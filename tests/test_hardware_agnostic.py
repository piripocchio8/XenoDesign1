"""Hardware/host-agnosticity contract tests (audit HW-1, HW-2, HW-5).

These pin the single device-resolution point and the externalized external-tool /
local-reference paths so the package runs on a CPU-only / non-ifrit box. All CPU-only:
torch is monkeypatched (never a real GPU), and no external tool is ever spawned.
"""
import importlib
import sys
import types

import pytest


# ── HW-1: central device resolver ────────────────────────────────────────────────

def _fake_torch(cuda_available=False, mps_available=False):
    """Build a stand-in torch module exposing just torch.cuda.is_available and
    torch.backends.mps.is_available so resolve_device can be exercised without a GPU."""
    t = types.SimpleNamespace()
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps_available))
    t.backends = backends
    return t


def test_resolve_device_cpu_when_no_cuda(monkeypatch):
    from xenodesign import config
    monkeypatch.delenv("XENO_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    assert config.resolve_device() == "cpu"


def test_resolve_device_cuda_when_available(monkeypatch):
    from xenodesign import config
    monkeypatch.delenv("XENO_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    assert config.resolve_device() == "cuda:0"


def test_resolve_device_mps_when_only_mps(monkeypatch):
    from xenodesign import config
    monkeypatch.delenv("XENO_DEVICE", raising=False)
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda_available=False, mps_available=True))
    assert config.resolve_device() == "mps"


def test_resolve_device_env_override_wins(monkeypatch):
    from xenodesign import config
    monkeypatch.setenv("XENO_DEVICE", "cuda:3")
    # Even with no cuda, the explicit env override is honoured verbatim.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    assert config.resolve_device() == "cuda:3"


def test_resolve_device_honours_explicit_cfg(monkeypatch):
    from xenodesign import config
    monkeypatch.delenv("XENO_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    cfg = config.DesignConfig(device="cuda:1")
    assert config.resolve_device(cfg) == "cuda:1"


def test_resolve_device_falls_through_unset_cfg(monkeypatch):
    """An unset cfg.device (sentinel) does NOT pin a device — it falls through to auto."""
    from xenodesign import config
    monkeypatch.delenv("XENO_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    cfg = config.DesignConfig()           # default device is the unset sentinel
    assert config.resolve_device(cfg) == "cpu"


def test_resolve_device_no_torch_is_cpu(monkeypatch):
    """If torch cannot be imported at all, the resolver degrades to 'cpu'."""
    from xenodesign import config
    monkeypatch.delenv("XENO_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    assert config.resolve_device() == "cpu"


def test_config_import_does_not_import_torch():
    """config.py must stay CPU-clean: importing it must not pull torch into sys.modules."""
    sys.modules.pop("torch", None)
    sys.modules.pop("xenodesign.config", None)
    importlib.import_module("xenodesign.config")
    assert "torch" not in sys.modules


# ── HW-2: externalized CARBonAra path ────────────────────────────────────────────

def test_carbonara_dir_from_env(monkeypatch):
    monkeypatch.setenv("CARBONARA_DIR", "/tmp/my-carbonara")
    sys.modules.pop("xenodesign.carbonara_backend", None)
    cb = importlib.import_module("xenodesign.carbonara_backend")
    assert str(cb._CARBONARA_DIR) == "/tmp/my-carbonara"
    assert str(cb._CARBONARA_VENV_PYTHON) == "/tmp/my-carbonara/.venv/bin/python"
    assert str(cb._CARBONARA_SCRIPT) == "/tmp/my-carbonara/carbonara.py"


def test_carbonara_dir_default_unchanged(monkeypatch):
    """With no env override, the ifrit default path is preserved (no behaviour change)."""
    monkeypatch.delenv("CARBONARA_DIR", raising=False)
    sys.modules.pop("xenodesign.carbonara_backend", None)
    cb = importlib.import_module("xenodesign.carbonara_backend")
    assert str(cb._CARBONARA_DIR) == "/home/user/CARBonAra"


# ── HW-2: MetalHawk env name overridable ─────────────────────────────────────────

def test_metalhawk_env_from_env(monkeypatch):
    monkeypatch.setenv("METALHAWK_ENV", "my_mh_env")
    sys.modules.pop("xenodesign.eval.metal_geometry_gate", None)
    mg = importlib.import_module("xenodesign.eval.metal_geometry_gate")
    assert mg._DEFAULT_METALHAWK_ENV == "my_mh_env"


def test_metalhawk_gate_failsoft_no_dir(monkeypatch, tmp_path):
    """Gate never raises when the dir is absent — returns a pass-through GateResult."""
    monkeypatch.delenv("METALHAWK_DIR", raising=False)
    from xenodesign.eval import metal_geometry_gate as mg
    res = mg.metal_geometry_gate(tmp_path / "missing.cif", metalhawk_dir=None)
    assert res.passed is True
    assert res.ok is False
    assert "METALHAWK_DIR" in (res.error or "")


# ── HW-5: configurable XenoDesign1_local_ref root ────────────────────────────────

def test_local_ref_default_relative():
    from xenodesign import config
    monkeyless = config.local_ref()
    assert monkeyless.name == "XenoDesign1_local_ref"


def test_local_ref_from_env(monkeypatch):
    from xenodesign import config
    monkeypatch.setenv("XENO_LOCAL_REF", "/data/refs")
    p = config.local_ref("9dxx_target_gate", "ha_target.fasta")
    assert str(p) == "/data/refs/9dxx_target_gate/ha_target.fasta"


# ── resolve_config stays torch-free; device is resolved at point-of-use ───────────

def test_resolve_config_leaves_device_sentinel(monkeypatch):
    """resolve_config must NOT auto-detect the device (it would import torch into the many
    CPU tests that call it). The sentinel is left for dispatch/backends to resolve lazily."""
    from xenodesign import config
    monkeypatch.delenv("XENO_DEVICE", raising=False)
    sys.modules.pop("torch", None)
    cfg = config.resolve_config("alpha")
    assert cfg.device == config.DEVICE_AUTO  # unset sentinel, torch never imported
    assert "torch" not in sys.modules


def test_resolve_config_device_cli_override_survives(monkeypatch):
    from xenodesign import config
    monkeypatch.delenv("XENO_DEVICE", raising=False)
    cfg = config.resolve_config("alpha", cli_overrides={"device": "cuda:2"})
    # An explicit --device override is preserved verbatim and resolve_device honours it.
    assert cfg.device == "cuda:2"
    assert config.resolve_device(cfg) == "cuda:2"
