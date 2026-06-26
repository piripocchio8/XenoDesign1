"""CPU unit tests for the migrated Alpha BinderClass (T5).

These exercise the Alpha adapter's hooks against the SAME helpers the validated
``run_alpha_design`` driver uses. The behaviour-preserving regression oracle is
``tests/test_design_alpha*.py`` (the legacy driver tests, kept green by the shim);
this file pins the NEW class-protocol surface. No GPU / no network: the predictor
and inverse-folding bases are mocked where needed.
"""
from __future__ import annotations

import numpy as np

from xenodesign.benchmark.cases import get_case
from xenodesign.classes.alpha import (
    Alpha,
    _loop_score_fn,
    make_ipsae_loop_score_fn,
    make_mixed_loop_score_fn,
)
from xenodesign.classes.base import SeedSpec
from xenodesign.config import resolve_config


# ── case_id + registry identity ────────────────────────────────────────────────

def test_alpha_case_id():
    assert Alpha().case_id == "alpha"


def test_registry_alpha_is_real_class():
    """base.CLASS_REGISTRY['alpha'] must be the migrated real Alpha (not the stub)."""
    from xenodesign.classes.base import CLASS_REGISTRY

    a = CLASS_REGISTRY["alpha"]
    assert isinstance(a, Alpha)
    assert a.case_id == "alpha"


# ── ss_bias delegates to the case SS-bias config (α -> reward helix) ────────────

def test_alpha_case_id_and_ss_bias():
    a = Alpha()
    assert a.case_id == "alpha"
    cfg = resolve_config("alpha", target_type="protein")
    ss = a.ss_bias(cfg, get_case("alpha"))
    assert ss.target_helix_frac == 1.0          # helix_ok


# ── seed (offline) reuses build_alpha_seed; deterministic length + Gly anchor ───

def test_alpha_seed_offline_is_deterministic_length():
    a = Alpha()
    cfg = resolve_config("alpha", target_type="protein",
                         cli_overrides={"use_pepmlm": False})
    spec = a.seed(cfg, target_seq="G" * 41)
    assert isinstance(spec, SeedSpec)
    assert len(spec.one_letter) == get_case("alpha").binder_length
    assert "G" in spec.one_letter               # C-term Gly anchor preserved
    assert spec.cys_positions == ()             # alpha pins no ICK cys
    assert spec.fixed_chirality == {}           # alpha pins no chirality


# ── objective routing: default iptm -> _loop_score_fn ──────────────────────────

def test_alpha_objective_default_is_iptm_score():
    a = Alpha()
    cfg = resolve_config("alpha", target_type="protein")
    fn = a.objective(cfg, wrapper=None)
    assert fn is _loop_score_fn

    class P:
        token_index = np.array([1, 1])
        plddt = np.array([70.0, 70.0])
        iptm = 0.6

    assert isinstance(fn(P()), float)


def test_alpha_objective_mixed_and_ipsae_route():
    a = Alpha()

    class _Wrapper:
        last_out_dir = None

    cfg_mixed = resolve_config("alpha", target_type="protein",
                               cli_overrides={"objective": "mixed"})
    cfg_ipsae = resolve_config("alpha", target_type="protein",
                               cli_overrides={"objective": "ipsae"})
    fn_mixed = a.objective(cfg_mixed, wrapper=_Wrapper())
    fn_ipsae = a.objective(cfg_ipsae, wrapper=_Wrapper())

    class _P:
        token_index = np.array([0, 0, 1, 1, 1])
        plddt = np.array([60.0, 60.0, 70.0, 70.0, 70.0])
        iptm = 0.5

    # with no last_out_dir both gracefully fall back to the reproducible ipTM score.
    assert fn_mixed(_P()) == _loop_score_fn(_P())
    assert fn_ipsae(_P()) == _loop_score_fn(_P())


# ── restraints honour cfg.restraints_on ────────────────────────────────────────

def test_alpha_restraints_off_returns_none():
    a = Alpha()
    cfg = resolve_config("alpha", target_type="protein",
                         cli_overrides={"restraints_on": False})
    assert a.restraints(cfg, get_case("alpha"), out_dir="/tmp/xd_unused",
                        target_ctx=None) is None


def test_alpha_restraints_on_delegates_to_build_alpha_restraint(tmp_path, monkeypatch):
    """restraints_on -> calls build_alpha_restraint(case, out_dir) with the run's chains."""
    import xenodesign.classes.alpha as alpha_mod

    seen = {}

    def _fake_build(case, out_dir, **kw):
        seen["case"] = case
        seen["out_dir"] = out_dir
        return out_dir

    monkeypatch.setattr(alpha_mod, "build_alpha_restraint", _fake_build)
    a = Alpha()
    cfg = resolve_config("alpha", target_type="protein")  # restraints_on default True
    case = get_case("alpha")
    out = a.restraints(cfg, case, out_dir=str(tmp_path), target_ctx=None)
    assert out == str(tmp_path)
    assert seen["case"] is case


# ── closure is empty for alpha (no head-to-tail / disulfide) ────────────────────

def test_alpha_closure_is_empty():
    a = Alpha()
    cfg = resolve_config("alpha", target_type="protein")
    spec = SeedSpec(one_letter="ACDEFGHIK")
    assert a.closure(cfg, spec) == []


# ── accept_fns: periodicity gate only when cfg.gates.periodicity ────────────────

def test_alpha_accept_fns_none_by_default():
    a = Alpha()
    cfg = resolve_config("alpha", target_type="protein")  # gates.periodicity default False
    assert a.accept_fns(cfg) is None


def test_alpha_accept_fns_periodicity_when_enabled():
    a = Alpha()
    cfg = resolve_config("alpha", target_type="protein",
                         cli_overrides={"gates.periodicity": True})
    gate = a.accept_fns(cfg)
    assert callable(gate)


# ── seq_update builds the drift-fixed updater (mocked base + wrapper) ───────────

def test_alpha_seq_update_returns_callable(monkeypatch):
    """seq_update wires make_alpha_seq_update_fn with the cfg's num_seqs/backend."""
    import scripts.design_alpha as _da

    seen = {}

    def _fake_seq_update(wrapper, num_seqs=8, backend="ligandmpnn", roles=None):
        seen["num_seqs"] = num_seqs
        seen["backend"] = backend
        seen["roles"] = roles
        return lambda pred: "ACDEFG"

    monkeypatch.setattr(_da, "make_alpha_seq_update_fn", _fake_seq_update)
    # Alpha.seq_update calls the module-level make_alpha_seq_update_fn directly; patch it there too.
    import xenodesign.classes.alpha as alpha_mod
    monkeypatch.setattr(alpha_mod, "make_alpha_seq_update_fn", _fake_seq_update)

    a = Alpha()
    cfg = resolve_config("alpha", target_type="protein",
                         cli_overrides={"loop.num_seqs": 4, "loop.backend": "carbonara"})

    class _Wrapper:
        last_out_dir = None

    fn = a.seq_update(cfg, _Wrapper(), SeedSpec(one_letter="ACDEFG"))
    assert callable(fn)
    assert seen == {"num_seqs": 4, "backend": "carbonara", "roles": None}


# ── referee builds a per-step RefereeScore reader (mocked CIF/judge) ────────────

def test_alpha_referee_returns_scoring_callable():
    a = Alpha()
    cfg = resolve_config("alpha", target_type="protein")
    fn = a.referee(cfg, loop_dir="/tmp/xd_loop_unused", esm_judge=None)
    assert callable(fn)
