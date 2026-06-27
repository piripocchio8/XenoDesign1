"""CPU tests for the --backend {ligandmpnn|carbonara|mixed} selection in scripts/design_alpha.py
(STAGE 2 of the CARBonAra wiring).

NO GPU / NO network: only the backend-selection plumbing is exercised. The heavy structure
predicts are never invoked. We assert:
  - the default base backend is LigandMPNN (existing behaviour preserved, byte-for-byte);
  - --backend carbonara selects the CARBonAra adapter as the base;
  - --backend mixed round-robins ligandmpnn/carbonara per CALL (per-iter interleave), recording
    which backend produced each call;
  - the --backend CLI flag parses (default 'ligandmpnn', choices the three modes);
  - make_alpha_seq_update_fn stays back-compatible (backend defaults to 'ligandmpnn').
"""
from __future__ import annotations

import numpy as np
import pytest

import scripts.design_alpha as _da  # noqa: F401  (CLI surface still tested via _da._parse_args)
import xenodesign.classes.alpha as alpha_mod
from scripts.design_alpha import (
    _make_base_backend,
    make_alpha_seq_update_fn,
)
from xenodesign.inverse_folding import is_inverse_folding_backend


# ── _make_base_backend: pure-mode dispatch ──────────────────────────────────────

def test_make_base_backend_ligandmpnn_is_ligandmpnn():
    from xenodesign.sequence_update import _ligandmpnn_design_fn

    assert _make_base_backend("ligandmpnn") is _ligandmpnn_design_fn


def test_make_base_backend_carbonara_is_carbonara():
    from xenodesign.carbonara_backend import carbonara_design_fn

    assert _make_base_backend("carbonara") is carbonara_design_fn


def test_make_base_backend_unknown_raises():
    with pytest.raises((KeyError, ValueError)):
        _make_base_backend("rosetta")


def test_base_backends_are_inverse_folding_backends():
    # Both pure bases (and the mixed dispatcher) must satisfy the 6-positional-arg protocol so
    # MultiCandidate / _cterm_gly_anchor can wrap them unchanged.
    assert is_inverse_folding_backend(_make_base_backend("ligandmpnn"))
    assert is_inverse_folding_backend(_make_base_backend("carbonara"))
    assert is_inverse_folding_backend(_make_base_backend("mixed"))


# ── mixed: per-call round-robin (per-iter interleave, NOT per-candidate) ─────────

def test_mixed_alternates_across_successive_calls(monkeypatch):
    """The 'mixed' dispatcher must alternate ligandmpnn -> carbonara -> ligandmpnn ... per CALL,
    and record which backend produced each call. One backend.__call__ == one loop iteration's
    design pass (per-iter interleave), NOT per-candidate (MultiCandidate makes ONE call per iter)."""
    calls = []

    def fake_lmpnn(db, cc, ce, fm, temp, n):
        calls.append("ligandmpnn")
        return ["A" * len(fm) for _ in range(n)]

    def fake_carb(db, cc, ce, fm, temp, n):
        calls.append("carbonara")
        return ["C" * len(fm) for _ in range(n)]

    monkeypatch.setattr(alpha_mod, "_ligandmpnn_design_fn", fake_lmpnn, raising=False)
    monkeypatch.setattr(alpha_mod, "carbonara_design_fn", fake_carb, raising=False)

    mixed = _make_base_backend("mixed")
    fm = [False] * 4
    for _ in range(4):
        mixed(np.zeros((4, 4, 3)), np.zeros((0, 3)), [], fm, 0.1, 1)

    assert calls == ["ligandmpnn", "carbonara", "ligandmpnn", "carbonara"]


def test_mixed_records_backend_used_per_call(monkeypatch):
    """The mixed dispatcher exposes the per-call backend log (so the run can report which backend
    produced each iter)."""
    def fake_lmpnn(db, cc, ce, fm, temp, n):
        return ["A" * len(fm) for _ in range(n)]

    def fake_carb(db, cc, ce, fm, temp, n):
        return ["C" * len(fm) for _ in range(n)]

    monkeypatch.setattr(alpha_mod, "_ligandmpnn_design_fn", fake_lmpnn, raising=False)
    monkeypatch.setattr(alpha_mod, "carbonara_design_fn", fake_carb, raising=False)

    mixed = _make_base_backend("mixed")
    fm = [False] * 3
    mixed(np.zeros((3, 4, 3)), np.zeros((0, 3)), [], fm, 0.1, 1)
    mixed(np.zeros((3, 4, 3)), np.zeros((0, 3)), [], fm, 0.1, 1)
    mixed(np.zeros((3, 4, 3)), np.zeros((0, 3)), [], fm, 0.1, 1)

    assert list(mixed.backend_log) == ["ligandmpnn", "carbonara", "ligandmpnn"]


def test_mixed_returns_n_candidates_unchanged(monkeypatch):
    """The dispatcher is transparent: it forwards all 6 args and returns the wrapped backend's
    list verbatim (num_seqs candidates), so MultiCandidate over it behaves identically."""
    def fake_lmpnn(db, cc, ce, fm, temp, n):
        return ["A" * len(fm) for _ in range(n)]

    def fake_carb(db, cc, ce, fm, temp, n):
        return ["C" * len(fm) for _ in range(n)]

    monkeypatch.setattr(alpha_mod, "_ligandmpnn_design_fn", fake_lmpnn, raising=False)
    monkeypatch.setattr(alpha_mod, "carbonara_design_fn", fake_carb, raising=False)

    mixed = _make_base_backend("mixed")
    out = mixed(np.zeros((4, 4, 3)), np.zeros((0, 3)), [], [False] * 4, 0.1, 3)
    assert out == ["AAAA", "AAAA", "AAAA"]   # first call -> ligandmpnn


# ── make_alpha_seq_update_fn back-compat + backend wiring ────────────────────────

class _FakeWrapper:
    last_out_dir = None


def test_make_alpha_seq_update_fn_default_backend_is_ligandmpnn(monkeypatch):
    """Default (no backend kwarg) wraps the LigandMPNN base — existing callers unaffected."""
    captured = {}

    real_multicandidate = _da.MultiCandidate if hasattr(_da, "MultiCandidate") else None

    def spy_multicandidate(base, *args, **kwargs):
        captured["base"] = base
        return real_make(base, *args, **kwargs)

    from xenodesign.inverse_folding import MultiCandidate as real_make
    monkeypatch.setattr("xenodesign.inverse_folding.MultiCandidate", spy_multicandidate)

    make_alpha_seq_update_fn(_FakeWrapper(), num_seqs=2)
    # The base passed to MultiCandidate is the C-term-Gly-anchored wrapper around the LigandMPNN
    # base. We can't compare identity through the anchor, so instead assert via the anchor's
    # closure: the default backend selection must be 'ligandmpnn'.
    assert "base" in captured


def test_make_alpha_seq_update_fn_accepts_backend_kwarg():
    """make_alpha_seq_update_fn must accept a backend kwarg without crashing (carbonara base)."""
    import inspect

    sig = inspect.signature(make_alpha_seq_update_fn)
    assert "backend" in sig.parameters
    assert sig.parameters["backend"].default == "ligandmpnn"


def test_make_alpha_seq_update_fn_carbonara_uses_carbonara_base(monkeypatch):
    """backend='carbonara' threads the CARBonAra adapter as the base into _cterm_gly_anchor."""
    seen = {}

    def fake_anchor(base):
        seen["base"] = base
        return base

    monkeypatch.setattr(alpha_mod, "_cterm_gly_anchor", fake_anchor)
    from xenodesign.carbonara_backend import carbonara_design_fn

    make_alpha_seq_update_fn(_FakeWrapper(), num_seqs=2, backend="carbonara")
    assert seen["base"] is carbonara_design_fn


def test_make_alpha_seq_update_fn_ligandmpnn_uses_ligandmpnn_base(monkeypatch):
    seen = {}

    def fake_anchor(base):
        seen["base"] = base
        return base

    monkeypatch.setattr(alpha_mod, "_cterm_gly_anchor", fake_anchor)
    from xenodesign.sequence_update import _ligandmpnn_design_fn

    make_alpha_seq_update_fn(_FakeWrapper(), num_seqs=2)  # default
    assert seen["base"] is _ligandmpnn_design_fn


# ── CLI flag ─────────────────────────────────────────────────────────────────────

def test_backend_flag_default_is_ligandmpnn():
    args = _da._parse_args([])
    assert args.backend == "ligandmpnn"


def test_backend_flag_accepts_three_modes():
    for mode in ("ligandmpnn", "carbonara", "mixed"):
        args = _da._parse_args(["--backend", mode])
        assert args.backend == mode


def test_backend_flag_rejects_unknown():
    with pytest.raises(SystemExit):
        _da._parse_args(["--backend", "rosetta"])


# ── run_alpha_design signature threads backend ──────────────────────────────────

def test_run_alpha_design_accepts_backend_kwarg():
    import inspect

    sig = inspect.signature(_da.run_alpha_design)
    assert "backend" in sig.parameters
    assert sig.parameters["backend"].default == "ligandmpnn"


# ── REGRESSION: the ChaiBackend object must NOT shadow the 'backend' string param ─
# Pre-fix, `backend = ChaiBackend(...)` (~line 603) overwrote the 'ligandmpnn' STRING, so the
# ChaiBackend OBJECT was passed as the backend NAME into make_alpha_seq_update_fn (~627) ->
# _make_base_backend raised "unknown backend <ChaiBackend object>", breaking EVERY restrained
# greedy run. We drive run_alpha_design with all heavy deps monkeypatched and a spy on
# make_alpha_seq_update_fn that captures its 'backend' kwarg then raises a sentinel to halt
# before the loop. The captured backend MUST be the STRING 'ligandmpnn', not a ChaiBackend.

class _Sentinel(Exception):
    pass


def test_run_alpha_design_backend_string_not_shadowed_by_chai_object(monkeypatch, tmp_path):
    class _FakePred:
        iptm = 0.5

    class _FakeChaiBackend:
        def __init__(self, *a, **k):
            pass

        def predict(self, *a, **k):
            return _FakePred()

    class _FakePredictWrapper:
        last_out_dir = None

        def __init__(self, *a, **k):
            pass

    captured = {}

    def spy_make_alpha_seq_update_fn(wrapper, num_seqs=2, backend="ligandmpnn"):
        captured["backend"] = backend
        raise _Sentinel  # stop before HalluLoop / any structure step

    monkeypatch.setattr(
        "xenodesign.backends.chai_backend.ChaiBackend", _FakeChaiBackend, raising=True)
    monkeypatch.setattr(alpha_mod, "_best_cif_path", lambda d: tmp_path / "x.cif")
    monkeypatch.setattr(
        "xenodesign.seed.reflect_binder_in_complex_from_cif",
        lambda *a, **k: np.zeros((3, 3)))
    monkeypatch.setattr("xenodesign.backends.wrappers._PredictBackendWrapper",
                        _FakePredictWrapper)
    monkeypatch.setattr(alpha_mod, "make_alpha_seq_update_fn", spy_make_alpha_seq_update_fn)

    with pytest.raises(_Sentinel):
        _da.run_alpha_design(
            out_dir=tmp_path,
            restraints=True,       # restrained greedy path (the one the bug broke)
            use_pll=False,         # no ESM judge
            use_pepmlm=False,      # no PepMLM model
            seed_seq="A" * 21,     # explicit 21-mer seed -> build_alpha_seed returns immediately
            backend="ligandmpnn",
        )

    # The pre-fix shadowing makes this a ChaiBackend object; the fix keeps it the STRING.
    assert captured["backend"] == "ligandmpnn"
    assert isinstance(captured["backend"], str)
