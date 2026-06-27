"""MOD-2 characterization: dispatch.run_design reproduces the legacy run_alpha_design
report, key-by-key, for a fixed CPU-faked alpha case.

This pins the unified-dispatch α path == the validated legacy driver BEFORE MOD-2's
refactor thins run_alpha_design into a wrapper over run_design. Both paths share the
SAME deterministic CPU fakes (predictor, seq-update, CIF reader, double-flip) and the
SAME seed, so any genuine wiring divergence would show up as a mismatched report key.

Keys that are inherently run-specific (the timed wall, the per-run temp out_dir, and
the out_dir-derived constraint path) are excluded from the comparison — they are not
behaviour, and the dispatcher measures wall/threads it deliberately (test_dispatch.py
already pins that contract).
"""
from __future__ import annotations

import numpy as np

import xenodesign.classes.alpha as alpha_mod
import scripts.design_alpha as _da
from xenodesign import dispatch
from xenodesign.classes.base import SeedSpec
from xenodesign.config import resolve_config
from xenodesign.loop import LoopStep


_SEED_SEQ = "ACDEFGHIKLMNPQRSTVWYG"   # fixed 21-mer α seed (ends in the Gly anchor)
_N_ITERS = 2


class _FakePred:
    """Deterministic Prediction stand-in shared by both paths."""

    coords = np.zeros((3, 3))
    iptm = 0.5
    token_index = np.array([1, 1, 1])
    plddt = np.array([80.0, 80.0, 80.0])


class _FakeChaiBackend:
    def __init__(self, *a, **k):
        pass

    def predict(self, *a, **k):
        return _FakePred()

    def truncated_refine(self, *a, **k):
        return _FakePred()


def _drop_runspecific(report: dict) -> dict:
    """Strip keys that are inherently run-specific (not behaviour)."""
    out = dict(report)
    for k in ("wall_time_s", "out_dir", "constraint_path"):
        out.pop(k, None)
    # trajectory carries no out_dir; metrics is None on the CPU-fake path (no chai_out).
    return out


def _common_fakes(monkeypatch, tmp_path):
    """Patch every shared GPU/IO seam to the SAME deterministic CPU fake, on every module
    object either path resolves it through (alpha_mod, the design_alpha shim, dispatch)."""
    # Structure predictor (legacy path constructs ChaiBackend directly).
    monkeypatch.setattr("xenodesign.backends.chai_backend.ChaiBackend", _FakeChaiBackend,
                        raising=True)
    # Dispatch builds its predictor through _make_predictor / target_entities / patches.
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakeChaiBackend(), lambda *a, **k: _FakePred()))

    # Double-flip reflection -> deterministic coords (no CIF needed).
    monkeypatch.setattr("xenodesign.seed.reflect_binder_in_complex_from_cif",
                        lambda *a, **k: np.zeros((3, 3)))

    # CIF helpers: _best_cif_path is used for the L-seed double-flip (we fake the reflection, so
    # the returned path is never opened) AND by the assembler to read the per-iter binder seq.
    # Return a dummy path so the L-seed step proceeds, but force the assembler's per-iter read to
    # fall back to step.state.d_fasta by making binder_seq_from_cif raise (no CIF on disk).
    _dummy = tmp_path / "dummy.cif"

    def _dummy_cif(*a, **k):
        return _dummy

    monkeypatch.setattr(alpha_mod, "_best_cif_path", _dummy_cif, raising=False)
    monkeypatch.setattr(_da, "_best_cif_path", _dummy_cif, raising=False)
    monkeypatch.setattr("xenodesign.cif_io._best_cif_path", _dummy_cif, raising=False)

    def _no_binder_seq(*a, **k):
        raise FileNotFoundError("cpu-fake: no scored CIF to read binder seq from")

    monkeypatch.setattr(alpha_mod, "binder_seq_from_cif", _no_binder_seq, raising=False)

    # Sequence update: identity-ish fn that just returns the seed (deterministic, no MPNN).
    def _fake_make_seq_update(wrapper, num_seqs=8, backend="ligandmpnn", roles=None):
        return lambda pred: _SEED_SEQ

    monkeypatch.setattr(alpha_mod, "make_alpha_seq_update_fn", _fake_make_seq_update)
    monkeypatch.setattr(_da, "make_alpha_seq_update_fn", _fake_make_seq_update)


def test_dispatch_matches_legacy_alpha(tmp_path, monkeypatch):
    _common_fakes(monkeypatch, tmp_path)

    # Force BOTH paths to start from the SAME seed: legacy via seed_seq=, dispatch by pinning
    # Alpha.seed to return the same SeedSpec (the seed SOURCE differs by design — legacy
    # build_alpha_seed vs unified PepMLM — so we hold it fixed to characterize the WIRING).
    monkeypatch.setattr(alpha_mod.Alpha, "seed",
                        lambda self, cfg, target_seq: SeedSpec(one_letter=_SEED_SEQ))
    # The legacy target read (read_target_sequence) and dispatch target (target_entities) must
    # yield the SAME target sequence so the assembled complexes match.
    _TARGET = "GSHMKVLITGGAGFIGSHLVDRL"
    monkeypatch.setattr("xenodesign.seed.read_target_sequence",
                        lambda *a, **k: _TARGET)
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "target",
                                       "sequence": _TARGET, "chirality": "L"}], None, None))

    # ── legacy driver ──
    legacy_out = tmp_path / "legacy"
    legacy = _da.run_alpha_design(
        n_iters=_N_ITERS, out_dir=legacy_out,
        restraints=False, use_pll=False, use_pepmlm=False, seed_seq=_SEED_SEQ,
        backend="ligandmpnn",
    )

    # ── unified dispatch ──
    dispatch_out = tmp_path / "dispatch"
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(dispatch_out),
                         cli_overrides={"loop.iters": _N_ITERS, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False,
                                        "loop.backend": "ligandmpnn"})
    unified = dispatch.run_design(cfg)

    a = _drop_runspecific(legacy)
    b = _drop_runspecific(unified)

    assert a.keys() == b.keys(), f"report key sets differ: {set(a) ^ set(b)}"
    for key in a:
        assert a[key] == b[key], f"report['{key}'] differs:\n  legacy={a[key]!r}\n  dispatch={b[key]!r}"
