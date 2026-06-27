"""CPU tests for the ABC fast-cycle fitness adapter (xenodesign/abc/fitness.py).

The just-decided objective (docs/results/2026-06-25-cyclization-calibration.md): feed the
bees on **pTM (primary) + C-N termini-distance closure proxy (secondary)** at K*=10-25 fast
Chai steps, with the head-to-tail CLOSURE restraint only (NO target-specific coordination).
DROP mainchain-pLDDT/chirality (non-discriminating) and the diluting 4-term aggregate.

All CPU / fake-backend — no GPU. We assert: the pTM-led weighting; termini-proximity
normalization in [0,1] (closer = higher); -inf on failure; and that the predict runs at
``k_star`` steps with the closure restraint.
"""
from __future__ import annotations

import math

import pytest

from xenodesign.abc.fitness import make_abc_fitness


class _FakeBackend:
    """Records the predict call + returns a fake prediction the fitness reads pTM/CIF from."""

    def __init__(self, *, ptm=0.8, cn_distance=3.0, raise_on_predict=False,
                 cif_missing=False):
        self.ptm = ptm
        self.cn_distance = cn_distance
        self.raise_on_predict = raise_on_predict
        self.cif_missing = cif_missing
        self.calls = []

    def predict(self, entities, out_dir, num_diffn_timesteps=200,
                constraint_path=None, **kw):
        self.calls.append({
            "entities": entities, "num_diffn_timesteps": num_diffn_timesteps,
            "constraint_path": constraint_path,
        })
        if self.raise_on_predict:
            raise RuntimeError("predict blew up")

        class _Pred:
            pass

        p = _Pred()
        p.ptm = self.ptm
        p._cif_path = None if self.cif_missing else "/tmp/fake.cif"
        return p


def _patch_closure_geometry(monkeypatch, cn_distance):
    """Stub head_to_tail_closure_geometry_from_cif so no real gemmi/CIF is needed."""
    import xenodesign.abc.fitness as fit_mod

    def _fake_geom(cif_path, chain_name="A"):
        return {"cn_distance": cn_distance, "closed": cn_distance <= 1.6}

    monkeypatch.setattr(fit_mod, "head_to_tail_closure_geometry_from_cif", _fake_geom)


def test_fitness_is_ptm_led(monkeypatch):
    # Two candidates identical in termini distance but different pTM → higher pTM wins,
    # and pTM carries the larger weight (default w_ptm=0.7 > w_termini=0.3).
    _patch_closure_geometry(monkeypatch, cn_distance=3.0)
    hi = make_abc_fitness(_FakeBackend(ptm=0.9, cn_distance=3.0), k_star=15)
    lo = make_abc_fitness(_FakeBackend(ptm=0.3, cn_distance=3.0), k_star=15)
    s_hi = hi("AGA", {0: "D", 1: "L", 2: "D"})
    s_lo = lo("AGA", {0: "D", 1: "L", 2: "D"})
    assert s_hi > s_lo
    # The pTM gap (0.6) weighted at 0.7 dominates → score gap ≈ 0.42.
    assert s_hi - s_lo == pytest.approx(0.7 * 0.6, abs=1e-6)


def test_fitness_runs_k_star_steps_with_closure_restraint(monkeypatch):
    _patch_closure_geometry(monkeypatch, cn_distance=2.0)
    be = _FakeBackend(ptm=0.5, cn_distance=2.0)
    fit = make_abc_fitness(be, k_star=12, closure=True)
    fit("AGA", {0: "D", 1: "L", 2: "D"})
    assert be.calls, "predict was never called"
    call = be.calls[0]
    assert call["num_diffn_timesteps"] == 12        # K* steps, not 200
    assert call["constraint_path"] is not None      # head-to-tail closure restraint present


def test_fitness_emits_per_position_handedness_fasta(monkeypatch):
    # The mixed-chirality peptide chain is emitted via mixed_chirality_fasta (T1 emit).
    _patch_closure_geometry(monkeypatch, cn_distance=2.0)
    be = _FakeBackend()
    fit = make_abc_fitness(be, k_star=15)
    fit("AGA", {0: "D", 1: "L", 2: "D"})
    binder = be.calls[0]["entities"][0]
    assert binder["sequence"] == "(DAL)G(DAL)"      # D at 0,2; Gly achiral; reused T1 emit


def test_termini_proximity_closer_is_higher(monkeypatch):
    # Same pTM; the candidate whose termini are CLOSER scores higher (proximity in [0,1]).
    near = _FakeBackend(ptm=0.5, cn_distance=1.4)   # ~closed
    far = _FakeBackend(ptm=0.5, cn_distance=18.0)   # wide open
    _patch_closure_geometry(monkeypatch, cn_distance=1.4)
    s_near = make_abc_fitness(near, k_star=15)("AA", {0: "D", 1: "D"})
    _patch_closure_geometry(monkeypatch, cn_distance=18.0)
    s_far = make_abc_fitness(far, k_star=15)("AA", {0: "D", 1: "D"})
    assert s_near > s_far


def test_termini_proximity_is_normalized_unit_interval(monkeypatch):
    # A perfectly-closed bond (cn≈1.33) → proximity≈1; a far-open one → proximity→0.
    _patch_closure_geometry(monkeypatch, cn_distance=1.33)
    s_closed = make_abc_fitness(_FakeBackend(ptm=0.0, cn_distance=1.33),
                                k_star=15, w_ptm=0.0, w_termini=1.0)("AA", {0: "D", 1: "D"})
    assert 0.9 <= s_closed <= 1.0
    _patch_closure_geometry(monkeypatch, cn_distance=40.0)
    s_open = make_abc_fitness(_FakeBackend(ptm=0.0, cn_distance=40.0),
                              k_star=15, w_ptm=0.0, w_termini=1.0)("AA", {0: "D", 1: "D"})
    assert 0.0 <= s_open <= 0.1


def test_fitness_minus_inf_on_predict_error(monkeypatch):
    _patch_closure_geometry(monkeypatch, cn_distance=2.0)
    fit = make_abc_fitness(_FakeBackend(raise_on_predict=True), k_star=15)
    assert fit("AA", {0: "D", 1: "D"}) == float("-inf")


def test_fitness_publishes_last_structure_side_channel(monkeypatch):
    # FIX 1: the fitness publishes the just-predicted structure (CIF path) on a `last_structure`
    # attribute so the ABC engine can record it onto the FoodSource and thread it to Variant A.
    _patch_closure_geometry(monkeypatch, cn_distance=2.0)
    fit = make_abc_fitness(_FakeBackend(ptm=0.5, cn_distance=2.0), k_star=15)
    assert hasattr(fit, "last_structure") and fit.last_structure is None  # nothing predicted yet
    fit("AGA", {0: "D", 1: "L", 2: "D"})
    assert fit.last_structure == "/tmp/fake.cif"   # the CIF the fake backend predicted


def test_fitness_clears_last_structure_on_failure(monkeypatch):
    # A failed eval must not re-publish a stale structure (the side-channel is reset up-front).
    _patch_closure_geometry(monkeypatch, cn_distance=2.0)
    fit = make_abc_fitness(_FakeBackend(ptm=0.5), k_star=15)
    fit("AGA", {0: "D", 1: "L", 2: "D"})
    assert fit.last_structure == "/tmp/fake.cif"
    fit("Z", {0: "D"})   # unknown 1-letter code 'Z' → mixed_chirality_fasta raises → -inf
    assert fit.last_structure is None


def test_fitness_falls_back_to_ptm_only_when_no_cif(monkeypatch):
    # No CIF on the prediction → termini proximity is unavailable; the score is the pTM term
    # alone (never crashes the search).
    _patch_closure_geometry(monkeypatch, cn_distance=2.0)
    be = _FakeBackend(ptm=0.6, cif_missing=True)
    score = make_abc_fitness(be, k_star=15)("AA", {0: "D", 1: "D"})
    assert score == pytest.approx(0.7 * 0.6, abs=1e-6)
    assert math.isfinite(score)
