"""CPU-only tests for scripts/generate_and_select.py — pure harvest/selection logic.

No GPU, torch, or chai required.  Run with:
    PYTHONPATH=$PWD micromamba run -n base python -m pytest tests/test_generate_and_select.py -v

Tests cover:
- harvest_clean_designs: panel-selected extraction, step scanning, deduplication, ranking
- compute_yield: counting, fraction, edge cases
- Integration: end-to-end with mock trajectory results
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_and_select import harvest_clean_designs, compute_yield


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_traj(
    traj_id: int,
    seed: int,
    panel_chirality: float = 0.05,
    panel_iptm: float = 0.7,
    panel_composite: float = 0.65,
    panel_pll: float | None = -2.0,
    traj_steps: list[dict] | None = None,
    error: str | None = None,
    d_fasta: str = "(DAL)G(DAL)",
    l_seq: str = "AGA",
) -> dict:
    """Build a fake trajectory result dict as returned by generate_and_select."""
    return {
        "traj_id": traj_id,
        "seed": seed,
        "panel_selected": {
            "iter": 3,
            "d_fasta": d_fasta,
            "l_seq": l_seq,
            "iptm": panel_iptm,
            "chirality": panel_chirality,
            "pll": panel_pll,
            "composite": panel_composite,
        } if not error else {},
        "trajectory": traj_steps or [],
        "wall_time_s": 120.0,
        "out_dir": f"/home/tmp/xd_gas_test/traj_{traj_id:03d}",
        "error": error,
    }


def _make_step(
    iter_idx: int,
    chirality: float = 0.0,
    iptm: float = 0.6,
    composite: float = 0.5,
    pll: float | None = -1.5,
    d_fasta: str = "(DAL)G",
    l_seq: str = "AG",
    vetoed: bool = False,
) -> dict:
    return {
        "iter": iter_idx,
        "d_fasta": d_fasta,
        "l_seq": l_seq,
        "iptm": iptm,
        "chirality": chirality,
        "pll": pll,
        "composite": composite,
        "vetoed": vetoed,
    }


# ── harvest_clean_designs ─────────────────────────────────────────────────────

class TestHarvestCleanDesigns:

    def test_panel_selected_below_threshold_is_harvested(self):
        trajs = [_make_traj(0, 1000, panel_chirality=0.05)]
        clean = harvest_clean_designs(trajs)
        assert len(clean) == 1
        assert clean[0]["traj_id"] == 0
        assert clean[0]["chirality"] == pytest.approx(0.05)
        assert clean[0]["source"] == "panel_selected"

    def test_panel_selected_above_threshold_is_not_harvested(self):
        trajs = [_make_traj(0, 1000, panel_chirality=0.15)]
        clean = harvest_clean_designs(trajs)
        assert len(clean) == 0

    def test_panel_exactly_at_threshold_is_harvested(self):
        """threshold is inclusive: chirality == 0.1 should be harvested."""
        trajs = [_make_traj(0, 1000, panel_chirality=0.1)]
        clean = harvest_clean_designs(trajs, chirality_threshold=0.1)
        assert len(clean) == 1

    def test_step_scan_finds_clean_step_not_in_panel(self):
        """A clean step in the trajectory scan (not panel-selected) should be added."""
        step = _make_step(iter_idx=1, chirality=0.0, iptm=0.5, composite=0.4,
                          d_fasta="(DAL)G(DLE)", l_seq="AGL")
        traj = _make_traj(0, 1000, panel_chirality=0.3,  # panel is dirty
                          traj_steps=[step])
        clean = harvest_clean_designs([traj])
        # Panel is dirty → not harvested; but scan step is clean
        assert len(clean) == 1
        assert clean[0]["source"] == "scan"
        assert clean[0]["chirality"] == pytest.approx(0.0)
        assert clean[0]["l_seq"] == "AGL"

    def test_step_already_in_panel_not_duplicated(self):
        """If the panel-selected step is also ≤ threshold, don't add it twice."""
        # Panel selects iter=3; scan also sees iter=3
        step = _make_step(iter_idx=3, chirality=0.05, iptm=0.7, composite=0.65,
                          d_fasta="(DAL)G(DAL)", l_seq="AGA")
        traj = _make_traj(0, 1000, panel_chirality=0.05,
                          panel_composite=0.65, d_fasta="(DAL)G(DAL)", l_seq="AGA",
                          traj_steps=[step])
        clean = harvest_clean_designs([traj])
        assert len(clean) == 1  # not doubled

    def test_error_trajectory_is_skipped(self):
        trajs = [_make_traj(0, 1000, error="Worker crashed")]
        clean = harvest_clean_designs(trajs)
        assert len(clean) == 0

    def test_ranked_by_composite_descending(self):
        """Multiple clean designs should be sorted highest composite first."""
        trajs = [
            _make_traj(0, 1000, panel_chirality=0.05, panel_composite=0.3,
                       panel_iptm=0.6, d_fasta="(DAL)G", l_seq="AG"),
            _make_traj(1, 1001, panel_chirality=0.0, panel_composite=0.9,
                       panel_iptm=0.85, d_fasta="(DAL)G(DLE)", l_seq="AGL"),
            _make_traj(2, 1002, panel_chirality=0.08, panel_composite=0.6,
                       panel_iptm=0.7, d_fasta="(DAL)(DLE)", l_seq="AL"),
        ]
        clean = harvest_clean_designs(trajs)
        composites = [d["composite"] for d in clean]
        assert composites == sorted(composites, reverse=True)

    def test_multiple_clean_steps_from_one_trajectory(self):
        """If one trajectory has multiple clean steps (via scan), all are returned."""
        steps = [
            _make_step(iter_idx=i, chirality=0.0, iptm=0.5 + i * 0.05,
                       composite=0.4 + i * 0.05,
                       d_fasta=f"(DAL)G{i}", l_seq=f"AG{i}")
            for i in range(3)
        ]
        traj = _make_traj(0, 1000, panel_chirality=0.5,  # panel is dirty
                          traj_steps=steps)
        clean = harvest_clean_designs([traj])
        assert len(clean) == 3

    def test_empty_trajectory_list_returns_empty(self):
        assert harvest_clean_designs([]) == []

    def test_pll_none_handled_gracefully(self):
        trajs = [_make_traj(0, 1000, panel_chirality=0.05, panel_pll=None)]
        clean = harvest_clean_designs(trajs)
        assert len(clean) == 1
        assert clean[0]["pll"] is None

    def test_custom_threshold(self):
        """Custom threshold 0.05 should exclude chirality=0.06."""
        trajs = [
            _make_traj(0, 1000, panel_chirality=0.04, d_fasta="(DAL)G", l_seq="AG"),
            _make_traj(1, 1001, panel_chirality=0.06, d_fasta="(DLE)G", l_seq="LG"),
        ]
        clean = harvest_clean_designs(trajs, chirality_threshold=0.05)
        assert len(clean) == 1
        assert clean[0]["traj_id"] == 0

    def test_scan_step_dirty_not_harvested(self):
        step = _make_step(iter_idx=2, chirality=0.4, iptm=0.8, composite=0.7)
        traj = _make_traj(0, 1000, panel_chirality=0.4, traj_steps=[step])
        clean = harvest_clean_designs([traj])
        assert len(clean) == 0

    def test_tiebreak_by_iptm(self):
        """When composite is tied, higher iptm should come first."""
        trajs = [
            _make_traj(0, 1000, panel_chirality=0.0, panel_composite=0.5,
                       panel_iptm=0.6, d_fasta="(DAL)G", l_seq="AG"),
            _make_traj(1, 1001, panel_chirality=0.0, panel_composite=0.5,
                       panel_iptm=0.9, d_fasta="(DLE)G", l_seq="LG"),
        ]
        clean = harvest_clean_designs(trajs)
        assert clean[0]["iptm"] == pytest.approx(0.9)


# ── compute_yield ──────────────────────────────────────────────────────────────

class TestComputeYield:

    def test_all_clean_yield_1(self):
        trajs = [
            _make_traj(i, 1000 + i, panel_chirality=0.0)
            for i in range(5)
        ]
        n_ok, n_total, frac = compute_yield(trajs)
        assert n_ok == 5
        assert n_total == 5
        assert frac == pytest.approx(1.0)

    def test_none_clean_yield_0(self):
        trajs = [_make_traj(i, 1000 + i, panel_chirality=0.5) for i in range(3)]
        n_ok, n_total, frac = compute_yield(trajs)
        assert n_ok == 0
        assert n_total == 3
        assert frac == pytest.approx(0.0)

    def test_mixed_yield(self):
        trajs = [
            _make_traj(0, 1000, panel_chirality=0.05),  # clean
            _make_traj(1, 1001, panel_chirality=0.5),   # dirty
            _make_traj(2, 1002, panel_chirality=0.0),   # clean
            _make_traj(3, 1003, panel_chirality=0.8),   # dirty
        ]
        n_ok, n_total, frac = compute_yield(trajs)
        assert n_ok == 2
        assert n_total == 4
        assert frac == pytest.approx(0.5)

    def test_error_trajectories_not_counted_in_total(self):
        trajs = [
            _make_traj(0, 1000, panel_chirality=0.0),
            _make_traj(1, 1001, error="Worker crashed"),
        ]
        n_ok, n_total, frac = compute_yield(trajs)
        assert n_total == 1  # error traj excluded from denominator
        assert n_ok == 1

    def test_scan_step_counts_as_success(self):
        """Trajectory with dirty panel but clean scan step should count as success."""
        step = _make_step(iter_idx=0, chirality=0.0)
        traj = _make_traj(0, 1000, panel_chirality=0.5, traj_steps=[step])
        n_ok, n_total, frac = compute_yield([traj])
        assert n_ok == 1
        assert n_total == 1
        assert frac == pytest.approx(1.0)

    def test_empty_yields_zero(self):
        n_ok, n_total, frac = compute_yield([])
        assert n_ok == 0
        assert n_total == 0
        assert frac == pytest.approx(0.0)

    def test_custom_threshold(self):
        trajs = [
            _make_traj(0, 1000, panel_chirality=0.04),
            _make_traj(1, 1001, panel_chirality=0.06),
        ]
        n_ok, n_total, frac = compute_yield(trajs, chirality_threshold=0.05)
        assert n_ok == 1
        assert n_total == 2

    def test_at_threshold_counts_as_success(self):
        """Chirality == threshold is inclusive (same as harvest)."""
        trajs = [_make_traj(0, 1000, panel_chirality=0.1)]
        n_ok, n_total, frac = compute_yield(trajs, chirality_threshold=0.1)
        assert n_ok == 1


# ── Integration: end-to-end with mock trajectories ────────────────────────────

class TestEndToEndMock:

    def test_five_trajectories_two_clean(self):
        trajs = [
            _make_traj(0, 1000, panel_chirality=0.02, panel_iptm=0.8, panel_composite=0.75,
                       d_fasta="(DAL)G(DAL)", l_seq="AGA"),
            _make_traj(1, 1001, panel_chirality=0.35),  # dirty
            _make_traj(2, 1002, panel_chirality=0.08, panel_iptm=0.65, panel_composite=0.55,
                       d_fasta="(DLE)G", l_seq="LG"),
            _make_traj(3, 1003, panel_chirality=0.25),  # dirty
            _make_traj(4, 1004, error="OOM"),            # error
        ]
        clean = harvest_clean_designs(trajs)
        n_ok, n_total, frac = compute_yield(trajs)

        assert n_total == 4  # 4 non-error
        assert n_ok == 2
        assert frac == pytest.approx(0.5)
        assert len(clean) == 2
        # Best design should be traj_id=0 (higher composite)
        assert clean[0]["traj_id"] == 0
        assert clean[0]["composite"] == pytest.approx(0.75)

    def test_all_error_trajectories(self):
        trajs = [_make_traj(i, 1000 + i, error="crash") for i in range(3)]
        clean = harvest_clean_designs(trajs)
        n_ok, n_total, frac = compute_yield(trajs)
        assert len(clean) == 0
        assert n_total == 0
        assert n_ok == 0
        assert frac == pytest.approx(0.0)

    def test_single_best_design_is_top_ranked(self):
        trajs = [
            _make_traj(0, 1000, panel_chirality=0.0, panel_composite=0.9,
                       panel_iptm=0.92, d_fasta="(DAL)G(DLE)", l_seq="AGL"),
            _make_traj(1, 1001, panel_chirality=0.12),  # just over threshold
        ]
        clean = harvest_clean_designs(trajs)
        assert len(clean) == 1
        assert clean[0]["l_seq"] == "AGL"
        assert clean[0]["chirality"] == pytest.approx(0.0)
