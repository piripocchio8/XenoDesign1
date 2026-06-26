"""End-to-end GPU test for scripts/design_alpha.py — run_alpha_design().

Codifies the first network-free α-case run: tiny params (n_iters=2, num_seqs=1),
fixed seed_seq bypassing PepMLM (use_pepmlm=False), no chirality gate.

Run with:
    pytest tests/gpu/test_design_alpha_gpu.py -m gpu -v -s
"""
from __future__ import annotations

import pytest

from tests.gpu.conftest import require_chai, require_cuda

# Fixed 21-char L seed containing a G (passes Chai's ≥1-canonical-residue guard).
_SEED_SEQ = "ACDEFGHIKLMNPQRSTVWYA"


@pytest.mark.gpu
def test_design_alpha_end_to_end(tmp_path):
    """End-to-end run of run_alpha_design() with tiny, network-free params.

    Wires the full α-case pipeline (seed → double-flip → HalluLoop × 2 →
    panel selection → interface metrics) with the smallest valid inputs:
      - seed_seq bypasses PepMLM (no network)
      - n_iters=2, num_seqs=1 (minimal compute)
      - chirality_gate=False (no gate overhead)

    Asserts on the returned result dict:
      - selected_l_seq has length 21 (matches case.binder_length)
      - trajectory has length 2 (one entry per iteration)
      - 'metrics' key is present (may be None if case_metrics fails gracefully)
      - if metrics is not None, its 'metrics' sub-dict exposes
        ipae_mean, ipsae, interface_iptm
      - beats_baseline is a bool
      - selected_chirality is a float in [0, 1]
    """
    require_cuda()
    require_chai()

    from scripts.design_alpha import run_alpha_design

    result = run_alpha_design(
        n_iters=2,
        num_seqs=1,
        device="cuda:0",
        use_pepmlm=False,
        seed_seq=_SEED_SEQ,
        chirality_gate=False,
        out_dir=tmp_path,
    )

    # ── Print key numbers ──────────────────────────────────────────────────────
    print(f"\n[test_design_alpha] selected_l_seq     : {result['selected_l_seq']}")
    print(f"[test_design_alpha] selected_iptm      : {result['selected_iptm']:.4f}")
    print(f"[test_design_alpha] selected_chirality : {result['selected_chirality']:.4f}")
    print(f"[test_design_alpha] beats_baseline     : {result['beats_baseline']}")
    print(f"[test_design_alpha] trajectory length  : {len(result['trajectory'])}")
    if result.get("metrics") is not None:
        m = result["metrics"].get("metrics", {})
        print(f"[test_design_alpha] ipae_mean          : {m.get('ipae_mean')}")
        print(f"[test_design_alpha] ipsae              : {m.get('ipsae')}")
        print(f"[test_design_alpha] interface_iptm     : {m.get('interface_iptm')}")
    else:
        print("[test_design_alpha] metrics            : None (case_metrics unavailable)")

    # ── Structural assertions ──────────────────────────────────────────────────

    # selected_l_seq must be 21 characters (= case.binder_length)
    assert isinstance(result["selected_l_seq"], str), (
        f"selected_l_seq is not a str: {type(result['selected_l_seq'])}"
    )
    assert len(result["selected_l_seq"]) == 21, (
        f"selected_l_seq length {len(result['selected_l_seq'])} != 21; "
        f"got: {result['selected_l_seq']!r}"
    )

    # trajectory must have exactly n_iters=2 entries
    assert isinstance(result["trajectory"], list), (
        f"trajectory is not a list: {type(result['trajectory'])}"
    )
    assert len(result["trajectory"]) == 2, (
        f"trajectory length {len(result['trajectory'])} != 2"
    )

    # 'metrics' key must be present (value may be None)
    assert "metrics" in result, "result dict missing 'metrics' key"

    # if metrics is not None, required sub-keys must be present
    if result["metrics"] is not None:
        sub = result["metrics"].get("metrics", {})
        for key in ("ipae_mean", "ipsae", "interface_iptm"):
            assert key in sub, (
                f"metrics['metrics'] missing key {key!r}; "
                f"available keys: {list(sub.keys())}"
            )

    # beats_baseline must be a bool
    assert isinstance(result["beats_baseline"], bool), (
        f"beats_baseline is not a bool: {type(result['beats_baseline'])!r} "
        f"= {result['beats_baseline']!r}"
    )

    # FIX #2: the HONEST 3-criterion gate (ipTM margin AND ipAE<10 AND chirality<=0.10) is
    # reported alongside the ipTM-only beats_baseline, and is also a bool.
    assert isinstance(result["beats_baseline_full"], bool), (
        f"beats_baseline_full is not a bool: {type(result['beats_baseline_full'])!r} "
        f"= {result['beats_baseline_full']!r}"
    )
    # The full gate can only be True if the ipTM-only gate is True (it's a strict superset).
    if result["beats_baseline_full"]:
        assert result["beats_baseline"], (
            "beats_baseline_full True but ipTM-only beats_baseline False — impossible")

    # selected_chirality must be a float in [0, 1]
    chir = result["selected_chirality"]
    assert isinstance(chir, float), (
        f"selected_chirality is not a float: {type(chir)!r} = {chir!r}"
    )
    assert 0.0 <= chir <= 1.0, (
        f"selected_chirality out of range [0, 1]: {chir:.4f}"
    )


@pytest.mark.gpu
def test_design_alpha_off_by_one_selected_seq_matches_cif(tmp_path):
    """#31 ACCEPTANCE: selected_l_seq MUST equal the chain-B sequence of the SELECTED iter's
    chai_out (the sequence that actually produced the scored metrics). Run unrestrained +
    no-PLL for speed; the off-by-one fix is independent of those features.
    """
    require_cuda()
    require_chai()

    from scripts.design_alpha import binder_seq_from_cif, run_alpha_design
    from scripts.design_demo import _best_cif_path

    result = run_alpha_design(
        n_iters=2, num_seqs=1, device="cuda:0",
        use_pepmlm=False, seed_seq=_SEED_SEQ, chirality_gate=False,
        restraints=False, use_pll=False, out_dir=tmp_path,
    )

    sel_idx = result["selected_iter"]
    sel_cif = _best_cif_path(tmp_path / "loop" / f"iter_{sel_idx:03d}" / "chai_out")
    cif_seq = binder_seq_from_cif(sel_cif, "B")

    print(f"\n[off-by-one] selected_iter      : {sel_idx}")
    print(f"[off-by-one] selected_l_seq     : {result['selected_l_seq']}")
    print(f"[off-by-one] chain-B of sel CIF : {cif_seq}")

    assert result["selected_l_seq"] == cif_seq, (
        f"#31 off-by-one: selected_l_seq {result['selected_l_seq']!r} != chain-B of the "
        f"selected iter's CIF {cif_seq!r}")
    # And every trajectory l_seq must match its own iter's chain-B CIF sequence.
    for t in result["trajectory"]:
        ic = _best_cif_path(tmp_path / "loop" / f"iter_{t['iter']:03d}" / "chai_out")
        assert t["l_seq"] == binder_seq_from_cif(ic, "B"), (
            f"trajectory iter {t['iter']} l_seq mismatched its scored CIF")


@pytest.mark.gpu
def test_design_alpha_restrained_threads_constraint(tmp_path, monkeypatch):
    """TASK 3 ACCEPTANCE: a restrained run writes the α restraint and passes constraint_path to
    EVERY chai call (L-seed predict + every per-iter predict). --no_restraints (restraints=False)
    reverts to the truncated_refine path. Spies on ChaiBackend to record calls without changing
    the science.
    """
    require_cuda()
    require_chai()

    from xenodesign.backends.chai_backend import ChaiBackend

    calls = {"predict": [], "truncated_refine": []}
    orig_predict = ChaiBackend.predict
    orig_trunc = ChaiBackend.truncated_refine

    def spy_predict(self, entities, out_dir, num_diffn_timesteps=200, constraint_path=None):
        calls["predict"].append(constraint_path)
        return orig_predict(self, entities, out_dir,
                            num_diffn_timesteps=num_diffn_timesteps,
                            constraint_path=constraint_path)

    def spy_trunc(self, structure, ref_time_steps, out_dir):
        calls["truncated_refine"].append(out_dir)
        return orig_trunc(self, structure, ref_time_steps, out_dir)

    monkeypatch.setattr(ChaiBackend, "predict", spy_predict)
    monkeypatch.setattr(ChaiBackend, "truncated_refine", spy_trunc)

    from scripts.design_alpha import run_alpha_design

    result = run_alpha_design(
        n_iters=2, num_seqs=1, device="cuda:0",
        use_pepmlm=False, seed_seq=_SEED_SEQ, chirality_gate=False,
        restraints=True, use_pll=False, out_dir=tmp_path,
    )

    # The restraint file was written and recorded in the result.
    cpath = result["constraint_path"]
    assert cpath is not None and (tmp_path / "alpha.restraints").exists()
    # Restrained → PREDICT mode: NO truncated_refine calls; every predict carries the path.
    assert calls["truncated_refine"] == [], (
        "restrained run must NOT use truncated_refine (no constraint support there)")
    assert len(calls["predict"]) >= 3, "expected L-seed predict + >=2 per-iter predicts"
    assert all(str(cp) == cpath for cp in calls["predict"]), (
        f"every chai predict call must carry the constraint path; got {calls['predict']}")

    # FIX #27 crash fix (the REAL proof this test now exercises): the run reached this point,
    # which means the L-seed predict + every per-iter predict LOADED the constraint without the
    # name-match assertion crashing. That is only possible because the α pin is now a POCKET
    # (binder = chain-level side, res_idxA EMPTY -> no identity check on the DESIGNED binder),
    # with the FIXED target anchor token carrying its REAL one-letter code. A contact (the old
    # behaviour) would have crashed at the very first predict on the 'X'/UNK binder anchor.
    from xenodesign.benchmark.restraints import parse_restraints

    rows = parse_restraints(tmp_path / "alpha.restraints")
    assert len(rows) == 1
    row = rows[0]
    assert row["connection_type"] == "pocket", (
        f"the α pin MUST be a pocket (#27 crash fix), got {row['connection_type']!r}")
    assert row["res_idxA"] == "", (
        f"binder (chain-level A) res_idxA MUST be empty, got {row['res_idxA']!r}")
    assert row["res_idxB"] and not row["res_idxB"].startswith("X"), (
        f"target token must carry a REAL one-letter code (asserted by chai against the "
        f"structure), got {row['res_idxB']!r}")
