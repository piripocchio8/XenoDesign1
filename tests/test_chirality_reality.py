"""CPU tests for the chirality-reality harness (#36).

Synthetic fixtures only (hand-made CIFs / coord arrays) — no GPU, no network.
Exercises:
  - trajectory_chirality_distribution: per-iter D-chirality distribution over a run dir,
    mean/max, pass-fraction (<=0.10), and per-iter Gly (achiral) fraction.
  - mirror_self_consistency: geometry-only discrepancy between a binder and its mirror.
  - _best_cif_in_chai_out: highest-aggregate-score model selection (degrades to lone CIF).

The only GPU/GT-touching test is skipped when the gitignored reference is absent.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from xenodesign.eval.chirality_reality import (
    backbone_chirality_fraction_from_cif,
    gly_fraction_from_cif,
    mirror_self_consistency,
    trajectory_chirality_distribution,
    _best_cif_in_chai_out,
)
from tests.conftest import IDEAL_L_ALA


# --------------------------------------------------------------------------------------
# Synthetic CIF builders. We build the structure via gemmi (so it parses back through
# backbone_by_residue_from_cif exactly like a real chai CIF), one binder chain (B).
# --------------------------------------------------------------------------------------

gemmi = pytest.importorskip("gemmi")  # CIF parsing dependency (present in CPU env)

# CB reflected across the z-axis gives a D-center; the L frame is the conftest ideal.
REFL_Z = np.diag([1.0, 1.0, -1.0])


def _mirror_frame(frame):
    return {k: v @ REFL_Z for k, v in frame.items()}


def _write_cif(path: Path, residues, chain_name="B"):
    """Build an mmCIF via gemmi and write it.

    residues: list of (comp_id, frame_dict) where frame_dict has N/CA/C(/CB) xyz.
    A residue with no 'CB' (e.g. GLY) is written without a CB atom.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    st = gemmi.Structure()
    st.name = "test"
    model = gemmi.Model("1")
    chain = gemmi.Chain(chain_name)
    for seq_id, (comp_id, frame) in enumerate(residues, start=1):
        res = gemmi.Residue()
        res.name = comp_id
        res.seqid = gemmi.SeqId(seq_id, " ")
        order = ["N", "CA", "C", "CB"] if "CB" in frame else ["N", "CA", "C"]
        for name in order:
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element(name[0])  # N or C; fine for backbone
            xyz = [float(v) for v in frame[name]]
            atom.pos = gemmi.Position(xyz[0], xyz[1], xyz[2])
            res.add_atom(atom)
        chain.add_residue(res)
    model.add_chain(chain)
    st.add_model(model)
    st.setup_entities()
    st.make_mmcif_document().write_file(str(path))
    return path


def _offset(frame, dx):
    """Translate a frame so successive residues don't overlap (gemmi is fine either way)."""
    return {k: np.asarray(v, float) + np.array([dx, 0.0, 0.0]) for k, v in frame.items()}


# --------------------------------------------------------------------------------------
# backbone_chirality_fraction_from_cif
# --------------------------------------------------------------------------------------

def test_all_D_binder_reads_zero_violation_when_labeled_D(tmp_path):
    """A binder whose every stereocenter is D, labeled all-D, has 0 D-violation fraction."""
    d = _mirror_frame(IDEAL_L_ALA)
    residues = [("DAL", _offset(d, 4.0 * i)) for i in range(5)]
    cif = _write_cif(tmp_path / "pred.cif", residues)
    frac = backbone_chirality_fraction_from_cif(cif, chain_name="B", chirality_label="D")
    assert frac == pytest.approx(0.0)


def test_all_L_binder_is_full_violation_when_labeled_D(tmp_path):
    """An L binder measured against the D label is 100% wrong (the survivorship trap)."""
    l = IDEAL_L_ALA
    residues = [("ALA", _offset(l, 4.0 * i)) for i in range(5)]
    cif = _write_cif(tmp_path / "pred.cif", residues)
    frac = backbone_chirality_fraction_from_cif(cif, chain_name="B", chirality_label="D")
    assert frac == pytest.approx(1.0)


def test_partial_D_binder_fraction(tmp_path):
    """3 D + 2 L stereocenters labeled all-D -> 2/5 = 0.4 violation fraction."""
    d = _mirror_frame(IDEAL_L_ALA)
    l = IDEAL_L_ALA
    frames = [d, d, d, l, l]
    residues = [("RES", _offset(f, 4.0 * i)) for i, f in enumerate(frames)]
    cif = _write_cif(tmp_path / "pred.cif", residues)
    frac = backbone_chirality_fraction_from_cif(cif, chain_name="B", chirality_label="D")
    assert frac == pytest.approx(0.4)


def test_glycine_residue_excluded_from_chirality_fraction(tmp_path):
    """A Gly (no CB) is achiral -> not counted as a stereocenter in the fraction."""
    d = _mirror_frame(IDEAL_L_ALA)
    gly = {k: v for k, v in IDEAL_L_ALA.items() if k != "CB"}
    # 2 D stereocenters + 1 Gly; labeled all-D -> 0 violations over 2 stereocenters.
    residues = [("DAL", _offset(d, 0)), ("GLY", _offset(gly, 4)), ("DAL", _offset(d, 8))]
    cif = _write_cif(tmp_path / "pred.cif", residues)
    frac = backbone_chirality_fraction_from_cif(cif, chain_name="B", chirality_label="D")
    assert frac == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# gly_fraction_from_cif  (achiral escape-hatch detector)
# --------------------------------------------------------------------------------------

def test_gly_fraction_counts_cb_less_residues(tmp_path):
    d = _mirror_frame(IDEAL_L_ALA)
    gly = {k: v for k, v in IDEAL_L_ALA.items() if k != "CB"}
    residues = [
        ("DAL", _offset(d, 0)), ("GLY", _offset(gly, 4)),
        ("GLY", _offset(gly, 8)), ("DAL", _offset(d, 12)),
    ]
    cif = _write_cif(tmp_path / "pred.cif", residues)
    assert gly_fraction_from_cif(cif, chain_name="B") == pytest.approx(0.5)


def test_gly_fraction_zero_when_no_glycine(tmp_path):
    d = _mirror_frame(IDEAL_L_ALA)
    residues = [("DAL", _offset(d, 4.0 * i)) for i in range(3)]
    cif = _write_cif(tmp_path / "pred.cif", residues)
    assert gly_fraction_from_cif(cif, chain_name="B") == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# _best_cif_in_chai_out
# --------------------------------------------------------------------------------------

def test_best_cif_picks_lone_cif_without_scores(tmp_path):
    """No scores.npz -> the single CIF present is returned (synthetic-fixture path)."""
    chai_out = tmp_path / "chai_out"
    chai_out.mkdir()
    cif = chai_out / "pred.model_idx_0.cif"
    cif.write_text("data_x\n")
    assert _best_cif_in_chai_out(chai_out) == cif


def test_best_cif_picks_highest_aggregate_score(tmp_path):
    """With scores.npz present, the highest-aggregate-score model's CIF is chosen."""
    chai_out = tmp_path / "chai_out"
    chai_out.mkdir()
    for idx, agg in [(0, 0.1), (1, 0.9), (2, 0.5)]:
        np.savez(
            chai_out / f"scores.model_idx_{idx}.npz",
            aggregate_score=np.array([agg]),
        )
        (chai_out / f"pred.model_idx_{idx}.cif").write_text("data_x\n")
    best = _best_cif_in_chai_out(chai_out)
    assert best.name == "pred.model_idx_1.cif"


def test_best_cif_missing_raises(tmp_path):
    chai_out = tmp_path / "chai_out"
    chai_out.mkdir()
    with pytest.raises(FileNotFoundError):
        _best_cif_in_chai_out(chai_out)


# --------------------------------------------------------------------------------------
# trajectory_chirality_distribution  (the anti-survivorship headline)
# --------------------------------------------------------------------------------------

def _make_run_dir(root: Path, per_iter_frames, chain="B"):
    """Build {root}/iter_000/chai_out/pred.cif ... from a list of per-iter residue lists.

    per_iter_frames[i] is a list of (comp_id, frame_dict) for iter i's binder chain.
    """
    for i, residues in enumerate(per_iter_frames):
        chai_out = root / f"iter_{i:03d}" / "chai_out"
        _write_cif(chai_out / "pred.model_idx_0.cif", residues, chain_name=chain)
    return root


def test_trajectory_distribution_all_clean(tmp_path):
    """Every iter all-D, labeled D -> distribution all-zeros, pass_fraction 1.0."""
    d = _mirror_frame(IDEAL_L_ALA)
    iters = [[("DAL", _offset(d, 4.0 * j)) for j in range(4)] for _ in range(3)]
    _make_run_dir(tmp_path, iters)
    out = trajectory_chirality_distribution(tmp_path, chirality_label="D")
    assert out["n_iters"] == 3
    assert out["per_iter"] == pytest.approx([0.0, 0.0, 0.0])
    assert out["mean"] == pytest.approx(0.0)
    assert out["max"] == pytest.approx(0.0)
    assert out["pass_fraction"] == pytest.approx(1.0)
    assert out["per_iter_gly_fraction"] == pytest.approx([0.0, 0.0, 0.0])


def test_trajectory_distribution_mixed_anti_survivorship(tmp_path):
    """Mixed run: some iters clean-D, some all-L. mean/max/pass_fraction reflect ALL iters,
    not just the best (the whole point of the harness)."""
    d = _mirror_frame(IDEAL_L_ALA)
    l = IDEAL_L_ALA
    clean = [("DAL", _offset(d, 4.0 * j)) for j in range(4)]   # 0.0 violation
    wrong = [("ALA", _offset(l, 4.0 * j)) for j in range(4)]   # 1.0 violation
    iters = [clean, wrong, clean, wrong]
    _make_run_dir(tmp_path, iters)
    out = trajectory_chirality_distribution(tmp_path, chirality_label="D")
    assert out["per_iter"] == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert out["mean"] == pytest.approx(0.5)
    assert out["max"] == pytest.approx(1.0)
    # pass threshold is <= 0.10; only the two clean iters pass -> 2/4.
    assert out["pass_fraction"] == pytest.approx(0.5)


def test_trajectory_distribution_pass_threshold_boundary(tmp_path):
    """A single iter at exactly 0.10 D-violation passes (<= 0.10, inclusive)."""
    d = _mirror_frame(IDEAL_L_ALA)
    l = IDEAL_L_ALA
    # 9 D + 1 L over 10 stereocenters -> 0.10 violation, exactly on the boundary.
    frames = [d] * 9 + [l]
    residues = [("RES", _offset(f, 4.0 * j)) for j, f in enumerate(frames)]
    _make_run_dir(tmp_path, [residues])
    out = trajectory_chirality_distribution(tmp_path, chirality_label="D")
    assert out["per_iter"][0] == pytest.approx(0.10)
    assert out["pass_fraction"] == pytest.approx(1.0)


def test_trajectory_distribution_tracks_gly_escape_hatch(tmp_path):
    """Per-iter Gly fraction is reported so an all-Gly 'achiral cheat' is visible."""
    d = _mirror_frame(IDEAL_L_ALA)
    gly = {k: v for k, v in IDEAL_L_ALA.items() if k != "CB"}
    iter0 = [("DAL", _offset(d, 0)), ("DAL", _offset(d, 4))]                  # 0 gly
    iter1 = [("GLY", _offset(gly, 0)), ("GLY", _offset(gly, 4))]             # all gly
    _make_run_dir(tmp_path, [iter0, iter1])
    out = trajectory_chirality_distribution(tmp_path, chirality_label="D")
    assert out["per_iter_gly_fraction"] == pytest.approx([0.0, 1.0])
    # An all-Gly iter has 0 stereocenters -> 0.0 chirality fraction; the Gly channel is
    # what exposes the escape hatch, not the (vacuously clean) chirality fraction.
    assert out["per_iter"][1] == pytest.approx(0.0)


def test_trajectory_distribution_empty_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        trajectory_chirality_distribution(tmp_path, chirality_label="D")


def test_trajectory_distribution_orders_iters_numerically(tmp_path):
    """iter_009 must sort before iter_010 (numeric, not lexical, ordering)."""
    d = _mirror_frame(IDEAL_L_ALA)
    l = IDEAL_L_ALA
    clean = [("DAL", _offset(d, 4.0 * j)) for j in range(3)]
    wrong = [("ALA", _offset(l, 4.0 * j)) for j in range(3)]
    # Build 11 iters: iters 0..8 clean, iter 9 wrong, iter 10 clean.
    iters = [clean] * 9 + [wrong, clean]
    _make_run_dir(tmp_path, iters)
    out = trajectory_chirality_distribution(tmp_path, chirality_label="D")
    assert out["n_iters"] == 11
    assert out["per_iter"][9] == pytest.approx(1.0)   # iter_009, not iter_010
    assert out["per_iter"][10] == pytest.approx(0.0)  # iter_010 is clean


# --------------------------------------------------------------------------------------
# mirror_self_consistency
# --------------------------------------------------------------------------------------

def test_mirror_self_consistency_zero_for_self_consistent(tmp_path):
    """A binder whose CIF coords are an exact mirror of a reference set -> ~0 discrepancy.

    Self-consistency here = the coord set compared against its own reflection-then-realign.
    We give a coord set that IS symmetric under reflect+realign by passing identical sets."""
    rng = np.random.RandomState(7)
    coords = rng.rand(20, 3)
    other = coords.copy()
    # mirror_self_consistency compares `coords` to the reflection of `other`; if other is
    # already the mirror of coords, discrepancy is ~0.
    from xenodesign.mirror import reflect_coords
    mirror_of_coords = reflect_coords(coords, axis=0)
    disc = mirror_self_consistency(coords, mirror_of_coords, axis=0)
    assert disc == pytest.approx(0.0, abs=1e-9)


def test_mirror_self_consistency_positive_for_inconsistent():
    rng = np.random.RandomState(11)
    coords = rng.rand(20, 3)
    other = rng.rand(20, 3)
    disc = mirror_self_consistency(coords, other, axis=0)
    assert disc > 0.1


def test_mirror_self_consistency_from_cif_single_arg(tmp_path):
    """Single-CIF form: read binder CA coords, compare to their own reflection.

    A perfectly mirror-symmetric (about the reflection axis after realign) chain scores ~0;
    a generic chain scores > 0. We just assert it returns a finite non-negative float."""
    d = _mirror_frame(IDEAL_L_ALA)
    residues = [("DAL", _offset(d, 4.0 * j)) for j in range(6)]
    cif = _write_cif(tmp_path / "pred.cif", residues)
    disc = mirror_self_consistency(cif, axis=0, chain_name="B")
    assert isinstance(disc, float)
    assert disc >= 0.0


def test_mirror_self_consistency_rigid_invariance():
    """Discrepancy is invariant to a rigid move of the mirror twin (Kabsch realigns)."""
    from xenodesign.mirror import reflect_coords
    rng = np.random.RandomState(13)
    coords = rng.rand(18, 3)
    twin = reflect_coords(coords, axis=0)
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    twin_moved = twin @ rot.T + np.array([5.0, -2.0, 1.0])
    disc = mirror_self_consistency(coords, twin_moved, axis=0)
    assert disc == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# Skip-if-absent ground-truth smoke test (gitignored DL-ABLE reference)
# --------------------------------------------------------------------------------------

_GT_RUN = Path(
    "/home/user/claude_projects/XenoDesign1/XenoDesign1_local_ref/"
    "dl_able_ground_truth/chai_run"
)


@pytest.mark.skipif(not _GT_RUN.exists(), reason="gitignored DL-ABLE GT reference absent")
def test_best_cif_on_real_gt_run_dir():
    """Smoke: _best_cif_in_chai_out selects a real chai model from the GT chai_run dir."""
    best = _best_cif_in_chai_out(_GT_RUN)
    assert best.exists()
    assert best.suffix == ".cif"
    # And it parses into per-residue backbones for the binder chain via the gate helper.
    from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif
    # DL-ABLE is a trimer; chain ids are chai-emitted. Just assert chain 'B' or 'A' parses.
    for chain in ("B", "A"):
        res = backbone_by_residue_from_cif(best, chain)
        if res:
            break
    assert res, "expected at least one parseable chain in the GT CIF"
