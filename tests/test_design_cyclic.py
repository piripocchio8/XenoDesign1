"""CPU tests for the 6UFA cyclic Zn-macrocycle design driver (task #9).

All tests here are pure-CPU: they exercise the seed construction, the mixed-chirality
FASTA builder, the Zn-ligand FASTA emission, the metal-coordination restraint wiring,
and the RMSD-to-deposit / Zn-N geometry scorers on SYNTHETIC coordinates. Nothing here
touches chai/torch (the GPU path is `run_cyclic_design`, marked elsewhere / not imported).

The unpublished design sequences are never inlined; the 6UFA DEPOSIT sequence is public
(RCSB 6UFA) but we still avoid hard-coding coords — synthetic arrays drive the geometry
tests so they are deterministic and deposit-file-independent.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.design_cyclic import (
    CYCLIC_HIS_CHIRALITY,
    ZN_SMILES,
    backbone_rmsd_to_deposit,
    build_closure_row,
    build_cyclic_input_fasta,
    build_cyclic_restraint_rows,
    build_cyclic_seed,
    mixed_chirality_fasta,
    write_cyclic_restraints,
    zn_coordination_geometry,
)
from xenodesign.benchmark.cases import get_case


# ── Seed construction (insert_fixed_chirality at the case His positions) ────────

def test_build_cyclic_seed_length_and_his_placement():
    case = get_case("cyclic")
    seed = build_cyclic_seed(case, seed_seq="ACDEFGHIKLMNACDEFGHIKLMN")  # len 24, deterministic
    # length preserved, His placed at the 4 deposit coordinating positions (1-based 6/12/18/24)
    assert seed.length == case.binder_length == 24
    assert len(seed.one_letter) == 24
    for pos in (6, 12, 18, 24):
        assert seed.one_letter[pos - 1] == "H"
    # fixed_chirality records the per-position handedness exactly as the policy table.
    assert seed.fixed_chirality == CYCLIC_HIS_CHIRALITY
    assert seed.conditioned is False  # cyclic is the unconditioned path (no metal-aware pLM)


def test_build_cyclic_seed_random_is_deterministic_with_seed():
    case = get_case("cyclic")
    a = build_cyclic_seed(case, rng_seed=7)
    b = build_cyclic_seed(case, rng_seed=7)
    assert a.one_letter == b.one_letter
    # His still pinned at the coordinating positions even for a random backbone seed.
    for pos in CYCLIC_HIS_CHIRALITY:
        assert a.one_letter[pos - 1] == "H"


def test_build_cyclic_seed_rejects_wrong_length_explicit_seed():
    case = get_case("cyclic")
    with pytest.raises(ValueError):
        build_cyclic_seed(case, seed_seq="TOOSHORT")  # != binder_length 12


# ── Mixed-chirality FASTA (per-position L vs D, NOT all-D) ──────────────────────

def test_mixed_chirality_fasta_places_d_only_at_d_positions():
    # 4-mer seq, His at all 4; mark positions 2 and 4 as D, 1 and 3 as L.
    seq = "HHHH"
    fixed = {1: "L", 2: "D", 3: "L", 4: "D"}
    out = mixed_chirality_fasta(seq, fixed)
    # L-His stays bare 'H'; D-His becomes the parenthesized D-CCD block (DHI).
    assert out == "H(DHI)H(DHI)"


def test_mixed_chirality_fasta_unmarked_positions_default_l():
    # Positions not in `fixed` are L (bare canonical) — this is the LINEAR phase-1 seed
    # where only the coordinating His are pinned to a handedness.
    seq = "ACDH"
    fixed = {4: "D"}  # only the His is D
    out = mixed_chirality_fasta(seq, fixed)
    assert out == "ACD(DHI)"


def test_mixed_chirality_fasta_glycine_stays_bare():
    # Glycine is achiral: stays a single 'G' regardless of any handedness request.
    out = mixed_chirality_fasta("GHG", {1: "D", 2: "L", 3: "D"})
    assert out == "GHG"


def test_mixed_chirality_fasta_passes_ncaa_blocks_through():
    # track #2: a Variant-B identity may already carry an ncAA as a (XXX) block; it must be
    # emitted verbatim (chai's modified-residue contract), not looked up as a 1-letter code.
    out = mixed_chirality_fasta("A(AIB)C", {1: "L", 3: "L"})
    assert out == "A(AIB)C"


def test_mixed_chirality_fasta_ncaa_block_unaffected_by_chirality_marks():
    # An ncAA block is one position; a D mark on a neighbouring canonical still applies, but the
    # (XXX) block itself is passed through unchanged (D-ncAA, if any, is already encoded in it).
    out = mixed_chirality_fasta("(NLE)H", {1: "L", 2: "D"})
    assert out == "(NLE)(DHI)"


# ── Zn-ligand FASTA emission (the metal/HETATM context) ─────────────────────────

def test_build_cyclic_input_fasta_has_protein_and_zn_ligand():
    fasta = build_cyclic_input_fasta(
        binder_mixed_seq="H(DHI)H(DHI)", binder_name="binder", zn_name="zn"
    )
    lines = fasta.strip().splitlines()
    # protein chain first (so chai labels it chain A), Zn ligand second (chain B).
    assert lines[0] == ">protein|binder"
    assert lines[1] == "H(DHI)H(DHI)"
    assert lines[2] == ">ligand|name=zn"
    assert lines[3] == ZN_SMILES  # the zinc SMILES, e.g. '[Zn+2]'


def test_zn_smiles_is_zinc_ion():
    # The cofactor enters Chai as a SMILES ligand; zinc(II) is '[Zn+2]'.
    assert ZN_SMILES == "[Zn+2]"


# ── Metal-coordination restraint wiring (His<->Zn, via build_for_case) ──────────

def test_build_cyclic_restraint_rows_one_contact_per_his():
    case = get_case("cyclic")
    rows = build_cyclic_restraint_rows(case)
    # 4 coordinating His in the full 24-mer (6/12/18/24, L/D/L/D) -> 4 inter-chain contact rows
    # (His chain A <-> Zn chain B).
    assert len(rows) == 4
    his_resnums = case.restraint.params["his_resnums"]
    assert len(his_resnums) == 4
    for row, hr in zip(rows, his_resnums):
        cols = row.split(",")
        # chainA = His chain, res token '<H><resnum>'; chainB = Zn metal chain.
        assert cols[0] == case.restraint.params["his_chain"]
        assert cols[1] == f"H{hr}"
        assert cols[2] == case.restraint.params["metal_chain"]
        assert cols[4] == "contact"


# ── Backbone heavy-atom RMSD to the deposit (the RECALL metric) ─────────────────

def test_backbone_rmsd_zero_for_identical_coords():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(12, 3))
    assert backbone_rmsd_to_deposit(coords, coords) == pytest.approx(0.0, abs=1e-9)


def test_backbone_rmsd_invariant_under_rigid_motion():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(12, 3))
    # rotate + translate b; Kabsch-aligned RMSD must be ~0.
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    b = a @ rot.T + np.array([3.0, -2.0, 5.0])
    assert backbone_rmsd_to_deposit(a, b) == pytest.approx(0.0, abs=1e-6)


def test_backbone_rmsd_positive_for_perturbed_coords():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(12, 3))
    b = a + rng.normal(scale=0.5, size=(12, 3))
    assert backbone_rmsd_to_deposit(a, b) > 0.0


def test_backbone_rmsd_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        backbone_rmsd_to_deposit(np.zeros((12, 3)), np.zeros((10, 3)))


# ── Zn-N coordination geometry (secondary metric) ───────────────────────────────

def test_zn_coordination_geometry_tetrahedral_distances():
    # Put 4 N atoms at unit distance along tetrahedral directions; Zn at origin.
    zn = np.zeros(3)
    tetra = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True) * 2.0  # 2.0 A each
    geom = zn_coordination_geometry(zn, tetra)
    assert geom["n_coordinating"] == 4
    assert geom["mean_zn_n_distance"] == pytest.approx(2.0, abs=1e-6)
    assert geom["max_zn_n_distance"] == pytest.approx(2.0, abs=1e-6)
    # ideal tetrahedral angle ~109.47 deg
    assert geom["mean_n_zn_n_angle"] == pytest.approx(109.47, abs=0.5)


def test_zn_coordination_geometry_counts_within_cutoff():
    zn = np.zeros(3)
    # two N within 2.6 A, one beyond (4.0 A)
    ns = np.array([[2.0, 0, 0], [0, 2.5, 0], [0, 0, 4.0]], dtype=float)
    geom = zn_coordination_geometry(zn, ns, cutoff=2.6)
    assert geom["n_coordinating"] == 2  # the 4.0-A N is not counted as coordinating


def test_zn_coordination_geometry_empty_when_no_neighbors():
    zn = np.zeros(3)
    ns = np.array([[10.0, 0, 0]], dtype=float)
    geom = zn_coordination_geometry(zn, ns, cutoff=2.6)
    assert geom["n_coordinating"] == 0
    assert geom["mean_zn_n_distance"] is None  # nothing within cutoff to average


# ── His-position deposit-vs-case consistency guard (documents the discrepancy) ──

def test_case_his_positions_synced_to_deposit():
    # 2026-06-24: the registry models the FULL S2-symmetric 6UFA 24-mer. The four coordinating
    # His are 6/12/18/24 with chirality L/D/L/D (module docstring / DEPOSIT REALITY) — a single
    # 12-mer cannot make the 4-coordinate [Zn(His)4] site. This asserts the corrected positions
    # AND chirality, plus self-consistency between the seeding map and the restraint his_resnums
    # (so the two never drift apart again).
    case = get_case("cyclic")
    assert case.restraint.params["his_resnums"] == (6, 12, 18, 24)
    assert CYCLIC_HIS_CHIRALITY == {6: "L", 12: "D", 18: "L", 24: "D"}
    assert tuple(sorted(CYCLIC_HIS_CHIRALITY)) == case.restraint.params["his_resnums"]


# ── P2b: head-to-tail covalent closure (#23) ──────────────────────────────────────

def test_build_closure_row_is_covalent_n_to_c():
    case = get_case("cyclic")
    seed = build_cyclic_seed(case, seed_seq="ACDEFGHIKLMNACDEFGHIKLMN")   # His placed at 6/12/18/24
    row = build_closure_row(seed)
    cols = row.split(",")
    assert cols[4] == "covalent"
    # C-term residue (pos 24) carbonyl C  <->  N-term residue (pos 1) amide N.
    assert cols[1].endswith("24@C") and cols[3].endswith("1@N")
    assert cols[3].startswith(seed.one_letter[0])   # N-term residue one-letter


def test_write_cyclic_restraints_closure_appends_one_covalent(tmp_path):
    case = get_case("cyclic")
    seed = build_cyclic_seed(case, seed_seq="ACDEFGHIKLMNACDEFGHIKLMN")
    base = write_cyclic_restraints(case, tmp_path / "noclose", seed_result=seed, closure=False)
    closed = write_cyclic_restraints(case, tmp_path / "close", seed_result=seed, closure=True)
    n_base = sum(1 for _ in base.read_text().splitlines()[1:])     # minus header
    n_closed = sum(1 for _ in closed.read_text().splitlines()[1:])
    assert n_closed == n_base + 1                                  # exactly one closure bond
    assert "covalent" in closed.read_text() and "cyclic_closure" in closed.read_text()


# ── Part F: result provenance — recorded sequence must match the PANEL-selected step ──

def test_assemble_records_panel_selected_sequence_not_greedy(tmp_path):
    """The recorded selected_d_fasta must come from the PANEL-selected step (whose CIF is the
    deposited model), NOT the greedy highest-score step — mirroring the alpha path. Here the
    greedy best (highest score) is step 0, but the panel selects step 1."""
    from xenodesign.classes.cyclic import _assemble_cyclic_result
    from xenodesign.config import resolve_config
    from xenodesign.judges.panel import PanelResult, RefereeScore

    class _Pred:
        def __init__(self, iptm, ptm):
            self.iptm, self.ptm = iptm, ptm

    class _State:
        def __init__(self, d_fasta):
            self.d_fasta = d_fasta

    class _Step:
        def __init__(self, d_fasta, iptm, ptm, score):
            self.prediction = _Pred(iptm, ptm)
            self.state = _State(d_fasta)
            self.score = score

    # step 0 has the HIGHER score (greedy would pick it); step 1 is the panel pick.
    history = [_Step("(DHI)GREEDY", 0.9, 0.9, 0.99),
               _Step("(DHI)PANEL", 0.5, 0.5, 0.10)]
    raw = [RefereeScore(chirality_violation=0.0, iptm=0.9),
           RefereeScore(chirality_violation=0.0, iptm=0.5)]
    panel = PanelResult(selected_idx=1, composite_scores=[0.1, 0.9],
                        vetoed=[False, False], raw_scores=raw)

    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path))
    case = get_case("cyclic")
    result = _assemble_cyclic_result(cfg, history, panel_result=panel,
                                     case=case, out_dir=tmp_path)
    assert result["selected_d_fasta"] == "(DHI)PANEL"
    assert result["selected_iptm"] == 0.5  # ipTM read from the SAME (panel) step
