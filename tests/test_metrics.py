# tests/test_metrics.py
import os
from pathlib import Path

import numpy as np
import pytest

from xenodesign.metrics import interface_pae

# ---------------------------------------------------------------------------
# Paths for the α-anchor (gitignored real-data) test
# ---------------------------------------------------------------------------
_LOCAL_REF = Path(__file__).resolve().parents[1] / "XenoDesign1_local_ref" / "dl_able_ground_truth" / "chai_run_pae"
_ANCHOR_NPZ = _LOCAL_REF / "confidence.model_idx_1.npz"
_ANCHOR_CIF = _LOCAL_REF / "pred.model_idx_1.cif"
_ANCHOR_PRESENT = _ANCHOR_NPZ.exists() and _ANCHOR_CIF.exists()


def test_aggregate_tokens_to_residues_mean():
    from xenodesign.metrics import aggregate_tokens_to_residues
    # tokens 0,1 = residue (asym 0, resid 0) [atom-tokenized]; token 2 = residue (asym 1, resid 0)
    pae = np.array([
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 6.0],
        [4.0, 6.0, 0.0],
    ])
    asym = np.array([0, 0, 1])
    resid = np.array([0, 0, 0])
    R, res_asym = aggregate_tokens_to_residues(pae, resid, asym, reduce="mean")
    assert R.shape == (2, 2)
    assert list(res_asym) == [0, 1]
    assert R[0, 1] == 5.0   # mean(4,6)
    assert R[1, 0] == 5.0


def test_ipsae_perfect_and_cutoff():
    from xenodesign.metrics import ipsae
    res_pae = np.zeros((4, 4))
    res_asym = np.array([0, 0, 1, 1])
    assert abs(ipsae(res_pae, res_asym, 0, 1, pae_cutoff=10.0) - 1.0) < 1e-9
    res_pae_bad = np.full((4, 4), 30.0)
    np.fill_diagonal(res_pae_bad, 0.0)
    assert ipsae(res_pae_bad, res_asym, 0, 1, pae_cutoff=10.0) == 0.0


def test_interface_plddt_from_cif_fixture():
    from xenodesign.metrics import interface_plddt_from_cif
    from pathlib import Path
    cif = str(Path(__file__).resolve().parents[1] / "data/benchmark/chai_output_fixture/pred.model_idx_0.cif")
    out = interface_plddt_from_cif(cif, contact_dist=10.0)
    assert set(out) and all(0.0 <= v["interface_plddt"] <= 100.0 for v in out.values())
    for ch, v in out.items():
        assert 0 <= v["n_interface"] <= v["n_residues"]
        assert 0.0 <= v["chain_plddt"] <= 100.0


def test_load_confidence_roundtrip(tmp_path):
    from xenodesign.metrics import load_confidence
    f = tmp_path / "confidence.model_idx_0.npz"
    np.savez(f, pae=np.zeros((3, 3)),
             token_asym_id=np.array([0, 0, 1]), token_residue_index=np.array([0, 0, 0]))
    c = load_confidence(f)
    assert c["pae"].shape == (3, 3)
    assert list(c["token_asym_id"]) == [0, 0, 1]
    assert list(c["token_residue_index"]) == [0, 0, 0]


def test_interface_pae_two_chain():
    # 4 tokens: tokens 0,1 = chain 0 ; tokens 2,3 = chain 1
    pae = np.array([
        [0.0, 1.0, 8.0, 9.0],
        [1.0, 0.0, 7.0, 6.0],
        [8.0, 7.0, 0.0, 2.0],
        [9.0, 6.0, 2.0, 0.0],
    ])
    asym = np.array([0, 0, 1, 1])
    out = interface_pae(pae, asym, 0, 1)
    # inter-chain block (both directions) = {8,9,7,6} twice -> mean 7.5, min 6.0
    assert out["ipae_mean"] == 7.5
    assert out["ipae_min"] == 6.0


def test_ipsae_nontrivial_value():
    from xenodesign.metrics import ipsae
    # 1 residue per chain, inter-chain PAE = 5.0, cutoff 10 → n_valid=1 → d0=1.0
    # ptm_term = 1/(1+(5/1)^2) = 1/26 ≈ 0.038462
    res_pae = np.array([[0.0, 5.0], [5.0, 0.0]])
    res_asym = np.array([0, 1])
    assert abs(ipsae(res_pae, res_asym, 0, 1, pae_cutoff=10.0) - (1.0 / 26.0)) < 1e-6


def test_ipsae_token_perfect_and_cutoff():
    from xenodesign.metrics import ipsae_token
    # All inter-chain PAE = 0 → every ptm-term = 1.0 → ipSAE = 1.0.
    pae = np.zeros((4, 4))
    asym = np.array([0, 0, 1, 1])
    assert abs(ipsae_token(pae, asym, 0, 1, pae_cutoff=10.0) - 1.0) < 1e-9
    # No inter-chain pair below cutoff → 0.0.
    pae_bad = np.full((4, 4), 30.0)
    np.fill_diagonal(pae_bad, 0.0)
    assert ipsae_token(pae_bad, asym, 0, 1, pae_cutoff=10.0) == 0.0


def test_ipsae_token_d0_uses_token_partner_count():
    """The canonical per-token form must derive d0 from the per-token partner COUNT, so a
    source token facing many low-PAE partner tokens scores far higher than the same PAE under
    a 1-partner d0 (which is what the residue-aggregated form collapses to). This is the exact
    mechanism behind the GT D-binder scale jump (task #32)."""
    from xenodesign.metrics import ipsae_token, _d0, _ptm_term
    # chain A = 1 token (id 0); chain B = 40 tokens (id 1). The single A token sees 40 partners
    # all at PAE = 2.0 → n_valid = 40 → d0 = _d0(40) ≈ 1.83 (> 1.0), so each ptm-term is lifted
    # above the n_valid=1 (d0=1.0) collapse. n_b chosen > 26 so _d0 clears 1.0 (see _d0 formula).
    n_b = 40
    assert _d0(n_b) > 1.0, "n_b must be large enough that d0 exceeds the 1.0 clamp"
    pae = np.full((1 + n_b, 1 + n_b), 2.0)
    np.fill_diagonal(pae, 0.0)
    asym = np.array([0] + [1] * n_b)
    got = ipsae_token(pae, asym, 0, 1, pae_cutoff=10.0)
    expected = _ptm_term(2.0, _d0(n_b))
    assert abs(got - expected) < 1e-9
    # And it must be strictly larger than the n_valid=1 (d0=1.0) collapse the resagg form gives.
    assert got > _ptm_term(2.0, 1.0) + 1e-6


# ---------------------------------------------------------------------------
# New tests: token_maps_from_cif  (Step 3a — synthetic)
# ---------------------------------------------------------------------------

def _make_synthetic_cif(tmp_path: Path) -> Path:
    """Two-residue, two-chain CIF: chain A = DLY (D-Lysine, 9 heavy atoms),
    chain B = ALA (standard L, 1 token).  The CIF uses heavy atoms only (no H),
    matching real Chai output."""
    # DLY heavy atoms: N, CA, C, O, CB, CG, CD, CE, NZ  → 9 atoms
    dly_atoms = [
        ("N",  "N",  0.0,  0.0,  0.0),
        ("C",  "CA", 1.5,  0.0,  0.0),
        ("C",  "C",  2.0,  1.4,  0.0),
        ("O",  "O",  3.2,  1.5,  0.0),
        ("C",  "CB", 1.5, -0.5,  1.3),
        ("C",  "CG", 1.5, -0.5,  2.8),
        ("C",  "CD", 1.5, -0.5,  4.3),
        ("C",  "CE", 1.5, -0.5,  5.8),
        ("N",  "NZ", 1.5, -0.5,  7.3),
    ]
    # ALA heavy atoms: N, CA, C, O, CB  → 5 atoms  (standard L; 1 token)
    ala_atoms = [
        ("N",  "N",  10.0, 0.0, 0.0),
        ("C",  "CA", 11.5, 0.0, 0.0),
        ("C",  "C",  12.0, 1.4, 0.0),
        ("O",  "O",  13.2, 1.5, 0.0),
        ("C",  "CB", 11.5, -0.5, 1.3),
    ]
    lines = [
        "data_test",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_seq_id",
        "_atom_site.label_asym_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.B_iso_or_equiv",
    ]
    atom_id = 1
    # chain A, residue 1, DLY
    for elem, atom_name, x, y, z in dly_atoms:
        lines.append(f"ATOM {atom_id} {elem} {atom_name} . DLY 1 A {x:.3f} {y:.3f} {z:.3f} 50.0")
        atom_id += 1
    # chain B, residue 1, ALA
    for elem, atom_name, x, y, z in ala_atoms:
        lines.append(f"ATOM {atom_id} {elem} {atom_name} . ALA 1 B {x:.3f} {y:.3f} {z:.3f} 50.0")
        atom_id += 1
    lines.append("#")
    cif_path = tmp_path / "synthetic.cif"
    cif_path.write_text("\n".join(lines) + "\n")
    return cif_path


def test_token_maps_from_cif_synthetic(tmp_path):
    """DLY (9 atoms) on chain A → 9 tokens; ALA on chain B → 1 token.
    Residue indices: DLY=0, ALA=1.  Chain ids: A=0, B=1."""
    from xenodesign.metrics import token_maps_from_cif

    cif_path = _make_synthetic_cif(tmp_path)
    asym, residx = token_maps_from_cif(cif_path)

    # Total tokens = 9 (DLY) + 1 (ALA)
    assert len(asym) == 10, f"expected 10 tokens, got {len(asym)}"
    assert len(residx) == 10

    # First 9 tokens: chain A (id 0), residue index 0
    assert list(asym[:9]) == [0] * 9, f"asym[:9]={list(asym[:9])}"
    assert list(residx[:9]) == [0] * 9, f"residx[:9]={list(residx[:9])}"

    # Last token: chain B (id 1), residue index 1
    assert asym[9] == 1, f"asym[9]={asym[9]}"
    assert residx[9] == 1, f"residx[9]={residx[9]}"


# ---------------------------------------------------------------------------
# Step 3b — α-anchor test on real Chai output (skipped if files absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _ANCHOR_PRESENT,
    reason="XenoDesign1_local_ref/dl_able_ground_truth/chai_run_pae/ files not present",
)
def test_score_interface_cif_derived_anchor():
    """End-to-end on the real GT D-binder: score_interface falls back to CIF-derived token maps
    when the npz only has 'pae'.

    ipSAE reconciliation (task #32). Three numbers exist for this complex:
      * 0.015  — the OLD residue-aggregated 'ipsae' (d0 collapses; under-scaled, the bug).
      * ~0.22  — the canonical per-TOKEN Dunbrack ipSAE on the TRUE chain split (asym 0/1, the
                 chain-A/B boundary at token 151). This is what score_interface now reports as
                 'ipsae_cut10'/'ipsae'.
      * ~0.73  — the value in the committed pae_summary.json / cases.py baseline. It comes from
                 run_groundtruth_pae.py splitting chains at the first NA=21 *tokens*, which falls
                 INSIDE the atom-tokenized 151-token D-block (true boundary is 151, not 21) — i.e.
                 a chain-split artifact, NOT a valid interface ipSAE. See the subagent report.
    We assert the correct-scale per-token value here, not the 0.73 artifact.
    """
    from xenodesign.metrics import score_interface

    m = score_interface(str(_ANCHOR_NPZ), str(_ANCHOR_CIF), chain_a=0, chain_b=1)
    print(f"\nα-anchor: ipae_mean={m['ipae_mean']:.4f}, ipsae_cut10={m['ipsae_cut10']:.4f}, "
          f"ipsae_cut15={m['ipsae_cut15']:.4f}, ipsae_resagg={m['ipsae_resagg']:.4f}")
    assert 11.5 <= m["ipae_mean"] <= 13.0, f"ipae_mean out of range: {m['ipae_mean']}"
    # canonical per-token Dunbrack ipSAE (true asym split): model_idx_1 ≈ 0.218.
    assert m["ipsae"] == m["ipsae_cut10"], "ipsae must alias ipsae_cut10"
    assert 0.18 <= m["ipsae_cut10"] <= 0.26, f"ipsae_cut10 out of range: {m['ipsae_cut10']}"
    assert m["ipsae_cut15"] >= m["ipsae_cut10"] - 1e-9, "cut15 admits ≥ as many pairs as cut10"
    assert 0.18 <= m["ipsae_cut15"] <= 0.30, f"ipsae_cut15 out of range: {m['ipsae_cut15']}"
    # the old residue-aggregated form is preserved for provenance and remains ~0.015.
    assert 0.0 <= m["ipsae_resagg"] <= 0.05, f"ipsae_resagg out of range: {m['ipsae_resagg']}"
