"""Real-PDB chirality benchmark — Tier-0a gate on experimental D-containing structures.

Run on your GPU (with open network for RCSB):
    pytest tests/gpu/test_chirality_benchmark_gpu.py -m "gpu and network" -v

For each curated PDB id: download the CIF from RCSB, find the chain(s) containing D-amino
acids (data-driven, via chai's D_partners), rebuild the Chai input from the *experimental*
residue codes, predict with Chai, and assert the predicted residues keep their D/L chirality
(violation fraction << 0.51, spec §3). Entries with no D residues are skipped.

NOTE (verify on your hardware): chain naming in Chai's predicted CIF may differ from the
experimental chain names (Chai relabels chains in entity order). If a case errors on chain
lookup, map the design chain to the corresponding predicted-CIF chain by order.

Extend `D_CONTAINING_PDBS` from MONDE-T (mondet.tuebingen.mpg.de) — filter its catalog to
chai-supported D codes (xenodesign.catalog.chai_supported_d_codes) and add their PDB ids.
"""
import pytest

from tests.gpu.conftest import require_chai, require_cuda

# Expanded benchmark set: 9 curated X-ray structures from the primary set in
# data/benchmark/d_codes/D_CONTAINING_PDBS.txt plus the extended MONDE-T pool, all confirmed
# to contain chai-supported D residues (DGL, DLY, DPR, DVA, DGN, DTH, DAR, DAS, DIL, DSN,
# DHI, DAL, DPN ⊂ D_TO_L) and to pass Chai 0.6.1 tokenization (≥1 canonical residue per chain).
#
# — Family A (jobs2 cyclic D/L peptides): 6UCX, 6UD9, 6UDR, 6UDZ, 6UF7, 6UDW, 6UF8
#     Single-chain, 6–10 aa, resolution 0.85–1.1 Å, computationally designed cyclic D/L.
#
# — Family B (diverse, non-cyclic-peptide structures):
#     8EF4: 20-aa linear D-Phe (DPN) peptide inhibitor, single chain, no ncAA.
#     2LOC: 24-aa linear scorpion-toxin-like peptide with D-Ala (DAL), single chain, no ncAA.
#
# Skipped / not selected from primary set:
#   7UBD/7UBF/7UBH: contain MLE (N-methylleucine) — not in chai's D_partners
#   7UBC/8CTO/8CWA: solution NMR (no single X-ray reference backbone)
#   6UGB/6UGC: 2–11 chains (multi-chain wall-clock cost without added chirality coverage)
#   6UF9/6UFA: 24 aa (4× the length vs primary set; adds time without new D-code types)
#   7UCP: contains MLE (N-methylleucine)
# Skipped from extended MONDE-T pool (diversity candidates evaluated but excluded):
#   7QDI: contains AIB (α-aminoisobutyric acid) and BF7 — not in chai's D_partners
#   1MAG/1GRM/1MIC: gramicidin A — contain FVA (formylvaline) and ETA — not in chai's D_partners
#   6UFU/6UG2: contain AIB — not in chai's D_partners
#   9B4I: 60 identical chains in the CIF (excessive multi-chain prediction load)
#   6B17: 6 identical chains of 13 aa in CIF (redundant; adds no new D-code types over 6UF8)
#   3LNJ: 11-aa all-D beta-hairpin inhibitor (chain B: DSG,DTR,DTY,DAL,DSG,DLE,DGL,DLY,DLE,
#          DLE,DAR) + 83-aa L-protein PCNA. The D-peptide chain is fully non-canonical (all-D,
#          no GLY or L residue). Chai 0.6.1 rejects chains with <1 canonical residue
#          (ValueError: "protein chain 'B' is fully non-canonical"). Would need mirror-into-L
#          path (spec §2.7) as a pre-processing step — deferred to the loop implementation.
D_CONTAINING_PDBS = [
    # Family A — jobs2 cyclic D/L peptides (existing 5 + 2 new from primary list)
    "6UCX", "6UD9", "6UDR", "6UDZ", "6UF7",
    "6UDW",   # 10 aa cyclic D/L: DGN, DTH, DAR, DPR, DAS (5 D-codes)
    "6UF8",   # 6 aa cyclic D/L: DAL, DLY, DHI, DGL (adds DHI to covered D-codes)
    # Family B — structurally diverse non-cyclic-peptide structures
    "8EF4",   # 20-aa linear D-Phe peptide inhibitor (DPN), single chain
    "2LOC",   # 24-aa linear peptide with D-Ala (DAL), scorpion-toxin scaffold
]


@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.parametrize("pdb_id", D_CONTAINING_PDBS)
def test_chai_preserves_chirality_on_real_structure(pdb_id, tmp_path):
    require_cuda()
    require_chai()
    pytest.importorskip("gemmi")

    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign.eval.gate_tier0a import GateCase, run_gate
    from xenodesign.pdb_extract import (
        chains_in_cif,
        chirality_labels,
        codes_to_entity_sequence,
        fetch_cif,
        has_d_residue,
        parse_cif_chain,
    )

    cif = fetch_cif(pdb_id, tmp_path)
    chains = list(dict.fromkeys(chains_in_cif(cif)))

    entities = []
    design_chain = None
    design_labels = None
    ref_backbone = None
    for ch in chains:
        codes, backbone = parse_cif_chain(cif, ch)
        if not codes:
            continue
        entities.append(
            {"type": "protein", "name": ch, "sequence": codes_to_entity_sequence(codes)}
        )
        if design_chain is None and has_d_residue(codes):
            design_chain = ch
            design_labels = chirality_labels(codes)
            ref_backbone = backbone

    if design_chain is None:
        pytest.skip(f"{pdb_id}: no D residues found in structure")

    # Chai relabels entities to chain A, B, … in entity order.  For single-chain PDBs
    # (experimental chain 'A', single entity), the predicted CIF also uses 'A' — no remapping
    # needed.  For multi-chain PDBs the position of design_chain in `chains` gives the index.
    design_chain_index = next(
        (i for i, e in enumerate(entities) if e["name"] == design_chain), 0
    )
    # Predicted chain names are A, B, C … by index (chai entity order).
    predicted_chain_name = chr(ord("A") + design_chain_index)

    # Override design_chain with the expected predicted-CIF chain name so run_gate can find it.
    case = GateCase(
        name=pdb_id,
        entities=entities,
        design_labels=design_labels,
        design_chain=predicted_chain_name,
        ref_backbone=ref_backbone,
    )
    overall, per_case = run_gate([case], ChaiBackend(device="cuda:0", seed=0), tmp_path)

    print(f"{pdb_id}: chirality violation = {overall.chirality_violation_frac:.3f}, "
          f"phi/psi violation = {overall.phi_psi_violation_frac:.3f}")
    assert overall.passed, (
        f"{pdb_id} FAILED Tier-0a: chirality violation "
        f"{overall.chirality_violation_frac:.3f} >= 0.51 (spec §3)."
    )
