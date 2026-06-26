"""GPU Tier-0a chirality gate — the go/no-go for the whole sub-project (spec §3).

Run on your local GPU:  pytest tests/gpu/test_chai_chirality_gate_gpu.py -m gpu -v

This runnable example uses a small all-D peptide and asks: does Chai PREDICT its residues
with correct D chirality? A PASS (violation fraction well below 0.51) is the minimal
go/no-go signal. For the rigorous gate, ADD real MONDE-T D-containing cases (e.g. PDB
7QDI D-AIB-310, polytheonamide B) as additional GateCase entries with `ref_backbone`
populated from the experimental structure (see GPU_TESTS.md).
"""
import pytest

from tests.gpu.conftest import require_chai, require_cuda


@pytest.mark.gpu
def test_tier0a_gate_all_D_peptide(tmp_path):
    require_cuda()
    require_chai()
    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign.eval.gate_tier0a import GateCase, run_gate

    peptide_seq = "ACDEFGHIK"  # 9 residues, all designed as D
    cases = [
        GateCase(
            name="all_D_smoke",
            entities=[
                {"type": "protein", "name": "target",
                 "sequence": "GSHMKVLITGGAGFIGSHLVDRL", "chirality": "L"},
                {"type": "protein", "name": "binder",
                 "sequence": peptide_seq, "chirality": "D"},
            ],
            design_labels=["D"] * len(peptide_seq),
            design_chain="binder",
            ref_backbone=None,  # no experimental reference -> phi/psi diagnostic skipped
        ),
    ]
    backend = ChaiBackend(device="cuda:0", seed=0)
    overall, per_case = run_gate(cases, backend, tmp_path)

    print(f"chirality violation fraction = {overall.chirality_violation_frac:.3f}")
    assert overall.passed, (
        f"GATE FAILED: chirality violation fraction "
        f"{overall.chirality_violation_frac:.3f} >= 0.51 — Chai does not preserve D "
        f"chirality in prediction; reconsider the route (spec §3)."
    )
