"""Re-gate scorer for the SINGLE-ANCHOR restrained 9DXX re-prediction.

Question: does one light POCKET anchor (peptide whole-chain <-> HA1 ASN21 = FASTA A:N13, 6 A) bring
the all-D DP93 peptide back onto its DEPOSITED pose? The unrestrained run found the right surface
patch but sat ~6 A off in orientation.

Method (placement RMSD, NOT a re-fit of the peptide):
  1. For each seed's best model, superpose the TWO RECEPTOR chains (pred A+B CA) onto the deposit
     (9dxx.cif chains A=HA1, B=HA2) by Kabsch (gemmi + numpy). The receptor frame is the anchor of
     comparison.
  2. Apply that SAME rigid transform to the predicted peptide (chain C) and measure chain-C CA RMSD
     vs the deposited peptide (chain E) WITHOUT any further peptide-only fitting. This is the
     placement RMSD: how far off the pose is once the receptor is aligned.
  3. Report chain-C-vs-receptor interface ipTM from scores.model_idx_*.npz (max over C-A and C-B
     pair entries of per_chain_pair_iptm).

Residue correspondence: deposit receptor chains A/B are an exact 0-offset prefix of the FASTA
HA1/HA2, so the k-th modeled receptor residue == FASTA position k; the predicted chains carry the
full FASTA, so we intersect on FASTA index. The peptide is 31 residues in both (same order) ->
positional CA correspondence i<->i.

Verdict: placement RMSD < ~5 A on the best/median model == anchor recovers the deposited pose.

CPU-only; gemmi + numpy. Run inside the container (has gemmi) or any env with gemmi+numpy:
  python scripts/gate_9dxx_restrained.py
  python scripts/gate_9dxx_restrained.py --self-test   # validate logic with deposit-vs-deposit (RMSD~0)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import gemmi
except ImportError:  # pragma: no cover
    gemmi = None

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "XenoDesign1_local_ref" / "9dxx_target_gate"
DEPOSIT = GATE / "9dxx.cif"
OUT_ROOT = REPO / "XenoDesign1_local_ref" / "benchmarks" / "restrained_single" / "9DXX" / "real"
SEEDS = (42, 43, 44)

AA3 = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
# Deposit receptor chain id -> predicted chain id (HA1->A, HA2->B). Peptide deposit E -> pred C.
RECEPTOR_DEP2PRED = {"A": "A", "B": "B"}
PEP_DEP, PEP_PRED = "E", "C"


def _ca(res):
    for a in res:
        if a.name == "CA":
            return np.array([a.pos.x, a.pos.y, a.pos.z])
    return None


def receptor_cas(model, chain_id: str, amino_only: bool):
    """Ordered list of CA coords for a chain's residues (amino acids only if amino_only)."""
    ch = next((c for c in model if c.name == chain_id), None)
    if ch is None:
        return []
    out = []
    for r in ch:
        if amino_only and r.name not in AA3:
            continue
        if r.name == "HOH":
            continue
        ca = _ca(r)
        if ca is not None:
            out.append(ca)
    return out


def peptide_cas(model, chain_id: str):
    """Ordered list of peptide CA coords (any residue with a CA, excluding waters/ligands w/o CA)."""
    ch = next((c for c in model if c.name == chain_id), None)
    if ch is None:
        return []
    out = []
    for r in ch:
        if r.name == "HOH":
            continue
        ca = _ca(r)
        if ca is not None:
            out.append(ca)
    return out


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid transform (R, t) that best maps P onto Q (minimise ||R P + t - Q||). Both (N,3)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def score_model(pred_model, dep_model, pred_pep_chain: str = PEP_PRED):
    """Receptor-superpose pred onto deposit, then return peptide placement RMSD (no peptide re-fit).

    pred_pep_chain lets the self-test reuse this with a deposit-as-prediction (peptide in chain E).
    """
    # Build matched receptor CA pairs by FASTA-index intersection. Deposit receptor chains are an
    # exact 0-offset prefix of the FASTA; predicted chains carry the full FASTA -> the k-th modeled
    # deposit residue corresponds to the k-th predicted residue (1..len(deposit_modeled)).
    P_list, Q_list = [], []
    for dep_id, pred_id in RECEPTOR_DEP2PRED.items():
        dep = receptor_cas(dep_model, dep_id, amino_only=True)
        pred = receptor_cas(pred_model, pred_id, amino_only=True)
        n = min(len(dep), len(pred))
        for i in range(n):
            P_list.append(pred[i])  # move the prediction
            Q_list.append(dep[i])  # onto the deposit
    P = np.array(P_list)
    Q = np.array(Q_list)
    R, t = kabsch(P, Q)

    # receptor fit RMSD (sanity)
    rec_rmsd = float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean()))

    # peptide placement RMSD: apply the SAME transform to predicted peptide, compare to deposit.
    dep_pep = np.array(peptide_cas(dep_model, PEP_DEP))
    pred_pep = np.array(peptide_cas(pred_model, pred_pep_chain))
    npep = min(len(dep_pep), len(pred_pep))
    moved = pred_pep[:npep] @ R.T + t
    place_rmsd = float(np.sqrt(((moved - dep_pep[:npep]) ** 2).sum(1).mean()))
    return {
        "receptor_pairs": len(P_list),
        "receptor_fit_rmsd": round(rec_rmsd, 3),
        "peptide_pairs": npep,
        "placement_rmsd": round(place_rmsd, 3),
    }


def interface_iptm(npz_path: Path):
    """chain-C-vs-receptor interface ipTM = max over C-A and C-B entries of per_chain_pair_iptm."""
    d = np.load(npz_path)
    if "per_chain_pair_iptm" not in d:
        return None, None, None
    m = np.asarray(d["per_chain_pair_iptm"]).reshape(3, 3)
    # chains order A,B,C -> indices 0,1,2; C is 2.
    cA = float(max(m[2, 0], m[0, 2]))
    cB = float(max(m[2, 1], m[1, 2]))
    return round(max(cA, cB), 3), round(cA, 3), round(cB, 3)


def best_model_idx(chai_out: Path):
    """Pick the model with the highest aggregate_score (chai's own ranking)."""
    best, best_score = None, -1.0
    for npz in sorted(chai_out.glob("scores.model_idx_*.npz")):
        d = np.load(npz)
        s = float(np.asarray(d["aggregate_score"]).flat[0]) if "aggregate_score" in d else -1.0
        if s > best_score:
            best, best_score = npz, s
    return best


def gate(seeds):
    assert gemmi is not None, "gemmi required"
    dep_model = gemmi.read_structure(str(DEPOSIT))[0]

    rows = []
    for seed in seeds:
        chai_out = OUT_ROOT / f"seed{seed}" / "chai_out"
        cifs = sorted(chai_out.glob("pred.model_idx_*.cif"))
        if not cifs:
            rows.append({"seed": seed, "status": "MISSING", "chai_out": str(chai_out)})
            continue
        best_npz = best_model_idx(chai_out)
        # match the CIF to the best npz by model index
        midx = best_npz.name.split("model_idx_")[1].split(".")[0]
        best_cif = chai_out / f"pred.model_idx_{midx}.cif"
        if not best_cif.is_file():
            best_cif = cifs[0]
        pred_model = gemmi.read_structure(str(best_cif))[0]
        geom = score_model(pred_model, dep_model)
        iface, cA, cB = interface_iptm(best_npz)
        rows.append(
            {
                "seed": seed,
                "status": "ok",
                "best_model": best_cif.name,
                **geom,
                "interface_iptm": iface,
                "iptm_C_HA1": cA,
                "iptm_C_HA2": cB,
            }
        )

    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {"system": "9DXX", "experiment": "single-anchor-restrained", "per_seed": rows}
    if ok:
        placements = sorted(r["placement_rmsd"] for r in ok)
        med = placements[len(placements) // 2]
        best = placements[0]
        summary["placement_rmsd_best"] = best
        summary["placement_rmsd_median"] = med
        summary["interface_iptm_median"] = sorted(
            r["interface_iptm"] for r in ok if r["interface_iptm"] is not None
        )[len(ok) // 2]
        summary["recovers_deposited_pose"] = bool(med < 5.0)
        summary["verdict"] = (
            f"placement RMSD median={med} A (best={best} A) over {len(ok)} seed(s): "
            + ("RECOVERS deposited pose (<5 A)." if med < 5.0 else "does NOT recover (>=5 A).")
        )
    else:
        summary["verdict"] = "no predictions present yet (CPU prep only); run the GPU lane first."
    return summary


def self_test():
    """Logic check: deposit-vs-deposit must give ~0 placement RMSD."""
    assert gemmi is not None, "gemmi required"
    dep = gemmi.read_structure(str(DEPOSIT))[0]

    # Build a pseudo-prediction from the deposit by renaming chains E->C and keeping A,B; the
    # receptor-superposition is identity so placement RMSD must be ~0. (Deposit receptor uses auth
    # numbering but our matcher is order-based, and deposit-vs-deposit order is identical.)
    # deposit's peptide is chain E (chain C in the deposit is a glycan), so point the peptide
    # extraction at E for both sides; receptor superposition is identity -> RMSD must be ~0.
    geom = score_model(dep, dep, pred_pep_chain=PEP_DEP)
    print("[self-test] deposit-vs-deposit:", json.dumps(geom))
    ok = geom["receptor_fit_rmsd"] < 1e-6 and geom["placement_rmsd"] < 1e-6
    print("[self-test]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true", help="emit only the JSON summary")
    args = ap.parse_args(argv)

    if gemmi is None:
        print("ERROR: gemmi not importable in this interpreter (run inside the container).")
        return 2
    if args.self_test:
        return self_test()

    summary = gate(args.seeds)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary, indent=2))
        print("\nVERDICT:", summary["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
