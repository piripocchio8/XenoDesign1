"""Gate the POCKET-RESTRAINED MDM2 re-predictions: which systems does Chai recapitulate?

Question: under the light pocket restraint, does Chai place the D-peptide binder back onto its
DEPOSITED pose for each MDM2 system (3LNJ, 7KJM, 3IWY)? Only recapitulating systems should join
the objective fit.

Method (placement RMSD, NOT a peptide re-fit) -- same logic as gate_9dxx_restrained.py:
  1. For each seed's best real model (highest aggregate_score), Kabsch-superpose the predicted
     TARGET (MDM2) CA onto the deposit target CA. The target frame is the anchor.
  2. Apply that SAME rigid transform to the predicted BINDER and measure binder CA placement RMSD
     vs the deposit binder, WITHOUT any further binder-only fitting.
  3. Report binder<->target interface ipTM (A-B entry of per_chain_pair_iptm).

CHAIN CONVENTION (differs from 9DXX):
  PREDICTION: chain A = binder (12-13mer incl. an added C-ter Gly), chain B = MDM2 target (~82-85aa).
  DEPOSIT   : the binder/target are different letters and there are several copies. We MATCH BY
              SEQUENCE/LENGTH, not letter:
                - target  = deposit AA chain whose seq aligns to the pred target (~82-85aa);
                - binder  = the cognate copy (closest in space to that target), best-covered copy.
  The added C-terminal Gly is NOT in the deposit -> EXCLUDED from the binder RMSD.

Residue correspondence:
  - TARGET: pred is renumbered 1..N; deposit uses native MDM2 numbering. The pred target sequence is
    an exact contiguous block of the deposit target sequence -> align by sequence, pair CA over the
    common block positionally.
  - BINDER: the deposit binder's AUTH SEQID equals the pred binder's 1-based sequential index
    (verified: 3LNJ dep seqid 2..12, 7KJM 1..12, 3IWY 1..12; pred binder index 1..L with the trailing
    Gly dropped). Map deposit seqid k -> pred index k; intersect on modeled residues.

Verdict: recapitulates if the BEST (min over seeds) binder placement RMSD < ~5 A.

CPU-only; gemmi + numpy.
  python3 scripts/gate_mdm2_restrained.py
  python3 scripts/gate_mdm2_restrained.py --self-test   # deposit-vs-deposit, RMSD ~0
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
PANEL_ROOT = REPO / "XenoDesign1_local_ref" / "benchmarks" / "gp41_mdm2_r2"
DEPOSIT_DIR = REPO / "XenoDesign1_local_ref" / "benchmarks" / "gp41_mdm2" / "cifs"
SYSTEMS = ("3LNJ", "7KJM", "3IWY")
SEEDS = (42, 43, 44)

# pred chains
PRED_BINDER, PRED_TARGET = "A", "B"


def _ca(res):
    for a in res:
        if a.name == "CA":
            return np.array([a.pos.x, a.pos.y, a.pos.z])
    return None


def _aa_residues(chain):
    """Ordered amino-acid residues of a chain (drops water/ligand/glycan)."""
    out = []
    for r in chain:
        t = gemmi.find_tabulated_residue(r.name)
        if t is not None and t.is_amino_acid():
            out.append(r)
    return out


def _one(r):
    return gemmi.find_tabulated_residue(r.name).one_letter_code.upper()


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid (R,t) best mapping P onto Q (minimise ||R P + t - Q||). Both (N,3)."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def _chain(model, name):
    return next((c for c in model if c.name == name), None)


def pick_deposit_pair(dep_model, pred_target_seq):
    """Pick (target_chain, binder_chain) from the deposit by sequence + spatial cognacy.

    The deposit holds several binder/target copies. A binder copy can graze a crystal-neighbour
    target it does not truly bind, so cognacy is chosen by INTERFACE SIZE (number of target residues
    contacted), NOT raw min CA-CA distance -- the latter mis-pairs binders onto a 94-97 crystal
    contact instead of the real p53 pocket (native 51-96).

      target  = AA chain whose seq aligns to the pred target block (~82-85aa, substring either way).
      binder  = best-covered short AA chain; its cognate target = the target copy with the most
                contacts; we then return THAT (target, binder) cognate pair.
    Returns (target_chain, binder_chain).
    """
    aa_chains = [c for c in dep_model if _aa_residues(c)]
    targets, binders = [], []
    for c in aa_chains:
        res = _aa_residues(c)
        seq = "".join(_one(r) for r in res)
        (binders if len(res) <= 25 else targets).append((c, res, seq))

    def is_target(seq):
        return (pred_target_seq in seq) or (seq in pred_target_seq)

    target_chains = [(c, res) for c, res, seq in targets if is_target(seq)] or [
        (targets[0][0], targets[0][1])
    ]

    def n_contacts(bres, tres, cutoff=8.0):
        b = np.array([_ca(r) for r in bres if _ca(r) is not None])
        t = np.array([_ca(r) for r in tres if _ca(r) is not None])
        if len(b) == 0 or len(t) == 0:
            return 0
        dmin = np.linalg.norm(b[:, None] - t[None], axis=2).min(0)
        return int((dmin < cutoff).sum())

    # pick the (binder copy, cognate target) pair with the largest interface; tie-break on binder
    # coverage so the best-modeled binder copy wins.
    best_pair, best_key = None, None
    for bc, bres, _ in binders:
        # cognate target = the target copy this binder contacts most
        tc, tres = max(target_chains, key=lambda tt: n_contacts(bres, tt[1]))
        nc = n_contacts(bres, tres)
        key = (nc, len(bres))
        if best_key is None or key > best_key:
            best_key, best_pair = key, (tc, bc)
    return best_pair


def align_target_pairs(pred_target_res, dep_target_res):
    """Positional CA pairs over the common target block (pred seq is a substring of deposit seq)."""
    pseq = "".join(_one(r) for r in pred_target_res)
    dseq = "".join(_one(r) for r in dep_target_res)
    # find offset so that pred aligns inside deposit (or vice-versa)
    if pseq in dseq:
        off_d, off_p, n = dseq.index(pseq), 0, len(pseq)
    elif dseq in pseq:
        off_p, off_d, n = pseq.index(dseq), 0, len(dseq)
    else:
        # fall back: longest common prefix from the natural alignment of identical leading block
        n = 0
        while n < min(len(pseq), len(dseq)) and pseq[n] == dseq[n]:
            n += 1
        off_d = off_p = 0
    P, Q = [], []
    for i in range(n):
        p = _ca(pred_target_res[off_p + i])
        q = _ca(dep_target_res[off_d + i])
        if p is not None and q is not None:
            P.append(p)
            Q.append(q)
    return np.array(P), np.array(Q)


def binder_pairs(pred_binder_res, dep_binder_res):
    """Match deposit binder by AUTH SEQID == pred binder 1-based sequential index. Drops the added
    trailing Gly automatically (it has no deposit seqid counterpart)."""
    pred_by_idx = {i + 1: r for i, r in enumerate(pred_binder_res)}  # 1-based sequential
    P, Q = [], []
    for r in dep_binder_res:
        sid = r.seqid.num
        pr = pred_by_idx.get(sid)
        if pr is None:
            continue
        p = _ca(pr)
        q = _ca(r)
        if p is not None and q is not None:
            P.append(p)
            Q.append(q)
    return np.array(P), np.array(Q)


def interface_iptm(npz_path: Path):
    """binder-target interface ipTM = A-B entry of per_chain_pair_iptm (max of the two off-diag)."""
    d = np.load(npz_path)
    if "per_chain_pair_iptm" not in d:
        return None
    m = np.asarray(d["per_chain_pair_iptm"])
    m = m.reshape(int(m.size ** 0.5), -1)
    return round(float(max(m[0, 1], m[1, 0])), 3)


def best_model_npz(chai_out: Path):
    best, best_score = None, -np.inf
    for npz in sorted(chai_out.glob("scores.model_idx_*.npz")):
        d = np.load(npz)
        s = float(np.asarray(d["aggregate_score"]).reshape(-1)[0]) if "aggregate_score" in d else -np.inf
        if s > best_score:
            best, best_score = npz, s
    return best


def score_seed(sys_name, seed, dep_model, dep_target, dep_binder):
    chai_out = PANEL_ROOT / sys_name / "real" / f"seed{seed}" / "chai_out"
    if not list(chai_out.glob("pred.model_idx_*.cif")):
        return {"seed": seed, "status": "MISSING"}
    best_npz = best_model_npz(chai_out)
    midx = best_npz.name.split("model_idx_")[1].split(".")[0]
    best_cif = chai_out / f"pred.model_idx_{midx}.cif"
    pred_model = gemmi.read_structure(str(best_cif))[0]

    pred_target_res = _aa_residues(_chain(pred_model, PRED_TARGET))
    pred_binder_res = _aa_residues(_chain(pred_model, PRED_BINDER))
    dep_target_res = _aa_residues(dep_target)
    dep_binder_res = _aa_residues(dep_binder)

    # target superposition (move prediction onto deposit)
    P, Q = align_target_pairs(pred_target_res, dep_target_res)
    R, t = kabsch(P, Q)
    rec_rmsd = float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean()))

    # binder placement: apply same transform, no refit
    Pb, Qb = binder_pairs(pred_binder_res, dep_binder_res)
    moved = Pb @ R.T + t
    place_rmsd = float(np.sqrt(((moved - Qb) ** 2).sum(1).mean()))

    return {
        "seed": seed,
        "status": "ok",
        "best_model": best_cif.name,
        "target_pairs": int(len(P)),
        "target_fit_rmsd": round(rec_rmsd, 3),
        "binder_pairs": int(len(Pb)),
        "placement_rmsd": round(place_rmsd, 3),
        "interface_iptm": interface_iptm(best_npz),
    }


def gate(systems, seeds):
    assert gemmi is not None, "gemmi required"
    out = {}
    for sys_name in systems:
        dep_path = DEPOSIT_DIR / f"{sys_name}.cif"
        dep_model = gemmi.read_structure(str(dep_path))[0]
        # need a pred target seq to choose the deposit target copy
        any_cif = next((PANEL_ROOT / sys_name / "real").glob("seed*/chai_out/pred.model_idx_0.cif"), None)
        pm = gemmi.read_structure(str(any_cif))[0]
        pred_target_seq = "".join(_one(r) for r in _aa_residues(_chain(pm, PRED_TARGET)))
        dep_target, dep_binder = pick_deposit_pair(dep_model, pred_target_seq)

        rows = [score_seed(sys_name, s, dep_model, dep_target, dep_binder) for s in seeds]
        ok = [r for r in rows if r.get("status") == "ok"]
        summary = {
            "system": sys_name,
            "deposit_target_chain": dep_target.name,
            "deposit_binder_chain": dep_binder.name,
            "per_seed": rows,
        }
        if ok:
            placements = sorted(r["placement_rmsd"] for r in ok)
            best = placements[0]
            med = placements[len(placements) // 2]
            summary["placement_rmsd_best"] = best
            summary["placement_rmsd_median"] = med
            iptms = [r["interface_iptm"] for r in ok if r["interface_iptm"] is not None]
            summary["interface_iptm_best"] = max(iptms) if iptms else None
            summary["recapitulates"] = bool(best < 5.0)
            summary["verdict"] = (
                f"best placement RMSD={best} A (median={med} A) over {len(ok)} seed(s): "
                + ("RECAPITULATES (<5 A) -> INCLUDE in fit."
                   if best < 5.0 else "does NOT recapitulate (>=5 A) -> EXCLUDE from fit.")
            )
        else:
            summary["recapitulates"] = None
            summary["verdict"] = "no predictions present."
        out[sys_name] = summary
    return out


def self_test():
    """deposit-vs-deposit sanity: target fit ~0 and binder placement ~0 for each system."""
    assert gemmi is not None, "gemmi required"
    allok = True
    for sys_name in SYSTEMS:
        dep_model = gemmi.read_structure(str(DEPOSIT_DIR / f"{sys_name}.cif"))[0]
        any_cif = next((PANEL_ROOT / sys_name / "real").glob("seed*/chai_out/pred.model_idx_0.cif"))
        pm = gemmi.read_structure(str(any_cif))[0]
        pred_target_seq = "".join(_one(r) for r in _aa_residues(_chain(pm, PRED_TARGET)))
        dep_target, dep_binder = pick_deposit_pair(dep_model, pred_target_seq)
        dtr = _aa_residues(dep_target)
        dbr = _aa_residues(dep_binder)
        # target onto itself
        P, Q = align_target_pairs(dtr, dtr)
        R, t = kabsch(P, Q)
        rec = float(np.sqrt(((P @ R.T + t - Q) ** 2).sum(1).mean()))
        # binder onto itself: pred index==deposit seqid only holds for the PRED binder; deposit binder
        # seqids are not 1..L sequential, so map deposit-binder by its own seqid against itself.
        bca = np.array([_ca(r) for r in dbr if _ca(r) is not None])
        moved = bca @ R.T + t
        place = float(np.sqrt(((moved - bca) ** 2).sum(1).mean()))
        ok = rec < 1e-6 and place < 1e-6
        allok = allok and ok
        print(f"[self-test] {sys_name} target={dep_target.name} binder={dep_binder.name} "
              f"rec_rmsd={rec:.2e} binder_self_rmsd={place:.2e} {'PASS' if ok else 'FAIL'}")
    print("[self-test]", "PASS" if allok else "FAIL")
    return 0 if allok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if gemmi is None:
        print("ERROR: gemmi not importable in this interpreter.")
        return 2
    if args.self_test:
        return self_test()
    out = gate(args.systems, args.seeds)
    print(json.dumps(out, indent=2))
    print("\n=== VERDICTS ===")
    for s, v in out.items():
        print(f"{s}: {v['verdict']}")
    inc = [s for s, v in out.items() if v.get("recapitulates")]
    exc = [s for s, v in out.items() if v.get("recapitulates") is False]
    print("\nINCLUDE in fit (recapitulate):", inc or "(none)")
    print("EXCLUDE from fit:", exc or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
