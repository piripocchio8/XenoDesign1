"""Decide the score question: is the structure mirror-correct, and are ipAE/ipSAE
L-biased like ipTM?  CPU-only post-analysis of the existing specular runs (T16).

For each design we have two predictions:
  orig   = L-target . D-binder   (the real design)
  mirror = D-target . L-binder   (its exact parity image)

(2) RMSD-AFTER-REFLECTION: reflect the mirror coords (x,y,z -> -x,y,z), Kabsch-superpose onto
    orig over all CA, report RMSD. A SMALL RMSD => Chai produced the correct mirror STRUCTURE
    (so the ipTM gap is a CONFIDENCE-head bias only, and a geometry-derived score would be
    parity-honest). A LARGE RMSD => Chai folds the mirror differently (structurally unreliable
    for D).

(3) SCORE COMPARISON: ipTM, ipAE, ipSAE for orig vs mirror. ipTM/ipAE/ipSAE are ALL derived
    from the same L-trained confidence head, so the expectation is they are ALL ~equally biased;
    this prints the numbers so we decide empirically.

Run (CPU; needs gemmi):  python scripts/specular_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from xenodesign.geometry import kabsch_rmsd

RUNS = {
    "p1_SEIRER": "XenoDesign1_local_ref/specular/p1",
    "prior_SLLNRT": "XenoDesign1_local_ref/specular/prior",
}


def _best_cif(chai_out: Path) -> Path:
    import re
    sf = sorted(chai_out.glob("scores.model_idx_*.npz"))
    if sf:
        bi, bv = 0, -np.inf
        for f in sf:
            agg = float(np.asarray(np.load(f)["aggregate_score"]).reshape(-1)[0])
            if agg > bv:
                bv, bi = agg, int(re.search(r"idx_(\d+)", f.name).group(1))
        c = chai_out / f"pred.model_idx_{bi}.cif"
        if c.exists():
            return c
    return sorted(chai_out.glob("*.cif"))[0]


def _all_ca(cif: Path) -> np.ndarray:
    """CA coords of ALL chains in CIF order (target chain then binder chain)."""
    import gemmi
    st = gemmi.read_structure(str(cif))
    out = []
    for m in st:
        for ch in m:
            for res in ch:
                a = res.find_atom("CA", "*")
                if a is not None:
                    out.append([a.pos.x, a.pos.y, a.pos.z])
        break
    return np.asarray(out, float)


def _metrics(chai_out: Path) -> dict:
    from xenodesign import metrics
    import re
    sf = sorted(chai_out.glob("scores.model_idx_*.npz"))
    bi = 0
    if sf:
        bv = -np.inf
        for f in sf:
            agg = float(np.asarray(np.load(f)["aggregate_score"]).reshape(-1)[0])
            if agg > bv:
                bv, bi = agg, int(re.search(r"idx_(\d+)", f.name).group(1))
    npz = chai_out / f"confidence.model_idx_{bi}.npz"
    cif = chai_out / f"pred.model_idx_{bi}.cif"
    b = dict(metrics.score_interface(npz, cif, chain_a=0, chain_b=1))
    # interface ipTM from the scores npz (per_chain_pair_iptm)
    iptm = None
    sc = chai_out / f"scores.model_idx_{bi}.npz"
    if sc.exists():
        d = np.load(sc)
        if "per_chain_pair_iptm" in d:
            m = np.asarray(d["per_chain_pair_iptm"]).reshape(-1)
            n = int(round(len(m) ** 0.5))
            if n >= 2:
                sq = m.reshape(n, n)
                iptm = float(max(sq[0, 1], sq[1, 0]))
    return {"interface_iptm": iptm,
            "ipae_mean": b.get("ipae_mean"),
            "ipsae_cut10": b.get("ipsae_cut10", b.get("ipsae"))}


def analyze(label: str, root: str) -> dict:
    root = Path(root)
    orig_co = root / "orig_Ltgt_Dbinder" / "chai_out"
    mir_co = root / "mirror_Dtgt_Lbinder" / "chai_out"
    orig_ca = _all_ca(_best_cif(orig_co))
    mir_ca = _all_ca(_best_cif(mir_co))
    rmsd = None
    if orig_ca.shape == mir_ca.shape and orig_ca.shape[0] > 0:
        mir_reflected = mir_ca.copy()
        mir_reflected[:, 0] *= -1.0   # reflect x -> -x
        rmsd = float(kabsch_rmsd(mir_reflected, orig_ca))
    om, mm = _metrics(orig_co), _metrics(mir_co)
    def gap(k):
        a, b = om.get(k), mm.get(k)
        return None if (a is None or b is None) else round(a - b, 4)
    return {
        "design": label,
        "rmsd_after_reflection_A": None if rmsd is None else round(rmsd, 3),
        "n_ca": int(orig_ca.shape[0]),
        "orig": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in om.items()},
        "mirror": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mm.items()},
        "orig_minus_mirror": {"interface_iptm": gap("interface_iptm"),
                              "ipae_mean": gap("ipae_mean"),
                              "ipsae_cut10": gap("ipsae_cut10")},
    }


def main():
    out = [analyze(lab, root) for lab, root in RUNS.items()]
    print(json.dumps(out, indent=2))
    Path("XenoDesign1_local_ref/specular_analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
    sys.exit(0)
