"""Comprehensive committed-artifact analysis for the full-length ABLE fold test (8JH6 racemic deposit).

Emits everything ADR-016 cites, so the numbers live in a committed JSON, not just in chat:
  - full-length L / D compactness + verified chirality (signed chiral volume; D must be 100% D-sign)
  - per-seed CA-RMSD: full pred-L vs deposit-L, full pred-D vs deposit-D
  - HLH sub-region (res 76-116) RMSD pred-D vs deposit-D, and embedded e2e (vs isolated long-helix)
  - chiral self-consistency reflect(pred-D) vs pred-L
  - racemic sanity reflect(deposit-D) vs deposit-L
  - GDDDS loop (res 96-100) per-residue deviation in the GLOBAL superposition frame (NOT self-aligned)
    + the global max-deviation residue  -> shows the turn is locally correct but lever-arm displaced
  - isolated-fragment contrast (K-variant vs true-Phe, all seeds) read from the fold_result.json files

Run (CPU; needs gemmi):
  python scripts/analyze_dable_fold.py --out XenoDesign1_local_ref/dable_full/dable_fold_analysis.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import gemmi
import numpy as np

from xenodesign.geometry import kabsch_rmsd, signed_chiral_volume
from xenodesign.mirror import reflect_coords

_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "DAL", "DAR", "DAS", "DCY", "DGN", "DGL", "DHI", "DIL", "DLE", "DLY", "MED",
    "DPN", "DPR", "DSG", "DSN", "DTH", "DTR", "DTY", "DVA",
}
_REF = "XenoDesign1_local_ref"
_HLH = slice(75, 116)        # full-ABLE residues 76-116 (the carved HLH fragment), 0-based
_LOOP = list(range(95, 100))  # GDDDS = full residues 96-100, 0-based


def _best_model_cif(cif_path) -> Path:
    p = Path(cif_path)
    if p.is_file():
        return p
    if (p / "chai_out").is_dir():
        p = p / "chai_out"
    sf = sorted(p.glob("scores.model_idx_*.npz"))
    if sf:
        bi, bv = 0, -np.inf
        for f in sf:
            a = float(np.load(f)["aggregate_score"].reshape(-1)[0])
            if a > bv:
                bv, bi = a, int(re.search(r"idx_(\d+)", f.name).group(1))
        return p / f"pred.model_idx_{bi}.cif"
    return sorted(p.glob("*.cif"))[0]


def backbone(cif_path):
    """First chain's amino-acid residues -> dict with ca array + (n,ca,c,cb) per residue."""
    st = gemmi.read_structure(str(_best_model_cif(cif_path)))
    rows = []
    for ch in st[0]:
        for r in ch:
            if r.name not in _AA:
                continue
            g = lambda nm: (lambda a: (a.pos.x, a.pos.y, a.pos.z) if a else None)(r.find_atom(nm, "*"))
            ca = g("CA")
            if ca is not None:
                rows.append({"name": r.name, "ca": ca, "n": g("N"), "c": g("C"), "cb": g("CB")})
        break
    return rows


def ca(rows):
    return np.asarray([r["ca"] for r in rows], float)


def chirality(rows):
    vols = [signed_chiral_volume(r["n"], r["ca"], r["c"], r["cb"])
            for r in rows if r["name"] != "GLY" and None not in (r["n"], r["c"], r["cb"])]
    v = np.asarray(vols, float)
    return {"mean_signed_chiral_volume": round(float(v.mean()), 2),
            "frac_D_sign": round(float((v < 0).mean()), 3),
            "frac_L_sign": round(float((v > 0).mean()), 3), "n_scored": int(len(v))}


def compact(c):
    return {"n_ca": int(len(c)), "end_to_end_A": round(float(np.linalg.norm(c[0] - c[-1])), 1),
            "radius_of_gyration_A": round(float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean())), 1)}


def kabsch_perres(a, b):
    """Per-residue distances after optimal superposition of a onto b (geometry.kabsch_rmsd convention)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ac = a - a.mean(0); bc = b - b.mean(0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return np.linalg.norm(ac @ rot.T - bc, axis=1)


def _frag_e2e(path):
    fr = Path(path) / "fold_result.json"
    if not fr.exists():
        return None
    d = json.loads(fr.read_text())
    return {"seed": d.get("seed"), "e2e_A": d["end_to_end_A"], "rg_A": d["radius_of_gyration_A"],
            "fold": d["fold"], "ptm": d.get("ptm")}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--deposit", default=f"{_REF}/dable_full/8jh6/8jh6.cif")
    p.add_argument("--deposit_L_chain", default="B")
    p.add_argument("--deposit_D_chain", default="A")
    p.add_argument("--pred_L", default=f"{_REF}/dable_full/L")
    p.add_argument("--pred_D", nargs="+", default=[f"{_REF}/dable_full/D",
                   f"{_REF}/dable_full/D_seed7", f"{_REF}/dable_full/D_seed123"])
    p.add_argument("--out", default=f"{_REF}/dable_full/dable_fold_analysis.json")
    a = p.parse_args(argv)

    dep = {}
    for ch in gemmi.read_structure(a.deposit)[0]:
        rows = [r for r in ch if r.name in _AA]
        cas = [r.find_atom("CA", "*") for r in rows]
        cas = [(x.pos.x, x.pos.y, x.pos.z) for x in cas if x]
        if cas:
            dep[ch.name] = np.asarray(cas, float)
    depL, depD = dep[a.deposit_L_chain], dep[a.deposit_D_chain]

    rowsL = backbone(a.pred_L); caL = ca(rowsL)
    res = {
        "deposit": Path(a.deposit).name, "deposit_L_chain": a.deposit_L_chain,
        "deposit_D_chain": a.deposit_D_chain, "loop_residues": "GDDDS = full-res 96-100",
        "full_length": {"L": {
            "compactness": compact(caL), "chirality": chirality(rowsL),
            "rmsd_vs_depositL_A": round(kabsch_rmsd(caL, depL), 2),
            "ptm": json.loads((Path(a.pred_L) / "fold_result.json").read_text()).get("ptm"),
        }, "D": []},
        "racemic_sanity_reflect_depositD_vs_depositL_A": round(kabsch_rmsd(reflect_coords(depD), depL), 2),
    }

    best = None
    for pd in a.pred_D:
        rowsD = backbone(pd); caD = ca(rowsD)
        fr = json.loads((Path(pd) / "fold_result.json").read_text())
        entry = {
            "dir": Path(pd).name, "seed": fr.get("seed"), "ptm": fr.get("ptm"),
            "compactness": compact(caD), "chirality": chirality(rowsD),
            "rmsd_vs_depositD_A": round(kabsch_rmsd(caD, depD), 2),
            "subregion76_116_rmsd_vs_depositD_A": round(kabsch_rmsd(caD[_HLH], depD[_HLH]), 2),
            "subregion76_116_e2e_A": round(float(np.linalg.norm(caD[_HLH][0] - caD[_HLH][-1])), 1),
            "chiral_selfconsistency_reflectD_vs_predL_A": round(kabsch_rmsd(reflect_coords(caD), caL), 2),
        }
        res["full_length"]["D"].append(entry)
        if best is None or entry["rmsd_vs_depositD_A"] < best[1]:
            best = (caD, entry["rmsd_vs_depositD_A"], entry["seed"])

    # GDDDS loop deviation in the GLOBAL frame (best D seed superposed on deposit-D over all 126 CA)
    caD = best[0]
    per = kabsch_perres(caD, depD)
    res["loop_global_frame_deviation"] = {
        "best_seed": best[2], "superposed_over_n_ca": int(len(per)),
        "median_dev_A": round(float(np.median(per)), 1),
        "loop_res96_100_dev_A": [round(float(per[i]), 1) for i in _LOOP],
        "global_max_residue_1based": int(np.argmax(per) + 1),
        "global_max_dev_A": round(float(per.max()), 1),
    }

    res["isolated_fragment_contrast"] = {
        "Kvariant_TATKQTGKSIK...": {k: _frag_e2e(f"{_REF}/hlh_fold/{k}") for k in ("D_r3", "D_r10", "L_r3")},
        "truePhe_TATFQTGKSIF...": {k: _frag_e2e(f"{_REF}/hlh_fold_truephe/{k}")
                                   for k in ("D_seed42", "D_seed7", "D_seed123", "L_seed42")},
        "embedded_subregion76_116_e2e_A_per_D_seed": {
            e["seed"]: e["subregion76_116_e2e_A"] for e in res["full_length"]["D"]},
    }

    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
