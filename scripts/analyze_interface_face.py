"""Which face of the L-HLH target does the D-binder pack against, and at what helix angle?
(Caveat: the GT binder lies PARALLEL on the Tyr-29 "outside" face = a clean 3-helix
bundle; the downloaded designs sit on the Met-14 "inside" face and ~PERPENDICULAR to the HLH
helices = wrong topology.) CPU-only, from an existing complex CIF.

Per complex it reports:
  * binder helix fraction (handedness-agnostic),
  * angle(binder axis, HLH helix-1 axis) and angle(binder axis, HLH helix-2 axis), acute 0-90 deg
    (~0-30 = parallel / bundle-like; ~60-90 = perpendicular / off-topology),
  * the TARGET residues the binder contacts (its footprint on the HLH),
  * min binder-CA distance to Met-14 and to Tyr-29, and which face is nearer.

HLH segmentation (target TATKQTGKSIKNAMVAAKKN GDDDS KESYLQALEKVTAKGE): helix-1 = res 1-20,
loop = 21-25 (GDDDS), helix-2 = 26-41. Met at 14, Tyr at 29.

Run (CPU; gemmi):
  python scripts/analyze_interface_face.py <chai_out|cif> --binder_chain B --target_chain A --label p1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HELIX1 = range(1, 21)     # 1-based target residues
_HELIX2 = range(26, 42)
_MET = 14
_TYR = 29


def _best_cif(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    import re
    sf = sorted(path.glob("scores.model_idx_*.npz"))
    if sf:
        bi, bv = 0, -np.inf
        for f in sf:
            agg = float(np.asarray(np.load(f)["aggregate_score"]).reshape(-1)[0])
            if agg > bv:
                bv, bi = agg, int(re.search(r"idx_(\d+)", f.name).group(1))
        c = path / f"pred.model_idx_{bi}.cif"
        if c.exists():
            return c
    return sorted(path.glob("*.cif"))[0]


def _ca(cif: Path, chain: str) -> np.ndarray:
    import gemmi
    st = gemmi.read_structure(str(cif))
    out = []
    for m in st:
        for ch in m:
            if ch.name != chain:
                continue
            for res in ch:
                a = res.find_atom("CA", "*")
                if a is not None:
                    out.append([a.pos.x, a.pos.y, a.pos.z])
        break
    return np.asarray(out, float)


def _axis(ca: np.ndarray) -> np.ndarray:
    """Helix axis = first principal component of the CA cloud."""
    c = ca - ca.mean(0)
    _, _, vt = np.linalg.svd(c)
    return vt[0]


def _acute_angle(u: np.ndarray, v: np.ndarray) -> float:
    cos = abs(float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
    return float(np.degrees(np.arccos(min(1.0, cos))))


def analyze(cif: Path, binder_chain: str, target_chain: str, label: str,
            cutoff: float = 8.0) -> dict:
    from xenodesign.secondary_structure import helix_fraction
    b = _ca(cif, binder_chain)
    t = _ca(cif, target_chain)
    nt = t.shape[0]
    # HLH segment axes (guard against short targets)
    h1 = t[[i - 1 for i in _HELIX1 if i <= nt]]
    h2 = t[[i - 1 for i in _HELIX2 if i <= nt]]
    b_axis = _axis(b)
    ang1 = _acute_angle(b_axis, _axis(h1)) if len(h1) >= 4 else None
    ang2 = _acute_angle(b_axis, _axis(h2)) if len(h2) >= 4 else None
    # binder footprint on the target (CA-CA)
    d = np.linalg.norm(b[:, None, :] - t[None, :, :], axis=2)   # (nb, nt)
    contacted = [int(j + 1) for j in range(nt) if d[:, j].min() < cutoff]
    met_d = float(d[:, _MET - 1].min()) if nt >= _MET else None
    tyr_d = float(d[:, _TYR - 1].min()) if nt >= _TYR else None
    face = None
    if met_d is not None and tyr_d is not None:
        face = "Met14(inside)" if met_d < tyr_d else "Tyr29(outside)"
    parallel = None
    if ang1 is not None:
        parallel = "parallel/bundle" if min(a for a in (ang1, ang2) if a is not None) <= 35 \
            else ("perpendicular/off-topology" if min(a for a in (ang1, ang2) if a is not None) >= 55
                  else "oblique")
    return {
        "label": label, "cif": str(cif),
        "binder_helix_fraction": round(float(helix_fraction(b)), 3),
        "angle_binder_vs_HLH_helix1_deg": None if ang1 is None else round(ang1, 1),
        "angle_binder_vs_HLH_helix2_deg": None if ang2 is None else round(ang2, 1),
        "orientation": parallel,
        "met14_min_dist_A": None if met_d is None else round(met_d, 2),
        "tyr29_min_dist_A": None if tyr_d is None else round(tyr_d, 2),
        "nearer_face": face,
        "n_contacted_target_res": len(contacted),
        "contacted_target_res_1based": contacted,
    }


def _parse(argv=None):
    p = argparse.ArgumentParser(description="HLH interface face + binder orientation")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--labels", default="")
    p.add_argument("--binder_chain", default="B")
    p.add_argument("--target_chain", default="A")
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = _parse(argv)
    labels = a.labels.split(",") if a.labels else [f"in{i}" for i in range(len(a.inputs))]
    out = []
    for inp, lab in zip(a.inputs, labels):
        try:
            out.append(analyze(_best_cif(Path(inp)), a.binder_chain, a.target_chain,
                               lab.strip(), a.cutoff))
        except Exception as exc:
            out.append({"label": lab, "error": f"{type(exc).__name__}: {exc}", "input": inp})
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
