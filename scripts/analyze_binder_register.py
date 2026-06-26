"""T18 (measurement half): determine a D-binder's helical register at the interface from an
EXISTING predicted complex CIF — no GPU, no re-prediction.

For each complex (binder = chain B, target = chain A by our run convention), we report:
  * helix_fraction of the binder (is it even helical?),
  * which binder residues are BURIED at the interface (min CA-CA distance to any target CA < cutoff),
  * the GAP pattern between consecutive buried residues — the direct readout of the repeat:
      heptad coiled-coil buries a+d -> gaps ~ 3,4,3,4 (period 7);
      hendecad buries with a 3-3-1 / 3-4-4 pattern (period 11);
  * the dominant period from the autocorrelation of the buried-indicator (sanity cross-check).

This makes NO assumption about 7 vs 11 — it reports the empirical pattern so the operator reads the
register off the data (shift the BINDER by ITS measured repeat, target fixed).

Run (CPU; needs gemmi):
    python scripts/analyze_binder_register.py <chai_out_dir|cif> [<...>] --labels p1,prior
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _best_cif(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    import re
    score_files = sorted(path.glob("scores.model_idx_*.npz"))
    if score_files:
        best_idx, best = 0, -np.inf
        for f in score_files:
            agg = float(np.asarray(np.load(f)["aggregate_score"]).reshape(-1)[0])
            if agg > best:
                best, best_idx = agg, int(re.search(r"idx_(\d+)", f.name).group(1))
        cif = path / f"pred.model_idx_{best_idx}.cif"
        if cif.exists():
            return cif
    return sorted(path.glob("*.cif"))[0]


def _ca_coords(cif_path: Path, chain_name: str) -> np.ndarray:
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    cas = []
    for model in st:
        for ch in model:
            if ch.name != chain_name:
                continue
            for res in ch:
                a = res.find_atom("CA", "*")
                if a is not None:
                    cas.append([a.pos.x, a.pos.y, a.pos.z])
        break
    return np.asarray(cas, dtype=float)


def _backbone_by_residue(cif_path: Path, chain_name: str):
    """Per-residue (N, CA, C) for one chain (None entry if any atom missing)."""
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    out = []
    for model in st:
        for ch in model:
            if ch.name != chain_name:
                continue
            for res in ch:
                atoms = {}
                for nm in ("N", "CA", "C"):
                    a = res.find_atom(nm, "*")
                    if a is not None:
                        atoms[nm] = np.array([a.pos.x, a.pos.y, a.pos.z], float)
                out.append(atoms if {"N", "CA", "C"} <= set(atoms) else None)
        break
    return out


def _cb(N, CA, C):
    """Idealised CB from backbone (the standard tetrahedral construction)."""
    b, c = CA - N, C - CA
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def facing_residues(cif_path: Path, binder_chain: str, target_chain: str,
                    cutoff: float) -> list:
    """1-based binder residues whose SIDE CHAIN points AT the target (the interface FACE) —
    register v2. A residue is 'facing' iff (CA within cutoff of some target CA) AND the CA->CB
    vector points toward the nearest target CA (angle < 90 deg). The index pattern of facing
    residues IS the register (heptad a/d -> gaps ~3,4; hendecad -> 3-3-1 / 3-4-4)."""
    bb = _backbone_by_residue(cif_path, binder_chain)
    tgt = _ca_coords(cif_path, target_chain)
    if tgt.size == 0:
        return []
    facing = []
    for i, res in enumerate(bb):
        if res is None:
            continue
        ca, cb = res["CA"], _cb(res["N"], res["CA"], res["C"])
        d = np.linalg.norm(tgt - ca, axis=1)
        j = int(np.argmin(d))
        if d[j] >= cutoff:
            continue
        v_side = cb - ca
        v_tgt = tgt[j] - ca
        cosang = float(np.dot(v_side, v_tgt) / (np.linalg.norm(v_side) * np.linalg.norm(v_tgt) + 1e-9))
        if cosang > 0.0:  # angle < 90 deg -> side chain points toward target
            facing.append(i + 1)
    return facing


def buried_residues(binder_ca: np.ndarray, target_ca: np.ndarray, cutoff: float) -> list:
    """1-based binder residue indices whose CA is within `cutoff` A of ANY target CA."""
    if binder_ca.size == 0 or target_ca.size == 0:
        return []
    d = np.linalg.norm(binder_ca[:, None, :] - target_ca[None, :, :], axis=2)  # (nb, nt)
    return [int(i + 1) for i in range(binder_ca.shape[0]) if d[i].min() < cutoff]


def autocorr_period(n: int, buried: list, max_lag: int = 14) -> int | None:
    """Dominant repeat lag of the buried-indicator signal (3..max_lag), or None."""
    if not buried:
        return None
    x = np.zeros(n)
    for b in buried:
        x[b - 1] = 1.0
    x = x - x.mean()
    if not np.any(x):
        return None
    best_lag, best_val = None, -np.inf
    for lag in range(3, min(max_lag, n - 1) + 1):
        v = float(np.dot(x[:-lag], x[lag:]))
        if v > best_val:
            best_val, best_lag = v, lag
    return best_lag


def analyze(cif_path: Path, label: str, binder_chain: str, target_chain: str,
            cutoff: float, face_cutoff: float = 12.0) -> dict:
    from xenodesign.secondary_structure import helix_fraction
    binder_ca = _ca_coords(cif_path, binder_chain)
    target_ca = _ca_coords(cif_path, target_chain)
    buried = buried_residues(binder_ca, target_ca, cutoff)
    facing = facing_residues(cif_path, binder_chain, target_chain, face_cutoff)  # v2 register
    face_gaps = [facing[i + 1] - facing[i] for i in range(len(facing) - 1)]
    return {
        "label": label,
        "cif": str(cif_path),
        "binder_len": int(binder_ca.shape[0]),
        "target_len": int(target_ca.shape[0]),
        "binder_helix_fraction": round(float(helix_fraction(binder_ca)), 3),  # now D-aware
        # v1 (coarse CA-CA, kept for reference)
        "n_buried_CA": len(buried),
        # v2 (FACE-based: side chain points at target) — read the register off THIS
        "facing_residues_1based": facing,
        "facing_gap_pattern": face_gaps,
        "facing_period_autocorr": autocorr_period(binder_ca.shape[0], facing),
        "face_cutoff_A": face_cutoff,
        "ca_cutoff_A": cutoff,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="measure a D-binder's interface register")
    p.add_argument("inputs", nargs="+", help="chai_out dirs or CIF files")
    p.add_argument("--labels", default="", help="comma-separated labels (one per input)")
    p.add_argument("--binder_chain", default="B")
    p.add_argument("--target_chain", default="A")
    p.add_argument("--cutoff", type=float, default=10.0, help="CA-CA interface cutoff (A)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    labels = args.labels.split(",") if args.labels else [f"in{i}" for i in range(len(args.inputs))]
    out = []
    for inp, lab in zip(args.inputs, labels):
        try:
            cif = _best_cif(Path(inp))
            out.append(analyze(cif, lab.strip(), args.binder_chain, args.target_chain, args.cutoff))
        except Exception as exc:
            out.append({"label": lab, "error": f"{type(exc).__name__}: {exc}", "input": inp})
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
