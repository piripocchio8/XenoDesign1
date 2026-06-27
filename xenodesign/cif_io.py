"""Shared CIF / entity-coordinate plumbing for the design loop.

These helpers were historically defined in ``scripts/design_demo.py`` and imported
back INTO the package (``classes/alpha.py``, ``classes/cyclic.py``, ``dispatch.py``,
``abc/calibration.py``) — an inverted dependency (package → scripts CLI). They are
pure CIF/array plumbing with no demo-specific logic, so they live in the package now
and ``scripts/design_demo.py`` re-exports them for its CLI (MOD-1).

Behaviour is byte-for-byte the same as the previous ``scripts.design_demo`` defs;
this was a move, not a rewrite.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _best_cif_path(out_dir: Path) -> Path:
    import re
    score_files = sorted(out_dir.rglob("scores.model_idx_*.npz"))
    if score_files:
        best_idx, best_agg = 0, -np.inf
        for f in score_files:
            d = np.load(f)
            agg = float(np.asarray(d["aggregate_score"]).reshape(-1)[0])
            idx = int(re.search(r"idx_(\d+)", f.name).group(1))
            if agg > best_agg:
                best_agg = agg
                best_idx = idx
        cif = next(out_dir.rglob(f"pred.model_idx_{best_idx}.cif"), None)
        if cif is not None:
            return cif
    return sorted(out_dir.rglob("*.cif"))[0]


def _all_atoms_from_chain(cif_path: Path, chain_name: str):
    import gemmi
    structure = gemmi.read_structure(str(cif_path))
    coords_list, elements_list = [], []
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                for atom in res:
                    coords_list.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    elements_list.append(atom.element.name)
        break
    if not coords_list:
        return np.zeros((0, 3), dtype=np.float32), []
    return np.array(coords_list, dtype=np.float32), elements_list


def _backbone_array_from_residues(residues: list[dict]) -> np.ndarray:
    arr = np.zeros((len(residues), 4, 3), dtype=np.float32)
    for i, res in enumerate(residues):
        arr[i, 0] = res["N"]
        arr[i, 1] = res["CA"]
        arr[i, 2] = res["C"]
        arr[i, 3] = res.get("CB", res["CA"])
    return arr


def _chirality_violation_frac_from_cif(cif_path: Path) -> float:
    from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif
    from xenodesign.chirality import is_chirality_violation

    residues = backbone_by_residue_from_cif(cif_path, "B")
    if not residues:
        residues = backbone_by_residue_from_cif(cif_path, "b")
    if not residues:
        return 0.0

    total = violations = 0
    for res in residues:
        if "CB" not in res:
            continue
        total += 1
        if is_chirality_violation(res["N"], res["CA"], res["C"], res["CB"], "D"):
            violations += 1
    return violations / total if total > 0 else 0.0
