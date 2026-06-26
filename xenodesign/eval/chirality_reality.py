"""Chirality-reality harness (#36): the anti-survivorship view of a D-design run.

A single "best" structure can pass the chirality gate by luck while most of the
trajectory is wrong (an L design that the inverse-folder happened to label D, a
chain that escaped into all-Glycine achirality, or a seed-pinned pose that never
re-derived its own handedness). This module measures the things a single headline
number hides:

  1. ``trajectory_chirality_distribution(loop_dir)`` — read the best CIF of EVERY
     ``iter_*/chai_out`` in a run dir, measure the binder chain's D-chirality
     violation fraction per iteration, and return the FULL distribution plus
     mean / max / pass-fraction (<=0.10) and the per-iter Gly (achiral) fraction.
     This is the headline that survivorship bias erases.

  2. ``mirror_self_consistency(cif_path | coords_a, coords_b)`` — the CPU geometry
     comparison underneath the mirror→re-predict→mirror-back protocol (the GPU
     step is documented below). Small = self-consistent (Chai re-derives the same
     pose from the mirror image, modulo reflection).

  3. The COLD-START D-FOLD protocol (documented below) — fold the all-D binder
     ALONE (or with target) with NO restraint and NO seed via a fresh Chai
     predict, then measure whether the handedness / helix / interface recover
     unaided. This module provides the analysis (1 and 2); the campaign driver
     invokes the GPU fold.

Everything here is CPU-only. All GPU/network work is the caller's: this harness
consumes CIFs and coord arrays that already exist on disk.

Reused, never reimplemented:
  - ``xenodesign.eval.gate_tier0a.backbone_by_residue_from_cif`` — CIF -> per-residue
    {'N','CA','C'(,'CB')} backbone dicts for one chain.
  - ``xenodesign.chirality.is_chirality_violation`` — signed-chiral-volume D/L test.
  - ``xenodesign.mirror.mirror_discrepancy`` / ``reflect_coords`` — the mirror geometry.


COLD-START D-FOLD PROTOCOL (the unaided-recovery test; GPU step run by the driver)
---------------------------------------------------------------------------------
The strongest reality check: does the all-D binder sequence fold to the designed
handedness/structure when Chai sees ONLY the sequence — no seed coordinates, no
restraints, no truncated-diffusion warm start? If it does, the design is real; if
it only looks chiral when pinned by a D-seed, the loop was decorating a seed.

Run it with a FRESH backend (default seed, full diffusion, unconstrained):

    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign.io_spec import glycine_satisfy_guard

    # binder_seq is the one-letter L sequence; chirality='D' triggers to_d_fasta in
    # build_fasta. glycine_satisfy_guard keeps a glycine-free all-D chain tokenizable.
    entities = [
        {"type": "protein", "name": "binder", "chirality": "D",
         "sequence": glycine_satisfy_guard(binder_seq)},
        # OPTIONAL target context (omit for the binder-alone fold):
        {"type": "protein", "name": "target", "chirality": "L", "sequence": target_seq},
    ]
    pred = ChaiBackend().predict(entities, cold_start_dir)  # device auto-resolves; NO seed/restraint

Then analyze on CPU with this module:

    cif  = _best_cif_in_chai_out(cold_start_dir / "chai_out")
    frac = backbone_chirality_fraction_from_cif(cif, chain_name="B", chirality_label="D")
    gly  = gly_fraction_from_cif(cif, chain_name="B")
    # frac <= 0.10 (and gly not ~1.0) => handedness recovered UNAIDED.

(The binder is chain "B" when the target is entity 0; for a binder-alone fold it is
chain "A". Pass the chain that holds the binder.)


MIRROR SELF-CONSISTENCY PROTOCOL (the full GPU round-trip; CPU does step 4)
--------------------------------------------------------------------------
``mirror_self_consistency`` here is only the geometry comparator (step 4). The full
round-trip the campaign driver runs on the GPU is:

    1. Take the designed D-complex CIF; extract binder coords + reflect them
       (``xenodesign.mirror.reflect_coords``) and remap residue codes D<->L
       (``xenodesign.mirror.mirror_residue_codes``) — the mirror image is an L design.
    2. Re-predict the mirrored (now-L) complex with a fresh ``ChaiBackend.predict``
       (L is in-manifold for Chai, so this is a clean, restraint-free fold).
    3. Mirror the re-predicted L pose back (reflect again) into D space.
    4. Compare the mirrored-back pose to the original D binder with
       ``mirror_self_consistency(original_coords, repredicted_coords, axis=...)``.
       A small discrepancy means Chai independently re-derived the same pose from
       the mirror image — strong evidence the handedness is intrinsic, not seeded.

Given two coord sets already on hand (e.g. original binder CA + mirrored-back binder
CA), call ``mirror_self_consistency(coords_a, coords_b, axis=0)`` directly.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

from xenodesign.chirality import is_chirality_violation
from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif
from xenodesign.mirror import mirror_discrepancy, reflect_coords

# The fraction at/below which a single iteration's D-chirality is considered to "pass"
# (spec §3 / #36: chir <= 0.10). Inclusive of the boundary.
PASS_THRESHOLD: float = 0.10


def _best_cif_in_chai_out(chai_out: str | Path) -> Path:
    """Return the best-model CIF in a chai output dir (highest aggregate_score).

    Mirrors ``backends.chai_backend.load_prediction``'s selection: rank
    ``scores.model_idx_*.npz`` by ``aggregate_score`` and return the matching
    ``pred.model_idx_{idx}.cif``. If no scores files are present (synthetic
    fixtures / partial runs), fall back to the lexicographically-first CIF so the
    harness still works on a lone hand-made CIF.

    Raises FileNotFoundError if the dir holds no CIF at all.
    """
    chai_out = Path(chai_out)
    score_files = sorted(chai_out.glob("scores.model_idx_*.npz"))
    if score_files:
        best_idx, best_agg = None, -np.inf
        for f in score_files:
            agg = float(np.asarray(np.load(f)["aggregate_score"]).reshape(-1)[0])
            if agg > best_agg:
                best_agg = agg
                best_idx = int(re.search(r"idx_(\d+)", f.name).group(1))
        cif = chai_out / f"pred.model_idx_{best_idx}.cif"
        if cif.exists():
            return cif
        # scores present but matching CIF missing — fall through to any CIF.
    cifs = sorted(chai_out.glob("*.cif"))
    if not cifs:
        raise FileNotFoundError(f"no *.cif under {chai_out}")
    return cifs[0]


def backbone_chirality_fraction_from_cif(
    cif_path: str | Path,
    chain_name: str = "B",
    chirality_label: str = "D",
    epsilon: float = 0.02,
) -> float:
    """D-chirality violation fraction for one chain of a predicted CIF.

    Reads the chain's per-residue backbones via ``backbone_by_residue_from_cif`` and
    counts stereocenters (residues with a CB) whose signed chiral volume disagrees
    with ``chirality_label`` (default 'D'). Glycine (no CB) is achiral and excluded.

    Returns violations / stereocenters, or 0.0 when the chain has no stereocenters
    (e.g. an all-Gly chain — vacuously "clean"; use ``gly_fraction_from_cif`` to
    catch that escape hatch).
    """
    residues = backbone_by_residue_from_cif(cif_path, chain_name)
    total = viol = 0
    for res in residues:
        if "CB" not in res or res["CB"] is None:
            continue
        total += 1
        if is_chirality_violation(
            res["N"], res["CA"], res["C"], res["CB"], chirality_label, epsilon=epsilon
        ):
            viol += 1
    if total == 0:
        return 0.0
    return viol / total


def gly_fraction_from_cif(cif_path: str | Path, chain_name: str = "B") -> float:
    """Fraction of a chain's residues that are achiral (no CB, i.e. glycine).

    This is the achiral escape-hatch detector: a binder that drifts toward all-Gly
    trivially "passes" the chirality fraction (no stereocenters to get wrong), so the
    Gly fraction must be reported alongside it. Returns 0.0 for an empty chain.
    """
    residues = backbone_by_residue_from_cif(cif_path, chain_name)
    if not residues:
        return 0.0
    gly = sum(1 for r in residues if "CB" not in r or r["CB"] is None)
    return gly / len(residues)


def _iter_index(iter_dir: Path) -> int:
    """Numeric index from an ``iter_NNN`` dir name (for numeric, not lexical, ordering)."""
    m = re.search(r"iter_(\d+)", iter_dir.name)
    return int(m.group(1)) if m else -1


def trajectory_chirality_distribution(
    loop_dir: str | Path,
    chain_name: str = "B",
    chirality_label: str = "D",
    pass_threshold: float = PASS_THRESHOLD,
    epsilon: float = 0.02,
) -> dict:
    """Full per-iteration D-chirality distribution over a run dir (anti-survivorship).

    For EVERY ``iter_*`` subdir of ``loop_dir``, locate the best CIF (in
    ``iter_*/chai_out`` if present, else directly in ``iter_*``), measure the binder
    chain's D-chirality violation fraction, and assemble the whole-trajectory view —
    not just the single best structure a downstream selector would surface.

    Args:
        loop_dir: run root containing ``iter_000/``, ``iter_001/``, ... subdirs.
        chain_name: the binder chain to measure (chai emits "B" for binder when the
            target is entity 0; "A" for a binder-alone fold).
        chirality_label: expected handedness of the binder ('D').
        pass_threshold: per-iter violation fraction counted as a "pass" (<=, inclusive).
        epsilon: near-planar tolerance forwarded to the chirality test.

    Returns a dict:
        {
          "n_iters": int,
          "iters": [int, ...],                 # numeric iter indices, ascending
          "per_iter": [float, ...],            # D-violation fraction per iter
          "per_iter_gly_fraction": [float, ...],  # achiral (Gly) fraction per iter
          "mean": float, "max": float, "min": float,
          "pass_fraction": float,              # share of iters with frac <= threshold
          "pass_threshold": float,
          "chain_name": str, "chirality_label": str,
        }

    Raises FileNotFoundError if no ``iter_*`` dir (or no CIF in any of them) is found.
    """
    loop_dir = Path(loop_dir)
    iter_dirs = sorted(
        (d for d in loop_dir.glob("iter_*") if d.is_dir()), key=_iter_index
    )
    if not iter_dirs:
        raise FileNotFoundError(f"no iter_* subdirs under {loop_dir}")

    iters: list[int] = []
    per_iter: list[float] = []
    per_iter_gly: list[float] = []
    for d in iter_dirs:
        chai_out = d / "chai_out"
        search_dir = chai_out if chai_out.is_dir() else d
        cif = _best_cif_in_chai_out(search_dir)
        per_iter.append(
            backbone_chirality_fraction_from_cif(
                cif, chain_name=chain_name, chirality_label=chirality_label, epsilon=epsilon
            )
        )
        per_iter_gly.append(gly_fraction_from_cif(cif, chain_name=chain_name))
        iters.append(_iter_index(d))

    arr = np.asarray(per_iter, dtype=float)
    n_pass = int(np.count_nonzero(arr <= pass_threshold))
    return {
        "n_iters": len(per_iter),
        "iters": iters,
        "per_iter": per_iter,
        "per_iter_gly_fraction": per_iter_gly,
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "pass_fraction": n_pass / len(per_iter),
        "pass_threshold": pass_threshold,
        "chain_name": chain_name,
        "chirality_label": chirality_label,
    }


def _ca_coords_from_cif(cif_path: str | Path, chain_name: str) -> np.ndarray:
    """Per-residue CA coordinates (n_res, 3) for one chain of a CIF."""
    residues = backbone_by_residue_from_cif(cif_path, chain_name)
    if not residues:
        raise ValueError(f"chain {chain_name!r} has no parseable residues in {cif_path}")
    return np.asarray([r["CA"] for r in residues], dtype=float)


def mirror_self_consistency(
    coords_a,
    coords_b=None,
    axis: int = 0,
    chain_name: str = "B",
) -> float:
    """Mirror self-consistency discrepancy (small = self-consistent). CPU geometry only.

    This is step 4 of the mirror→re-predict→mirror-back protocol documented in the
    module docstring: given two coordinate sets that should be mirror images of one
    another (the original D binder and the independently re-predicted, mirrored-back
    pose), return the Kabsch RMSD between ``coords_a`` and the reflection of
    ``coords_b`` (delegating to ``xenodesign.mirror.mirror_discrepancy``). Zero means
    B is the exact mirror of A up to a rigid motion.

    Two call forms:
      - ``mirror_self_consistency(coords_a, coords_b, axis=...)`` — two coord arrays
        (each (n, 3)); the GPU round-trip's comparison step.
      - ``mirror_self_consistency(cif_path, axis=..., chain_name=...)`` — single CIF:
        reads the chain's CA coords and compares them to their OWN reflection. This is
        the cheap CPU pre-screen (a chain symmetric about the reflection axis after
        realignment scores ~0); it does NOT substitute for the full re-predict round
        trip, which is the real test.

    Returns a non-negative float (RMSD in CIF coordinate units, Å).
    """
    if coords_b is None:
        # Single-CIF self form: compare the binder's CA coords to their own reflection.
        ca = _ca_coords_from_cif(coords_a, chain_name)
        return mirror_discrepancy(ca, reflect_coords(ca, axis=axis), axis=axis)
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    return mirror_discrepancy(a, b, axis=axis)
