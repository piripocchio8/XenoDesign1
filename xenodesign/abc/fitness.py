"""ABC fast-cycle fitness adapter (spec §5.3; objective re-decided 2026-06-25).

The bee fitness is a SHORT low-diffusion-step Chai predict of the single mixed-chirality
peptide, scored on the only terms the cyclization calibration found discriminating:

    score = w_ptm * pTM  +  w_termini * termini_proximity

(docs/results/2026-06-25-cyclization-calibration.md). **pTM is primary** (ΔpTM ≈ +0.21,
large/stable/already-saturated at K=10); the **C-N termini-distance closure proxy is the
secondary** discriminator (always positive; essential for short peptides whose pTM chai
under-estimates). We DROP mainchain-pLDDT and chirality (both saturate for real and fake
alike) and the diluting 4-term aggregate.

The predict runs at ``k_star`` (10-25) steps WITH the head-to-tail CLOSURE restraint ONLY —
NO target-specific coordination terms — so the fitness generalizes to ANY mixed-chirality
design (cyclic + no-target). The mixed-chirality chain is emitted via ``mixed_chirality_fasta``
(the same per-position-handedness emit as the T1 mirror-back). Heavy imports (gemmi via
``head_to_tail_closure_geometry_from_cif``) stay deferred so this module imports CPU-clean.

The adapter is **best-effort / guarded**: any predict / parse failure returns ``-inf`` so a
single bad evaluation never crashes the colony.

WEIGHTS ARE USER-TUNABLE: ``w_ptm`` / ``w_termini`` are knobs (defaults 0.7 / 0.3); the
termini-proximity decay ``_PROXIMITY_SCALE`` is the one calibrated constant.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

# Bound at module level so the fitness tests can monkeypatch
# ``head_to_tail_closure_geometry_from_cif`` without a real CIF; gemmi itself is deferred inside
# that helper, so this import stays CPU-clean. Prime ``classes.base`` BEFORE ``classes.cyclic`` to
# break the base<->cyclic import cycle (base defines SeedSpec before re-importing cyclic.Cyclic;
# importing cyclic first hits a partially-initialised-module ImportError) — same guard as
# sequence_update.py.
import xenodesign.classes.base  # noqa: F401
from xenodesign.classes.cyclic import head_to_tail_closure_geometry_from_cif

# Ideal head-to-tail peptide-bond C-N distance (A); proximity == 1.0 at/below this.
_PEPTIDE_BOND_A = 1.33
# Exponential decay length (A) for the termini-proximity normalization: at 1.33 A proximity
# is 1.0, decaying smoothly toward 0 as the termini open up (≈0.06 at 18 A, ≈0 at 40 A).
_PROXIMITY_SCALE = 6.0


def termini_proximity(cn_distance: float | None) -> float:
    """Normalize a C-N termini distance (A) to a closeness in [0, 1] (closer = higher).

    1.0 at/below the ideal peptide-bond distance (~1.33 A); a smooth exponential decay as the
    termini open. ``None`` (closure geometry unavailable) → 0.0.
    """
    if cn_distance is None:
        return 0.0
    excess = max(0.0, float(cn_distance) - _PEPTIDE_BOND_A)
    return float(math.exp(-excess / _PROXIMITY_SCALE))


def _closure_restraint_path(sequence: str, out_dir: Path, *, chain: str = "A") -> Path:
    """Write the head-to-tail COVALENT closure restraint (NO coordination terms) and return it.

    Reuses ``cyclic.build_closure_row`` (== ``head_to_tail_closure_row``): C-term carbonyl C of
    the last residue bonded to the N-term amide N of residue 1, on the binder chain. This is the
    ONLY restraint — it generalizes to any mixed-chirality design (no Zn / no metal coordination).
    """
    from types import SimpleNamespace

    from xenodesign.benchmark.restraints import write_restraints
    from xenodesign.classes.cyclic import build_closure_row

    seed_result = SimpleNamespace(one_letter=sequence)
    row = build_closure_row(seed_result, chain=chain)
    return write_restraints(Path(out_dir) / "abc_closure.restraints", [row])


def make_abc_fitness(
    backend,
    *,
    k_star: int = 15,
    w_ptm: float = 0.7,
    w_termini: float = 0.3,
    closure: bool = True,
    chain_name: str = "A",
    out_root: "str | Path | None" = None,
):
    """Build ``fitness(sequence, chirality_pattern) -> float`` over a fast Chai predict.

    Runs a SHORT (``k_star`` step) full predict of the single mixed-chirality peptide, emitted
    via ``mixed_chirality_fasta`` from ``sequence`` + ``chirality_pattern``, WITH the head-to-tail
    closure restraint (when ``closure``). Reads pTM + the C-N termini distance from the predicted
    CIF and returns ``w_ptm*pTM + w_termini*termini_proximity``.

    Args:
        backend: a Chai backend exposing ``predict(entities, out_dir,
            num_diffn_timesteps=, constraint_path=) -> prediction`` (prediction carries ``ptm``
            and an attached ``_cif_path``). The dispatcher wires the real ``ChaiBackend``; tests
            inject a fake.
        k_star: fast diffusion-step count (10-25; the calibrated cheap operating point).
        w_ptm / w_termini: objective weights (user-tunable). pTM is primary; termini is
            secondary (essential for short peptides where chai under-estimates pTM).
        closure: emit + pass the head-to-tail closure restraint (default True).
        chain_name: the peptide chain letter (default ``"A"`` — single-chain mixed-chirality).
        out_root: where per-evaluation chai dirs are written (default a temp dir per call).

    Returns:
        ``fitness(sequence, chirality_pattern) -> float`` (``-inf`` on any failure).
    """
    import tempfile

    from xenodesign.classes.cyclic import mixed_chirality_fasta

    base_out = Path(out_root) if out_root is not None else None
    _counter = {"n": 0}

    def fitness(sequence: str, chirality_pattern: Mapping[int, str]) -> float:
        # Side-channel reset: the engine reads ``fitness.last_structure`` after each call to record
        # the just-predicted structure onto the FoodSource and thread it to Variant A's design_fn
        # (FIX 1). Cleared up-front so a failed eval never re-publishes a stale structure.
        fitness.last_structure = None
        try:
            # S2.3: canonical-residue anchor (invariant #3) — an all-D chain crashes Chai
            # tokenization; ensure >=1 canonical residue (a C-terminal Gly when no L/Gly present)
            # before the emit. Variant-agnostic (acts on the chain ABOUT to be encoded). Flag off
            # keeps the legacy emit byte-identical. frozen positions are not pinned here (the
            # fitness has no coordinator set in scope — S3's restraints unification threads them);
            # for S2 the anchor's own frozen=set() default is correct for the no-coordinator case.
            import os
            seq_for_emit = sequence
            if os.environ.get("XENO_SEQ_STAGE", "0") != "0":
                from xenodesign.seq_stage import SequenceUpdate
                seq_for_emit = SequenceUpdate().ensure_canonical_anchor(
                    sequence, chirality_pattern=dict(chirality_pattern))
            # Per-position-handedness emit (T1): 0-based pattern → 1-based for mixed_chirality_fasta.
            fixed_chirality = {pos + 1: hand for pos, hand in chirality_pattern.items()}
            d_fasta = mixed_chirality_fasta(seq_for_emit, fixed_chirality=fixed_chirality)

            _counter["n"] += 1
            if base_out is not None:
                out_dir = base_out / f"abc_eval_{_counter['n']:05d}"
            else:
                out_dir = Path(tempfile.mkdtemp(prefix="abc_eval_"))
            out_dir.mkdir(parents=True, exist_ok=True)

            constraint_path = None
            if closure:
                constraint_path = _closure_restraint_path(sequence, out_dir, chain=chain_name)

            entities = [
                {"type": "protein", "name": "binder", "sequence": d_fasta, "chirality": "L"},
            ]
            pred = backend.predict(
                entities, out_dir,
                num_diffn_timesteps=int(k_star),
                constraint_path=constraint_path,
            )

            ptm = float(getattr(pred, "ptm", 0.0) or 0.0)
            ptm = max(0.0, min(1.0, ptm))

            cif = getattr(pred, "_cif_path", None)
            # Publish the predicted structure (CIF path) for the engine to thread into Variant A's
            # structure-aware re-design (FIX 1). The candidate's backbone now drives MPNN.
            fitness.last_structure = cif
            if cif is None:
                # No structure to read termini from → pTM-led fallback (never crash the search).
                return float(w_ptm * ptm)
            geom = head_to_tail_closure_geometry_from_cif(cif, chain_name=chain_name) or {}
            prox = termini_proximity(geom.get("cn_distance"))
            return float(w_ptm * ptm + w_termini * prox)
        except Exception:
            return float("-inf")

    fitness.last_structure = None  # structure side-channel the ABC engine reads (FIX 1)
    return fitness
