"""ABC design_fn variants — the A/B axis split (spec §4.3 / §5.4).

Each variant is a ``design_fn(identity, chirality_pattern) -> identity`` the ABC engine calls
to fill / mutate the identity sequence for a (possibly perturbed) chirality pattern:

- **Variant A** (``abc_variant_a_design_fn``): ABC owns the CHIRALITY pattern; identity is filled
  per-pattern by MPNN via ``SequenceUpdater.update(..., chirality_pattern=pattern)`` (the T1
  per-position-handed subset-reflection path). Identity has a cheap prior — don't search it blindly.
- **Variant B** (``abc_variant_b_design_fn``): ABC searches IDENTITY (point mutations over the 20
  canonical AAs) + chirality; MPNN is warm-start only (the initial population). This variant does
  NOT call MPNN per move — it perturbs the warm-start identity directly.

Both are pure given their injected backend / rng so they are CPU-testable with fakes before any GPU.
The engine treats ``design_fn`` as a black box, so the A-vs-B difference lives entirely here.
"""
from __future__ import annotations

import random
from typing import Mapping

import numpy as np

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"  # the 20 standard amino acids (Variant B alphabet)


def _resolve_design_backbone(last_structure, n: int) -> np.ndarray:
    """Coerce the engine's opaque ``last_structure`` into an ``(n, 4, 3)`` N/CA/C/CB backbone for
    the coordinate-only LigandMPNN adapter (FIX 1).

    - ``None`` → a zero placeholder of length ``n`` (warm-start / no structure yet; the prior
      structure-blind behaviour, kept for the first eval).
    - an array-like ``(n, 4, 3)`` → used as-is (a backbone already decoded by the caller / tests).
    - a CIF path (str / ``os.PathLike``) → parsed best-effort into per-residue N/CA/C/CB via
      ``backbone_by_residue_from_cif`` (GLY → CB filled from CA so the array stays ``(n, 4, 3)``);
      on any parse failure or length mismatch we fall back to the zero placeholder so a single bad
      structure never crashes the search.
    """
    if last_structure is None:
        return np.zeros((n, 4, 3), dtype=float)

    arr = np.asarray(last_structure, dtype=float) if not _is_path_like(last_structure) else None
    if arr is not None and arr.shape == (n, 4, 3):
        return arr

    if _is_path_like(last_structure):
        bb = _backbone_from_cif(last_structure, n)
        if bb is not None:
            return bb
    return np.zeros((n, 4, 3), dtype=float)


def _is_path_like(x) -> bool:
    import os
    return isinstance(x, (str, os.PathLike))


def _context_from_structure(last_structure):
    """Resolve (context_coords, context_elements) from the candidate's last structure.

    For S2: a CIF path -> the binder chain's all-atom context via _all_atoms_from_chain; anything
    else (None / a bare backbone array / parse failure) -> empty context (the legacy fallback, so a
    missing structure never crashes the search). The single-chain mixed-chirality ABC design has no
    separate target chain, so context here is best-effort; the full target-driven context belongs to
    S3's restraints/Target-axis unification."""
    if not _is_path_like(last_structure):
        return np.zeros((0, 3)), []
    try:
        from xenodesign.cif_io import _all_atoms_from_chain
        ctx_coords, ctx_elements = _all_atoms_from_chain(last_structure, "A")
        return np.asarray(ctx_coords), list(ctx_elements)
    except Exception:
        return np.zeros((0, 3)), []


def _backbone_from_cif(cif_path, n: int):  # pragma: no cover (gemmi/gpu path)
    """Best-effort parse a CIF into an ``(n, 4, 3)`` N/CA/C/CB backbone (binder chain 'A')."""
    try:
        from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif

        residues = backbone_by_residue_from_cif(cif_path, chain_name="A")
        if len(residues) != n:
            return None
        bb = np.zeros((n, 4, 3), dtype=float)
        for i, res in enumerate(residues):
            bb[i, 0] = res["N"]
            bb[i, 1] = res["CA"]
            bb[i, 2] = res["C"]
            bb[i, 3] = res.get("CB", res["CA"])  # GLY has no CB → reuse CA so the frame is filled
        return bb
    except Exception:
        return None


def abc_variant_a_design_fn(backend, *, roles=None, frozen=None):
    """Variant A: MPNN fills IDENTITY per chirality pattern (T1 per-position-handed path).

    Returns ``design_fn(identity, chirality_pattern) -> identity`` (the PLAIN one-letter L seq;
    the fitness adapter applies the per-position mixed_chirality_fasta emit). ``backend`` is an
    InverseFoldingBackend (the coordinate-only LigandMPNN adapter in production; a fake in tests).

    S2.2 (SequenceUpdate routing, flag-on): when ``XENO_SEQ_STAGE != "0"``, the REAL evolving
    ``identity`` is fed to the backend as ``known_seq`` (built via ``SequenceUpdate.build_known_seq``
    so declared-coordinator ``frozen`` positions keep their identity) — replacing the legacy
    all-Ala ``design_codes`` starvation (variants.py:104-106) — and the candidate's real backbone +
    ligand/Zn context are resolved from ``last_structure`` (the CIF the fitness publishes) instead
    of the legacy empty context (variants.py:111-112). Flag off => the legacy all-Ala/empty-context
    body, byte-identical. ``roles``/``frozen`` are only consulted on the flag-on path.
    """
    import os

    from xenodesign.sequence_update import SequenceUpdater

    updater = SequenceUpdater(design_fn=backend,
                              frozen_positions=set(frozen) if frozen else None)
    routed = os.environ.get("XENO_SEQ_STAGE", "0") != "0"
    stage = None
    if routed:
        from xenodesign.seq_stage import SequenceUpdate
        stage = SequenceUpdate(roles=roles, frozen=set(frozen or ()))

    def design_fn(identity: str, chirality_pattern: Mapping[int, str],
                  last_structure=None) -> str:
        n = len(chirality_pattern)
        # Per-position-handed design codes so choose_reflection lands in the majority frame and the
        # searched handedness is preserved by the chirality_pattern emit.
        design_codes = [
            "DAL" if chirality_pattern[i] == "D" else "ALA" for i in range(n)
        ]
        design_backbone = _resolve_design_backbone(last_structure, n)
        if routed:
            # Invariant #1: known_seq is the REAL evolving identity (frozen positions pinned).
            known = stage.build_known_seq(prev_l_seq=identity[:n].ljust(n, "A"))
            # Real ligand/Zn context from the candidate CIF (S2 scope: the seq-update needs the
            # context so MPNN conditions free positions on the real interface, not empty arrays).
            ctx_coords, ctx_elements = _context_from_structure(last_structure)
            result = updater.update(
                design_backbone=design_backbone,
                design_codes=design_codes,
                context_coords=ctx_coords,
                context_elements=ctx_elements,
                chirality_pattern=dict(chirality_pattern),
                known_seq=known,
            )
            return result.one_letter
        result = updater.update(
            design_backbone=design_backbone,
            design_codes=design_codes,
            context_coords=np.zeros((0, 3)),
            context_elements=[],
            chirality_pattern=dict(chirality_pattern),
        )
        # Return the PLAIN one-letter L identity: the fitness adapter re-emits the per-position
        # handedness via mixed_chirality_fasta. (Returning result.d_fasta double-encodes → -inf.)
        return result.one_letter

    return design_fn


def abc_variant_b_design_fn(*, rng: random.Random | None = None,
                            mutation_rate: float = 0.3,
                            alphabet: str = CANONICAL_AA,
                            ncaa_palette=None,
                            frozen=None):
    """Variant B: ABC point-mutates the warm-start IDENTITY (MPNN is initial-population only).

    Returns ``design_fn(identity, chirality_pattern) -> identity`` that, at each call, mutates
    each position of ``identity`` to a fresh canonical AA with probability ``mutation_rate``.
    ``mutation_rate=0`` returns the identity unchanged (the MPNN warm-start is kept). The chirality
    pattern is not consumed here — the engine perturbs chirality separately; identity is the only
    axis this design_fn owns.

    track #2 — when ``ncaa_palette`` is non-empty, each call ALSO applies one ncAA identity move
    (``moves.ncaa_identity_move``): a non-frozen position may be set to a palette ncAA ``(XXX)``
    block (or an existing ncAA reverted to canonical). An empty/absent palette keeps the prior
    canonical-only behaviour. ``frozen`` (0-based declared-coordinator positions) are never
    mutated — neither by the canonical point mutation nor the ncAA move.

    Args:
        rng: randomness source (defaults to a fresh ``Random()``); inject for reproducibility.
        mutation_rate: per-position mutation probability in [0, 1].
        alphabet: the candidate amino-acid alphabet (default the 20 canonical AAs).
        ncaa_palette: VALIDATED CCD codes the ncAA move may propose (empty/None → ncAA OFF).
        frozen: 0-based positions (declared coordinators) never mutated.
    """
    from xenodesign.abc.moves import identity_tokens, ncaa_identity_move

    rng = rng or random.Random()
    palette = list(ncaa_palette or ())
    frozen = set(frozen or ())

    def design_fn(identity: str, chirality_pattern: Mapping[int, str]) -> str:
        if mutation_rate > 0.0:
            # Per-position canonical point mutation. Tokenize so ncAA ``(XXX)`` blocks are treated
            # as single positions (preserved unless explicitly reverted by the ncAA move) and the
            # position index lines up with ``frozen``.
            out = []
            for i, tok in enumerate(identity_tokens(identity)):
                if i in frozen or tok.startswith("("):
                    out.append(tok)  # never mutate a frozen position or an existing ncAA block here
                elif rng.random() < mutation_rate:
                    # Draw a DIFFERENT residue so an effective rate-1.0 move always changes identity.
                    choices = [c for c in alphabet if c != tok] or list(alphabet)
                    out.append(rng.choice(choices))
                else:
                    out.append(tok)
            identity = "".join(out)
        if palette:
            identity = ncaa_identity_move(identity, rng, palette=palette, frozen=frozen)
        return identity

    return design_fn
