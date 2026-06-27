"""Strategy-uniform run stages for the unified pipeline spine (S3a).

These are FREE functions (not BinderClass hooks) precisely because the concerns they own —
sequence_constraints production (spec §3.3), Restraints emission (§3.4), and the composed Gates —
must be identical across greedy / beam / ABC, none of which share a class loop. ``run_design`` and
``dispatch._run_abc`` both call them. Pure CPU; heavy imports (chai/gemmi/MetalHawk) are deferred.

All S3a routing is gated on ``XENO_SEQ_STAGE`` (default OFF) at the CALL SITES (dispatch); these
helpers themselves are pure and flag-agnostic so they are independently unit-testable.
"""
from __future__ import annotations

from typing import Optional

from pathlib import Path

from xenodesign.seq_stage import FrozenPosition


def frozen_from_coord_residues(coord_residues) -> set:
    """The spec §3.3 metal-coordination producer: declared coordinator tuples -> the FrozenPosition
    set carrying IDENTITY + CHIRALITY (not the position-only set S2 used).

    Each tuple is ``(pos1based, one_letter[, three_letter, chirality, atom])`` as stored by the
    ``--coord_residues`` flag (see ``classes/cyclic.py:_coord_residues``). The 4th element
    (chirality 'L'/'D') is carried so ``encode_d_fasta`` / ``ensure_canonical_anchor`` honour the
    donor handedness; the 2nd (one_letter, e.g. 'H') is carried as ``identity`` so
    ``build_known_seq`` pins the donor identity. Back-compat: a 2-tuple yields ``chirality=None``.
    """
    out = set()
    for t in (coord_residues or ()):
        pos0 = int(t[0]) - 1
        identity = t[1] if len(t) > 1 else None
        chirality = t[3] if len(t) > 3 else None
        out.add(FrozenPosition(position0=pos0, identity=identity, chirality=chirality))
    return out


def build_run_restraints(cfg, *, out_dir, case=None, target_ctx=None, roles=None):
    """The spec §3.4 uniform restraint emitter: coordination + closure rows for ONE run, reusing the
    cyclic class's metal/closure logic and the benchmark @atom covalent grammar.

    Returns the written ``.restraints`` Path, or ``None`` when restraints are off. The metal case
    (target_type != 'none' with declared coordinators) gets native-covalent ``@atom`` coordination
    rows + a head-to-tail closure row; a non-metal cyclic gets closure-only. Delegates to the SAME
    ``Cyclic.restraints`` hook the greedy path uses, so the rows are byte-identical across strategies
    — the unification the spec asks for (ABC currently emits NONE of these).
    """
    if not getattr(cfg, "restraints_on", False):
        return None

    from xenodesign.benchmark.cases import get_case
    from xenodesign.classes.cyclic import Cyclic

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = case or get_case("cyclic")
    # Cyclic.restraints owns the metal-coordination + closure emission (and the declared-coordinator
    # REQUIRE guard); it reads coord_residues + the closure default from cfg. target_ctx carries the
    # assembled entity list when available (drives the Zn/binder chain split); None -> legacy order.
    return Cyclic().restraints(cfg, case, out_dir, target_ctx)
