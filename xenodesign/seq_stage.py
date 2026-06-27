"""SequenceUpdate — the single owner of extract · known_seq · chirality · canonical-residue
anchor · de-gaming for the unified pipeline spine (spec §3.2). S1 builds the stage + its four
invariant contract tests and routes the GREEDY path through it (parity vs the S0 goldens);
beam/ABC migrate in S2. The `frozen` input is a generic set of FrozenPosition so it is ready for
the sequence_constraints config model (spec §3.3, S3) without an interface change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FrozenPosition:
    """A position the stage must hold fixed, decoupled from WHY (spec §3.3).

    position0: 0-based index in the binder chain.
    identity:  one-letter L identity to hold (None = hold geometry only).
    chirality: 'L' | 'D' | None (None = inherit the position's existing handedness).
    """
    position0: int
    identity: Optional[str] = None
    chirality: Optional[str] = None


class SequenceUpdate:
    """Owns the five seq-update concerns. S1.2–S1.5 fill in the behavior behind the four
    invariant contracts; S1.1 is just the constructor + field storage."""

    def __init__(self, *, roles=None, frozen=None, num_seqs: int = 8,
                 design_fn=None):
        self.roles = roles
        self.frozen = set(frozen or ())
        self.num_seqs = int(num_seqs)
        self._design_fn = design_fn

    def encode_d_fasta(self, one_letter: str, chirality_pattern=None) -> str:
        """Invariant #2 (chirality survives): ONE encoder for the next-cycle Chai sequence.

        `chirality_pattern` is 0-based {pos: 'L'|'D'}. None (or an all-'D' pattern) reproduces the
        whole-chain all-D `to_d_fasta` — the SPECIAL CASE, never a silent override (the bug at
        io_spec.py:23 was applying to_d_fasta unconditionally, flattening declared-L coordinators).
        Otherwise delegate to the shared `mixed_chirality_fasta` primitive (1-based keys), so L
        positions stay bare canonical and D positions become parenthesized D-CCD; Gly stays 'G'.
        """
        from xenodesign.io_spec import to_d_fasta
        if chirality_pattern is None or set(chirality_pattern.values()) <= {"D"}:
            return to_d_fasta(one_letter)
        import xenodesign.classes.base  # noqa: F401  (prime the base<->cyclic import cycle)
        from xenodesign.classes.cyclic import mixed_chirality_fasta
        fixed = {pos + 1: hand for pos, hand in chirality_pattern.items()}
        return mixed_chirality_fasta(one_letter, fixed_chirality=fixed)

    def ensure_canonical_anchor(self, one_letter: str, chirality_pattern=None,
                                frozen=None) -> str:
        """Invariant #3: THE single canonical-residue rule (collapses the 3 divergent anchors:
        _alpha_internals.py:243/:256, non_alpha.py:152, cyclic.py:324).

        Ensure >=1 Chai-canonical residue. Add a C-terminal Gly ONLY when the chain has no
        L-residue AND no Gly already; place it at the last NON-frozen position so a declared
        coordinator (even at the C-terminus) is never modified. A chain that already carries an
        L residue (e.g. the 2L+2D-His cycle) is tokenizable as-is and gets NO Gly.

        `chirality_pattern` is 0-based {pos: 'L'|'D'}; positions absent default to 'D' (the
        historical all-D backbone default). `frozen` positions are excluded from placement.
        """
        if "G" in one_letter:
            return one_letter
        n = len(one_letter)
        pat = dict(chirality_pattern or {})
        hands = {pat.get(i, "D") for i in range(n)}
        if "L" in hands:
            return one_letter            # an L residue is tokenizable -> no Gly needed
        pinned = {fp.position0 for fp in (frozen if frozen is not None else self.frozen)}
        for i in range(n - 1, -1, -1):   # C-terminal-most NON-frozen position
            if i not in pinned:
                return one_letter[:i] + "G" + one_letter[i + 1:]
        return one_letter                # every position frozen (impossible in practice)

    def build_loop_fn(self, extract_fn, chirality_pattern=None):
        """Assemble the four primitives into ``fn(prediction) -> str`` for HalluLoop.

        `extract_fn(prediction)` returns design_backbone / context_coords / context_elements /
        prev_l_seq. Per call: build the real known_seq (invariant #1), run the de-gamed
        SequenceUpdater (MultiCandidate + sequence_quality_key, invariant: de-gaming), apply the
        canonical anchor (invariant #3), then emit one_letter (all-D path) or the per-position
        d_fasta (invariant #2). frozen positions are forced fixed in the MPNN mask.
        """
        from xenodesign.inverse_folding import MultiCandidate
        from xenodesign.io_spec import AA1_TO_AA3
        from xenodesign.mirror import L_TO_D
        from xenodesign.scorer import sequence_quality_key
        from xenodesign.sequence_update import SequenceUpdater

        frozen0 = {fp.position0 for fp in self.frozen}
        design_fn = MultiCandidate(self._design_fn, num_seqs=self.num_seqs,
                                   key_fn=sequence_quality_key)
        updater = SequenceUpdater(design_fn=design_fn,
                                  frozen_positions=frozen0 or None)

        def _fn(prediction) -> str:
            ext = extract_fn(prediction)
            prev = ext["prev_l_seq"]
            known = self.build_known_seq(prev_l_seq=prev)
            n = len(known)
            # design_codes carry the REAL identity (L-projected) per position so the SequenceUpdater
            # feeds it as known_seq — invariant #1. Frozen donors keep their declared handedness.
            pat = dict(chirality_pattern or {})
            codes = []
            for i, ch in enumerate(known):
                three = AA1_TO_AA3.get(ch.upper(), "ALA")
                codes.append(L_TO_D.get(three, three) if pat.get(i, "D") == "D" else three)
            result = updater.update(
                design_backbone=ext["design_backbone"],
                design_codes=codes,
                context_coords=ext["context_coords"],
                context_elements=ext["context_elements"],
                chirality_pattern=({i: pat.get(i, "D") for i in range(n)}
                                   if chirality_pattern is not None else None),
            )
            anchored = self.ensure_canonical_anchor(result.one_letter,
                                                    chirality_pattern=chirality_pattern)
            if chirality_pattern is None:
                return anchored
            return self.encode_d_fasta(anchored, chirality_pattern=chirality_pattern)

        return _fn

    def build_known_seq(self, prev_l_seq: str, frozen=None) -> str:
        """Invariant #1 (real evolving context) + frozen-identity override.

        Start from the REAL previous-iteration L sequence (each char a one-letter L residue, the
        chirality-agnostic L projection of whatever the CIF held), then stamp any frozen position's
        declared identity over it. This is what becomes the backend's `known_seq` so fixed positions
        keep their donor identity and FREE positions condition on real context — replacing the
        all-Ala `design_codes=["DAL"]*n` starvation (root cause: _alpha_internals.py:472).
        """
        chars = list(prev_l_seq)
        for fp in (frozen if frozen is not None else self.frozen):
            if fp.identity is not None and 0 <= fp.position0 < len(chars):
                chars[fp.position0] = fp.identity.upper()
        return "".join(chars)
