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
