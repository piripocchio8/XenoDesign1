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
