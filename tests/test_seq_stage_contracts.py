"""The FOUR invariant contract tests for the unified SequenceUpdate stage (spec §4).

Pure CPU, fake inverse-folding backend. Each invariant gets a focused unit test; these are the
single place the recurring sequence-collapse / all-D-flatten / all-D-crash bug class is pinned.
"""
from __future__ import annotations

import numpy as np

from xenodesign.seq_stage import FrozenPosition, SequenceUpdate


def test_frozen_position_fields():
    fp = FrozenPosition(position0=5, identity="H", chirality="D")
    assert (fp.position0, fp.identity, fp.chirality) == (5, "H", "D")


def test_invariant1_known_seq_carries_real_free_positions():
    """Invariant #1: known_seq is the prior iteration's ACTUAL free-position residues, not all-Ala."""
    stage = SequenceUpdate()
    prev = "MKWVTFGLAG"   # the real evolving sequence from the previous iteration's CIF
    known = stage.build_known_seq(prev_l_seq=prev, frozen=set())
    assert known == prev            # free positions preserved verbatim — NOT "AAAAAAAAAA"
    assert "A" * len(prev) != known


def test_invariant1_frozen_identity_overrides_at_its_position():
    """A frozen position with a declared identity is forced to that identity in known_seq;
    the rest of the chain still carries the real evolving residues."""
    stage = SequenceUpdate()
    prev = "MKWVTFGLAG"
    frozen = {FrozenPosition(position0=2, identity="H", chirality="D")}
    known = stage.build_known_seq(prev_l_seq=prev, frozen=frozen)
    assert known[2] == "H"          # frozen His donor identity wins
    assert known[:2] == "MK" and known[3:] == "VTFGLAG"   # free positions untouched
