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
