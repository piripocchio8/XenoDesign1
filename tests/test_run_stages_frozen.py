"""S3a.1: frozen_from_coord_residues lifts declared coordinator tuples into identity+chirality
FrozenPosition (the spec §3.3 metal producer), upgrading S2's position-only frozen so the donor
identity (His) and handedness (L/D) survive into known_seq/encode/anchor for greedy AND ABC."""
from __future__ import annotations

from xenodesign.run_stages import frozen_from_coord_residues
from xenodesign.seq_stage import FrozenPosition


def test_frozen_from_coord_residues_carries_identity_and_chirality():
    coord = [(6, "H", "HIS", "L", "ND1"), (12, "H", "HIS", "D", "ND1")]
    frozen = frozen_from_coord_residues(coord)
    assert frozen == {
        FrozenPosition(position0=5, identity="H", chirality="L"),
        FrozenPosition(position0=11, identity="H", chirality="D"),
    }


def test_frozen_from_coord_residues_handles_short_and_empty():
    assert frozen_from_coord_residues([]) == set()
    assert frozen_from_coord_residues(None) == set()
    # 2-tuple back-compat (pos, one_letter) — chirality unknown -> None.
    assert frozen_from_coord_residues([(3, "C")]) == {
        FrozenPosition(position0=2, identity="C", chirality=None)}


from xenodesign.seq_stage import SequenceUpdate


def test_build_known_seq_pins_coordinator_identity():
    coord = [(2, "H", "HIS", "L", "ND1")]            # His at 1-based pos 2 (0-based 1)
    stage = SequenceUpdate(frozen=frozen_from_coord_residues(coord))
    # The real evolving seq has 'A' at the coordinator slot; the frozen identity stamps 'H' over it.
    assert stage.build_known_seq(prev_l_seq="AAAAA") == "AHAAA"
