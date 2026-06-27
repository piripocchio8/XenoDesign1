"""S3a.1b (#18): harden ensure_canonical_anchor edge cases for the metal path — empty chain,
all-frozen chain, a chain that already contains a Gly at a frozen position, and a genuinely mixed
L/D chain (already tokenizable -> never anchored).

Also pins the VALIDATION hardening: the primitive must RAISE on a genuinely-unknown residue
letter (not in AA1_TO_AA3) so a future caller cannot silently rescue invalid input via the
Gly-anchor path. Pre-encoded ncAA '(XXX)' blocks must be SKIPPED (not treated as invalid single
letters). All valid L/D 1-letter codes + Gly must pass unchanged.
"""
from __future__ import annotations

import pytest

from xenodesign.seq_stage import FrozenPosition, SequenceUpdate


# ── edge-case pins (metal path) ──────────────────────────────────────────────

def test_anchor_empty_chain_returns_empty():
    assert SequenceUpdate().ensure_canonical_anchor("") == ""


def test_anchor_existing_gly_is_noop():
    # A Gly anywhere already satisfies the >=1-canonical requirement (Gly is achiral).
    assert SequenceUpdate().ensure_canonical_anchor("AAGAA", chirality_pattern=None) == "AAGAA"


def test_anchor_mixed_LD_chain_not_anchored():
    # An L residue present -> tokenizable as-is -> no Gly added.
    pat = {0: "D", 1: "L", 2: "D"}
    assert SequenceUpdate().ensure_canonical_anchor("HKH", chirality_pattern=pat) == "HKH"


def test_anchor_all_frozen_chain_unchanged():
    # Every position frozen (a fully-pinned coordinator scaffold) -> no free slot -> chain unchanged.
    frozen = {FrozenPosition(position0=i, identity="H", chirality="D") for i in range(3)}
    out = SequenceUpdate(frozen=frozen).ensure_canonical_anchor(
        "HHH", chirality_pattern={0: "D", 1: "D", 2: "D"})
    assert out == "HHH"


def test_anchor_all_D_gets_one_cterm_gly_at_last_free():
    frozen = {FrozenPosition(position0=0, identity="H", chirality="D")}  # coordinator at N-term
    out = SequenceUpdate(frozen=frozen).ensure_canonical_anchor(
        "HKW", chirality_pattern={0: "D", 1: "D", 2: "D"})
    assert out == "HKG"          # last NON-frozen position becomes Gly; the frozen His is untouched


# ── validation hardening ─────────────────────────────────────────────────────

def test_anchor_raises_on_unknown_residue():
    """HARDENING: a letter not in AA1_TO_AA3 must raise ValueError (not silently become Gly)."""
    with pytest.raises((ValueError, KeyError)):
        SequenceUpdate().ensure_canonical_anchor(
            "AZC", chirality_pattern={0: "D", 1: "D", 2: "D"})


def test_anchor_raises_on_unknown_residue_all_d_path():
    """Unknown letter in a purely all-D chain (no Gly, no L) must still raise."""
    with pytest.raises((ValueError, KeyError)):
        SequenceUpdate().ensure_canonical_anchor(
            "Z", chirality_pattern={0: "D"})


def test_anchor_does_not_raise_on_valid_L_residues():
    """Standard L 1-letter codes must pass through unchanged (with an L present, no Gly added)."""
    pat = {0: "L", 1: "L", 2: "L", 3: "L"}
    result = SequenceUpdate().ensure_canonical_anchor("ACDE", chirality_pattern=pat)
    assert result == "ACDE"


def test_anchor_does_not_raise_on_valid_D_residues():
    """Standard D residue letters (valid AA1_TO_AA3 keys) must not raise; Gly anchor is added."""
    pat = {0: "D", 1: "D", 2: "D", 3: "D"}
    result = SequenceUpdate().ensure_canonical_anchor("ACDE", chirality_pattern=pat)
    assert result == "ACDG"  # last position -> Gly anchor


def test_anchor_skips_ncaa_blocks():
    """Pre-encoded ncAA '(XXX)' blocks must be SKIPPED (not treated as individual invalid chars)."""
    # Sequence with an ncAA block at position 0 and valid residues after; L present -> no Gly added
    # but critically no raise on the '(' chars in the flat string view.
    pat = {0: "D", 1: "L", 2: "D"}
    # Pass the FLAT one-letter view (before encoding); ncAA blocks appear as '(' in flat string
    # only after encoding. The primitive receives the one-letter identity string, not yet encoded.
    # So we test the actual scenario: a sequence like "(AIB)AH" which has ncAA already embedded.
    result = SequenceUpdate().ensure_canonical_anchor("(AIB)AH", chirality_pattern={0: "D", 1: "L", 2: "D"})
    # L present (pos 1) -> no Gly added; (AIB) block is a valid ncAA token, not an unknown letter
    assert result == "(AIB)AH"


def test_anchor_does_not_raise_on_gly():
    """Gly 'G' must always pass (and short-circuit the all-D path immediately)."""
    result = SequenceUpdate().ensure_canonical_anchor("G", chirality_pattern={0: "D"})
    assert result == "G"
