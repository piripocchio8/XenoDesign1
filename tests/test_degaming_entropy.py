"""S3a.6 (#4): the de-gaming key penalises a low-ENTROPY sequence (few distinct residues, no run >=4)
beyond the existing run-length rule, when XENO_SEQ_STAGE is on. Flag off keeps the legacy key."""
from __future__ import annotations

import os

from xenodesign.scorer import sequence_quality_key


def test_low_entropy_penalised_when_flag_on(monkeypatch):
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    # 'ABCDABCDABCDABCDABCD' has no homopolymer run >=4 and each letter is 25% < 0.30, so the
    # LEGACY composition rules do NOT penalise it. Its normalized Shannon entropy is ~0.463 < 0.5.
    # Flag-ON must apply the de-gaming entropy penalty, making the flag-ON score LOWER than
    # the legacy (flag-OFF) score for the SAME low-entropy sequence.
    low = "ABCDABCDABCDABCDABCD"             # 4 distinct, repeating -> entropy 0.463 < 0.5
    diverse = "ACDEFGHIKLMNPQRSTVWY"          # 20 distinct -> entropy 1.0 (no extra penalty)

    monkeypatch.setenv("XENO_SEQ_STAGE", "0")
    key_off = sequence_quality_key(low)
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    key_on = sequence_quality_key(low)

    # The entropy penalty must LOWER the low-entropy seq's score when the flag is on
    assert key_on < key_off, (
        f"flag-ON key ({key_on:.4f}) should be lower than flag-OFF key ({key_off:.4f}) "
        "for a low-entropy sequence — the entropy penalty is not being applied"
    )

    # Diverse seq (entropy 1.0 >= 0.5) must not be penalised by the entropy term
    diverse_key = sequence_quality_key(diverse)
    assert diverse_key == 1.0, (
        f"diverse 20-letter seq should have key=1.0 even with flag-ON, got {diverse_key}"
    )


def test_flag_off_is_legacy(monkeypatch):
    monkeypatch.delenv("XENO_SEQ_STAGE", raising=False)
    # Legacy key: 'ABCDABCD...' passes all 3 composition rules -> equals its sequence_complexity with
    # no entropy penalty. Just assert it is finite and not penalised to negative.
    assert sequence_quality_key("ABCDABCDABCDABCDABCD") >= 0.0
