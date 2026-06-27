"""build_loop_fn assembles the four primitives into a loop sequence_update_fn(prediction)->str,
reusing SequenceUpdater + MultiCandidate(sequence_quality_key) for de-gaming. CPU fake backend."""
from __future__ import annotations

import numpy as np

from xenodesign.seq_stage import FrozenPosition, SequenceUpdate


def _fake_extract(prediction):
    # The stage's extract seam returns the four SequenceUpdater.update inputs + the prev L seq.
    # prev_l_seq is deliberately non-Ala ("ACDE") so the test can verify the real sequence
    # propagates through build_known_seq → design_fn → output (not the all-Ala starvation path).
    return {
        "design_backbone": np.zeros((4, 4, 3)),
        "context_coords": np.zeros((0, 3)),
        "context_elements": [],
        "prev_l_seq": "ACDE",
    }


def _fake_backend(design_backbone, context_coords, context_elements,
                  fixed_mask, temperature, num_seqs, known_seq=None):
    # Echo the known_seq so the test can assert the real sequence reaches the backend.
    n = np.asarray(design_backbone).shape[0]
    return [(known_seq or "A" * n)[:n].ljust(n, "A") for _ in range(num_seqs)]


def test_build_loop_fn_emits_one_letter_for_all_d():
    """build_loop_fn must propagate the real prev_l_seq (not emit all-Ala starvation output).

    The fake backend echoes its ``known_seq`` argument, which build_known_seq derives from
    prev_l_seq="ACDE".  ensure_canonical_anchor then replaces the last position (E→G) to
    guarantee a Chai-tokenizable residue, so the expected output is "ACDG".

    Regression guard: if the fn reverted to feeding all-Ala to the backend the echo would
    return "AAAG" (all-Ala + Gly anchor), not "ACDG", and the assertion below would fail.
    """
    stage = SequenceUpdate(num_seqs=4, design_fn=_fake_backend)
    fn = stage.build_loop_fn(extract_fn=_fake_extract, chirality_pattern=None)
    out = fn(prediction=object())
    # Shape contract.
    assert isinstance(out, str) and len(out) == 4
    # Real-sequence contract: positions 0–2 must echo "ACD" from prev_l_seq, not "AAA".
    # (Position 3 becomes G via ensure_canonical_anchor — this is correct, not a regression.)
    assert out == "ACDG", (
        f"Expected 'ACDG' (real prev_l_seq round-trip + Gly anchor) but got {out!r}; "
        "this indicates build_known_seq is not feeding the real sequence to the backend"
    )
    assert out != "AAAG", "Output must not be all-Ala+anchor (starvation regression guard)"


def test_build_loop_fn_emits_d_fasta_when_pattern_present():
    stage = SequenceUpdate(num_seqs=4, design_fn=_fake_backend,
                           frozen={FrozenPosition(position0=1, identity="H", chirality="L")})
    pattern = {0: "D", 1: "L", 2: "D", 3: "D"}
    fn = stage.build_loop_fn(extract_fn=_fake_extract, chirality_pattern=pattern)
    out = fn(prediction=object())
    # L coordinator at pos 1 stays bare; D positions parenthesized.
    assert "(" in out and out.count("(") == 3
