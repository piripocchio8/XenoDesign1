"""build_loop_fn assembles the four primitives into a loop sequence_update_fn(prediction)->str,
reusing SequenceUpdater + MultiCandidate(sequence_quality_key) for de-gaming. CPU fake backend."""
from __future__ import annotations

import numpy as np

from xenodesign.seq_stage import FrozenPosition, SequenceUpdate


class _Wrapper:
    last_out_dir = "out"


def _fake_extract(prediction):
    # The stage's extract seam returns the four SequenceUpdater.update inputs + the prev L seq.
    return {
        "design_backbone": np.zeros((4, 4, 3)),
        "context_coords": np.zeros((0, 3)),
        "context_elements": [],
        "prev_l_seq": "ACDE",
    }


def _fake_backend(design_backbone, context_coords, context_elements,
                  fixed_mask, temperature, num_seqs, known_seq=None):
    n = np.asarray(design_backbone).shape[0]
    return [(known_seq or "A" * n)[:n].ljust(n, "A") for _ in range(num_seqs)]


def test_build_loop_fn_emits_one_letter_for_all_d():
    stage = SequenceUpdate(num_seqs=4, design_fn=_fake_backend)
    fn = stage.build_loop_fn(extract_fn=_fake_extract, chirality_pattern=None)
    out = fn(prediction=object())
    assert isinstance(out, str) and len(out) == 4


def test_build_loop_fn_emits_d_fasta_when_pattern_present():
    stage = SequenceUpdate(num_seqs=4, design_fn=_fake_backend,
                           frozen={FrozenPosition(position0=1, identity="H", chirality="L")})
    pattern = {0: "D", 1: "L", 2: "D", 3: "D"}
    fn = stage.build_loop_fn(extract_fn=_fake_extract, chirality_pattern=pattern)
    out = fn(prediction=object())
    # L coordinator at pos 1 stays bare; D positions parenthesized.
    assert "(" in out and out.count("(") == 3
