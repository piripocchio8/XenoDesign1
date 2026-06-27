"""S2.1 intentional-improvement pin: the beam path routed through SequenceUpdate (XENO_SEQ_STAGE=1)
carries the REAL evolving sequence into beam expansion — NOT the legacy all-Ala starvation. This
pins the invariant-#1 fix for the beam search (spec §1: the beam.py:162 all-Ala known_seq bug)."""
from __future__ import annotations

import numpy as np

from xenodesign.beam import BeamState, expand_state


def test_routed_beam_expansion_uses_real_seq_not_all_ala():
    seen = {}

    def design_fn(design_backbone, context_coords, context_elements,
                  fixed_mask, temperature, num_seqs, known_seq=None):
        seen["known_seq"] = known_seq
        n = np.asarray(design_backbone).shape[0]
        return [(known_seq or "A" * n)[:n].ljust(n, "A")]

    parent = BeamState(d_fasta="", coords=np.zeros((5, 3)), l_seq="MKWVT", id=1)
    kids = expand_state(parent, design_fn,
                        extract_fn=lambda p: {"design_backbone": np.zeros((5, 4, 3)),
                                              "design_codes": ["DAL"] * 5,
                                              "context_coords": np.zeros((0, 3)),
                                              "context_elements": []},
                        known_seq_fn=lambda p: p.l_seq,
                        encode_fn=lambda s: s)
    assert seen["known_seq"] == "MKWVT"              # the real evolving seq, not None/all-Ala
    assert kids[0].l_seq == "MKWVT" and "A" * 5 != kids[0].l_seq
