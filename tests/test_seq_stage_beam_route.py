"""S2.1: beam.expand_state threads a real known_seq + stage-encode when injected (flag-on path);
with no injection it is byte-identical to the legacy all-Ala expansion.
S2.1.2: beam_search forwards known_seq_fn/encode_fn to expand_state."""
from __future__ import annotations

import numpy as np

from xenodesign.beam import BeamState, CostAccount, beam_search, expand_state


def _extract(parent):
    return {"design_backbone": np.zeros((4, 4, 3)),
            "design_codes": ["DAL"] * 4,
            "context_coords": np.zeros((0, 3)), "context_elements": []}


def _capturing_design_fn(seen):
    def design_fn(design_backbone, context_coords, context_elements,
                  fixed_mask, temperature, num_seqs, known_seq=None):
        seen["known_seq"] = known_seq
        n = np.asarray(design_backbone).shape[0]
        # Echo known_seq so the d_fasta reflects the threaded real sequence (else all-Ala).
        return [(known_seq or "A" * n)[:n].ljust(n, "A")]
    return design_fn


def test_expand_state_legacy_is_all_ala(monkeypatch):
    """No known_seq_fn injected: design_fn is called with known_seq=None (legacy beam bug)."""
    seen = {}
    parent = BeamState(d_fasta="", coords=np.zeros((4, 3)), l_seq="MKWV", id=1)
    kids = expand_state(parent, _capturing_design_fn(seen), _extract)
    assert seen["known_seq"] is None                 # legacy: no real context threaded
    assert kids and kids[0].d_fasta == "(DAL)(DAL)(DAL)(DAL)"   # all-Ala -> all-D Ala


def test_expand_state_threads_known_seq_and_encode(monkeypatch):
    """known_seq_fn + encode_fn injected: design_fn sees the parent's real l_seq; the child's
    d_fasta is produced by encode_fn (here: identity passthrough, to make the assertion exact)."""
    seen = {}
    parent = BeamState(d_fasta="", coords=np.zeros((4, 3)), l_seq="MKWV", id=1)
    kids = expand_state(parent, _capturing_design_fn(seen), _extract,
                        known_seq_fn=lambda p: p.l_seq,
                        encode_fn=lambda l_seq: f"<{l_seq}>")
    assert seen["known_seq"] == "MKWV"               # REAL evolving seq threaded (invariant #1)
    assert kids[0].l_seq == "MKWV"                   # echoed by the design_fn
    assert kids[0].d_fasta == "<MKWV>"               # emitted via the injected stage encode


def test_beam_search_forwards_known_seq_fn(monkeypatch):
    """beam_search threads known_seq_fn/encode_fn down to expand_state."""
    seen = {}

    def predict_fn(state, ref_time_steps, out_dir):
        from tests.test_characterization_goldens import _FakePred
        return _FakePred()

    from xenodesign.judges.panel import JudgePanel, RefereeScore
    panel = JudgePanel()
    referee_fn = lambda c: RefereeScore(chirality_violation=0.0, iptm=0.5)
    seed = BeamState(d_fasta="", coords=np.zeros((4, 3)), l_seq="MKWV", cycle=0, id=99)
    pool, cost = beam_search(
        seed, design_fn=_capturing_design_fn(seen), predict_fn=predict_fn,
        extract_fn=_extract, referee_fn=referee_fn, panel=panel,
        beam_width=1, children_per_branch=1, cycles=1, cost=CostAccount(),
        known_seq_fn=lambda p: p.l_seq, encode_fn=lambda s: f"<{s}>",
    )
    assert seen.get("known_seq") == "MKWV"           # forwarded all the way to expand_state
