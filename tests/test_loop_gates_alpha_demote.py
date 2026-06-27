"""S3a.3a: alpha_demote_gated_accept rejects an over-helical non_alpha candidate (knottins are
anti-alpha) and accepts a low-helix one. Mirrors periodicity_gated_accept's accept_fn shape."""
from __future__ import annotations

from xenodesign.loop import LoopState, LoopStep, alpha_demote_gated_accept


class _Pred:
    def __init__(self, helix):
        self.helix_fraction = helix
        self.iptm = 0.5


def _step(helix):
    return LoopStep(state=LoopState(d_fasta="", coords=None), prediction=_Pred(helix), score=0.0)


def test_alpha_demote_rejects_helical():
    gate = alpha_demote_gated_accept(max_helix_frac=0.5)
    assert gate(_step(0.9), _step(0.1)) is False     # too helical for a knottin -> reject


def test_alpha_demote_accepts_low_helix():
    gate = alpha_demote_gated_accept(max_helix_frac=0.5)
    assert gate(_step(0.2), _step(0.1)) is True


def test_alpha_demote_missing_helix_accepts():
    # An unreadable helix (None / no attr) never silently kills a trajectory -> accept.
    class _NoHelix:
        iptm = 0.5
    step = LoopStep(state=LoopState(d_fasta="", coords=None), prediction=_NoHelix(), score=0.0)
    assert alpha_demote_gated_accept(max_helix_frac=0.5)(step, step) is True
