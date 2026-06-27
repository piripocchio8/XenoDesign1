"""S3a.3b: metalhawk_gated_accept vetoes a step whose metal-coordination geometry FAILS MetalHawk
(ok=True, passed=False), accepts a clean one (passed=True), and accepts a pass-through (ok=False) so
a gate that cannot run never vetoes a design (MetalHawk best-effort contract)."""
from __future__ import annotations

from xenodesign.eval.metal_geometry_gate import GateResult
from xenodesign.loop import LoopState, LoopStep, metalhawk_gated_accept


def _step():
    return LoopStep(state=LoopState(d_fasta="", coords=None), prediction=object(), score=0.0)


def test_metalhawk_rejects_failing_geometry():
    gate = metalhawk_gated_accept(lambda step: GateResult(geometry="LIN", perplexity=4.0,
                                                          threshold=1.5, passed=False, ok=True))
    assert gate(_step(), _step()) is False


def test_metalhawk_accepts_clean_geometry():
    gate = metalhawk_gated_accept(lambda step: GateResult(geometry="TET", perplexity=1.1,
                                                          threshold=1.5, passed=True, ok=True))
    assert gate(_step(), _step()) is True


def test_metalhawk_passthrough_accepts():
    gate = metalhawk_gated_accept(lambda step: GateResult(passed=True, ok=False, error="no dir"))
    assert gate(_step(), _step()) is True
