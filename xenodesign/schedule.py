# xenodesign/schedule.py
"""AnnealSchedule — per-iteration resolution of the forward loop's exploration knobs
(lane B_loop_control of docs/boltzdesign-halludesign-feature-map.md).

Forward-only re-cast of HalluDesign's ref_time_steps_anneal + design_epoch_begin and
BoltzDesign's soft->hard (explore->exploit) staged schedule, unified into ONE object so the
loop has a single source of per-iteration ref_time_steps, num_seqs, and optional MPNN
sampling temperature.

DEFAULT-OFF contract: with only base_ref_time_steps set (no anneal_start, no
design_epoch_begin, no temp_start), every resolver returns the constant base value for every
iteration — reproducing the existing constant-ref_time_steps loop byte-for-byte. The loop
treats schedule=None as 'use the scalar args' (see loop.py), so a None schedule and a
constant AnnealSchedule are equivalent.

Pure Python; no torch / chai / GPU. CPU-unit-tested in tests/test_schedule.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _anneal_progress(i: int, iterations: int, anneal_frac: float) -> float:
    """Fraction in [0, 1] of the way through the anneal WINDOW at iteration i.

    The window is the first anneal_frac of the run (>= 1 iteration). 0.0 at the first
    iteration, 1.0 at/after the end of the window. Safe for iterations <= 1.
    """
    if iterations <= 1:
        return 0.0
    window = max(1, int(iterations * anneal_frac))
    if window <= 1:
        return 1.0 if i >= 1 else 0.0
    return min(1.0, i / window)


def _lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation start->end for t in [0, 1]."""
    return start + (end - start) * t


@dataclass
class AnnealSchedule:
    """Per-iteration schedule for the forward HalluDesign loop.

    Parameters
    ----------
    base_ref_time_steps:
        The floor / steady-state diffusion truncation depth (the value the existing loop uses
        as its constant ref_time_steps, e.g. 50). Always the value returned once the anneal
        window has elapsed, and the value returned for EVERY iter when anneal_start is None.
    anneal_start:
        If set (e.g. 150), ref_time_steps linearly anneals anneal_start -> base over the
        anneal window, then holds at base. If None (default), ref_time_steps is constant
        base (current behaviour).
    anneal_frac:
        Fraction of the run that forms the anneal window (default 1/3, per spec 'first ~1/3').
    design_epoch_begin:
        Warm-up: force num_seqs == 1 for iterations i < design_epoch_begin (single-seq
        warm-up before multi-candidate design). Default 0 = no warm-up (full num_seqs always).
    temp_start:
        If set, MPNN sampling temperature anneals temp_start -> base_temperature over the
        anneal window. If None (default), temperature is the constant base everywhere.
    """

    base_ref_time_steps: int
    anneal_start: Optional[int] = None
    anneal_frac: float = 1.0 / 3.0
    design_epoch_begin: int = 0
    temp_start: Optional[float] = None

    def ref_time_steps(self, i: int, iterations: int) -> int:
        """Resolve ref_time_steps for iteration i (0-based) of an iterations-iter run."""
        if self.anneal_start is None:
            return int(self.base_ref_time_steps)
        t = _anneal_progress(i, iterations, self.anneal_frac)
        return int(round(_lerp(float(self.anneal_start), float(self.base_ref_time_steps), t)))

    def num_seqs(self, i: int, base_num_seqs: int) -> int:
        """Resolve num_seqs for iteration i: 1 during warm-up, else base_num_seqs."""
        if i < self.design_epoch_begin:
            return 1
        return int(base_num_seqs)

    def mpnn_temperature(self, i: int, iterations: int, base_temperature: float) -> float:
        """Resolve MPNN sampling temperature for iteration i (constant base if no temp_start)."""
        if self.temp_start is None:
            return float(base_temperature)
        t = _anneal_progress(i, iterations, self.anneal_frac)
        return float(_lerp(float(self.temp_start), float(base_temperature), t))
