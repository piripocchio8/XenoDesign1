# tests/test_schedule.py
"""CPU tests for xenodesign.schedule.AnnealSchedule (lane B, default-OFF).

The null/constant schedule MUST reproduce the existing constant-ref_time_steps behaviour
byte-for-byte; the annealed schedule interpolates 150->50 over the first fraction of iters and
the warm-up gate forces num_seqs=1 for the first design_epoch_begin iters."""
import pytest
from xenodesign.schedule import AnnealSchedule


def test_constant_schedule_reproduces_current_behaviour():
    s = AnnealSchedule(base_ref_time_steps=50)
    for i in range(30):
        assert s.ref_time_steps(i, iterations=30) == 50
        assert s.num_seqs(i, base_num_seqs=8) == 8
        assert s.mpnn_temperature(i, iterations=30, base_temperature=0.1) == 0.1


def test_ref_time_steps_anneal_endpoints_and_midpoint():
    s = AnnealSchedule(base_ref_time_steps=50, anneal_start=150, anneal_frac=1.0 / 3.0)
    n = 30
    assert s.ref_time_steps(0, iterations=n) == 150
    assert s.ref_time_steps(n - 1, iterations=n) == 50
    assert s.ref_time_steps(10, iterations=n) == 50
    assert s.ref_time_steps(5, iterations=n) == 100
    seq = [s.ref_time_steps(i, iterations=n) for i in range(n)]
    assert all(seq[i] >= seq[i + 1] for i in range(n - 1))
    assert all(50 <= v <= 150 for v in seq)


def test_design_epoch_begin_warmup_gate():
    s = AnnealSchedule(base_ref_time_steps=50, design_epoch_begin=3)
    assert s.num_seqs(0, base_num_seqs=8) == 1
    assert s.num_seqs(2, base_num_seqs=8) == 1
    assert s.num_seqs(3, base_num_seqs=8) == 8
    assert s.num_seqs(10, base_num_seqs=8) == 8


def test_mpnn_temperature_anneal():
    s = AnnealSchedule(base_ref_time_steps=50, temp_start=0.3, anneal_frac=1.0 / 3.0)
    n = 30
    assert s.mpnn_temperature(0, iterations=n, base_temperature=0.1) == pytest.approx(0.3)
    assert s.mpnn_temperature(n - 1, iterations=n, base_temperature=0.1) == pytest.approx(0.1)
    s2 = AnnealSchedule(base_ref_time_steps=50)
    assert s2.mpnn_temperature(0, iterations=n, base_temperature=0.1) == pytest.approx(0.1)


def test_single_iteration_is_safe():
    s = AnnealSchedule(base_ref_time_steps=50, anneal_start=150)
    assert s.ref_time_steps(0, iterations=1) == 150
