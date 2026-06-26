"""CPU tests for xenodesign.eval.watchdog (#25).

No GPU, no network, no pynvml, no psutil required.  All hardware-dependent paths
are exercised through pure helpers (parse_nvidia_smi_csv, evaluate_thresholds) or
via the ``_sampler_fn`` injection point on ResourceWatchdog.

Run with:
    pytest tests/test_watchdog.py -m "not gpu and not network" -q
"""
from __future__ import annotations

import itertools
import threading
import time
from typing import Any

import pytest

from xenodesign.eval.watchdog import (
    ResourceWatchdog,
    evaluate_thresholds,
    parse_nvidia_smi_csv,
)

# ---------------------------------------------------------------------------
# 1.  parse_nvidia_smi_csv — pure CSV parser
# ---------------------------------------------------------------------------

class TestParseNvidiaSmiCsv:
    """Tests for the pure nvidia-smi CSV parser."""

    SAMPLE_TWO_GPUS = "0, 1024, 20480, 45\n1, 512, 20480, 12\n"

    def test_two_gpu_count(self):
        rows = parse_nvidia_smi_csv(self.SAMPLE_TWO_GPUS)
        assert len(rows) == 2

    def test_index_parsed(self):
        rows = parse_nvidia_smi_csv(self.SAMPLE_TWO_GPUS)
        assert rows[0]["index"] == 0
        assert rows[1]["index"] == 1

    def test_mem_used_mb(self):
        rows = parse_nvidia_smi_csv(self.SAMPLE_TWO_GPUS)
        assert rows[0]["mem_used_mb"] == 1024.0
        assert rows[1]["mem_used_mb"] == 512.0

    def test_mem_total_mb(self):
        rows = parse_nvidia_smi_csv(self.SAMPLE_TWO_GPUS)
        assert rows[0]["mem_total_mb"] == 20480.0

    def test_util_pct(self):
        rows = parse_nvidia_smi_csv(self.SAMPLE_TWO_GPUS)
        assert rows[0]["util_pct"] == 45.0
        assert rows[1]["util_pct"] == 12.0

    def test_no_util_column_gives_none(self):
        text = "0, 1024, 20480\n"
        rows = parse_nvidia_smi_csv(text)
        assert len(rows) == 1
        assert rows[0]["util_pct"] is None

    def test_empty_string_returns_empty_list(self):
        assert parse_nvidia_smi_csv("") == []

    def test_comment_lines_skipped(self):
        text = "# header comment\n0, 100, 20480, 30\n"
        rows = parse_nvidia_smi_csv(text)
        assert len(rows) == 1

    def test_malformed_line_skipped(self):
        text = "not_a_number, garbage\n0, 500, 20480, 55\n"
        rows = parse_nvidia_smi_csv(text)
        assert len(rows) == 1
        assert rows[0]["index"] == 0

    def test_whitespace_around_values(self):
        text = " 0 ,  2048 ,  20480 ,  70 \n"
        rows = parse_nvidia_smi_csv(text)
        assert rows[0]["mem_used_mb"] == 2048.0

    def test_single_gpu_high_vram(self):
        text = "0, 19800, 20480, 99\n"
        rows = parse_nvidia_smi_csv(text)
        assert rows[0]["mem_used_mb"] / rows[0]["mem_total_mb"] > 0.96


# ---------------------------------------------------------------------------
# 2.  evaluate_thresholds — pure threshold checker
# ---------------------------------------------------------------------------

def _make_sample(
    gpu_fracs: list[float] | None = None,
    free_disk_gb: float | None = None,
    mem_total_mb: float = 20480.0,
) -> dict:
    """Build a synthetic resource sample for threshold tests."""
    gpus = None
    if gpu_fracs is not None:
        gpus = [
            {
                "index": i,
                "mem_used_mb": frac * mem_total_mb,
                "mem_total_mb": mem_total_mb,
                "util_pct": None,
            }
            for i, frac in enumerate(gpu_fracs)
        ]
    return {
        "gpus": gpus,
        "free_disk_gb": free_disk_gb,
        "ram_used_gb": None,
    }


_DEFAULT_CFG = {"vram_frac_warn": 0.92, "min_free_disk_gb": 5.0}


class TestEvaluateThresholds:
    """Tests for the pure threshold evaluator."""

    def test_no_breach_returns_empty_list(self):
        sample = _make_sample(gpu_fracs=[0.50, 0.50], free_disk_gb=100.0)
        assert evaluate_thresholds(sample, _DEFAULT_CFG) == []

    def test_vram_breach_one_gpu(self):
        sample = _make_sample(gpu_fracs=[0.95, 0.50])
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        assert len(reasons) == 1
        assert "GPU 0" in reasons[0]

    def test_vram_breach_both_gpus(self):
        sample = _make_sample(gpu_fracs=[0.95, 0.93])
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        gpu_indices = [r.split()[1] for r in reasons]
        assert "0" in gpu_indices
        assert "1" in gpu_indices

    def test_vram_exactly_at_threshold_triggers(self):
        # fraction == warn level should trigger (>=)
        sample = _make_sample(gpu_fracs=[0.92])
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        assert len(reasons) == 1

    def test_vram_just_below_threshold_no_breach(self):
        # 0.919... is just under 0.92 — should NOT trigger
        sample = _make_sample(gpu_fracs=[0.919])
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        assert reasons == []

    def test_disk_breach(self):
        sample = _make_sample(free_disk_gb=2.0)
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        assert len(reasons) == 1
        assert "disk" in reasons[0]

    def test_disk_exactly_at_floor_no_breach(self):
        # free_disk == min_free_disk_gb should NOT trigger (strict <)
        sample = _make_sample(free_disk_gb=5.0)
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        assert reasons == []

    def test_disk_and_vram_both_breach(self):
        sample = _make_sample(gpu_fracs=[0.99], free_disk_gb=1.0)
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        assert len(reasons) == 2

    def test_no_gpus_in_sample(self):
        sample = _make_sample(gpu_fracs=None, free_disk_gb=100.0)
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        assert reasons == []

    def test_custom_thresholds_respected(self):
        cfg = {"vram_frac_warn": 0.70, "min_free_disk_gb": 20.0}
        sample = _make_sample(gpu_fracs=[0.75], free_disk_gb=15.0)
        reasons = evaluate_thresholds(sample, cfg)
        # Both should fire with the tighter custom cfg
        assert len(reasons) == 2

    def test_reason_string_contains_useful_info(self):
        sample = _make_sample(gpu_fracs=[0.95], free_disk_gb=1.0)
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        # VRAM reason should mention percentage and MiB
        vram_reason = next((r for r in reasons if "GPU" in r), None)
        assert vram_reason is not None
        assert "%" in vram_reason
        assert "MiB" in vram_reason
        # Disk reason should mention GiB
        disk_reason = next((r for r in reasons if "disk" in r), None)
        assert disk_reason is not None
        assert "GiB" in disk_reason


# ---------------------------------------------------------------------------
# 3.  ResourceWatchdog — summary aggregation with injected sampler
# ---------------------------------------------------------------------------

def _make_synthetic_samples(
    gpu_fracs_sequence: list[list[float]],
    free_disk_sequence: list[float],
    mem_total_mb: float = 20480.0,
) -> list[dict]:
    """Build a list of synthetic samples for injecting into the watchdog."""
    samples = []
    for sample_idx, (fracs, disk) in enumerate(
        zip(gpu_fracs_sequence, free_disk_sequence)
    ):
        samples.append(
            {
                "gpus": [
                    {
                        "index": gpu_i,
                        "mem_used_mb": f * mem_total_mb,
                        "mem_total_mb": mem_total_mb,
                        "util_pct": None,
                    }
                    for gpu_i, f in enumerate(fracs)
                ],
                "ram_used_gb": 10.0 + sample_idx * 0.5,
                "free_disk_gb": disk,
                "ts": time.time(),
            }
        )
    return samples


class TestResourceWatchdogSummaryAggregation:
    """Tests for summary() aggregation logic using injected samplers."""

    def _make_watchdog_with_samples(self, samples: list[dict]) -> ResourceWatchdog:
        """Create a watchdog that emits *samples* in order, then blocks."""
        it = iter(samples)

        def sampler() -> dict:
            try:
                return next(it)
            except StopIteration:
                # Return a benign sample after the list is exhausted.
                return {"gpus": None, "ram_used_gb": None, "free_disk_gb": None}

        return ResourceWatchdog(
            interval=0.01,
            _sampler_fn=sampler,
        )

    def test_n_samples_matches_emitted(self):
        samples = _make_synthetic_samples([[0.5], [0.6]], [100.0, 90.0])
        w = ResourceWatchdog(interval=0.01, _sampler_fn=iter(samples).__next__)
        w.start()
        time.sleep(0.05)
        w.stop()
        s = w.summary()
        assert s["n_samples"] >= 1  # at least one sample taken

    def test_peak_vram_is_maximum(self):
        # Two samples: GPU0 uses 0.5 then 0.8 of total.  Peak should be 0.8.
        samples = _make_synthetic_samples([[0.5, 0.3], [0.8, 0.2]], [100.0, 100.0])
        w = ResourceWatchdog(
            interval=0.001,
            vram_frac_warn=0.99,  # don't trip
            _sampler_fn=None,
        )
        # Inject samples directly to avoid threading timing issues.
        w._samples = samples
        s = w.summary()
        expected_gpu0 = 0.8 * 20480.0 / 1024.0
        assert abs(s["peak_vram_used_gb"][0] - expected_gpu0) < 0.01

    def test_min_free_disk_is_minimum(self):
        samples = _make_synthetic_samples([[0.5]], [100.0])
        samples += _make_synthetic_samples([[0.5]], [10.0])
        samples += _make_synthetic_samples([[0.5]], [50.0])
        w = ResourceWatchdog(interval=0.001)
        w._samples = samples
        s = w.summary()
        assert abs(s["min_free_disk_gb"] - 10.0) < 0.01

    def test_peak_ram_is_maximum(self):
        samples = [
            {"gpus": None, "ram_used_gb": 20.0, "free_disk_gb": 100.0},
            {"gpus": None, "ram_used_gb": 30.0, "free_disk_gb": 100.0},
            {"gpus": None, "ram_used_gb": 25.0, "free_disk_gb": 100.0},
        ]
        w = ResourceWatchdog(interval=0.001)
        w._samples = samples
        s = w.summary()
        assert abs(s["peak_ram_used_gb"] - 30.0) < 0.01

    def test_summary_empty_samples_returns_nones(self):
        w = ResourceWatchdog(interval=0.001)
        # No start/stop, no samples injected
        s = w.summary()
        assert s["n_samples"] == 0
        assert s["peak_vram_used_gb"] == {}
        assert s["peak_ram_used_gb"] is None
        assert s["min_free_disk_gb"] is None
        assert s["duration_s"] is None
        assert not s["tripped"]
        assert s["reasons"] == []

    def test_duration_recorded_after_stop(self):
        w = ResourceWatchdog(interval=0.05)
        # Simulate manual start/stop with a no-op sampler
        w._sampler_fn = lambda: {"gpus": None, "ram_used_gb": None, "free_disk_gb": None}
        w.start()
        time.sleep(0.1)
        w.stop()
        s = w.summary()
        assert s["duration_s"] is not None
        assert s["duration_s"] >= 0.05  # at least as long as the sleep


# ---------------------------------------------------------------------------
# 4.  Start/stop lifecycle + context manager
# ---------------------------------------------------------------------------

class TestWatchdogLifecycle:
    """Tests for start(), stop(), context manager, and __dunder__ contracts."""

    def _noop_sampler(self) -> dict:
        return {"gpus": None, "ram_used_gb": None, "free_disk_gb": None}

    def test_start_stop_thread_terminates(self):
        w = ResourceWatchdog(interval=0.05, _sampler_fn=self._noop_sampler)
        w.start()
        assert w._thread is not None
        assert w._thread.is_alive()
        w.stop()
        assert not w._thread.is_alive()

    def test_context_manager_starts_and_stops(self):
        with ResourceWatchdog(
            interval=0.05, _sampler_fn=self._noop_sampler
        ) as w:
            assert w._thread is not None and w._thread.is_alive()
        assert not w._thread.is_alive()

    def test_context_manager_returns_self(self):
        with ResourceWatchdog(
            interval=0.05, _sampler_fn=self._noop_sampler
        ) as w:
            assert isinstance(w, ResourceWatchdog)

    def test_at_least_one_sample_after_start(self):
        # The watchdog takes one immediate sample on start.
        w = ResourceWatchdog(interval=1.0, _sampler_fn=self._noop_sampler)
        w.start()
        time.sleep(0.05)
        w.stop()
        assert w.summary()["n_samples"] >= 1

    def test_double_start_does_not_spawn_second_thread(self):
        w = ResourceWatchdog(interval=0.05, _sampler_fn=self._noop_sampler)
        w.start()
        thread1 = w._thread
        w.start()  # second call — should no-op
        thread2 = w._thread
        w.stop()
        assert thread1 is thread2

    def test_stop_without_start_is_safe(self):
        w = ResourceWatchdog(interval=0.05)
        w.stop()  # should not raise
        assert w.summary()["n_samples"] == 0


# ---------------------------------------------------------------------------
# 5.  Tripped flag fires on synthetic threshold breach
# ---------------------------------------------------------------------------

class TestWatchdogTrip:
    """Tests that tripped/reasons are set when thresholds are exceeded."""

    def _high_vram_sampler(self):
        """Return a sample that always exceeds the VRAM threshold."""
        return {
            "gpus": [
                {
                    "index": 0,
                    "mem_used_mb": 19000.0,   # ~92.8% of 20480
                    "mem_total_mb": 20480.0,
                    "util_pct": None,
                }
            ],
            "ram_used_gb": 10.0,
            "free_disk_gb": 100.0,
            "ts": time.time(),
        }

    def _low_disk_sampler(self):
        """Return a sample that always exceeds the disk threshold."""
        return {
            "gpus": None,
            "ram_used_gb": 10.0,
            "free_disk_gb": 0.5,   # below default 5 GiB
            "ts": time.time(),
        }

    def test_vram_breach_trips_watchdog(self):
        w = ResourceWatchdog(
            interval=0.01,
            vram_frac_warn=0.92,
            _sampler_fn=self._high_vram_sampler,
        )
        w.start()
        time.sleep(0.08)
        w.stop()
        assert w.tripped
        assert len(w.reasons) >= 1
        assert any("GPU 0" in r for r in w.reasons)

    def test_disk_breach_trips_watchdog(self):
        w = ResourceWatchdog(
            interval=0.01,
            min_free_disk_gb=5.0,
            _sampler_fn=self._low_disk_sampler,
        )
        w.start()
        time.sleep(0.08)
        w.stop()
        assert w.tripped
        assert any("disk" in r for r in w.reasons)

    def test_on_trip_callback_called(self):
        fired_reasons: list[list[str]] = []

        def callback(reasons):
            fired_reasons.append(reasons)

        w = ResourceWatchdog(
            interval=0.01,
            vram_frac_warn=0.92,
            _sampler_fn=self._high_vram_sampler,
            on_trip=callback,
        )
        w.start()
        time.sleep(0.08)
        w.stop()
        assert len(fired_reasons) >= 1  # callback was invoked
        assert all(isinstance(r, list) for r in fired_reasons)

    def test_on_trip_callback_called_only_once(self):
        """on_trip must fire only on the FIRST breach, not on every sample."""
        call_count: list[int] = [0]

        def callback(reasons):
            call_count[0] += 1

        w = ResourceWatchdog(
            interval=0.01,
            vram_frac_warn=0.92,
            _sampler_fn=self._high_vram_sampler,
            on_trip=callback,
        )
        w.start()
        time.sleep(0.15)  # let several breach samples accumulate
        w.stop()
        assert call_count[0] == 1  # exactly one callback invocation

    def test_no_trip_when_below_thresholds(self):
        def safe_sampler():
            return {
                "gpus": [
                    {
                        "index": 0,
                        "mem_used_mb": 5000.0,
                        "mem_total_mb": 20480.0,
                        "util_pct": None,
                    }
                ],
                "ram_used_gb": 10.0,
                "free_disk_gb": 100.0,
            }

        w = ResourceWatchdog(
            interval=0.01,
            vram_frac_warn=0.92,
            min_free_disk_gb=5.0,
            _sampler_fn=safe_sampler,
        )
        w.start()
        time.sleep(0.08)
        w.stop()
        assert not w.tripped
        assert w.reasons == []

    def test_summary_tripped_and_reasons_match(self):
        w = ResourceWatchdog(
            interval=0.01,
            min_free_disk_gb=5.0,
            _sampler_fn=self._low_disk_sampler,
        )
        w.start()
        time.sleep(0.08)
        w.stop()
        s = w.summary()
        assert s["tripped"] == w.tripped
        assert s["reasons"] == w.reasons


# ---------------------------------------------------------------------------
# 6.  Graceful degradation: import works without pynvml / psutil
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Confirm the module imports and runs pure logic without optional deps."""

    def test_module_importable(self):
        """Simply importing the module must not raise even without pynvml."""
        import xenodesign.eval.watchdog  # noqa: F401 — import check

    def test_evaluate_thresholds_no_gpu_data(self):
        """evaluate_thresholds handles None gpus without crashing."""
        sample = {"gpus": None, "free_disk_gb": 100.0}
        reasons = evaluate_thresholds(sample, _DEFAULT_CFG)
        assert reasons == []

    def test_watchdog_runs_with_injected_sampler(self):
        """Watchdog lifecycle works with a pure sampler, no hardware needed."""
        safe_sample = {
            "gpus": None,
            "ram_used_gb": 8.0,
            "free_disk_gb": 50.0,
        }
        with ResourceWatchdog(
            interval=0.02,
            _sampler_fn=lambda: safe_sample,
        ) as w:
            time.sleep(0.06)
        s = w.summary()
        assert s["n_samples"] >= 1
        assert not s["tripped"]
