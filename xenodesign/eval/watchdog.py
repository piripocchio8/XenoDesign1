"""Resource watchdog for long GPU campaigns (#25).

Samples GPU VRAM, host RAM, and free disk at a configurable interval on a
background thread.  Issues ``logging.WARNING`` messages and sets a ``.tripped``
flag when configurable thresholds are exceeded.  Does **not** kill or restart
anything — advisory + flag only, matching the project's "warn, don't fail
silently" idiom.

Dependencies are **lazily imported and gracefully degrade**:
- ``pynvml``  — preferred GPU backend (NVIDIA Management Library Python bindings).
- ``subprocess``/ ``nvidia-smi`` — fallback GPU backend when pynvml is absent.
- ``psutil``  — preferred host-memory + disk backend.
- ``os.statvfs`` / ``/proc/meminfo`` — fallback when psutil is absent.

The CPU test suite can import this module and exercise all pure logic without any
of those optional dependencies installed.

Quick-start for campaign drivers
---------------------------------
Wire the watchdog around a campaign with a context manager::

    from xenodesign.eval.watchdog import ResourceWatchdog

    cfg = dict(
        vram_frac_warn=0.92,      # warn when any GPU uses ≥ 92 % of VRAM
        min_free_disk_gb=5.0,     # warn when free disk drops below 5 GB
        disk_path=".",            # path to monitor (default "."); /scratch is opt-in on HPC
        interval=30.0,            # sampling interval in seconds
    )

    def _on_trip(reasons):
        # e.g. pause job submission
        print("WATCHDOG TRIPPED:", reasons)

    with ResourceWatchdog(**cfg, on_trip=_on_trip) as w:
        run_campaign(...)          # your GPU campaign loop

    print(w.summary())

``summary()`` returns a dict with peak_vram_used_gb (per GPU), peak_ram_used_gb,
min_free_disk_gb_observed, n_samples, duration_s, tripped, reasons.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helper: parse nvidia-smi CSV output
# ---------------------------------------------------------------------------

def parse_nvidia_smi_csv(text: str) -> list[dict]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` output.

    Each output line is expected to be comma-separated values in this order:
    ``index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]``
    (the ``[unit]`` annotations are stripped by ``nounits``).

    Returns a list of dicts, one per GPU line, with keys:
    ``index`` (int), ``mem_used_mb`` (float), ``mem_total_mb`` (float),
    ``util_pct`` (float | None).

    Lines that cannot be parsed are skipped.  An empty / all-comment input
    returns an empty list.

    This is a **pure** function with no external calls — fully unit-testable.

    Example
    -------
    >>> rows = parse_nvidia_smi_csv("0, 1024, 20480, 45\\n1, 512, 20480, 12")
    >>> rows[0]["mem_used_mb"]
    1024.0
    """
    results: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
            util: Optional[float] = float(parts[3]) if len(parts) >= 4 else None
            results.append(
                {
                    "index": index,
                    "mem_used_mb": mem_used,
                    "mem_total_mb": mem_total,
                    "util_pct": util,
                }
            )
        except (ValueError, IndexError):
            log.debug("parse_nvidia_smi_csv: skipping malformed line %r", line)
    return results


# ---------------------------------------------------------------------------
# Pure helper: threshold evaluation
# ---------------------------------------------------------------------------

def evaluate_thresholds(sample: dict, cfg: dict) -> list[str]:
    """Return a list of human-readable breach reason strings for *sample*.

    If the sample is within all thresholds the list is empty.

    Parameters
    ----------
    sample:
        A single resource snapshot dict as produced by :class:`ResourceWatchdog`'s
        internal sampler.  Expected keys (all optional — missing keys are
        silently skipped):

        - ``gpus``: list of dicts with ``mem_used_mb`` / ``mem_total_mb``.
        - ``ram_used_gb``: float, host RAM used.
        - ``free_disk_gb``: float, free disk on the monitored path.

    cfg:
        Configuration dict.  Recognised keys:

        - ``vram_frac_warn`` (float, default 0.92): fraction of total VRAM that
          triggers a warning.
        - ``min_free_disk_gb`` (float, default 5.0): free-disk floor in GiB.

    Returns
    -------
    list[str]
        One entry per threshold breached.  The strings are suitable for logging
        and for the watchdog's ``reasons`` list.

    This is a **pure** function — no I/O, no side-effects.
    """
    vram_warn = float(cfg.get("vram_frac_warn", 0.92))
    disk_floor = float(cfg.get("min_free_disk_gb", 5.0))
    reasons: list[str] = []

    # GPU VRAM thresholds
    for gpu in sample.get("gpus") or []:
        total = gpu.get("mem_total_mb")
        used = gpu.get("mem_used_mb")
        if total and used is not None and total > 0:
            frac = used / total
            if frac >= vram_warn:
                idx = gpu.get("index", "?")
                reasons.append(
                    f"GPU {idx} VRAM {frac*100:.1f}% >= {vram_warn*100:.1f}% "
                    f"({used:.0f}/{total:.0f} MiB)"
                )

    # Free disk threshold
    free_disk = sample.get("free_disk_gb")
    if free_disk is not None and free_disk < disk_floor:
        reasons.append(
            f"free disk {free_disk:.2f} GiB < {disk_floor:.2f} GiB"
        )

    return reasons


# ---------------------------------------------------------------------------
# Hardware samplers (lazy, gracefully degrading)
# ---------------------------------------------------------------------------

def _sample_gpu_pynvml() -> Optional[list[dict]]:
    """Sample GPU info via pynvml.  Returns None if pynvml is unavailable."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        results = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except Exception:
                util = None
            results.append(
                {
                    "index": i,
                    "mem_used_mb": mem.used / 1024**2,
                    "mem_total_mb": mem.total / 1024**2,
                    "util_pct": float(util) if util is not None else None,
                }
            )
        return results
    except Exception:
        return None


def _sample_gpu_smi() -> Optional[list[dict]]:
    """Sample GPU info via nvidia-smi subprocess.  Returns None on failure."""
    import subprocess  # stdlib

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return parse_nvidia_smi_csv(result.stdout) or None
    except Exception:
        return None


def _sample_gpu(use_pynvml: bool = True) -> Optional[list[dict]]:
    """Sample GPU info, trying pynvml first then nvidia-smi."""
    if use_pynvml:
        result = _sample_gpu_pynvml()
        if result is not None:
            return result
    return _sample_gpu_smi()


def _sample_host_psutil(disk_path: str = ".") -> dict:
    """Sample host RAM and disk via psutil."""
    import psutil  # type: ignore

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(disk_path)
    return {
        "ram_used_gb": vm.used / 1024**3,
        "ram_total_gb": vm.total / 1024**3,
        "free_disk_gb": disk.free / 1024**3,
    }


def _sample_host_fallback(disk_path: str = ".") -> dict:
    """Sample host RAM (/proc/meminfo) and disk (os.statvfs) without psutil."""
    import os

    # Disk via statvfs
    try:
        stat = os.statvfs(disk_path)
        free_disk_gb: Optional[float] = (stat.f_bavail * stat.f_frsize) / 1024**3
    except Exception:
        free_disk_gb = None

    # RAM via /proc/meminfo
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    try:
        meminfo: dict[str, float] = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        meminfo[key] = float(parts[1])  # kB
                    except ValueError:
                        pass
        if "MemTotal" in meminfo and "MemAvailable" in meminfo:
            ram_total_gb = meminfo["MemTotal"] / 1024**2
            used_kb = meminfo["MemTotal"] - meminfo["MemAvailable"]
            ram_used_gb = used_kb / 1024**2
    except Exception:
        pass

    return {
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "free_disk_gb": free_disk_gb,
    }


def _sample_host(disk_path: str = ".") -> dict:
    """Sample host RAM and disk, preferring psutil, falling back to os/proc."""
    try:
        import psutil  # noqa: F401 — presence check
        return _sample_host_psutil(disk_path)
    except ImportError:
        return _sample_host_fallback(disk_path)


# ---------------------------------------------------------------------------
# ResourceWatchdog
# ---------------------------------------------------------------------------

class ResourceWatchdog:
    """Background resource monitor for long GPU campaigns.

    Samples GPU VRAM, host RAM, and free disk on a background thread.  When
    configurable thresholds are breached it logs a ``WARNING``, sets
    ``.tripped = True``, appends to ``.reasons``, and optionally calls
    ``on_trip(reasons)``.

    Parameters
    ----------
    interval:
        Sampling interval in seconds (default 30).
    vram_frac_warn:
        Fraction of total VRAM that triggers a VRAM warning (default 0.92).
    min_free_disk_gb:
        Free-disk floor in GiB; warn when free disk drops below this (default 5).
    disk_path:
        Filesystem path to monitor for free disk (default ``"."``).
    on_trip:
        Optional callable ``(reasons: list[str]) -> None`` invoked once on the
        first breach.  Called from the background thread.
    _sampler_fn:
        Internal hook for testing — replace the default hardware sampler with a
        callable ``() -> dict`` that returns a resource snapshot.  The dict must
        have the same shape as a real sample (keys: ``gpus``, ``ram_used_gb``,
        ``free_disk_gb``).

    Usage (context manager)
    -----------------------
    >>> with ResourceWatchdog(interval=30) as w:
    ...     run_campaign()
    >>> print(w.summary())

    Usage (manual)
    --------------
    >>> w = ResourceWatchdog(interval=30)
    >>> w.start()
    >>> run_campaign()
    >>> w.stop()
    >>> print(w.summary())
    """

    def __init__(
        self,
        *,
        interval: float = 30.0,
        vram_frac_warn: float = 0.92,
        min_free_disk_gb: float = 5.0,
        disk_path: str = ".",
        on_trip: Optional[Callable[[list[str]], Any]] = None,
        _sampler_fn: Optional[Callable[[], dict]] = None,
    ) -> None:
        self.interval = interval
        self._cfg: dict = {
            "vram_frac_warn": vram_frac_warn,
            "min_free_disk_gb": min_free_disk_gb,
        }
        self._disk_path = disk_path
        self._on_trip = on_trip
        self._sampler_fn = _sampler_fn  # None → use real hardware

        # Accumulated state
        self._samples: list[dict] = []
        self.tripped: bool = False
        self.reasons: list[str] = []

        # Threading
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_start: Optional[float] = None
        self._t_stop: Optional[float] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ResourceWatchdog":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sampling thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("ResourceWatchdog.start() called while already running")
            return
        self._stop_event.clear()
        self._t_start = time.monotonic()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="ResourceWatchdog",
            daemon=True,
        )
        self._thread.start()
        log.debug("ResourceWatchdog started (interval=%.1fs)", self.interval)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 5)
        self._t_stop = time.monotonic()
        log.debug(
            "ResourceWatchdog stopped; %d samples collected", len(self._samples)
        )

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Thread body: sample repeatedly until stop() is called."""
        # Take one immediate sample so summary() is non-empty even for short runs.
        self._take_sample()
        while not self._stop_event.wait(timeout=self.interval):
            self._take_sample()

    def _take_sample(self) -> None:
        """Collect one resource snapshot, check thresholds, store the sample."""
        try:
            sample = self._collect_sample()
            self._samples.append(sample)
            self._check_thresholds(sample)
        except Exception:
            log.exception("ResourceWatchdog: error collecting sample")

    def _collect_sample(self) -> dict:
        """Return one resource snapshot dict."""
        if self._sampler_fn is not None:
            return self._sampler_fn()

        # Real hardware path
        gpus = _sample_gpu()
        host = _sample_host(self._disk_path)
        return {
            "gpus": gpus,
            "ram_used_gb": host.get("ram_used_gb"),
            "ram_total_gb": host.get("ram_total_gb"),
            "free_disk_gb": host.get("free_disk_gb"),
            "ts": time.time(),
        }

    def _check_thresholds(self, sample: dict) -> None:
        """Evaluate thresholds against *sample*; log and trip on breach."""
        new_reasons = evaluate_thresholds(sample, self._cfg)
        if not new_reasons:
            return
        log.warning(
            "ResourceWatchdog threshold breach: %s", "; ".join(new_reasons)
        )
        self.reasons.extend(new_reasons)
        already_tripped = self.tripped
        self.tripped = True
        if not already_tripped and self._on_trip is not None:
            try:
                self._on_trip(new_reasons)
            except Exception:
                log.exception("ResourceWatchdog: on_trip callback raised")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary dict of all collected samples.

        Keys
        ----
        n_samples : int
            Number of samples collected.
        duration_s : float | None
            Wall-clock seconds from start() to stop() (None if not yet stopped).
        peak_vram_used_gb : dict[int, float]
            Peak VRAM used per GPU index (GiB).  Empty if no GPU data collected.
        peak_ram_used_gb : float | None
            Peak host RAM used across all samples (GiB).
        min_free_disk_gb : float | None
            Minimum free disk observed across all samples (GiB).
        tripped : bool
            Whether any threshold was ever breached.
        reasons : list[str]
            All threshold-breach reason strings accumulated.
        """
        peak_vram: dict[int, float] = {}
        peak_ram: Optional[float] = None
        min_disk: Optional[float] = None

        for s in self._samples:
            # GPU VRAM
            for gpu in s.get("gpus") or []:
                idx = gpu.get("index", 0)
                used_gb = (gpu.get("mem_used_mb") or 0.0) / 1024.0
                if idx not in peak_vram or used_gb > peak_vram[idx]:
                    peak_vram[idx] = used_gb

            # RAM
            ram = s.get("ram_used_gb")
            if ram is not None:
                peak_ram = ram if peak_ram is None else max(peak_ram, ram)

            # Disk
            disk = s.get("free_disk_gb")
            if disk is not None:
                min_disk = disk if min_disk is None else min(min_disk, disk)

        # Duration
        duration: Optional[float] = None
        if self._t_start is not None and self._t_stop is not None:
            duration = self._t_stop - self._t_start

        return {
            "n_samples": len(self._samples),
            "duration_s": duration,
            "peak_vram_used_gb": peak_vram,
            "peak_ram_used_gb": peak_ram,
            "min_free_disk_gb": min_disk,
            "tripped": self.tripped,
            "reasons": list(self.reasons),
        }
