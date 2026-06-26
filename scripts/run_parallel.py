"""Multi-GPU persistent-worker runner for XenoDesign batch jobs.

Worker-pool pattern: one long-lived subprocess per GPU, each pinned to a single device
via CUDA_VISIBLE_DEVICES (so every worker sees its own GPU as cuda:0). chai weights load
ONCE per worker and are reused for every case that worker pulls from a shared queue. This
keeps each GPU saturated by a single heavy chai process and amortises the multi-GB weight
load across the whole batch. Defaults to 2 GPUs (ifrit) but scales to any N via --n_gpus
/ run_cases(n_gpus=...); on a SLURM allocation, use one task (1 GPU) per worker.

Public API
----------
run_cases(cases, n_gpus=2, workers_per_gpu=1, seed=42) -> list[CaseResult]
    Run a list of cases across n_gpus.  Importable without torch/chai (lazy imports).

CLI
---
python scripts/run_parallel.py --fasta a.fasta b.fasta --out_dir /tmp/out [--n_gpus 2]
python scripts/run_parallel.py --cases_json cases.json --out_dir /tmp/out

Design notes
------------
- One subprocess per GPU, pinned via CUDA_VISIBLE_DEVICES=<i>; each sees its GPU as
  cuda:0.  Only one worker per GPU for heavy chai (20 GB VRAM, 1 worker saturates it).
- Protocol: parent → worker over stdin (one JSON line per case), worker → parent over
  stdout (one JSON response per case).  All chai/tqdm output goes to stderr (relayed).
- OOM / dead worker: the case is re-queued and the worker is restarted.
- Pure helpers (plan_worker_slots, collate_results) are CPU-testable with no torch/chai.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Pure, CPU-testable data structures and helpers
# ---------------------------------------------------------------------------

@dataclass
class CaseSpec:
    """One design/predict case to run through the worker pool.

    Attributes
    ----------
    case_id : str
        Unique identifier for this case (used in output paths and result logging).
    entities : list[dict]
        Sequence entities as expected by ChaiBackend.predict() / write_inputs().
        Each dict: {"type": "protein", "name": ..., "sequence": ..., "chirality": ...}
    out_dir : str | Path
        Directory where this case's outputs should be written.
    params : dict
        Extra kwargs forwarded to ChaiBackend.predict(), e.g. num_diffn_timesteps.
    """
    case_id: str
    entities: list[dict]
    out_dir: str | Path
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)


@dataclass
class CaseResult:
    """Result returned by run_cases() for one CaseSpec."""
    case_id: str
    status: str          # "ok" | "error"
    out_dir: Path
    gpu_idx: int         # which GPU ran this case
    elapsed_s: float     # wall time for this case
    error: Optional[str] = None


def plan_worker_slots(n_gpus: int, workers_per_gpu: int = 1) -> list[tuple[int, int]]:
    """Return list of (gpu_idx, worker_num) slots — pure, CPU-testable.

    For heavy chai jobs on 20 GB GPUs the safe choice is workers_per_gpu=1 (one slot
    per GPU).  A pure planner with no runtime VRAM query, so it runs on a CPU host.

    >>> plan_worker_slots(2, 1)
    [(0, 0), (1, 0)]
    >>> plan_worker_slots(2, 2)
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    """
    return [(gpu_idx, w) for gpu_idx in range(n_gpus) for w in range(workers_per_gpu)]


def collate_results(results: dict[str, CaseResult], cases: list[CaseSpec]) -> list[CaseResult]:
    """Return results in input order, preserving original case order.

    Cases that have no result entry (shouldn't happen normally) are included as
    error results so callers always get len(cases) entries back.

    >>> from pathlib import Path
    >>> cs = [CaseSpec("a", [], Path("/x")), CaseSpec("b", [], Path("/y"))]
    >>> ra = CaseResult("a", "ok", Path("/x"), 0, 1.2)
    >>> rb = CaseResult("b", "error", Path("/y"), 1, 0.5, "oops")
    >>> collate_results({"a": ra, "b": rb}, cs) == [ra, rb]
    True
    """
    ordered = []
    for spec in cases:
        if spec.case_id in results:
            ordered.append(results[spec.case_id])
        else:
            ordered.append(CaseResult(
                case_id=spec.case_id,
                status="error",
                out_dir=spec.out_dir,
                gpu_idx=-1,
                elapsed_s=0.0,
                error="no result recorded (worker pool did not process this case)",
            ))
    return ordered


# ---------------------------------------------------------------------------
# Persistent-worker subprocess script (the worker side)
# ---------------------------------------------------------------------------

# This script runs as a separate Python process, pinned to one GPU via
# CUDA_VISIBLE_DEVICES (set by the parent before spawning).  It:
#   1. Imports chai once (loading weights into GPU memory)
#   2. Loops reading JSON lines from stdin, running predict, writing JSON to stdout
#
# Protocol channel: only JSON goes to real stdout; everything else (chai print/tqdm)
# is redirected to stderr so it doesn't corrupt the protocol.

_WORKER_SCRIPT = '''\
import os, sys, json, time, tempfile, shutil
from pathlib import Path

gpu_idx = int(sys.argv[1])
# CUDA_VISIBLE_DEVICES already set by parent — this process sees only one GPU as cuda:0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_idx))

# Redirect print/logging to stderr; real stdout is our JSON protocol channel
_proto_out = sys.stdout
sys.stdout = sys.stderr

# ---- Load chai (weights land in GPU memory here — this is the once-per-worker load) ----
import torch
device_str = "cuda:0"
device = torch.device(device_str)

# Trigger import (weights won't load until first call, but module is ready)
from chai_lab.chai1 import run_inference
from xenodesign.backends.chai_backend import write_inputs, load_prediction, _save_confidence_npz
import numpy as np

print(f"[Worker GPU{gpu_idx}] Ready. device={device_str} pid={os.getpid()}", flush=True)

load_counter = 0  # counts how many times we load a case (weights load once)

for line in iter(sys.stdin.readline, ""):
    line = line.strip()
    if not line:
        continue
    t0 = time.perf_counter()
    try:
        spec = json.loads(line)
        case_id  = spec["case_id"]
        entities = spec["entities"]
        out_dir  = Path(spec["out_dir"])
        params   = spec.get("params", {})
        seed     = spec.get("seed", 42)
        num_diffn_timesteps = params.get("num_diffn_timesteps", 200)

        out_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = write_inputs(entities, out_dir)

        # chai 0.6.1: output_dir must be empty; FASTA is in out_dir so use subdir
        chai_out = out_dir / "chai_out"
        if chai_out.exists():
            shutil.rmtree(chai_out)
        chai_out.mkdir(parents=True, exist_ok=True)

        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=chai_out,
            device=device_str,
            seed=seed,
            num_diffn_timesteps=num_diffn_timesteps,
            use_esm_embeddings=True,
            use_msa_server=False,
        )
        _save_confidence_npz(candidates, chai_out)
        torch.cuda.empty_cache()

        load_counter += 1
        elapsed = time.perf_counter() - t0
        _proto_out.write(json.dumps({
            "status": "ok",
            "case_id": case_id,
            "gpu_idx": gpu_idx,
            "elapsed_s": elapsed,
            "load_counter": load_counter,
        }) + "\\n")
        _proto_out.flush()
        print(f"[Worker GPU{gpu_idx}] case={case_id} done in {elapsed:.1f}s "
              f"(case #{load_counter} this worker)", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        elapsed = time.perf_counter() - t0
        _proto_out.write(json.dumps({
            "status": "error",
            "case_id": spec.get("case_id", "?") if "spec" in dir() else "?",
            "gpu_idx": gpu_idx,
            "elapsed_s": elapsed,
            "message": str(e),
        }) + "\\n")
        _proto_out.flush()
'''

# ---------------------------------------------------------------------------
# Worker process management (host side)
# ---------------------------------------------------------------------------

_workers: dict[tuple[int, int], subprocess.Popen] = {}
_worker_io_locks: dict[tuple[int, int], threading.Lock] = {}
_worker_lock = threading.Lock()


def _start_worker(gpu_idx: int, worker_num: int, python_exe: str,
                  worker_script_path: str, work_dir: str) -> subprocess.Popen:
    """Spawn a persistent worker subprocess pinned to gpu_idx."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    proc = subprocess.Popen(
        [python_exe, "-u", worker_script_path, str(gpu_idx)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=work_dir,
        text=True,
        env=env,
    )
    key = (gpu_idx, worker_num)
    _workers[key] = proc
    _worker_io_locks.setdefault(key, threading.Lock())

    # Relay stderr from worker to our stderr
    t = threading.Thread(
        target=_relay_stderr, args=(proc, gpu_idx, worker_num), daemon=True
    )
    t.start()
    print(f"[Pool] Started GPU{gpu_idx}:W{worker_num} pid={proc.pid}", file=sys.stderr)
    return proc


def _relay_stderr(proc: subprocess.Popen, gpu_idx: int, worker_num: int) -> None:
    try:
        for line in proc.stderr:
            print(f"[GPU{gpu_idx}:W{worker_num}] {line}", end="", file=sys.stderr)
    except (ValueError, OSError):
        pass


def _get_worker(gpu_idx: int, worker_num: int,
                python_exe: str, worker_script_path: str, work_dir: str) -> subprocess.Popen:
    """Return live worker, restarting if dead."""
    key = (gpu_idx, worker_num)
    with _worker_lock:
        proc = _workers.get(key)
        if proc is None or proc.poll() is not None:
            if proc is not None:
                print(f"[Pool] GPU{gpu_idx}:W{worker_num} died "
                      f"(rc={proc.returncode}), restarting...", file=sys.stderr)
            proc = _start_worker(gpu_idx, worker_num, python_exe,
                                 worker_script_path, work_dir)
        return proc


def _send_to_worker(gpu_idx: int, worker_num: int, payload: dict,
                    python_exe: str, worker_script_path: str,
                    work_dir: str) -> dict:
    """Send one case to a worker and block for its JSON response."""
    key = (gpu_idx, worker_num)
    lock = _worker_io_locks.setdefault(key, threading.Lock())
    with lock:
        proc = _get_worker(gpu_idx, worker_num, python_exe,
                           worker_script_path, work_dir)
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            print(f"[Pool] GPU{gpu_idx}:W{worker_num} pipe broken, restarting...",
                  file=sys.stderr)
            proc = _start_worker(gpu_idx, worker_num, python_exe,
                                 worker_script_path, work_dir)
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"Worker GPU{gpu_idx}:W{worker_num} died (no response)"
            )
        return json.loads(line)


def _kill_workers() -> None:
    with _worker_lock:
        for (gpu_idx, worker_num), proc in list(_workers.items()):
            if proc and proc.poll() is None:
                print(f"[Pool] Killing GPU{gpu_idx}:W{worker_num} "
                      f"(pid={proc.pid})", file=sys.stderr)
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        _workers.clear()
        _worker_io_locks.clear()


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_cases(
    cases: Sequence[CaseSpec],
    n_gpus: int = 2,
    workers_per_gpu: int = 1,
    seed: int = 42,
    python_exe: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> list[CaseResult]:
    """Run cases across n_gpus with persistent workers; return results in input order.

    Parameters
    ----------
    cases : list[CaseSpec]
        Cases to run.  Each carries its own out_dir — outputs land there.
    n_gpus : int
        Number of GPUs to use (default 2 for ifrit's A4500s).
    workers_per_gpu : int
        Workers per GPU (keep at 1 for heavy chai; 20 GB VRAM saturates with 1 worker).
    seed : int
        RNG seed forwarded to ChaiBackend.predict() for reproducibility.
    python_exe : str | None
        Python interpreter to use for worker subprocesses.  Defaults to sys.executable
        (i.e. the same interpreter that is running this script).
    work_dir : str | None
        Working directory for worker subprocesses (needed so `xenodesign` is on the
        import path inside the Docker container).  Defaults to the repo root (/work
        inside container, or the directory containing this script's parent).

    Returns
    -------
    list[CaseResult]
        One result per input case, in the same order, even if some failed.
    """
    if not cases:
        return []

    python_exe = python_exe or sys.executable
    if work_dir is None:
        # Default: parent of the `scripts/` dir (the repo root)
        work_dir = str(Path(__file__).parent.parent)

    slots = plan_worker_slots(n_gpus, workers_per_gpu)
    if not slots:
        raise ValueError(f"No worker slots: n_gpus={n_gpus}, workers_per_gpu={workers_per_gpu}")

    # Write worker script to a temp file
    tmp_dir = tempfile.mkdtemp(prefix="xd_workers_")
    worker_script_path = os.path.join(tmp_dir, "_persistent_worker.py")
    with open(worker_script_path, "w") as f:
        f.write(_WORKER_SCRIPT)

    # Shared queue of (original_index, CaseSpec) pairs
    case_queue: queue.Queue[tuple[int, CaseSpec]] = queue.Queue()
    for i, spec in enumerate(cases):
        case_queue.put((i, spec))

    results: dict[str, CaseResult] = {}
    results_lock = threading.Lock()
    # stop_event: wires KeyboardInterrupt → graceful consumer exit.
    # Consumers check is_set() each iteration; set() is called from a SIGINT handler below.
    stop_event = threading.Event()

    def _consumer(gpu_idx: int, worker_num: int) -> None:
        wid = f"GPU{gpu_idx}:W{worker_num}"
        while not stop_event.is_set():
            try:
                orig_idx, spec = case_queue.get_nowait()
            except queue.Empty:
                break  # nothing left, consumer exits

            print(f"[{wid}] Processing case={spec.case_id}", file=sys.stderr)
            payload = {
                "case_id": spec.case_id,
                "entities": spec.entities,
                "out_dir": str(spec.out_dir),
                "params": spec.params,
                "seed": seed,
            }
            try:
                resp = _send_to_worker(
                    gpu_idx, worker_num, payload,
                    python_exe, worker_script_path, work_dir,
                )
                if resp.get("status") == "ok":
                    result = CaseResult(
                        case_id=spec.case_id,
                        status="ok",
                        out_dir=spec.out_dir,
                        gpu_idx=resp["gpu_idx"],
                        elapsed_s=resp["elapsed_s"],
                    )
                else:
                    result = CaseResult(
                        case_id=spec.case_id,
                        status="error",
                        out_dir=spec.out_dir,
                        gpu_idx=gpu_idx,
                        elapsed_s=resp.get("elapsed_s", 0.0),
                        error=resp.get("message", "unknown error"),
                    )
            except RuntimeError as e:
                # Worker died (OOM or crash) — re-queue then restart and continue so THIS
                # consumer drains the re-queued item.  With a single consumer per GPU the
                # old break would cause the re-queued case to be silently dropped (the queue
                # becomes non-empty after all consumers have exited).
                err_str = str(e)
                print(f"[{wid}] Worker error: {err_str}, re-queuing case={spec.case_id} "
                      f"and restarting worker", file=sys.stderr)
                case_queue.put((orig_idx, spec))
                # Restart the dead worker so this consumer can process the re-queued case
                # on the next loop iteration.  _get_worker() already handles restart, but
                # we force it here by removing the stale entry from _workers so the next
                # _get_worker call unconditionally spawns a fresh process.
                with _worker_lock:
                    key = (gpu_idx, worker_num)
                    dead = _workers.pop(key, None)
                    if dead is not None and dead.poll() is None:
                        dead.kill()
                continue  # pick up the re-queued case on the next iteration

            with results_lock:
                results[spec.case_id] = result

    try:
        threads = []
        for gpu_idx, worker_num in slots:
            t = threading.Thread(
                target=_consumer,
                args=(gpu_idx, worker_num),
                name=f"xd-consumer-{gpu_idx}-{worker_num}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n[Pool] KeyboardInterrupt — stopping consumers...", file=sys.stderr)
            stop_event.set()
            for t in threads:
                t.join(timeout=5)

    finally:
        _kill_workers()
        # Clean up temp worker script
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return collate_results(results, list(cases))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_cases_from_fastas(fasta_paths: list[str], out_dir: Path,
                              params: dict) -> list[CaseSpec]:
    """Build CaseSpec list from raw FASTA paths (CLI helper).

    Each FASTA is treated as a single case; entities are parsed from the FASTA header
    lines (>protein|name or >ligand|name).
    """
    cases = []
    for fp in fasta_paths:
        fp = Path(fp)
        case_id = fp.stem
        # Parse entities from FASTA
        entities = []
        with open(fp) as fh:
            current_header = None
            seq_lines: list[str] = []
            for line in fh:
                line = line.rstrip()
                if line.startswith(">"):
                    if current_header is not None and seq_lines:
                        entities.append(_header_to_entity(current_header, "".join(seq_lines)))
                    current_header = line[1:]
                    seq_lines = []
                else:
                    seq_lines.append(line)
            if current_header is not None and seq_lines:
                entities.append(_header_to_entity(current_header, "".join(seq_lines)))
        case_out = out_dir / case_id
        cases.append(CaseSpec(case_id=case_id, entities=entities,
                              out_dir=case_out, params=params))
    return cases


def _header_to_entity(header: str, sequence: str) -> dict:
    """Parse a FASTA header like 'protein|name' into an entity dict."""
    parts = header.split("|", 1)
    entity_type = parts[0].strip().lower() if parts else "protein"
    name = parts[1].strip() if len(parts) > 1 else header
    return {"type": entity_type, "name": name, "sequence": sequence, "chirality": "L"}


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-GPU persistent-worker runner for XenoDesign batch jobs."
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--fasta", nargs="+", help="FASTA file(s) to predict")
    grp.add_argument("--cases_json", help="JSON file with list of CaseSpec dicts")

    parser.add_argument("--out_dir", required=True, help="Root output directory")
    parser.add_argument("--n_gpus", type=int, default=2,
                        help="Number of GPUs to use (default: 2)")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Workers per GPU (default: 1; keep 1 for heavy chai)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--num_diffn_timesteps", type=int, default=200,
                        help="Number of diffusion timesteps (default: 200)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {"num_diffn_timesteps": args.num_diffn_timesteps}

    if args.fasta:
        cases = _build_cases_from_fastas(args.fasta, out_dir, params)
    else:
        with open(args.cases_json) as f:
            raw = json.load(f)
        cases = [
            CaseSpec(
                case_id=d["case_id"],
                entities=d["entities"],
                out_dir=Path(d["out_dir"]),
                params=d.get("params", params),
            )
            for d in raw
        ]

    print(f"[run_parallel] {len(cases)} cases, {args.n_gpus} GPU(s), "
          f"{args.workers_per_gpu} worker(s)/GPU", file=sys.stderr)

    t0 = time.perf_counter()
    results = run_cases(
        cases,
        n_gpus=args.n_gpus,
        workers_per_gpu=args.workers_per_gpu,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    ok = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status == "error")
    print(f"\n[run_parallel] Done in {elapsed:.1f}s: {ok} ok, {err} errors",
          file=sys.stderr)

    for r in results:
        status_str = "OK" if r.status == "ok" else f"ERROR: {r.error}"
        print(f"  {r.case_id}  GPU{r.gpu_idx}  {r.elapsed_s:.1f}s  {status_str}")


if __name__ == "__main__":
    _main()
