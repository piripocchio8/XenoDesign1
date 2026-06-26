"""Generate-and-Select: harvest chirality-clean D-designs across N independent trajectories.

Pragmatic workaround for the fundamental L-bias (documented in
docs/results/2026-06-chirality-drift-diagnosis.md): LigandMPNN + Chai-1 are
L-trained, so the loop cannot reliably hold D-chirality via iterative refinement.
Instead, run N independent trajectories with different seeds and HARVEST the ones
that yield at least one chirality-clean step (violation < 0.1).

Pipeline per trajectory
-----------------------
1. Double-flip D-correct seeding (reflect_binder_in_complex_from_cif)
2. HalluLoop.run() — 7 iters, 50-step truncated_refine + context-aware LigandMPNN
3. Scan all loop steps for chirality violation < 0.1
4. Panel selection: pick the best non-vetoed step across the trajectory

Output / API
------------
generate_and_select(n, n_gpus) -> tuple[list[dict], list[dict]]
    Returns (clean_designs, trajectory_results):
    - clean_designs: chirality-clean designs ranked by composite score.
      Each dict: {traj_id, seed, step_idx, d_fasta, l_seq,
                  iptm, chirality, pll, composite, source, out_dir}
    - trajectory_results: raw per-trajectory result dicts (including errors).

CLI
---
python scripts/generate_and_select.py --n 10 --gpus 2
    Prints YIELD and a ranked table; writes outputs to /home/tmp/xd_gas_<pid>/

Design notes
------------
- Each trajectory is an independent subprocess call to design_demo.run_design_demo().
  The worker pool from run_parallel.py handles GPU dispatch — we use one "case" per
  trajectory but the design_demo runs the full pipeline (L-seed → flip → loop → panel).
- Seeds are varied per trajectory for diversity (base_seed + trajectory_idx).
- A trajectory "succeeds" if it yields >= 1 step with chirality < 0.1 AND the
  panel-selected design is chirality-clean.  We also report any individual clean
  steps found in trajectory scans.
- Pure harvest/selection logic is CPU-testable (no GPU).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Public CPU-testable data structures ───────────────────────────────────────

def harvest_clean_designs(
    trajectory_results: list[dict],
    chirality_threshold: float = 0.1,
) -> list[dict]:
    """Extract chirality-clean designs from a list of trajectory result dicts.

    Pure Python/numpy — no GPU needed.  Importable for CPU tests.

    Parameters
    ----------
    trajectory_results:
        Each dict must have the keys returned by _run_single_trajectory_in_process:
        traj_id, seed, panel_selected (dict with chirality/iptm/pll/composite/d_fasta/l_seq),
        trajectory (list of per-iter dicts), out_dir, error (or None).
    chirality_threshold:
        Steps with chirality_violation <= threshold are considered clean.
        Default 0.1 (panel veto threshold).

    Returns
    -------
    list[dict]
        Clean design records, each with:
        traj_id, seed, step_idx, d_fasta, l_seq,
        iptm, chirality, pll, composite, source ("panel_selected" | "scan"),
        out_dir.
        Ranked by composite score (descending).
    """
    clean: list[dict] = []
    for tr in trajectory_results:
        if tr.get("error"):
            continue
        traj_id = tr["traj_id"]
        seed = tr["seed"]
        out_dir = tr.get("out_dir", "")

        # Primary: panel-selected design
        ps = tr.get("panel_selected", {})
        if ps and ps.get("chirality", 1.0) <= chirality_threshold:
            clean.append({
                "traj_id": traj_id,
                "seed": seed,
                "step_idx": ps.get("iter", -1),
                "d_fasta": ps.get("d_fasta", ""),
                "l_seq": ps.get("l_seq", ""),
                "iptm": ps.get("iptm", 0.0),
                "chirality": ps.get("chirality", 0.0),
                "pll": ps.get("pll"),
                "composite": ps.get("composite", 0.0),
                "source": "panel_selected",
                "out_dir": out_dir,
            })

        # Secondary: scan all trajectory steps for any clean step not already captured
        seen_keys: set = set()
        for entry in clean:
            if entry["traj_id"] == traj_id:
                seen_keys.add((entry["step_idx"], entry["d_fasta"]))

        for step in tr.get("trajectory", []):
            if step.get("chirality", 1.0) <= chirality_threshold:
                key = (step.get("iter", -1), step.get("d_fasta", ""))
                if key not in seen_keys:
                    seen_keys.add(key)
                    clean.append({
                        "traj_id": traj_id,
                        "seed": seed,
                        "step_idx": step.get("iter", -1),
                        "d_fasta": step.get("d_fasta", ""),
                        "l_seq": step.get("l_seq", ""),
                        "iptm": step.get("iptm", 0.0),
                        "chirality": step.get("chirality", 0.0),
                        "pll": step.get("pll"),
                        "composite": step.get("composite", 0.0),
                        "source": "scan",
                        "out_dir": out_dir,
                    })

    # Rank by composite (descending); break ties by iptm
    clean.sort(key=lambda d: (d["composite"], d["iptm"]), reverse=True)
    return clean


def compute_yield(
    trajectory_results: list[dict],
    chirality_threshold: float = 0.1,
) -> tuple[int, int, float]:
    """Compute yield: (n_successful, n_total, fraction).

    A trajectory is successful if it produced >= 1 chirality-clean design
    (panel-selected OR any scan step with violation <= threshold).

    Pure Python — no GPU.
    """
    n_total = len([t for t in trajectory_results if not t.get("error")])
    n_error = len([t for t in trajectory_results if t.get("error")])
    n_ok = 0
    for tr in trajectory_results:
        if tr.get("error"):
            continue
        ps = tr.get("panel_selected", {})
        if ps and ps.get("chirality", 1.0) <= chirality_threshold:
            n_ok += 1
            continue
        for step in tr.get("trajectory", []):
            if step.get("chirality", 1.0) <= chirality_threshold:
                n_ok += 1
                break
    frac = n_ok / n_total if n_total > 0 else 0.0
    return n_ok, n_total, frac


# ── GPU-side: run one trajectory (called from worker process) ─────────────────

def _run_single_trajectory(
    traj_id: int,
    seed: int,
    device: str,
    out_base: Path,
    n_iters: int = 7,
    ref_time_steps: int = 50,
) -> dict:
    """Run one complete design trajectory and return a result dict.

    Called inside a GPU worker process.  Returns:
    {traj_id, seed, panel_selected: {iter, d_fasta, l_seq, iptm, chirality, pll, composite},
     trajectory: [{iter, d_fasta, l_seq, iptm, chirality, pll, composite, vetoed}],
     wall_time_s, out_dir, error (None on success)}
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))  # ensure /work is on path

    traj_dir = out_base / f"traj_{traj_id:03d}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    try:
        from scripts.design_demo import run_design_demo

        result = run_design_demo(
            device=device,
            out_dir=traj_dir,
            seed=seed,
        )

        traj = result["trajectory"]
        panel_idx = result["panel_best_idx"]
        sel = traj[panel_idx]

        return {
            "traj_id": traj_id,
            "seed": seed,
            "panel_selected": {
                "iter": panel_idx,
                "d_fasta": sel["d_fasta"],
                "l_seq": sel["l_seq"],
                "iptm": sel["iptm"],
                "chirality": sel["chirality"],
                "pll": sel["pll"],
                "composite": sel["composite"],
            },
            "trajectory": [
                {
                    "iter": s["iter"],
                    "d_fasta": s["d_fasta"],
                    "l_seq": s["l_seq"],
                    "iptm": s["iptm"],
                    "chirality": s["chirality"],
                    "pll": s["pll"],
                    "composite": s["composite"],
                    "vetoed": s.get("vetoed", False),
                }
                for s in traj
            ],
            "wall_time_s": result["wall_time_s"],
            "out_dir": str(traj_dir),
            "error": None,
        }
    except Exception as exc:
        import traceback
        return {
            "traj_id": traj_id,
            "seed": seed,
            "panel_selected": {},
            "trajectory": [],
            "wall_time_s": 0.0,
            "out_dir": str(traj_dir),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


# ── Worker script embedded string (runs inside the GPU subprocess) ────────────

_GAS_WORKER_SCRIPT = '''\
import os, sys, json, time
from pathlib import Path

gpu_idx = int(sys.argv[1])
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_idx))

# Redirect print/logging to stderr so only JSON protocol goes to stdout
_proto_out = sys.stdout
sys.stdout = sys.stderr

# Add /work (repo root) to path inside Docker
repo_root = os.environ.get("GAS_REPO_ROOT", "/work")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, "scripts"))

print(f"[GAS Worker GPU{gpu_idx}] Starting — pid={os.getpid()}", flush=True)

# We import run_design_demo lazily to avoid loading chai until the first case
from scripts.generate_and_select import _run_single_trajectory

print(f"[GAS Worker GPU{gpu_idx}] Ready", flush=True)

for line in iter(sys.stdin.readline, ""):
    line = line.strip()
    if not line:
        continue
    t0 = time.perf_counter()
    try:
        spec = json.loads(line)
        traj_id = spec["traj_id"]
        seed    = spec["seed"]
        out_dir = Path(spec["out_dir"])
        device  = f"cuda:0"   # each worker sees one GPU (CUDA_VISIBLE_DEVICES set by parent)
        n_iters = spec.get("n_iters", 7)
        ref_time_steps = spec.get("ref_time_steps", 50)

        result = _run_single_trajectory(
            traj_id=traj_id,
            seed=seed,
            device=device,
            out_base=out_dir,
            n_iters=n_iters,
            ref_time_steps=ref_time_steps,
        )
        result["gpu_idx"] = gpu_idx
        result["elapsed_s"] = time.perf_counter() - t0

        _proto_out.write(json.dumps(result, default=str) + "\\n")
        _proto_out.flush()

        status = "ok" if not result["error"] else "error"
        print(f"[GAS Worker GPU{gpu_idx}] traj={traj_id} seed={seed} {status} "
              f"in {result['elapsed_s']:.1f}s", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _proto_out.write(json.dumps({
            "traj_id": spec.get("traj_id", -1) if "spec" in locals() else -1,
            "seed": spec.get("seed", -1) if "spec" in locals() else -1,
            "panel_selected": {},
            "trajectory": [],
            "wall_time_s": 0.0,
            "out_dir": str(spec.get("out_dir", "")) if "spec" in locals() else "",
            "gpu_idx": gpu_idx,
            "elapsed_s": time.perf_counter() - t0,
            "error": str(e),
        }, default=str) + "\\n")
        _proto_out.flush()
'''


# ── Parallel runner ────────────────────────────────────────────────────────────

def generate_and_select(
    n: int = 10,
    n_gpus: int = 2,
    base_seed: int = 1000,
    out_dir: Optional[Path] = None,
    n_iters: int = 7,
    ref_time_steps: int = 50,
    chirality_threshold: float = 0.1,
    python_exe: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> list[dict]:
    """Run N independent design trajectories and return ranked chirality-clean designs.

    Parameters
    ----------
    n : int
        Number of independent trajectories to run.
    n_gpus : int
        Number of GPUs to use (workers_per_gpu=1 for heavy chai).
    base_seed : int
        Seeds are base_seed + traj_idx for diversity.
    out_dir : Path | None
        Root output directory.  Defaults to /home/tmp/xd_gas_<pid>.
    n_iters : int
        Number of HalluLoop iterations per trajectory (default 7).
    ref_time_steps : int
        Truncated-refine diffusion steps per iteration (default 50).
    chirality_threshold : float
        Harvest designs with chirality_violation <= this value (default 0.1).
    python_exe : str | None
        Python interpreter for worker subprocesses.
    repo_root : str | None
        Path to repo root (for PYTHONPATH in workers).

    Returns
    -------
    tuple[list[dict], list[dict]]
        (clean_designs, trajectory_results) where clean_designs are ranked chirality-clean
        designs (composite descending), each dict:
        {traj_id, seed, step_idx, d_fasta, l_seq, iptm, chirality, pll, composite,
         source, out_dir}; and trajectory_results are the raw per-trajectory dicts.
    """
    import queue
    import subprocess
    import tempfile
    import threading

    if out_dir is None:
        out_dir = Path(f"/home/tmp/xd_gas_{os.getpid()}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    python_exe = python_exe or sys.executable
    repo_root = repo_root or str(Path(__file__).parent.parent)

    # Write worker script to temp file
    tmp_dir = tempfile.mkdtemp(prefix="xd_gas_workers_")
    worker_script_path = os.path.join(tmp_dir, "_gas_worker.py")
    with open(worker_script_path, "w") as f:
        f.write(_GAS_WORKER_SCRIPT)

    # Build trajectory specs
    specs = [
        {
            "traj_id": i,
            "seed": base_seed + i,
            "out_dir": str(out_dir),
            "n_iters": n_iters,
            "ref_time_steps": ref_time_steps,
        }
        for i in range(n)
    ]

    # Queue of (traj_id, spec)
    spec_queue: queue.Queue = queue.Queue()
    for spec in specs:
        spec_queue.put(spec)

    trajectory_results: list[dict] = []
    results_lock = threading.Lock()

    # One persistent worker per GPU
    workers: dict[int, subprocess.Popen] = {}
    worker_locks: dict[int, threading.Lock] = {i: threading.Lock() for i in range(n_gpus)}

    def _start_worker(gpu_idx: int) -> subprocess.Popen:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        env["GAS_REPO_ROOT"] = repo_root
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.Popen(
            [python_exe, "-u", worker_script_path, str(gpu_idx)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=repo_root,
            text=True,
            env=env,
        )
        workers[gpu_idx] = proc
        t = threading.Thread(
            target=_relay_stderr_gas, args=(proc, gpu_idx), daemon=True
        )
        t.start()
        print(f"[GaS Pool] Started GPU{gpu_idx} pid={proc.pid}", file=sys.stderr, flush=True)
        return proc

    def _relay_stderr_gas(proc: subprocess.Popen, gpu_idx: int) -> None:
        try:
            for line in proc.stderr:
                print(f"[GPU{gpu_idx}] {line}", end="", file=sys.stderr, flush=True)
        except (ValueError, OSError):
            pass

    def _consumer(gpu_idx: int) -> None:
        while True:
            try:
                spec = spec_queue.get_nowait()
            except queue.Empty:
                break

            with worker_locks[gpu_idx]:
                proc = workers.get(gpu_idx)
                if proc is None or proc.poll() is not None:
                    proc = _start_worker(gpu_idx)

                try:
                    proc.stdin.write(json.dumps(spec, default=str) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    proc = _start_worker(gpu_idx)
                    proc.stdin.write(json.dumps(spec, default=str) + "\n")
                    proc.stdin.flush()

                line = proc.stdout.readline()
                if not line:
                    print(
                        f"[GaS Pool] GPU{gpu_idx} worker died for traj={spec['traj_id']}",
                        file=sys.stderr, flush=True
                    )
                    with results_lock:
                        trajectory_results.append({
                            "traj_id": spec["traj_id"],
                            "seed": spec["seed"],
                            "panel_selected": {},
                            "trajectory": [],
                            "wall_time_s": 0.0,
                            "out_dir": str(out_dir),
                            "error": "Worker died (no response)",
                        })
                    continue

                try:
                    result = json.loads(line)
                except json.JSONDecodeError as e:
                    result = {
                        "traj_id": spec["traj_id"],
                        "seed": spec["seed"],
                        "panel_selected": {},
                        "trajectory": [],
                        "wall_time_s": 0.0,
                        "out_dir": str(out_dir),
                        "error": f"JSON parse error: {e}  raw={line!r}",
                    }

                with results_lock:
                    trajectory_results.append(result)

    # Start consumers (one per GPU, sequentially — they block on workers)
    try:
        threads = []
        for gpu_idx in range(n_gpus):
            _start_worker(gpu_idx)

        for gpu_idx in range(n_gpus):
            t = threading.Thread(
                target=_consumer,
                args=(gpu_idx,),
                name=f"gas-consumer-{gpu_idx}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    finally:
        # Kill all workers
        for gpu_idx, proc in list(workers.items()):
            if proc and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    # Harvest clean designs
    clean = harvest_clean_designs(trajectory_results, chirality_threshold)
    return clean, trajectory_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_table(clean_designs: list[dict], trajectory_results: list[dict],
                 n_ok: int, n_total: int, frac: float, wall_time_s: float) -> None:
    """Print results table to stdout."""
    print(f"\n{'='*100}")
    print(f"Generate-and-Select Results")
    print(f"  N run: {n_total}  |  N errors: {len([t for t in trajectory_results if t.get('error')])} "
          f"|  Wall-clock: {wall_time_s/60:.1f} min")
    print(f"  YIELD: {n_ok}/{n_total} trajectories ({frac*100:.1f}%) produced ≥1 chirality-clean design "
          f"(violation ≤ 0.1)")
    print(f"  Total clean designs harvested: {len(clean_designs)}")
    print(f"{'='*100}")

    if not clean_designs:
        print("\nNo chirality-clean designs found in any trajectory.\n")
        return

    print(f"\n{'Rank':>4}  {'Traj':>4}  {'Seed':>6}  {'Step':>4}  "
          f"{'Sequence (L)':>20}  {'ipTM':>6}  {'Chir':>6}  {'PLL':>7}  "
          f"{'Composite':>9}  {'Source':>14}")
    print(f"{'─'*100}")
    for rank, d in enumerate(clean_designs, 1):
        pll_str = f"{d['pll']:7.3f}" if d['pll'] is not None else f"{'N/A':>7}"
        print(f"  {rank:2d}  {d['traj_id']:4d}  {d['seed']:6d}  {d['step_idx']:4d}  "
              f"  {d['l_seq']:>20}  {d['iptm']:6.4f}  {d['chirality']:6.3f}  "
              f"{pll_str}  {d['composite']:9.4f}  {d['source']:>14}")

    best = clean_designs[0]
    pll_best = f"{best['pll']:.3f}" if best['pll'] is not None else "N/A"
    print(f"\n{'='*100}")
    print(f"BEST CLEAN DESIGN:")
    print(f"  Traj: {best['traj_id']}  Seed: {best['seed']}  Step: {best['step_idx']}")
    print(f"  D-CCD:     {best['d_fasta']}")
    print(f"  L-seq:     {best['l_seq']}")
    print(f"  ipTM:      {best['iptm']:.4f}")
    print(f"  Chirality: {best['chirality']:.4f}  (< 0.1 ✓)")
    print(f"  PLL:       {pll_best}")
    print(f"  Composite: {best['composite']:.4f}")
    print(f"  Output:    {best['out_dir']}")
    print(f"{'='*100}\n")

    # Per-trajectory summary
    print("Per-trajectory summary:")
    print(f"  {'Traj':>4}  {'Seed':>6}  {'Status':>8}  {'PanelChir':>9}  {'PanelIPTM':>9}  "
          f"{'CleanSteps':>10}  {'Error':}")
    print(f"  {'─'*80}")
    for tr in sorted(trajectory_results, key=lambda x: x["traj_id"]):
        err_str = tr.get("error", "")
        if err_str and len(err_str) > 60:
            err_str = err_str[:60] + "..."
        ps = tr.get("panel_selected", {})
        chir_str = f"{ps.get('chirality', float('nan')):.3f}" if ps else "N/A"
        iptm_str = f"{ps.get('iptm', float('nan')):.4f}" if ps else "N/A"
        traj_steps = tr.get("trajectory", [])
        n_clean = sum(1 for s in traj_steps if s.get("chirality", 1.0) <= 0.1)
        status = "ERROR" if tr.get("error") else ("CLEAN" if n_clean > 0 else "drift")
        print(f"  {tr['traj_id']:4d}  {tr['seed']:6d}  {status:>8}  "
              f"{chir_str:>9}  {iptm_str:>9}  {n_clean:>10}  {err_str or ''}")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate-and-Select: harvest chirality-clean D-designs across N trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=10,
                        help="Number of independent design trajectories")
    parser.add_argument("--gpus", type=int, default=2,
                        help="Number of GPUs to use")
    parser.add_argument("--base_seed", type=int, default=1000,
                        help="Base seed; traj i uses seed base_seed+i")
    parser.add_argument("--out_dir", default=None,
                        help="Output root dir (default: /home/tmp/xd_gas_<pid>)")
    parser.add_argument("--n_iters", type=int, default=7,
                        help="HalluLoop iterations per trajectory")
    parser.add_argument("--ref_time_steps", type=int, default=50,
                        help="Truncated-refine diffusion steps per iteration")
    parser.add_argument("--chirality_threshold", type=float, default=0.1,
                        help="Chirality violation threshold for harvest (default 0.1)")
    parser.add_argument("--results_json", default=None,
                        help="Write full trajectory results to this JSON file")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"/home/tmp/xd_gas_{os.getpid()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[GaS] Starting generate-and-select: N={args.n}, {args.gpus} GPU(s), "
          f"base_seed={args.base_seed}", file=sys.stderr, flush=True)
    print(f"[GaS] Out dir: {out_dir}", file=sys.stderr, flush=True)
    print(f"[GaS] Pipeline: {args.n_iters} HalluLoop iters × {args.ref_time_steps} refine "
          f"steps/iter per trajectory", file=sys.stderr, flush=True)

    t0 = time.time()
    clean_designs, trajectory_results = generate_and_select(
        n=args.n,
        n_gpus=args.gpus,
        base_seed=args.base_seed,
        out_dir=out_dir,
        n_iters=args.n_iters,
        ref_time_steps=args.ref_time_steps,
        chirality_threshold=args.chirality_threshold,
    )
    wall_time = time.time() - t0

    n_ok, n_total, frac = compute_yield(
        trajectory_results, args.chirality_threshold
    )

    _print_table(clean_designs, trajectory_results, n_ok, n_total, frac, wall_time)

    # Save JSON results
    if args.results_json:
        results_path = Path(args.results_json)
    else:
        results_path = out_dir / "results.json"

    with open(results_path, "w") as f:
        json.dump({
            "n": args.n,
            "n_gpus": args.gpus,
            "base_seed": args.base_seed,
            "chirality_threshold": args.chirality_threshold,
            "n_ok": n_ok,
            "n_total": n_total,
            "yield_frac": frac,
            "wall_time_s": wall_time,
            "clean_designs": clean_designs,
            "trajectory_results": trajectory_results,
        }, f, indent=2, default=str)
    print(f"[GaS] Results saved to {results_path}", file=sys.stderr, flush=True)

    # Exit code: 0 if any clean design found, 1 otherwise
    sys.exit(0 if clean_designs else 1)


if __name__ == "__main__":
    _main()
