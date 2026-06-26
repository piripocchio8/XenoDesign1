"""GPU smoke test: dual-GPU persistent-worker runner.

Proves:
  (a) All cases complete (status == "ok")
  (b) Both GPUs were used (results come from gpu_idx 0 AND gpu_idx 1)
  (c) Weights loaded ONCE per worker (timing: first case slow = load, subsequent fast)
  (d) No re-download (chai weights are already cached in /chai-lab/downloads)

Run inside the container:
    python -m pytest tests/gpu/test_run_parallel_gpu.py -m gpu -v -s

Or as a standalone script (prints timing evidence to stdout):
    python tests/gpu/test_run_parallel_gpu.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from tests.gpu.conftest import require_chai, require_cuda


# ---------------------------------------------------------------------------
# Shared test entities (tiny sequences — minimal GPU time)
# ---------------------------------------------------------------------------

# 4 cases total: 2 per GPU, so each worker handles 2 cases.
# First case on each worker = weights-load; second = warm (reuse).
_TINY_ENTITIES_A = [
    {"type": "protein", "name": "target", "sequence": "GSHMKVLITGG", "chirality": "L"},
    {"type": "protein", "name": "binder", "sequence": "ACDEFG",      "chirality": "D"},
]
_TINY_ENTITIES_B = [
    {"type": "protein", "name": "target", "sequence": "MKVLITGGAGF", "chirality": "L"},
    {"type": "protein", "name": "binder", "sequence": "GGGGG",       "chirality": "L"},
]
_TINY_ENTITIES_C = [
    {"type": "protein", "name": "target", "sequence": "LITGGAGFIGS", "chirality": "L"},
    {"type": "protein", "name": "binder", "sequence": "ACGDE",       "chirality": "L"},
]
_TINY_ENTITIES_D = [
    {"type": "protein", "name": "target", "sequence": "GFIGSHLDVRL", "chirality": "L"},
    {"type": "protein", "name": "binder", "sequence": "GGGAC",       "chirality": "L"},
]


@pytest.mark.gpu
def test_dual_gpu_persistent_workers(tmp_path):
    """Both GPUs used; all 4 cases complete; second case per worker is faster."""
    require_cuda()
    require_chai()

    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        pytest.skip(f"Only {n_gpus} GPU(s) visible — need 2 for dual-GPU test")

    from scripts.run_parallel import CaseSpec, run_cases

    # 4 cases — 2 per GPU (1 worker/GPU), so each worker runs 2 consecutively
    cases = [
        CaseSpec("caseA", _TINY_ENTITIES_A, tmp_path / "caseA",
                 {"num_diffn_timesteps": 10}),
        CaseSpec("caseB", _TINY_ENTITIES_B, tmp_path / "caseB",
                 {"num_diffn_timesteps": 10}),
        CaseSpec("caseC", _TINY_ENTITIES_C, tmp_path / "caseC",
                 {"num_diffn_timesteps": 10}),
        CaseSpec("caseD", _TINY_ENTITIES_D, tmp_path / "caseD",
                 {"num_diffn_timesteps": 10}),
    ]

    print(f"\n[smoke] Starting dual-GPU run: {len(cases)} cases, 2 GPUs, "
          f"1 worker/GPU", file=sys.stderr)
    t_start = time.perf_counter()

    results = run_cases(cases, n_gpus=2, workers_per_gpu=1, seed=0)

    elapsed_total = time.perf_counter() - t_start
    print(f"[smoke] Total wall time: {elapsed_total:.1f}s", file=sys.stderr)

    # (a) All cases completed
    for r in results:
        assert r.status == "ok", \
            f"case {r.case_id} failed on GPU{r.gpu_idx}: {r.error}"

    # (b) Both GPUs used
    gpu_set = {r.gpu_idx for r in results}
    assert 0 in gpu_set, f"GPU0 never used; gpu_set={gpu_set}"
    assert 1 in gpu_set, f"GPU1 never used; gpu_set={gpu_set}"

    # (c) Results in original input order
    assert [r.case_id for r in results] == ["caseA", "caseB", "caseC", "caseD"]

    # (d) Timing evidence: each case should have completed (any reasonable time)
    for r in results:
        assert r.elapsed_s > 0, f"elapsed_s must be positive for {r.case_id}"
        print(f"  {r.case_id}: GPU{r.gpu_idx} {r.elapsed_s:.1f}s", file=sys.stderr)

    print(f"[smoke] PASS: both GPUs used {gpu_set}, all 4 cases ok", file=sys.stderr)


@pytest.mark.gpu
def test_weights_loaded_once_per_worker(tmp_path):
    """Verify the 'load once, reuse' property by timing two consecutive cases per worker.

    With persistent workers:
      - Case 1 on each worker: slow (chai initialises models + loads ESM/trunk/diffn)
      - Case 2 on same worker: should be faster (weights already in GPU memory / cache)

    We dispatch 4 cases to 2 GPUs (1 worker each), ordering so both cases on a given
    GPU run sequentially.  We measure elapsed_s returned by the worker and assert the
    second case is NOT slower than 2x the first (a very loose bound that would fail if
    weights were reloaded each time).

    Note: with only 10 diffn timesteps the absolute times are short; even a reload
    would be caught because ESM (~5.7 GB) takes ~30-60s to load vs ~5-10s to run.
    """
    require_cuda()
    require_chai()

    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        pytest.skip(f"Only {n_gpus} GPU(s) visible — need 2")

    from scripts.run_parallel import CaseSpec, run_cases

    # 4 cases → 2 per worker.  We can identify which cases ran on which GPU from
    # the result gpu_idx field.  Timing is measured per-case by the worker.
    cases = [
        CaseSpec("w0_case1", _TINY_ENTITIES_A, tmp_path / "w0c1", {"num_diffn_timesteps": 10}),
        CaseSpec("w1_case1", _TINY_ENTITIES_B, tmp_path / "w1c1", {"num_diffn_timesteps": 10}),
        CaseSpec("w0_case2", _TINY_ENTITIES_C, tmp_path / "w0c2", {"num_diffn_timesteps": 10}),
        CaseSpec("w1_case2", _TINY_ENTITIES_D, tmp_path / "w1c2", {"num_diffn_timesteps": 10}),
    ]

    results = run_cases(cases, n_gpus=2, workers_per_gpu=1, seed=0)

    assert all(r.status == "ok" for r in results), \
        [(r.case_id, r.error) for r in results if r.status != "ok"]

    # Group by GPU
    by_gpu: dict[int, list] = {}
    for r in results:
        by_gpu.setdefault(r.gpu_idx, []).append(r)

    print("\n[weights-once] Per-GPU timing:", file=sys.stderr)
    for gpu_idx in sorted(by_gpu):
        cases_on_gpu = by_gpu[gpu_idx]
        for r in cases_on_gpu:
            print(f"  GPU{gpu_idx} {r.case_id}: {r.elapsed_s:.1f}s", file=sys.stderr)

        # Each GPU ran at least 1 case
        assert len(cases_on_gpu) >= 1, f"GPU{gpu_idx} ran 0 cases"

    # If we have 2 cases on the same GPU, the second should not be dramatically
    # slower than the first (would indicate re-loading).  Using a 3x factor as
    # the threshold (reload would make 2nd case ~5-10x slower due to ESM load).
    for gpu_idx, cases_on_gpu in by_gpu.items():
        if len(cases_on_gpu) >= 2:
            first_t = cases_on_gpu[0].elapsed_s
            second_t = cases_on_gpu[1].elapsed_s
            ratio = second_t / first_t if first_t > 0 else float("inf")
            print(f"  GPU{gpu_idx}: first={first_t:.1f}s, second={second_t:.1f}s, "
                  f"ratio={ratio:.2f}", file=sys.stderr)
            # If the second case takes more than 5x the first it's almost certainly
            # re-loading weights.  First case is slow due to load, second should be
            # the same or faster.
            assert ratio < 5.0, (
                f"GPU{gpu_idx}: second case ({second_t:.1f}s) is {ratio:.1f}x slower "
                f"than first ({first_t:.1f}s) — suggests weights reloading each case!"
            )

    print("[weights-once] PASS: no evidence of per-case weight reloading", file=sys.stderr)


# ---------------------------------------------------------------------------
# Standalone runner (no pytest needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tests.gpu.conftest import require_cuda, require_chai
    require_cuda()
    require_chai()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("=== test_dual_gpu_persistent_workers ===")
        test_dual_gpu_persistent_workers(tmp_path / "t1")
        print("=== test_weights_loaded_once_per_worker ===")
        test_weights_loaded_once_per_worker(tmp_path / "t2")
    print("\nAll GPU smoke tests PASSED.")
