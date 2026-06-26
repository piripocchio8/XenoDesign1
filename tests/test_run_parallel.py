"""CPU-only tests for scripts/run_parallel.py pure helpers.

No torch, chai_lab, or GPU required — run with:
    pytest tests/test_run_parallel.py -m "not gpu and not network" -q

Tests cover:
- plan_worker_slots: slot planning
- collate_results: ordering + missing-result sentinel
- CaseSpec post_init: Path coercion
- _build_cases_from_fastas: FASTA parsing
- _header_to_entity: header parsing
- run_cases mock-backend: queue dispatch, result collation, stub workers
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import only pure helpers; do NOT import torch/chai at collection time
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_parallel import (
    CaseResult,
    CaseSpec,
    _build_cases_from_fastas,
    _header_to_entity,
    collate_results,
    plan_worker_slots,
)


# ---------------------------------------------------------------------------
# plan_worker_slots
# ---------------------------------------------------------------------------

class TestPlanWorkerSlots:
    def test_two_gpus_one_worker_each(self):
        assert plan_worker_slots(2, 1) == [(0, 0), (1, 0)]

    def test_two_gpus_two_workers_each(self):
        assert plan_worker_slots(2, 2) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_one_gpu(self):
        assert plan_worker_slots(1, 1) == [(0, 0)]

    def test_zero_gpus(self):
        assert plan_worker_slots(0, 1) == []

    def test_four_gpus(self):
        slots = plan_worker_slots(4, 1)
        gpu_indices = [s[0] for s in slots]
        assert gpu_indices == [0, 1, 2, 3]

    def test_worker_num_increments_per_gpu(self):
        slots = plan_worker_slots(1, 3)
        assert slots == [(0, 0), (0, 1), (0, 2)]


# ---------------------------------------------------------------------------
# CaseSpec
# ---------------------------------------------------------------------------

class TestCaseSpec:
    def test_out_dir_coerced_to_path(self, tmp_path):
        spec = CaseSpec("id1", [], str(tmp_path))
        assert isinstance(spec.out_dir, Path)
        assert spec.out_dir == tmp_path

    def test_params_defaults_to_empty_dict(self, tmp_path):
        spec = CaseSpec("id2", [], tmp_path)
        assert spec.params == {}

    def test_params_preserved(self, tmp_path):
        spec = CaseSpec("id3", [], tmp_path, {"num_diffn_timesteps": 50})
        assert spec.params["num_diffn_timesteps"] == 50


# ---------------------------------------------------------------------------
# collate_results
# ---------------------------------------------------------------------------

class TestCollateResults:
    def _make_spec(self, cid: str) -> CaseSpec:
        return CaseSpec(cid, [], Path(f"/tmp/{cid}"))

    def _make_result(self, cid: str, gpu: int = 0, elapsed: float = 1.0) -> CaseResult:
        return CaseResult(cid, "ok", Path(f"/tmp/{cid}"), gpu, elapsed)

    def test_preserves_input_order(self):
        specs = [self._make_spec(c) for c in ["a", "b", "c"]]
        rmap = {
            "a": self._make_result("a", 0),
            "b": self._make_result("b", 1),
            "c": self._make_result("c", 0),
        }
        ordered = collate_results(rmap, specs)
        assert [r.case_id for r in ordered] == ["a", "b", "c"]

    def test_reversal_still_gives_input_order(self):
        specs = [self._make_spec("x"), self._make_spec("y")]
        rmap = {
            "y": self._make_result("y", 1),
            "x": self._make_result("x", 0),
        }
        ordered = collate_results(rmap, specs)
        assert ordered[0].case_id == "x"
        assert ordered[1].case_id == "y"

    def test_missing_result_becomes_sentinel_error(self):
        specs = [self._make_spec("a"), self._make_spec("b")]
        rmap = {"a": self._make_result("a")}
        ordered = collate_results(rmap, specs)
        assert ordered[0].status == "ok"
        assert ordered[1].status == "error"
        assert ordered[1].case_id == "b"
        assert "no result recorded" in (ordered[1].error or "")

    def test_empty_cases_returns_empty(self):
        assert collate_results({}, []) == []

    def test_single_case(self):
        specs = [self._make_spec("solo")]
        r = self._make_result("solo")
        assert collate_results({"solo": r}, specs) == [r]


# ---------------------------------------------------------------------------
# _header_to_entity
# ---------------------------------------------------------------------------

class TestHeaderToEntity:
    def test_pipe_separated(self):
        e = _header_to_entity("protein|mychain", "ACDEF")
        assert e["type"] == "protein"
        assert e["name"] == "mychain"
        assert e["sequence"] == "ACDEF"

    def test_ligand_type(self):
        e = _header_to_entity("ligand|ATP", "ATP")
        assert e["type"] == "ligand"

    def test_no_pipe_uses_header_as_name(self):
        e = _header_to_entity("myseq", "GGGGG")
        assert e["name"] == "myseq"
        assert e["type"] == "myseq"  # no pipe => type = header

    def test_chirality_defaults_to_L(self):
        e = _header_to_entity("protein|x", "A")
        assert e["chirality"] == "L"


# ---------------------------------------------------------------------------
# _build_cases_from_fastas
# ---------------------------------------------------------------------------

class TestBuildCasesFromFastas:
    def test_single_fasta_two_chains(self, tmp_path):
        fa = tmp_path / "test.fasta"
        fa.write_text(">protein|receptor\nACDEFGHIK\n>protein|binder\nGGGGG\n")
        cases = _build_cases_from_fastas([str(fa)], tmp_path / "out", {})
        assert len(cases) == 1
        assert cases[0].case_id == "test"
        assert len(cases[0].entities) == 2
        assert cases[0].entities[0]["name"] == "receptor"
        assert cases[0].entities[1]["sequence"] == "GGGGG"

    def test_multiple_fastas_produce_multiple_cases(self, tmp_path):
        for name in ["case1", "case2"]:
            f = tmp_path / f"{name}.fasta"
            f.write_text(f">protein|{name}\nACDEF\n")
        cases = _build_cases_from_fastas(
            [str(tmp_path / "case1.fasta"), str(tmp_path / "case2.fasta")],
            tmp_path / "out", {}
        )
        assert len(cases) == 2
        assert {c.case_id for c in cases} == {"case1", "case2"}

    def test_params_passed_through(self, tmp_path):
        fa = tmp_path / "x.fasta"
        fa.write_text(">protein|x\nACDEF\n")
        cases = _build_cases_from_fastas([str(fa)], tmp_path, {"num_diffn_timesteps": 50})
        assert cases[0].params["num_diffn_timesteps"] == 50

    def test_out_dir_per_case_is_subdir(self, tmp_path):
        fa = tmp_path / "mycase.fasta"
        fa.write_text(">protein|x\nACDEF\n")
        cases = _build_cases_from_fastas([str(fa)], tmp_path / "out", {})
        assert cases[0].out_dir == tmp_path / "out" / "mycase"


# ---------------------------------------------------------------------------
# run_cases with mock backend (no GPU / chai needed)
# ---------------------------------------------------------------------------

class TestRunCasesMocked:
    """Verify queue dispatch, result collation and OOM-restart logic without GPU."""

    def _fake_worker_script(self, tmp_path: Path) -> str:
        """Write a tiny Python script that echoes JSON back (simulates the worker)."""
        script = tmp_path / "_fake_worker.py"
        script.write_text(
            "import sys, json, time\n"
            "gpu_idx = int(sys.argv[1])\n"
            "_proto = sys.stdout\n"
            "sys.stdout = sys.stderr\n"
            "for line in iter(sys.stdin.readline, ''):\n"
            "    line = line.strip()\n"
            "    if not line: continue\n"
            "    spec = json.loads(line)\n"
            "    time.sleep(0.05)\n"
            "    _proto.write(json.dumps({\n"
            "        'status': 'ok',\n"
            "        'case_id': spec['case_id'],\n"
            "        'gpu_idx': gpu_idx,\n"
            "        'elapsed_s': 0.05,\n"
            "        'load_counter': 1,\n"
            "    }) + '\\n')\n"
            "    _proto.flush()\n"
        )
        return str(script)

    def test_all_cases_complete(self, tmp_path):
        """Two cases across two fake GPU workers all complete."""
        import importlib, types

        # We need to patch _WORKER_SCRIPT in run_parallel so it uses our fake
        fake_script = self._fake_worker_script(tmp_path)

        cases = [
            CaseSpec("c0", [{"type": "protein", "name": "x", "sequence": "A", "chirality": "L"}],
                     tmp_path / "c0"),
            CaseSpec("c1", [{"type": "protein", "name": "y", "sequence": "G", "chirality": "L"}],
                     tmp_path / "c1"),
        ]

        from scripts import run_parallel as rp
        # Patch the worker script written to disk with our fake
        orig_script = rp._WORKER_SCRIPT
        rp._WORKER_SCRIPT = open(fake_script).read()
        try:
            results = rp.run_cases(cases, n_gpus=2, workers_per_gpu=1,
                                   python_exe=sys.executable,
                                   work_dir=str(Path(__file__).parent.parent))
        finally:
            rp._WORKER_SCRIPT = orig_script

        assert len(results) == 2
        assert all(r.status == "ok" for r in results), \
            [(r.case_id, r.status, r.error) for r in results]
        # Results in input order
        assert results[0].case_id == "c0"
        assert results[1].case_id == "c1"

    def test_results_in_input_order(self, tmp_path):
        """Four cases dispatched to 2 GPUs; results come back ordered by input."""
        fake_script = self._fake_worker_script(tmp_path)
        cases = [CaseSpec(f"case{i}", [], tmp_path / f"case{i}") for i in range(4)]

        from scripts import run_parallel as rp
        orig_script = rp._WORKER_SCRIPT
        rp._WORKER_SCRIPT = open(fake_script).read()
        try:
            results = rp.run_cases(cases, n_gpus=2, workers_per_gpu=1,
                                   python_exe=sys.executable,
                                   work_dir=str(Path(__file__).parent.parent))
        finally:
            rp._WORKER_SCRIPT = orig_script

        assert [r.case_id for r in results] == [f"case{i}" for i in range(4)]

    def test_empty_cases_returns_empty(self):
        from scripts.run_parallel import run_cases
        assert run_cases([], n_gpus=2) == []

    def _crash_then_ok_worker_script(self, tmp_path: Path) -> str:
        """Worker that crashes (exits immediately) on the first case it receives,
        then on a fresh spawn handles the second (re-queued) case normally.

        Simulates OOM / worker death: the first spawn exits without writing any
        JSON response (EOF on stdout).  The consumer must re-queue and restart;
        after restart the re-queued case must complete as "ok".
        """
        # The crash-script reads ONE line and exits immediately (no JSON response).
        # After restart (new process), all subsequent cases are handled normally.
        # We use a shared counter file to distinguish first vs subsequent spawns.
        counter_file = tmp_path / "_spawn_count.txt"
        counter_file.write_text("0")

        script = tmp_path / "_crash_worker.py"
        script.write_text(
            f"import sys, json, time\n"
            f"from pathlib import Path\n"
            f"gpu_idx = int(sys.argv[1])\n"
            f"_proto = sys.stdout\n"
            f"sys.stdout = sys.stderr\n"
            f"counter_file = Path({str(counter_file)!r})\n"
            f"count = int(counter_file.read_text().strip())\n"
            f"count += 1\n"
            f"counter_file.write_text(str(count))\n"
            f"if count == 1:\n"
            f"    # First spawn: read one line then exit without responding (simulates OOM)\n"
            f"    sys.stdin.readline()\n"
            f"    sys.exit(0)\n"
            f"# Subsequent spawns: handle all cases normally\n"
            f"for line in iter(sys.stdin.readline, ''):\n"
            f"    line = line.strip()\n"
            f"    if not line: continue\n"
            f"    spec = json.loads(line)\n"
            f"    time.sleep(0.02)\n"
            f"    _proto.write(json.dumps({{\n"
            f"        'status': 'ok',\n"
            f"        'case_id': spec['case_id'],\n"
            f"        'gpu_idx': gpu_idx,\n"
            f"        'elapsed_s': 0.02,\n"
            f"        'load_counter': 1,\n"
            f"    }}) + '\\n')\n"
            f"    _proto.flush()\n"
        )
        return str(script)

    def test_worker_death_requeues_and_case_not_dropped(self, tmp_path):
        """Simulate worker death on first case; re-queued case must complete — not be dropped.

        This is the regression test for the CRITICAL fix in run_parallel.py:
        when a single-GPU consumer catches RuntimeError (worker EOF → no response),
        it re-queues the case and used to 'break', silently losing the case.
        After the fix it 'continue's (with worker restart), so the case is processed.
        """
        crash_script = self._crash_then_ok_worker_script(tmp_path)

        # Single GPU, single worker — the crash drops one item and re-queues it.
        # With the old break, that item would be lost.  With the new continue it
        # should be picked up and processed by the restarted worker.
        cases = [
            CaseSpec("victim", [], tmp_path / "victim"),   # this one triggers the crash
            CaseSpec("safe",   [], tmp_path / "safe"),     # this one follows after
        ]

        from scripts import run_parallel as rp
        orig_script = rp._WORKER_SCRIPT
        rp._WORKER_SCRIPT = open(crash_script).read()
        try:
            results = rp.run_cases(
                cases, n_gpus=1, workers_per_gpu=1,
                python_exe=sys.executable,
                work_dir=str(Path(__file__).parent.parent),
            )
        finally:
            rp._WORKER_SCRIPT = orig_script

        # Both cases must appear (not dropped)
        case_ids = {r.case_id for r in results}
        assert "victim" in case_ids, (
            f"'victim' case was silently dropped after worker death! results={results}"
        )
        assert "safe" in case_ids, (
            f"'safe' case missing from results: {results}"
        )
        assert len(results) == 2, f"Expected 2 results, got {len(results)}: {results}"

        # Both must complete as "ok" (the restarted worker handles them)
        by_id = {r.case_id: r for r in results}
        assert by_id["victim"].status == "ok", (
            f"'victim' case re-queued after crash but did not complete: {by_id['victim']}"
        )
        assert by_id["safe"].status == "ok", (
            f"'safe' case did not complete: {by_id['safe']}"
        )
