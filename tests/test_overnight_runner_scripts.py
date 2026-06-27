"""Static checks for the overnight GPU runner scripts (no GPU, no network).

Guards the two ready-to-run launchers under scripts/ so they stay:
  * present and syntactically valid bash (`bash -n`),
  * host-agnostic (no literal /home/user — env-driven per scripts/run_design_smoke.sh),
  * writing under runs/overnight/<name>/,
  * and so the heavy full-200 re-check refuses to run without explicit opt-in.

These never launch the scripts; they only inspect/parse them.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# No gpu/network markers: these are pure static/parse checks, so they run under
# `-m "not gpu and not network"` by default (the host test command).

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

VARIANT_A = SCRIPTS / "run_variant_a_repilot.sh"
FULL200 = SCRIPTS / "run_full200_recheck.sh"
ALL_SCRIPTS = [VARIANT_A, FULL200]


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_script_exists(script: Path) -> None:
    assert script.is_file(), f"missing runner script: {script}"


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_script_is_valid_bash(script: Path) -> None:
    bash = shutil.which("bash")
    assert bash, "bash not found on PATH"
    proc = subprocess.run(
        [bash, "-n", str(script)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"bash -n failed for {script.name}:\n{proc.stderr}"


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_no_literal_home_user(script: Path) -> None:
    text = script.read_text()
    assert "/home/user" not in text, (
        f"{script.name} contains a literal /home/user path; use env vars "
        "(XENO_REPO_ROOT/CHAI_IMAGE/CHAI_WEIGHTS) like run_design_smoke.sh"
    )


def test_variant_a_writes_under_runs_overnight() -> None:
    assert "runs/overnight/variant_a_repilot" in VARIANT_A.read_text()


def test_full200_writes_under_runs_overnight() -> None:
    assert "runs/overnight/full200_recheck" in FULL200.read_text()


def test_variant_a_is_the_matched_apples_to_apples_a_pilot() -> None:
    text = VARIANT_A.read_text()
    # the only search difference vs the B pilot is --abc_variant a
    assert "--abc_variant a" in text
    # matched B-pilot budget knobs
    assert "--abc_cycles 20" in text
    assert "--colony_size 12" in text
    # same class as the B pilot: free macrocycle, no target
    assert "--binder_class cyclic" in text
    assert "--target_type none" in text


def test_full200_has_confirm_guard() -> None:
    text = FULL200.read_text()
    # refuses to run without explicit opt-in
    assert 'XENO_CONFIRM_FULL200:?' in text, (
        "run_full200_recheck.sh must guard on XENO_CONFIRM_FULL200 so it cannot "
        "be launched unattended"
    )
