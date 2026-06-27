"""HW-4: run_design_smoke.sh must be host-agnostic and gate the compose-label workaround."""
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_design_smoke.sh"


def test_smoke_script_no_hardcoded_host():
    text = SCRIPT.read_text()
    lines = text.splitlines()

    # No literal host paths / dev-box image tag.
    assert "/home/user" not in text, "found literal /home/user host path"
    assert "gradio_design-gradio-design" not in text, "found literal dev-box image tag"

    # The compose-label workaround lines must only appear inside the guard branch.
    guard = "XENO_COMPOSE_GUARD"
    label_idxs = [i for i, ln in enumerate(lines) if "com.docker.compose" in ln]
    assert label_idxs, "expected the compose --label lines to still exist (behind a guard)"
    guard_idxs = [i for i, ln in enumerate(lines) if guard in ln]
    assert guard_idxs, "expected an XENO_COMPOSE_GUARD guard branch"

    # Every label line must come after the guard opening and before its 'fi'.
    guard_open = min(guard_idxs)
    fi_idxs = [i for i, ln in enumerate(lines) if ln.strip() == "fi" and i > guard_open]
    assert fi_idxs, "guard branch must be closed with 'fi'"
    guard_close = min(fi_idxs)
    for i in label_idxs:
        assert guard_open < i < guard_close, (
            f"--label line at {i} is outside the XENO_COMPOSE_GUARD branch "
            f"({guard_open}..{guard_close})"
        )
