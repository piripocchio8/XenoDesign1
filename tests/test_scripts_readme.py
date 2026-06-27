"""MOD-4: scripts/README.md must document the core driver scripts (drift guard)."""
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "scripts" / "README.md"

# The ~8 main drivers (Tier-1) — unified dispatcher, multi-GPU runner,
# the three per-class loop drivers, and the predict/score/select drivers.
CORE_DRIVERS = [
    "design.py",
    "run_parallel.py",
    "design_alpha.py",
    "design_cyclic.py",
    "design_nonalpha.py",
    "generate_and_select.py",
    "predict_complex.py",
    "score_complex.py",
]


def test_scripts_readme_lists_core_drivers():
    assert README.exists(), "scripts/README.md must exist"
    text = README.read_text()
    for name in CORE_DRIVERS:
        assert name in text, f"core driver {name} not documented in scripts/README.md"
