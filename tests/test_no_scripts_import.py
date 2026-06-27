"""MOD-1 guard: the xenodesign package must not import its shared CIF/backend
plumbing from ``scripts.design_demo`` (the inverted dependency the audit flags as
the #1 readability/modularity blocker — §2.1).

Walks every ``xenodesign/**/*.py`` and asserts none import from
``scripts.design_demo``. Those helpers (``_best_cif_path``, ``_all_atoms_from_chain``,
``_backbone_array_from_residues``, ``_chirality_violation_frac_from_cif``,
``_LoopBackendWrapper``, ``_PredictBackendWrapper``) now live in
``xenodesign.cif_io`` / ``xenodesign.backends.wrappers``; ``scripts/design_demo.py``
re-exports them for its own CLI.
"""
from __future__ import annotations

import pathlib
import re

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent / "xenodesign"

# Matches `from scripts.design_demo import ...` and `import scripts.design_demo`.
_BAD = re.compile(r"(from\s+scripts\.design_demo\b|import\s+scripts\.design_demo\b)")


def test_no_package_imports_from_scripts():
    offenders = []
    for path in _PKG_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            # ignore comments/docstring mentions: only flag actual import statements
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _BAD.search(stripped) and (
                stripped.startswith("from ") or stripped.startswith("import ")
            ):
                offenders.append(f"{path.relative_to(_PKG_ROOT.parent)}:{lineno}: {stripped}")
    assert not offenders, (
        "xenodesign package must not import from scripts.design_demo "
        "(move plumbing into the package, re-export in the demo). Offenders:\n"
        + "\n".join(offenders)
    )
