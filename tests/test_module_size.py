"""MOD-3 guardrail: the public ``classes/alpha.py`` (and ``classes/cyclic.py``) stay a thin
CONTRACT (BinderClass adapter + driver), with the restraint/seed/objective/referee internals
extracted into ``_alpha_internals.py`` / ``_cyclic_internals.py``.

A guardrail (not a hard rule): a sharp regrowth of the public module means internals are
creeping back in. Thresholds are set just above the post-split sizes (pre-split: alpha 1060 LOC,
cyclic 944 LOC) so the modules can absorb small contract-level edits without churn.
"""
from __future__ import annotations

import pathlib

_CLASSES = pathlib.Path(__file__).resolve().parent.parent / "xenodesign" / "classes"

# Post-split public-module size guardrails. Chosen just above the actual post-split LOC so a
# minor contract edit won't trip them, but a re-merge of the internals (hundreds of lines) will.
_ALPHA_MAX_LOC = 400
_CYCLIC_MAX_LOC = 400


def _loc(path: pathlib.Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_alpha_module_under_400_loc():
    loc = _loc(_CLASSES / "alpha.py")
    assert loc < _ALPHA_MAX_LOC, (
        f"classes/alpha.py is {loc} LOC (>= {_ALPHA_MAX_LOC}); the restraint/seed/objective/"
        "referee internals belong in classes/_alpha_internals.py (MOD-3)."
    )


def test_cyclic_module_under_400_loc():
    loc = _loc(_CLASSES / "cyclic.py")
    assert loc < _CYCLIC_MAX_LOC, (
        f"classes/cyclic.py is {loc} LOC (>= {_CYCLIC_MAX_LOC}); the intramolecular-objective/"
        "no-target-seed/Zn-restraint internals belong in classes/_cyclic_internals.py (MOD-3)."
    )
