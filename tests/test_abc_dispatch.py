"""CPU tests for the ``--search abc`` dispatch branch (xenodesign/dispatch.py) + CLI flags.

Mixed-chirality cases (cyclic + target_type=none) route through ``abc_search`` with the fast
fitness adapter + the chosen variant; homochiral classes (alpha / all-D non_alpha) are guarded
out. All CPU — the predictor and ``abc_search`` are mocked (mirrors tests/test_dispatch.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from xenodesign import dispatch
from xenodesign.config import resolve_config


class _FakePred:
    coords = np.zeros((3, 3))
    iptm = 0.5
    ptm = 0.8


def _wire_predictor(monkeypatch):
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr(dispatch, "target_entities", lambda cfg: ([], None, None))


def test_abc_branch_invoked_for_mixed_chirality(monkeypatch):
    called = {}

    def fake_abc(init_pop, fitness_fn, design_fn=None, **k):
        called["yes"] = True
        called["n_init"] = len(init_pop)
        called["has_design_fn"] = design_fn is not None
        from xenodesign.abc.engine import FoodSource
        best = FoodSource(identity="(DAL)(DAL)(DAL)",
                          chirality_pattern={0: "D", 1: "D", 2: "D"},
                          last_structure=None, nectar=0.9)
        return best, [{"cycle": 0, "best_nectar": 0.9}]

    monkeypatch.setattr(dispatch, "abc_search", fake_abc, raising=False)
    _wire_predictor(monkeypatch)

    cfg = resolve_config("cyclic", target_type="none", out_dir="/tmp/xd_abc_test",
                         cli_overrides={"loop.search": "abc", "abc.cycles": 2,
                                        "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False})
    result = dispatch.run_design(cfg)

    assert called.get("yes") is True
    assert called["has_design_fn"] is True
    assert result["search"] == "abc"
    assert result["selected_nectar"] == 0.9
    assert result["selected_d_fasta"] == "(DAL)(DAL)(DAL)"


def test_abc_variant_b_selected(monkeypatch):
    seen = {}

    def fake_abc(init_pop, fitness_fn, design_fn=None, **k):
        # Variant B design_fn mutates identity (not a per-position-handed FASTA emit).
        seen["out"] = design_fn("AAA", {0: "D", 1: "L", 2: "D"})
        from xenodesign.abc.engine import FoodSource
        return FoodSource("AAA", {0: "D", 1: "L", 2: "D"}, None, 0.7), []

    monkeypatch.setattr(dispatch, "abc_search", fake_abc, raising=False)
    _wire_predictor(monkeypatch)

    cfg = resolve_config("cyclic", target_type="none", out_dir="/tmp/xd_abc_b",
                         cli_overrides={"loop.search": "abc", "abc.variant": "b",
                                        "abc.cycles": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    dispatch.run_design(cfg)
    # Variant B returns a 3-char identity string (point-mutated), never a (DXX) block emit.
    assert isinstance(seen["out"], str) and len(seen["out"]) == 3
    assert "(" not in seen["out"]


def test_abc_rejects_homochiral_class(monkeypatch):
    _wire_predictor(monkeypatch)
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "t",
                                       "sequence": "AAAA"}], None, None))
    cfg = resolve_config("alpha", target_type="protein", out_dir="/tmp/xd_abc_homo",
                         cli_overrides={"loop.search": "abc", "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    with pytest.raises(ValueError, match="mixed-chirality"):
        dispatch.run_design(cfg)


def test_cli_parses_abc_flags():
    import scripts.design as d
    a = d._parse_args(["--binder_class", "cyclic", "--target_type", "none",
                       "--search", "abc", "--abc_variant", "b", "--abc_cycles", "30",
                       "--colony_size", "12", "--scout_limit", "4"])
    assert a.search == "abc" and a.abc_variant == "b" and a.abc_cycles == 30
    assert a.colony_size == 12 and a.scout_limit == 4
    o = d._overrides(a)
    assert o["loop.search"] == "abc" and o["abc.variant"] == "b"
    assert o["abc.cycles"] == 30 and o["abc.colony_size"] == 12 and o["abc.scout_limit"] == 4
