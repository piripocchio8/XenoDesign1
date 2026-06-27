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
                         cli_overrides={"mixed_chirality": "A", "abc.cycles": 2,
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
                         cli_overrides={"mixed_chirality": "B",
                                        "abc.cycles": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    dispatch.run_design(cfg)
    # Variant B returns a 3-RESIDUE identity (point-mutated). With the default --ncaa_dict
    # d_only palette now ON, a residue may be emitted as a (DXX) block, so count residues
    # (identity_tokens), not raw chars.
    from xenodesign.abc.moves import identity_tokens
    assert isinstance(seen["out"], str)
    assert len(identity_tokens(seen["out"])) == 3


def test_abc_passes_frozen_coordinators(monkeypatch):
    # track #2 / track #1 gap: declared coordinators must reach abc_search as `frozen` (0-based),
    # so the coordinator-chirality freeze (engine-side) actually activates for cyclic-metal ABC.
    captured = {}

    def fake_abc(init_pop, fitness_fn, design_fn=None, **k):
        captured["frozen"] = k.get("frozen")
        from xenodesign.abc.engine import FoodSource
        return FoodSource("AAAA", {0: "L", 1: "L", 2: "L", 3: "L"}, None, 0.5), []

    monkeypatch.setattr(dispatch, "abc_search", fake_abc, raising=False)
    _wire_predictor(monkeypatch)

    cfg = resolve_config("cyclic", target_type="none", out_dir="/tmp/xd_abc_frozen",
                         cli_overrides={"mixed_chirality": "A", "abc.cycles": 1,
                                        "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False})
    # Declare two coordinators (1-based positions 1 and 3) the way the CLI flag wiring stores them.
    cfg.restraint.params["coord_residues"] = [
        (1, "H", "HIS", "L"), (3, "H", "HIS", "D"),
    ]
    dispatch.run_design(cfg)
    assert captured["frozen"] == {0, 2}   # 1-based -> 0-based


def test_abc_variant_b_palette_reaches_design_fn(monkeypatch):
    from xenodesign.abc.moves import identity_tokens
    seen = {}

    def fake_abc(init_pop, fitness_fn, design_fn=None, **k):
        # Drive the design_fn enough times to observe an ncAA proposal from the palette.
        outs = [design_fn("AAAAA", {i: "L" for i in range(5)}) for _ in range(200)]
        seen["any_ncaa"] = any("(" in o for o in outs)
        seen["blocks_in_palette"] = all(
            tok[1:-1] in ("AIB", "ORN")
            for o in outs for tok in identity_tokens(o) if tok.startswith("(")
        )
        from xenodesign.abc.engine import FoodSource
        return FoodSource("AAAAA", {i: "L" for i in range(5)}, None, 0.6), []

    monkeypatch.setattr(dispatch, "abc_search", fake_abc, raising=False)
    _wire_predictor(monkeypatch)

    cfg = resolve_config("cyclic", target_type="none", out_dir="/tmp/xd_abc_pal",
                         cli_overrides={"mixed_chirality": "B",
                                        "abc.cycles": 1, "abc.ncaa_palette": ["AIB", "ORN"],
                                        "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False})
    dispatch.run_design(cfg)
    assert seen["any_ncaa"] is True
    assert seen["blocks_in_palette"] is True


def test_homochiral_class_defaults_to_no_abc(tmp_path, monkeypatch):
    # alpha defaults mixed_chirality="none" -> greedy loop, abc_search never called (decoupled
    # from --search; the old "homochiral rejection" of --search abc is gone).
    from tests.test_dispatch import _FakeClass, _wire

    def boom(*a, **k):
        raise AssertionError("abc_search should not run for homochiral default (none)")

    monkeypatch.setattr(dispatch, "abc_search", boom, raising=False)
    fake = _FakeClass()
    _wire(monkeypatch, fake)
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    assert cfg.mixed_chirality == "none"
    result = dispatch.run_design(cfg)
    assert result["case_id"] == "alpha"


def test_mixed_chirality_A_routes_to_abc_variant_a(monkeypatch):
    captured = {}

    def fake_abc(init_pop, fitness_fn, design_fn=None, **k):
        captured["called"] = True
        from xenodesign.abc.engine import FoodSource
        return FoodSource("AAA", {0: "D", 1: "L", 2: "D"}, None, 0.8), []

    monkeypatch.setattr(dispatch, "abc_search", fake_abc, raising=False)
    _wire_predictor(monkeypatch)

    cfg = resolve_config("cyclic", target_type="none", out_dir="/tmp/xd_mc_a",
                         cli_overrides={"mixed_chirality": "A", "abc.cycles": 1,
                                        "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False})
    result = dispatch.run_design(cfg)
    assert captured.get("called") is True
    assert result["search"] == "abc"
    assert result["abc_variant"] == "a"  # Variant A flag supersedes cfg.abc.variant


def test_mixed_chirality_B_routes_to_abc_variant_b(monkeypatch):
    seen = {}

    def fake_abc(init_pop, fitness_fn, design_fn=None, **k):
        # Variant B design_fn point-mutates the identity string.
        seen["out"] = design_fn("AAA", {0: "D", 1: "L", 2: "D"})
        from xenodesign.abc.engine import FoodSource
        return FoodSource("AAA", {0: "D", 1: "L", 2: "D"}, None, 0.7), []

    monkeypatch.setattr(dispatch, "abc_search", fake_abc, raising=False)
    _wire_predictor(monkeypatch)

    cfg = resolve_config("cyclic", target_type="none", out_dir="/tmp/xd_mc_b",
                         cli_overrides={"mixed_chirality": "B", "abc.cycles": 1,
                                        "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False})
    result = dispatch.run_design(cfg)
    assert result["abc_variant"] == "b"
    # 3 RESIDUES (the default d_only ncAA palette may emit a (DXX) block per residue).
    from xenodesign.abc.moves import identity_tokens
    assert isinstance(seen["out"], str)
    assert len(identity_tokens(seen["out"])) == 3


def test_mixed_chirality_flag_supersedes_cfg_abc_variant(monkeypatch):
    # The flag wins over cfg.abc.variant: mixed_chirality="A" runs Variant A even though
    # abc.variant was set to "b".
    def fake_abc(init_pop, fitness_fn, design_fn=None, **k):
        from xenodesign.abc.engine import FoodSource
        return FoodSource("AAA", {0: "D", 1: "L", 2: "D"}, None, 0.6), []

    monkeypatch.setattr(dispatch, "abc_search", fake_abc, raising=False)
    _wire_predictor(monkeypatch)

    cfg = resolve_config("cyclic", target_type="none", out_dir="/tmp/xd_mc_super",
                         cli_overrides={"mixed_chirality": "A", "abc.variant": "b",
                                        "abc.cycles": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    result = dispatch.run_design(cfg)
    assert result["abc_variant"] == "a"


def test_mixed_chirality_allowed_for_cyclic_metal(monkeypatch):
    # The old "cyclic + target_type=none ONLY" restriction is removed: cyclic+metal routes to ABC.
    def fake_abc(init_pop, fitness_fn, design_fn=None, **k):
        from xenodesign.abc.engine import FoodSource
        return FoodSource("AAAA", {i: "L" for i in range(4)}, None, 0.5), []

    monkeypatch.setattr(dispatch, "abc_search", fake_abc, raising=False)
    _wire_predictor(monkeypatch)
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "ligand", "name": "Zn",
                                       "smiles": "[Zn+2]"}], None, None))

    # cyclic preset defaults mixed_chirality="A" even for the metal target.
    cfg = resolve_config("cyclic", target_type="metal", out_dir="/tmp/xd_mc_metal",
                         cli_overrides={"abc.cycles": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    assert cfg.mixed_chirality == "A"
    result = dispatch.run_design(cfg)
    assert result["search"] == "abc"


def test_mixed_chirality_none_runs_greedy_not_abc(tmp_path, monkeypatch):
    # mixed_chirality="none" must NOT route to abc_search; it falls through to the greedy loop.
    # Reuse test_dispatch's full _FakeClass/_wire (which drives the greedy path on CPU).
    from tests.test_dispatch import _FakeClass, _wire

    def boom(*a, **k):
        raise AssertionError("abc_search should not be called for mixed_chirality=none")

    monkeypatch.setattr(dispatch, "abc_search", boom, raising=False)
    fake = _FakeClass()
    _wire(monkeypatch, fake)

    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    assert cfg.mixed_chirality == "none"
    result = dispatch.run_design(cfg)  # reaches the loop without calling abc_search
    assert result["case_id"] == "alpha"


def test_search_abc_backcompat_maps_to_mixed_chirality_A():
    # BACK-COMPAT: `--search abc` maps to --mixed_chirality A (b if --abc_variant b).
    import scripts.design as d
    a = d._parse_args(["--binder_class", "cyclic", "--target_type", "none", "--search", "abc"])
    o = d._overrides(a)
    assert o.get("mixed_chirality") == "A"
    assert "loop.search" not in o or o["loop.search"] != "abc"

    a2 = d._parse_args(["--binder_class", "cyclic", "--target_type", "none",
                        "--search", "abc", "--abc_variant", "b"])
    o2 = d._overrides(a2)
    assert o2.get("mixed_chirality") == "B"


def test_cli_parses_mixed_chirality_flag():
    import scripts.design as d
    a = d._parse_args(["--binder_class", "cyclic", "--target_type", "metal",
                       "--mixed_chirality", "B"])
    assert a.mixed_chirality == "B"
    o = d._overrides(a)
    assert o["mixed_chirality"] == "B"

    # Absent flag -> no override (preset wins).
    a2 = d._parse_args(["--binder_class", "cyclic", "--target_type", "metal"])
    assert "mixed_chirality" not in d._overrides(a2)


def test_cli_parses_abc_flags():
    import scripts.design as d
    a = d._parse_args(["--binder_class", "cyclic", "--target_type", "none",
                       "--search", "abc", "--abc_variant", "b", "--abc_cycles", "30",
                       "--colony_size", "12", "--scout_limit", "4"])
    assert a.search == "abc" and a.abc_variant == "b" and a.abc_cycles == 30
    assert a.colony_size == 12 and a.scout_limit == 4
    o = d._overrides(a)
    # `--search abc` is back-compat mapped to mixed_chirality (B here, via --abc_variant b),
    # NOT a literal loop.search="abc" override anymore.
    assert o.get("mixed_chirality") == "B"
    assert o.get("loop.search") != "abc"
    assert o["abc.variant"] == "b"
    assert o["abc.cycles"] == 30 and o["abc.colony_size"] == 12 and o["abc.scout_limit"] == 4
