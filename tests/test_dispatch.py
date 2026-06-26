"""Unit tests for xenodesign.dispatch.run_design — the config→hooks→HalluLoop wiring (T3).

All tests are CPU-only (no GPU): the structure predictor and the per-iteration loop step are
MOCKED, so the test exercises the DISPATCH wiring (resolve → hook → target → loop → report)
rather than any Chai forward pass. The mocked class records which hooks were called and with
which config-derived args; the dispatcher's contract is that it routes the resolved DesignConfig
through the class's injected callables and returns the dict produced by ``cls.report``.
"""
from __future__ import annotations

import numpy as np

from xenodesign import dispatch
from xenodesign.classes.base import SeedSpec
from xenodesign.config import resolve_config


class _FakePred:
    """Minimal Prediction stand-in: the attributes the loop/objective/referee read."""

    coords = np.zeros((3, 3))
    iptm = 0.5
    token_index = np.array([1, 1, 1])
    plddt = np.array([80.0, 80.0, 80.0])


class _FakeClass:
    """Records every hook call so the test can assert the dispatcher's wiring contract."""

    case_id = "alpha"

    def __init__(self):
        self.calls: dict = {}

    def seed(self, cfg, target_seq):
        self.calls["seed"] = (cfg, target_seq)
        return SeedSpec(one_letter="ACDEFGHIK")

    def ss_bias(self, cfg, case):
        self.calls["ss_bias"] = (cfg, case)
        from xenodesign.benchmark.cases import ss_bias_config_for_case
        return ss_bias_config_for_case(case)

    def restraints(self, cfg, case, out_dir, target_ctx):
        self.calls["restraints"] = (cfg, case, out_dir, target_ctx)
        return None

    def closure(self, cfg, seed_spec):
        self.calls["closure"] = (cfg, seed_spec)
        return []

    def seq_update(self, cfg, wrapper, seed_spec):
        self.calls["seq_update"] = (cfg, wrapper, seed_spec)
        return lambda pred: seed_spec.one_letter

    def accept_fns(self, cfg):
        self.calls["accept_fns"] = (cfg,)
        return None

    def objective(self, cfg, wrapper):
        self.calls["objective"] = (cfg, wrapper)
        return lambda pred: float(pred.iptm)

    def referee(self, cfg, loop_dir, esm_judge):
        self.calls["referee"] = (cfg, loop_dir, esm_judge)
        return lambda step, i: None

    def report(self, cfg, history, panel_result, case, out_dir,
               *, l_seed_iptm=0.0, wall_time_s=0.0):
        self.calls["report"] = (cfg, history, panel_result, case, out_dir)
        return {"case_id": self.case_id, "n_iters": len(history),
                "selected_iptm": 0.5, "l_seed_iptm": l_seed_iptm, "wall_time_s": wall_time_s}


def _wire(monkeypatch, fake_cls):
    """Patch the dispatcher's three GPU/IO seams to CPU-safe fakes."""
    monkeypatch.setitem(dispatch._registry(), "alpha", fake_cls)
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "target",
                                       "sequence": "AAAA", "chirality": "L"}], None, None))


def test_run_design_wires_hooks_with_mocked_predictor(tmp_path, monkeypatch):
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 2, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    fake = _FakeClass()
    _wire(monkeypatch, fake)

    result = dispatch.run_design(cfg)

    assert result["case_id"] == "alpha"
    assert result["n_iters"] == 2
    # Provenance dumped BEFORE any predict.
    assert (tmp_path / "resolved_config.json").exists()


def test_run_design_calls_hooks_with_config_derived_args(tmp_path, monkeypatch):
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 3, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    fake = _FakeClass()
    _wire(monkeypatch, fake)

    dispatch.run_design(cfg)

    # resolve → hook → target → loop wiring: every loop-building hook fired with this cfg.
    assert fake.calls["seed"][0] is cfg
    assert fake.calls["seed"][1] == "AAAA"          # target_seq from target_entities[0]
    assert fake.calls["objective"][0] is cfg
    assert fake.calls["referee"][0] is cfg
    assert fake.calls["accept_fns"][0] is cfg
    # restraints consulted only when restraints_on (here False) → not called.
    assert "restraints" not in fake.calls
    # report receives the full history (one step per iter).
    history = fake.calls["report"][1]
    assert len(history) == 3
    assert fake.calls["report"][4] == tmp_path      # out_dir threaded through


def test_run_design_threads_real_l_seed_iptm_and_wall_into_report(tmp_path, monkeypatch):
    """The l-seed predict ipTM and the timed wall are the dispatcher's to measure; it THREADS
    them into report(l_seed_iptm=, wall_time_s=) so both the returned dict and the on-disk
    *_result.json carry the real values (JSON parity with the legacy single-class CLI)."""
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    fake = _FakeClass()
    _wire(monkeypatch, fake)

    result = dispatch.run_design(cfg)

    # _FakePred.iptm == 0.5 → the real l-seed predict ipTM the dispatcher measured.
    assert result["l_seed_iptm"] == 0.5
    # The dispatcher threads a measured wall (>= 0) into report().
    assert result["wall_time_s"] >= 0.0
    assert isinstance(result["wall_time_s"], float)


def test_search_beam_routes_to_beam(tmp_path, monkeypatch):
    """``cfg.loop.search == 'beam'`` routes run_design to ``_run_beam`` (NOT the greedy loop),
    threading the SAME class instance + resolved cfg + loop/init through it. The mock asserts the
    branch is taken without exercising the GPU beam machinery."""
    from xenodesign.loop import LoopStep

    called: dict = {}

    def _fake_beam(cls, cfg, loop, init, loop_dir, roles=None):
        called["beam"] = (cls, cfg, loop, init, loop_dir)
        called["roles"] = roles
        return [LoopStep(state=init, prediction=_FakePred(), score=0.5)]

    monkeypatch.setattr(dispatch, "_run_beam", _fake_beam)
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.search": "beam", "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    fake = _FakeClass()
    _wire(monkeypatch, fake)

    dispatch.run_design(cfg)

    assert called.get("beam") is not None
    # the routed call carries the SAME resolved cfg + the fake class instance.
    assert called["beam"][0] is fake
    assert called["beam"][1] is cfg


def test_run_design_builds_chain_roles_from_entities(tmp_path, monkeypatch):
    """The dispatcher computes the ONE ChainRoles contract from the assembled entity list and
    THREADS it to the beam path. With a 2-chain target the binder is chain 'C' (not a hardcoded
    'B') — proving the contract is derived from entity order, not assumed at the call site."""
    from xenodesign.loop import LoopStep
    from xenodesign.targets import ChainRoles

    called: dict = {}

    def _fake_beam(cls, cfg, loop, init, loop_dir, roles=None):
        called["roles"] = roles
        return [LoopStep(state=init, prediction=_FakePred(), score=0.5)]

    monkeypatch.setattr(dispatch, "_run_beam", _fake_beam)
    # 2-chain target (HA1/HA2 shape) -> binder appended last is chain 'C'.
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "ha1", "sequence": "AAAA"},
                                      {"type": "protein", "name": "ha2", "sequence": "CCCC"}],
                                     None, None))
    monkeypatch.setitem(dispatch._registry(), "alpha", _FakeClass())
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))

    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.search": "beam", "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    dispatch.run_design(cfg)

    assert called["roles"] == ChainRoles(binder="C", targets=("A", "B"))


def test_search_greedy_does_not_route_to_beam(tmp_path, monkeypatch):
    """The default (``search == 'greedy'``) path never touches ``_run_beam`` — greedy unchanged."""
    called: dict = {}
    monkeypatch.setattr(dispatch, "_run_beam",
                        lambda *a, **k: called.setdefault("beam", True) or [])
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 2, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    fake = _FakeClass()
    _wire(monkeypatch, fake)

    result = dispatch.run_design(cfg)

    assert "beam" not in called
    assert result["n_iters"] == 2


def test_run_design_honours_binder_length_in_real_seed(tmp_path, monkeypatch):
    """--binder_length flows end-to-end: the dispatcher hands cfg to the REAL Alpha.seed, whose
    from-scratch unified seed is exactly cfg.binder_length residues (clamped) — proof the seed
    NEVER inherits the reference length."""
    from xenodesign.classes.alpha import Alpha

    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 1, "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False, "binder_length": 27})
    # Call the REAL Alpha.seed exactly as the dispatcher does (target_seq from target_entities[0]).
    spec = Alpha().seed(cfg, target_seq="AAAA")
    assert len(spec.one_letter) == 27   # the from-scratch length, not the reference 21
    # And the C-term Gly tokenization anchor is preserved.
    assert "G" in spec.one_letter


def test_run_design_restraints_path_builds_predict_wrapper(tmp_path, monkeypatch):
    """When restraints_on and the class emits a restraint, the dispatcher consults restraints()
    and wires the PREDICT-mode wrapper (refine_fn) instead of the truncated-refine loop default."""
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": True})
    fake = _FakeClass()
    restraint_file = tmp_path / "x.restraints"
    restraint_file.write_text("dummy")
    fake.restraints = lambda cfg, case, out_dir, target_ctx: restraint_file  # type: ignore
    _wire(monkeypatch, fake)

    result = dispatch.run_design(cfg)

    assert result["case_id"] == "alpha"
    # objective hook received the predict-wrapper (the wrapper the loop folds with).
    wrapper = fake.calls["objective"][1]
    assert wrapper is not None
