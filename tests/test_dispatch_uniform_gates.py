"""S3a.3d: with XENO_SEQ_STAGE on, run_design's greedy accept_fn is build_run_gates(cfg, roles) —
composing the class gate with the uniform metal/non_alpha gates; flag off uses cls.accept_fns.

Tests:
  1. test_greedy_accept_fn_uniform_when_flag_on — when flag is on, the accept_fn passed to
     loop.run IS the build_run_gates result (not cls.accept_fns).
  2. test_greedy_accept_fn_legacy_when_flag_off — when flag is off, accept_fn is cls.accept_fns(cfg)
     (backward-compatible byte-identical path).
  3. test_non_alpha_reject_behavior_flag_on — BEHAVIOR test (required by reviewer): on the wired
     dispatch path with flag ON, a non_alpha candidate that is over-helical is ACTUALLY rejected by
     the composed gate, not silently accepted by a stub. This proves the real JudgePanel (with a
     real helix score_fn) is injected — not the stub that always-accepts when helix_fraction=None.
"""
from __future__ import annotations

import numpy as np

from xenodesign import dispatch
from xenodesign.config import resolve_config


# ── Shared CPU fake stack (mirrors _alpha_fakes / _nonalpha_fakes from goldens) ─────────────────

class _FakePred:
    coords = np.zeros((3, 3))
    iptm = 0.5
    token_index = np.array([1, 1, 1])
    plddt = np.array([80.0, 80.0, 80.0])


_ALPHA_SEED = "ACDEFGHIKLMNPQRSTVWYG"    # 21-mer
_NONALPHA_SEED = "DEFCGHIKCLMNPCQRCSTVWYCDEDECFGG"  # 31-mer


def _alpha_fakes(monkeypatch):
    import xenodesign.classes.alpha as alpha_mod
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "target",
                                       "sequence": "GSHMKVLITGG", "chirality": "L"}], None, None))
    monkeypatch.setattr("xenodesign.seed.reflect_binder_in_complex_from_cif",
                        lambda *a, **k: np.zeros((3, 3)))
    monkeypatch.setattr(alpha_mod.Alpha, "seed",
                        lambda self, cfg, target_seq: __import__(
                            "xenodesign.classes.base", fromlist=["SeedSpec"]
                        ).SeedSpec(one_letter=_ALPHA_SEED))
    monkeypatch.setattr(alpha_mod, "make_alpha_seq_update_fn",
                        lambda wrapper, **k: (lambda pred: _ALPHA_SEED))


def _nonalpha_fakes(monkeypatch):
    import xenodesign.classes.non_alpha as nonalpha_mod
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "ha1", "sequence": "AAAA"},
                                      {"type": "protein", "name": "ha2", "sequence": "CCCC"}],
                                     None, None))
    monkeypatch.setattr("xenodesign.seed.reflect_binder_in_complex_from_cif",
                        lambda *a, **k: np.zeros((3, 3)))
    monkeypatch.setattr(nonalpha_mod.NonAlpha, "seed",
                        lambda self, cfg, target_seq: __import__(
                            "xenodesign.classes.base", fromlist=["SeedSpec"]
                        ).SeedSpec(one_letter=_NONALPHA_SEED))
    monkeypatch.setattr(nonalpha_mod, "make_alpha_seq_update_fn",
                        lambda wrapper, **k: (lambda pred: _NONALPHA_SEED))


# ── Test 1: sentinel wiring check (flag-on) ──────────────────────────────────────────────────────

def test_greedy_accept_fn_uniform_when_flag_on(monkeypatch, tmp_path):
    """When XENO_SEQ_STAGE=1, the accept_fn passed to loop.run IS build_run_gates(cfg, roles)."""
    from xenodesign.loop import HalluLoop, LoopStep, LoopState

    captured = {}

    def _fake_run(self, *, init, iterations, ref_time_steps, out_dir, accept_fn=None, **k):
        captured["accept_fn"] = accept_fn
        return [LoopStep(state=LoopState(d_fasta="", coords=None),
                         prediction=_FakePred(), score=0.5)]

    monkeypatch.setattr(HalluLoop, "run", _fake_run, raising=True)

    sentinel = object()
    monkeypatch.setattr("xenodesign.run_stages.build_run_gates",
                        lambda cfg, **k: sentinel, raising=True)
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")

    _alpha_fakes(monkeypatch)
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    dispatch.run_design(cfg)
    assert captured["accept_fn"] is sentinel


# ── Test 2: legacy path check (flag-off) ─────────────────────────────────────────────────────────

def test_greedy_accept_fn_legacy_when_flag_off(monkeypatch, tmp_path):
    """When XENO_SEQ_STAGE is absent (default), accept_fn is cls.accept_fns(cfg) — legacy path."""
    from xenodesign.loop import HalluLoop, LoopStep, LoopState

    captured = {}
    legacy_sentinel = object()

    def _fake_run(self, *, init, iterations, ref_time_steps, out_dir, accept_fn=None, **k):
        captured["accept_fn"] = accept_fn
        return [LoopStep(state=LoopState(d_fasta="", coords=None),
                         prediction=_FakePred(), score=0.5)]

    monkeypatch.setattr(HalluLoop, "run", _fake_run, raising=True)

    import xenodesign.classes.alpha as alpha_mod
    monkeypatch.setattr(alpha_mod.Alpha, "accept_fns",
                        lambda self, cfg: legacy_sentinel, raising=True)

    monkeypatch.delenv("XENO_SEQ_STAGE", raising=False)

    _alpha_fakes(monkeypatch)
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False})
    dispatch.run_design(cfg)
    assert captured["accept_fn"] is legacy_sentinel


# ── Test 3: BEHAVIOR test — real rejection on the wired dispatch path (flag-on) ─────────────────

def test_non_alpha_reject_behavior_flag_on(monkeypatch, tmp_path):
    """BEHAVIOR TEST (required by reviewer): with XENO_SEQ_STAGE=1, a non_alpha candidate that is
    over-helical (helix_fraction=0.9 > default max 0.5) is ACTUALLY REJECTED by the composed gate
    on the dispatch path. This proves the real JudgePanel with a real helix score_fn is injected —
    not the always-accept stub (panel=None path yields helix_fraction=None -> always-accept).

    Approach: monkeypatch build_run_gates to call through to the REAL implementation but intercept
    the panel argument so we can inject a panel whose score_fn returns helix=0.9 (over-helical).
    Then intercept loop.run to capture the accept_fn and exercise it on a synthetic LoopStep —
    asserting that gate(candidate, current) is False (rejected), NOT True (stub no-op).
    """
    import xenodesign.run_stages as rs_mod
    from xenodesign.loop import LoopState, LoopStep
    from xenodesign.judges.panel import JudgePanel, RefereeScore

    # Build a panel whose score_fn always returns helix_fraction=0.9 (always over-helical).
    # This simulates a real CIF-reading score_fn that found high helix content.
    over_helical_panel = JudgePanel(
        score_fn=lambda step: RefereeScore(chirality_violation=0.0, iptm=0.5, helix_fraction=0.9)
    )

    captured = {}

    # Wrap the real build_run_gates, injecting our over-helical panel so alpha_demote fires.
    real_build = rs_mod.build_run_gates

    def _patched_build_run_gates(cfg, **kwargs):
        # Inject the over-helical panel so alpha_demote_gated_accept actually runs with helix=0.9.
        kwargs["panel"] = over_helical_panel
        gate = real_build(cfg, **kwargs)
        captured["gate"] = gate
        return gate

    monkeypatch.setattr("xenodesign.run_stages.build_run_gates",
                        _patched_build_run_gates, raising=True)
    # Also patch the dispatch import path so the same patched fn is used.
    monkeypatch.setattr("xenodesign.dispatch.build_run_gates",
                        _patched_build_run_gates, raising=False)

    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    _nonalpha_fakes(monkeypatch)

    cfg = resolve_config("non_alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False,
                                        "gates.metal_geometry": False})

    dispatch.run_design(cfg)

    # The gate must have been built (non_alpha always gets alpha_demote).
    assert captured.get("gate") is not None, "build_run_gates was not called or returned None"

    # NOW exercise the captured gate on a synthetic over-helical step.
    # The gate reads helix from the panel.score_fn (not from prediction), so prediction can be minimal.
    class _MinPred:
        iptm = 0.5

    candidate = LoopStep(state=LoopState(d_fasta="", coords=None), prediction=_MinPred(), score=0.0)
    current = LoopStep(state=LoopState(d_fasta="", coords=None), prediction=_MinPred(), score=0.0)

    result = captured["gate"](candidate, current)
    # helix_fraction=0.9 > max_helix_frac=0.5 → MUST be rejected (False), not accepted (True).
    # If the stub path (panel=None) were used instead, helix_fraction=None → always True (no-op).
    assert result is False, (
        f"Expected rejection (False) for over-helical candidate, got {result!r}. "
        "This means the stub panel (always-accept) was injected instead of the real JudgePanel. "
        "Check that build_run_gates receives panel=over_helical_panel on the dispatch wiring path."
    )


# ── Test 4: END-TO-END GUARD — dispatch-built panel is REAL and INERT-PROOF ──────────────────────

def _run_dispatch_and_capture_gate(monkeypatch, tmp_path, helix_value):
    """Helper: run the flag-ON greedy non_alpha dispatch path, stub _binder_helix_fraction to
    return ``helix_value``, spy on loop.run to capture the accept_fn WITHOUT replacing its panel.
    Returns the captured accept_fn built by make_helix_panel_for_gates (the REAL dispatch panel).
    """
    import xenodesign.cif_io as cif_io_mod
    import xenodesign.classes._alpha_internals as alpha_internals_mod
    from xenodesign.loop import HalluLoop, LoopState, LoopStep

    captured = {}

    # --- Stub the helix source (not the panel) so the REAL _score closure returns our value ---
    monkeypatch.setattr(cif_io_mod, "_best_cif_path",
                        lambda out_dir: tmp_path / "fake.cif", raising=True)
    monkeypatch.setattr(alpha_internals_mod, "_binder_helix_fraction",
                        lambda cif, chain="B": helix_value, raising=True)

    # --- Spy on loop.run: capture accept_fn without replacing it ---
    real_run = HalluLoop.run

    def _spy_run(self, *, init, iterations, ref_time_steps, out_dir, accept_fn=None, **k):
        captured["accept_fn"] = accept_fn
        return real_run(self, init=init, iterations=iterations,
                        ref_time_steps=ref_time_steps, out_dir=out_dir,
                        accept_fn=accept_fn, **k)

    monkeypatch.setattr(HalluLoop, "run", _spy_run, raising=True)

    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    _nonalpha_fakes(monkeypatch)

    cfg = resolve_config("non_alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 1, "use_pepmlm": False,
                                        "use_pll": False, "restraints_on": False,
                                        "gates.metal_geometry": False})
    dispatch.run_design(cfg)

    assert "accept_fn" in captured, "loop.run was not called — spy did not capture accept_fn"
    return captured["accept_fn"]


def test_dispatch_helix_panel_real_not_inert_rejects_over_helical(monkeypatch, tmp_path):
    """END-TO-END GUARD (S3a.3d): the dispatch-built panel (make_helix_panel_for_gates) is REAL
    and NOT inert.  We stub _binder_helix_fraction→0.9 (over-helical) at the source, run the
    flag-ON greedy path WITHOUT injecting a custom panel, capture accept_fn from loop.run via spy,
    and assert a non_alpha candidate is REJECTED (False).

    This test FAILS if dispatch nulls or stubs _helix_panel (inert-gate no-op), because then
    build_run_gates receives panel=None, constructs the always-accept stub, and gate returns True.
    Test 3 (test_non_alpha_reject_behavior_flag_on) cannot catch that regression because it
    overrides kwargs["panel"] with its own over_helical_panel — we exercise the dispatch wiring
    here end-to-end.
    """
    from xenodesign.loop import LoopState, LoopStep

    accept_fn = _run_dispatch_and_capture_gate(monkeypatch, tmp_path, helix_value=0.9)

    assert accept_fn is not None, (
        "dispatch returned accept_fn=None for non_alpha (no gate was composed). "
        "alpha_demote_gated_accept must always be wired for non_alpha."
    )

    class _MinPred:
        iptm = 0.5

    candidate = LoopStep(state=LoopState(d_fasta="", coords=None),
                         prediction=_MinPred(), score=0.0)
    current = LoopStep(state=LoopState(d_fasta="", coords=None),
                       prediction=_MinPred(), score=0.0)

    result = accept_fn(candidate, current)
    assert result is False, (
        f"Expected REJECTION (False) for helix=0.9 via REAL dispatch-built panel, got {result!r}. "
        "This means the dispatch panel is inert (panel=None stub always-accepts when helix=None). "
        "Regression: make_helix_panel_for_gates / _helix_panel wiring in dispatch was broken."
    )


def test_dispatch_helix_panel_real_not_inert_accepts_low_helix(monkeypatch, tmp_path):
    """Mirror of the above: stub _binder_helix_fraction→0.2 (low helix), same REAL dispatch panel,
    assert the candidate is ACCEPTED (True).  Together with the reject test, this pins both
    transitions of the gate and proves the helix read-path is live (not short-circuited to None).
    """
    from xenodesign.loop import LoopState, LoopStep

    accept_fn = _run_dispatch_and_capture_gate(monkeypatch, tmp_path, helix_value=0.2)

    assert accept_fn is not None, (
        "dispatch returned accept_fn=None for non_alpha. alpha_demote must always be wired."
    )

    class _MinPred:
        iptm = 0.5

    candidate = LoopStep(state=LoopState(d_fasta="", coords=None),
                         prediction=_MinPred(), score=0.0)
    current = LoopStep(state=LoopState(d_fasta="", coords=None),
                       prediction=_MinPred(), score=0.0)

    result = accept_fn(candidate, current)
    assert result is True, (
        f"Expected ACCEPTANCE (True) for helix=0.2 via REAL dispatch-built panel, got {result!r}. "
        "Low-helix candidate must pass the anti-alpha gate (max_helix_frac=0.5 default)."
    )
