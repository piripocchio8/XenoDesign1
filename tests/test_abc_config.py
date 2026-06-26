"""CPU tests for AbcKnobs config (xenodesign/config.py) — ABC tunables (spec §5.4).

The ABC tunables the dispatcher + pilot read: colony size, cycles, scout limit, chirality-move
rate, variant ∈ {a,b}, eval budget, and fitness_steps (= K*=10-25 fast Chai steps) plus the
pTM/termini objective weights (the just-decided 2026-06-25 driver). Existing config untouched.
"""
from __future__ import annotations

from xenodesign.config import AbcKnobs, resolve_config


def test_abc_knobs_defaults():
    k = AbcKnobs()
    assert k.colony_size > 0 and k.cycles > 0 and k.scout_limit > 0
    assert k.variant == "a"
    assert k.chai_eval_budget > 0
    # The objective: K*=10-25 fast steps, pTM-primary + termini-secondary (sum to 1.0).
    assert 10 <= k.fitness_steps <= 25
    assert k.w_ptm > k.w_termini
    assert abs((k.w_ptm + k.w_termini) - 1.0) < 1e-9


def test_resolve_sets_abc_overrides():
    cfg = resolve_config("cyclic", target_type="none",
                         cli_overrides={"abc.variant": "b", "abc.cycles": 30})
    assert cfg.abc.variant == "b" and cfg.abc.cycles == 30


def test_abc_knobs_present_on_all_presets():
    for cls in ("alpha", "non_alpha", "cyclic"):
        cfg = resolve_config(cls)
        assert isinstance(cfg.abc, AbcKnobs)
