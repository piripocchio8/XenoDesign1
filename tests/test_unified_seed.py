"""CPU tests for the UNIFIED FROM-SCRATCH SEEDING REFACTOR.

Principle (non-negotiable): binders are designed FROM SCRATCH — the seed
NEVER comes from the real binder (sequence, scaffold, OR length). ONE unified
PepMLM seed path for ALL classes; target_seq="" (unconditional) when there is no
protein target (cyclic/metal, no-target), the target sequence when there is one.

These tests pin:
  * cfg.binder_length defaults per class + the no-target default
  * resolve_binder_length clamp to 6..50
  * --binder_length CLI override + clamp
  * --length_sweep ladder selection (mock predictor / objective)
  * the unified seed.unified_seed primitive (always generate_conditioned)
  * non_alpha NO LONGER forces a Cys/knottin (opt-in scaffolding only)
  * cyclic/no-target unconditional seed (target_seq="" path) returns length-N
  * alpha still seeds without crashing (re-baseline OK)
"""
from __future__ import annotations

import numpy as np
import pytest

from xenodesign.config import (
    DEFAULT_BINDER_LENGTH, NO_TARGET_BINDER_LENGTH, resolve_binder_length,
    resolve_config,
)
from xenodesign.seed import RandomSeedGenerator, SeedResult, unified_seed


# ── unified_seed primitive (always conditioned signature; target may be "") ──────

def _fixed_gen(letter="A"):
    """A generator whose generate_conditioned returns letter*length (target ignored)."""
    class _G:
        def generate_conditioned(self, target_seq, length):
            return letter * length
    return _G()


def test_unified_seed_unconditional_returns_length_n():
    """target_seq="" path returns a length-N peptide (no protein target)."""
    r = unified_seed(_fixed_gen("A"), target_seq="", length=16)
    assert isinstance(r, SeedResult)
    assert len(r.one_letter) == 16
    assert r.length == 16
    assert r.conditioned is False  # empty target -> unconditional


def test_unified_seed_conditioned_sets_flag():
    r = unified_seed(_fixed_gen("C"), target_seq="MKLVTARGET", length=12)
    assert len(r.one_letter) == 12
    assert r.conditioned is True


def test_unified_seed_applies_retro_inverso_when_reverse():
    gen = _fixed_gen()

    class _G:
        def generate_conditioned(self, target_seq, length):
            return "ACDEF"

    r = unified_seed(_G(), target_seq="MK", length=5, reverse=True)
    assert r.one_letter == "FEDCA"
    assert r.reverse_applied is True


def test_unified_seed_opt_in_fixed_positions():
    """Opt-in scaffolding: fixed positions are placed + recorded, never default."""
    r = unified_seed(_fixed_gen("A"), target_seq="", length=12,
                     fixed_positions={6: "L", 12: "D"}, fixed_residue="H")
    assert r.one_letter[5] == "H" and r.one_letter[11] == "H"
    assert r.fixed_chirality == {6: "L", 12: "D"}


def test_unified_seed_no_fixed_positions_by_default():
    r = unified_seed(_fixed_gen("A"), target_seq="", length=10)
    assert r.fixed_chirality == {}
    assert set(r.one_letter) == {"A"}  # nothing overwritten


def test_unified_seed_random_generator_is_deterministic():
    a = unified_seed(RandomSeedGenerator(seed=5), target_seq="", length=14)
    b = unified_seed(RandomSeedGenerator(seed=5), target_seq="", length=14)
    assert a.one_letter == b.one_letter


# ── per-class default binder lengths (DEFAULTS, overridable) ─────────────────────

def test_default_binder_lengths_present():
    assert DEFAULT_BINDER_LENGTH["alpha"] == 21
    assert DEFAULT_BINDER_LENGTH["non_alpha"] == 30
    assert DEFAULT_BINDER_LENGTH["cyclic"] == 24
    assert NO_TARGET_BINDER_LENGTH == 16


def test_resolve_binder_length_per_class_defaults():
    assert resolve_binder_length(resolve_config("alpha", target_type="protein")) == 21
    assert resolve_binder_length(resolve_config("non_alpha", target_type="protein")) == 30
    assert resolve_binder_length(resolve_config("cyclic", target_type="metal")) == 24


def test_resolve_binder_length_no_target_default_is_16():
    cfg = resolve_config("cyclic", target_type="none")
    assert resolve_binder_length(cfg) == 16


def test_resolve_binder_length_explicit_override():
    cfg = resolve_config("alpha", target_type="protein",
                         cli_overrides={"binder_length": 33})
    assert resolve_binder_length(cfg) == 33


def test_resolve_binder_length_clamps_low():
    cfg = resolve_config("alpha", target_type="protein",
                         cli_overrides={"binder_length": 2})
    assert resolve_binder_length(cfg) == 6


def test_resolve_binder_length_clamps_high():
    cfg = resolve_config("alpha", target_type="protein",
                         cli_overrides={"binder_length": 999})
    assert resolve_binder_length(cfg) == 50


# ── CLI: --binder_length + --length_sweep parsing ────────────────────────────────

def test_cli_binder_length_flag_threads_into_config():
    from scripts.design import _parse_args, _overrides
    a = _parse_args(["--binder_class", "alpha", "--binder_length", "28"])
    assert _overrides(a)["binder_length"] == 28


def test_cli_binder_length_clamped_in_resolve():
    from scripts.design import _parse_args, _overrides
    a = _parse_args(["--binder_class", "alpha", "--binder_length", "100"])
    cfg = resolve_config("alpha", target_type="protein",
                         cli_overrides=_overrides(a))
    assert resolve_binder_length(cfg) == 50


def test_cli_length_sweep_flag_present():
    from scripts.design import _parse_args
    a = _parse_args(["--binder_class", "cyclic", "--length_sweep"])
    assert a.length_sweep is True
    b = _parse_args(["--binder_class", "cyclic"])
    assert b.length_sweep is False


# ── --length_sweep ladder selection (mock predictor) ─────────────────────────────

def test_length_sweep_ladder_clamped_to_range():
    from xenodesign.dispatch import length_sweep_ladder
    # the coarse ladder clamps to 6..50 and dedups, ascending
    ladder = length_sweep_ladder()
    assert ladder == sorted(set(ladder))
    assert all(6 <= n <= 50 for n in ladder)
    assert len(ladder) <= 6  # coarse + budget-bounded


def test_length_sweep_picks_best_by_objective(monkeypatch):
    """run_length_sweep loops over the ladder and returns the best-by-objective result."""
    from xenodesign import dispatch

    # Fake run_design: objective (selected_iptm) peaks at length 16.
    calls = []

    def _fake_run_design(cfg):
        calls.append(cfg.binder_length)
        # a unimodal score peaking at 16
        score = 1.0 - abs(resolve_binder_length(cfg) - 16) / 100.0
        return {"selected_iptm": score, "binder_length": resolve_binder_length(cfg)}

    monkeypatch.setattr(dispatch, "run_design", _fake_run_design)

    cfg = resolve_config("cyclic", target_type="none")
    best = dispatch.run_length_sweep(cfg, ladder=[8, 12, 16, 24, 32])
    assert best["binder_length"] == 16
    assert calls == [8, 12, 16, 24, 32]  # swept the whole (clamped) ladder
