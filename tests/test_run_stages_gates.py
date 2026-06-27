"""S3a.3c: build_run_gates composes the right accept gates per (binder_class, target, gate-knobs)
and is None when none apply. alpha -> periodicity; non_alpha -> alpha-demote; metal -> MetalHawk;
pll_veto knob -> pLM gate. All AND-composed via compose_accept_fns (None-safe)."""
from __future__ import annotations

from xenodesign.config import resolve_config
from xenodesign.run_stages import build_run_gates


def test_alpha_gets_periodicity_when_enabled():
    cfg = resolve_config("alpha", target_type="protein",
                         cli_overrides={"gates.periodicity": True})
    assert build_run_gates(cfg, roles=None) is not None


def test_non_alpha_gets_alpha_demote():
    cfg = resolve_config("non_alpha", target_type="protein",
                         cli_overrides={"gates.metal_geometry": False})
    gate = build_run_gates(cfg, roles=None)
    assert gate is not None      # alpha-demote always on for the anti-alpha class


def test_metal_gets_metalhawk_when_enabled():
    cfg = resolve_config("cyclic", target_type="metal",
                         cli_overrides={"gates.metal_geometry": True,
                                        "restraint.params": {"coord_residues": [(6, "H", "HIS", "L", "ND1")]}})
    assert build_run_gates(cfg, roles=None) is not None


def test_no_gates_is_none():
    cfg = resolve_config("cyclic", target_type="none",
                         cli_overrides={"gates.periodicity": False, "gates.metal_geometry": False,
                                        "gates.pll_veto": False})
    assert build_run_gates(cfg, roles=None) is None
