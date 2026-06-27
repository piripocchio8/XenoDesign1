"""CPU tests: --ncaa_dict / --ncaa_top_x on DesignConfig + scripts/design.py wiring."""
from __future__ import annotations

import pytest

from xenodesign.config import DesignConfig, resolve_config


def test_default_ncaa_dict_is_d_only():
    cfg = DesignConfig()
    assert cfg.ncaa_dict == "d_only"
    assert cfg.ncaa_top_x == 20


def test_resolve_config_default_ncaa_dict():
    cfg = resolve_config("cyclic")
    assert cfg.ncaa_dict == "d_only"
    assert cfg.ncaa_top_x == 20


def test_resolve_config_accepts_valid_ncaa_dict():
    cfg = resolve_config("cyclic", cli_overrides={"ncaa_dict": "all", "ncaa_top_x": 5})
    assert cfg.ncaa_dict == "all"
    assert cfg.ncaa_top_x == 5


def test_resolve_config_rejects_invalid_ncaa_dict():
    with pytest.raises(ValueError):
        resolve_config("cyclic", cli_overrides={"ncaa_dict": "bogus"})


def test_resolve_config_rejects_negative_ncaa_top_x():
    with pytest.raises(ValueError):
        resolve_config("cyclic", cli_overrides={"ncaa_top_x": -1})


def test_design_cli_threads_ncaa_flags():
    from scripts.design import _parse_args, _overrides
    a = _parse_args(["--binder_class", "cyclic", "--ncaa_dict", "d_common", "--ncaa_top_x", "7"])
    o = _overrides(a)
    assert o["ncaa_dict"] == "d_common"
    assert o["ncaa_top_x"] == 7


def test_design_cli_omits_absent_ncaa_flags():
    from scripts.design import _parse_args, _overrides
    a = _parse_args(["--binder_class", "cyclic"])
    o = _overrides(a)
    assert "ncaa_dict" not in o
    assert "ncaa_top_x" not in o
