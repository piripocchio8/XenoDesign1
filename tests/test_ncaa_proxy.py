import pytest
from xenodesign.ncaa_proxy import proxy_for, CONFORMATIONAL_PROXY


def test_known_proxies_grounded_in_mondet():
    assert proxy_for("MSE") == "MET"   # selenomethionine ~ Met (identical distribution)
    assert proxy_for("AIB") == "ALA"   # alpha-aminoisobutyric, alpha-helical
    assert proxy_for("SEC") == "CYS"
    assert proxy_for("PYL") == "LYS"
    assert proxy_for("HYP") == "PRO"
    assert proxy_for("SEP") == "SER"   # phosphoserine
    assert proxy_for("TPO") == "THR"
    assert proxy_for("PTR") == "TYR"


def test_case_insensitive():
    assert proxy_for("mse") == "MET"


def test_unknown_returns_none():
    assert proxy_for("ZZZ") is None


def test_proxy_targets_are_canonical_three_letter():
    from xenodesign.io_spec import AA1_TO_AA3
    canonical = set(AA1_TO_AA3.values())
    for target in CONFORMATIONAL_PROXY.values():
        assert target in canonical
