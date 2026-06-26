import pytest
from xenodesign.classes.base import BinderClass, SeedSpec, CLASS_REGISTRY


def test_seedspec_defaults():
    s = SeedSpec(one_letter="ACDEFG")
    assert s.one_letter == "ACDEFG"
    assert s.fixed_chirality == {}
    assert s.cys_positions == ()


def test_registry_maps_cli_axis_to_case_id():
    assert set(CLASS_REGISTRY) == {"alpha", "non_alpha", "cyclic"}
    assert CLASS_REGISTRY["non_alpha"].case_id == "nonalpha"   # underscore→registry key
    assert CLASS_REGISTRY["alpha"].case_id == "alpha"
    assert CLASS_REGISTRY["cyclic"].case_id == "cyclic"


def test_registry_entries_satisfy_protocol():
    for cls in CLASS_REGISTRY.values():
        for m in ("seed", "ss_bias", "restraints", "closure",
                  "accept_fns", "objective", "referee", "report"):
            assert callable(getattr(cls, m))
