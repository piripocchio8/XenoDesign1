# tests/test_seeding.py
import pytest

from xenodesign.benchmark.cases import get_case
from xenodesign.benchmark.seeding import (
    SEED_POLICIES, seed_policy, build_seed_for_case,
)
from xenodesign.seed import PepMLMSeedGenerator, RandomSeedGenerator, SeedResult


def _fake_pepmlm():
    return PepMLMSeedGenerator(
        generate_fn=lambda target_seq, length: ("ACDEFGHIKLMNPQRSTVWYG" * 3)[:length],
        reverse=True,
    )


def test_every_case_has_a_seed_policy():
    assert set(SEED_POLICIES) == {"alpha", "cyclic", "nonalpha"}


def test_alpha_policy_is_conditioned_retro():
    p = seed_policy("alpha")
    assert p["mode"] == "pepmlm_conditioned"
    assert p["reverse"] is True
    assert p["length"] == 21
    assert p.get("fixed_chirality_positions", {}) == {}


def test_cyclic_policy_is_unconditioned_with_his_positions():
    p = seed_policy("cyclic")
    assert p["mode"] == "unconditioned"
    assert p["reverse"] is False
    assert p["length"] == 24
    # full S2-symmetric 6UFA 24-mer: His 6/12/18/24, chirality L/D/L/D
    assert set(p["fixed_chirality_positions"]) == {6, 12, 18, 24}
    assert set(p["fixed_chirality_positions"].values()) <= {"D", "L"}


def test_build_alpha_seed_conditioned_and_retro_inverted():
    case = get_case("alpha")
    res = build_seed_for_case(case, generator=_fake_pepmlm(), target_seq="MKLVTARGET")
    assert isinstance(res, SeedResult)
    assert res.length == 21
    assert len(res.one_letter) == 21
    assert res.conditioned is True
    assert res.reverse_applied is True
    fwd = ("ACDEFGHIKLMNPQRSTVWYG" * 3)[:21]
    assert res.one_letter == fwd[::-1]


def test_build_nonalpha_seed_conditioned_len31():
    case = get_case("nonalpha")
    res = build_seed_for_case(case, generator=_fake_pepmlm(), target_seq="HAHAHA")
    assert res.length == 31 and len(res.one_letter) == 31
    assert res.conditioned is True and res.reverse_applied is True


def test_build_cyclic_seed_unconditioned_with_fixed_his():
    case = get_case("cyclic")
    res = build_seed_for_case(case, generator=RandomSeedGenerator(seed=0))
    assert res.length == 24 and len(res.one_letter) == 24
    assert res.conditioned is False
    assert res.reverse_applied is False
    assert res.fixed_chirality == seed_policy("cyclic")["fixed_chirality_positions"]
    for pos in res.fixed_chirality:
        assert res.one_letter[pos - 1] == "H"


def test_build_conditioned_case_requires_target_seq():
    case = get_case("alpha")
    with pytest.raises(ValueError, match="target_seq"):
        build_seed_for_case(case, generator=_fake_pepmlm())  # no target_seq


def test_build_seed_for_case_does_not_import_torch():
    import sys
    build_seed_for_case(get_case("cyclic"), generator=RandomSeedGenerator(seed=1))
    assert "torch" not in sys.modules and "transformers" not in sys.modules


# ── P4: Tier-1 mirror self-consistency filter wired into seed selection ───────
import numpy as np
from xenodesign.benchmark.seeding import filter_seeds_by_mirror_consistency


def _seed(one_letter):
    return SeedResult(one_letter=one_letter, length=len(one_letter))


def test_mirror_filter_keeps_self_consistent_seed():
    rng = np.random.RandomState(0)
    a = rng.rand(10, 3)
    good_twin = a.copy(); good_twin[:, 0] *= -1.0      # exact mirror -> ~0 discrepancy
    bad_twin = rng.rand(10, 3)                          # unrelated -> high discrepancy
    candidates = [
        {"seed": _seed("AAAA"), "coords": a, "twin_coords": good_twin},
        {"seed": _seed("CCCC"), "coords": a, "twin_coords": bad_twin},
    ]
    kept = filter_seeds_by_mirror_consistency(candidates, axis=0, threshold=0.1)
    assert [s.one_letter for s in kept] == ["AAAA"]


def test_mirror_filter_keeps_seed_with_no_twin_yet():
    candidates = [{"seed": _seed("GGGG"), "coords": None, "twin_coords": None}]
    kept = filter_seeds_by_mirror_consistency(candidates, axis=0, threshold=0.1)
    assert [s.one_letter for s in kept] == ["GGGG"]


def test_mirror_filter_empty_input_is_empty():
    assert filter_seeds_by_mirror_consistency([], axis=0, threshold=0.1) == []
