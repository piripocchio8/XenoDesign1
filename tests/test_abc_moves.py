"""ABC T5 — structured chirality moves + priors (spec §4.5 / §5.4).

A chirality pattern is ``dict[int -> 'L'|'D']`` keyed by 0-based position. All
generators/perturbations are pure and deterministic given an injected ``rng``
(``random.Random``) — never module-level ``random``.
"""
import random

import pytest

from xenodesign.abc.moves import (
    alternating_pattern,
    flip_position,
    shift_boundary,
    shift_boundary_perturb,
    d_at_turn_apices,
    seed_chirality_pattern,
)


# ── priors / generators ───────────────────────────────────────────────────────

def test_alternating_pattern():
    assert alternating_pattern(4, start="L") == {0: "L", 1: "D", 2: "L", 3: "D"}


def test_alternating_pattern_start_D():
    assert alternating_pattern(3, start="D") == {0: "D", 1: "L", 2: "D"}


def test_alternating_pattern_default_start_is_L():
    assert alternating_pattern(2) == {0: "L", 1: "D"}


def test_alternating_pattern_rejects_bad_start():
    with pytest.raises(ValueError):
        alternating_pattern(3, start="X")


def test_d_at_turn_apices_sets_only_apices_D():
    # turn apices are D; everything else L.
    assert d_at_turn_apices(6, apices=[2, 4]) == {
        0: "L", 1: "L", 2: "D", 3: "L", 4: "D", 5: "L",
    }


def test_d_at_turn_apices_ignores_out_of_range():
    assert d_at_turn_apices(3, apices=[1, 9]) == {0: "L", 1: "D", 2: "L"}


def test_shift_boundary_generator():
    # boundary at k → positions <k are L, ≥k are D.
    assert shift_boundary(3, k=1) == {0: "L", 1: "D", 2: "D"}


def test_shift_boundary_k0_all_D():
    assert shift_boundary(3, k=0) == {0: "D", 1: "D", 2: "D"}


def test_shift_boundary_kn_all_L():
    assert shift_boundary(3, k=3) == {0: "L", 1: "L", 2: "L"}


# ── perturbations (the bees' moves) ────────────────────────────────────────────

def test_flip_position_swaps_one():
    p = {0: "L", 1: "D", 2: "L"}
    assert flip_position(p, 1) == {0: "L", 1: "L", 2: "L"}


def test_flip_position_returns_new_dict():
    p = {0: "L", 1: "D"}
    out = flip_position(p, 0)
    assert out == {0: "D", 1: "D"}
    assert p == {0: "L", 1: "D"}  # input untouched


def test_flip_position_out_of_range_raises():
    with pytest.raises(KeyError):
        flip_position({0: "L"}, 5)


def test_shift_boundary_perturb_moves_the_LD_split():
    # Given a pattern with an L→D boundary, perturbing nudges it by ±1.
    p = {0: "L", 1: "L", 2: "D", 3: "D"}
    out = shift_boundary_perturb(p, rng=random.Random(0))
    assert set(out.values()) <= {"L", "D"} and len(out) == 4
    # it is still a contiguous L*/D* split (a *shifted* boundary), not arbitrary
    ls = [i for i in sorted(out) if out[i] == "L"]
    ds = [i for i in sorted(out) if out[i] == "D"]
    assert ls == list(range(len(ls)))               # all L's are a prefix
    assert ds == list(range(len(ls), len(out)))     # all D's are the suffix
    assert out != p                                  # boundary actually moved


def test_shift_boundary_perturb_deterministic_given_rng():
    p = {0: "L", 1: "L", 2: "D", 3: "D"}
    a = shift_boundary_perturb(p, rng=random.Random(7))
    b = shift_boundary_perturb(p, rng=random.Random(7))
    assert a == b


# ── structured random seed ─────────────────────────────────────────────────────

def test_seed_respects_required_handedness():
    # metal-coordinator positions have a REQUIRED handedness priors must honour.
    p = seed_chirality_pattern(6, required={2: "D", 4: "L"}, rng=random.Random(0))
    assert p[2] == "D" and p[4] == "L" and len(p) == 6
    assert set(p.values()) <= {"L", "D"}


def test_seed_is_deterministic_given_rng():
    a = seed_chirality_pattern(8, rng=random.Random(3))
    b = seed_chirality_pattern(8, rng=random.Random(3))
    assert a == b


def test_seed_varies_with_rng_seed():
    # Different seeds give (with overwhelming probability) different draws —
    # i.e. it is NOT a constant pattern.
    draws = {
        tuple(seed_chirality_pattern(12, rng=random.Random(s)).items())
        for s in range(8)
    }
    assert len(draws) > 1


def test_seed_required_overrides_even_when_prior_would_differ():
    # Whatever the prior draws, required positions are forced.
    for s in range(5):
        p = seed_chirality_pattern(10, required={0: "D", 9: "D"}, rng=random.Random(s))
        assert p[0] == "D" and p[9] == "D"


# ── Part D: frozen coordinator positions are never chirality-mutated ─────────────
def test_perturb_chirality_never_flips_frozen_positions():
    from xenodesign.abc.engine import _perturb_chirality

    n = 24
    frozen = {5, 11, 17, 23}
    # seed pattern with a fixed handedness at the frozen (coordinator) positions.
    pattern = {i: "L" for i in range(n)}
    for i in frozen:
        pattern[i] = "D"
    rng = random.Random(0)
    for _ in range(2000):
        out = _perturb_chirality(pattern, rng, frozen=frozen)
        for i in frozen:
            assert out[i] == pattern[i]   # coordinator handedness never changes
