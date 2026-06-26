"""Structured chirality moves + priors for the ABC engine (spec §4.5 / §5.4).

A **chirality pattern** is ``dict[int -> 'L'|'D']`` keyed by 0-based position.
These functions are the bees' moves over the chirality axis: structured priors
(``alternating_pattern`` / ``d_at_turn_apices`` / ``shift_boundary``) and local
perturbations (``flip_position`` / ``shift_boundary_perturb``), plus a structured
random initializer (``seed_chirality_pattern``).

The search is **structured, not blind** — we lean on D-peptide stereochemistry
priors (alternating L/D, D at β-turn / loop apices, a single L→D boundary) and
always honour ``required`` handedness (e.g. a metal coordinator whose chirality
is fixed by the geometry). Everything is pure and deterministic given an injected
``random.Random`` — never module-level ``random`` — so the search reproduces
under a fixed ``rng_seed``.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence

ChiralityPattern = dict  # dict[int, str]  ('L'|'D' by 0-based position)

_VALID = ("L", "D")


def _check_label(label: str) -> str:
    up = label.upper()
    if up not in _VALID:
        raise ValueError(f"chirality label must be 'L' or 'D', got {label!r}")
    return up


# ── structured priors / generators ────────────────────────────────────────────

def alternating_pattern(n: int, start: str = "L") -> ChiralityPattern:
    """Alternating L/D pattern of length ``n`` beginning with ``start``."""
    start = _check_label(start)
    other = "D" if start == "L" else "L"
    return {i: (start if i % 2 == 0 else other) for i in range(n)}


def d_at_turn_apices(n: int, apices: Sequence[int]) -> ChiralityPattern:
    """All-L backbone with D only at the given turn/loop apex positions.

    A common motif: L mainchain with a D residue placed at a β-turn apex to flip
    the turn handedness. Apex indices outside ``range(n)`` are ignored.
    """
    apex_set = {i for i in apices if 0 <= i < n}
    return {i: ("D" if i in apex_set else "L") for i in range(n)}


def shift_boundary(n: int, k: int) -> ChiralityPattern:
    """Single L→D boundary at ``k``: positions ``< k`` are L, ``>= k`` are D.

    ``k == 0`` → all D; ``k == n`` → all L.
    """
    if not 0 <= k <= n:
        raise ValueError(f"boundary k must be in [0, {n}], got {k}")
    return {i: ("L" if i < k else "D") for i in range(n)}


# ── local perturbations (employed/onlooker moves) ─────────────────────────────

def flip_position(pattern: Mapping[int, str], i: int) -> ChiralityPattern:
    """Return a copy of ``pattern`` with position ``i`` flipped L<->D."""
    if i not in pattern:
        raise KeyError(f"position {i} not in pattern")
    out = dict(pattern)
    out[i] = "D" if out[i] == "L" else "L"
    return out


def shift_boundary_perturb(
    pattern: Mapping[int, str], rng: random.Random
) -> ChiralityPattern:
    """Nudge a contiguous L→D boundary by +/-1, returning a new pattern.

    Treats ``pattern`` as a single L-prefix / D-suffix split (``k`` = number of
    leading L's) and moves that boundary by one in a random direction, clamped to
    ``[0, n]``. If clamping would leave the boundary unmoved (already at an edge),
    it moves the other way so the move is always effective.
    """
    n = len(pattern)
    # k = length of the contiguous leading-L prefix → a clean L*/D* split.
    k = 0
    for i in sorted(pattern):
        if pattern[i] == "L":
            k += 1
        else:
            break
    step = rng.choice((-1, 1))
    new_k = k + step
    if not 0 <= new_k <= n:
        new_k = k - step  # bounce off the edge
    return shift_boundary(n, new_k)


# ── structured random initializer (scout / Variant-B init draw) ───────────────

# Prior families a fresh structured draw samples from. Each takes (n, rng).
def _prior_alternating(n: int, rng: random.Random) -> ChiralityPattern:
    return alternating_pattern(n, start=rng.choice(_VALID))


def _prior_boundary(n: int, rng: random.Random) -> ChiralityPattern:
    return shift_boundary(n, k=rng.randint(0, n))


def _prior_apices(n: int, rng: random.Random) -> ChiralityPattern:
    n_apex = rng.randint(1, max(1, n // 3))
    apices = rng.sample(range(n), k=min(n_apex, n))
    return d_at_turn_apices(n, apices)


def _prior_independent(n: int, rng: random.Random) -> ChiralityPattern:
    return {i: rng.choice(_VALID) for i in range(n)}


_PRIORS = (_prior_alternating, _prior_boundary, _prior_apices, _prior_independent)


def seed_chirality_pattern(
    n: int,
    required: Mapping[int, str] | None = None,
    rng: random.Random | None = None,
) -> ChiralityPattern:
    """Draw a structured chirality pattern of length ``n``.

    Samples one of the structured priors (alternating / single-boundary /
    D-at-apices / per-position) using ``rng``, then forces every ``required``
    position to its mandated handedness (e.g. metal coordinators). Deterministic
    given ``rng``; varies across rng seeds.
    """
    rng = rng or random.Random()
    pattern = rng.choice(_PRIORS)(n, rng)
    if required:
        for pos, label in required.items():
            if 0 <= pos < n:
                pattern[pos] = _check_label(label)
    return pattern
