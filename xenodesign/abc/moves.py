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


# ── track #2: ncAA identity move (Variant-B only) ───────────────────────────────

# Reverting an ncAA block to a canonical residue draws from this small neutral set (the move
# only needs a plausible canonical residue to put back; identity refinement is MPNN's job).
_REVERT_CANONICAL = "AGLSV"


def identity_tokens(identity: str) -> list[str]:
    """Split a Variant-B identity into per-position tokens.

    A token is either a single canonical 1-letter code (``'A'``) or a parenthesized ncAA/D-CCD
    block (``'(AIB)'``). This is the per-RESIDUE view both the ncAA move and the FASTA emit use,
    so position indices line up with the chirality pattern. Malformed (unclosed ``'('``) input
    yields the rest of the string as one trailing token (best-effort; never raises)."""
    out: list[str] = []
    i = 0
    while i < len(identity):
        if identity[i] == "(":
            j = identity.find(")", i)
            if j == -1:
                out.append(identity[i:])
                break
            out.append(identity[i:j + 1])
            i = j + 1
        else:
            out.append(identity[i])
            i += 1
    return out


def ncaa_positions(identity: str) -> set:
    """0-based positions of the ncAA ``(XXX)`` blocks in a Variant-B identity.

    These are the positions MPNN must keep FIXED (MPNN cannot emit ncAA): the caller unions
    this into ``SequenceUpdater(frozen_positions=...)`` (the same seam as declared coordinators)
    so an ABC-chosen ncAA is never overwritten by an inverse-folding re-design."""
    return {i for i, tok in enumerate(identity_tokens(identity)) if tok.startswith("(")}


def ncaa_identity_move(
    identity: str,
    rng: random.Random,
    palette: Sequence[str],
    frozen: set | None = None,
) -> str:
    """Set ONE non-frozen position's identity to a palette ncAA (or revert an ncAA to canonical).

    Returns ``identity`` unchanged when ``palette`` is empty (ncAA OFF) — the existing Variant-B
    behaviour. Otherwise picks one position not in ``frozen`` (0-based) and either swaps in a
    random palette ncAA as a ``(XXX)`` block, or — if that position is already an ncAA block —
    reverts it to a canonical residue. Frozen positions (declared coordinators) are never chosen.
    Pure/deterministic given ``rng``."""
    if not palette:
        return identity
    frozen = frozen or set()
    toks = identity_tokens(identity)
    choices = [i for i in range(len(toks)) if i not in frozen]
    if not choices:
        return identity
    i = rng.choice(choices)
    if toks[i].startswith("("):
        toks[i] = rng.choice(_REVERT_CANONICAL)  # revert the ncAA back to a canonical residue
    else:
        toks[i] = f"({rng.choice(list(palette))})"
    return "".join(toks)


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
