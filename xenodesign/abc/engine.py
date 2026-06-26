"""ABC / evolutionary search engine over (identity, chirality) (spec §5.4).

Artificial Bee Colony adapted to mixed-chirality peptide design. A **food
source** carries an identity sequence, a chirality pattern (``dict[int->'L'|'D']``)
and its nectar (= fitness). The colony runs three phases per cycle:

- **employed bees** perturb each source's chirality (and, in Variant B, its
  identity via the injected ``design_fn``), evaluate, and greedily keep the new
  source iff it is better (else bump the source's stagnation ``trials``);
- **onlooker bees** roulette-select high-nectar sources (exploit) and perturb
  again with the same greedy-keep rule;
- **scout bees** replace any source stagnant for ``> scout_limit`` cycles with a
  fresh structured draw (via ``seed_fn`` if injected, else a chirality-prior
  re-seed + ``design_fn`` identity refill).

The engine is **fitness-agnostic** — it only ever calls ``fitness_fn(sequence,
chirality_pattern) -> float`` — and honours a hard ``eval_budget`` (max
``fitness_fn`` calls). It imports nothing heavy (no torch/chai), so it is
CPU-testable with a synthetic fitness. Determinism: all randomness flows through
a single injected ``random.Random``.
"""
from __future__ import annotations

import inspect
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from xenodesign.abc import moves

FitnessFn = Callable[[str, Mapping[int, str]], float]
DesignFn = Callable[[str, Mapping[int, str]], str]
SeedFn = Callable[[random.Random], "FoodSource"]


@dataclass
class FoodSource:
    """One candidate: identity sequence + chirality pattern + cached fitness.

    ``trials`` counts consecutive cycles without improvement (scout trigger).
    ``last_structure`` is an opaque slot for a cached prediction (unused by the
    pure engine; the GPU fitness adapter may populate it).
    """

    identity: str
    chirality_pattern: dict
    last_structure: Any = None
    nectar: Optional[float] = None
    trials: int = 0


# ── eval-budget guard ─────────────────────────────────────────────────────────

class _BudgetExhausted(Exception):
    """Raised internally when the fitness-eval budget is spent."""


@dataclass
class _Evaluator:
    fitness_fn: FitnessFn
    remaining: int
    last_structure: Any = None  # structure the fitness published for the most recent eval

    def __call__(self, identity: str, pattern: Mapping[int, str]) -> float:
        if self.remaining <= 0:
            raise _BudgetExhausted
        self.remaining -= 1
        nectar = self.fitness_fn(identity, pattern)
        # Side-channel: a structure-aware fitness (the chai adapter) publishes the structure it
        # just predicted on ``fitness_fn.last_structure``; capture it so the engine can record it
        # onto the FoodSource and thread it back to a structure-aware design_fn (Variant A).
        self.last_structure = getattr(self.fitness_fn, "last_structure", None)
        return nectar


# ── helpers ────────────────────────────────────────────────────────────────────

def _accepts_last_structure(design_fn: DesignFn) -> bool:
    """True iff ``design_fn`` opts into structure-aware re-design by declaring a
    ``last_structure`` parameter. Keeps the engine's fitness-/design-agnostic 2-arg contract:
    legacy ``design_fn(identity, pattern)`` callables (and the synthetic-engine tests) are called
    unchanged; only Variant A (which accepts ``last_structure``) is handed the candidate backbone."""
    try:
        return "last_structure" in inspect.signature(design_fn).parameters
    except (TypeError, ValueError):  # builtins / C-callables with no signature
        return False


def _call_design_fn(design_fn: DesignFn, identity: str, pattern: Mapping[int, str],
                    last_structure: Any) -> str:
    """Invoke ``design_fn``, threading ``last_structure`` only when it opts in (see above)."""
    if last_structure is not None and _accepts_last_structure(design_fn):
        return design_fn(identity, pattern, last_structure=last_structure)
    return design_fn(identity, pattern)


def _perturb_chirality(pattern: Mapping[int, str], rng: random.Random) -> dict:
    """Apply one structured chirality move (flip a position or shift the L/D
    boundary), returning a new pattern."""
    if not pattern:
        return dict(pattern)
    if rng.random() < 0.5:
        i = rng.choice(sorted(pattern))
        return moves.flip_position(pattern, i)
    return moves.shift_boundary_perturb(pattern, rng)


def _default_seed_fn(template: FoodSource, design_fn: Optional[DesignFn]):
    """Fallback scout seeder: a fresh chirality prior over the template's length,
    with identity refilled by ``design_fn`` (Variant A/B) or kept as the
    template's identity (chirality-only search)."""

    n = len(template.chirality_pattern)
    base_identity = template.identity

    def seed(rng: random.Random) -> FoodSource:
        pattern = moves.seed_chirality_pattern(n, rng=rng)
        identity = design_fn(base_identity, pattern) if design_fn else base_identity
        return FoodSource(identity, pattern, None, None, trials=0)

    return seed


def _roulette_select(sources: Sequence[FoodSource], rng: random.Random) -> int:
    """Fitness-proportionate index. Nectar is shifted so the worst source still
    has a small positive weight (handles negative / equal fitnesses)."""
    nectars = [s.nectar if s.nectar is not None else float("-inf") for s in sources]
    finite = [x for x in nectars if x != float("-inf")]
    if not finite:
        return rng.randrange(len(sources))
    lo = min(finite)
    weights = [(x - lo + 1.0) if x != float("-inf") else 0.0 for x in nectars]
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(sources))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(sources) - 1


# ── public API ─────────────────────────────────────────────────────────────────

def abc_search(
    init_pop: Sequence[FoodSource],
    fitness_fn: FitnessFn,
    design_fn: Optional[DesignFn] = None,
    *,
    n_cycles: int,
    colony_size: int,
    scout_limit: int,
    chai_eval_budget: Optional[int] = None,
    eval_budget: Optional[int] = None,
    seed_fn: Optional[SeedFn] = None,
    rng: Optional[random.Random] = None,
    rng_seed: Optional[int] = None,
) -> tuple[FoodSource, list[dict]]:
    """Run the ABC search and return ``(best_source, history)``.

    Args:
        init_pop: seed food sources (warm-start). Grown/cloned up to
            ``colony_size``; the caller's objects are never mutated.
        fitness_fn: ``(identity, pattern) -> float``. The ONLY oracle the engine
            calls; everything else (Chai, objectives) lives behind it.
        design_fn: optional ``(identity, pattern) -> identity`` to refill/mutate
            identity for a (possibly new) chirality pattern. ``None`` → identity is
            carried through unchanged (chirality-only search).
        n_cycles: max employed+onlooker+scout cycles.
        colony_size: number of food sources maintained.
        scout_limit: a source stagnant for ``> scout_limit`` cycles is re-seeded.
        chai_eval_budget / eval_budget: hard cap on ``fitness_fn`` calls
            (aliases; ``eval_budget`` wins if both given).
        seed_fn: optional ``(rng) -> FoodSource`` for scouts; defaults to a
            structured chirality re-draw + ``design_fn`` identity refill.
        rng / rng_seed: randomness source. ``rng`` wins; else ``Random(rng_seed)``;
            else a fresh ``Random()``.

    Returns:
        ``(best, history)`` where ``best`` is the highest-nectar source seen and
        ``history`` is a per-cycle list of ``{"cycle", "best_nectar",
        "evals_used"}``.
    """
    if rng is None:
        rng = random.Random(rng_seed)
    budget = eval_budget if eval_budget is not None else chai_eval_budget
    if budget is None:
        budget = 10 ** 9  # effectively unbounded
    ev = _Evaluator(fitness_fn, budget)

    if not init_pop:
        raise ValueError("init_pop must contain at least one FoodSource")
    if colony_size < 1:
        raise ValueError("colony_size must be >= 1")

    template = init_pop[0]
    seeder = seed_fn or _default_seed_fn(template, design_fn)

    # Build the colony: deep-copy the seeds (never touch the caller's objects),
    # then top up to colony_size via the seeder.
    def _clone(s: FoodSource) -> FoodSource:
        return replace(s, chirality_pattern=dict(s.chirality_pattern), trials=0)

    colony: list[FoodSource] = []
    history: list[dict] = []
    best: Optional[FoodSource] = None

    def _track(s: FoodSource) -> None:
        nonlocal best
        if s.nectar is None:
            return
        if best is None or s.nectar > best.nectar:
            best = replace(s, chirality_pattern=dict(s.chirality_pattern))  # carries last_structure

    try:
        for s in init_pop[:colony_size]:
            c = _clone(s)
            if c.nectar is None:
                c.nectar = ev(c.identity, c.chirality_pattern)
                c.last_structure = ev.last_structure
            colony.append(c)
            _track(c)
        while len(colony) < colony_size:
            c = seeder(rng)
            c = replace(c, chirality_pattern=dict(c.chirality_pattern), trials=0)
            c.nectar = ev(c.identity, c.chirality_pattern)
            c.last_structure = ev.last_structure
            colony.append(c)
            _track(c)

        for cycle in range(n_cycles):
            # ── employed bees: perturb every source, greedy-keep ──
            for idx in range(len(colony)):
                _try_neighbour(colony, idx, ev, design_fn, rng, _track)

            # ── onlooker bees: roulette-select, perturb again ──
            for _ in range(len(colony)):
                idx = _roulette_select(colony, rng)
                _try_neighbour(colony, idx, ev, design_fn, rng, _track)

            # ── scout bees: re-seed the most-stagnant over-limit source ──
            stale = [i for i, s in enumerate(colony) if s.trials > scout_limit]
            if stale:
                idx = max(stale, key=lambda i: colony[i].trials)
                c = seeder(rng)
                c = replace(c, chirality_pattern=dict(c.chirality_pattern), trials=0)
                c.nectar = ev(c.identity, c.chirality_pattern)
                c.last_structure = ev.last_structure
                colony[idx] = c
                _track(c)

            history.append(
                {
                    "cycle": cycle,
                    "best_nectar": best.nectar if best else None,
                    "evals_used": budget - ev.remaining,
                }
            )
    except _BudgetExhausted:
        # Budget spent mid-cycle: stop cleanly, return the best found so far.
        if not history:
            history.append(
                {
                    "cycle": 0,
                    "best_nectar": best.nectar if best else None,
                    "evals_used": budget - ev.remaining,
                }
            )

    if best is None:
        # Degenerate budget (exhausted before any evaluation): return the
        # warm-start seed unevaluated rather than crash the caller.
        best = _clone(template)
    return best, history


def _try_neighbour(
    colony: list[FoodSource],
    idx: int,
    ev: _Evaluator,
    design_fn: Optional[DesignFn],
    rng: random.Random,
    track: Callable[[FoodSource], None],
) -> None:
    """Generate a neighbour of ``colony[idx]``, evaluate, greedy-keep or bump
    stagnation. May raise ``_BudgetExhausted`` (propagated to stop the search)."""
    cur = colony[idx]
    new_pattern = _perturb_chirality(cur.chirality_pattern, rng)
    # Re-design identity on the candidate's ACTUAL last structure (Variant A): the source's
    # cached backbone is threaded to a structure-aware design_fn; legacy 2-arg design_fns ignore it.
    if design_fn:
        new_identity = _call_design_fn(design_fn, cur.identity, new_pattern, cur.last_structure)
    else:
        new_identity = cur.identity
    nectar = ev(new_identity, new_pattern)  # may raise _BudgetExhausted
    if cur.nectar is None or nectar > cur.nectar:
        # Record the structure the fitness just predicted so the next move designs on it.
        colony[idx] = FoodSource(new_identity, new_pattern, ev.last_structure, nectar, trials=0)
        track(colony[idx])
    else:
        cur.trials += 1
