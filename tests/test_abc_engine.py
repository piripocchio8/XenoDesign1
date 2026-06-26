"""ABC T6 — the ABC/EA search engine (spec §5.4).

Pure-Python artificial-bee-colony over (identity sequence, chirality pattern).
Fitness-agnostic: the engine takes an injected ``fitness_fn(sequence, pattern)
-> float`` and is exercised here with a SYNTHETIC fitness (no GPU). We assert it
(1) CLIMBS toward a hidden optimum, (2) RESPECTS the hard eval budget, (3) SCOUTS
re-seed stagnant sources, and (4) is DETERMINISTIC under a fixed seed.
"""
import random

from xenodesign.abc.engine import FoodSource, abc_search


# ── synthetic fitnesses ────────────────────────────────────────────────────────

def _synthetic_fitness(seq, pattern):
    # Optimum = all-'D' pattern; smooth, deterministic → ABC must climb toward it.
    return sum(1 for v in pattern.values() if v == "D")


def _identity_design_fn(seq, pattern):
    return seq  # Variant-A-style: identity untouched by the engine in this fake


# ── plan's locked contract ─────────────────────────────────────────────────────

def test_abc_converges_toward_optimum():
    n = 6
    init = [
        FoodSource(
            identity="A" * n,
            chirality_pattern={i: "L" for i in range(n)},
            last_structure=None,
            nectar=None,
        )
    ]
    best, history = abc_search(
        init,
        _synthetic_fitness,
        _identity_design_fn,
        n_cycles=40,
        colony_size=8,
        scout_limit=5,
        chai_eval_budget=10_000,
        rng=random.Random(0),
    )
    assert best.nectar == n  # found the all-D optimum
    assert len(history) >= 1


def test_abc_respects_eval_budget():
    calls = {"n": 0}

    def counting_fitness(seq, pattern):
        calls["n"] += 1
        return 0.0

    init = [FoodSource("AA", {0: "L", 1: "L"}, None, None)]
    abc_search(
        init,
        counting_fitness,
        _identity_design_fn,
        n_cycles=10_000,
        colony_size=4,
        scout_limit=3,
        chai_eval_budget=20,
        rng=random.Random(0),
    )
    assert calls["n"] <= 20  # hard Chai-eval budget enforced


def test_scout_replaces_stagnant_source():
    seen_identities = set()

    def fitness(seq, pattern):
        seen_identities.add(seq)
        return 0.0  # flat → everything stagnates

    init = [FoodSource("AAAA", {i: "L" for i in range(4)}, None, None)]
    abc_search(
        init,
        fitness,
        lambda s, p: s + "x",
        n_cycles=20,
        colony_size=2,
        scout_limit=2,
        chai_eval_budget=10_000,
        rng=random.Random(1),
    )
    assert len(seen_identities) > 1  # scouts introduced fresh sources


# ── prompt's additional evidence ───────────────────────────────────────────────

def test_climb_history_is_monotonic_nondecreasing():
    # best-so-far nectar recorded per cycle must never go DOWN (greedy elitism).
    n = 8
    init = [FoodSource("A" * n, {i: "L" for i in range(n)}, None, None)]
    best, history = abc_search(
        init,
        _synthetic_fitness,
        _identity_design_fn,
        n_cycles=30,
        colony_size=10,
        scout_limit=5,
        chai_eval_budget=10_000,
        rng=random.Random(2),
    )
    bests = [h["best_nectar"] for h in history]
    assert bests == sorted(bests)  # non-decreasing
    assert bests[-1] >= bests[0]
    assert best.nectar == max(bests)


def test_determinism_under_fixed_seed():
    n = 6
    init = [FoodSource("A" * n, {i: "L" for i in range(n)}, None, None)]
    kw = dict(
        n_cycles=25, colony_size=6, scout_limit=4, chai_eval_budget=10_000
    )
    b1, h1 = abc_search(
        init, _synthetic_fitness, _identity_design_fn, rng=random.Random(123), **kw
    )
    b2, h2 = abc_search(
        init, _synthetic_fitness, _identity_design_fn, rng=random.Random(123), **kw
    )
    assert b1.nectar == b2.nectar
    assert b1.chirality_pattern == b2.chirality_pattern
    assert [h["best_nectar"] for h in h1] == [h["best_nectar"] for h in h2]


def test_does_not_mutate_caller_init_pop():
    init = [FoodSource("AAA", {0: "L", 1: "L", 2: "L"}, None, None)]
    snapshot = (init[0].identity, dict(init[0].chirality_pattern), init[0].nectar)
    abc_search(
        init,
        _synthetic_fitness,
        _identity_design_fn,
        n_cycles=10,
        colony_size=4,
        scout_limit=3,
        chai_eval_budget=500,
        rng=random.Random(0),
    )
    assert (init[0].identity, dict(init[0].chirality_pattern), init[0].nectar) == snapshot


def test_prompt_kwarg_aliases_and_seed_fn():
    # The prompt's interface: design_fn keyword, eval_budget alias, rng_seed int,
    # and an injected seed_fn used by scouts.
    seeded = {"n": 0}

    def seed_fn(rng):
        seeded["n"] += 1
        n = 4
        return FoodSource("S" * n, {i: "D" for i in range(n)}, None, None)

    init = [FoodSource("AAAA", {i: "L" for i in range(4)}, None, None)]
    best, history = abc_search(
        init,
        _synthetic_fitness,
        design_fn=_identity_design_fn,
        seed_fn=seed_fn,
        n_cycles=15,
        colony_size=3,
        scout_limit=2,
        eval_budget=10_000,  # alias for chai_eval_budget
        rng_seed=0,  # alias for rng=random.Random(0)
    )
    assert best.nectar is not None
    # flat-ish landscape + tiny scout_limit → scouts fire → seed_fn invoked
    assert seeded["n"] >= 1


def test_default_design_fn_keeps_identity():
    # design_fn defaults to None → identity is left as-is (chirality-only search).
    n = 5
    init = [FoodSource("A" * n, {i: "L" for i in range(n)}, None, None)]
    best, _ = abc_search(
        init,
        _synthetic_fitness,
        n_cycles=20,
        colony_size=6,
        scout_limit=4,
        eval_budget=5_000,
        rng_seed=1,
    )
    assert best.nectar == n  # still climbs on the chirality axis alone


# ── last_structure threading (FIX 1: Variant A backbone injection) ──────────────

def test_two_arg_design_fn_still_supported():
    # A legacy 2-arg design_fn (no last_structure param) must keep working untouched — the
    # engine only threads last_structure to design_fns that accept it.
    n = 5
    init = [FoodSource("A" * n, {i: "L" for i in range(n)}, None, None)]
    best, _ = abc_search(
        init, _synthetic_fitness, _identity_design_fn,
        n_cycles=10, colony_size=4, scout_limit=3, eval_budget=2_000, rng_seed=2,
    )
    assert best.nectar == n


def test_fitness_last_structure_recorded_and_threaded_to_design_fn():
    # The fitness exposes its last-evaluated structure via a `last_structure` attribute; the
    # engine records it onto the FoodSource and threads it to a design_fn that opts in (via a
    # `last_structure` kwarg). So Variant A re-designs identity on the candidate's ACTUAL backbone.
    n = 4

    class _StructFitness:
        """Synthetic fitness that, like the chai adapter, publishes the structure it just
        'predicted' on a side-channel attribute the engine reads."""
        def __init__(self):
            self.last_structure = None

        def __call__(self, seq, pattern):
            self.last_structure = f"struct::{seq}"   # opaque token (CIF path in production)
            return sum(1 for v in pattern.values() if v == "D")

    seen = {"backbones": []}

    def design_fn(identity, pattern, last_structure=None):
        seen["backbones"].append(last_structure)
        return identity

    fitness = _StructFitness()
    init = [FoodSource("A" * n, {i: "L" for i in range(n)}, None, None)]
    best, _ = abc_search(
        init, fitness, design_fn,
        n_cycles=3, colony_size=2, scout_limit=10, eval_budget=2_000, rng_seed=0,
    )
    # design_fn was handed a real (non-None) last_structure recorded from a prior eval.
    assert any(b is not None for b in seen["backbones"])
    assert any(isinstance(b, str) and b.startswith("struct::") for b in seen["backbones"])
    # the winning source carries the structure the fitness recorded for it.
    assert best.last_structure is not None and best.last_structure.startswith("struct::")
