"""Per-case seeding dispatcher (spec §2 "Seeding philosophy", §5; tracker #6).

Mirrors benchmark.restraints.build_for_case: a SEED_POLICIES table keyed by case_id +
build_seed_for_case(case, ...) -> seed.SeedResult. Composes the seed.py primitives
(PepMLMSeedGenerator / RandomSeedGenerator / retro_inverso / insert_fixed_chirality); it
does NOT reimplement them. CPU-only: no torch/transformers/chai import at module level —
the real PepMLM model is only reached when a generator with generate_fn=None is passed
(GPU/network path), never in CPU tests (which inject a fake generator).

Per-case policy:
  alpha    : PepMLM conditioned on the L-HLH target, len 21, retro-inverso ON.
  nonalpha : PepMLM conditioned on the HA protomer, len 31, retro-inverso ON.
  cyclic   : PepMLM CANNOT condition on a metal -> UNCONDITIONED generation at the baseline
             length 24 (the full S2-symmetric 6UFA 24-mer), retro-inverso OFF, manual D/L His at
             the metal-coordinating positions (taken from the case's metal_coordination his_resnums).
"""
from __future__ import annotations

from typing import Optional

from xenodesign.seed import SeedResult, insert_fixed_chirality, retro_inverso, select_seeds_by_mirror_consistency

# His D/L pattern at the 6UFA coordinating positions of the FULL 24-mer: the deposited tetrahedral
# [Zn(L-His)2(D-His)2] uses His 6/12/18/24 with chirality L/D/L/D. RE-SYNCED 2026-06-24 to the full
# S2-symmetric 24-mer deposit (was the wrong single-12-mer (6 L, 12 D) — a 12-mer cannot make a
# 4-coordinate site); positions match the cyclic case's metal_coordination his_resnums (6,12,18,24).
_CYCLIC_HIS_CHIRALITY = {6: "L", 12: "D", 18: "L", 24: "D"}

SEED_POLICIES: dict = {
    "alpha": {
        "mode": "pepmlm_conditioned",
        "length": 21,
        "reverse": True,
        "target_record": "target",
        "fixed_chirality_positions": {},
    },
    "nonalpha": {
        "mode": "pepmlm_conditioned",
        "length": 31,
        "reverse": True,
        "target_record": "target",
        "fixed_chirality_positions": {},
    },
    "cyclic": {
        "mode": "unconditioned",
        "length": 24,
        "reverse": False,
        "target_record": None,
        "fixed_chirality_positions": dict(_CYCLIC_HIS_CHIRALITY),
    },
}


def seed_policy(case_id: str) -> dict:
    """Return the seeding policy for a case_id (raises KeyError listing known ids)."""
    try:
        return SEED_POLICIES[case_id]
    except KeyError:
        raise KeyError(
            f"no seed policy for case {case_id!r}; known: {sorted(SEED_POLICIES)}"
        ) from None


def build_seed_for_case(case, generator, target_seq: Optional[str] = None) -> SeedResult:
    """Build the per-case starting seed as a SeedResult (spec §2; tracker #6).

    Args:
        case: a BenchmarkCase (benchmark.cases). Drives the policy via case.case_id.
        generator: a seed generator exposing generate(target_seq, length) for conditioned
            cases (e.g. seed.PepMLMSeedGenerator with an injected fake in CPU tests), or
            generate_conditioned(target_seq, length) for the unconditioned cyclic path
            (e.g. seed.RandomSeedGenerator).
        target_seq: the target sequence to condition on (REQUIRED for pepmlm_conditioned
            cases). Production callers read it via seed.read_target_sequence(case.fasta_path);
            tests pass it directly. Ignored for the unconditioned cyclic case.

    Returns:
        SeedResult(one_letter, length, reverse_applied, conditioned, fixed_chirality).
    """
    policy = seed_policy(case.case_id)
    length = policy["length"]

    if policy["mode"] == "pepmlm_conditioned":
        if target_seq is None:
            raise ValueError(
                f"case {case.case_id!r} is pepmlm_conditioned; a target_seq is required "
                f"(read it via seed.read_target_sequence(case.fasta_path))")
        # PepMLMSeedGenerator.generate already applies retro-inverso per its own `reverse`
        # flag (set at construction); the policy 'reverse' is the source of truth for the
        # recorded provenance flag.
        one_letter = generator.generate(target_seq=target_seq, length=length)
        return SeedResult(
            one_letter=one_letter, length=length,
            reverse_applied=bool(policy["reverse"]), conditioned=True,
            fixed_chirality={},
        )

    # Unconditioned path (cyclic): use generate_conditioned with an empty target,
    # which RandomSeedGenerator silently ignores.
    raw = generator.generate_conditioned(target_seq="", length=length)
    if policy["reverse"]:
        raw = retro_inverso(raw, reverse=True)
    positions = dict(policy["fixed_chirality_positions"])
    one_letter, fixed = insert_fixed_chirality(raw, positions=positions, residue="H")
    return SeedResult(
        one_letter=one_letter, length=length,
        reverse_applied=bool(policy["reverse"]), conditioned=False,
        fixed_chirality=fixed,
    )


def filter_seeds_by_mirror_consistency(candidates, axis: int = 0, threshold: float = 0.1):
    """Tier-1 mirror self-consistency filter over candidate seeds (spec §2.2, §5; #6).

    Wires seed.select_seeds_by_mirror_consistency into the seed-selection flow. Each
    candidate is a dict {'seed': SeedResult, 'coords': (n,3) or None, 'twin_coords':
    (n,3) or None}. A candidate whose mirror twin matches its prediction within `threshold`
    PASSES; one whose twin is too far is dropped.

    DEFAULT-SAFE: a candidate without predicted coords/twin (None) cannot be positively
    rejected, so it is KEPT — the filter only removes seeds it can prove inconsistent. This
    keeps the seed flow usable before any structure prediction has been run.

    Returns the surviving SeedResults (order preserved).
    """
    kept, testable = [], []
    for c in candidates:
        if c.get("coords") is None or c.get("twin_coords") is None:
            kept.append(c["seed"])  # not yet testable -> keep
        else:
            testable.append(c)
    passing = select_seeds_by_mirror_consistency(testable, axis=axis, threshold=threshold)
    kept.extend(c["seed"] for c in passing)
    order = {id(c["seed"]): i for i, c in enumerate(candidates)}
    return sorted(kept, key=lambda s: order[id(s)])
