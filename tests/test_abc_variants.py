"""CPU tests for the ABC design_fn variants (xenodesign/abc/variants.py).

Two pilot arms, wired as ``design_fn(identity, chirality_pattern) -> identity`` the engine calls:

- **Variant A:** ABC perturbs CHIRALITY; MPNN fills IDENTITY per pattern via
  ``SequenceUpdater.update(..., chirality_pattern=...)`` (the T1 per-position-handed path). So the
  search owns handedness, MPNN designs identity. The design_fn returns the PLAIN one-letter L
  identity — the fitness adapter applies the per-position ``mixed_chirality_fasta`` emit. (A
  parenthesized ``(DXX)`` return double-encodes in the fitness → -inf → frozen search.)
- **Variant B:** ABC mutates IDENTITY (point mutations over the 20 AAs) AND chirality; MPNN is
  warm-start only (initial population).

All CPU / fakes — no GPU.
"""
from __future__ import annotations

import random

from xenodesign.abc.variants import abc_variant_a_design_fn, abc_variant_b_design_fn


class _FakeMpnnBackend:
    """A fake inverse-folding backend (InverseFoldingBackend 6-arg protocol): returns an
    all-'G' designed chain of the right length so SequenceUpdater can emit a mixed FASTA.

    Records the fixed_mask it was handed so the test can assert MPNN was actually invoked.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, design_backbone, context_coords, context_elements,
                 fixed_mask, temperature, num_seqs):
        self.calls.append({"n": design_backbone.shape[0], "fixed_mask": list(fixed_mask)})
        # All-glycine designed chain (canonical L; reflection-clean so update() never rejects it).
        return ["G" * design_backbone.shape[0]] * num_seqs


def test_variant_a_calls_mpnn_with_chirality_pattern():
    be = _FakeMpnnBackend()
    fn = abc_variant_a_design_fn(be)
    pattern = {0: "D", 1: "L", 2: "D"}
    out = fn("AAA", pattern)
    # MPNN was invoked (identity is MPNN-designed, NOT carried through).
    assert be.calls, "MPNN backend was never called"
    # The returned identity is the PLAIN one-letter L sequence MPNN designed (no (DXX) blocks).
    assert out == "GGG"


def test_variant_a_returns_plain_identity_not_dfasta():
    # REGRESSION (pilot 2026-06-25): the design_fn must return a PLAIN one-letter identity, NOT a
    # parenthesized (DXX) FASTA. The fitness adapter re-emits handedness via mixed_chirality_fasta;
    # a parenthesized return double-encodes there (KeyError on '(') → fitness -inf → the search can
    # never keep an MPNN candidate and freezes on the warm-start seed.
    import xenodesign.classes.base  # noqa: F401  (prime base<->cyclic import cycle)
    from xenodesign.classes.cyclic import mixed_chirality_fasta

    class _AlaBackend:
        def __call__(self, db, cc, ce, fm, temperature, num_seqs):
            return ["A" * db.shape[0]] * num_seqs

    fn = abc_variant_a_design_fn(_AlaBackend())
    pattern = {0: "D", 1: "L", 2: "D"}
    out = fn("AAA", pattern)
    assert out == "AAA"           # plain one-letter identity, no parentheses
    assert "(" not in out
    # And it must round-trip through the fitness emit without raising (the bug that broke the pilot):
    emitted = mixed_chirality_fasta(out, fixed_chirality={p + 1: h for p, h in pattern.items()})
    assert emitted == "(DAL)A(DAL)"   # D at 0,2 → D-Ala block; L at 1 → bare 'A' (now done by fitness)


def test_variant_b_mutates_identity():
    fn = abc_variant_b_design_fn(rng=random.Random(0), mutation_rate=1.0)
    out = fn("AAA", {0: "D", 1: "L", 2: "D"})
    assert isinstance(out, str) and len(out) == 3
    assert out != "AAA"            # every position mutated at rate 1.0


def test_variant_b_zero_rate_keeps_warm_start():
    fn = abc_variant_b_design_fn(rng=random.Random(0), mutation_rate=0.0)
    assert fn("ACD", {0: "D", 1: "D", 2: "D"}) == "ACD"   # MPNN warm-start identity untouched


def test_variant_b_mutations_are_valid_amino_acids():
    fn = abc_variant_b_design_fn(rng=random.Random(7), mutation_rate=1.0)
    out = fn("AAAAAAAA", {i: "L" for i in range(8)})
    assert set(out) <= set("ACDEFGHIKLMNPQRSTVWY")   # 20 canonical AAs only
    assert len(out) == 8


# ── Variant A backbone injection (FIX 1: structure-aware re-design) ─────────────

class _RecordingBackend:
    """Fake InverseFoldingBackend that records the design_backbone it was handed so the test
    can assert MPNN re-designed on the REAL candidate backbone (not a zero placeholder)."""

    def __init__(self):
        self.backbones = []

    def __call__(self, design_backbone, context_coords, context_elements,
                 fixed_mask, temperature, num_seqs):
        import numpy as np
        self.backbones.append(np.asarray(design_backbone, dtype=float).copy())
        return ["G" * design_backbone.shape[0]] * num_seqs


def test_variant_a_uses_real_backbone_when_provided():
    # REGRESSION (ABC pilot 2026-06-25): Variant A MUST design on the candidate's ACTUAL last
    # structure, not a zero/empty backbone (which is structure-blind → search stalls on the seed).
    import numpy as np

    be = _RecordingBackend()
    fn = abc_variant_a_design_fn(be)
    pattern = {0: "D", 1: "L", 2: "D"}
    # A non-trivial backbone (n_res, 4, 3): N, CA, C, CB — the engine threads this in as the
    # candidate's last predicted structure (coordinate-only LigandMPNN adapter consumes it).
    real_bb = np.arange(3 * 4 * 3, dtype=float).reshape(3, 4, 3) + 1.0
    out = fn("AAA", pattern, last_structure=real_bb)

    assert be.backbones, "MPNN backend was never called"
    used = be.backbones[-1]
    assert used.shape == (3, 4, 3)
    assert not np.allclose(used, 0.0), "Variant A designed on a ZERO backbone (structure-blind)"
    # The adapter may reflect the backbone into the majority L/D frame (a sign flip on one axis),
    # so assert the REAL geometry was used up to that reflection (coordinate magnitudes match).
    np.testing.assert_allclose(np.abs(used), np.abs(real_bb))
    assert out == "GGG"


def test_variant_a_falls_back_to_zero_backbone_without_structure():
    # No structure yet (warm-start / first eval) → current behaviour: a zero placeholder backbone
    # of the right length, so the design_fn stays callable with the legacy 2-arg signature.
    import numpy as np

    be = _RecordingBackend()
    fn = abc_variant_a_design_fn(be)
    out = fn("AAA", {0: "D", 1: "L", 2: "D"})   # no last_structure
    assert be.backbones[-1].shape == (3, 4, 3)
    assert np.allclose(be.backbones[-1], 0.0)
    assert out == "GGG"
