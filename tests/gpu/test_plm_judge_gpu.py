"""GPU test: ESM-2 pseudo-log-likelihood discrimination.

Adversarial framing
-------------------
The design loop uses ESM-2 as a frozen *discriminator / adversary*: the loop must
produce sequences that ESM-2 finds natural (high PLL).  For this to be useful, ESM-2
must be able to DISCRIMINATE between:
  (a) natural / functional peptide sequences  → high PLL
  (b) random / shuffled sequences             → low PLL

This test PROVES the discrimination with concrete numbers.  If ESM-2 cannot separate
natural from random, the adversarial judge is worthless — and we say so explicitly.

Test structure
--------------
1. Positive set: real peptide sequences drawn from well-characterised proteins /
   peptide binders (mix of α-helical, β-strand, and mixed topologies).
2. Negative set: shuffled versions of the positive set (same composition, random order)
   + purely random sequences.
3. Assertion: mean PLL(natural) > mean PLL(random) by at least `_MIN_GAP` nats/residue.
   The exact numbers are printed so they appear in the test log.

Model: facebook/esm2_t33_650M_UR50D (650M params, ~2.5 GB in fp32 / ~1.2 GB in fp16).
GPU: cuda:0 (pinned).
"""
from __future__ import annotations

import random
import statistics

import pytest

from tests.gpu.conftest import require_cuda, require_transformers

# ── Test sequences ──────────────────────────────────────────────────────────────
# Positive set: natural peptide fragments from well-known proteins.
# Selected for diversity (helical, strand, loop; not all from one protein).
_NATURAL_PEPTIDES = [
    # Short functional segments of model proteins
    "GSHMKVLITGGAGFIGS",     # NAD-binding Rossmann fold β-strand (GAPDH-like)
    "ACDEFGHIKLMNPQRSTW",    # synthetic run through standard AA (canonical)
    "LLTEHQFNLLHKLSELTQ",   # coiled-coil heptad repeat (leucine-zipper)
    "RKLFRGVQGLAKKLREQ",     # helix-kink peptide (antimicrobial helix)
    "WGQGTSVTVSS",           # IgG framework loop (light-chain J region)
    "SYSMEISNSGAVPALYK",     # Triosephosphate isomerase barrel strand
    "KVFERCELARTLKRLGM",     # lysozyme α-helix segment
    "DYKDDDDK",              # FLAG tag — evolutionarily sourced epitope
    "MGSSHHHHHHSSGLVPR",     # his-tag + enterokinase — semi-natural
    "ACSNLYVSSQLRSYSSL",     # de novo helix from Rosetta designs (natural-like)
]

# Negative set: shuffled + random sequences (same lengths as positive set).
# Shuffled: same amino-acid composition as the corresponding positive; random order.
# Random: uniform draw from 20 standard AAs (maximum entropy).
_AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
_RANDOM_SEED  = 42


def _shuffled(seq: str, rng: random.Random) -> str:
    lst = list(seq)
    rng.shuffle(lst)
    return "".join(lst)


def _random_seq(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(_AA_ALPHABET) for _ in range(length))


def _make_negative_set(naturals: list[str], seed: int = _RANDOM_SEED) -> list[str]:
    rng = random.Random(seed)
    neg = []
    for seq in naturals:
        neg.append(_shuffled(seq, rng))       # composition-matched scramble
        neg.append(_random_seq(len(seq), rng)) # uniform random
    return neg


# ── Gap threshold ────────────────────────────────────────────────────────────────
# We require the mean PLL gap (natural − random) to be at least 0.3 nats/residue.
# ESM-2 650M on real sequences typically yields PLL ≈ −1.0 to −2.5 nats;
# random sequences typically ≈ −4.0 to −6.0 nats.  A 0.3 gap is conservative.
_MIN_GAP_NATS = 0.3


# ── Test ─────────────────────────────────────────────────────────────────────────

@pytest.mark.gpu
def test_esm_pll_discriminates_natural_from_random():
    """ESM-2 PLL must be significantly higher for natural than random sequences.

    Proves the adversarial judge can discriminate.  Reports exact PLL numbers.
    """
    require_cuda()
    require_transformers()

    import torch
    from xenodesign.judges.plm_judge import ESMPseudoLogLikelihood

    device = "cuda:0"
    judge = ESMPseudoLogLikelihood(device=device)

    negative_seqs = _make_negative_set(_NATURAL_PEPTIDES)

    print("\n\n── ESM-2 PLL Discrimination Report ──────────────────────────────────")
    print(f"Model: {judge._model_id}")
    print(f"Device: {device}  CUDA available: {torch.cuda.is_available()}")
    print(f"\nNatural sequences (n={len(_NATURAL_PEPTIDES)}):")

    natural_plls = []
    for seq in _NATURAL_PEPTIDES:
        pll = judge(seq)
        natural_plls.append(pll)
        print(f"  {seq[:30]:<30s}  PLL = {pll:+.4f} nats")

    print(f"\nNegative sequences (shuffled + random; n={len(negative_seqs)}):")
    random_plls = []
    for seq in negative_seqs:
        pll = judge(seq)
        random_plls.append(pll)
        print(f"  {seq[:30]:<30s}  PLL = {pll:+.4f} nats")

    mean_natural = statistics.mean(natural_plls)
    mean_random  = statistics.mean(random_plls)
    gap          = mean_natural - mean_random
    std_natural  = statistics.stdev(natural_plls) if len(natural_plls) > 1 else 0.0
    std_random   = statistics.stdev(random_plls)  if len(random_plls)  > 1 else 0.0

    print(f"\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  Mean PLL (natural):  {mean_natural:+.4f} ± {std_natural:.4f} nats")
    print(f"  Mean PLL (random):   {mean_random:+.4f} ± {std_random:.4f} nats")
    print(f"  Gap (natural−random): {gap:+.4f} nats")
    print(f"  Required gap:         ≥ {_MIN_GAP_NATS:.2f} nats")
    print(f"  DISCRIMINATION {'PASSES' if gap >= _MIN_GAP_NATS else 'FAILS'}")

    # What fraction of natural sequences beat the mean random PLL?
    n_natural_above_mean_random = sum(1 for p in natural_plls if p > mean_random)
    frac_above = n_natural_above_mean_random / len(natural_plls)
    print(f"  Natural seqs above mean-random PLL: "
          f"{n_natural_above_mean_random}/{len(natural_plls)} ({frac_above:.0%})")

    assert gap >= _MIN_GAP_NATS, (
        f"ESM-2 PLL discrimination INSUFFICIENT: "
        f"mean natural PLL = {mean_natural:.4f}, "
        f"mean random PLL = {mean_random:.4f}, "
        f"gap = {gap:.4f} nats (required >= {_MIN_GAP_NATS} nats). "
        f"The adversarial judge cannot distinguish natural from random sequences — "
        f"check model loading (is ESM-2 actually on GPU?), tokenisation, or model size."
    )


@pytest.mark.gpu
def test_esm_pll_monotone_with_sequence_quality():
    """PLL should rank a well-known natural peptide above a scrambled version.

    Secondary discrimination check: for each natural sequence, its PLL should exceed
    the PLL of its composition-matched scramble (same AAs, random order).
    We require ≥70% of pairs to satisfy this (not 100%, due to stochasticity in
    short sequences where positional context is limited).
    """
    require_cuda()
    require_transformers()

    from xenodesign.judges.plm_judge import ESMPseudoLogLikelihood

    device = "cuda:0"
    judge = ESMPseudoLogLikelihood(device=device)

    rng = random.Random(_RANDOM_SEED + 1)

    print("\n\n── Pairwise ranking: natural vs. scrambled (same composition) ───────")
    wins = 0
    total = 0
    for seq in _NATURAL_PEPTIDES:
        scrambled = _shuffled(seq, rng)
        pll_nat   = judge(seq)
        pll_scr   = judge(scrambled)
        win = pll_nat > pll_scr
        wins += int(win)
        total += 1
        mark = "WIN" if win else "LOSS"
        print(f"  [{mark}]  {seq[:20]:<20s} {pll_nat:+.4f}  vs  "
              f"{scrambled[:20]:<20s} {pll_scr:+.4f}")

    frac_win = wins / total
    print(f"\n  Natural > scrambled: {wins}/{total} = {frac_win:.0%}")

    assert frac_win >= 0.70, (
        f"ESM-2 pairwise ranking failed: natural > scrambled in only "
        f"{wins}/{total} cases ({frac_win:.0%}); expected >= 70%. "
        f"The adversarial judge may not reliably prefer natural sequences."
    )


@pytest.mark.gpu
def test_panel_selects_chirality_clean_design_on_mock_loop():
    """Integration: panel selects chirality-clean step over naive best, using real PLL.

    Constructs a mock loop history with:
      - Step 0: high ipTM (0.85), chirality VIOLATED (0.5), arbitrary sequence
      - Step 1: moderate ipTM (0.65), chirality CLEAN (0.0), natural sequence
    The naive best() picks step 0 (higher ipTM). The panel must pick step 1 (chirality
    veto kills step 0).  Additionally computes real ESM-2 PLL for both sequences,
    showing the system end-to-end.
    """
    require_cuda()
    require_transformers()

    import numpy as np
    from xenodesign.judges.plm_judge import ESMPseudoLogLikelihood
    from xenodesign.judges.panel import JudgePanel, RefereeScore
    from xenodesign.loop import HalluLoop, LoopState, LoopStep

    device = "cuda:0"
    judge = ESMPseudoLogLikelihood(device=device)

    # Sequences for the two mock loop steps.
    # Step 0: a random sequence (low PLL expected) with chirality violation.
    seq_bad   = "MCWFHQLAPTR"   # somewhat random composition
    # Step 1: a natural-like fragment with clean chirality.
    seq_good  = "KVFERCELARTL"  # lysozyme α-helix (natural)

    pll_bad  = judge(seq_bad)
    pll_good = judge(seq_good)

    print(f"\n── Mock loop integration ────────────────────────────────────────────")
    print(f"  Step 0 (violated): seq={seq_bad}  ipTM=0.85  chir_viol=0.50  PLL={pll_bad:.4f}")
    print(f"  Step 1 (clean):    seq={seq_good} ipTM=0.65  chir_viol=0.00  PLL={pll_good:.4f}")

    coords = np.zeros((3, 3))

    def _make_loop_step(iptm, chir_viol, pll, seq):
        pred = type("P", (), {
            "iptm": iptm,
            "interface_plddt": 80.0,
            "chirality_violation_frac": chir_viol,
        })()
        return LoopStep(
            state=LoopState(d_fasta=f"({seq})", coords=coords),
            prediction=pred,
            score=iptm,
        ), pll

    step0, pll0 = _make_loop_step(0.85, 0.50, pll_bad,  seq_bad)
    step1, pll1 = _make_loop_step(0.65, 0.00, pll_good, seq_good)
    history = [step0, step1]

    def score_fn_with_real_pll(step):
        """Extract scores + attach real ESM-2 PLL."""
        idx = history.index(step)
        real_pll = [pll0, pll1][idx]
        return RefereeScore(
            chirality_violation=step.prediction.chirality_violation_frac,
            iptm=step.prediction.iptm,
            interface_plddt=step.prediction.interface_plddt,
            pll=real_pll,
        )

    panel = JudgePanel(score_fn=score_fn_with_real_pll)

    naive_best   = HalluLoop.best(history)
    panel_choice = HalluLoop.select_by_panel(history, panel)

    raw_scores = [score_fn_with_real_pll(s) for s in history]
    panel_result = panel.combine(raw_scores)

    print(f"\n  Naive best() → step {history.index(naive_best)} (ipTM={naive_best.prediction.iptm:.2f})")
    print(f"  Panel choice → step {history.index(panel_choice)} (ipTM={panel_choice.prediction.iptm:.2f})")
    print(f"  Panel composites: {[f'{c:.4f}' for c in panel_result.composite_scores]}")
    print(f"  Vetoed:           {panel_result.vetoed}")
    print(f"  Fallback used:    {panel_result.fallback_used}")

    # Naive best() picks step 0 (higher ipTM).
    assert naive_best is step0, "Naive best() should pick the higher-ipTM step"

    # Panel vetoes step 0 (chirality violation), must pick step 1.
    assert panel_choice is step1, (
        f"Panel should pick chirality-clean step 1, but picked step {history.index(panel_choice)}. "
        f"Veto status: {panel_result.vetoed}. Composites: {panel_result.composite_scores}."
    )

    # Chirality veto must be active on step 0.
    assert panel_result.vetoed[0], "Step 0 (chir_viol=0.5) must be vetoed"
    assert not panel_result.vetoed[1], "Step 1 (chir_viol=0.0) must not be vetoed"

    print("\n  PASS: panel selected chirality-clean step with real ESM-2 PLL scoring.")
