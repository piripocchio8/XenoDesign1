"""Chirality-aware scoring and top-k selection for designed candidates (spec §2.4, §5)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def design_score(
    iptm: float,
    interface_plddt: float,
    chirality_violation_frac: float,
    chirality_weight: float = 1.0,
    *,
    helix_frac: float = 0.0,
    helix_weight: float = 0.0,
    num_intra_contacts: int = 0,
    target_intra_contacts: int = 1,
    intra_contact_weight: float = 0.0,
    num_inter_contacts: int = 0,
    target_inter_contacts: int = 1,
    inter_contact_weight: float = 0.0,
) -> float:
    """Composite design score: higher is better.

    Rewards interface confidence (ipTM in [0,1], interface pLDDT in [0,100] normalized to
    [0,1]) and penalizes chirality violation (fraction in [0,1]).

    Optional, DEFAULT-OFF forward terms (lane A_scorer; all weights default 0.0 so the score is
    byte-for-byte unchanged unless a caller sets a weight):

    * helix_weight * helix_frac — secondary-structure (helix) reward. helix_frac is the
      CA-geometry helix fraction in [0,1] from secondary_structure.helix_fraction. A POSITIVE
      helix_weight rewards helix (alpha case); a NEGATIVE helix_weight penalizes it
      (anti-alpha case). See also ss_bias_score for the per-case profile form.
    * intra_contact_weight * min(num_intra_contacts, target_intra_contacts)/target_intra_contacts
      — compactness / core-packing satisfaction (hinge saturating at the target).
    * inter_contact_weight * min(num_inter_contacts, target_inter_contacts)/target_inter_contacts
      — interface contact satisfaction (hinge saturating at the target).

    The contact counts come from secondary_structure.count_contacts. Targets default to 1 to
    avoid division by zero when the corresponding weight is 0.0 (the term is then 0-weighted
    anyway). The metal coordination-number reward is deferred to task #9 (not added here).
    """
    confidence = 0.5 * iptm + 0.5 * (interface_plddt / 100.0)
    score = confidence - chirality_weight * chirality_violation_frac
    if helix_weight != 0.0:
        score += helix_weight * helix_frac
    if intra_contact_weight != 0.0:
        sat = min(num_intra_contacts, target_intra_contacts) / max(target_intra_contacts, 1)
        score += intra_contact_weight * sat
    if inter_contact_weight != 0.0:
        sat = min(num_inter_contacts, target_inter_contacts) / max(target_inter_contacts, 1)
        score += inter_contact_weight * sat
    return score


def sequence_complexity(seq: str) -> float:
    """Normalized Shannon entropy H / log2(20) of the amino-acid composition, in [0, 1].

    0.0 for empty input or a homopolymer; a uniform 20-letter sequence -> 1.0. A coarse
    sequence-diversity measure (no structure, no GPU)."""
    from collections import Counter

    s = (seq or "").upper()
    n = len(s)
    if n == 0:
        return 0.0
    counts = Counter(s)
    if len(counts) == 1:
        return 0.0
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return h / math.log2(20)


def sequence_quality_key(seq: str) -> float:
    """MultiCandidate keep-best key (spec §5 oversample->filter; the de-gaming re-rank lever).

    Re-ranks oversampled inverse-folding candidates AWAY from the low-complexity poly-Ala /
    homopolymer basin that gamed Chai's helix optimism (ADR-007), keeping the diverse,
    non-degenerate candidate. Higher = better. It is a PURE function of the sequence (no
    structure, no Chai call), so wiring it into MultiCandidate multiplies only the cheap
    LigandMPNN forward passes — never the GPU Chai predicts. This directly attacks the ~20%
    chirality-clean+composition-passing yield (the single biggest "more good designs" lever
    per the 2026-06-15 feature-map) without any extra prediction cost.

    Returns ``sequence_complexity(seq)`` minus a penalty per low-complexity hallmark, using the
    SAME thresholds as the design driver's composition veto (single-AA >0.30; Ala+Gly >0.40;
    any homopolymer run >= 4): a clean diverse sequence -> its entropy in [0, 1]; a degenerate
    one -> negative. Empty input -> -1.0 (never selected when any real candidate exists)."""
    from collections import Counter

    s = (seq or "").upper()
    if not s:
        return -1.0
    n = len(s)
    counts = Counter(s)
    penalty = 0.0
    if max(counts.values()) / n > 0.30:
        penalty += 1.0
    if (counts.get("A", 0) + counts.get("G", 0)) / n > 0.40:
        penalty += 1.0
    run = max_run = 1
    for i in range(1, n):
        run = run + 1 if s[i] == s[i - 1] else 1
        max_run = max(max_run, run)
    if max_run >= 4:
        penalty += 1.0
    # S3a.6 (#4): de-gaming entropy penalty (flag-gated). Beyond the run-length rule, penalise a
    # sequence whose normalized Shannon entropy is low (a repeating few-letter "diverse-looking" seq
    # still games Chai). Flag OFF keeps the legacy key byte-identical for the default suite.
    # Raising the oversample count (num_seqs) to further de-game is a cfg override (--loop.num_seqs),
    # not a code change — see build_loop_fn in seq_stage.py for the de-gaming rationale.
    import os
    if os.environ.get("XENO_SEQ_STAGE", "0") != "0":
        ent = sequence_complexity(s)
        if ent < 0.5:                       # below half of the max 20-letter entropy -> graded penalty
            penalty += (0.5 - ent)          # smooth, never a cliff; subtracted from the complexity below
    return sequence_complexity(s) - penalty


def select_topk(items: Sequence[T], key: Callable[[T], float], frac: float) -> list[T]:
    """Return the top `frac` of items by `key` (descending). Always returns >= 1 item."""
    if not items:
        return []
    ordered = sorted(items, key=key, reverse=True)
    n = max(1, math.ceil(len(items) * frac))
    return ordered[:n]


@dataclass
class SSBiasConfig:
    """Per-case secondary-structure bias (#21, lane A / #20 umbrella).

    Expresses the per-case SS objective as proximity to a target helix fraction:

    * alpha case (D/L-ABLE): target_helix_frac=1.0 with weight>0 -> REWARD helix.
    * cyclic / non-alpha case (6UFA, 9DXX knottin): target_helix_frac=0.0 with weight>0
      -> anti-alpha PENALTY (more helix scores lower).

    Default (weight=0.0) is NEUTRAL: contributes 0.0 regardless of helix fraction, so it is
    a no-op until a case config sets a weight.
    """

    target_helix_frac: float = 0.0
    weight: float = 0.0


def ss_bias_score(helix_frac: float, config: SSBiasConfig) -> float:
    """Per-case SS-bias contribution: weight * (1 - |helix_frac - target_helix_frac|).

    Higher = closer to the case's desired secondary-structure profile. Usable BOTH as an
    additive term in a composite score AND as a standalone semi-greedy selection key (rank
    candidates by ss_bias_score to push toward / away from helix per case). Returns 0.0 when
    config.weight == 0.0 (neutral default).
    """
    if config.weight == 0.0:
        return 0.0
    proximity = 1.0 - abs(float(helix_frac) - float(config.target_helix_frac))
    return float(config.weight) * proximity
