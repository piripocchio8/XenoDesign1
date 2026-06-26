"""Benchmark case registry + per-case config (spec §2, tracker #5).

Each BenchmarkCase pins a fixed TARGET + a designed BINDER, a chirality class, a measured
BASELINE-TO-BEAT, an optional restraint spec, and per-case knobs (spec §6 overrides). Only
the binder is ever designed; the target is fixed context.

CPU-only data module — no chai/torch import. The unpublished ALPHA sequences are NOT inlined:
the registry holds only the gitignored FASTA path + lengths + measured baseline numbers
(spec §2.1 gitignored caveat). The NON-ALPHA 9DXX target-gate (#29) is RESOLVED (spec §2.3,
2026-06-14): the HA target is prepared with a precomputed MSA (target_prep='msa'); the binder
is designed MSA-free as usual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from xenodesign.config import local_ref


@dataclass(frozen=True)
class BaselineMetrics:
    """The measured baseline-to-beat for a case (spec §2 / §8). Values a design must
    clear by more than Chai's ~0.02 ipTM non-determinism to count as a win.

    interface_iptm: Chai per_chain_pair_iptm at the binder/target interface.
    ipae: mean inter-chain PAE (Bennett pae_interaction), A.
    ipsae_cut10: Dunbrack ipSAE at the 10-A cutoff (interface-localised pSAE); None unless measured.
    chirality: chirality-violation score of the designed chain (0.0 = clean).
    backbone_rmsd: optional Ca/heavy-atom RMSD-to-deposit (geometry-recovery cases).
    notes: free-text provenance.
    """
    interface_iptm: Optional[float] = None
    ipae: Optional[float] = None
    ipsae_cut10: Optional[float] = None
    chirality: Optional[float] = None
    backbone_rmsd: Optional[float] = None
    notes: str = ''


@dataclass(frozen=True)
class RestraintSpec:
    """A per-case Chai .restraints intent (spec §2 knobs, #27). `kind` selects the builder
    in benchmark/restraints.py; `params` carries its kwargs. Validated/emitted there, not here.

    kind: 'pin_polarity' | 'contact' | 'pocket' | 'metal_coordination'.
    params: builder kwargs (e.g. anchor residues, distances, confidence).
    """
    kind: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark case (spec §2). Only the binder is designed; the target is fixed context.

    case_id: short stable id ('alpha' | 'cyclic' | 'nonalpha').
    description: human summary.
    chirality_class: 'all_D' | 'mixed' (L+D) — drives reflection/glycine handling.
    target_chains: tuple of fixed target chain ids.
    binder_chain: the single designed chain id.
    binder_length: designed-chain length (matched to the reference binder).
    baseline: BaselineMetrics (the baseline-to-beat).
    restraint: optional RestraintSpec (per-case knob).
    knobs: per-case tuning overrides (spec §6).
    fasta_path: gitignored input FASTA path (NEVER inline the sequences).
    cofactor: optional cofactor descriptor (e.g. 'Zn' for the cyclic case), else None.
    target_prep: 'fixed_input' (alpha/cyclic) | 'msa' (9DXX — target uses precomputed MSA; gate #29 RESOLVED per spec §2.3).
    """
    case_id: str
    description: str
    chirality_class: str
    target_chains: tuple
    binder_chain: str
    binder_length: int
    baseline: BaselineMetrics
    restraint: Optional[RestraintSpec] = None
    knobs: dict = field(default_factory=dict)
    fasta_path: str = ''
    cofactor: Optional[str] = None
    target_prep: str = 'fixed_input'


# --- ALPHA: trimer D/L-ABLE (the prove-it case; GROUND TRUTH GENERATED) ---
# Designed binder = chain A (21-res D-helix); fixed target = chain B (41-res L-HLH).
# BASELINE = the GENUINE PER-TOKEN Chai ground-truth (#33, corrected 2026-06-15): interface
# ipTM 0.434 (range 0.43-0.45), ipAE 12.2 A, Dunbrack ipSAE_cut10 ~0.22. chain-A chirality 0.000.
# CHAIN-SPLIT CAVEAT (load-bearing): the gitignored pae_summary.json shows ipAE 8.59 / ipsae_cut10
# 0.73 — those are NA=21 chain-split ARTIFACTS. run_groundtruth_pae.py split at residue-count 21,
# but Chai ATOM-TOKENIZES the D-helix to 151 tokens, so the true chain A/B asym boundary is at
# token 151, not 21 (splitting at 21 cuts mid-D-helix → a fake intra-chain "interface"). metrics
# .score_interface uses token_asym_id (the correct split) → 12.2 / ~0.22; the baseline below matches
# it so case_metrics vs_baseline deltas are meaningful. interface_iptm 0.44 is chai-native (split-
# independent). Polarity intent: A N-terminus -> HLH loop, antiparallel to helix-1.
_ALPHA = BenchmarkCase(
    case_id='alpha',
    description='trimer D/L-ABLE: 21-res all-D helix binder vs 41-res L-HLH target; '
                'tighten the interface (ipTM>0.50, ipAE<10) keeping chirality clean.',
    chirality_class='all_D',
    target_chains=('B',),
    binder_chain='A',
    binder_length=21,
    baseline=BaselineMetrics(
        interface_iptm=0.44, ipae=12.2, ipsae_cut10=0.22, chirality=0.0,
        notes='GENUINE per-token Chai ground-truth (#33, corrected 2026-06-15) on the '
              'token_asym_id chain split (D-helix atom-tokenizes to 151 tokens; true asym '
              'boundary at token 151, NOT residue-count 21). interface ipTM mean 0.434 '
              '(range 0.43-0.45; chai per_chain_pair_iptm, split-independent); ipAE ~12.2 A '
              '(per-token; matches spec §2.1 and metrics.score_interface); Dunbrack ipsae_cut10 '
              '~0.22 (per-token). 19/19 D clean; loose interface (above Bennett <10), room to '
              'tighten. The gitignored pae_summary.json 8.59/0.73 are NA=21 chain-split '
              'artifacts — do NOT use (see comment above).',
    ),
    restraint=RestraintSpec(
        kind='pin_polarity',
        params={'binder_chain': 'A', 'binder_anchor_resnum': 1,
                'target_chain': 'B', 'target_anchor_resnum': 21,
                'max_distance': 12.0, 'confidence': 0.5},
    ),
    knobs={'binding_weight': 0.45, 'ss_bias': 'helix_ok'},
    fasta_path=str(local_ref('dl_able_ground_truth', 'trimer_DL_ABLE.fasta')),
    cofactor=None,
    target_prep='fixed_input',
)


# --- CYCLIC: 6UFA (Zn macrocycle; metal coordination) ---
# Single-chain holo geometry-recovery (spec §2.2): recover the deposited macrocycle + the
# tetrahedral [Zn(L-His)2(D-His)2] site. NO interface. Baseline-to-beat = backbone heavy-atom
# RMSD <= 1 A to the 0.77 A deposit + Zn-coordination geometry. Mixed chirality (L+D His);
# first-shell coordinating His fixed across cycles. 0.10-noise checkpoint is a GPU run-knob.
# His<->Zn restraint (#27) drives diffusion into the coordinated subspace. NOTE: 6UFA is the
# FULL 24-mer (S2-symmetric repeat of the 12-mer); the 4 coordinating His are 6/12/18/24, L/D/L/D.
# his_resnums RE-SYNCED 2026-06-24 to the FULL DEPOSIT: 6UFA is S2-symmetric — the deposit is a
# 24-mer (the 12-mer repeat unit TWICE). Modeling only the single 12-mer was WRONG: a 12-mer
# carries only 2 of the 4 coordinating His and CANNOT form a 4-coordinate [Zn(His)4] site. We now
# model the full 24-mer with all FOUR coordinating His — 6/12/18/24, chirality L/D/L/D — giving the
# real tetrahedral [Zn(L-His)2(D-His)2] geometry (Zn-N ~2.02 A). (Was the 12-mer (6,12).)
_CYCLIC = BenchmarkCase(
    case_id='cyclic',
    description='6UFA Zn macrocycle: single-chain holo geometry-recovery of the deposited '
                '24-mer ring + tetrahedral [Zn(L-His)2(D-His)2] site (backbone RMSD<=1 A).',
    chirality_class='mixed',
    target_chains=(),
    binder_chain='A',
    binder_length=24,
    baseline=BaselineMetrics(
        backbone_rmsd=1.0, chirality=None,
        notes='recover the 0.77 A 6UFA deposit; Zn-N coordination geometry is a secondary '
              'metric; P_near>=0.9 added to the metric stack in P6/P7.',
    ),
    restraint=RestraintSpec(
        kind='metal_coordination',
        params={'metal_chain': 'B', 'metal_resnum': 1, 'metal_atom': 'ZN',
                'his_chain': 'A', 'his_resnums': (6, 12, 18, 24),
                'max_distance': 2.6, 'confidence': 0.8},
    ),
    knobs={'checkpoint_noise': 0.10, 'fix_first_shell': True,
           'ss_bias': 'anti_alpha', 'pre_relax': True},
    fasta_path='',
    cofactor='Zn',
    target_prep='fixed_input',
)


# --- NON-ALPHA: 9DXX (D-knottin : influenza HA) ---
# Genuine 2-chain L-target + D-binder complex (spec §2.3): target = one HA protomer (~504 aa);
# binder = DP93, a 31-res all-D cystine-knot peptide (non-helical). Post-cutoff novelty test.
#
# GATE #29 RESOLVED (spec §2.3, 2026-06-14): MSA-free Chai fails the HA target (pTM 0.43,
# HA1-HA2 ipTM 0.15, HA1 Ca-RMSD 18.8 A) but with an MSA reproduces it near-natively (pTM
# 0.92, HA1-HA2 ipTM 0.90, HA1 1.7 A / HA2 0.6 A). Fix: the fixed HA target is given a
# precomputed MSA (cached at .../9dxx_target_gate/chai_pred_msa/msas/); the D-binder is
# designed MSA-free as usual. target_prep='msa' encodes this per-case adaptation.
# Binder baseline (interface_iptm) stays None — that's a separate future measurement.
# Binder length 31 (DP93); SS-bias anti-alpha; weak pocket restraint to the HA-stem epitope
# (params TBD on GT, #27). Disulfide / ncAA handling and PepMLM seeding are P4/P6, not here.
_NONALPHA = BenchmarkCase(
    case_id='nonalpha',
    description='9DXX D-knottin:HA — 31-res all-D cystine-knot binder (DP93) vs an HA '
                'protomer target. Target uses precomputed MSA (gate #29 resolved, spec §2.3); '
                'binder designed MSA-free.',
    chirality_class='all_D',
    target_chains=('B',),
    binder_chain='A',
    binder_length=31,
    baseline=BaselineMetrics(
        interface_iptm=None, ipae=None, backbone_rmsd=None,
        notes='baseline-to-beat = binder Ca-RMSD + interface ipTM/ipSAE vs the 2.37 A crystal; '
              'PNAS-2025 design model is the prior-art output to beat. Binder baseline numbers '
              'are a separate future measurement (gate #29 resolved target prep only).',
    ),
    restraint=RestraintSpec(
        kind='pocket',
        params={'binder_chain': 'A', 'target_chain': 'B', 'target_resnums': (),
                'max_distance': 14.0, 'confidence': 0.3, 'pending_gate': 29},
    ),
    knobs={'ss_bias': 'anti_alpha', 'disulfide_constraints': True},
    fasta_path='',
    cofactor=None,
    target_prep='msa',
)


CASES: dict = {'alpha': _ALPHA, 'cyclic': _CYCLIC, 'nonalpha': _NONALPHA}


def get_case(case_id: str) -> BenchmarkCase:
    """Look up a registered benchmark case by id. Raises KeyError listing known ids."""
    try:
        return CASES[case_id]
    except KeyError:
        raise KeyError(
            f'unknown benchmark case {case_id!r}; registered: {sorted(CASES)}'
        ) from None


# Map the per-case knobs['ss_bias'] tag to a target helix fraction (#21, P1b).
#   'helix_ok'   (α)                       -> 1.0 : reward a helical design
#   'anti_alpha' (cyclic 6UFA / 9DXX knot) -> 0.0 : penalise helix (anti-α)
_SS_BIAS_TARGETS = {'helix_ok': 1.0, 'anti_alpha': 0.0}


def ss_bias_config_for_case(case: BenchmarkCase, weight: float = 1.0):
    """Build the per-case secondary-structure bias config from ``case.knobs['ss_bias']``.

    Returns an ``scorer.SSBiasConfig`` whose ``target_helix_frac`` encodes the case's desired
    SS (helix for α, no-helix for the anti-α cases). An absent/unknown tag -> the NEUTRAL config
    (weight 0.0), so the SS-bias term is a no-op for cases that don't request it. `weight` sets
    the config's standalone weight (used by ``scorer.ss_bias_score`` as a semi-greedy key); the
    JudgePanel ignores it and takes its magnitude from ``weights['ss_bias']`` instead."""
    from xenodesign.scorer import SSBiasConfig

    tag = case.knobs.get('ss_bias')
    if tag not in _SS_BIAS_TARGETS:
        return SSBiasConfig()  # neutral (weight 0.0)
    return SSBiasConfig(target_helix_frac=_SS_BIAS_TARGETS[tag], weight=weight)


def target_gate_note(case_id: str) -> str:
    """Return a human note iff the case's target prep is DEFERRED behind a gate, else ''.

    Gate #29 (9DXX HA1/HA2 strand-swap reproducibility) is RESOLVED (spec §2.3): the HA target
    now uses a precomputed MSA (target_prep='msa'), so nonalpha returns '' here. This refusal
    only fires for a case still marked target_prep=='pending_gate_29' (none at present); it keeps
    a runner from launching a design against an unprepared target.
    """
    case = get_case(case_id)
    if case.target_prep == 'pending_gate_29':
        return (
            f'case {case_id!r}: target prep DEFERRED behind gate #29 — predict the HA target '
            f'ALONE and verify Chai reproduces the deposited HA1/HA2 strand-swap interface '
            f'before designing a binder. Not implemented in P3 (scaffolding only).'
        )
    return ''
