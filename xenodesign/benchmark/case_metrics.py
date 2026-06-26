"""Integration glue: run P1 interface metrics for a benchmark case and compare to its baseline
(spec §8, tracker #5). CPU-only — given a prediction_dir (fed by a test or, later, by a GPU run),
NEVER runs Chai. GPU case-runs are DEFERRED.

For an INTERFACE case (alpha / nonalpha) we run metrics.score_interface over the best model and
also surface Chai's per_chain_pair_iptm (interface ipTM) from the scores npz if present, then
compute a vs_baseline delta. For a SINGLE-CHAIN case (cyclic) there is no interface -> we skip the
interface metrics and return a note (backbone-RMSD / P_near geometry recovery is P6/P7, GPU).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from xenodesign import metrics

# Chai inference is GPU-non-deterministic (~0.02 ipTM run-to-run, spec §8): a design only
# "beats baseline" if it clears the baseline by MORE than this margin.
IPTM_NONDETERMINISM_MARGIN = 0.02


def _best_model_idx(prediction_dir: Path) -> int:
    """The model index with the highest aggregate_score (chai's own ranking); 0 if no scores."""
    best_idx, best_agg = 0, -np.inf
    for f in sorted(prediction_dir.glob('scores.model_idx_*.npz')):
        agg = float(np.asarray(np.load(f)['aggregate_score']).reshape(-1)[0])
        if agg > best_agg:
            best_agg = agg
            best_idx = int(re.search(r'idx_(\d+)', f.name).group(1))
    return best_idx


def _interface_iptm_from_scores(prediction_dir: Path, idx: int):
    """Read chai's per_chain_pair_iptm[0,0,1] (interface ipTM) from scores.model_idx_{idx}.npz,
    or None if the file/key is absent. Chai handles chains correctly here (spec §8: trust it)."""
    f = prediction_dir / f'scores.model_idx_{idx}.npz'
    if not f.exists():
        return None
    d = np.load(f)
    if 'per_chain_pair_iptm' not in d.files:
        return None
    arr = np.asarray(d['per_chain_pair_iptm'], dtype=float)
    if arr.ndim == 3:
        return float(arr[0, 0, 1])
    if arr.ndim == 2:
        return float(arr[0, 1])
    m = arr.reshape(-1)
    return float(m[1]) if m.size > 1 else None


def case_metrics(case, prediction_dir) -> dict:
    """Per-case interface metric bundle vs the case baseline (spec §8).

    Args:
        case: a BenchmarkCase (from benchmark.cases).
        prediction_dir: a directory holding confidence.model_idx_*.npz + pred.model_idx_*.cif
            (+ optional scores.model_idx_*.npz). Fed by a test or a GPU run; not produced here.

    Returns:
        {'case_id', 'model_idx', 'metrics': {...}, 'vs_baseline': {...}, 'note': str}.
        For a single-chain case (no interface) the interface metrics are None with a note.
    """
    prediction_dir = Path(prediction_dir)
    idx = _best_model_idx(prediction_dir)
    note = ''

    if not case.target_chains:
        return {
            'case_id': case.case_id, 'model_idx': idx,
            'metrics': {'ipae_mean': None, 'ipsae': None, 'interface_iptm': None},
            'vs_baseline': {'baseline_backbone_rmsd': case.baseline.backbone_rmsd},
            'note': 'single-chain geometry-recovery case: no interface metrics in P3; '
                    'backbone-RMSD / P_near recovery is a GPU step (P6/P7).',
        }

    npz = prediction_dir / f'confidence.model_idx_{idx}.npz'
    cif = prediction_dir / f'pred.model_idx_{idx}.cif'
    bundle = dict(metrics.score_interface(npz, cif, chain_a=0, chain_b=1))

    iptm = _interface_iptm_from_scores(prediction_dir, idx)
    bundle['interface_iptm'] = iptm

    b = case.baseline
    vs = {'baseline_ipae': b.ipae, 'baseline_interface_iptm': b.interface_iptm,
          'baseline_chirality': b.chirality, 'baseline_ipsae_cut10': b.ipsae_cut10}
    if b.ipae is not None:
        vs['ipae_delta'] = b.ipae - bundle['ipae_mean']
    if b.interface_iptm is not None and iptm is not None:
        vs['iptm_delta'] = iptm - b.interface_iptm
    # #32: surface the Dunbrack ipSAE_cut10 delta (higher is better, so design - baseline).
    if b.ipsae_cut10 is not None and bundle.get('ipsae_cut10') is not None:
        vs['ipsae_cut10_delta'] = bundle['ipsae_cut10'] - b.ipsae_cut10

    return {'case_id': case.case_id, 'model_idx': idx,
            'metrics': bundle, 'vs_baseline': vs, 'note': note}


def beats_baseline(result: dict) -> bool:
    """True iff the measured interface ipTM clears the case baseline by MORE than Chai's
    ~0.02 non-determinism margin (spec §8 beat-by-margin). False if either value is missing.

    NOTE: this is the ipTM-ONLY criterion. The driver reports the full 3-criterion claim via
    ``beats_baseline_full`` (ipTM margin AND ipAE AND chirality) — do not confuse the two."""
    iptm = result.get('metrics', {}).get('interface_iptm')
    base = result.get('vs_baseline', {}).get('baseline_interface_iptm')
    if iptm is None or base is None:
        return False
    return iptm > base + IPTM_NONDETERMINISM_MARGIN


# The α driver prints "BEATS BASELINE (bar: ipTM>0.50 by >0.02, ipAE<10, chirality<=0.10)" —
# a 3-criterion CLAIM. beats_baseline (ipTM-only) enforced just ONE, so the printed line
# overstated the gate. These thresholds make the reported claim honest (spec §8).
IPAE_BEAT_THRESHOLD = 10.0       # interface ipAE must be STRICTLY below this (Bennett <10)
CHIRALITY_BEAT_THRESHOLD = 0.10  # designed-chain chirality violation must be AT MOST this


def beats_baseline_full(result: dict, selected_chirality) -> bool:
    """The HONEST 3-criterion baseline gate the α driver prints (spec §8). True iff ALL of:

      1. interface_iptm > baseline_interface_iptm + IPTM_NONDETERMINISM_MARGIN (the ipTM
         beat-by-margin, == ``beats_baseline``), AND
      2. ipae_mean < IPAE_BEAT_THRESHOLD (10.0; tighter interface than the loose baseline), AND
      3. selected_chirality <= CHIRALITY_BEAT_THRESHOLD (0.10; the design stays D-clean).

    Each criterion is guarded for a missing/None value -> the whole gate is False (an
    unmeasurable criterion can NEVER be claimed as cleared). ``selected_chirality`` is the
    chirality of the SELECTED design (the driver passes the panel-selected iter's value); it is
    NOT in ``result`` because case_metrics scores a structure, not the loop's chirality channel.
    """
    if not beats_baseline(result):
        return False
    ipae = result.get('metrics', {}).get('ipae_mean')
    if ipae is None or not (ipae < IPAE_BEAT_THRESHOLD):
        return False
    if selected_chirality is None or not (selected_chirality <= CHIRALITY_BEAT_THRESHOLD):
        return False
    return True
