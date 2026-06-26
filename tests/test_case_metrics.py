# tests/test_case_metrics.py
from pathlib import Path

import numpy as np

from xenodesign.benchmark.case_metrics import (
    beats_baseline, beats_baseline_full, case_metrics,
)
from xenodesign.benchmark.cases import get_case


def _write_two_chain_prediction(d: Path, inter_chain_pae: float = 6.5):
    """Write a synthetic 2-chain prediction. ``inter_chain_pae`` sets the off-diagonal
    inter-chain PAE block so ipae_mean is controllable (default 6.5 -> below the <10 beat bar;
    pass a large value, e.g. 30.0, to exercise a 'good ipTM but BAD ipAE' case)."""
    lo = inter_chain_pae - 0.5
    hi = inter_chain_pae + 0.5
    pae = np.array([
        [0.0, 0.5, 0.5, lo, hi],
        [0.5, 0.0, 0.5, lo, hi],
        [0.5, 0.5, 0.0, lo, hi],
        [lo, lo, lo, 0.0, 0.5],
        [hi, hi, hi, 0.5, 0.0],
    ])
    np.savez(d / 'confidence.model_idx_0.npz', pae=pae,
             token_asym_id=np.array([0, 0, 0, 1, 1]),
             token_residue_index=np.array([0, 1, 2, 0, 1]))
    lines = [
        'data_test', '#', 'loop_',
        '_atom_site.group_PDB', '_atom_site.id', '_atom_site.type_symbol',
        '_atom_site.label_atom_id', '_atom_site.label_alt_id', '_atom_site.label_comp_id',
        '_atom_site.label_seq_id', '_atom_site.label_asym_id',
        '_atom_site.Cartn_x', '_atom_site.Cartn_y', '_atom_site.Cartn_z',
        '_atom_site.B_iso_or_equiv',
    ]
    atoms = [
        ('A', 1, 0.0, 0.0, 0.0, 80.0), ('A', 2, 3.8, 0.0, 0.0, 82.0),
        ('A', 3, 7.6, 0.0, 0.0, 78.0),
        ('B', 1, 1.0, 5.0, 0.0, 70.0), ('B', 2, 4.0, 5.0, 0.0, 72.0),
    ]
    aid = 1
    for ch, seq, x, y, z, b in atoms:
        lines.append(f'ATOM {aid} C CA . ALA {seq} {ch} {x} {y} {z} {b}')
        aid += 1
    lines.append('#')
    (d / 'pred.model_idx_0.cif').write_text('\n'.join(lines) + '\n')


def test_case_metrics_interface_bundle_and_vs_baseline(tmp_path):
    _write_two_chain_prediction(tmp_path)
    case = get_case('alpha')
    out = case_metrics(case, tmp_path)
    assert 'ipae_mean' in out['metrics'] and out['metrics']['ipae_mean'] >= 0.0
    assert 'ipsae' in out['metrics']
    # #33 (corrected): alpha baseline ipae is the genuine PER-TOKEN GT 12.2 (token_asym split;
    # the 8.59 was an NA=21 chain-split artifact). Matches metrics.score_interface on the same split.
    assert out['vs_baseline']['baseline_ipae'] == 12.2
    assert 'ipae_delta' in out['vs_baseline']
    # #32: the Dunbrack ipSAE_cut10 is surfaced through the bundle and gets a vs-baseline delta.
    assert 'ipsae_cut10' in out['metrics']
    assert out['vs_baseline']['baseline_ipsae_cut10'] == 0.22
    assert 'ipsae_cut10_delta' in out['vs_baseline']
    assert out['case_id'] == 'alpha'


def test_case_metrics_reads_interface_iptm_from_scores_if_present(tmp_path):
    _write_two_chain_prediction(tmp_path)
    np.savez(tmp_path / 'scores.model_idx_0.npz',
             aggregate_score=np.array([0.9]), ptm=np.array([0.8]), iptm=np.array([0.5]),
             per_chain_pair_iptm=np.array([[[0.0, 0.61], [0.61, 0.0]]]),
             has_inter_chain_clashes=np.array([False]))
    case = get_case('alpha')
    out = case_metrics(case, tmp_path)
    assert out['metrics']['interface_iptm'] == 0.61
    assert beats_baseline(out) is True


def test_case_metrics_single_chain_case_skips_interface(tmp_path):
    _write_two_chain_prediction(tmp_path)
    case = get_case('cyclic')
    out = case_metrics(case, tmp_path)
    assert out['metrics'].get('ipae_mean') is None
    assert 'single-chain' in out['note'].lower()


def test_beats_baseline_false_when_below_margin(tmp_path):
    _write_two_chain_prediction(tmp_path)
    np.savez(tmp_path / 'scores.model_idx_0.npz',
             aggregate_score=np.array([0.9]), ptm=np.array([0.8]), iptm=np.array([0.5]),
             per_chain_pair_iptm=np.array([[[0.0, 0.45], [0.45, 0.0]]]),
             has_inter_chain_clashes=np.array([False]))
    out = case_metrics(get_case('alpha'), tmp_path)
    assert beats_baseline(out) is False


# ── FIX #2: beats_baseline_full is the HONEST 3-criterion gate the driver prints ─────────

def _write_scores(d: Path, iptm: float):
    np.savez(d / 'scores.model_idx_0.npz',
             aggregate_score=np.array([0.9]), ptm=np.array([0.8]), iptm=np.array([0.5]),
             per_chain_pair_iptm=np.array([[[0.0, iptm], [iptm, 0.0]]]),
             has_inter_chain_clashes=np.array([False]))


def test_beats_baseline_full_true_when_all_three_criteria_clear(tmp_path):
    """ipTM clears the baseline by > margin, ipAE 6.5 < 10, chirality 0.0 <= 0.10 -> True."""
    _write_two_chain_prediction(tmp_path, inter_chain_pae=6.5)   # ipae_mean 6.5 (< 10)
    _write_scores(tmp_path, iptm=0.61)                            # 0.61 > 0.44 + 0.02
    out = case_metrics(get_case('alpha'), tmp_path)
    assert beats_baseline(out) is True
    assert beats_baseline_full(out, selected_chirality=0.0) is True


def test_beats_baseline_full_false_when_ipae_is_bad(tmp_path):
    """CORE FIX #2 case: a GOOD ipTM (clears the margin, so the ipTM-only beats_baseline is
    True) but a BAD ipAE (>=10) -> the honest 3-criterion full gate must be False. This is the
    overstatement the old ipTM-only print hid."""
    _write_two_chain_prediction(tmp_path, inter_chain_pae=30.0)  # ipae_mean 30.0 (>= 10)
    _write_scores(tmp_path, iptm=0.61)                           # good ipTM, clears the margin
    out = case_metrics(get_case('alpha'), tmp_path)
    # ipTM-only would (mis)claim a win:
    assert beats_baseline(out) is True
    # but the full 3-criterion gate is honest about the bad ipAE:
    assert beats_baseline_full(out, selected_chirality=0.0) is False


def test_beats_baseline_full_false_when_chirality_is_bad(tmp_path):
    """Good ipTM + good ipAE but a chirality violation > 0.10 -> full gate False."""
    _write_two_chain_prediction(tmp_path, inter_chain_pae=6.5)
    _write_scores(tmp_path, iptm=0.61)
    out = case_metrics(get_case('alpha'), tmp_path)
    assert beats_baseline(out) is True
    assert beats_baseline_full(out, selected_chirality=0.25) is False
    # the boundary 0.10 is inclusive (<=):
    assert beats_baseline_full(out, selected_chirality=0.10) is True


def test_beats_baseline_full_false_when_iptm_below_margin(tmp_path):
    """A weak ipTM (below the margin) fails the full gate even with good ipAE/chirality."""
    _write_two_chain_prediction(tmp_path, inter_chain_pae=6.5)
    _write_scores(tmp_path, iptm=0.45)   # 0.45 is NOT > 0.44 + 0.02
    out = case_metrics(get_case('alpha'), tmp_path)
    assert beats_baseline(out) is False
    assert beats_baseline_full(out, selected_chirality=0.0) is False


def test_beats_baseline_full_false_when_chirality_is_none(tmp_path):
    """A missing (None) selected chirality can NEVER be claimed clear -> full gate False."""
    _write_two_chain_prediction(tmp_path, inter_chain_pae=6.5)
    _write_scores(tmp_path, iptm=0.61)
    out = case_metrics(get_case('alpha'), tmp_path)
    assert beats_baseline_full(out, selected_chirality=None) is False


def test_beats_baseline_full_false_when_iptm_missing(tmp_path):
    """No scores npz -> interface_iptm None -> full gate False (unmeasurable, never claimed)."""
    _write_two_chain_prediction(tmp_path, inter_chain_pae=6.5)   # no scores npz written
    out = case_metrics(get_case('alpha'), tmp_path)
    assert out['metrics']['interface_iptm'] is None
    assert beats_baseline_full(out, selected_chirality=0.0) is False
