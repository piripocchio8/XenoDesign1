"""CPU contract tests for parsing a real Chai-1 output directory into a Prediction.

Pins `load_prediction` (the offline, chai_lab-free disk parser) against the committed
fixture `data/benchmark/chai_output_fixture/`, so the GPU `predict` path can be verified
without a GPU. Contract: docs/benchmark-prior-art-map.md §2.
"""
import glob

import numpy as np
import pytest

from xenodesign.backends.chai_backend import Prediction, load_prediction, per_chain_plddt

FIXTURE = "data/benchmark/chai_output_fixture"


def _fixture_aggregates():
    aggs = []
    for f in sorted(glob.glob(f"{FIXTURE}/scores.model_idx_*.npz")):
        aggs.append(float(np.load(f)["aggregate_score"].reshape(-1)[0]))
    return aggs


def test_load_prediction_returns_prediction():
    assert isinstance(load_prediction(FIXTURE), Prediction)


def test_load_prediction_selects_best_model_by_aggregate_score():
    pred = load_prediction(FIXTURE)
    assert pred.aggregate_score == pytest.approx(max(_fixture_aggregates()))


def test_load_prediction_recovers_scalar_scores():
    pred = load_prediction(FIXTURE)
    assert isinstance(pred.iptm, float)
    assert isinstance(pred.ptm, float)
    assert pred.iptm == pytest.approx(0.0)  # single-chain fixture: no interface
    assert 0.80 < pred.ptm < 0.81


def test_load_prediction_clash_flag_is_python_bool():
    pred = load_prediction(FIXTURE)
    assert isinstance(pred.has_inter_chain_clashes, bool)
    assert pred.has_inter_chain_clashes is False  # every fixture model: no clash


def test_load_prediction_coords_from_cif():
    pred = load_prediction(FIXTURE)
    assert pred.coords.shape == (223, 3)


def test_load_prediction_plddt_per_residue():
    pred = load_prediction(FIXTURE)
    assert pred.plddt.shape == (30,)
    assert 60.0 < float(pred.plddt.mean()) < 75.0


def test_load_prediction_chain_bookkeeping_single_chain():
    pred = load_prediction(FIXTURE)
    # token_index carries per-residue chain bookkeeping (not a fake arange).
    assert pred.token_index.shape == (30,)
    assert len(np.unique(pred.token_index)) == 1


def test_per_chain_plddt_single_chain_matches_global_mean():
    pred = load_prediction(FIXTURE)
    pc = per_chain_plddt(pred)
    assert set(pc) == {0}
    assert pc[0] == pytest.approx(float(pred.plddt.mean()))
