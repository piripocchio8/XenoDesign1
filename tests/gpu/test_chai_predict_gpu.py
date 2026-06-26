"""GPU smoke test: Chai-1 forward prediction of a small D-peptide : L-target complex.

Run on your local GPU:  pytest tests/gpu/test_chai_predict_gpu.py -m gpu -v

Validates the BEST-EFFORT ChaiBackend.predict + _to_prediction path (parsing
StructureCandidates -> Prediction). If chai-lab's API differs in your version, this is
where you'll see it; adjust xenodesign/backends/chai_backend.py accordingly.
"""
import pytest

from tests.gpu.conftest import require_chai, require_cuda


@pytest.mark.gpu
def test_predict_returns_wellformed_prediction(tmp_path):
    require_cuda()
    require_chai()
    from xenodesign.backends.chai_backend import ChaiBackend, Prediction

    entities = [
        {"type": "protein", "name": "target",
         "sequence": "GSHMKVLITGGAGFIGSHLVDRL", "chirality": "L"},
        {"type": "protein", "name": "binder",
         "sequence": "ACDEFGHIK", "chirality": "D"},
    ]
    backend = ChaiBackend(device="cuda:0", seed=0)
    # Fewer diffusion steps to keep the smoke test fast.
    pred = backend.predict(entities, tmp_path, num_diffn_timesteps=50)

    assert isinstance(pred, Prediction)
    assert pred.coords.ndim == 2 and pred.coords.shape[1] == 3
    assert pred.coords.shape[0] > 0
    assert pred.plddt.ndim == 1 and pred.plddt.shape[0] > 0
    assert 0.0 <= pred.iptm <= 1.0
