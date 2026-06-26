from pathlib import Path
import numpy as np
from xenodesign.backends.chai_backend import Prediction, write_inputs


def test_write_inputs_creates_fasta(tmp_path):
    entities = [
        {"type": "protein", "name": "target", "sequence": "MAK", "chirality": "L"},
        {"type": "protein", "name": "binder", "sequence": "ACG", "chirality": "D"},
    ]
    fasta_path = write_inputs(entities, tmp_path)
    assert Path(fasta_path).exists()
    text = Path(fasta_path).read_text()
    assert ">protein|binder" in text
    assert "(DAL)(DCY)G" in text  # glycine anchors chai tokenization


def test_prediction_dataclass_roundtrip():
    coords = np.zeros((5, 3))
    pred = Prediction(coords=coords, plddt=np.ones(5), iptm=0.9, token_index=np.arange(5))
    assert pred.iptm == 0.9
    assert pred.coords.shape == (5, 3)
    assert pred.plddt.mean() == 1.0


from xenodesign.eval.gate_tier0a import GateResult, aggregate_gate_report


def test_aggregate_gate_report_pass():
    # 100 stereocenters, 2 violations -> 2% << 51% -> PASS; phi/psi all within tol.
    res = aggregate_gate_report(
        chirality_violations=2, n_stereocenters=100,
        phi_psi_violations=1, n_torsions=100,
        violation_threshold=0.51,
    )
    assert isinstance(res, GateResult)
    assert res.chirality_violation_frac == 0.02
    assert res.passed is True


def test_aggregate_gate_report_fail_on_high_chirality_violation():
    res = aggregate_gate_report(
        chirality_violations=60, n_stereocenters=100,
        phi_psi_violations=0, n_torsions=100,
        violation_threshold=0.51,
    )
    assert res.passed is False


def test_aggregate_gate_report_handles_zero_division():
    res = aggregate_gate_report(0, 0, 0, 0, violation_threshold=0.51)
    assert res.chirality_violation_frac == 0.0
    assert res.passed is True


# ── P3a: ChaiBackend.predict threads msa_directory + constraint_path to run_inference ──

def _fake_chai_predict(monkeypatch, tmp_path):
    """Install a fake chai_lab.chai1.run_inference + no-op post-processing; return the
    kwargs-capture dict so a caller can assert what ChaiBackend.predict forwarded."""
    import sys
    import types

    import xenodesign.backends.chai_backend as cb

    captured: dict = {}

    fake_chai1 = types.ModuleType("chai_lab.chai1")

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return object()  # opaque fake StructureCandidates

    fake_chai1.run_inference = fake_run_inference
    sys.modules.setdefault("chai_lab", types.ModuleType("chai_lab"))
    monkeypatch.setitem(sys.modules, "chai_lab.chai1", fake_chai1)
    # short-circuit the on-disk post-processing (no real CIF/scores to parse)
    sentinel = object()
    monkeypatch.setattr(cb, "_save_confidence_npz", lambda *a, **k: None)
    monkeypatch.setattr(cb, "load_prediction", lambda *a, **k: sentinel)
    return captured, sentinel, cb


def test_predict_threads_msa_directory(tmp_path, monkeypatch):
    captured, sentinel, cb = _fake_chai_predict(monkeypatch, tmp_path)
    backend = cb.ChaiBackend(device="cpu", seed=7)
    entities = [{"type": "protein", "name": "t", "sequence": "MAK", "chirality": "L"}]
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()

    out = backend.predict(entities, tmp_path / "out", msa_directory=msa_dir)
    assert out is sentinel
    assert captured["msa_directory"] == Path(msa_dir)
    # use_msa_server stays False so chai reads the LOCAL MSAs only (never the network).
    assert captured["use_msa_server"] is False


def test_predict_msa_directory_defaults_none(tmp_path, monkeypatch):
    captured, _sentinel, cb = _fake_chai_predict(monkeypatch, tmp_path)
    backend = cb.ChaiBackend(device="cpu", seed=7)
    entities = [{"type": "protein", "name": "t", "sequence": "MAK", "chirality": "L"}]
    backend.predict(entities, tmp_path / "out")
    assert captured["msa_directory"] is None        # MSA-free default (unchanged behaviour)
    assert captured["constraint_path"] is None
