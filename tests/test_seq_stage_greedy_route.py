"""make_alpha_seq_update_fn delegates to SequenceUpdate when XENO_SEQ_STAGE is on (default)."""
from __future__ import annotations

import numpy as np

import xenodesign.classes._alpha_internals as ai
import xenodesign.classes.alpha as alpha_mod


class _Wrapper:
    last_out_dir = "out"


def test_make_alpha_seq_update_routes_through_stage(monkeypatch):
    seen = {}
    real = ai.SequenceUpdate.build_loop_fn

    def _spy(self, extract_fn, chirality_pattern=None):
        seen["routed"] = True
        return real(self, extract_fn, chirality_pattern=chirality_pattern)

    monkeypatch.setattr(ai.SequenceUpdate, "build_loop_fn", _spy)
    # _make_base_backend is resolved via _self() -> alpha_mod (monkeypatch contract).
    monkeypatch.setattr(alpha_mod, "_make_base_backend",
                        lambda backend="ligandmpnn":
                        (lambda *a, **k: ["ACDEG" for _ in range(k.get("num_seqs", a[5] if len(a) > 5 else 1))]),
                        raising=True)
    # _stage_extract reads the CIF via _self() -> alpha_mod; fake the IO leaves it touches.
    monkeypatch.setattr(alpha_mod, "_best_cif_path", lambda *a, **k: "cif", raising=False)
    monkeypatch.setattr("xenodesign.eval.gate_tier0a.backbone_by_residue_from_cif",
                        lambda cif, chain: [object()] * 5)
    monkeypatch.setattr(ai, "_backbone_array_from_residues",
                        lambda res: np.zeros((len(res), 4, 3)), raising=False)
    monkeypatch.setattr(alpha_mod, "_all_atoms_from_chain",
                        lambda cif, chain: (np.zeros((0, 3)), []), raising=False)
    monkeypatch.setattr(alpha_mod, "binder_seq_from_cif", lambda cif, chain: "ACDEG", raising=False)

    fn = ai.make_alpha_seq_update_fn(_Wrapper(), num_seqs=4, backend="ligandmpnn")
    out = fn(prediction=object())
    assert seen.get("routed") is True
    assert isinstance(out, str) and len(out) == 5
