"""S0 characterization goldens: pin current CPU dispatch behavior before the S1 refactor.

Each test runs dispatch.run_design with a fixed seed through a deterministic CPU fake stack
(no GPU, no network) and asserts the report equals a committed golden JSON. Regenerate the
goldens (e.g. after an intentional behavior change) with:  XENO_REGOLD=1 pytest -k golden
The S1 parity tests reuse the SAME fakes + goldens, so any greedy-routing divergence shows up
here as a mismatched key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from xenodesign import dispatch
from xenodesign.config import resolve_config

_GOLDEN_DIR = Path(__file__).parent / "golden"


class _FakePred:
    """Deterministic Prediction stand-in: the attributes loop/objective/referee read."""
    coords = np.zeros((3, 3))
    iptm = 0.5
    token_index = np.array([1, 1, 1])
    plddt = np.array([80.0, 80.0, 80.0])


def _drop_runspecific(report: dict) -> dict:
    """Strip keys that are inherently run-specific (not behavior).

    ``msa_dir`` is a machine-local path (produced by local_ref() on the host where the golden was
    captured) and must not be committed to the golden — it would fail on any other machine.
    """
    out = dict(report)
    for k in ("wall_time_s", "out_dir", "constraint_path", "msa_dir"):
        out.pop(k, None)
    return out


def _load_or_regold(name: str, report: dict) -> dict:
    """Compare ``report`` against tests/golden/<name>.json, or (re)write it under XENO_REGOLD=1.

    Returns the golden dict to compare against. JSON round-trips the report so numpy scalars /
    tuples are normalized the same way on both sides (the on-disk golden is the source of truth).
    """
    path = _GOLDEN_DIR / f"{name}.json"
    normalized = json.loads(json.dumps(_drop_runspecific(report), default=str))
    if os.environ.get("XENO_REGOLD") == "1":
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, indent=2, sort_keys=True))
    if not path.exists():
        raise AssertionError(
            f"golden {path} missing; regenerate with XENO_REGOLD=1 pytest -k golden")
    return json.loads(path.read_text())


def test_golden_helper_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setenv("XENO_REGOLD", "1")
    got = _load_or_regold("_helper_selftest", {"a": 1, "wall_time_s": 9.9})
    assert got == {"a": 1}            # run-specific key dropped, value round-tripped


import xenodesign.classes.alpha as alpha_mod

_ALPHA_SEED = "ACDEFGHIKLMNPQRSTVWYG"   # fixed 21-mer (ends in the Gly anchor)
_ALPHA_TARGET = "GSHMKVLITGGAGFIGSHLVDRL"


def _alpha_fakes(monkeypatch):
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "target",
                                       "sequence": _ALPHA_TARGET, "chirality": "L"}], None, None))
    monkeypatch.setattr("xenodesign.seed.reflect_binder_in_complex_from_cif",
                        lambda *a, **k: np.zeros((3, 3)))
    monkeypatch.setattr(alpha_mod.Alpha, "seed",
                        lambda self, cfg, target_seq: __import__(
                            "xenodesign.classes.base", fromlist=["SeedSpec"]
                        ).SeedSpec(one_letter=_ALPHA_SEED))
    # Deterministic seq-update: re-emit the seed every iteration (no MPNN/GPU).
    monkeypatch.setattr(alpha_mod, "make_alpha_seq_update_fn",
                        lambda wrapper, **k: (lambda pred: _ALPHA_SEED))


def test_golden_alpha_greedy(tmp_path, monkeypatch):
    _alpha_fakes(monkeypatch)
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 2, "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False, "loop.backend": "ligandmpnn"})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("alpha_greedy", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden


import xenodesign.classes._alpha_internals as ai_mod
import xenodesign.classes.non_alpha as nonalpha_mod

_NONALPHA_SEED = "DEFCGHIKCLMNPCQRCSTVWYCDEDECFGG"   # 31-mer matching case binder_length=31; Cys at
# 0-indexed positions [3,8,13,16,22,27] == knottin_cys_positions(31) [4,9,14,17,23,28]; C-term G


def _nonalpha_fakes(monkeypatch):
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "ha1", "sequence": "AAAA"},
                                      {"type": "protein", "name": "ha2", "sequence": "CCCC"}],
                                     None, None))
    monkeypatch.setattr("xenodesign.seed.reflect_binder_in_complex_from_cif",
                        lambda *a, **k: np.zeros((3, 3)))
    monkeypatch.setattr(nonalpha_mod.NonAlpha, "seed",
                        lambda self, cfg, target_seq: __import__(
                            "xenodesign.classes.base", fromlist=["SeedSpec"]
                        ).SeedSpec(one_letter=_NONALPHA_SEED))
    monkeypatch.setattr(nonalpha_mod, "make_alpha_seq_update_fn",
                        lambda wrapper, **k: (lambda pred: _NONALPHA_SEED))


def test_golden_nonalpha_greedy(tmp_path, monkeypatch):
    _nonalpha_fakes(monkeypatch)
    cfg = resolve_config("non_alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"loop.iters": 2, "use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False, "loop.backend": "ligandmpnn"})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("nonalpha_greedy", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden


# ── S0.4 cyclic-metal greedy golden (coordinator/chirality path pinned) ───────────────────────────

# The declared 6UFA coordinators as 5-tuples (pos, one_letter, three_letter, chirality, atom).
# This is the EXACT in-memory format stored in cfg.restraint.params['coord_residues'] by the CLI.
# Token meaning (from xenodesign.coordinators.parse_coord_residues):
#   H6   -> (6,  'H', 'HIS', 'L', 'ND1')   L-His at position 6
#   DHI12 -> (12, 'H', 'DHI', 'D', 'ND1')  D-His at position 12
#   H18   -> (18, 'H', 'HIS', 'L', 'ND1')  L-His at position 18
#   DHI24 -> (24, 'H', 'DHI', 'D', 'ND1')  D-His at position 24
def _cyclic_metal_coord_residues():
    from xenodesign.coordinators import parse_coord_residues
    return [(c.pos, c.one_letter, c.three_letter, c.chirality, c.atom)
            for c in parse_coord_residues("H6,DHI12,H18,DHI24")]


def _fake_ifold_backend(design_backbone, context_coords, context_elements,
                        fixed_mask, temperature, num_seqs, known_seq=None):
    """Deterministic InverseFoldingBackend: echo known_seq (the real evolving seq) so the
    extract->known_seq->chirality round-trip is exercised without torch. Falls back to all-Ala."""
    n = np.asarray(design_backbone).shape[0]
    base = (known_seq or "A" * n)[:n].ljust(n, "A")
    return [base for _ in range(num_seqs)]


def _cyclic_metal_fakes(monkeypatch, tmp_path):
    """Patch only the GPU/IO leaves; real Cyclic.seq_update + make_alpha_seq_update_fn run."""
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr("xenodesign.seed.reflect_binder_in_complex_from_cif",
                        lambda *a, **k: np.zeros((3, 3)))
    # _make_base_backend is resolved through _self() -> xenodesign.classes.alpha (public module).
    # Patch on alpha_mod so shim._make_base_backend picks up the deterministic CPU stub.
    monkeypatch.setattr(alpha_mod, "_make_base_backend",
                        lambda backend="ligandmpnn": _fake_ifold_backend)
    # _extract reads binder backbone + context from the per-iter CIF via:
    #   backbone_by_residue_from_cif -> lazy-imported from xenodesign.eval.gate_tier0a
    #   _backbone_array_from_residues -> direct call in _alpha_internals (NOT via _self())
    #   _best_cif_path / _all_atoms_from_chain -> via _self() -> alpha_mod
    dummy_cif = tmp_path / "iter.cif"
    dummy_cif.write_text("")
    # 24-residue cyclic backbone (matches DEFAULT_BINDER_LENGTH["cyclic"]=24; coordinators at 6,12,18,24).
    monkeypatch.setattr("xenodesign.eval.gate_tier0a.backbone_by_residue_from_cif",
                        lambda cif, chain: [object()] * 24)
    monkeypatch.setattr(ai_mod, "_backbone_array_from_residues",
                        lambda res: np.zeros((len(res), 4, 3)))
    monkeypatch.setattr(alpha_mod, "_best_cif_path",
                        lambda *a, **k: dummy_cif)
    monkeypatch.setattr(alpha_mod, "_all_atoms_from_chain",
                        lambda cif, chain: (np.zeros((0, 3)), []))


def test_golden_cyclic_metal_greedy(tmp_path, monkeypatch):
    _cyclic_metal_fakes(monkeypatch, tmp_path)
    # coord_residues: the REAL in-memory 5-tuple format (pos, one_letter, three_letter, chirality, atom).
    # mixed_chirality="none" overrides the cyclic preset default ("A") to force the greedy HalluLoop.
    cfg = resolve_config(
        "cyclic", target_type="metal", out_dir=str(tmp_path),
        cli_overrides={"loop.iters": 2, "use_pepmlm": False, "use_pll": False,
                       "restraints_on": False, "loop.backend": "ligandmpnn",
                       "mixed_chirality": "none",
                       "restraint.params": {"coord_residues": _cyclic_metal_coord_residues()}})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("cyclic_metal_greedy", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden
