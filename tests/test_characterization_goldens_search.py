"""S2 characterization goldens for the SEARCH paths (beam + ABC), reusing the S0 fake stack.

Each test runs a search path with a fixed seed through a deterministic CPU fake stack (no GPU,
no network) and asserts the result equals a committed golden JSON. Regenerate with:
    XENO_REGOLD=1 PYTHONPATH=$PWD python -m pytest -k golden_search -q
S2.0 captures the CURRENT (buggy: all-Ala beam known_seq, all-Ala ABC-A identity, no ABC anchor)
baseline; S2.1-S2.3 route through SequenceUpdate behind XENO_SEQ_STAGE and REGOLD to the corrected
behaviour, with the diff documented in the commit and an explicit not-all-Ala assertion test.
"""
from __future__ import annotations

import json

import numpy as np

from xenodesign import dispatch
from xenodesign.config import resolve_config

# Reuse the S0 fakes + helpers — ONE source of truth for the fake stack and the goldens.
from tests.test_characterization_goldens import (
    _FakePred,
    _drop_runspecific,
    _load_or_regold,
)


# ── beam fakes ─────────────────────────────────────────────────────────────────
# The dispatch beam path imports the design_alpha_beam helpers and the beam machinery.
# We fake: the predictor (dispatch._make_predictor), target_entities, the CIF reflect, the
# MPNN base design_fn (a deterministic echo that surfaces known_seq so the routing is visible),
# and the CIF-reading leaves the beam extractor + referee touch.

_BEAM_TARGET = "GSHMKVLITGGAGFIGSHLVDRL"
_BEAM_SEED = "ACDEFGHIKLMNPQRSTVWYG"   # 21-mer, ends in Gly anchor


def _echo_mpnn(design_backbone, context_coords, context_elements,
               fixed_mask, temperature, num_seqs, known_seq=None):
    """Deterministic MPNN stand-in: echo known_seq (so the routed real-seq context is OBSERVABLE
    in the output), falling back to all-Ala when known_seq is None (the CURRENT buggy beam call,
    which never threads known_seq)."""
    n = np.asarray(design_backbone).shape[0]
    base = (known_seq or "A" * n)[:n].ljust(n, "A")
    return [base for _ in range(num_seqs)]


def _beam_fakes(monkeypatch, tmp_path):
    import scripts.design_alpha_beam as dab
    import xenodesign.beam as beam_mod

    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)

    class _PredWrapper:
        """A predict-wrapper stand-in: callable(state, ref_time_steps, out_dir) -> _FakePred,
        with last_out_dir set so the beam referee/extractor have a CIF dir."""
        last_out_dir = str(tmp_path)

        def __call__(self, *a, **k):
            return _FakePred()

        def truncated_refine(self, *a, **k):
            return _FakePred()

    # The dispatch beam path selects wrapper=_LoopBackendWrapper (non-callable) when
    # restraints_on=False, but beam.predict_children calls predict_fn(state, ...) which needs
    # a callable. We replace _LoopBackendWrapper in the wrappers module with our fake so the
    # dispatch's `from xenodesign.backends.wrappers import _LoopBackendWrapper` inside run_design
    # picks up the callable stand-in. This fakes the wrapper at the module-import seam.
    import xenodesign.backends.wrappers as _wrappers_mod
    _orig_lbw = _wrappers_mod._LoopBackendWrapper

    class _CallableLoopWrapper(_PredWrapper):
        """Drop-in for _LoopBackendWrapper that is also callable (needed by beam.predict_children).
        Stores last_out_dir as an INSTANCE attribute set on __call__ so the beam extractor sees it.
        """
        def __init__(self, *a, **k):
            self.last_out_dir = str(tmp_path)

    monkeypatch.setattr(_wrappers_mod, "_LoopBackendWrapper", _CallableLoopWrapper)

    # dispatch._make_predictor returns (backend, predict_fn); _run_beam uses loop._backend as the
    # beam predict_fn, which is the wrapper built around this. We fake the whole predictor stack.
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_PredWrapper(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr(dispatch, "target_entities",
                        lambda cfg: ([{"type": "protein", "name": "target",
                                       "sequence": _BEAM_TARGET, "chirality": "L"}], None, None))
    monkeypatch.setattr("xenodesign.seed.reflect_binder_in_complex_from_cif",
                        lambda *a, **k: np.zeros((3, 3)))

    # The beam design_fn wraps _ligandmpnn_design_fn (dispatch.py:188). Swap the base for the echo.
    monkeypatch.setattr("xenodesign.sequence_update._ligandmpnn_design_fn", _echo_mpnn,
                        raising=True)

    # The beam extractor reads the parent CIF; fake the leaves (design_alpha_beam._make_extract_fn).
    dummy_cif = tmp_path / "beam.cif"
    dummy_cif.write_text("")
    monkeypatch.setattr(dab, "_best_cif_path", lambda *a, **k: dummy_cif, raising=True)
    monkeypatch.setattr("xenodesign.eval.gate_tier0a.backbone_by_residue_from_cif",
                        lambda cif, chain: [object()] * len(_BEAM_SEED))
    monkeypatch.setattr(dab, "_backbone_array_from_residues",
                        lambda res: np.zeros((len(res), 4, 3)), raising=True)
    monkeypatch.setattr(dab, "_all_atoms_from_chain",
                        lambda cif, chain: (np.zeros((0, 3)), []), raising=True)
    # The beam referee reads chirality/seq/helix from the CIF; stub to constants so selection is
    # deterministic (we are pinning the seq-update, not the referee).
    monkeypatch.setattr(dab, "_chirality_violation_frac_from_cif", lambda cif: 0.0, raising=True)
    monkeypatch.setattr(dab, "_binder_helix_fraction", lambda cif: 0.9, raising=True)
    monkeypatch.setattr(dab, "binder_seq_from_cif", lambda cif, chain: _BEAM_SEED, raising=True)
    monkeypatch.setattr(dab, "composition_violation", lambda seq: False, raising=True)
    # Seed the binder so the dispatch L-seed predict + double-flip uses our fixed seed.
    import xenodesign.classes.alpha as alpha_mod
    import xenodesign.classes._alpha_internals as ai_mod
    monkeypatch.setattr(alpha_mod.Alpha, "seed",
                        lambda self, cfg, target_seq: __import__(
                            "xenodesign.classes.base", fromlist=["SeedSpec"]
                        ).SeedSpec(one_letter=_BEAM_SEED))
    # The anneal HalluLoop's sequence_update_fn resolves its backend via
    # _self()._make_base_backend (which reads shim._ligandmpnn_design_fn on alpha_mod).
    # Patch _make_base_backend to return our echo directly, so the anneal seq-update
    # is the same deterministic echo as the beam design_fn (no GPU / no torch).
    monkeypatch.setattr(alpha_mod, "_make_base_backend",
                        lambda backend="ligandmpnn": _echo_mpnn)
    # The anneal HalluLoop's sequence_update_fn calls _extract → _self()._best_cif_path
    # and _self()._all_atoms_from_chain / _backbone_array_from_residues via the alpha shim.
    # Patch these so the anneal seq-update runs deterministically on CPU.
    monkeypatch.setattr(alpha_mod, "_best_cif_path",
                        lambda *a, **k: dummy_cif)
    monkeypatch.setattr(alpha_mod, "_all_atoms_from_chain",
                        lambda cif, chain: (np.zeros((0, 3)), []))
    monkeypatch.setattr(ai_mod, "_backbone_array_from_residues",
                        lambda res: np.zeros((len(res), 4, 3)))
    # S2.1.3: the flag-ON stage path (_stage_extract in _alpha_internals.py) calls
    # _self().binder_seq_from_cif (= alpha_mod.binder_seq_from_cif), which is different from the
    # dab.binder_seq_from_cif already patched above (the beam referee's CIF reads come through dab;
    # the anneal loop's _stage_extract reads come through alpha_mod). Patch both so the dummy empty
    # CIF does not reach gemmi.read_structure under XENO_SEQ_STAGE=1.
    monkeypatch.setattr(alpha_mod, "binder_seq_from_cif",
                        lambda cif, chain: _BEAM_SEED)


def test_golden_search_beam_alpha(tmp_path, monkeypatch):
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")   # S2.1: the corrected (real-known_seq) beam is now the contract
    _beam_fakes(monkeypatch, tmp_path)
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False, "loop.backend": "ligandmpnn",
                                        "loop.search": "beam", "loop.beam_width": 2,
                                        "loop.beam_cycles": 2, "loop.num_seqs": 2})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("beam_alpha", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden


# ── ABC Variant-A baseline ─────────────────────────────────────────────────────
# Captures the CURRENT (buggy) ABC-A output: all-Ala design_codes + empty context_coords /
# context_elements in abc_variant_a_design_fn (variants.py:104-112). This is the S2.0b
# baseline; the POINT is to pin the CURRENT behaviour — do NOT fix anything here.

import xenodesign.dispatch as dispatch_mod  # noqa: F401 — explicit seam reference per brief


def _abc_fakes(monkeypatch):
    """Fake the ABC fitness (deterministic, structure-publishing) so abc_search runs on CPU with
    the REAL variant design_fns + engine. The fitness scores by sequence diversity so selection is
    deterministic and the design_fn's output is what drives the result."""
    def _fake_make_fitness(backend, **k):
        def fitness(sequence, chirality_pattern):
            fitness.last_structure = None   # no CIF in the CPU fake (Variant A falls back to zeros)
            # Reward identity diversity (favours a non-degenerate sequence) deterministically.
            return float(len(set(sequence)))
        fitness.last_structure = None
        return fitness

    monkeypatch.setattr("xenodesign.abc.fitness.make_abc_fitness", _fake_make_fitness)
    # _run_abc imports make_abc_fitness from xenodesign.abc.fitness at call time; patch both the
    # source and the dispatch-local import path to be safe.
    monkeypatch.setattr("xenodesign.dispatch.target_entities",
                        lambda cfg: ([], None, None))
    monkeypatch.setattr(dispatch, "_ensure_patches", lambda: None)
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (_FakePred(), lambda *a, **k: _FakePred()))
    # The Variant-A design_fn wraps _ligandmpnn_design_fn (dispatch.py:317). Echo known_seq so the
    # routed-vs-buggy difference is observable in the selected identity.
    monkeypatch.setattr("xenodesign.sequence_update._ligandmpnn_design_fn", _echo_mpnn,
                        raising=True)


def test_golden_search_abc_a(tmp_path, monkeypatch):
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")   # S2.2: the corrected (real-known_seq) ABC-A is now the contract
    _abc_fakes(monkeypatch)
    cfg = resolve_config("cyclic", target_type="none", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False, "mixed_chirality": "A",
                                        "abc.cycles": 2, "abc.colony_size": 3,
                                        "abc.scout_limit": 2, "abc.chai_eval_budget": 12})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("abc_a", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden


# ── ABC Variant-B baseline ─────────────────────────────────────────────────────
# Captures the CURRENT (no-MPNN, point-mutates identity directly) ABC-B output.
# Variant B does identity+chirality point-mutation (no MPNN, no known_seq) —
# its baseline mainly characterises the result-dict shape + the (currently
# anchor-less) emitted chain. Do NOT fix anything here.


def test_golden_search_abc_b(tmp_path, monkeypatch):
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")   # S2.3: flag-ON is now the contract for ABC-B
    _abc_fakes(monkeypatch)
    cfg = resolve_config("cyclic", target_type="none", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False, "use_pll": False,
                                        "restraints_on": False, "mixed_chirality": "B",
                                        "abc.cycles": 2, "abc.colony_size": 3,
                                        "abc.scout_limit": 2, "abc.chai_eval_budget": 12})
    report = dispatch.run_design(cfg)
    golden = _load_or_regold("abc_b", report)
    assert _drop_runspecific(json.loads(json.dumps(report, default=str))) == golden
