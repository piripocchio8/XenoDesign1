"""CPU tests for the non-α (ICK D-knottin) BinderClass — real-loop hooks (T6).

Pure-CPU: exercise the NonAlpha hooks (case_id, anti-α SS-bias, ICK Cys seed,
default-off disulfide closure, restraint gating, objective/seq_update/referee
reuse of the alpha machinery, and the result assembly). The predictor is mocked;
no GPU. ``scripts/design_nonalpha.py`` remains the predict-only oracle
(``tests/test_design_nonalpha.py``); this file covers the promotion to a real
loop class.
"""
from __future__ import annotations

import json

import numpy as np

from xenodesign.benchmark.cases import get_case
from xenodesign.classes.base import CLASS_REGISTRY, SeedSpec
from xenodesign.classes.non_alpha import NonAlpha
from xenodesign.config import resolve_config


# ── identity + registry ──────────────────────────────────────────────────────────

def test_non_alpha_case_id_is_registry_key():
    assert NonAlpha().case_id == "nonalpha"


def test_registry_non_alpha_is_real_class():
    cls = CLASS_REGISTRY["non_alpha"]
    assert isinstance(cls, NonAlpha)
    assert cls.case_id == "nonalpha"


# ── SS-bias is anti-α (the defining non-α decision) ──────────────────────────────

def test_non_alpha_ss_bias_is_anti_alpha():
    n = NonAlpha()
    cfg = resolve_config("non_alpha", target_type="protein")
    ss = n.ss_bias(cfg, get_case("nonalpha"))
    assert ss.target_helix_frac == 0.0


# ── FROM-SCRATCH seed (NO forced knottin; Cys/disulfide is OPT-IN) ───────────────

# Offline (use_pepmlm=False) so CPU tests never load the real PepMLM weights.
def _na_cfg(**ov):
    ov.setdefault("use_pepmlm", False)
    return resolve_config("non_alpha", target_type="protein", cli_overrides=ov)


def test_non_alpha_seed_from_scratch_no_forced_cys():
    """The seed NO LONGER forces a knottin: default has NO mandatory Cys scaffold, and its
    length is the from-scratch DEFAULT (30), NOT the reference DP93 length (31)."""
    from xenodesign.config import resolve_binder_length
    n = NonAlpha()
    cfg = _na_cfg()
    spec = n.seed(cfg, target_seq="A" * 50)
    assert isinstance(spec, SeedSpec)
    assert len(spec.one_letter) == resolve_binder_length(cfg) == 30
    assert spec.cys_positions == ()                         # no forced ICK Cys
    assert len(spec.one_letter) != get_case("nonalpha").binder_length  # not the reference length


def test_non_alpha_seed_cys_is_opt_in():
    """Cys scaffold is OPT-IN via cfg.restraint.params['cys_positions']."""
    n = NonAlpha()
    cfg = _na_cfg(**{"restraint.params": {"cys_positions": (4, 10, 18)}})
    spec = n.seed(cfg, target_seq="A" * 50)
    assert spec.cys_positions == (4, 10, 18)
    assert all(spec.one_letter[p - 1] == "C" for p in spec.cys_positions)


def test_non_alpha_seed_deterministic_for_seed():
    n = NonAlpha()
    cfg = _na_cfg()
    a = n.seed(cfg, target_seq="A" * 50)
    b = n.seed(cfg, target_seq="A" * 50)
    assert a.one_letter == b.one_letter
    assert a.cys_positions == b.cys_positions


# ── closure (D-Cys disulfides off by default) ────────────────────────────────────

def test_non_alpha_closure_disulfides_off_by_default():
    n = NonAlpha()
    cfg = _na_cfg()
    spec = n.seed(cfg, target_seq="A" * 50)
    assert n.closure(cfg, spec) == []  # D-Cys covalent rejected by chai → default off


# ── restraint gating (nonalpha pocket is a shell → None, never crash) ────────────

def test_non_alpha_restraints_none_for_shell_pocket(tmp_path):
    n = NonAlpha()
    cfg = resolve_config("non_alpha", target_type="protein", out_dir=str(tmp_path))
    # nonalpha case pocket has empty target_resnums (pending gate #29) → unbuildable → None.
    assert n.restraints(cfg, get_case("nonalpha"), tmp_path, None) is None


def test_non_alpha_restraints_none_when_off(tmp_path):
    n = NonAlpha()
    cfg = resolve_config("non_alpha", target_type="protein", out_dir=str(tmp_path),
                         cli_overrides={"restraints_on": False})
    assert n.restraints(cfg, get_case("nonalpha"), tmp_path, None) is None


# ── objective is a real loop score_fn (reuses alpha machinery) ───────────────────

def test_non_alpha_objective_default_is_iptm_score():
    n = NonAlpha()
    cfg = resolve_config("non_alpha", target_type="protein")
    fn = n.objective(cfg, wrapper=None)

    class P:
        token_index = np.array([1, 1])
        plddt = np.array([70.0, 70.0])
        iptm = 0.6

    assert isinstance(fn(P()), float)


def test_non_alpha_seq_update_is_callable():
    """seq_update reuses the alpha MultiCandidate builder (real loop, not a one-shot)."""
    n = NonAlpha()
    cfg = _na_cfg()

    class _Wrapper:
        last_out_dir = None

    fn = n.seq_update(cfg, _Wrapper(), n.seed(cfg, target_seq="A" * 50))
    assert callable(fn)


def test_non_alpha_referee_is_callable():
    n = NonAlpha()
    cfg = resolve_config("non_alpha", target_type="protein")
    fn = n.referee(cfg, loop_dir="/tmp/none", esm_judge=None)
    assert callable(fn)


def test_non_alpha_accept_fns_default_is_accept_all():
    # No gates configured → compose_accept_fns returns None (the loop's accept-always
    # default), matching alpha's behaviour when no periodicity/chirality gate is set.
    n = NonAlpha()
    cfg = resolve_config("non_alpha", target_type="protein")
    af = n.accept_fns(cfg)
    assert af is None or callable(af)


# ── report assembles a result dict + writes JSON ─────────────────────────────────

def test_non_alpha_report_writes_json(tmp_path):
    n = NonAlpha()
    cfg = resolve_config("non_alpha", target_type="protein", out_dir=str(tmp_path))
    result = n.report(cfg, history=[], panel_result=None,
                      case=get_case("nonalpha"), out_dir=tmp_path)
    assert result["case_id"] == "nonalpha"
    assert (tmp_path / "nonalpha_result.json").exists()
    data = json.loads((tmp_path / "nonalpha_result.json").read_text())
    assert data["case_id"] == "nonalpha"
    assert data["ss_bias_target_helix_frac"] == 0.0


def test_non_alpha_runs_real_loop_through_dispatch(tmp_path, monkeypatch):
    """The promotion: non_alpha runs a REAL multi-iteration HalluLoop (not a one-shot predict)
    over the 2-chain MSA'd HA target via the dispatcher, with a CPU-mocked predictor.

    Asserts the loop actually iterated (n_iters == cfg.loop.iters) and produced a nonalpha
    result dict — the predict-only stub could only ever return a single prediction."""
    from xenodesign import dispatch

    class _FakePred:
        coords = np.zeros((3, 3))
        iptm = 0.42
        ptm = 0.5
        token_index = np.array([0, 1, 1])
        plddt = np.array([80.0, 80.0, 80.0])

    cfg = resolve_config(
        "non_alpha", target_type="protein", out_dir=str(tmp_path),
        cli_overrides={"loop.iters": 2, "use_pepmlm": False, "use_pll": False,
                       "restraints_on": False})

    # Patch the three GPU/IO seams: predictor, target builder (no MSA files), and the class's
    # seq_update (so no real CIF parse is needed for the mocked loop step).
    monkeypatch.setattr(dispatch, "_make_predictor",
                        lambda cfg: (object(), lambda *a, **k: _FakePred()))
    monkeypatch.setattr(
        dispatch, "target_entities",
        lambda cfg: ([{"type": "protein", "name": "HA1", "sequence": "AAAA",
                       "chirality": "L"},
                      {"type": "protein", "name": "HA2", "sequence": "CCCC",
                       "chirality": "L"}], "some/msas", None))
    monkeypatch.setattr(NonAlpha, "seq_update",
                        lambda self, cfg, wrapper, seed_spec: (lambda pred: "G" * 31))

    result = dispatch.run_design(cfg)
    assert result["case_id"] == "nonalpha"
    assert result["n_iters"] == 2  # REAL loop iterated twice (vs the old one-shot predict)
    assert (tmp_path / "resolved_config.json").exists()
    assert (tmp_path / "nonalpha_result.json").exists()
    # JSON parity: the dispatcher's measured l_seed_iptm / wall_time_s reach the ON-DISK file,
    # not just the returned dict (the l-seed predict stub returns iptm 0.42).
    import json
    on_disk = json.loads((tmp_path / "nonalpha_result.json").read_text())
    assert on_disk["l_seed_iptm"] == 0.42
    assert on_disk["wall_time_s"] >= 0.0


def test_non_alpha_report_selects_best_iptm_step(tmp_path):
    """report selects the highest-score trajectory step and reports its ipTM (real loop)."""
    n = NonAlpha()
    cfg = resolve_config("non_alpha", target_type="protein", out_dir=str(tmp_path))

    class _Pred:
        def __init__(self, iptm):
            self.iptm = iptm
            self.ptm = 0.5

    class _State:
        d_fasta = "(DAL)"

    class _Step:
        def __init__(self, iptm, score):
            self.prediction = _Pred(iptm)
            self.state = _State()
            self.score = score

    history = [_Step(0.30, 0.30), _Step(0.55, 0.55), _Step(0.40, 0.40)]
    result = n.report(cfg, history=history, panel_result=None,
                      case=get_case("nonalpha"), out_dir=tmp_path)
    assert result["selected_iptm"] == 0.55
    assert result["n_iters"] == 3
