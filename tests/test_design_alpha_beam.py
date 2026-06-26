"""CPU tests for the frozen beam+anneal driver (scripts/design_alpha_beam.py, ADR-018).

NO GPU / NO network: only the CLI parsing, the smoke-budget math, and the read-only reuse of
design_alpha helpers are exercised. The heavy run (run_alpha_beam_design) is GPU-only and never
invoked here — these tests pin the driver's contract (superset CLI of design_alpha) so the lead
can commit a stable, reproducible artifact.
"""
import scripts.design_alpha_beam as dab
from scripts.design_alpha_beam import _parse_args


# ── module imports CPU-clean (no GPU/network at import) ─────────────────────────

def test_module_importable():
    assert dab is not None
    # It reuses design_alpha helpers by IMPORT (read-only), never re-defines them.
    from scripts.design_alpha import composition_violation, _ensure_cterm_glycine
    assert dab.composition_violation is composition_violation
    assert dab._ensure_cterm_glycine is _ensure_cterm_glycine


# ── new beam knobs exist with the spec'd defaults ───────────────────────────────

def test_beam_defaults():
    args = _parse_args([])
    assert args.beam_width == 3
    assert args.children_per_branch == 3
    assert args.cycles == 3
    assert args.num_seqs == 8
    assert args.anneal_steps == 5
    assert args.anneal_top_n == 3
    assert args.anneal_ref_start == 200
    assert args.anneal_temp_start == 0.3
    assert args.prune_metric == "composite"
    assert args.dedup is True
    assert args.ref_time_steps == 50


# ── inherited design_alpha knobs are still present (superset CLI) ───────────────

def test_inherits_design_alpha_knobs():
    args = _parse_args([])
    for knob in ("device", "seed", "out_dir", "no_pll", "no_pepmlm", "smoke"):
        assert hasattr(args, knob), f"missing inherited knob --{knob}"
    # --no_restraints / --restraint_file are the inherited restraint controls.
    assert hasattr(args, "no_restraints")


# ── --no_dedup flips the dedup flag off ─────────────────────────────────────────

def test_no_dedup_flag():
    assert _parse_args([]).dedup is True
    assert _parse_args(["--no_dedup"]).dedup is False


# ── children_per_branch must be <= num_seqs (top_k constraint) ──────────────────

def test_children_per_branch_capped_to_num_seqs():
    # The driver resolves an effective m = min(children_per_branch, num_seqs) so
    # MultiCandidate(top_k=m) never violates 1 <= top_k <= num_seqs.
    assert dab._effective_children(children_per_branch=10, num_seqs=8) == 8
    assert dab._effective_children(children_per_branch=3, num_seqs=8) == 3


# ── --smoke forces B=2,m=2,C=2,anneal_steps=2 => 13 predicts ───────────────────

def test_smoke_budget_is_13_predicts():
    # Spec: smoke beam = 1 + m + (C-1)*B*m = 1 + 2 + 1*2*2 = 7; anneal = top_n*steps.
    B, m, C, a, top_n = dab._smoke_knobs()
    assert (B, m, C, a) == (2, 2, 2, 2)
    beam_predicts = 1 + m + (C - 1) * B * m
    anneal_predicts = top_n * a
    assert beam_predicts == 7
    assert beam_predicts + anneal_predicts == 13


# ── default budget matches the beam.py formula (31 predicts) ──────────

def test_default_budget_matches_formula():
    args = _parse_args([])
    B, m, C = args.beam_width, args.children_per_branch, args.cycles
    a, top_n = args.anneal_steps, args.anneal_top_n
    # B=3,m=3,C=3 => beam = 1 + 3 + (3-1)*3*3 = 1+3+18 = 22 (matches beam.py test 6 formula);
    # + anneal top_n*steps = 3*5 = 15 => 37 predicts (the spec's ~37 ceiling).
    beam_predicts = 1 + m + (C - 1) * B * m
    assert beam_predicts == 22
    assert beam_predicts + top_n * a == 37


# ── --objective {iptm|mixed} + --periodicity_gate (beam-dep) ───────

def test_objective_default_iptm():
    """Reproducible DEFAULT objective is ipTM (byte-for-byte baseline)."""
    assert _parse_args([]).objective == "iptm"


def test_objective_accepts_mixed():
    assert _parse_args(["--objective", "mixed"]).objective == "mixed"


def test_periodicity_gate_default_off():
    args = _parse_args([])
    assert args.periodicity_gate is False
    assert args.heptad_thresh == 0.35


def test_periodicity_gate_on_with_thresh():
    args = _parse_args(["--periodicity_gate", "--heptad_thresh", "0.42"])
    assert args.periodicity_gate is True
    assert args.heptad_thresh == 0.42


def test_reuses_design_alpha_mixed_helpers():
    """The beam driver reuses design_alpha's mixed-objective helpers by IMPORT (not re-defined)."""
    from scripts.design_alpha import make_mixed_loop_score_fn, mixed_objective_from_cif
    assert dab.make_mixed_loop_score_fn is make_mixed_loop_score_fn
    assert dab.mixed_objective_from_cif is mixed_objective_from_cif


# ── _select_beam_final: the beam-dep final pick (periodicity gate + mixed re-rank) ──

def _make_anneal_step(one_letter, iptm):
    """A LoopStep whose state carries the D-CCD of the given binder seq and a mock prediction."""
    import numpy as np
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.loop import LoopState, LoopStep

    pred = type("P", (), {"iptm": iptm, "token_index": np.array([0, 0, 1, 1, 1]),
                          "plddt": np.array([60.0, 60.0, 70.0, 70.0, 70.0]),
                          "chirality_violation_frac": 0.0})()
    return LoopStep(state=LoopState(d_fasta=to_d_fasta(one_letter), coords=np.zeros((1, 3))),
                    prediction=pred, score=iptm)


def test_select_beam_final_periodicity_gate_drops_periodic():
    """With periodicity_gate on, a heptad-periodic anneal state is filtered out and an achievable
    one is selected even if the periodic one has higher ipTM."""
    from scripts.design_alpha_beam import _select_beam_final, _make_anneal_referee_fn
    from xenodesign.judges.panel import JudgePanel

    periodic = _make_anneal_step("LEKIQSN" * 4, iptm=0.99)   # register UNachievable
    diverse = _make_anneal_step("ACDEFGHIKLMNPQRSTVWYA", iptm=0.50)  # register achievable
    states = [periodic, diverse]

    chosen = _select_beam_final(
        states, _make_anneal_referee_fn(None), JudgePanel(),
        objective="iptm", periodicity_gate=True, heptad_thresh=0.35, default_step=periodic)
    assert chosen is diverse, "the periodic (register-UNachievable) state must be filtered out"


def test_select_beam_final_iptm_picks_highest_when_gate_off():
    """objective=iptm, no gate: pick the highest-ipTM state (baseline behaviour)."""
    from scripts.design_alpha_beam import _select_beam_final, _make_anneal_referee_fn
    from xenodesign.judges.panel import JudgePanel

    low = _make_anneal_step("ACDEFGHIKLMNPQRSTVWYA", iptm=0.4)
    high = _make_anneal_step("MKLNDEQRSTVWYAGHICFP", iptm=0.8)
    chosen = _select_beam_final(
        [low, high], _make_anneal_referee_fn(None), JudgePanel(),
        objective="iptm", periodicity_gate=False, heptad_thresh=0.35, default_step=low)
    assert chosen is high


def test_select_beam_final_gate_inert_if_all_periodic():
    """If the periodicity gate would remove EVERY state, it is treated as inert (never leave the
    run with no design to report) — it falls back to ranking all states."""
    from scripts.design_alpha_beam import _select_beam_final, _make_anneal_referee_fn
    from xenodesign.judges.panel import JudgePanel

    p1 = _make_anneal_step("LEKIQSN" * 4, iptm=0.6)
    p2 = _make_anneal_step("LEKIQSN" * 3, iptm=0.9)
    chosen = _select_beam_final(
        [p1, p2], _make_anneal_referee_fn(None), JudgePanel(),
        objective="iptm", periodicity_gate=True, heptad_thresh=0.35, default_step=p1)
    # both periodic -> gate inert -> iptm ranking picks the higher-ipTM p2.
    assert chosen is p2
