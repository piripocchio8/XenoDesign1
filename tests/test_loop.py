import numpy as np
import pytest
from xenodesign.loop import HalluLoop, LoopState, LoopStep, chirality_gated_accept
from xenodesign.judges.panel import JudgePanel, RefereeScore


class _FakeBackend:
    """Returns predictions with monotonically improving iptm so we can assert progress."""
    def __init__(self):
        self.calls = 0

    def truncated_refine(self, state, ref_time_steps, out_dir):
        self.calls += 1
        coords = np.zeros((3, 3))
        return type("P", (), {"coords": coords, "iptm": 0.5 + 0.1 * self.calls,
                              "interface_plddt": 80.0, "chirality_violation_frac": 0.0})()


def _fake_seq_update(prediction):
    return "AAA"  # constant designed sequence


def test_loop_runs_n_iterations_greedy():
    backend = _FakeBackend()
    loop = HalluLoop(backend=backend, sequence_update_fn=_fake_seq_update,
                     score_fn=lambda p: p.iptm)
    init = LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=np.zeros((3, 3)))
    history = loop.run(init, iterations=4, ref_time_steps=50, out_dir="/tmp/x")
    assert backend.calls == 4
    assert len(history) == 4
    # greedy: every step accepted, score increases monotonically here.
    scores = [h.score for h in history]
    assert scores == sorted(scores)


def test_loop_best_returns_highest_score():
    backend = _FakeBackend()
    loop = HalluLoop(backend=backend, sequence_update_fn=_fake_seq_update,
                     score_fn=lambda p: p.iptm)
    init = LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=np.zeros((3, 3)))
    history = loop.run(init, iterations=3, ref_time_steps=50, out_dir="/tmp/x")
    best = loop.best(history)
    assert best.score == max(h.score for h in history)


# ─────────────────────────────────────────────────────────────────────────────
# Chirality-gated acceptance gate — CPU/mock tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_pred(iptm: float, chirality_violation_frac: float) -> object:
    """Build a minimal mock prediction object."""
    return type("P", (), {
        "coords": np.zeros((3, 3)),
        "iptm": iptm,
        "interface_plddt": 80.0,
        "chirality_violation_frac": chirality_violation_frac,
    })()


def _make_panel_with_score_fn() -> JudgePanel:
    """Build a JudgePanel with a score_fn that reads from mock predictions."""
    def _score_fn(step: LoopStep) -> RefereeScore:
        pred = step.prediction
        return RefereeScore(
            chirality_violation=pred.chirality_violation_frac,
            iptm=pred.iptm,
            interface_plddt=pred.interface_plddt,
        )
    return JudgePanel(score_fn=_score_fn)


def _make_step(iptm: float, chirality_violation_frac: float) -> LoopStep:
    pred = _make_pred(iptm, chirality_violation_frac)
    coords = np.zeros((3, 3))
    return LoopStep(
        state=LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=coords),
        prediction=pred,
        score=iptm,
    )


class TestChiralityGatedAccept:
    """CPU tests for the chirality_gated_accept factory and HalluLoop.run accept_fn."""

    def test_accept_fn_requires_panel_with_score_fn(self):
        """chirality_gated_accept must raise if panel has no score_fn."""
        panel_no_fn = JudgePanel()  # no score_fn
        with pytest.raises(ValueError, match="score_fn"):
            chirality_gated_accept(panel_no_fn)

    def test_gate_accepts_clean_candidate(self):
        """Candidate with chirality=0.0 should be accepted."""
        panel = _make_panel_with_score_fn()
        gate = chirality_gated_accept(panel, max_violation=0.1)

        candidate = _make_step(iptm=0.7, chirality_violation_frac=0.0)
        current   = _make_step(iptm=0.5, chirality_violation_frac=0.0)
        assert gate(candidate, current) is True

    def test_gate_rejects_chirality_violated_candidate(self):
        """Candidate with chirality > max_violation should be rejected."""
        panel = _make_panel_with_score_fn()
        gate = chirality_gated_accept(panel, max_violation=0.1)

        candidate = _make_step(iptm=0.9, chirality_violation_frac=0.5)
        current   = _make_step(iptm=0.5, chirality_violation_frac=0.0)
        assert gate(candidate, current) is False

    def test_gate_boundary_exactly_at_threshold_accepted(self):
        """Candidate with chirality == max_violation (not strictly greater) is accepted."""
        panel = _make_panel_with_score_fn()
        gate = chirality_gated_accept(panel, max_violation=0.1)

        candidate = _make_step(iptm=0.7, chirality_violation_frac=0.1)
        current   = _make_step(iptm=0.5, chirality_violation_frac=0.0)
        # violation=0.1 is NOT > 0.1 → accepted.
        assert gate(candidate, current) is True

    def test_gate_boundary_just_above_threshold_rejected(self):
        """Candidate with chirality = threshold + epsilon is rejected."""
        panel = _make_panel_with_score_fn()
        gate = chirality_gated_accept(panel, max_violation=0.1)

        candidate = _make_step(iptm=0.7, chirality_violation_frac=0.101)
        current   = _make_step(iptm=0.5, chirality_violation_frac=0.0)
        assert gate(candidate, current) is False

    def test_loop_default_accept_fn_is_greedy(self):
        """Without accept_fn, loop behaves identically to original (accept-always)."""
        class _DriftingBackend:
            """Alternates clean and chirality-violated predictions."""
            def __init__(self):
                self.calls = 0
            def truncated_refine(self, state, ref_time_steps, out_dir):
                self.calls += 1
                viol = 0.5 if self.calls % 2 == 0 else 0.0
                return _make_pred(iptm=0.5 + 0.05 * self.calls,
                                  chirality_violation_frac=viol)

        backend = _DriftingBackend()
        loop = HalluLoop(backend=backend, sequence_update_fn=_fake_seq_update,
                         score_fn=lambda p: p.iptm)
        init = LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=np.zeros((3, 3)))
        # No accept_fn → all 4 steps accepted including chirality-violated ones.
        history = loop.run(init, iterations=4, ref_time_steps=50, out_dir="/tmp/x_greedy")
        assert len(history) == 4
        assert backend.calls == 4

    def test_loop_chirality_gate_rejects_violated_steps(self):
        """With chirality_gated_accept, violated iterations don't advance state."""
        # Predictions: iter 0 clean, iter 1 violated (chir=0.5), iter 2 clean, iter 3 violated.
        chirality_seq = [0.0, 0.5, 0.0, 0.5]
        iptm_seq = [0.6, 0.7, 0.65, 0.75]

        call_idx = [0]

        class _PatterndBackend:
            def truncated_refine(self, state, ref_time_steps, out_dir):
                i = call_idx[0]
                call_idx[0] += 1
                return _make_pred(iptm=iptm_seq[i],
                                  chirality_violation_frac=chirality_seq[i])

        backend = _PatterndBackend()
        panel = _make_panel_with_score_fn()
        gate = chirality_gated_accept(panel, max_violation=0.1)
        loop = HalluLoop(backend=backend, sequence_update_fn=_fake_seq_update,
                         score_fn=lambda p: p.iptm)
        init = LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=np.zeros((3, 3)))
        history = loop.run(init, iterations=4, ref_time_steps=50,
                           out_dir="/tmp/x_gated", accept_fn=gate)

        # All 4 iterations run (backend called 4×), history has 4 entries.
        assert len(history) == 4
        assert call_idx[0] == 4

        # iter 0 (clean): accepted → state advances to new coords from pred.
        # iter 1 (violated): rejected → state retained (coords still zeros from iter 0 pred).
        # iter 2 (clean): accepted.
        # iter 3 (violated): rejected.

        # The score is always the candidate's score (for analysis logging).
        assert history[0].score == pytest.approx(0.6)
        assert history[1].score == pytest.approx(0.7)   # candidate recorded even when rejected
        assert history[2].score == pytest.approx(0.65)
        assert history[3].score == pytest.approx(0.75)  # candidate recorded even when rejected

    def test_loop_chirality_gate_retains_state_on_reject(self):
        """When a step is rejected, state.d_fasta must match the PREVIOUS accepted step."""
        chirality_seq = [0.0, 0.5]  # iter 0 clean, iter 1 violated
        call_idx = [0]
        seqs = ["AAA", "CCC"]  # seq_update returns different seqs each call

        class _Backend2:
            def truncated_refine(self, state, ref_time_steps, out_dir):
                i = call_idx[0]
                call_idx[0] += 1
                return _make_pred(iptm=0.5 + i * 0.1, chirality_violation_frac=chirality_seq[i])

        seq_call_idx = [0]
        def _seq_update(pred):
            s = seqs[seq_call_idx[0] % len(seqs)]
            seq_call_idx[0] += 1
            return s

        backend = _Backend2()
        panel = _make_panel_with_score_fn()
        gate = chirality_gated_accept(panel, max_violation=0.1)
        loop = HalluLoop(backend=backend, sequence_update_fn=_seq_update,
                         score_fn=lambda p: p.iptm)
        init = LoopState(d_fasta="(DINIT)", coords=np.zeros((3, 3)))
        history = loop.run(init, iterations=2, ref_time_steps=50,
                           out_dir="/tmp/x_retain", accept_fn=gate)

        # iter 0: clean, accepted → state.d_fasta updated from seq_update("AAA")
        # iter 1: violated, rejected → state.d_fasta must match iter 0's d_fasta.
        d_fasta_iter0 = history[0].state.d_fasta
        d_fasta_iter1 = history[1].state.d_fasta
        assert d_fasta_iter0 == d_fasta_iter1, (
            f"Rejected step should retain previous state; "
            f"iter0={d_fasta_iter0!r}, iter1={d_fasta_iter1!r}"
        )


# AnnealSchedule wiring into HalluLoop.run — CPU tests (lane B, default-OFF)
from xenodesign.schedule import AnnealSchedule


class _RecordingBackend:
    """Records the ref_time_steps passed to each truncated_refine call."""
    def __init__(self):
        self.ref_time_steps_seen = []

    def truncated_refine(self, state, ref_time_steps, out_dir):
        self.ref_time_steps_seen.append(ref_time_steps)
        return type("P", (), {"coords": np.zeros((3, 3)), "iptm": 0.6,
                              "interface_plddt": 80.0,
                              "chirality_violation_frac": 0.0})()


def test_loop_default_schedule_none_uses_constant_ref_time_steps():
    backend = _RecordingBackend()
    loop = HalluLoop(backend=backend, sequence_update_fn=_fake_seq_update,
                     score_fn=lambda p: p.iptm)
    init = LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=np.zeros((3, 3)))
    loop.run(init, iterations=4, ref_time_steps=50, out_dir="/tmp/x_sched_none")
    assert backend.ref_time_steps_seen == [50, 50, 50, 50]


def test_loop_schedule_anneals_ref_time_steps_per_iter():
    backend = _RecordingBackend()
    loop = HalluLoop(backend=backend, sequence_update_fn=_fake_seq_update,
                     score_fn=lambda p: p.iptm)
    init = LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=np.zeros((3, 3)))
    sched = AnnealSchedule(base_ref_time_steps=50, anneal_start=150, anneal_frac=1.0 / 3.0)
    loop.run(init, iterations=30, ref_time_steps=50, out_dir="/tmp/x_sched_anneal",
             schedule=sched, iterations_for_schedule=30)
    seen = backend.ref_time_steps_seen
    assert len(seen) == 30
    assert seen[0] == 150
    assert seen[-1] == 50
    assert all(seen[i] >= seen[i + 1] for i in range(len(seen) - 1))


def test_loop_constant_schedule_matches_scalar_path():
    b_scalar, b_sched = _RecordingBackend(), _RecordingBackend()
    init = LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=np.zeros((3, 3)))
    HalluLoop(b_scalar, _fake_seq_update, lambda p: p.iptm).run(
        init, iterations=5, ref_time_steps=50, out_dir="/tmp/x_scalar")
    HalluLoop(b_sched, _fake_seq_update, lambda p: p.iptm).run(
        init, iterations=5, ref_time_steps=50, out_dir="/tmp/x_const_sched",
        schedule=AnnealSchedule(base_ref_time_steps=50))
    assert b_scalar.ref_time_steps_seen == b_sched.ref_time_steps_seen


# greedy_iptm_accept — strict score/ipTM hill-climb accept_fn (lane B, semi_greedy)
from xenodesign.loop import greedy_iptm_accept


class TestGreedyIptmAccept:
    """The accept-half of semi_greedy_steps: reject any candidate that does not strictly
    improve on the current step's score (stays 'greedy' — no Metropolis)."""

    def test_accepts_improvement(self):
        gate = greedy_iptm_accept(min_delta=0.0)
        cand = _make_step(iptm=0.7, chirality_violation_frac=0.0)
        curr = _make_step(iptm=0.5, chirality_violation_frac=0.0)
        assert gate(cand, curr) is True

    def test_rejects_no_improvement(self):
        gate = greedy_iptm_accept(min_delta=0.0)
        cand = _make_step(iptm=0.5, chirality_violation_frac=0.0)
        curr = _make_step(iptm=0.5, chirality_violation_frac=0.0)
        assert gate(cand, curr) is False

    def test_rejects_regression(self):
        gate = greedy_iptm_accept(min_delta=0.0)
        cand = _make_step(iptm=0.4, chirality_violation_frac=0.0)
        curr = _make_step(iptm=0.6, chirality_violation_frac=0.0)
        assert gate(cand, curr) is False

    def test_min_delta_threshold(self):
        gate = greedy_iptm_accept(min_delta=0.05)
        curr = _make_step(iptm=0.50, chirality_violation_frac=0.0)
        assert gate(_make_step(iptm=0.54, chirality_violation_frac=0.0), curr) is False
        assert gate(_make_step(iptm=0.56, chirality_violation_frac=0.0), curr) is True

    def test_first_iter_sentinel_accepted(self):
        gate = greedy_iptm_accept(min_delta=0.0)
        cand = _make_step(iptm=0.3, chirality_violation_frac=0.0)
        sentinel = LoopStep(state=LoopState(d_fasta="(DAL)", coords=np.zeros((3, 3))),
                            prediction=None, score=-1.0)
        assert gate(cand, sentinel) is True

    def test_loop_greedy_iptm_hill_climb(self):
        iptm_seq = [0.6, 0.5, 0.7, 0.65]
        call_idx = [0]

        class _SeqBackend:
            def truncated_refine(self, state, ref_time_steps, out_dir):
                i = call_idx[0]; call_idx[0] += 1
                return _make_pred(iptm=iptm_seq[i], chirality_violation_frac=0.0)

        loop = HalluLoop(backend=_SeqBackend(), sequence_update_fn=_fake_seq_update,
                         score_fn=lambda p: p.iptm)
        init = LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=np.zeros((3, 3)))
        history = loop.run(init, iterations=4, ref_time_steps=50,
                           out_dir="/tmp/x_greedy_iptm", accept_fn=greedy_iptm_accept())
        assert len(history) == 4
        assert call_idx[0] == 4
        assert [round(h.score, 2) for h in history] == [0.6, 0.5, 0.7, 0.65]


# ─────────────────────────────────────────────────────────────────────────────
# Periodicity-gated acceptance gate — CPU/sequence-only tests
# ─────────────────────────────────────────────────────────────────────────────
from xenodesign.loop import periodicity_gated_accept, compose_accept_fns
from xenodesign.io_spec import to_d_fasta


def _seq_step(one_letter: str) -> LoopStep:
    """A LoopStep whose state carries the D-CCD encoding of the given one-letter binder seq.
    The periodicity gate decodes state.d_fasta -> one-letter L for the hydropathy autocorrelation."""
    return LoopStep(
        state=LoopState(d_fasta=to_d_fasta(one_letter), coords=np.zeros((1, 3))),
        prediction=None, score=0.0,
    )


# A textbook coiled-coil heptad (period 7) -> register UNACHIEVABLE -> gate REJECTS.
_PERIODIC_SEQ = "LEKIQSN" * 4           # 28 aa, cleanly 7-periodic hydropathy
# An irregular amphipath -> register achievable -> gate ACCEPTS.
_DIVERSE_SEQ = "ACDEFGHIKLMNPQRSTVWYA"  # 21 aa, aperiodic


class TestPeriodicityGatedAccept:
    """CPU/sequence-only tests for periodicity_gated_accept (mirrors the chirality gate, but the
    gate is reference-free and sequence-only — no panel, no CIF, no GPU)."""

    def test_rejects_heptad_periodic_candidate(self):
        gate = periodicity_gated_accept(heptad_thresh=0.35)
        cand = _seq_step(_PERIODIC_SEQ)
        curr = _seq_step(_DIVERSE_SEQ)
        assert gate(cand, curr) is False

    def test_accepts_register_achievable_candidate(self):
        gate = periodicity_gated_accept(heptad_thresh=0.35)
        cand = _seq_step(_DIVERSE_SEQ)
        curr = _seq_step(_DIVERSE_SEQ)
        assert gate(cand, curr) is True

    def test_empty_sequence_is_accepted(self):
        """An empty/unreadable candidate seq cannot be PROVEN periodic -> accept (never silently
        kill a trajectory on a parse failure; the gate only ever rejects a proven-periodic seq)."""
        gate = periodicity_gated_accept(heptad_thresh=0.35)
        empty = LoopStep(state=LoopState(d_fasta="", coords=np.zeros((1, 3))),
                         prediction=None, score=0.0)
        assert gate(empty, _seq_step(_DIVERSE_SEQ)) is True

    def test_threshold_controls_rejection(self):
        """A very high heptad_thresh (>1.0, unreachable) makes every seq register-achievable, so the
        periodic seq is no longer rejected — confirming the threshold is the live control."""
        loose = periodicity_gated_accept(heptad_thresh=1.5)
        assert loose(_seq_step(_PERIODIC_SEQ), _seq_step(_DIVERSE_SEQ)) is True

    def test_loop_periodicity_gate_rejects_periodic_steps(self):
        """End-to-end through HalluLoop.run: a heptad-periodic designed seq is rejected, so the
        loop retains the previous (achievable) state's d_fasta."""
        seqs = [_DIVERSE_SEQ, _PERIODIC_SEQ]  # iter0 achievable (accept), iter1 periodic (reject)
        seq_idx = [0]

        def _seq_update(pred):
            s = seqs[seq_idx[0] % len(seqs)]
            seq_idx[0] += 1
            return s

        class _Backend:
            def truncated_refine(self, state, ref_time_steps, out_dir):
                return _make_pred(iptm=0.6, chirality_violation_frac=0.0)

        loop = HalluLoop(backend=_Backend(), sequence_update_fn=_seq_update,
                         score_fn=lambda p: p.iptm)
        init = LoopState(d_fasta=to_d_fasta("MKLVG"), coords=np.zeros((1, 3)))
        history = loop.run(init, iterations=2, ref_time_steps=50,
                           out_dir="/tmp/x_periodicity",
                           accept_fn=periodicity_gated_accept(heptad_thresh=0.35))
        assert len(history) == 2
        # iter0 (diverse): accepted -> state advances to the diverse seq.
        # iter1 (periodic): rejected -> state retained == iter0's accepted d_fasta.
        assert history[0].state.d_fasta == to_d_fasta(_DIVERSE_SEQ)
        assert history[1].state.d_fasta == history[0].state.d_fasta


class TestComposeAcceptFns:
    """compose_accept_fns AND-composes gates None-safely (used to stack chirality + periodicity)."""

    def test_no_gates_returns_none(self):
        assert compose_accept_fns(None, None) is None

    def test_single_gate_passes_through(self):
        g = lambda c, p: True
        assert compose_accept_fns(None, g) is g

    def test_and_composition_rejects_if_any_rejects(self):
        accept = lambda c, p: True
        reject = lambda c, p: False
        composed = compose_accept_fns(accept, reject)
        step = _seq_step(_DIVERSE_SEQ)
        assert composed(step, step) is False

    def test_and_composition_accepts_if_all_accept(self):
        accept = lambda c, p: True
        composed = compose_accept_fns(accept, accept, accept)
        step = _seq_step(_DIVERSE_SEQ)
        assert composed(step, step) is True

    def test_chirality_and_periodicity_stack(self):
        """The '-dep' stack: chirality gate AND periodicity gate. A periodic but D-clean candidate
        is rejected by the periodicity arm even though it passes chirality."""
        chir = chirality_gated_accept(_make_panel_with_score_fn(), max_violation=0.1)
        period = periodicity_gated_accept(heptad_thresh=0.35)
        composed = compose_accept_fns(chir, period)
        # D-clean (chirality passes) but heptad-periodic (periodicity rejects) -> overall reject.
        cand = LoopStep(
            state=LoopState(d_fasta=to_d_fasta(_PERIODIC_SEQ), coords=np.zeros((1, 3))),
            prediction=_make_pred(iptm=0.7, chirality_violation_frac=0.0), score=0.7,
        )
        curr = LoopStep(
            state=LoopState(d_fasta=to_d_fasta(_DIVERSE_SEQ), coords=np.zeros((1, 3))),
            prediction=_make_pred(iptm=0.5, chirality_violation_frac=0.0), score=0.5,
        )
        assert composed(cand, curr) is False
