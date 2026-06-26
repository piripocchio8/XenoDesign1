"""CPU tests for the beam+anneal search (xenodesign/beam.py, WF-1 spec, ADR-018).

NO GPU / NO torch / NO Chai. Everything is driven through injected callables:

  * a FAKE predict-wrapper backend that mirrors the real
    ``_PredictBackendWrapper.__call__(state, ref_time_steps, out_dir)`` shape — it returns a
    Prediction-like object exposing ``.coords / .iptm / .token_index / .plddt /
    .chirality_violation_frac`` (NOT test_loop's truncated_refine fake, which has none of
    token_index/plddt/chirality_violation_frac).
  * a FAKE design_fn that mirrors MultiCandidate(top_k=m): a 6-arg InverseFoldingBackend-shaped
    call returning the m best L sequences, best first.
  * a FAKE extract_fn that builds the inverse-folding inputs dict from a scored state.

The fakes are deterministic so the cost-accounting and selection assertions are exact.
D-peptide sequences are reported lowercase (Gly=G) in any output the driver emits; the tests
work in the L (mirror) frame, where backends emit one-letter L letters, so they stay uppercase.
"""
import numpy as np
import pytest

from xenodesign.beam import (
    BeamState,
    CostAccount,
    expand_state,
    predict_children,
    dedup,
    prune,
    beam_search,
    anneal_best,
)
from xenodesign.judges.panel import JudgePanel, RefereeScore
from xenodesign.loop import HalluLoop, LoopState, greedy_iptm_accept
from xenodesign.schedule import AnnealSchedule


# ── Fakes (deterministic, CPU-only) ────────────────────────────────────────────

_ALPHA_WEIGHTS = {"chirality": 0.40, "binding": 0.40, "pll": 0.05, "mirror": 0.05,
                  "ss_bias": 0.10}


def _fake_prediction(l_seq, *, iptm, chirality):
    """A Prediction-like object mirroring the real predict-wrapper output shape."""
    n = len(l_seq)
    coords = np.zeros((n, 3))
    token_index = np.ones(n, dtype=int)        # all design-chain tokens (mask == 1)
    plddt = np.full(n, 80.0)
    return type("P", (), {
        "coords": coords,
        "iptm": float(iptm),
        "token_index": token_index,
        "plddt": plddt,
        "chirality_violation_frac": float(chirality),
        "l_seq": l_seq,
    })()


class _FakePredictBackend:
    """Mirrors _PredictBackendWrapper: __call__(state, ref_time_steps, out_dir) -> Prediction.

    ipTM / chirality are looked up per-l_seq from an injected table (default: deterministic
    function of the sequence) so every test can pin exactly what each child scores. Counts its
    own calls so dedup/cost assertions can read the real Chai-predict count.
    """
    def __init__(self, iptm_for=None, chir_for=None):
        self.calls = 0
        self.seen_seqs = []
        # Mirror the real _PredictBackendWrapper, which records last_out_dir on every predict so
        # predict_children can derive cif_path from it (the seed bootstrap relies on this).
        self.last_out_dir = None
        self._iptm_for = iptm_for or (lambda s: 0.5 + 0.01 * len(set(s)))
        self._chir_for = chir_for or (lambda s: 0.0)

    def __call__(self, state, ref_time_steps, out_dir):
        self.calls += 1
        self.last_out_dir = out_dir
        l_seq = getattr(state, "l_seq", None) or _l_of(state.d_fasta)
        self.seen_seqs.append(l_seq)
        return _fake_prediction(l_seq, iptm=self._iptm_for(l_seq),
                                chirality=self._chir_for(l_seq))


def _l_of(d_fasta: str) -> str:
    """Cheap L-seq read for fakes (the real path reads the scored CIF chain)."""
    from xenodesign.io_spec import d_fasta_to_one_letter
    return d_fasta_to_one_letter(d_fasta)


def _fake_extract(state):
    """Build the inverse-folding inputs dict from a (scored) state. CPU-only stand-in for the
    design_alpha _extract closure (CIF -> backbone + context).

    Mirrors the real _extract guard (scripts/design_alpha_beam.py): a parent reaches extract only
    AFTER being predicted, so cif_path must be set. RAISE on a None cif_path exactly like the real
    closure — this is what catches the seed-bootstrap bug (seed predicted but cif_path left None).
    """
    if getattr(state, "cif_path", None) is None:
        raise RuntimeError("extract_fn called on an unpredicted parent (cif_path is None)")
    n = len(getattr(state, "l_seq", None) or _l_of(state.d_fasta))
    return {
        "design_backbone": np.zeros((n, 4, 3)),
        "design_codes": ["DAL"] * n,
        "context_coords": np.zeros((0, 3)),
        "context_elements": [],
    }


def _seq_pool_design_fn(children_seqs):
    """Make a 6-arg design_fn (MultiCandidate-shaped) that returns a fixed list of m children
    L-seqs, best first, each the length of the design backbone."""
    def _design(design_backbone, context_coords, context_elements,
                fixed_mask, temperature, num_seqs):
        n = design_backbone.shape[0]
        out = [s[:n].ljust(n, "A") for s in children_seqs]
        return out[:num_seqs] if num_seqs else out
    return _design


def _seed_state(l_seq="ACDEFGHIK"):
    from xenodesign.io_spec import to_d_fasta
    n = len(l_seq)
    return BeamState(
        d_fasta=to_d_fasta(l_seq), coords=np.zeros((n, 3)), l_seq=l_seq,
        cif_path=None, iptm=0.5, chirality=0.0, composite=0.0,
        parent_id=None, id=0, cycle=0,
    )


# ── 1. Stage-1 top_k (mirror; stage-1 may already cover) ────────────────────────

def test_multicandidate_topk_returns_m():
    from xenodesign.inverse_folding import MultiCandidate

    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["AAAA", "CAAA", "CCAA", "CCCA"][:num_seqs]

    mc = MultiCandidate(backend, num_seqs=4, key_fn=lambda s: s.count("C"), top_k=3)
    out = mc(np.zeros((4, 4, 3)), np.zeros((0, 3)), [], [False] * 4,
             temperature=0.1, num_seqs=1)
    assert out == ["CCCA", "CCAA", "CAAA"]              # m best, descending

    # top_k=1 byte-identical to the single-winner behaviour.
    mc1 = MultiCandidate(backend, num_seqs=4, key_fn=lambda s: s.count("C"))
    assert mc1(np.zeros((4, 4, 3)), np.zeros((0, 3)), [], [False] * 4,
               temperature=0.1, num_seqs=1) == ["CCCA"]

    # top_k > num_seqs raises in __init__.
    with pytest.raises((AssertionError, ValueError)):
        MultiCandidate(backend, num_seqs=4, top_k=5)


# ── 2. expand keeps B*m children with correct parent_id ─────────────────────────

def test_expand_keeps_m_children_with_parent_id():
    parent = _seed_state(l_seq="ACDEFGHIK")
    parent.id = 7
    parent.cif_path = "/tmp/beam/predicted_parent"   # expand is only ever called on a predicted parent
    design_fn = _seq_pool_design_fn(["MMMMMMMMM", "WWWWWWWWW", "YYYYYYYYY"])
    children = expand_state(parent, design_fn, _fake_extract, next_id=lambda: _ctr())
    assert len(children) == 3
    assert all(isinstance(c, BeamState) for c in children)
    assert all(c.parent_id == 7 for c in children)
    assert all(c.cycle == parent.cycle + 1 for c in children)
    # children carry distinct ids and the designed L-seqs (length preserved).
    assert [c.l_seq for c in children] == ["MMMMMMMMM", "WWWWWWWWW", "YYYYYYYYY"]
    assert len({c.id for c in children}) == 3


_CTR = [100]


def _ctr():
    _CTR[0] += 1
    return _CTR[0]


# ── 3. prune keeps top-B by COMPOSITE (vetoed excluded even with high ipTM) ─────

def _referee_fn(child):
    """RefereeScore straight from the child's predicted fields (CPU stand-in for the reused
    design_alpha referee that reads the scored CIF)."""
    return RefereeScore(
        chirality_violation=child.chirality, iptm=child.iptm,
        interface_plddt=80.0, pll=None, mirror_discrepancy=None,
        composition_violation=False, helix_fraction=None,
    )


def _scored_child(l_seq, iptm, chir, cid):
    c = _seed_state(l_seq)
    c.id, c.iptm, c.chirality = cid, iptm, chir
    return c


def test_prune_keeps_top_b_by_composite():
    # Four clean children with increasing ipTM; binding term is relative (min-max) so the
    # highest-ipTM pair should be the top-2 kept.
    children = [
        _scored_child("AAAAAAAAA", iptm=0.40, chir=0.0, cid=1),
        _scored_child("CCCCCCCCC", iptm=0.60, chir=0.0, cid=2),
        _scored_child("DDDDDDDDD", iptm=0.80, chir=0.0, cid=3),
        _scored_child("EEEEEEEEE", iptm=0.70, chir=0.0, cid=4),
    ]
    panel = JudgePanel(weights=_ALPHA_WEIGHTS)
    kept, pool, result = prune(children, _referee_fn, panel, keep_b=2)
    assert len(kept) == 2
    kept_ids = {c.id for c in kept}
    assert kept_ids == {3, 4}                            # the two highest ipTM
    # every non-vetoed child carries its composite; pool holds all non-vetoed children.
    assert all(c.composite is not None for c in kept)
    assert len(pool) == 4


# ── 4. prune is chirality-VETOED, not ipTM-thresholded (WINNER POOL invariant) ──

def test_prune_vetoes_chirality_not_iptm_threshold():
    # Highest-ipTM child is chirality-dirty (0.5 > 0.10 veto). The compliance invariant is that
    # it is hard-VETOED (chirality FIRST, binding only a relative gradient — never an absolute
    # ipTM cutoff) and so NEVER enters the WINNER POOL, no matter how good its binding is.
    #
    # Post-ADR (beam GENERALIZES greedy): prune now returns (advance, clean_pool, result).
    # ``advance`` ranks ALL children by the SOFT score and MAY carry a dirty child forward (the
    # search advances through dirty intermediates, like the accept-always greedy loop). The hard
    # guarantee lives in ``clean_pool``: the vetoed child is excluded there, so it can never be
    # SELECTED as a winner.
    children = [
        _scored_child("DDDDDDDDD", iptm=0.95, chir=0.50, cid=1),   # best ipTM, DIRTY
        _scored_child("CCCCCCCCC", iptm=0.55, chir=0.00, cid=2),   # clean, lower ipTM
        _scored_child("EEEEEEEEE", iptm=0.50, chir=0.00, cid=3),   # clean, lower ipTM
    ]
    panel = JudgePanel(weights=_ALPHA_WEIGHTS)
    advance, clean_pool, result = prune(children, _referee_fn, panel, keep_b=2)
    # The dirty child is hard-vetoed and excluded from the selectable WINNER POOL.
    assert result.vetoed[0] is True
    assert {c.id for c in clean_pool} == {2, 3}
    assert all(c.id != 1 for c in clean_pool)             # vetoed never enters the winner pool
    # Advancement is by SOFT score and is allowed to include the dirty child (no hard veto on
    # the beam) — but it is still non-empty and bounded by keep_b.
    assert 0 < len(advance) <= 2


# ── 5. dedup drops identical child seqs (call-count + child==parent) ────────────

def test_dedup_drops_identical_child_seqs():
    parent = _seed_state(l_seq="ACDEFGHIK")
    # one child equals the parent (skip), two collide with each other (keep first only),
    # one is novel.
    c_parent = _scored_child("ACDEFGHIK", 0.5, 0.0, 1)
    c_dup_a = _scored_child("MMMMMMMMM", 0.5, 0.0, 2)
    c_dup_b = _scored_child("MMMMMMMMM", 0.6, 0.0, 3)
    c_new = _scored_child("WWWWWWWWW", 0.5, 0.0, 4)
    seen = {parent.l_seq}
    kept = dedup([c_parent, c_dup_a, c_dup_b, c_new], seen, parent_l_seq=parent.l_seq)
    kept_seqs = [c.l_seq for c in kept]
    assert "ACDEFGHIK" not in kept_seqs                   # child == parent skipped
    assert kept_seqs.count("MMMMMMMMM") == 1              # intra-batch collision deduped
    assert "WWWWWWWWW" in kept_seqs
    assert len(kept) == 2

    # The predict backend is only called for the surviving (deduped) children.
    backend = _FakePredictBackend()
    cost = CostAccount()
    predict_children(kept, backend, cost, ref_time_steps=50, out_dir="/tmp/beam")
    assert backend.calls == 2
    assert cost.predicts == 2


# ── 6. full-cycle predict count == 1 + m + (C-1)*B*m ────────────────────────────

def test_beam_search_full_cycle_predict_count():
    B, m, C = 2, 2, 3
    seed = _seed_state(l_seq="ACDEFGHIK")
    backend = _FakePredictBackend(iptm_for=lambda s: 0.5 + 0.001 * (hash(s) % 100))
    cost = CostAccount()
    panel = JudgePanel(weights=_ALPHA_WEIGHTS)

    # A collision-free design_fn: every child seq is unique per (cycle, branch, candidate) so the
    # cost formula is exact (no dedup hits).
    def design_fn(design_backbone, context_coords, context_elements,
                  fixed_mask, temperature, num_seqs):
        n = design_backbone.shape[0]
        design_fn.k += 1
        base = "CDEFGHIKLMNPQRSTVWY"
        return [(base[(design_fn.k * 3 + j) % len(base)] * n)[:n]
                for j in range(m)]
    design_fn.k = 0

    pool, cost = beam_search(
        seed, design_fn=design_fn, predict_fn=backend, extract_fn=_fake_extract,
        referee_fn=_referee_fn, panel=panel, beam_width=B, children_per_branch=m,
        cycles=C, cost=cost, dedup_on=False,
    )
    # The seed must be PREDICTED with its cif_path set before the first expand — otherwise
    # expand_state(seed)->extract_fn raises (the real _extract guards cif_path is None). This
    # asserting the bug-fix: the pre-fix bootstrap set prediction/iptm/chirality but NOT cif_path,
    # so _fake_extract (now guarded like the real one) would raise on cycle 1.
    assert seed.cif_path is not None
    assert cost.predicts == 1 + m + (C - 1) * B * m       # 1 + 2 + 2*2*2 = 11
    assert backend.calls == cost.predicts
    assert len(pool) >= 1


# ── 7. anneal monotonicity + greedy accept => final >= seed ─────────────────────

def test_anneal_schedule_monotone_and_greedy_improves():
    sched = AnnealSchedule(base_ref_time_steps=50, anneal_start=200, temp_start=0.3)
    N = 5
    rts = [sched.ref_time_steps(i, N) for i in range(N)]
    temps = [sched.mpnn_temperature(i, N, 0.1) for i in range(N)]
    # ref_time_steps decreasing (200 -> 50) and temperature cooling (0.3 -> 0.1).
    assert all(b <= a for a, b in zip(rts, rts[1:]))
    assert all(b <= a for a, b in zip(temps, temps[1:]))
    assert rts[0] >= rts[-1] and temps[0] >= temps[-1]

    # greedy(zero-T) accept over a monotonically improving fake loop => final score >= seed.
    class _Improving:
        def __init__(self):
            self.calls = 0

        def __call__(self, state, ref_time_steps, out_dir):
            self.calls += 1
            return _fake_prediction("ACDEFGHIK", iptm=0.5 + 0.1 * self.calls, chirality=0.0)

    loop = HalluLoop(backend=None, sequence_update_fn=lambda p: "ACDEFGHIK",
                     score_fn=lambda p: p.iptm, refine_fn=_Improving())
    init = LoopState(d_fasta="(DAL)(DAL)(DAL)", coords=np.zeros((9, 3)))
    history = loop.run(init, iterations=5, ref_time_steps=50, out_dir="/tmp/anneal",
                       accept_fn=greedy_iptm_accept(min_delta=0.0), schedule=sched)
    assert HalluLoop.best(history).score >= history[0].score


# ── 8. cost accounting matches the formula (with / without anneal) ──────────────

def test_cost_accounting_matches_formula():
    B, m, C, A = 3, 3, 3, 5
    cost = CostAccount()
    # seed (1) + first cycle expansion (m) + (C-1) cycles of B*m.
    expected_search = 1 + m + (C - 1) * B * m
    cost.add_predicts(expected_search)
    assert cost.predicts == expected_search

    # anneal adds anneal_top_n * anneal_steps predicts (3 * 5 = 15) by default.
    cost.add_predicts(3 * A)
    assert cost.predicts == expected_search + 3 * A
    summary = cost.summary()
    assert "predict" in summary.lower()
    assert str(cost.predicts) in summary


# ── 9. beam reduces to greedy when B=1, m=1, anneal=0 ───────────────────────────

def test_beam_reduces_to_greedy_when_b1_m1():
    B, m, C = 1, 1, 4
    seed = _seed_state(l_seq="ACDEFGHIK")
    # strictly improving ipTM per distinct seq so the single-beam path hill-climbs.
    table = {}

    def iptm_for(s):
        table.setdefault(s, 0.5 + 0.05 * len(table))
        return table[s]

    backend = _FakePredictBackend(iptm_for=iptm_for)
    cost = CostAccount()
    panel = JudgePanel(weights=_ALPHA_WEIGHTS)

    # distinct child each cycle so B=1 single path keeps moving (no dedup collapse).
    def design_fn(design_backbone, context_coords, context_elements,
                  fixed_mask, temperature, num_seqs):
        n = design_backbone.shape[0]
        design_fn.k += 1
        return ["CDEFGHIKLMNPQRSTVWY"[design_fn.k % 19] * n]
    design_fn.k = 0

    pool, cost = beam_search(
        seed, design_fn=design_fn, predict_fn=backend, extract_fn=_fake_extract,
        referee_fn=_referee_fn, panel=panel, beam_width=B, children_per_branch=m,
        cycles=C, cost=cost, dedup_on=True,
    )
    # B=1,m=1 => 1 (seed) + 1 (first cycle) + (C-1)*1*1 = 1 + N predicts (no anneal here).
    assert cost.predicts == 1 + C
    # the pool's best is a single chain that improved over the seed (greedy reduction).
    best = max(pool, key=lambda c: c.composite)
    assert best.iptm >= seed.iptm


# ── 10. beam GENERALIZES greedy: advance THROUGH vetoed intermediates ────────────
#
# These tests pin the design change (ADR — beam must mirror the greedy loop): the next
# cycle's BEAM advances by a SOFT score (chirality as a heavy soft penalty, NOT a hard
# veto), so an all-vetoed cycle still yields a non-empty beam and the search reaches clean
# basins like greedy iter2+. The WINNER POOL stays hard-veto-clean. The only stop condition
# is literally zero children to expand.

def _multi_design_fn(per_call_seqs):
    """6-arg design_fn that returns a fresh list of L-seqs per CALL (one call per parent per
    cycle). ``per_call_seqs`` is a list-of-lists indexed by call number; each inner list is the
    m candidate L-seqs for that call. Pads/truncates each candidate to the backbone length."""
    def _design(design_backbone, context_coords, context_elements,
                fixed_mask, temperature, num_seqs):
        n = design_backbone.shape[0]
        k = _design.k
        _design.k += 1
        seqs = per_call_seqs[k] if k < len(per_call_seqs) else per_call_seqs[-1]
        out = [(s * n)[:n] if len(s) < n else s[:n] for s in seqs]
        return out[:num_seqs] if num_seqs else out
    _design.k = 0
    return _design


def test_beam_advances_through_all_vetoed_cycle():
    # Cycle 1 (the seed's single beam) yields m children that are ALL hard-vetoed (chirality
    # 0.5 > 0.10). The beam must still advance: a NON-EMPTY next beam (top-B by SOFT score),
    # so cycle 2 expands and the search proceeds rather than dying at cycle 0.
    B, m, C = 2, 3, 2
    seed = _seed_state(l_seq="ACDEFGHIK")

    # Every distinct seq is chirality-dirty -> every child hard-vetoed in BOTH cycles.
    backend = _FakePredictBackend(
        iptm_for=lambda s: 0.5 + 0.001 * (hash(s) % 100),
        chir_for=lambda s: 0.50,
    )
    cost = CostAccount()
    panel = JudgePanel(weights=_ALPHA_WEIGHTS)

    # cycle1: seed expands once -> 3 dirty children; cycle2: B(=2) parents expand -> 2 calls.
    design_fn = _multi_design_fn([
        ["MMMMMMMMM", "WWWWWWWWW", "YYYYYYYYY"],   # cycle1 (seed)
        ["FFFFFFFFF", "PPPPPPPPP", "RRRRRRRRR"],   # cycle2 parent A
        ["KKKKKKKKK", "HHHHHHHHH", "NNNNNNNNN"],   # cycle2 parent B
    ])

    pool, cost = beam_search(
        seed, design_fn=design_fn, predict_fn=backend, extract_fn=_fake_extract,
        referee_fn=_referee_fn, panel=panel, beam_width=B, children_per_branch=m,
        cycles=C, cost=cost, dedup_on=False,
    )
    # The search must NOT have stopped at cycle 1: cycle 2 ran (B*m more predicts).
    #   1 (seed) + m (cycle1) + (C-1)*B*m (cycle2) = 1 + 3 + 1*2*3 = 10
    assert cost.predicts == 1 + m + (C - 1) * B * m
    assert backend.calls == cost.predicts


def test_winner_pool_excludes_vetoed():
    # Even though the beam ADVANCES through dirty children, the WINNER POOL accumulates ONLY
    # hard-veto-passing (clean) children. Here every child in every cycle is dirty -> pool empty.
    B, m, C = 2, 3, 2
    seed = _seed_state(l_seq="ACDEFGHIK")
    backend = _FakePredictBackend(
        iptm_for=lambda s: 0.5 + 0.001 * (hash(s) % 100),
        chir_for=lambda s: 0.50,
    )
    cost = CostAccount()
    panel = JudgePanel(weights=_ALPHA_WEIGHTS)
    design_fn = _multi_design_fn([
        ["MMMMMMMMM", "WWWWWWWWW", "YYYYYYYYY"],
        ["FFFFFFFFF", "PPPPPPPPP", "RRRRRRRRR"],
        ["KKKKKKKKK", "HHHHHHHHH", "NNNNNNNNN"],
    ])
    pool, cost = beam_search(
        seed, design_fn=design_fn, predict_fn=backend, extract_fn=_fake_extract,
        referee_fn=_referee_fn, panel=panel, beam_width=B, children_per_branch=m,
        cycles=C, cost=cost, dedup_on=False,
    )
    # advanced through dirty intermediates, but no clean child anywhere -> empty winner pool.
    assert pool == []


def test_beam_recovers_like_greedy():
    # Mirrors greedy iter0/iter1 dirty -> iter2 clean: cycle1 children all hard-vetoed, but
    # one cycle2 child is CLEAN (chir 0). The final winner pool must contain that clean child
    # (and only clean children), so a downstream anneal/select picks it.
    B, m, C = 2, 3, 2
    seed = _seed_state(l_seq="ACDEFGHIK")
    clean_seq = "FFFFFFFFF"   # the lone clean child, born in cycle 2

    backend = _FakePredictBackend(
        iptm_for=lambda s: 0.80 if s == clean_seq else 0.5 + 0.001 * (hash(s) % 50),
        chir_for=lambda s: 0.0 if s == clean_seq else 0.50,
    )
    cost = CostAccount()
    panel = JudgePanel(weights=_ALPHA_WEIGHTS)
    design_fn = _multi_design_fn([
        ["MMMMMMMMM", "WWWWWWWWW", "YYYYYYYYY"],   # cycle1: all dirty
        [clean_seq, "PPPPPPPPP", "RRRRRRRRR"],     # cycle2 parent A: ONE clean
        ["KKKKKKKKK", "HHHHHHHHH", "NNNNNNNNN"],   # cycle2 parent B: all dirty
    ])
    pool, cost = beam_search(
        seed, design_fn=design_fn, predict_fn=backend, extract_fn=_fake_extract,
        referee_fn=_referee_fn, panel=panel, beam_width=B, children_per_branch=m,
        cycles=C, cost=cost, dedup_on=False,
    )
    # The clean child survived to the pool; every pool member is hard-veto-clean.
    assert any(c.l_seq == clean_seq for c in pool)
    assert all(c.chirality <= 0.10 for c in pool)
    # The winner (top composite over the clean pool) is the clean child.
    best = max(pool, key=lambda c: c.composite)
    assert best.l_seq == clean_seq


def test_no_clean_anywhere_explores_fully():
    # If NO child EVER passes the hard veto, the winner pool is empty BUT the search still
    # EXPLORES all C cycles (no early stop at cycle 0). Cost must reflect the full budget.
    B, m, C = 2, 2, 3
    seed = _seed_state(l_seq="ACDEFGHIK")
    backend = _FakePredictBackend(
        iptm_for=lambda s: 0.5 + 0.001 * (hash(s) % 100),
        chir_for=lambda s: 0.50,   # always dirty
    )
    cost = CostAccount()
    panel = JudgePanel(weights=_ALPHA_WEIGHTS)

    # A collision-free design_fn so the full budget is charged (no dedup hits).
    def design_fn(design_backbone, context_coords, context_elements,
                  fixed_mask, temperature, num_seqs):
        n = design_backbone.shape[0]
        design_fn.k += 1
        base = "CDEFGHIKLMNPQRSTVWY"
        return [(base[(design_fn.k * 3 + j) % len(base)] * n)[:n] for j in range(m)]
    design_fn.k = 0

    pool, cost = beam_search(
        seed, design_fn=design_fn, predict_fn=backend, extract_fn=_fake_extract,
        referee_fn=_referee_fn, panel=panel, beam_width=B, children_per_branch=m,
        cycles=C, cost=cost, dedup_on=False,
    )
    # No clean child anywhere -> empty pool, but FULL exploration cost (all C cycles).
    assert pool == []
    assert cost.predicts == 1 + m + (C - 1) * B * m
    assert backend.calls == cost.predicts
