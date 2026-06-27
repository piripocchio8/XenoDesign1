"""Dispatcher: resolve a DesignConfig → wire a BinderClass's hooks into the shared HalluLoop.

``run_design(cfg)`` is the single entry point behind ``scripts/design.py``. It reproduces the
wiring that the validated ``xenodesign.classes.alpha.run_alpha_design`` driver uses — seed →
(restraint) → L-seed predict + double-flip → wrapper → loop (seq-update / objective / refine /
accept) → referee → JudgePanel → ``cls.report`` — but parameterised over the binder class so
``alpha`` / ``cyclic`` / ``non_alpha`` all run through one path. The HalluLoop control flow and
``xenodesign/loop.py`` are never touched; only WHICH callables the class injects changes.

GPU/IO seams are isolated for CPU testing: ``_registry()`` (the class lookup), ``_make_predictor``
(the ChaiBackend + its predict callable), and ``target_entities`` (the fixed-context entity list)
are the three monkeypatch points the dispatch unit test stubs out.

l_seed_iptm / wall_time_s: the per-class ``report`` hooks (read-only here) hardcode ``0.0`` for
these two fields because they cannot measure them. The DISPATCHER owns both — the L-seed predict's
ipTM and the timed end-to-end wall — so it overwrites those two keys in the report dict with the
real measured values, so the returned report matches ``run_alpha_design``'s output.
"""
from __future__ import annotations

import time
from pathlib import Path

from xenodesign.config import DesignConfig, dump_config
from xenodesign.targets import target_entities

# Re-exported at module level so the dispatch unit test can monkeypatch ``dispatch.abc_search``
# (the engine itself is pure / CPU-clean — no heavy import cost).
from xenodesign.abc.engine import abc_search


def _registry() -> dict:
    """The CLI-axis → BinderClass instance map (monkeypatched in the dispatch unit test)."""
    from xenodesign.classes.base import CLASS_REGISTRY
    return CLASS_REGISTRY


def _ensure_patches():  # pragma: no cover (gpu) — monkeypatched in tests
    """Install the shared chai restraint patches (pocket-name + token-dist match).

    Indirection through this module-level seam so the dispatch unit test can stub it out (the
    real one imports chai). Delegates to the single shared installer used by the legacy
    ``run_restrained_batch`` driver too, so behaviour is identical across both entry points."""
    from xenodesign.chai_patches import ensure_patches
    ensure_patches()


def _make_predictor(cfg):  # pragma: no cover (gpu) — monkeypatched in tests
    """Return ``(backend, predict_fn)``: the Chai backend and its full-predict callable.

    ``predict_fn(entities, out_dir, num_diffn_timesteps=..., constraint_path=...)`` runs a full
    Chai prediction. The dispatch unit test replaces this with a CPU fake.
    """
    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign.config import resolve_device
    backend = ChaiBackend(device=resolve_device(cfg), seed=cfg.seed)
    return backend, backend.predict


class _PredictAdapter:
    """Routes the structure step through the ``predict_fn`` seam while preserving the underlying
    backend's ``truncated_refine`` for the unrestrained legacy path.

    The design_demo wrappers (``_PredictBackendWrapper`` / ``_LoopBackendWrapper``) call
    ``self._backend.predict`` / ``self._backend.truncated_refine``. Wrapping the real backend in
    this adapter lets the dispatch test supply only a ``predict_fn`` (no real backend method) while
    production wiring delegates ``truncated_refine`` to the genuine ChaiBackend method, byte-for-
    byte preserving ``run_alpha_design``'s unrestrained truncated-refine behaviour.
    """

    def __init__(self, backend, predict_fn):
        self._backend = backend
        self._predict_fn = predict_fn

    def predict(self, *a, **k):
        return self._predict_fn(*a, **k)

    def truncated_refine(self, *a, **k):
        fn = getattr(self._backend, "truncated_refine", None)
        if fn is None:  # mocked backend without truncated_refine → fall back to predict.
            return self._predict_fn(*a, **k)
        return fn(*a, **k)


def _seed_to_seq_update(cls, cfg, wrapper, seed_spec, roles=None):
    """Build the loop's sequence_update_fn from the class hook, tolerating a missing ``seq_update``.

    Real classes expose ``seq_update(cfg, wrapper, seed_spec, roles) -> (prediction) -> one_letter``;
    a bare-Protocol mock that omits it falls back to re-emitting the seed (the loop only needs a
    callable that returns a one-letter L sequence). ``roles`` is the dispatch chain contract
    (``ChainRoles``) computed ONCE from the assembled entity list, threaded so the seq-update
    extractor reads the binder chain the contract declares — never a hardcoded letter. It is passed
    positionally only when the hook accepts it, so mocks with a 3-arg ``seq_update`` still work."""
    import inspect

    fn = getattr(cls, "seq_update", None)
    if fn is None:
        return lambda prediction: seed_spec.one_letter
    try:
        accepts_roles = "roles" in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        accepts_roles = False
    if accepts_roles:
        return fn(cfg, wrapper, seed_spec, roles=roles)
    return fn(cfg, wrapper, seed_spec)


def _call_referee(cls, cfg, loop_dir, esm_judge, roles):
    """Call the class's referee hook, threading the chain contract ``roles`` only when the hook
    accepts it (so a 3-arg mock referee in the unit tests still works). The referee reads the
    binder chain for its helix/seq metrics; routing ``roles`` makes a non-'B' binder read right."""
    import inspect

    fn = cls.referee
    try:
        accepts_roles = "roles" in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        accepts_roles = False
    if accepts_roles:
        return fn(cfg, loop_dir, esm_judge, roles=roles)
    return fn(cfg, loop_dir, esm_judge)


def _maybe_esm(cfg):
    """Lazy ESM-PLL judge for the referee, or None when PLL is off (CPU-clean when off)."""
    if not cfg.use_pll:
        return None
    from xenodesign.judges.plm_judge import ESMPseudoLogLikelihood
    return ESMPseudoLogLikelihood(device=cfg.gates.esm_device or "cpu")


def _run_beam(cls, cfg, loop, init, loop_dir, roles=None):  # pragma: no cover (gpu wiring beyond mocked branch)
    """Route the design loop through the BEAM+ANNEAL search instead of the greedy HalluLoop.

    Wires the existing ``xenodesign.beam`` machinery (``beam_search`` + ``anneal_best``) through
    the SAME per-class hooks the greedy path uses: the loop's predict-wrapper (``loop._backend``)
    is the beam ``predict_fn``; the class supplies the panel ss_bias and (via the dispatcher's
    ``_seed_to_seq_update``) the anneal sequence-update; the design-time beam helpers
    (``_make_extract_fn`` / ``_make_beam_referee_fn`` / ``_make_anneal_referee_fn``) are reused
    from ``scripts.design_alpha_beam`` (the validated α beam driver — the A/B chain convention the
    α and non_alpha classes share). Returns a ``history`` list[LoopStep] compatible with the
    panel-select tail of :func:`run_design` (so the SAME referee → JudgePanel → ``cls.report``
    runs over the beam/anneal trajectory).
    """
    from xenodesign.beam import BeamState, CostAccount, anneal_best, beam_search
    from xenodesign.benchmark.cases import get_case, ss_bias_config_for_case
    from xenodesign.inverse_folding import MultiCandidate
    from xenodesign.judges.panel import JudgePanel
    from xenodesign.loop import HalluLoop, LoopState
    from xenodesign.scorer import sequence_quality_key
    from xenodesign.sequence_update import _ligandmpnn_design_fn

    from scripts.design_alpha import _cterm_gly_anchor, _ensure_cterm_glycine
    from scripts.design_alpha_beam import (
        _make_anneal_referee_fn, _make_beam_referee_fn, _make_extract_fn,
    )

    wrapper = loop._backend                 # the class-built predict-wrapper (also the predict_fn)
    case = get_case(cls.case_id)
    esm_judge = _maybe_esm(cfg)

    # ── Injected beam callables (built from the class hooks + shared α-shaped seams) ──
    design_fn = MultiCandidate(_cterm_gly_anchor(_ligandmpnn_design_fn),
                               num_seqs=cfg.loop.num_seqs, key_fn=sequence_quality_key,
                               top_k=cfg.loop.beam_width)
    extract_fn = _make_extract_fn(roles=roles)  # beam extractor reads the binder chain from the contract
    referee_fn = _make_beam_referee_fn(esm_judge=esm_judge)
    panel = JudgePanel(ss_bias=cls.ss_bias(cfg, case))
    cost = CostAccount()

    # ── BEAM search over cfg.loop.beam_cycles cycles at width cfg.loop.beam_width ──
    seed_state = BeamState(d_fasta=init.d_fasta, coords=init.coords, cycle=0)
    pool, cost = beam_search(
        seed_state, design_fn=design_fn, predict_fn=wrapper, extract_fn=extract_fn,
        referee_fn=referee_fn, panel=panel, beam_width=cfg.loop.beam_width,
        children_per_branch=cfg.loop.beam_width, cycles=cfg.loop.beam_cycles, cost=cost,
        anchor_fn=_ensure_cterm_glycine, ref_time_steps=cfg.loop.ref_time_steps,
        out_dir=loop_dir / "beam",
    )

    # ── ANNEAL polish over the clean pool, reusing the class's anneal seq-update + objective ──
    def _make_loop_fn(_state):
        return HalluLoop(backend=wrapper,
                         sequence_update_fn=_seed_to_seq_update(cls, cfg, wrapper, _state, roles=roles),
                         score_fn=loop._score_fn, refine_fn=wrapper)

    def _make_init_fn(state):
        return LoopState(d_fasta=state.d_fasta, coords=state.coords)

    _best_step, anneal_states, cost = anneal_best(
        pool, make_loop_fn=_make_loop_fn, make_init_fn=_make_init_fn,
        score_fn=loop._score_fn, panel=panel, referee_fn=_make_anneal_referee_fn(esm_judge),
        top_n=cfg.loop.beam_width, anneal_steps=cfg.loop.beam_cycles,
        base_ref_time_steps=cfg.loop.ref_time_steps, cost=cost, out_dir=loop_dir / "anneal",
    )

    # ``anneal_states`` is a list[LoopStep] (HalluLoop history) — exactly the shape run_design's
    # referee → JudgePanel → cls.report tail consumes.
    return anneal_states


def length_sweep_ladder() -> list:
    """A coarse, budget-bounded from-scratch length ladder, clamped to [6, 50] (ascending, deduped).

    Deliberately short (≤6 rungs) so the sweep stays cheap: the dispatcher runs ONE full design
    per rung. By design: keep it coarse + budget-bounded."""
    from xenodesign.config import BINDER_LENGTH_MAX, BINDER_LENGTH_MIN
    raw = [8, 12, 16, 24, 32]
    return sorted({max(BINDER_LENGTH_MIN, min(BINDER_LENGTH_MAX, n)) for n in raw})


def run_length_sweep(cfg: DesignConfig, ladder=None) -> dict:
    """Run a coarse length ladder and return the single best-by-objective design result.

    Dispatches :func:`run_design` once per rung (a fresh copy of ``cfg`` with that
    ``binder_length``), then picks the rung whose result maximises the objective
    (``selected_iptm`` — the same field every class report writes; higher is better for the
    ipTM/mixed/ipsae objectives). The winning result dict carries ``binder_length`` so the CLI
    can report which length won. ``run_design`` is referenced via the module global so tests can
    monkeypatch it."""
    import copy
    from xenodesign.config import resolve_binder_length

    rungs = ladder if ladder is not None else length_sweep_ladder()
    best, best_score = None, float("-inf")
    for n in rungs:
        rung_cfg = copy.deepcopy(cfg)
        rung_cfg.binder_length = int(n)
        result = run_design(rung_cfg)
        result.setdefault("binder_length", resolve_binder_length(rung_cfg))
        score = result.get("selected_iptm")
        score = float(score) if score is not None else float("-inf")
        if score > best_score:
            best, best_score = result, score
    return best


def _is_mixed_chirality(cfg: DesignConfig) -> bool:
    """The ABC search applies ONLY to mixed-chirality cases: cyclic + target_type=none (a free
    mixed-chirality macrocycle / peptide). Homochiral classes (alpha, all-D non_alpha) keep the
    existing greedy/beam loop."""
    return cfg.binder_class == "cyclic" and cfg.target.target_type == "none"


def _run_abc(cfg: DesignConfig, backend, seed_one_letter: str, out_dir: "Path") -> dict:
    """Wire + run the ABC mixed-chirality search and assemble the result dict.

    Objective (decided 2026-06-25): pTM (primary) + C-N termini-distance closure proxy
    (secondary) at K*=10-25 fast Chai steps with the head-to-tail CLOSURE restraint only (NO
    target-specific coordination). Variant A: ABC searches chirality, MPNN fills identity;
    Variant B: ABC searches identity+chirality, MPNN warm-start only.

    ``abc_search`` is referenced via the module global so the dispatch test can monkeypatch it.
    ``loop.py`` is never touched — this is a parallel search path, not a HalluLoop step.
    """
    import random

    from xenodesign.abc.engine import FoodSource
    from xenodesign.abc.fitness import make_abc_fitness
    from xenodesign.abc.moves import seed_chirality_pattern
    from xenodesign.abc.variants import abc_variant_a_design_fn, abc_variant_b_design_fn

    knobs = cfg.abc
    rng = random.Random(cfg.seed)

    fitness_fn = make_abc_fitness(
        backend, k_star=knobs.fitness_steps,
        w_ptm=knobs.w_ptm, w_termini=knobs.w_termini, closure=True,
        out_root=out_dir / "abc_evals",
    )

    if knobs.variant == "b":
        design_fn = abc_variant_b_design_fn(rng=rng, mutation_rate=knobs.chirality_move_rate)
    else:
        # Variant A: the coordinate-only LigandMPNN adapter (default SequenceUpdater backend).
        from xenodesign.sequence_update import _ligandmpnn_design_fn
        design_fn = abc_variant_a_design_fn(_ligandmpnn_design_fn)

    n = len(seed_one_letter)
    init_pattern = seed_chirality_pattern(n, rng=rng)
    init_pop = [FoodSource(identity=seed_one_letter, chirality_pattern=init_pattern,
                           last_structure=None, nectar=None)]

    best, history = abc_search(
        init_pop, fitness_fn, design_fn,
        n_cycles=knobs.cycles, colony_size=knobs.colony_size,
        scout_limit=knobs.scout_limit, chai_eval_budget=knobs.chai_eval_budget, rng=rng,
    )

    result = {
        "case_id": "cyclic",
        "search": "abc",
        "abc_variant": knobs.variant,
        "fitness_steps": knobs.fitness_steps,
        "selected_nectar": (float(best.nectar) if best is not None and best.nectar is not None
                            else None),
        "selected_d_fasta": (best.identity if best is not None else None),
        "selected_chirality_pattern": (dict(best.chirality_pattern) if best is not None else None),
        "n_cycles": len(history),
        "history": history,  # per-cycle [{"cycle","best_nectar","evals_used"}] — convergence curve
        "out_dir": str(out_dir),
    }
    import json
    (out_dir / "abc_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def run_design(cfg: DesignConfig) -> dict:
    """Run the multi-class hallucination design loop end-to-end and return the result dict.

    Mirrors ``run_alpha_design``'s wiring, parameterised over ``cfg.binder_class``:
    resolve class → build target → seed → (restraint) → L-seed predict + double-flip →
    wrapper → HalluLoop(seq_update / objective / refine / accept) → referee → JudgePanel →
    ``cls.report``. ``resolved_config.json`` is dumped BEFORE any predict (provenance), and the
    real L-seed ipTM + timed wall overwrite the report's placeholder fields.
    """
    from xenodesign.benchmark.cases import get_case
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.judges.panel import JudgePanel
    from xenodesign.loop import HalluLoop, LoopState
    from xenodesign.seed import reflect_binder_in_complex_from_cif

    from xenodesign.cif_io import _best_cif_path
    from xenodesign.backends.wrappers import _LoopBackendWrapper, _PredictBackendWrapper

    t0 = time.time()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out_dir)  # provenance BEFORE any predict

    # Install the chai restraint patches (idempotent) BEFORE target resolution / predict, so the
    # metal target gate is open and pocket/coordination restraints apply on the dispatch path —
    # mirroring scripts.run_restrained_batch. Skipped when restraints are off (no chai import in
    # CPU tests, which run --no_restraints and monkeypatch target_entities/_make_predictor).
    if cfg.restraints_on:
        _ensure_patches()

    cls = _registry()[cfg.binder_class]
    case = get_case(cls.case_id)
    entities, msa_dir, _hint = target_entities(cfg)
    target_entity = entities[0] if entities else None
    target_seq = (target_entity.get("sequence", "") if target_entity else "")

    # CHAIN CONTRACT (build ONCE, thread everywhere): chai labels chains A,B,C… by entity order
    # and the binder is appended LAST, so the assembled target ``entities`` fully determine the
    # binder/target chain letters. Every downstream consumer that would otherwise guess a chain
    # (the seq-update extractor, the double-flip reflection, the beam extractor) reads from this —
    # no scattered ``chr(...)`` / hardcoded "B". This makes chain-misidentification impossible for
    # ANY binder class / target chemistry (alpha->B, non_alpha [HA1,HA2]->C, cyclic [Zn]->B, none->A).
    from xenodesign.targets import ChainRoles
    roles = ChainRoles.from_entities(entities)

    seed_spec = cls.seed(cfg, target_seq)
    backend, predict_fn = _make_predictor(cfg)
    adapter = _PredictAdapter(backend, predict_fn)

    # ── ABC mixed-chirality search branch (--search abc) ──────────────────────────
    # Routes mixed-chirality cases (cyclic + target_type=none) through abc_search with the fast
    # pTM+termini fitness + the chosen variant, IN PLACE of the per-iter HalluLoop step. Homochiral
    # classes are guarded out (greedy/beam only). ``loop.py`` is never touched.
    if cfg.loop.search == "abc":
        if not _is_mixed_chirality(cfg):
            raise ValueError(
                "--search abc is mixed-chirality only (cyclic + target_type=none); "
                f"got binder_class={cfg.binder_class!r} target_type={cfg.target.target_type!r}. "
                "Homochiral classes use --search greedy/beam.")
        return _run_abc(cfg, adapter, seed_spec.one_letter, out_dir)

    # Restraint emission (class-specific; consulted only when restraints are on).
    constraint_path = (cls.restraints(cfg, case, out_dir, (entities, msa_dir))
                       if cfg.restraints_on else None)

    # ── L-seed predict (real l_seed_iptm) + double-flip → D-correct seed coords ──
    # Mirrors run_alpha_design's [2/4] step. The binder is appended LAST so multi-chain / ligand
    # targets order correctly; its chain letter is chr('A'+len(entities)) (B for α single-target,
    # C for the 2-chain HA target). MSA is forwarded so an MSA'd target seeds faithfully. CIF
    # reflection is GPU/IO; on any failure (e.g. the mocked predictor) fall back to coords=None.
    binder_chain = roles.binder  # from the ONE chain contract — not a re-derived chr(...)
    l_seed_dir = out_dir / "p0_l_seed"
    l_seed_pred = predict_fn(
        [*entities,
         {"type": "protein", "name": "binder",
          "sequence": seed_spec.one_letter, "chirality": "L"}],
        l_seed_dir, num_diffn_timesteps=200, constraint_path=constraint_path,
        msa_directory=msa_dir)
    l_seed_iptm = float(getattr(l_seed_pred, "iptm", 0.0))
    try:
        d_seed_coords = reflect_binder_in_complex_from_cif(
            _best_cif_path(l_seed_dir), binder_chain=binder_chain, axis=0)
    except Exception:
        d_seed_coords = None

    # ── Wrapper: restrained → PREDICT-mode (constraints honoured); else truncated-refine ──
    if constraint_path is not None:
        wrapper = _PredictBackendWrapper(adapter, entities,
                                         constraint_path=constraint_path,
                                         msa_directory=msa_dir)
        refine_fn = wrapper
    else:
        wrapper = _LoopBackendWrapper(adapter, entities, msa_directory=msa_dir)
        refine_fn = None

    loop = HalluLoop(
        backend=wrapper,
        sequence_update_fn=_seed_to_seq_update(cls, cfg, wrapper, seed_spec, roles=roles),
        score_fn=cls.objective(cfg, wrapper),
        refine_fn=refine_fn,
    )
    init = LoopState(d_fasta=to_d_fasta(seed_spec.one_letter), coords=d_seed_coords)
    loop_dir = out_dir / "loop"

    if cfg.loop.search == "beam":
        history = _run_beam(cls, cfg, loop, init, loop_dir, roles=roles)
    else:
        history = loop.run(init=init, iterations=cfg.loop.iters,
                           ref_time_steps=cfg.loop.ref_time_steps, out_dir=loop_dir,
                           accept_fn=cls.accept_fns(cfg))

    # ── Referee + adversarial panel selection ──────────────────────────────────
    # Classes whose referee returns None per step (e.g. the cyclic recall case) get a neutral
    # RefereeScore built from the step's prediction, so JudgePanel.combine never sees a None.
    from xenodesign.judges.panel import RefereeScore

    esm_judge = _maybe_esm(cfg)
    referee_fn = _call_referee(cls, cfg, loop_dir, esm_judge, roles)
    referee_scores = []
    for i, step in enumerate(history):
        rs = referee_fn(step, i)
        if rs is None:
            rs = RefereeScore(chirality_violation=0.0,
                              iptm=float(getattr(step.prediction, "iptm", 0.0)))
        referee_scores.append(rs)
    panel = JudgePanel(score_fn=lambda step: None, ss_bias=cls.ss_bias(cfg, case))
    panel_result = panel.combine(referee_scores)

    # l_seed_iptm / wall_time_s are measured by the dispatcher (the report hooks can't see them);
    # thread them INTO report() so both the returned dict AND the on-disk *_result.json carry the
    # real values (JSON parity with the legacy single-class CLI).
    return cls.report(cfg, history, panel_result, case, out_dir,
                      l_seed_iptm=l_seed_iptm, wall_time_s=float(time.time() - t0))
