"""α-class INTERNALS (MOD-3 split): restraint / seed / backend-selector / sequence-update /
objective / referee / result-assembly helpers extracted out of ``classes/alpha.py`` so that
module stays a thin CONTRACT (the :class:`~xenodesign.classes.alpha.Alpha` BinderClass adapter +
the ``run_alpha_design`` driver). Behaviour is byte-for-byte identical — this is a move.

Monkeypatch contract (preserved): collaborators that the legacy CPU tests patch on the
``xenodesign.classes.alpha`` module (``_best_cif_path`` / ``_all_atoms_from_chain`` /
``binder_seq_from_cif`` / ``build_alpha_restraint`` / ``make_alpha_seq_update_fn`` /
``_make_base_backend`` / ``_cterm_gly_anchor`` / ``_ligandmpnn_design_fn`` /
``carbonara_design_fn``) are resolved at CALL TIME through :func:`_self`, which returns the
PUBLIC ``xenodesign.classes.alpha`` module object (which re-exports every name defined here).
So a test patching one of those names on the public module is honoured even though the body
lives here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# CIF/backend plumbing (package-resident since MOD-1). Imported into THIS module's namespace so
# the public alpha module re-exports them and tests can patch them on the public module.
from xenodesign.cif_io import (
    _all_atoms_from_chain,
    _backbone_array_from_residues,
    _best_cif_path,
    _chirality_violation_frac_from_cif,
)
from xenodesign.backends.wrappers import _LoopBackendWrapper

# Inverse-folding bases (CPU-clean to import — heavy deps deferred to call time).
from xenodesign.carbonara_backend import carbonara_design_fn
from xenodesign.sequence_update import _ligandmpnn_design_fn


_DEFAULT_N_ITERS = 30
_DEFAULT_REF_TIME_STEPS = 50
_DEFAULT_NUM_SEQS = 8          # spec §5 drift lever (oversample); cheap — LigandMPNN only
_DEFAULT_DEVICE = None  # unset -> resolve_device() (XENO_DEVICE / cuda:0 if avail / mps / cpu)
# ALPHA-SPECIFIC (not framework-general): the α case FASTA holds the binder (record A) AND the
# fixed L-HLH target (record B); this names record B. Only the α class default-target path reads
# it — the shared targets.target_entities reaches it via the alpha-guarded _alpha_target_record(cfg).
# Imported + asserted by name in the alpha oracle (tests/test_design_alpha.py, tests/test_targets.py),
# so it is part of the monkeypatch contract: do NOT rename without a back-compat alias.
_TARGET_RECORD = "trimer_DL_ABLE_B"   # chain B = the fixed L-HLH target (NOT record A = binder)

# RUN chain convention (TASK 2): _LoopBackendWrapper builds entities=[target, binder], so Chai
# labels TARGET=chain A, BINDER=chain B. The binder (designed) chain is therefore 'B' EVERYWHERE
# in this run — chirality read, CIF->seq, metrics chain_b=1. The cases.py _ALPHA.restraint
# nominal params (binder_chain='A'/target_chain='B', the GT fasta order) are INVERTED vs this run
# and MUST be re-emitted with the run's chains when building the restraint (build_alpha_restraint).
#
# REDUNDANT on the shared dispatch path: these are now only the α-STANDALONE-DRIVER defaults.
# dispatch.run_design builds the authoritative chain assignment ONCE from the assembled entity
# list (targets.ChainRoles.from_entities) and THREADS ``roles`` into every reuse of these helpers,
# so each consumer below collapses to ``roles.binder``/``roles.context`` and only falls back to
# these constants when ``roles is None`` (the α-only standalone CLI). Kept (not deleted) because
# (a) they ARE the correct α fallback and (b) design_alpha_beam.py imports them by name.
_RUN_BINDER_CHAIN = "B"
_RUN_TARGET_CHAIN = "A"

# Composition floor (TASK 5, anti poly-Ala) thresholds.
_COMP_MAX_SINGLE_AA_FRAC = 0.30   # reject if any single residue exceeds this fraction
_COMP_MAX_ALA_GLY_FRAC = 0.40     # reject if (Ala+Gly) fraction exceeds this
_COMP_MIN_NORM_ENTROPY = 0.55     # reject if normalized Shannon entropy H/log2(20) below this
_COMP_MAX_HOMOPOLYMER_RUN = 3     # reject if any homopolymer run is >= 4 (i.e. > this)

# ALPHA-SPECIFIC panel composite weights (#37 PLL demotion + P1c SS-bias carve) — NOT the
# framework default. Other classes build their own JudgePanel weights; this dict is the α tuning.
# Mirrored by name in scripts/design_alpha_beam.py and tests/test_beam.py (oracle), so it is part
# of the monkeypatch contract: rename only with a back-compat alias.
_ALPHA_WEIGHTS = {"chirality": 0.40, "binding": 0.40, "pll": 0.05, "mirror": 0.05,
                  "ss_bias": 0.10}


def _self():
    """Return the PUBLIC ``xenodesign.classes.alpha`` module, for monkeypatch-honouring
    call-time attribute lookups.

    Helpers below read a collaborator at call time (``_ligandmpnn_design_fn`` /
    ``carbonara_design_fn`` / ``_make_base_backend`` / ``_cterm_gly_anchor`` / ``_best_cif_path`` /
    ``_all_atoms_from_chain`` / ``binder_seq_from_cif`` / ``make_alpha_seq_update_fn`` /
    ``build_alpha_restraint``) through this, so a test patching one of those names on the public
    alpha module is honoured. (MOD-3: the bodies live in this internals module but the public
    module re-exports them and is what tests patch.)
    """
    import importlib
    return importlib.import_module("xenodesign.classes.alpha")


# ── CIF → binder-chain sequence (TASK 1 off-by-one fix) ────────────────────────

def binder_seq_from_cif(cif_path, chain_name: str = _RUN_BINDER_CHAIN) -> str:
    """Extract the binder chain's one-letter L sequence from a scored CIF.

    This is the sequence that ACTUALLY produced the metrics read from that CIF — the fix for
    the #31 off-by-one: the loop's ``step.state.d_fasta`` is the NEXT sequence designed that
    iteration, mismatched-by-one with the structure/metrics at ``iter_{i}/chai_out``. Reporting
    the chain read here guarantees ``selected_l_seq`` == the chain-B sequence that was scored.

    Reads residue 3-letter codes (D-CCD like 'DAL' or canonical 'GLY') from the named chain via
    gemmi and maps them D->L->one-letter through xenodesign.io_spec maps. Achiral glycine stays
    'G'; modified residues (e.g. MSE) normalize to their standard parent.
    """
    import gemmi

    from xenodesign.io_spec import AA3_TO_AA1, MODIFIED_TO_STANDARD
    from xenodesign.mirror import D_TO_L

    structure = gemmi.read_structure(str(cif_path))
    out: list[str] = []
    for model in structure:
        chain = None
        for ch in model:
            if ch.name == chain_name or ch.name == chain_name.lower():
                chain = ch
                break
        if chain is None:
            raise RuntimeError(
                f"chain {chain_name!r} not found in {cif_path} "
                f"(chains: {[c.name for c in model]})")
        for res in chain:
            three = res.name.upper()
            three = D_TO_L.get(three, three)            # D-CCD -> L 3-letter
            three = MODIFIED_TO_STANDARD.get(three, three)
            try:
                out.append(AA3_TO_AA1[three])
            except KeyError:
                raise RuntimeError(
                    f"residue {res.name!r} in chain {chain_name} of {cif_path} has no "
                    f"one-letter mapping") from None
        break   # first model only
    if not out:
        raise RuntimeError(f"chain {chain_name!r} in {cif_path} has no residues")
    return "".join(out)


# ── Composition floor (TASK 5, anti poly-Ala) ──────────────────────────────────

def composition_violation(seq: str) -> bool:
    """Pure-CPU low-complexity veto for a one-letter L sequence (anti poly-Ala).

    Returns True (REJECT) when the sequence is degenerate by any of:
      - a single amino acid exceeds 30% of the sequence, OR
      - (Ala + Gly) together exceed 40%, OR
      - normalized Shannon entropy H / log2(20) < 0.55, OR
      - any homopolymer run of length >= 4.

    A real, diverse binder passes all four; poly-Ala / low-entropy junk fails. Empty/whitespace
    input is treated as a violation (nothing to select). Calibrated so the genuine α GT binder
    passes (asserted programmatically in the CPU tests; the GT seq is never inlined).
    """
    import math

    s = (seq or "").upper()
    n = len(s)
    if n == 0:
        return True

    from collections import Counter
    counts = Counter(s)

    # 1. single-AA dominance
    if max(counts.values()) / n > _COMP_MAX_SINGLE_AA_FRAC:
        return True

    # 2. Ala + Gly floor
    if (counts.get("A", 0) + counts.get("G", 0)) / n > _COMP_MAX_ALA_GLY_FRAC:
        return True

    # 3. normalized Shannon entropy (base-2) against a 20-letter max
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    if h / math.log2(20) < _COMP_MIN_NORM_ENTROPY:
        return True

    # 4. homopolymer run >= 4
    run = 1
    for i in range(1, n):
        run = run + 1 if s[i] == s[i - 1] else 1
        if run > _COMP_MAX_HOMOPOLYMER_RUN:
            return True

    return False


# ── Restraint emission with the RUN's chains (TASK 2 + TASK 3) ─────────────────

def build_alpha_restraint(case, out_dir, binder_chain: str = _RUN_BINDER_CHAIN,
                          target_chain: str = _RUN_TARGET_CHAIN):
    """Emit the α pin-polarity .restraints file with the RUN's chain letters (#27, TASK 2/3).

    The case's nominal restraint params encode the GT-fasta chain order (binder='A',
    target='B'), which is INVERTED vs this run (the loop wrapper builds entities=[target,
    binder] -> Chai labels TARGET=A, BINDER=B). We therefore override the chain letters to the
    run's convention (binder->'B', target->'A') while preserving the case's anchor resnums,
    max_distance and confidence. Writes ``out_dir/alpha.restraints`` and returns its Path.

    POCKET pin (#27 crash fix): the pin is a POCKET (binder = chain-level side, FIXED target
    anchor = token side), NOT a contact. A contact would assert BOTH endpoint identities match
    the structure, but the binder anchor is DESIGNED (unknown identity) so that crashes at the
    first predict. The pocket only asserts the FIXED target anchor token's identity, which IS
    known: we read it here from the target FASTA (the target is fixed context, resnum is known)
    and pass it as ``target_anchor_one_letter`` so the emitted code matches the real residue.
    """
    from pathlib import Path

    from xenodesign.benchmark.restraints import pin_polarity_rows, write_restraints
    from xenodesign.seed import read_target_sequence

    spec = case.restraint
    if spec is None or spec.kind != "pin_polarity":
        raise ValueError(
            f"build_alpha_restraint expects a pin_polarity restraint; got {spec!r}")
    p = dict(spec.params)

    # The pocket token side is the FIXED target anchor — read its REAL one-letter code from the
    # target FASTA (resnum is 1-based; the target is read at runtime, never inlined). The chai
    # pocket generator still asserts this token identity matches the structure, so it MUST be
    # the genuine residue (achiral Gly at the GT α target -> L/D 3-letter codes coincide).
    target_anchor_resnum = p["target_anchor_resnum"]
    target_seq = read_target_sequence(case.fasta_path, name=_TARGET_RECORD)
    if not (1 <= target_anchor_resnum <= len(target_seq)):
        raise ValueError(
            f"target_anchor_resnum {target_anchor_resnum} out of range for target "
            f"length {len(target_seq)} ({_TARGET_RECORD})")
    target_anchor_one_letter = target_seq[target_anchor_resnum - 1]

    run_params = {
        "binder_chain": binder_chain,
        "binder_anchor_resnum": p["binder_anchor_resnum"],
        "target_chain": target_chain,
        "target_anchor_resnum": target_anchor_resnum,
        "target_anchor_one_letter": target_anchor_one_letter,
        "max_distance": p["max_distance"],
        "confidence": p["confidence"],
    }
    rows = pin_polarity_rows(run_params)
    return write_restraints(Path(out_dir) / "alpha.restraints", rows)


# ── Seed construction ──────────────────────────────────────────────────────────

def _ensure_cterm_glycine(one_letter: str) -> str:
    """Chai needs ≥1 canonical residue per chain to tokenize an all-D peptide. Put that anchor
    at the **C-TERMINUS** (the helix terminus), NOT the core. Forces the last residue to 'G' when
    the seq is glycine-free; if a Gly already exists anywhere the seq is unchanged.

    Corrects the earlier `_ensure_glycine`, which slammed a Gly into the helix MIDPOINT (a
    helix-breaker, and not a designed residue) — see ADR-011 / the FASTA audit. The C-terminal Gly
    is held NON-DESIGNABLE by ``_cterm_gly_anchor`` during the loop so the core stays fully designed."""
    if "G" in one_letter:
        return one_letter
    return one_letter[:-1] + "G"


def _cterm_gly_anchor(backend_fn):
    """Wrap an InverseFoldingBackend so the binder's C-TERMINAL position is a FIXED Gly anchor:
    non-designable by LigandMPNN (``fixed_mask[-1] = True``) and forced to 'G' in every candidate.
    Keeps the helix core fully designed (no central-Gly artifact). 6-positional-arg signature so it
    is itself an InverseFoldingBackend (wrappable by MultiCandidate)."""
    def _wrapped(design_backbone, context_coords, context_elements,
                 fixed_mask, temperature, num_seqs):
        fm = list(fixed_mask)
        if fm:
            fm[-1] = True   # C-terminal position is non-designable
        cands = backend_fn(design_backbone, context_coords, context_elements,
                           fm, temperature, num_seqs)
        return [(c[:-1] + "G") if c else c for c in cands]   # force the C-terminal Gly anchor
    return _wrapped


def build_alpha_seed(case, target_seq: str, use_pepmlm: bool = True,
                     seed_seq: str | None = None, reverse: bool = True,
                     pepmlm_seed: int | None = None, pepmlm_temperature: float = 1.0) -> str:
    """Return the 21-res L seed sequence for the α binder.

    Default: a real target-aware PepMLM seed (network — weights download on first use).
    `seed_seq` overrides with an explicit sequence (e.g. for an offline/repeat run);
    `use_pepmlm=False` without a seed_seq falls back to a uniform-random seed.

    pepmlm_seed (correction): PepMLM's `generate()` is argmax (deterministic) by default, so every
    campaign run got the IDENTICAL seed. Passing a per-run `pepmlm_seed` switches PepMLM to
    temperature sampling so each run starts from a genuinely DIFFERENT target-conditioned seed."""
    from xenodesign.benchmark.seeding import build_seed_for_case

    if seed_seq is not None:
        if len(seed_seq) != case.binder_length:
            raise ValueError(
                f"seed_seq length {len(seed_seq)} != case.binder_length {case.binder_length}")
        return _ensure_cterm_glycine(seed_seq.upper())

    from xenodesign.seed import PepMLMSeedGenerator
    if use_pepmlm:
        generator = PepMLMSeedGenerator(reverse=reverse, seed=pepmlm_seed,
                                        temperature=pepmlm_temperature)  # real model
    else:
        # Offline fallback: a deterministic generator injected into the SAME
        # pepmlm_conditioned dispatch (alpha policy calls generate(target_seq=, length=);
        # RandomSeedGenerator's generate(length) signature would not fit that path).
        import random as _random
        from xenodesign.seed import AA_ALPHABET
        _rng = _random.Random(0)

        def _fake(target_seq: str, length: int) -> str:   # ignores target (no model)
            return "".join(_rng.choice(AA_ALPHABET) for _ in range(length))

        generator = PepMLMSeedGenerator(generate_fn=_fake, reverse=reverse)

    result = build_seed_for_case(case, generator=generator, target_seq=target_seq)
    return _ensure_cterm_glycine(result.one_letter.upper())


# ── Inverse-folding backend selector (STAGE 2: --backend {ligandmpnn|carbonara|mixed}) ──

class _MixedBackend:
    """Per-call round-robin between the LigandMPNN and CARBonAra bases (an InverseFoldingBackend).

    'mixed' interleaves the two inverse-folding models PER LOOP ITERATION, NOT per candidate:
    MultiCandidate makes exactly ONE backend call per iteration (oversampling `num_seqs` candidates
    in that single call), so alternating here on each ``__call__`` gives one model per iter
    (ligandmpnn, carbonara, ligandmpnn, ...). ``backend_log`` records which base produced each call
    so the run can report the per-iter backend assignment.

    The bases are looked up from THIS module at call time
    (``_self()._ligandmpnn_design_fn`` / ``_self().carbonara_design_fn``) so monkeypatching either
    name on that module is honoured (the validated behaviour + legacy test contract). 6-positional-
    arg call signature -> itself an InverseFoldingBackend, wrappable by _cterm_gly_anchor /
    MultiCandidate.
    """

    _NAMES = ("ligandmpnn", "carbonara")

    def __init__(self):
        self._i = 0
        self.backend_log: list[str] = []

    def __call__(self, design_backbone, context_coords, context_elements,
                 fixed_mask, temperature, num_seqs):
        name = self._NAMES[self._i % len(self._NAMES)]
        self._i += 1
        self.backend_log.append(name)
        shim = _self()
        base = shim._ligandmpnn_design_fn if name == "ligandmpnn" else shim.carbonara_design_fn
        return base(design_backbone, context_coords, context_elements,
                    fixed_mask, temperature, num_seqs)


def _make_base_backend(backend: str = "ligandmpnn"):
    """Select the base InverseFoldingBackend for the loop's sequence-update design_fn.

    'ligandmpnn' (default, ALL current behaviour preserved) and 'carbonara' return the pure
    adapters; 'mixed' returns a fresh _MixedBackend that round-robins the two PER CALL (per-iter
    interleave). Raises ValueError for an unknown mode (fail-fast). The selected backend's heavy
    deps are only touched when it is actually CALLED, so selection itself stays CPU-clean.

    The pure bases are looked up from THIS module at call time so a test
    monkeypatch of ``_self()._ligandmpnn_design_fn`` / ``_self().carbonara_design_fn`` on that module is honoured.
    """
    if backend == "mixed":
        return _MixedBackend()
    shim = _self()
    try:
        return {"ligandmpnn": shim._ligandmpnn_design_fn,
                "carbonara": shim.carbonara_design_fn}[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; expected one of "
            "'ligandmpnn', 'carbonara', 'mixed'") from None


# ── Drift-fixed sequence-update closure (spec §5 oversample lever) ──────────────

def make_alpha_seq_update_fn(wrapper: _LoopBackendWrapper, num_seqs: int = _DEFAULT_NUM_SEQS,
                             backend: str = "ligandmpnn", roles=None,
                             frozen_positions=None):
    """Build the loop's sequence_update_fn(prediction) -> one-letter L seq, wiring the
    P2 drift fix: a SequenceUpdater whose design_fn is a MultiCandidate over the selected
    context-aware inverse-folding base (oversample `num_seqs`, spec §5).

    `backend` (STAGE 2): which inverse-folding base to wrap — 'ligandmpnn' (default; ALL
    prior behaviour preserved byte-for-byte), 'carbonara', or 'mixed' (per-iter round-robin
    of the two). The base is C-term-Gly-anchored then oversampled by MultiCandidate exactly as
    before; only the base callable changes, so the double-flip and D-CCD re-encode are untouched.

    P1a (real key_fn): MultiCandidate now keeps the best of `num_seqs` oversampled
    candidates by ``scorer.sequence_quality_key`` — a pure-sequence de-gaming
    re-rank that favours diverse, non-degenerate sequences over the poly-Ala / homopolymer
    basin (ADR-007). It is CHEAP: a function of the sequence string only, so it multiplies
    just the inverse-folding forward passes, never the Chai predicts. An orthogonal per-candidate
    Chai re-score (chirality-clean then ipTM) remains the expensive P7 variant; this
    sequence-level key is the high-value/low-cost lever the feature-map ranked first.

    Monkeypatch contract: ``_self()._make_base_backend`` / ``_self()._cterm_gly_anchor`` / ``_self()._best_cif_path`` are
    resolved through THIS module at call time so the legacy CPU tests that
    patch them here are honoured (validated behaviour preserved).
    """
    from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif
    from xenodesign.inverse_folding import MultiCandidate
    from xenodesign.scorer import sequence_quality_key
    from xenodesign.sequence_update import SequenceUpdater, make_sequence_update_fn

    shim = _self()
    # C-terminal Gly anchor (non-designable) wraps the SELECTED base backend so the helix CORE is
    # never overwritten by the tokenization Gly (correction; ADR-011 / FASTA audit). backend
    # defaults to 'ligandmpnn' => identical to the prior MultiCandidate(_cterm_gly_anchor(...)).
    base = shim._make_base_backend(backend)
    design_fn = MultiCandidate(shim._cterm_gly_anchor(base), num_seqs=num_seqs,
                               key_fn=sequence_quality_key)
    # frozen_positions (0-based): declared coordinator positions forced fixed in the MPNN
    # mask so pinned donors never drift (cyclic-metal). alpha path passes None → no change.
    updater = SequenceUpdater(design_fn=design_fn, frozen_positions=frozen_positions)

    # CHAIN CONTRACT: the binder + context chain letters come from the ChainRoles value the
    # dispatcher computed ONCE from the assembled entity list — NOT a hardcoded letter. roles=None
    # preserves the legacy alpha default (binder 'B', context 'A') byte-for-byte, so the standalone
    # alpha drivers and the alpha oracle are unaffected. With roles threaded, non_alpha (binder 'C'),
    # cyclic metal (binder 'B') and the no-target case (binder/context BOTH 'A') all read the RIGHT
    # chain — the chain-misidentification bug is structurally impossible.
    binder_chain = roles.binder if roles is not None else "B"
    context_chain = roles.context if roles is not None else "A"

    def _extract(prediction) -> dict:
        out_dir = wrapper.last_out_dir
        if out_dir is None:
            raise RuntimeError("seq_update called before any structure step")
        cif = _self()._best_cif_path(out_dir)
        binder_res = (backbone_by_residue_from_cif(cif, binder_chain)
                      or backbone_by_residue_from_cif(cif, binder_chain.lower()))
        if not binder_res:
            raise RuntimeError(f"cannot extract binder chain {binder_chain!r} from {cif}")
        design_backbone = _backbone_array_from_residues(binder_res)
        ctx_coords, ctx_elements = _self()._all_atoms_from_chain(cif, context_chain)
        if ctx_coords.shape[0] == 0:
            ctx_coords, ctx_elements = _self()._all_atoms_from_chain(cif, context_chain.lower())
        return {
            "design_backbone": design_backbone,
            "design_codes": ["DAL"] * design_backbone.shape[0],
            "context_coords": ctx_coords,
            "context_elements": ctx_elements,
        }

    base_fn = make_sequence_update_fn(updater, _extract, emit="one_letter")

    def _guarded(prediction) -> str:
        return _ensure_cterm_glycine(base_fn(prediction))

    return _guarded


# ── Mixed parity-aware objective from a per-candidate score_complex panel ──

def mixed_objective_from_cif(cif_path, chai_dir=None,
                             chain_a: str = _RUN_TARGET_CHAIN,
                             chain_b: str = _RUN_BINDER_CHAIN):
    """Build a score_complex panel for ONE candidate CIF and score it with mixed_objective.score().

    This is the directive-#6 selection objective: instead of ipTM-only, route selection through the
    parity-aware composite (bsa/contacts/pack/sc/ipsae/iptm/ipae/hbond) calibrated on the real
    heterochiral D-binder interfaces (T20 / ADR-014). The panel is CHEAP on CPU: a single
    score_complex.structural() pass over the candidate's binder/target chains (freesasa + numpy),
    plus — when the chai_out dir is given — the L-trained confidence terms (iptm/ipae/ipsae).

    Chain convention (α RUN): the loop wrapper builds entities=[target, binder] -> Chai labels
    TARGET=chain A, BINDER=chain B. structural() is chain-order-symmetric for bsa/contacts/sc/hbond;
    confidence()'s ``na`` is the number of FIRST-block (target, chain A) residues, matching chai's
    [target, binder] token order. We therefore pass chain_a=A (target), chain_b=B (binder).

    Args:
        cif_path: path to the candidate's scored CIF.
        chai_dir: the candidate's chai_out dir (holds scores.model_idx_*.npz + confidence npz). When
            None, only the parity-invariant geometry terms are scored (iptm/ipae/ipsae imputed by
            mixed_objective.normalize to their 0-defaults — the geometry+sc+hbond terms still drive).
        chain_a / chain_b: CIF chain labels for the target / binder (defaults = the α RUN's A / B).

    Returns:
        ``(composite, panel)`` — the mixed_objective composite float in ~[0,1] and the raw
        score_complex panel dict (merged structural + confidence) that produced it.
    """
    from scripts import mixed_objective, score_complex

    panel = score_complex.structural(str(cif_path), chain_a, chain_b)
    if chai_dir is not None:
        try:
            na = panel.get("n_res_a") or 0
            panel.update(score_complex.confidence(str(chai_dir), na))
        except Exception:
            pass   # confidence terms missing -> mixed_objective imputes their 0-defaults
    composite, _parts = mixed_objective.score(panel)
    return composite, panel


def ipsae_objective_from_cif(cif_path, chai_dir=None,
                             chain_a: str = _RUN_TARGET_CHAIN,
                             chain_b: str = _RUN_BINDER_CHAIN):
    """Return ONE candidate's ipSAE (the parity-aware confidence axis) and the panel it came from.

    The ipSAE objective: rank candidates by ipSAE alone — the
    confidence axis that was the least-bad for the rigid 8GQP heterochiral case — rather than the
    ipTM+pLDDT design_score or the mixed composite. Reuses ``mixed_objective_from_cif``'s machinery
    verbatim: it already builds a score_complex panel containing 'ipsae' (read by
    score_complex.confidence from the chai confidence npz). We simply pull the panel's 'ipsae'.

    Returns ``(ipsae, panel)`` — the float ipSAE in ~[0,1] and the raw merged panel dict. When the
    confidence terms could not be read (no chai_dir, or the npz was missing/unreadable so
    score_complex.confidence emitted no 'ipsae'), ipSAE is absent → returns 0.0 (the worst-case
    ranking value, consistent with mixed_objective.normalize's 0-default for the term)."""
    _composite, panel = mixed_objective_from_cif(cif_path, chai_dir=chai_dir,
                                                 chain_a=chain_a, chain_b=chain_b)
    return float(panel.get("ipsae", 0.0)), panel


# ── Panel referee (chirality from the per-iter CIF) ─────────────────────────────

def _make_referee_fn(loop_dir, esm_judge=None, roles=None):
    """RefereeScore for a LoopStep, reading chirality AND the scored sequence from that iter's CIF.

    The loop appends one LoopStep per iteration in order, so the step's position in
    `history` is its iter index; the CIF lives at loop_dir/iter_{idx:03d}.

    #31 off-by-one fix: the binder sequence is read from the SCORED CIF (binder_seq_from_cif),
    i.e. the sequence that actually produced this iter's metrics — NOT step.state.d_fasta (which
    is the NEXT sequence designed that iter). The ESM-PLL (TASK 4) and composition floor (TASK 5)
    both score that scored-CIF sequence.

    Composition floor (TASK 5): a low-complexity candidate (poly-Ala / low-entropy) is vetoed via
    the panel's INDEPENDENT composition-veto channel (RefereeScore.composition_violation=True),
    NOT by overwriting chirality_violation. The chirality field therefore ALWAYS carries the real
    measured chirality (#37 de-conflation: reusing the chirality channel for composition rejects
    corrupted the reported chirality distribution; the panel now vetoes composition separately).
    """
    from pathlib import Path

    from xenodesign.judges.panel import RefereeScore

    loop_dir = Path(loop_dir)
    # Binder chain from the ONE chain contract (roles=None -> alpha default 'B', byte-identical);
    # threaded so the non_alpha 2-chain referee reads chain 'C', not HA2 ('B'). The reads are
    # already try/except graceful, but reading the RIGHT chain makes the helix/seq metrics correct
    # instead of silently None for a non-'B' binder.
    binder_chain = roles.binder if roles is not None else _RUN_BINDER_CHAIN

    def _score(step, idx):
        iter_dir = loop_dir / f"iter_{idx:03d}"
        cif = None
        try:
            cif = _self()._best_cif_path(iter_dir)
            chir = _chirality_violation_frac_from_cif(cif)
        except Exception:
            chir = 1.0   # unverifiable → treat as a chirality violation, never false-clean
        ti = np.asarray(step.prediction.token_index)
        mask = ti == 1
        iface_plddt = (float(step.prediction.plddt[mask].mean()) if mask.any()
                       else float(step.prediction.plddt.mean()))

        # P1c: forward CA-geometry helix fraction of the binder chain → the panel SS-bias term
        # (α rewards helix). Robust: any parse failure → None (imputed to the population mean).
        helix = None
        if cif is not None:
            try:
                helix = _binder_helix_fraction(cif, chain=binder_chain)
            except Exception:
                helix = None

        # The sequence that PRODUCED these metrics: the binder chain of the scored CIF (#31).
        scored_l_seq = None
        if cif is not None:
            try:
                scored_l_seq = _self().binder_seq_from_cif(cif, binder_chain)
            except Exception:
                scored_l_seq = None

        # Composition floor (TASK 5): veto low-complexity scored sequences on the OWN channel —
        # the chirality value above is left untouched (#37). Unreadable seq → cannot vet → False.
        comp_viol = (scored_l_seq is not None and composition_violation(scored_l_seq))

        pll = None
        if esm_judge is not None and scored_l_seq is not None:
            try:
                pll = esm_judge(scored_l_seq)
            except Exception:
                pll = None

        return RefereeScore(
            chirality_violation=chir, iptm=step.prediction.iptm,
            interface_plddt=iface_plddt, pll=pll, mirror_discrepancy=None,
            composition_violation=comp_viol, helix_fraction=helix,
        )

    return _score


def _binder_helix_fraction(cif, chain: str = _RUN_BINDER_CHAIN):  # pragma: no cover (gpu/CIF)
    """Forward CA-geometry helix fraction of the binder chain in a scored CIF (P1c).

    Reads the binder chain's per-residue backbone, takes the CA column, and returns
    secondary_structure.helix_fraction over it (in [0, 1]). Used to populate
    RefereeScore.helix_fraction so the panel's per-case SS-bias term can reward the α design
    toward a helical conformation. Returns None when the chain cannot be read."""
    from xenodesign.eval.gate_tier0a import backbone_by_residue_from_cif
    from xenodesign.secondary_structure import helix_fraction

    res = (backbone_by_residue_from_cif(cif, chain)
           or backbone_by_residue_from_cif(cif, chain.lower()))
    if not res:
        return None
    ca = np.asarray(_backbone_array_from_residues(res))[:, 1, :]   # (n_res, 3) CA
    return float(helix_fraction(ca))


# ── Loop objectives ──────────────────────────────────────────────────────────────

def _loop_score_fn(prediction) -> float:
    """Loop objective (DEFAULT, --objective iptm): ipTM + binder-chain pLDDT composite (chirality
    scored separately by the panel / accept gate). Reproducible / byte-for-byte unchanged."""
    from xenodesign.scorer import design_score
    ti = np.asarray(prediction.token_index)
    mask = ti == 1
    iface_plddt = (float(prediction.plddt[mask].mean()) if mask.any()
                   else float(prediction.plddt.mean()))
    return design_score(iptm=prediction.iptm, interface_plddt=iface_plddt,
                        chirality_violation_frac=0.0)


def make_mixed_loop_score_fn(wrapper):
    """Build the per-iteration loop score_fn for --objective mixed.

    The HalluLoop calls ``score_fn(prediction)`` right after the structure step, so the candidate's
    scored CIF lives at ``wrapper.last_out_dir`` (the per-iter chai_out). This closure builds the
    mixed parity-aware panel from that CIF (``mixed_objective_from_cif``) and returns its composite
    in place of the ipTM-only ``design_score``. GRACEFUL FALLBACK: any failure to build/score the
    panel (missing CIF, freesasa error, etc.) falls back to ``_loop_score_fn`` so a single bad
    iteration never crashes the loop and the run degrades to the reproducible ipTM objective.
    """
    def _score(prediction) -> float:
        out_dir = getattr(wrapper, "last_out_dir", None)
        if out_dir is None:
            return _loop_score_fn(prediction)
        try:
            cif = _self()._best_cif_path(out_dir)
            chai_dir = cif.parent
            composite, _panel = mixed_objective_from_cif(cif, chai_dir=chai_dir)
            return float(composite)
        except Exception:
            return _loop_score_fn(prediction)   # graceful fallback to the reproducible ipTM objective

    return _score


def make_ipsae_loop_score_fn(wrapper):
    """Build the per-iteration loop score_fn for --objective ipsae.

    Identical plumbing to ``make_mixed_loop_score_fn`` (the candidate's scored CIF is at
    ``wrapper.last_out_dir`` when score_fn is called), but the per-iter score is the candidate's
    raw ipSAE (``ipsae_objective_from_cif``) instead of the mixed composite. GRACEFUL FALLBACK:
    any failure to build/read the panel falls back to the reproducible ipTM ``_loop_score_fn`` so a
    single bad iteration never crashes the loop.
    """
    def _score(prediction) -> float:
        out_dir = getattr(wrapper, "last_out_dir", None)
        if out_dir is None:
            return _loop_score_fn(prediction)
        try:
            cif = _self()._best_cif_path(out_dir)
            ipsae, _panel = ipsae_objective_from_cif(cif, chai_dir=cif.parent)
            return float(ipsae)
        except Exception:
            return _loop_score_fn(prediction)   # graceful fallback to the reproducible ipTM objective

    return _score


# ── Trajectory + panel-select + metrics assembly (extracted from run_alpha_design) ──

def _assemble_alpha_result(history, referee_scores, panel_result, case, *,
                           loop_dir, out_dir, l_seed_iptm, n_iters, num_seqs,
                           ref_time_steps, chirality_gate, objective, periodicity_gate,
                           heptad_thresh, restraints, constraint_path, use_pll, backend,
                           wall_time_s):
    """Assemble the α trajectory + panel selection + (optional mixed/ipsae re-rank) + metrics +
    result dict, and write ``out_dir/alpha_result.json`` — behaviour-preserving extraction of the
    tail of :func:`run_alpha_design` (same trajectory shape, same selection logic, same JSON).

    Returns the result dict.
    """
    import json
    from pathlib import Path

    from xenodesign.benchmark.case_metrics import (
        beats_baseline, beats_baseline_full, case_metrics,
    )
    from xenodesign.io_spec import d_fasta_to_one_letter, to_d_fasta

    loop_dir = Path(loop_dir)
    out_dir = Path(out_dir)

    # #31 off-by-one fix: report the binder sequence READ FROM the scored CIF (the sequence
    # that actually produced this iter's metrics), NOT step.state.d_fasta (the NEXT designed
    # sequence). Falls back to the (mismatched) state only if the CIF cannot be read.
    trajectory = []
    for i, (step, ref) in enumerate(zip(history, referee_scores)):
        try:
            l_seq = _self().binder_seq_from_cif(_self()._best_cif_path(loop_dir / f"iter_{i:03d}"),
                                        _RUN_BINDER_CHAIN)
            d_fasta = to_d_fasta(l_seq)
        except Exception:
            l_seq = d_fasta_to_one_letter(step.state.d_fasta)
            d_fasta = step.state.d_fasta
        trajectory.append({
            "iter": i, "d_fasta": d_fasta,
            "l_seq": l_seq,
            "iptm": float(step.prediction.iptm), "chirality": float(ref.chirality_violation),
            "score": float(step.score), "composite": float(panel_result.composite_scores[i]),
            "vetoed": bool(panel_result.vetoed[i]),
        })

    sel_idx = panel_result.selected_idx
    # --objective mixed / ipsae: re-select among NON-VETOED iters by the parity-aware objective.
    mixed_selection = None
    if objective in ("mixed", "ipsae"):
        objective_scores = {}
        for i in range(len(history)):
            if panel_result.vetoed[i]:
                continue
            try:
                cif = _self()._best_cif_path(loop_dir / f"iter_{i:03d}")
                if objective == "ipsae":
                    val, _panel = ipsae_objective_from_cif(cif, chai_dir=cif.parent)
                else:
                    val, _panel = mixed_objective_from_cif(cif, chai_dir=cif.parent)
                objective_scores[i] = float(val)
            except Exception:
                continue
        if objective_scores:
            sel_idx = max(objective_scores, key=objective_scores.get)
            mixed_selection = {"selected_idx": sel_idx, "scores": objective_scores,
                               "objective": objective}
    # argmax by score WITHOUT history.index (LoopStep is an unfrozen dataclass; equal steps
    # would make .index() return the first match, mislabelling the naive-best marker).
    naive_idx = max(range(len(history)), key=lambda i: history[i].score)

    print(f"\n{'iter':>4} {'l_seq':>23} {'ipTM':>6} {'chir':>6} {'compos':>7} "
          f"{'veto':>4} {'sel':>5}")
    for t in trajectory:
        mark = "PANEL" if t["iter"] == sel_idx else ("best" if t["iter"] == naive_idx else "")
        print(f"{t['iter']+1:>4} {t['l_seq']:>23} {t['iptm']:6.4f} {t['chirality']:6.3f} "
              f"{t['composite']:7.4f} {'Y' if t['vetoed'] else '.':>4} {mark:>5}")

    # ── Interface metrics of the selected design vs baseline ─────────────
    print(f"\n[4/4] Scoring selected design (iter {sel_idx+1}) vs baseline ...")
    sel_chai_out = loop_dir / f"iter_{sel_idx:03d}" / "chai_out"
    sel = trajectory[sel_idx]
    metrics_result, beats, beats_full = None, False, False
    try:
        metrics_result = case_metrics(case, sel_chai_out)
        beats = beats_baseline(metrics_result)
        beats_full = beats_baseline_full(metrics_result, sel["chirality"])
    except Exception as exc:   # pragma: no cover — GPU-path robustness
        print(f"    [metrics] WARNING: case_metrics failed on {sel_chai_out}: {exc}")

    result = {
        "case_id": "alpha",
        "selected_iter": sel_idx,
        "selected_d_fasta": sel["d_fasta"],
        "selected_l_seq": sel["l_seq"],
        "selected_iptm": sel["iptm"],
        "selected_chirality": sel["chirality"],
        "selected_composite": sel["composite"],
        "panel_fallback_used": bool(panel_result.fallback_used),
        "metrics": metrics_result,
        "beats_baseline": bool(beats),
        "beats_baseline_full": bool(beats_full),
        "baseline": {"interface_iptm": case.baseline.interface_iptm,
                     "ipae": case.baseline.ipae, "chirality": case.baseline.chirality},
        "l_seed_iptm": float(l_seed_iptm),
        "trajectory": trajectory,
        "n_iters": n_iters, "num_seqs": num_seqs, "ref_time_steps": ref_time_steps,
        "chirality_gate": chirality_gate, "wall_time_s": wall_time_s,
        "objective": objective,
        "periodicity_gate": bool(periodicity_gate),
        "heptad_thresh": float(heptad_thresh),
        "mixed_selection": mixed_selection,
        "restraints": bool(restraints),
        "constraint_path": str(constraint_path) if constraint_path is not None else None,
        "use_pll": bool(use_pll),
        "backend": backend,
        "out_dir": str(out_dir),
    }

    (out_dir / "alpha_result.json").write_text(
        json.dumps(result, indent=2, default=lambda o: getattr(o, "tolist", lambda: str(o))()))

    print(f"\n{'='*78}")
    print(f"SELECTED (iter {sel_idx+1}): ipTM {sel['iptm']:.4f}  chir {sel['chirality']:.3f}")
    if metrics_result is not None:
        m = metrics_result.get("metrics", {})
        vb = metrics_result.get("vs_baseline", {})
        print(f"  interface ipTM {m.get('interface_iptm')}  ipAE {m.get('ipae_mean')}  "
              f"ipSAE {m.get('ipsae')}")
        print(f"  vs baseline: iptm_delta {vb.get('iptm_delta')}  ipae_delta {vb.get('ipae_delta')}")
    print(f"  BEATS BASELINE: {beats_full}   "
          f"(bar: ipTM>baseline by >0.02 AND ipAE<10 AND chirality<=0.10)")
    print(f"    breakdown: ipTM-margin {beats} | full-3-criterion {beats_full}")
    print(f"  wall {wall_time_s/60:.1f} min | result -> {out_dir/'alpha_result.json'}\n{'='*78}")
    return result

