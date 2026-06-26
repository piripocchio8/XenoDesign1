"""CPU-only tests for scripts/design_alpha.py pure helpers.

Covers:
  - _ensure_glycine: correctness, idempotency, length preservation
  - build_alpha_seed (use_pepmlm=False): offline fake generator, returned seed properties
  - build_alpha_seed seed_seq override: valid passthrough + wrong-length ValueError
  - binder_seq_from_cif: CIF -> binder-chain one-letter seq (TASK 1 off-by-one fix)
  - build_alpha_restraint: emits the RUN's chains (binder=B, target=A) (TASK 2)
  - composition_violation: anti poly-Ala low-complexity floor (TASK 5)
  - Module importability and default constants
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ── Module-level import (importability check) ──────────────────────────────────

import scripts.design_alpha as _da

from scripts.design_alpha import (
    _DEFAULT_N_ITERS,
    _DEFAULT_NUM_SEQS,
    _TARGET_RECORD,
    _cterm_gly_anchor,
    _ensure_cterm_glycine,
    binder_seq_from_cif,
    build_alpha_restraint,
    build_alpha_seed,
    composition_violation,
)
from xenodesign.benchmark.cases import get_case

# A synthetic target sequence that avoids any gitignored FASTA read.
_SYNTHETIC_TARGET = "GSHMKVLITGGAGFIGSHLVDRLMERGHEVVVLDNL"

_AA_SET = set("ARNDCQEGHILKMFPSTWYV")


# ── C-terminal Gly anchor (correction: was a helix-core midpoint Gly) ────────────

def test_ensure_cterm_glycine_places_at_cterminus_not_core():
    seq = "ACDEFHIKLMNPQRSTVWY"   # 19 chars, no G
    result = _ensure_cterm_glycine(seq)
    assert len(result) == len(seq)           # length preserved
    assert result[-1] == "G"                 # anchor at the C-TERMINUS
    assert "G" not in result[:-1]            # NOT in the helix core


def test_ensure_cterm_glycine_noop_when_G_present():
    seq = "ACGDEFHIKLM"                       # already has a G
    assert _ensure_cterm_glycine(seq) == seq


def test_ensure_cterm_glycine_idempotent():
    seq = "ACDEFHIKLMNPQRSTVWY"               # no G
    once = _ensure_cterm_glycine(seq)
    assert _ensure_cterm_glycine(once) == once


def test_cterm_gly_anchor_fixes_last_position_nondesignable():
    """The anchor wrapper marks the C-terminal position non-designable in the fixed_mask AND
    forces every candidate's last residue to 'G' (the core is left untouched)."""
    captured = {}

    def fake_backend(db, cc, ce, fixed_mask, temp, n):
        captured["fixed_mask"] = list(fixed_mask)
        return ["KKKKK" for _ in range(n)]    # 5-mer, no G, core = KKKK

    wrapped = _cterm_gly_anchor(fake_backend)
    out = wrapped(None, [], [], [False] * 5, 0.1, 3)
    assert captured["fixed_mask"][-1] is True         # C-terminal non-designable
    assert captured["fixed_mask"][:-1] == [False] * 4  # core designability untouched
    assert all(s[-1] == "G" for s in out)             # forced C-terminal Gly
    assert all(s[:-1] == "KKKK" for s in out)         # core sequence preserved


def test_cterm_glycine_replaces_old_midpoint_behaviour():
    """Correction: the anchor Gly is now at the C-terminus, NEVER the helix midpoint."""
    seq = "AAAAAAAAAA"  # 10 chars, no G
    result = _ensure_cterm_glycine(seq)
    assert result == "AAAAAAAAAG"            # last residue only
    assert result[len(seq) // 2] != "G"      # the old midpoint slot is NOT a Gly


# ── build_alpha_seed (offline, use_pepmlm=False) ──────────────────────────────

def test_build_alpha_seed_offline_len():
    """Offline seed must have exactly binder_length (21) characters."""
    case = get_case("alpha")
    seed = build_alpha_seed(case, _SYNTHETIC_TARGET, use_pepmlm=False)
    assert len(seed) == case.binder_length


def test_build_alpha_seed_offline_contains_G():
    """Offline seed must contain at least one 'G' (glycine guard)."""
    case = get_case("alpha")
    seed = build_alpha_seed(case, _SYNTHETIC_TARGET, use_pepmlm=False)
    assert "G" in seed


def test_build_alpha_seed_offline_uppercase():
    """Seed must be returned as all-uppercase."""
    case = get_case("alpha")
    seed = build_alpha_seed(case, _SYNTHETIC_TARGET, use_pepmlm=False)
    assert seed == seed.upper()


def test_build_alpha_seed_offline_chars_in_aa_alphabet():
    """All characters in the seed must belong to the standard 20-AA alphabet."""
    case = get_case("alpha")
    seed = build_alpha_seed(case, _SYNTHETIC_TARGET, use_pepmlm=False)
    assert set(seed) <= _AA_SET


# ── build_alpha_seed seed_seq override ───────────────────────────────────────

def test_build_alpha_seed_seed_seq_override_passthrough():
    """An explicit 21-char seed_seq (already containing G) is returned unchanged
    (modulo upper-casing and glycine-ensure which is a no-op here)."""
    case = get_case("alpha")
    explicit = "ACDEFGHIKLMNPQRSTVWYG"   # 21 chars, contains G
    result = build_alpha_seed(case, _SYNTHETIC_TARGET, use_pepmlm=False,
                              seed_seq=explicit)
    assert result == explicit.upper()
    assert len(result) == 21


def test_build_alpha_seed_seed_seq_ensures_glycine():
    """When an explicit seed_seq lacks 'G', the result still contains one (same length)."""
    case = get_case("alpha")
    no_g = "ACDEFHIKLMNPQRSTVWYAC"   # 21 chars, no G
    result = build_alpha_seed(case, _SYNTHETIC_TARGET, use_pepmlm=False,
                              seed_seq=no_g)
    assert "G" in result
    assert len(result) == 21


def test_build_alpha_seed_seed_seq_wrong_length_raises():
    """A seed_seq of wrong length must raise ValueError."""
    case = get_case("alpha")
    wrong = "ACDEFGHIK"   # only 9 chars
    with pytest.raises(ValueError, match="seed_seq length"):
        build_alpha_seed(case, _SYNTHETIC_TARGET, use_pepmlm=False, seed_seq=wrong)


def test_build_alpha_seed_seed_seq_too_long_raises():
    """A seed_seq that is too long must also raise ValueError."""
    case = get_case("alpha")
    too_long = "A" * 30
    with pytest.raises(ValueError, match="seed_seq length"):
        build_alpha_seed(case, _SYNTHETIC_TARGET, use_pepmlm=False, seed_seq=too_long)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_default_n_iters():
    assert _DEFAULT_N_ITERS == 30


def test_default_num_seqs():
    assert _DEFAULT_NUM_SEQS == 8


def test_target_record():
    assert _TARGET_RECORD == "trimer_DL_ABLE_B"


def test_module_importable():
    """The module itself must be importable without GPU or network."""
    assert _da is not None


# ── binder_seq_from_cif (TASK 1 off-by-one fix) ───────────────────────────────

def _write_tiny_cif(tmp_path) -> Path:
    """Build a tiny 2-chain mmCIF via gemmi (the same writer chai uses), so it round-trips
    cleanly: chain A (target, canonical L ALA/LYS) + chain B (binder, D-CCD DAL/DGL + GLY).
    The helper reads only residue NAMES, so 1 CA atom per residue suffices."""
    import gemmi

    st = gemmi.Structure()
    st.name = "tiny"
    model = gemmi.Model("1")

    def _mkchain(name, residue_names):
        ch = gemmi.Chain(name)
        for i, rn in enumerate(residue_names, start=1):
            res = gemmi.Residue()
            res.name = rn
            res.seqid = gemmi.SeqId(i, " ")
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            atom.element = gemmi.Element("C")
            atom.b_iso = 50.0
            res.add_atom(atom)
            ch.add_residue(res)
        return ch

    model.add_chain(_mkchain("A", ["ALA", "LYS"]))             # target, L
    model.add_chain(_mkchain("B", ["DAL", "DGL", "GLY"]))      # binder, D + achiral Gly
    st.add_model(model)
    st.setup_entities()

    cif = tmp_path / "pred.cif"
    st.make_mmcif_document().write_file(str(cif))
    return cif


def test_binder_seq_from_cif_reads_chain_B(tmp_path):
    """Chain B (the binder) DAL/DGL/GLY must decode to one-letter L 'AEG'."""
    cif = _write_tiny_cif(tmp_path)
    # DAL -> ALA -> A ; DGL -> GLU -> E ; GLY -> G
    assert binder_seq_from_cif(cif, "B") == "AEG"


def test_binder_seq_from_cif_default_is_chain_B(tmp_path):
    """Default chain is the run's binder chain 'B'."""
    cif = _write_tiny_cif(tmp_path)
    assert binder_seq_from_cif(cif) == "AEG"


def test_binder_seq_from_cif_reads_target_chain_A(tmp_path):
    """Chain A (the target) is canonical L: ALA/LYS -> 'AK'."""
    cif = _write_tiny_cif(tmp_path)
    assert binder_seq_from_cif(cif, "A") == "AK"


def test_binder_seq_from_cif_missing_chain_raises(tmp_path):
    """A chain that does not exist raises a clear RuntimeError."""
    cif = _write_tiny_cif(tmp_path)
    with pytest.raises(RuntimeError, match="not found"):
        binder_seq_from_cif(cif, "Z")


# ── build_alpha_restraint (TASK 2 chain convention) ───────────────────────────

def test_build_alpha_restraint_uses_run_chains(tmp_path):
    """The emitted restraint MUST use the RUN's chains: binder=B, target=A — the INVERSE of
    the case's nominal (binder='A'/target='B', GT-fasta order).

    #27 crash fix: the pin is a POCKET, so the binder is the chain-level side (chainA, res_idxA
    EMPTY -> no identity assertion on the DESIGNED binder) and the FIXED target anchor is the
    token side (chainB) with its REAL one-letter code read from the FASTA. Skipped if the
    gitignored GT FASTA is absent (the real target code can't be read)."""
    from xenodesign.benchmark.restraints import parse_restraints

    case = get_case("alpha")
    if not Path(case.fasta_path).exists():
        pytest.skip(f"GT FASTA absent ({case.fasta_path}); cannot read target anchor code")
    path = build_alpha_restraint(case, tmp_path)
    rows = parse_restraints(path)
    assert len(rows) == 1
    row = rows[0]
    # pocket: binder = chain-level side (chainA), target = token side (chainB).
    assert row["chainA"] == "B", f"binder chain should be 'B', got {row['chainA']!r}"
    assert row["chainB"] == "A", f"target chain should be 'A', got {row['chainB']!r}"
    assert row["connection_type"] == "pocket"
    # binder (chainA) is chain-level -> res_idxA MUST be empty (no identity on the designed
    # binder). target (chainB) token res_idx is '<REAL one-letter><pos>' — NOT 'X' (the fixed
    # target identity is known and asserted by chai against the structure).
    assert row["res_idxA"] == "", f"binder res_idxA must be empty (chain-level), got {row['res_idxA']!r}"
    assert row["res_idxB"] != "" and not row["res_idxB"].startswith("X"), (
        f"target token must carry a REAL one-letter code, got {row['res_idxB']!r}")


def test_build_alpha_restraint_target_code_is_real_fasta_residue(tmp_path):
    """The target token's one-letter code MUST be the REAL target-FASTA residue at the anchor
    resnum (so chai's pocket-token identity assertion passes against the structure)."""
    from xenodesign.benchmark.restraints import parse_restraints
    from xenodesign.seed import read_target_sequence
    from scripts.design_alpha import _TARGET_RECORD

    case = get_case("alpha")
    if not Path(case.fasta_path).exists():
        pytest.skip(f"GT FASTA absent ({case.fasta_path}); cannot read target anchor code")
    p = case.restraint.params
    target_seq = read_target_sequence(case.fasta_path, name=_TARGET_RECORD)
    expected_code = target_seq[p["target_anchor_resnum"] - 1]

    path = build_alpha_restraint(case, tmp_path)
    row = parse_restraints(path)[0]
    assert row["res_idxB"] == f"{expected_code}{p['target_anchor_resnum']}", (
        f"target token must be the real FASTA residue '{expected_code}' at "
        f"resnum {p['target_anchor_resnum']}, got {row['res_idxB']!r}")


def test_build_alpha_restraint_preserves_case_params(tmp_path):
    """Anchor resnum, max_distance and confidence come from the case spec; pocket connection."""
    from xenodesign.benchmark.restraints import parse_restraints

    case = get_case("alpha")
    if not Path(case.fasta_path).exists():
        pytest.skip(f"GT FASTA absent ({case.fasta_path}); cannot read target anchor code")
    p = case.restraint.params
    path = build_alpha_restraint(case, tmp_path)
    row = parse_restraints(path)[0]
    # pocket token position is the preserved target anchor resnum. Distance column is the
    # chai-0.6.1 '*_angstrom' name.
    assert row["res_idxB"].endswith(str(p["target_anchor_resnum"]))
    assert float(row["max_distance_angstrom"]) == float(p["max_distance"])
    assert float(row["confidence"]) == float(p["confidence"])
    assert row["connection_type"] == "pocket"


def test_build_alpha_restraint_writes_file(tmp_path):
    """The restraint file is actually written to out_dir/alpha.restraints."""
    case = get_case("alpha")
    path = build_alpha_restraint(case, tmp_path)
    assert path.exists()
    assert path.name == "alpha.restraints"


# ── composition_violation (TASK 5 anti poly-Ala floor) ────────────────────────

def test_composition_violation_rejects_high_ala():
    """A 57%-Ala string is low-complexity → rejected."""
    # 12 A out of 21 = 0.571 > 0.30 single-AA and > 0.40 Ala+Gly.
    seq = "AAAAAAAAAAAA" + "CDEFHIKLM"   # 21 chars, 12 A
    assert composition_violation(seq) is True


def test_composition_violation_rejects_homopolymer_run():
    """An 'AAAA'-containing string is rejected by the homopolymer-run rule."""
    # Construct a sequence that ONLY trips the homopolymer rule (so the test is specific):
    # diverse residues but with an AAAA run of length 4.
    seq = "CDEFHIKLMNAAAAPQRSTVW"   # 21 chars, single-AA frac low, but AAAA run = 4
    assert composition_violation(seq) is True


def test_composition_violation_aaaa_run_specifically():
    """A run of exactly 4 identical residues trips the homopolymer veto."""
    assert composition_violation("CDEFAAAAHIKLMNPQRSTVW") is True


def test_composition_violation_accepts_diverse_sequence():
    """A diverse 20-AA sequence passes the floor."""
    assert composition_violation("ACDEFGHIKLMNPQRSTVWY") is False


def test_composition_violation_empty_is_rejected():
    """Empty input is treated as a violation (nothing to select)."""
    assert composition_violation("") is True


# ── _make_referee_fn: off-by-one seq read + composition veto (TASK 1 + TASK 5) ─

def _write_binder_cif(chai_out: Path, binder_residue_names: list[str],
                      target_residue_names=("ALA", "LYS")) -> Path:
    """Write a chai-style output dir (scores.model_idx_0.npz + pred.model_idx_0.cif) with the
    given binder chain-B residues, so _best_cif_path + binder_seq_from_cif resolve it."""
    import gemmi
    import numpy as np

    chai_out.mkdir(parents=True, exist_ok=True)
    # scores npz so _best_cif_path picks model 0.
    np.savez(chai_out / "scores.model_idx_0.npz",
             aggregate_score=np.array([1.0]), ptm=np.array([0.5]),
             iptm=np.array([0.5]), has_inter_chain_clashes=np.array([False]))

    st = gemmi.Structure()
    st.name = "iter"
    model = gemmi.Model("1")

    def _mkchain(name, names):
        ch = gemmi.Chain(name)
        for i, rn in enumerate(names, start=1):
            res = gemmi.Residue()
            res.name = rn
            res.seqid = gemmi.SeqId(i, " ")
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.pos = gemmi.Position(float(i), 0.0, 0.0)
            atom.element = gemmi.Element("C")
            atom.b_iso = 70.0
            res.add_atom(atom)
            ch.add_residue(res)
        return ch

    model.add_chain(_mkchain("A", list(target_residue_names)))
    model.add_chain(_mkchain("B", list(binder_residue_names)))
    st.add_model(model)
    st.setup_entities()
    st.make_mmcif_document().write_file(str(chai_out / "pred.model_idx_0.cif"))
    return chai_out / "pred.model_idx_0.cif"


class _FakePred:
    def __init__(self):
        import numpy as np
        # 2 target tokens (chain index 0) + binder tokens (chain index 1).
        self.token_index = np.array([0, 0, 1, 1, 1])
        self.plddt = np.array([60.0, 60.0, 70.0, 70.0, 70.0])
        self.iptm = 0.5


class _FakeStep:
    def __init__(self, d_fasta):
        from xenodesign.loop import LoopState, LoopStep
        import numpy as np
        self._inner = LoopStep(state=LoopState(d_fasta=d_fasta, coords=np.zeros((1, 3))),
                               prediction=_FakePred(), score=0.5)

    @property
    def state(self):
        return self._inner.state

    @property
    def prediction(self):
        return self._inner.prediction


def test_referee_reads_scored_cif_not_state(tmp_path):
    """#31 acceptance (CPU proxy): the referee scores the CIF's chain-B seq, and a
    poly-Ala scored sequence is composition-vetoed even though step.state.d_fasta is diverse.

    #37: the composition veto now rides its OWN channel (RefereeScore.composition_violation),
    NOT the chirality channel — so the reported chirality stays the real measured value for
    BOTH iters (never forced to 1.0).
    """
    from scripts.design_alpha import _make_referee_fn, binder_seq_from_cif

    loop_dir = tmp_path / "loop"
    # iter 0: diverse binder in the CIF → NOT vetoed.
    cif0 = _write_binder_cif(loop_dir / "iter_000" / "chai_out",
                             ["DAL", "DCY", "DAS", "DGL", "DPN", "GLY", "DHI"])
    # iter 1: poly-Ala binder in the CIF → composition veto.
    cif1 = _write_binder_cif(loop_dir / "iter_001" / "chai_out",
                             ["DAL"] * 6 + ["GLY"])

    referee = _make_referee_fn(loop_dir, esm_judge=None)

    # step.state.d_fasta is DELIBERATELY diverse for BOTH (the off-by-one trap): if the referee
    # read state instead of the CIF, neither would be vetoed.
    from xenodesign.io_spec import to_d_fasta
    diverse_state = to_d_fasta("ACDEFGHIKLMNPQRSTVWYA")
    s0 = _FakeStep(diverse_state)
    s1 = _FakeStep(diverse_state)

    r0 = referee(s0, 0)
    r1 = referee(s1, 1)

    # The CIF-read sequences are what the referee scores.
    assert binder_seq_from_cif(cif0, "B")  # diverse
    assert binder_seq_from_cif(cif1, "B") == "AAAAAAG"  # poly-Ala

    # iter 0 diverse → composition-clean; iter 1 poly-Ala → composition-vetoed.
    assert r0.composition_violation is False
    assert r1.composition_violation is True
    # #37 CRITICAL: chirality is NEVER corrupted — both iters carry their REAL measured
    # chirality (read from the tiny synthetic CIFs), never forced to 1.0 by the comp veto.
    assert r0.chirality_violation < 0.5
    assert r1.chirality_violation < 0.5
    assert r1.chirality_violation != 1.0


def test_referee_composition_veto_deselects_low_complexity(tmp_path):
    """A low-complexity (poly-Ala) candidate must be DE-SELECTED by the panel."""
    from scripts.design_alpha import _make_referee_fn
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.judges.panel import JudgePanel

    loop_dir = tmp_path / "loop"
    # iter 0: poly-Ala (should be vetoed). iter 1: diverse (should win).
    _write_binder_cif(loop_dir / "iter_000" / "chai_out", ["DAL"] * 6 + ["GLY"])
    _write_binder_cif(loop_dir / "iter_001" / "chai_out",
                      ["DAL", "DCY", "DAS", "DGL", "DPN", "GLY", "DHI"])

    referee = _make_referee_fn(loop_dir, esm_judge=None)
    steps = [_FakeStep(to_d_fasta("ACDEFGHIKLMNPQRSTVWYA")) for _ in range(2)]
    scores = [referee(s, i) for i, s in enumerate(steps)]

    panel = JudgePanel(score_fn=lambda step: None)
    result = panel.combine(scores)

    assert result.vetoed[0] is True, "poly-Ala candidate must be vetoed"
    assert result.vetoed[1] is False
    assert result.selected_idx == 1, "panel must select the diverse, non-vetoed candidate"
    # #37: the poly-Ala step is de-selected via composition_violation, NOT by corrupting its
    # chirality — its reported chirality_violation stays the real measured value (< 1.0).
    assert scores[0].composition_violation is True
    assert result.raw_scores[0].chirality_violation != 1.0
    assert result.raw_scores[0].chirality_violation < 0.5


def test_referee_uses_pll_judge_on_scored_seq(tmp_path):
    """The injected ESM-PLL judge is called on the CIF-read sequence (TASK 4 wiring)."""
    from scripts.design_alpha import _make_referee_fn
    from xenodesign.io_spec import to_d_fasta

    loop_dir = tmp_path / "loop"
    _write_binder_cif(loop_dir / "iter_000" / "chai_out",
                      ["DAL", "DCY", "DAS", "DGL", "DPN", "GLY", "DHI"])

    seen = {}

    def _mock_pll(seq):
        seen["seq"] = seq
        return -1.5

    referee = _make_referee_fn(loop_dir, esm_judge=_mock_pll)
    score = referee(_FakeStep(to_d_fasta("ACDEFGHIKLMNPQRSTVWYA")), 0)

    # PLL was computed on the scored-CIF binder seq, not on step.state.
    assert score.pll == -1.5
    assert seen["seq"] == "ACDEFGH"  # DAL DCY DAS DGL DPN GLY DHI -> A C D E F G H


# ── constraint threading via the backend wrappers (TASK 3 wiring) ─────────────

class _SpyBackend:
    """Records every (method, constraint_path) chai call for the wrapper threading test."""

    def __init__(self):
        self.calls = []

    def predict(self, entities, out_dir, num_diffn_timesteps=200, constraint_path=None,
                msa_directory=None):
        self.calls.append(("predict", constraint_path))
        self.last_entities = entities
        self.last_msa_directory = msa_directory
        return _FakePred()

    def truncated_refine(self, structure, ref_time_steps, out_dir):
        self.calls.append(("truncated_refine", None))
        return _FakePred()


def test_predict_wrapper_threads_constraint_path(tmp_path):
    """_PredictBackendWrapper passes its constraint_path to every backend.predict call."""
    from scripts.design_demo import _PredictBackendWrapper
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.loop import LoopState

    spy = _SpyBackend()
    target = {"type": "protein", "name": "target", "sequence": "ACDE", "chirality": "L"}
    cpath = tmp_path / "alpha.restraints"
    cpath.write_text("header\n")
    wrapper = _PredictBackendWrapper(spy, target, constraint_path=cpath)

    state = LoopState(d_fasta=to_d_fasta("ACDEFG"), coords=__import__("numpy").zeros((1, 3)))
    wrapper(state, ref_time_steps=50, out_dir=tmp_path / "iter0")
    wrapper(state, ref_time_steps=50, out_dir=tmp_path / "iter1")

    assert len(spy.calls) == 2
    assert all(method == "predict" for method, _ in spy.calls)
    assert all(cp == cpath for _, cp in spy.calls), (
        f"every predict call must carry the constraint path; got {spy.calls}")
    assert wrapper.last_out_dir == tmp_path / "iter1"


def test_predict_wrapper_none_constraint_threads_none(tmp_path):
    """With no constraint, _PredictBackendWrapper passes constraint_path=None."""
    from scripts.design_demo import _PredictBackendWrapper
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.loop import LoopState

    spy = _SpyBackend()
    target = {"type": "protein", "name": "target", "sequence": "ACDE", "chirality": "L"}
    wrapper = _PredictBackendWrapper(spy, target)  # no constraint_path
    state = LoopState(d_fasta=to_d_fasta("ACDEFG"), coords=__import__("numpy").zeros((1, 3)))
    wrapper(state, ref_time_steps=50, out_dir=tmp_path / "iter0")
    assert spy.calls == [("predict", None)]


def test_predict_wrapper_single_dict_is_binder_chain_B(tmp_path):
    """Legacy single-dict target → entities [target, binder]; binder stays chain B (α-identical)."""
    from scripts.design_demo import _PredictBackendWrapper
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.loop import LoopState

    spy = _SpyBackend()
    target = {"type": "protein", "name": "target", "sequence": "ACDE", "chirality": "L"}
    wrapper = _PredictBackendWrapper(spy, target)
    state = LoopState(d_fasta=to_d_fasta("ACDEFG"), coords=__import__("numpy").zeros((1, 3)))
    wrapper(state, ref_time_steps=50, out_dir=tmp_path / "iter0")

    ents = spy.last_entities
    assert [e["name"] for e in ents] == ["target", "binder"]  # binder LAST → chain B
    assert ents[-1]["chirality"] == "D"
    assert spy.last_msa_directory is None


def test_predict_wrapper_multi_entity_target_plus_msa(tmp_path):
    """Multi-chain target LIST + msa_directory: binder appended LAST (chain C) and MSA forwarded."""
    from scripts.design_demo import _PredictBackendWrapper
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.loop import LoopState

    spy = _SpyBackend()
    ha = [
        {"type": "protein", "name": "HA1", "sequence": "ACDE", "chirality": "L"},
        {"type": "protein", "name": "HA2", "sequence": "FGHI", "chirality": "L"},
    ]
    wrapper = _PredictBackendWrapper(spy, ha, msa_directory=tmp_path / "msas")
    state = LoopState(d_fasta=to_d_fasta("ACDEFG"), coords=__import__("numpy").zeros((1, 3)))
    wrapper(state, ref_time_steps=50, out_dir=tmp_path / "iter0")

    ents = spy.last_entities
    assert [e["name"] for e in ents] == ["HA1", "HA2", "binder"]  # binder LAST → chain C
    assert ents[-1]["chirality"] == "D"
    assert spy.last_msa_directory == tmp_path / "msas"


def test_predict_wrapper_no_target_is_binder_only(tmp_path):
    """No-target mode (entities None) → [binder] only (free mixed-chirality peptide, chain A)."""
    from scripts.design_demo import _PredictBackendWrapper
    from xenodesign.io_spec import to_d_fasta
    from xenodesign.loop import LoopState

    spy = _SpyBackend()
    wrapper = _PredictBackendWrapper(spy, None)
    state = LoopState(d_fasta=to_d_fasta("ACDEFG"), coords=__import__("numpy").zeros((1, 3)))
    wrapper(state, ref_time_steps=50, out_dir=tmp_path / "iter0")

    ents = spy.last_entities
    assert [e["name"] for e in ents] == ["binder"]  # binder only → chain A


# ── --objective {iptm|mixed} + --periodicity_gate CLI flags ────

def test_objective_flag_default_is_iptm():
    """The reproducible DEFAULT objective is ipTM (must stay byte-for-byte)."""
    from scripts.design_alpha import _parse_args
    assert _parse_args([]).objective == "iptm"


def test_objective_flag_accepts_mixed():
    from scripts.design_alpha import _parse_args
    assert _parse_args(["--objective", "mixed"]).objective == "mixed"


def test_objective_flag_accepts_ipsae():
    from scripts.design_alpha import _parse_args
    assert _parse_args(["--objective", "ipsae"]).objective == "ipsae"


def test_periodicity_gate_flag_default_off():
    from scripts.design_alpha import _parse_args
    args = _parse_args([])
    assert args.periodicity_gate is False
    assert args.heptad_thresh == 0.35


def test_periodicity_gate_flag_on_with_thresh():
    from scripts.design_alpha import _parse_args
    args = _parse_args(["--periodicity_gate", "--heptad_thresh", "0.4"])
    assert args.periodicity_gate is True
    assert args.heptad_thresh == 0.4


# ── mixed_objective_from_cif: per-candidate score_complex panel -> mixed composite ──

def _write_complex_cif(tmp_path, with_scores: bool = False) -> Path:
    """A tiny 2-chain CIF with FULL backbone+CB per residue (chain A target L, chain B binder D),
    arranged so the two chains form a real (contacting) interface — enough for score_complex.
    structural() to compute bsa/contacts/sc/hbond. Optionally also writes a chai scores npz so the
    confidence() terms can be read from the same dir."""
    import gemmi
    import numpy as np

    def _mkres(name, i, base):
        res = gemmi.Residue(); res.name = name; res.seqid = gemmi.SeqId(i, " ")
        for an, el, off in [("N", "N", (0, 0, 0)), ("CA", "C", (1.5, 0, 0)),
                            ("C", "C", (2.0, 1.4, 0)), ("O", "O", (1.2, 2.3, 0)),
                            ("CB", "C", (2.0, -1.0, 1.0))]:
            a = gemmi.Atom(); a.name = an; a.element = gemmi.Element(el)
            a.pos = gemmi.Position(base[0] + off[0], base[1] + off[1], base[2] + off[2])
            res.add_atom(a)
        return res

    st = gemmi.Structure(); m = gemmi.Model("1")
    chA = gemmi.Chain("A")
    for i, (nm, b) in enumerate([("ALA", (0, 0, 0)), ("LYS", (3.8, 0, 0)),
                                 ("GLU", (7.6, 0, 0))], 1):
        chA.add_residue(_mkres(nm, i, b))
    chB = gemmi.Chain("B")
    for i, (nm, b) in enumerate([("DAL", (0.5, 4.0, 0)), ("DGL", (4.3, 4.0, 0)),
                                 ("GLY", (8.1, 4.0, 0))], 1):
        chB.add_residue(_mkres(nm, i, b))
    m.add_chain(chA); m.add_chain(chB); st.add_model(m)

    out = tmp_path / "chai_out"; out.mkdir(parents=True, exist_ok=True)
    cif = out / "pred.model_idx_0.cif"
    st.make_mmcif_document().write_file(str(cif))
    if with_scores:
        np.savez(out / "scores.model_idx_0.npz",
                 aggregate_score=np.array([1.0]), ptm=np.array([0.5]),
                 iptm=np.array([0.5]),
                 per_chain_pair_iptm=np.array([[0.9, 0.4], [0.4, 0.9]]),
                 has_inter_chain_clashes=np.array([False]))
    return cif


def test_mixed_objective_from_cif_returns_composite_and_panel(tmp_path):
    """mixed_objective_from_cif builds a score_complex panel for the candidate CIF and returns a
    mixed_objective composite in [0,1] plus the raw panel dict (geometry terms present)."""
    from scripts.design_alpha import mixed_objective_from_cif

    cif = _write_complex_cif(tmp_path)
    composite, panel = mixed_objective_from_cif(cif, chai_dir=None)
    assert 0.0 <= composite <= 1.0
    # the parity-invariant geometry terms score_complex.structural produces:
    for key in ("bsa_A2", "n_residue_contacts", "sc_normal_opp", "hbond_density"):
        assert key in panel


def test_mixed_objective_from_cif_uses_confidence_when_chai_dir_given(tmp_path):
    """With chai_dir, the iptm confidence term is read into the panel (so it can drive the
    mixed composite's iptm/ipsae/ipae terms)."""
    from scripts.design_alpha import mixed_objective_from_cif

    cif = _write_complex_cif(tmp_path, with_scores=True)
    composite, panel = mixed_objective_from_cif(cif, chai_dir=cif.parent)
    assert 0.0 <= composite <= 1.0
    assert "iptm" in panel  # confidence() read the per_chain_pair_iptm off-diagonal


# ── make_mixed_loop_score_fn: per-iter mixed score + graceful ipTM fallback ────

class _MixedPred:
    def __init__(self, iptm=0.5):
        import numpy as np
        self.token_index = np.array([0, 0, 1, 1, 1])
        self.plddt = np.array([60.0, 60.0, 70.0, 70.0, 70.0])
        self.iptm = iptm


def test_mixed_loop_score_fn_falls_back_to_iptm_when_no_out_dir():
    """When the wrapper has no last_out_dir (no structure step yet / unreadable), the mixed loop
    score_fn falls back to the reproducible ipTM design_score (never crashes the loop)."""
    from scripts.design_alpha import make_mixed_loop_score_fn, _loop_score_fn

    class _Wrapper:
        last_out_dir = None

    score_fn = make_mixed_loop_score_fn(_Wrapper())
    pred = _MixedPred(iptm=0.5)
    assert score_fn(pred) == _loop_score_fn(pred)


def test_mixed_loop_score_fn_uses_panel_when_cif_present(tmp_path):
    """When the wrapper points at a scored CIF dir, the mixed loop score_fn returns the
    mixed_objective composite (a DIFFERENT value than the ipTM-only design_score)."""
    from scripts.design_alpha import make_mixed_loop_score_fn, mixed_objective_from_cif

    cif = _write_complex_cif(tmp_path, with_scores=True)

    class _Wrapper:
        last_out_dir = cif.parent  # the chai_out dir holding the CIF + scores

    score_fn = make_mixed_loop_score_fn(_Wrapper())
    got = score_fn(_MixedPred(iptm=0.5))
    expected, _ = mixed_objective_from_cif(cif, chai_dir=cif.parent)
    assert got == pytest.approx(expected)
    assert 0.0 <= got <= 1.0


# ── --objective ipsae: per-candidate ipSAE + per-iter loop score_fn ──

def test_ipsae_objective_from_cif_returns_panel_ipsae(tmp_path):
    """ipsae_objective_from_cif returns the panel's 'ipsae' value (reusing the mixed panel
    machinery) — the SAME panel mixed_objective_from_cif builds — not the mixed composite."""
    from scripts.design_alpha import ipsae_objective_from_cif, mixed_objective_from_cif

    cif = _write_complex_cif(tmp_path, with_scores=True)
    ipsae, panel = ipsae_objective_from_cif(cif, chai_dir=cif.parent)
    _composite, mixed_panel = mixed_objective_from_cif(cif, chai_dir=cif.parent)
    # the returned float is exactly the panel's ipsae (0.0 when the term is absent/unreadable)
    assert ipsae == pytest.approx(float(mixed_panel.get("ipsae", 0.0)))
    # the geometry panel is the same machinery as the mixed objective
    assert panel.get("bsa_A2") == mixed_panel.get("bsa_A2")


def test_ipsae_loop_score_fn_falls_back_to_iptm_when_no_out_dir():
    """No last_out_dir → the ipsae loop score_fn falls back to the reproducible ipTM design_score."""
    from scripts.design_alpha import make_ipsae_loop_score_fn, _loop_score_fn

    class _Wrapper:
        last_out_dir = None

    score_fn = make_ipsae_loop_score_fn(_Wrapper())
    pred = _MixedPred(iptm=0.5)
    assert score_fn(pred) == _loop_score_fn(pred)


def test_ipsae_loop_score_fn_uses_panel_ipsae_when_cif_present(tmp_path):
    """With a scored CIF dir, the ipsae loop score_fn returns the candidate's raw ipSAE (the
    panel's 'ipsae'), NOT the ipTM design_score nor the mixed composite."""
    from scripts.design_alpha import make_ipsae_loop_score_fn, ipsae_objective_from_cif

    cif = _write_complex_cif(tmp_path, with_scores=True)

    class _Wrapper:
        last_out_dir = cif.parent

    score_fn = make_ipsae_loop_score_fn(_Wrapper())
    got = score_fn(_MixedPred(iptm=0.5))
    expected, _ = ipsae_objective_from_cif(cif, chai_dir=cif.parent)
    assert got == pytest.approx(expected)


def test_composition_violation_gt_binder_passes():
    """CALIBRATION: the real α GT binder (chain-A record of the gitignored GT FASTA) MUST pass
    the composition floor. The sequence is unpublished — we load it at runtime and assert it
    passes; we never inline it. Skipped if the gitignored FASTA is absent."""
    from xenodesign.benchmark.cases import get_case
    from xenodesign.io_spec import d_fasta_to_one_letter
    from xenodesign.seed import read_target_sequence

    case = get_case("alpha")
    fasta = Path(case.fasta_path)
    if not fasta.exists():
        pytest.skip(f"GT FASTA absent ({fasta}); calibration check skipped")
    # The GT binder is the chain-A record (all-D); decode its D-CCD to one-letter L.
    gt_d_fasta = read_target_sequence(fasta, name="trimer_DL_ABLE_A")
    gt_l_seq = d_fasta_to_one_letter(gt_d_fasta)
    assert composition_violation(gt_l_seq) is False, (
        "the genuine GT binder must PASS the composition floor — recalibrate thresholds")
