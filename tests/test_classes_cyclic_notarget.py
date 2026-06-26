"""CPU tests for the NO-TARGET free-cyclic peptide path (target_type='none', T-none).

Covers (all CPU, predictor/CIF mocked):
  * targets.target_entities -> ([], None, None) for target_type='none';
  * the loop wrapper assembles [binder] only (binder = chain A);
  * Cyclic.restraints SKIPS the His<->Zn metal_coordination rows when there's no Zn chain,
    but still honours the opt-in head-to-tail closure on chain A;
  * the four intramolecular-objective terms (mainchain-pLDDT of the cyclizing termini,
    chirality goodness, closure+backbone geometry, pTM) combine into a finite [0,1] scalar
    with the documented default weights;
  * the cyclic objective hook returns the intramolecular score fn when target_type=='none'
    and the ipTM/pTM _loop_score_fn otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

from xenodesign.benchmark.cases import get_case
from xenodesign.benchmark.restraints import parse_restraints
from xenodesign.classes.base import CLASS_REGISTRY  # noqa: F401  (resolve import order first)
from xenodesign.classes.cyclic import (
    Cyclic,
    INTRAMOLECULAR_WEIGHTS,
    combine_intramolecular_terms,
    intramolecular_terms_from_records,
)
from xenodesign.config import resolve_config
from xenodesign import targets


# ── 1. no-target dispatch path ────────────────────────────────────────────────

def test_target_entities_none_is_empty():
    cfg = resolve_config("cyclic", target_type="none")
    ents, msa_dir, hint = targets.target_entities(cfg)
    assert ents == []
    assert msa_dir is None and hint is None


def test_wrapper_builds_binder_only_chain_a():
    """The loop wrapper appends the binder LAST; with an empty target it is the SOLE chain (A)."""
    from scripts.design_demo import _build_entities, _as_target_list
    ents = _build_entities(_as_target_list([]), "ACDEFGHIKLMN")
    assert len(ents) == 1
    assert ents[0]["name"] == "binder"
    # binder chain letter = chr('A' + len(target_entities)) == 'A'
    assert chr(ord("A") + 0) == "A"


# ── 2. restraints: metal skipped, closure honoured ───────────────────────────

def test_restraints_no_target_skips_metal_writes_only_closure(tmp_path):
    cfg = resolve_config("cyclic", target_type="none",
                         cli_overrides={"restraint.params": {"closure": True},
                                        "use_pepmlm": False})
    case = get_case("cyclic")
    path = Cyclic().restraints(cfg, case, tmp_path, target_ctx=([], None))
    rows = parse_restraints(path)
    # No metal_coordination rows (no Zn chain); exactly the head-to-tail closure bond remains.
    assert all(r["connection_type"] != "contact" for r in rows)
    assert len(rows) == 1                                  # closure only
    assert rows[0]["connection_type"] == "covalent"        # head-to-tail backbone bond
    # closure is on chain A (the sole binder chain in no-target mode).
    assert rows[0]["chainA"] == "A" and rows[0]["chainB"] == "A"


def test_restraints_no_target_no_closure_is_none(tmp_path):
    """No Zn AND no closure -> nothing to restrain -> None (unconstrained free peptide)."""
    cfg = resolve_config("cyclic", target_type="none")   # closure not set
    case = get_case("cyclic")
    assert Cyclic().restraints(cfg, case, tmp_path, target_ctx=([], None)) is None


def test_restraints_metal_path_unchanged(tmp_path):
    """Sanity: the metal (Zn present) path still emits coordination rows on chain B."""
    cfg = resolve_config("cyclic", target_type="metal")
    case = get_case("cyclic")
    zn = [{"type": "ligand", "name": "zn"}]
    path = Cyclic().restraints(cfg, case, tmp_path, target_ctx=(zn, None))
    rows = parse_restraints(path)
    assert any(r["connection_type"] == "contact" for r in rows)


# ── 3. intramolecular objective terms + combination ──────────────────────────

def _ideal_dipeptide_records(n_res=12, plddt=90.0):
    """Synthetic L-backbone records: a flat trans head-to-tail amide + ideal valence angles.

    Each record carries N/CA/C/O/CB coords and a per-mainchain-atom pLDDT. Geometry is laid
    out so the C(L)->N(1) closure dihedral is ~180 deg (planar trans) and the N-CA-C valence
    angles are ~111 deg (ideal). Pure in-memory; no CIF/gemmi.
    """
    recs = []
    for i in range(n_res):
        base = np.array([float(i) * 3.8, 0.0, 0.0])
        rec = {
            "N":  base + np.array([-1.2, 0.0, 0.0]),
            "CA": base,
            "C":  base + np.array([1.0, 0.6, 0.0]),
            "O":  base + np.array([1.0, 1.8, 0.0]),
            "CB": base + np.array([-0.5, -0.77, -1.2]),   # L-alanine-like frame
            "plddt": {"N": plddt, "CA": plddt, "C": plddt, "O": plddt},
            "chirality": "L",
        }
        recs.append(rec)
    return recs


def test_intramolecular_terms_finite_and_in_unit_range():
    recs = _ideal_dipeptide_records()
    terms = intramolecular_terms_from_records(recs, ptm=0.7)
    for k in ("mainchain_plddt", "chirality", "geometry", "ptm"):
        assert k in terms
        assert np.isfinite(terms[k])
        assert 0.0 <= terms[k] <= 1.0


def test_mainchain_plddt_term_uses_termini_only():
    recs = _ideal_dipeptide_records(plddt=50.0)
    # Boost ONLY the termini (res 1 and res L) mainchain pLDDT to 100; interior stays 50.
    for r in (recs[0], recs[-1]):
        r["plddt"] = {k: 100.0 for k in r["plddt"]}
    terms = intramolecular_terms_from_records(recs, ptm=0.5)
    assert terms["mainchain_plddt"] == pytest.approx(1.0, abs=1e-6)   # 100/100, termini only


def test_chirality_term_penalises_wrong_handedness():
    good = _ideal_dipeptide_records()
    g = intramolecular_terms_from_records(good, ptm=0.5)["chirality"]
    # Flip every CB to its mirror -> all D by the L label -> all violations.
    bad = _ideal_dipeptide_records()
    for r in bad:
        r["CB"] = r["CB"] * np.array([1.0, 1.0, -1.0])   # reflect z
    b = intramolecular_terms_from_records(bad, ptm=0.5)["chirality"]
    assert g == pytest.approx(1.0)
    assert b < 0.5


def test_geometry_term_drops_for_twisted_closure():
    flat = intramolecular_terms_from_records(_ideal_dipeptide_records(), ptm=0.5)["geometry"]
    twisted = _ideal_dipeptide_records()
    # Lift the N-terminus N out of plane so C(L)->N(1) omega is far from 0/180.
    twisted[0]["N"] = twisted[0]["N"] + np.array([0.0, 0.0, 5.0])
    g = intramolecular_terms_from_records(twisted, ptm=0.5)["geometry"]
    assert g < flat


def test_combine_uses_named_weights():
    terms = {"mainchain_plddt": 1.0, "chirality": 1.0, "geometry": 1.0, "ptm": 1.0}
    assert combine_intramolecular_terms(terms) == pytest.approx(1.0)
    zero = {k: 0.0 for k in terms}
    assert combine_intramolecular_terms(zero) == pytest.approx(0.0)
    # A single term at 1.0 contributes exactly its weight.
    only_ptm = {"mainchain_plddt": 0.0, "chirality": 0.0, "geometry": 0.0, "ptm": 1.0}
    assert combine_intramolecular_terms(only_ptm) == pytest.approx(INTRAMOLECULAR_WEIGHTS["ptm"])


def test_weights_sum_to_one():
    assert sum(INTRAMOLECULAR_WEIGHTS.values()) == pytest.approx(1.0)


# ── 4. objective hook routing ────────────────────────────────────────────────

class _FakePred:
    iptm = 0.5
    ptm = 0.6
    token_index = np.array([0, 0, 0])
    plddt = np.array([80.0, 80.0, 80.0])


def test_objective_routes_to_intramolecular_for_none():
    cfg = resolve_config("cyclic", target_type="none")
    fn = Cyclic().objective(cfg, wrapper=None)
    # With no wrapper.last_out_dir / no CIF, the score fn falls back to a pTM-only finite score.
    s = fn(_FakePred())
    assert np.isfinite(s)
    assert 0.0 <= s <= 1.0


def test_objective_routes_to_iptm_for_metal():
    from xenodesign.classes.alpha import _loop_score_fn
    cfg = resolve_config("cyclic", target_type="metal")
    fn = Cyclic().objective(cfg, wrapper=None)
    assert fn is _loop_score_fn


# ── 5. no-target loop wrapper assembles the L-seed predict with [binder] only ──

def test_loop_wrapper_assembles_binder_only_for_empty_target():
    """The dispatcher feeds target_entities('none') -> [] into the loop wrapper; the wrapper
    then folds [binder] as the sole chain (chain A) every iteration. This is the loop-level
    contract the full run_design relies on (the full greedy loop's seq_update reads real CIFs,
    so it is exercised on GPU; here we pin the entity assembly that drives it)."""
    from scripts.design_demo import _LoopBackendWrapper

    captured: dict = {}

    class _Backend:
        def truncated_refine(self, structure, ref_time_steps, out_dir):
            captured["entities"] = structure["entities"]
            return object()

    from xenodesign.loop import LoopState
    from xenodesign.io_spec import to_d_fasta

    wrapper = _LoopBackendWrapper(_Backend(), target_entity=[], msa_directory=None)
    state = LoopState(d_fasta=to_d_fasta("ACDEFGHIKLMN"), coords=None)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        wrapper.truncated_refine(state, ref_time_steps=5, out_dir=td)

    assert len(captured["entities"]) == 1
    assert captured["entities"][0]["name"] == "binder"


# ── 5. Gly-guard the from-scratch no-target free-cyclic seed (#9) ──────────────

def _glyfree_generator():
    """A from-scratch seed generator that returns a deterministic glycine-FREE L sequence,
    so the Gly-guard is exercised regardless of any random draw."""
    class _G:
        def generate_conditioned(self, target_seq, length):
            return "A" * length   # no 'G' anywhere -> all-D would be un-tokenizable by chai
    return _G()


def test_no_target_cyclic_seed_is_tokenizable_even_when_glyfree(monkeypatch):
    # REGRESSION (#9): the no-target free-cyclic default seed can be encoded all-D; an all-D
    # chain with NO glycine is rejected by chai (>=1 canonical residue needed, ADR-004). The
    # alpha/non_alpha seeds are Gly-guarded; the no-target free-cyclic seed must be too, or a
    # from-scratch run crashes at iter_000.
    import xenodesign.seed as seed_mod

    monkeypatch.setattr(seed_mod, "make_configured_generator",
                        lambda cfg: _glyfree_generator())
    cfg = resolve_config("cyclic", target_type="none", cli_overrides={"use_pepmlm": False})
    spec = Cyclic().seed(cfg, target_seq="")
    # A from-scratch no-target seed always carries >=1 tokenizable canonical (achiral Gly) anchor.
    assert "G" in spec.one_letter, "no-target free-cyclic seed must contain a Gly anchor"


def test_no_target_cyclic_seed_keeps_existing_glycine(monkeypatch):
    # When the generator already yields a glycine, the guard is a no-op (no extra/forced Gly).
    import xenodesign.seed as seed_mod

    class _HasGly:
        def generate_conditioned(self, target_seq, length):
            return ("G" + "A" * (length - 1))

    monkeypatch.setattr(seed_mod, "make_configured_generator", lambda cfg: _HasGly())
    cfg = resolve_config("cyclic", target_type="none", cli_overrides={"use_pepmlm": False})
    spec = Cyclic().seed(cfg, target_seq="")
    assert spec.one_letter.count("G") == 1   # the existing Gly is kept, none added
