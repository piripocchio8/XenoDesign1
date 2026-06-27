"""CPU tests for the Cyclic BinderClass adapter (xenodesign.classes.cyclic, T7).

Pure-CPU: exercises the cyclic class hooks (seed / closure / restraints / objective /
ss_bias / report) against a MOCKED predictor — never touches chai/torch. The migrated
geometry/seed helpers retain their behavioural coverage in tests/test_design_cyclic.py
(the regression oracle, which now imports them through the re-export shim).
"""
from __future__ import annotations

import numpy as np

from xenodesign.classes.base import SeedSpec
from xenodesign.classes.cyclic import Cyclic
from xenodesign.config import resolve_binder_length, resolve_config
from xenodesign.benchmark.cases import get_case


# FROM-SCRATCH seeding: use the offline (use_pepmlm=False) generator so CPU tests never load
# the real PepMLM weights, and the length is the per-class DEFAULT (24, not the deposit's 12).
def _cyclic_cfg(**ov):
    ov.setdefault("use_pepmlm", False)
    return resolve_config("cyclic", target_type="metal", cli_overrides=ov)


def test_cyclic_case_id():
    assert Cyclic().case_id == "cyclic"


def test_cyclic_seed_pins_his_chirality():
    c = Cyclic()
    cfg = _cyclic_cfg()
    spec = c.seed(cfg, target_seq=None)
    assert isinstance(spec, SeedSpec)
    # From-scratch DEFAULT length (24 = S2-symmetric 6UFA full length), NOT the deposit's 12.
    assert len(spec.one_letter) == resolve_binder_length(cfg) == 24
    # Opt-in coordinating His (restraints on by default): L-His@6, D-His@12 (1-based).
    assert spec.fixed_chirality.get(6) == "L" and spec.fixed_chirality.get(12) == "D"
    assert spec.one_letter[5] == "H" and spec.one_letter[11] == "H"


def test_cyclic_seed_is_deterministic_in_cfg_seed():
    c = Cyclic()
    cfg = _cyclic_cfg(seed=7)
    a = c.seed(cfg, target_seq=None)
    b = c.seed(cfg, target_seq=None)
    assert a.one_letter == b.one_letter


def test_cyclic_seed_his_opt_in_off_when_restraints_off():
    """His placement is OPT-IN: with restraints off the seed carries no mandatory His scaffold."""
    c = Cyclic()
    cfg = _cyclic_cfg(restraints_on=False)
    spec = c.seed(cfg, target_seq=None)
    assert spec.fixed_chirality == {}


def test_cyclic_closure_empty_unless_requested():
    c = Cyclic()
    cfg = _cyclic_cfg()
    spec = c.seed(cfg, target_seq=None)
    assert c.closure(cfg, spec) == []  # default: LINEAR + emergent closure
    cfg2 = _cyclic_cfg(**{"restraint.params": {"closure": True}})
    rows = c.closure(cfg2, spec)
    assert len(rows) == 1  # one head-to-tail COVALENT row
    assert rows[0].split(",")[4] == "covalent"


def test_cyclic_ss_bias_is_anti_alpha():
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal")
    ss = c.ss_bias(cfg, get_case("cyclic"))
    assert ss.target_helix_frac == 0.0


def test_cyclic_objective_default_is_iptm_score():
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal")
    fn = c.objective(cfg, wrapper=None)

    class P:
        token_index = np.array([1, 1])
        plddt = np.array([70.0, 70.0])
        iptm = 0.6

    assert isinstance(fn(P()), float)


def test_cyclic_restraints_off_when_disabled():
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal",
                         cli_overrides={"restraints_on": False})
    assert c.restraints(cfg, get_case("cyclic"), out_dir="/tmp", target_ctx=None) is None


def _metal_coords():
    """The declared 6UFA coordinators (required now for a metal cyclic run)."""
    from xenodesign.coordinators import parse_coord_residues
    return [(c.pos, c.one_letter, c.three_letter, c.chirality, c.atom)
            for c in parse_coord_residues("H6,DHI12,H18,DHI24")]


def test_cyclic_restraints_writes_metal_coordination(tmp_path):
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False,
                                        "restraint.params": {"coord_residues": _metal_coords()}})
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=None)
    assert path is not None and path.exists()
    text = path.read_text()
    assert "contact" in text
    # FIX A: metal coordination is CONTACT-ONLY by default (covalent-to-Zn crashes Chai); the
    # metal_coord rows are contacts, NOT covalent. The head-to-tail closure covalent still rides
    # along by default (a real, working backbone bond).
    assert "metal_coord_" in text and "cyclic_closure" in text
    coord_covalents = [r for r in text.splitlines()
                       if "metal_coord_" in r and ",covalent," in r]
    assert coord_covalents == [], "metal coordination must be contact-only by default"


def test_cyclic_restraints_chain_aware_binder_last(tmp_path):
    """With the assembled entity order ([Zn]+binder), the His<->Zn restraint must reference the
    binder chain (B, last) and the Zn chain (A) — not the legacy peptide=A/Zn=B order. This is
    the dispatcher gap: binder appended LAST inverts Zn to chain A."""
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False,
                                        "restraint.params": {"coord_residues": _metal_coords()}})
    zn = {"type": "ligand", "name": "zn", "smiles": "[Zn+2]"}
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=([zn], None))
    rows = [r for r in path.read_text().splitlines() if "metal_coord" in r]
    assert rows, "expected His-Zn coordination rows"
    for r in rows:
        f = r.split(",")
        # contact/covalent columns: chainA, resA, chainB, resB, ... -> His on B (binder), Zn on A.
        assert f[0] == "B", f"His must be on the binder chain B, got {f[0]} in {r}"
        assert f[2] == "A", f"Zn must be on chain A, got {f[2]} in {r}"


def test_cyclic_restraints_legacy_chains_without_ctx(tmp_path):
    """No target_ctx -> legacy standalone-driver order (peptide=A, Zn=B), unchanged."""
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False,
                                        "restraint.params": {"coord_residues": _metal_coords()}})
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=None)
    rows = [r for r in path.read_text().splitlines() if "metal_coord" in r]
    assert rows
    for r in rows:
        f = r.split(",")
        assert f[0] == "A" and f[2] == "B"


def test_cyclic_restraints_appends_closure_when_requested(tmp_path):
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"restraint.params": {"closure": True,
                                                             "coord_residues": _metal_coords()},
                                        "use_pepmlm": False})
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=None)
    text = path.read_text()
    assert "covalent" in text and "cyclic_closure" in text


# ── WT-RESTRAINTS: auto-closure + atom-level coordination + require-coord ──────────

def _coord_tuples(spec):
    """Parse a --coord_residues flag string into the stored 5-tuples (pos, one, three, chir, atom)."""
    from xenodesign.coordinators import parse_coord_residues
    return [(c.pos, c.one_letter, c.three_letter, c.chirality, c.atom)
            for c in parse_coord_residues(spec)]


def test_cyclic_metal_emits_coordination_and_closure_together(tmp_path):
    """WT-RESTRAINTS #1+#3 / FIX A: a cyclic metal run (coord_residues with liganding atoms) must
    write BOTH the 4 residue-level CONTACT coordination rows AND an auto head-to-tail closure row —
    the 6UFA analogue is a CYCLE, so closure rides WITH the coordination rows by default (no opt-in).
    Metal coordination is CONTACT-ONLY by default (covalent-to-Zn crashes Chai)."""
    c = Cyclic()
    coords = _coord_tuples("H6,DHI12,H18,DHI24")
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False,
                                        "restraint.params": {"coord_residues": coords}})
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=None)
    text = path.read_text()
    # 4 residue-level CONTACT His->Zn coordination rows (one per declared coordinator).
    coord_contacts = [r for r in text.splitlines() if "metal_coord_" in r and ",contact," in r]
    assert len(coord_contacts) == 4, f"expected 4 contact coordination rows, got {coord_contacts}"
    # NO covalent-to-metal coordination rows by default (FIX A: they crash Chai).
    coord_covalents = [r for r in text.splitlines() if "metal_coord_" in r and ",covalent," in r]
    assert coord_covalents == [], f"metal coordination must be contact-only, got {coord_covalents}"
    # AND exactly one auto head-to-tail closure row (covalent backbone bond), without any opt-in.
    closures = [r for r in text.splitlines() if "cyclic_closure" in r]
    assert len(closures) == 1, f"expected one auto-closure row, got {closures}"


def test_cyclic_metal_requires_coord_residues(tmp_path):
    """WT-RESTRAINTS #4: a cyclic METAL target with no declared --coord_residues must raise a
    clear ValueError (the liganding chemistry must be declared, not silently defaulted)."""
    import pytest
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False})
    with pytest.raises(ValueError) as ei:
        c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=None)
    assert "coord_residues" in str(ei.value)


def test_cyclic_metal_closure_can_be_disabled(tmp_path):
    """The auto-closure has a disable path (closure=False) for the LINEAR + emergent-closure run."""
    c = Cyclic()
    coords = _coord_tuples("H6,DHI12,H18,DHI24")
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False,
                                        "restraint.params": {"coord_residues": coords,
                                                             "closure": False}})
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=None)
    text = path.read_text()
    assert "cyclic_closure" not in text          # closure suppressed
    assert any("metal_coord_" in r for r in text.splitlines())  # coordination still emitted


def test_cyclic_referee_is_noop():
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal")
    ref = c.referee(cfg, loop_dir="/tmp", esm_judge=None)
    assert ref("step", 0) is None


# ── Part E: achiral Gly anchor for all-one-handedness designs ──────────────────

def test_anchor_forced_cterm_for_all_d_design():
    """All-D design + all-D coordinators: among NON-coordinator positions there is no L,
    so an achiral Gly must be ensured AT THE C-TERMINUS (last non-coordinator position)."""
    one = "AAAAHH"          # H@5,H@6 are coordinators
    fixed = {5: "D", 6: "D"}  # coordinators are D-handed
    out = Cyclic._ensure_canonical_anchor(one, fixed, default_chirality="D")
    assert "G" in out
    # C-terminal non-coordinator position is index 3 (1-based 4); 5,6 are pinned coordinators.
    assert out == "AAAGHH"
    # coordinators are never overwritten
    assert out[4] == "H" and out[5] == "H"


def test_anchor_noop_when_glycine_present():
    """A design already containing a Gly is unchanged (the anchor already exists)."""
    one = "AAGAHH"
    fixed = {5: "D", 6: "D"}
    out = Cyclic._ensure_canonical_anchor(one, fixed, default_chirality="D")
    assert out == one


def test_anchor_not_forced_for_genuinely_mixed_design():
    """A genuinely mixed L/D design (both handedness present among non-coordinators) needs no
    forced Gly: chai can tokenize the L residues, so the all-one-handedness trigger is off."""
    one = "AAAAHH"
    fixed = {5: "D", 6: "D"}  # coordinators only
    # Non-coordinator handedness: pos 1='L', pos 2='D' -> both present among non-coords.
    chir = {1: "L", 2: "D"}
    out = Cyclic._ensure_canonical_anchor(one, fixed, chirality_map=chir,
                                          default_chirality="D")
    assert "G" not in out
    assert out == one


# ── Part G: MetalHawk post-selection verification (recorded, not a hard gate) ──

def _fake_history(d_fasta="H(DHI)", iptm=0.42, ptm=0.55, score=0.42):
    class _Pred:
        pass
    p = _Pred()
    p.iptm, p.ptm = iptm, ptm

    class _State:
        pass
    s = _State()
    s.d_fasta = d_fasta

    class _Step:
        pass
    step = _Step()
    step.prediction, step.state, step.score = p, s, score
    return [step]


def test_metal_geometry_recorded_for_metal_target(tmp_path, monkeypatch):
    """When the target is a metal, _assemble_cyclic_result runs the (reused) metal_geometry_gate
    on the SELECTED CIF and records geometry+perplexity+passed into the result dict."""
    import xenodesign.classes.cyclic as cyc
    from xenodesign.eval.metal_geometry_gate import GateResult

    calls = []

    def _fake_gate(cif_path, **kw):
        calls.append(cif_path)
        return GateResult(geometry="TET", perplexity=1.2, passed=True, ok=True)

    monkeypatch.setattr(cyc, "metal_geometry_gate", _fake_gate)
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path))
    case = get_case("cyclic")
    result = cyc._assemble_cyclic_result(cfg, _fake_history(), panel_result=None,
                                         case=case, out_dir=tmp_path)
    assert calls, "metal_geometry_gate must be called for a metal target"
    assert result["metal_geometry"]["geometry"] == "TET"
    assert result["metal_geometry"]["perplexity"] == 1.2
    assert result["metal_geometry"]["passed"] is True


def test_metal_geometry_skipped_for_non_metal(tmp_path, monkeypatch):
    """No-target (target_type='none'): no metal site, so the gate is never called and the
    metal_geometry field is absent."""
    import xenodesign.classes.cyclic as cyc

    called = []
    monkeypatch.setattr(cyc, "metal_geometry_gate",
                        lambda *a, **k: called.append(True))
    cfg = resolve_config("cyclic", target_type="none", out_dir=str(tmp_path),
                         cli_overrides={"use_pepmlm": False})
    case = get_case("cyclic")
    result = cyc._assemble_cyclic_result(cfg, _fake_history(), panel_result=None,
                                         case=case, out_dir=tmp_path)
    assert not called, "metal_geometry_gate must NOT be called for a non-metal target"
    assert "metal_geometry" not in result


def test_cyclic_report_assembles_result_dict(tmp_path):
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path))
    case = get_case("cyclic")

    class _Pred:
        iptm = 0.42
        ptm = 0.55

    class _State:
        d_fasta = "H(DHI)"

    class _Step:
        prediction = _Pred()
        state = _State()
        score = 0.42

    history = [_Step()]
    result = c.report(cfg, history, panel_result=None, case=case, out_dir=tmp_path)
    assert result["case_id"] == "cyclic"
    assert result["n_iters"] == 1
    assert result["selected_iptm"] == 0.42
    assert (tmp_path / "cyclic_result.json").exists()
