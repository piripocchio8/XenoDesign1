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


def test_cyclic_restraints_writes_metal_coordination(tmp_path):
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path))
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=None)
    assert path is not None and path.exists()
    text = path.read_text()
    assert "contact" in text
    # default (closure off): no covalent closure row
    assert "covalent" not in text


def test_cyclic_restraints_chain_aware_binder_last(tmp_path):
    """With the assembled entity order ([Zn]+binder), the His<->Zn restraint must reference the
    binder chain (B, last) and the Zn chain (A) — not the legacy peptide=A/Zn=B order. This is
    the dispatcher gap: binder appended LAST inverts Zn to chain A."""
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path))
    zn = {"type": "ligand", "name": "zn", "smiles": "[Zn+2]"}
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=([zn], None))
    rows = [r for r in path.read_text().splitlines() if "zn_coord" in r]
    assert rows, "expected His-Zn contact rows"
    for r in rows:
        f = r.split(",")
        # contact_row columns: chainA, resA, chainB, resB, ... -> His on B (binder), Zn on A.
        assert f[0] == "B", f"His must be on the binder chain B, got {f[0]} in {r}"
        assert f[2] == "A", f"Zn must be on chain A, got {f[2]} in {r}"


def test_cyclic_restraints_legacy_chains_without_ctx(tmp_path):
    """No target_ctx -> legacy standalone-driver order (peptide=A, Zn=B), unchanged."""
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path))
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=None)
    rows = [r for r in path.read_text().splitlines() if "zn_coord" in r]
    assert rows
    for r in rows:
        f = r.split(",")
        assert f[0] == "A" and f[2] == "B"


def test_cyclic_restraints_appends_closure_when_requested(tmp_path):
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path),
                         cli_overrides={"restraint.params": {"closure": True},
                                        "use_pepmlm": False})
    path = c.restraints(cfg, get_case("cyclic"), out_dir=tmp_path, target_ctx=None)
    text = path.read_text()
    assert "covalent" in text and "cyclic_closure" in text


def test_cyclic_referee_is_noop():
    c = Cyclic()
    cfg = resolve_config("cyclic", target_type="metal")
    ref = c.referee(cfg, loop_dir="/tmp", esm_judge=None)
    assert ref("step", 0) is None


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
