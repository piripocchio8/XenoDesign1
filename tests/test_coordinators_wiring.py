"""CPU tests for the DECLARATIVE coordinator/scaffold flag wiring (--coord_residues,
--cys_positions): the cyclic seed placing the declared coordinators at the right
positions/chirality, the metal restraint rows referencing them (generalizing past His),
the case-default fall-back when the flag is absent, and the CLI flag plumbing.
"""
from __future__ import annotations

from xenodesign.classes.base import SeedSpec  # noqa: F401  (import base first to avoid cyclic-import)
from xenodesign.classes.cyclic import Cyclic
from xenodesign.benchmark.cases import get_case
from xenodesign.benchmark.restraints import metal_coordination_rows, parse_restraints
from xenodesign.config import resolve_binder_length, resolve_config


def _cyclic_cfg(**ov):
    ov.setdefault("use_pepmlm", False)
    return resolve_config("cyclic", target_type="metal", cli_overrides=ov)


def _coords(spec):
    """Build the stored (pos, one_letter, three_letter, chirality) tuples from a flag string,
    exactly as scripts/design.py._apply_declarative_flags does."""
    from xenodesign.coordinators import parse_coord_residues
    return [(c.pos, c.one_letter, c.three_letter, c.chirality)
            for c in parse_coord_residues(spec)]


# ── seed: declared coordinators placed at the right positions + chirality ──────────

def test_declared_coordinators_drive_seed_positions_and_chirality():
    c = Cyclic()
    coords = _coords("H6,DHI12,H18,DHI24")
    cfg = _cyclic_cfg(**{"restraint.params": {"coord_residues": coords}})
    spec = c.seed(cfg, target_seq=None)
    # His at all four declared positions; chirality from the token FORM (L for H, D for DHI).
    for pos in (6, 12, 18, 24):
        assert spec.one_letter[pos - 1] == "H"
    assert spec.fixed_chirality == {6: "L", 12: "D", 18: "L", 24: "D"}


def test_declared_coordinators_generalize_beyond_his():
    """Cys/Asp donors (S/O) at declared positions — not just His/N."""
    c = Cyclic()
    coords = _coords("DCY4,E9,DAS15")
    cfg = _cyclic_cfg(**{"restraint.params": {"coord_residues": coords}})
    spec = c.seed(cfg, target_seq=None)
    assert spec.one_letter[3] == "C"   # D-Cys -> one-letter C
    assert spec.one_letter[8] == "E"   # L-Glu
    assert spec.one_letter[14] == "D"  # D-Asp -> one-letter D
    assert spec.fixed_chirality == {4: "D", 9: "L", 15: "D"}


def test_declared_coordinators_opt_in_off_when_restraints_off():
    c = Cyclic()
    coords = _coords("H6,DHI12")
    cfg = _cyclic_cfg(restraints_on=False,
                      **{"restraint.params": {"coord_residues": coords}})
    spec = c.seed(cfg, target_seq=None)
    assert spec.fixed_chirality == {}


# ── restraints: rows reference the declared coordinators (and their identities) ────

def test_declared_coordinators_drive_restraint_rows(tmp_path):
    c = Cyclic()
    coords = _coords("H6,DHI12,DCY18")
    cfg = _cyclic_cfg(out_dir=str(tmp_path),
                      **{"restraint.params": {"coord_residues": coords}})
    path = c.restraints(cfg, get_case("cyclic"), tmp_path, target_ctx=None)
    rows = parse_restraints(path)
    # One contact per declared coordinator, at the declared positions, with REAL identities.
    assert len(rows) == 3
    coord_tokens = sorted(r["res_idxA"] for r in rows)
    assert coord_tokens == ["C18", "H12", "H6"]
    assert all(r["connection_type"] == "contact" for r in rows)


def test_metal_rows_declarative_overrides_his_resnums():
    """The (pos, one_letter) coord_residues REPLACE the case his_resnums in the row builder."""
    rows = metal_coordination_rows({
        "metal_chain": "B", "metal_resnum": 1, "his_chain": "A",
        "his_resnums": (6, 12, 18, 24),                  # should be IGNORED
        "coord_residues": [(4, "C"), (9, "D")],          # declarative wins
        "max_distance": 2.6, "confidence": 0.8,
    })
    assert len(rows) == 2
    assert rows[0].split(",")[1] == "C4" and rows[1].split(",")[1] == "D9"


# ── default fall-back: cyclic still works with case defaults when flag absent ──────

def test_cyclic_defaults_unchanged_when_coord_residues_absent(tmp_path):
    c = Cyclic()
    cfg = _cyclic_cfg(out_dir=str(tmp_path))
    spec = c.seed(cfg, target_seq=None)
    # Case-default His-only coordinators (6/12/18/24, L/D/L/D) — unchanged behaviour.
    assert spec.fixed_chirality == {6: "L", 12: "D", 18: "L", 24: "D"}
    assert len(spec.one_letter) == resolve_binder_length(cfg) == 24
    path = c.restraints(cfg, get_case("cyclic"), tmp_path, target_ctx=None)
    rows = parse_restraints(path)
    assert [r["res_idxA"] for r in rows] == ["H6", "H12", "H18", "H24"]
    assert all(r["comment"] == "His-Zn" for r in rows)


# ── seq-update: coordinators threaded so post-design identity/chirality is re-imposed ──

def test_cyclic_seq_update_passes_coordinators_and_frozen(monkeypatch):
    """Cyclic.seq_update must thread the declared coordinators (0-based pos, one_letter) into
    make_alpha_seq_update_fn so the post-design anchor can re-impose the pinned His identity
    (GPU-confirmed bug: fixed_mask alone blanked them to D-Ala). frozen_positions stays too."""
    import xenodesign.classes.alpha as alpha_mod

    captured = {}

    def fake_make(wrapper, **kw):
        captured.update(kw)
        return lambda prediction: "stub"

    monkeypatch.setattr(alpha_mod, "make_alpha_seq_update_fn", fake_make)

    c = Cyclic()
    coords = _coords("H6,DHI12,H18,DHI24")
    cfg = _cyclic_cfg(**{"restraint.params": {"coord_residues": coords}})
    c.seq_update(cfg, wrapper=object(), seed_spec=None, roles=None)

    # coordinators: 0-based position + one-letter identity.
    assert captured["coordinators"] == [(5, "H"), (11, "H"), (17, "H"), (23, "H")]
    # frozen_positions kept (both mark the positions non-designable; harmless overlap).
    assert captured["frozen_positions"] == {5, 11, 17, 23}


# ── CLI flag plumbing (scripts/design.py) ──────────────────────────────────────────

def test_cli_cys_positions_parsed_into_params():
    from scripts.design import _apply_declarative_flags, _parse_args
    a = _parse_args(["--binder_class", "non_alpha", "--cys_positions", "3,7,12,18,22,26"])
    cfg = resolve_config("non_alpha", cli_overrides={"use_pepmlm": False})
    _apply_declarative_flags(cfg, a)
    assert cfg.restraint.params["cys_positions"] == (3, 7, 12, 18, 22, 26)


def test_cli_cys_positions_drive_nonalpha_seed():
    """--cys_positions flows end-to-end: the non_alpha seed places Cys at those positions
    (opt-in, not the old mandatory knottin) and reports them to drive the disulfide rows."""
    from scripts.design import _apply_declarative_flags, _parse_args
    from xenodesign.classes.non_alpha import NonAlpha
    a = _parse_args(["--binder_class", "non_alpha", "--cys_positions", "3,7,12,18,22,26"])
    cfg = resolve_config("non_alpha", cli_overrides={"use_pepmlm": False})
    _apply_declarative_flags(cfg, a)
    spec = NonAlpha().seed(cfg, target_seq="ACDEFGHIKLMNPQRSTVWY")
    assert spec.cys_positions == (3, 7, 12, 18, 22, 26)
    assert all(spec.one_letter[p - 1] == "C" for p in (3, 7, 12, 18, 22, 26))


def test_cli_coord_residues_parsed_into_params():
    from scripts.design import _apply_declarative_flags, _parse_args
    a = _parse_args(["--binder_class", "cyclic", "--coord_residues", "H6,DHI12"])
    cfg = resolve_config("cyclic", target_type="metal",
                         cli_overrides={"use_pepmlm": False})
    _apply_declarative_flags(cfg, a)
    # Stored tuple widened to (pos, one_letter, three_letter, chirality, atom) — atom last
    # for back-compat (omitted @atom defaults per element: His -> ND1).
    assert cfg.restraint.params["coord_residues"] == [
        (6, "H", "HIS", "L", "ND1"), (12, "H", "DHI", "D", "ND1")]


def test_cli_flags_absent_leave_params_untouched():
    from scripts.design import _apply_declarative_flags, _parse_args
    a = _parse_args(["--binder_class", "cyclic"])
    cfg = resolve_config("cyclic", target_type="metal",
                         cli_overrides={"use_pepmlm": False})
    before = dict(cfg.restraint.params)
    _apply_declarative_flags(cfg, a)
    assert cfg.restraint.params == before
    assert "coord_residues" not in cfg.restraint.params
    assert "cys_positions" not in cfg.restraint.params
