"""target_entities(cfg) — entity lists per target_type + metal gating (T4, spec §5/§6)."""
import pytest

from xenodesign.config import resolve_config, TargetSpec
from xenodesign import targets


def _cfg(**tk):
    c = resolve_config("alpha", target_type=tk.pop("target_type", "protein"))
    for k, v in tk.items():
        setattr(c.target, k, v)
    return c


def test_protein_single_chain(tmp_path):
    fa = tmp_path / "t.fasta"
    fa.write_text(">A\nACDEFG\n")
    ents, msa_dir, _ = targets.target_entities(
        _cfg(target_type="protein", fasta_path=str(fa), chains=("A",)))
    assert len(ents) == 1
    assert ents[0]["type"] == "protein" and ents[0]["chirality"] == "L"
    assert ents[0]["sequence"] == "ACDEFG"
    assert msa_dir is None


def test_protein_multichain_with_msa(tmp_path, monkeypatch):
    monkeypatch.setattr(targets, "load_ha_entities",
                        lambda p: [{"type": "protein", "name": "HA1", "sequence": "AA", "chirality": "L"},
                                   {"type": "protein", "name": "HA2", "sequence": "CC", "chirality": "L"}])
    c = _cfg(target_type="protein", fasta_path="x", chains=("HA1", "HA2"),
             msa=True, msa_dir="some/msas")
    ents, msa_dir, _ = targets.target_entities(c)
    assert len(ents) == 2 and msa_dir == "some/msas"


def test_metal_emits_ligand_and_gate(monkeypatch):
    c = resolve_config("cyclic", target_type="metal")     # smiles '[Zn+2]'
    # gate: refuses until the dist-restraint patch is verified-applied
    monkeypatch.setattr(targets, "_metal_patch_verified", lambda: False)
    with pytest.raises(RuntimeError) as ei:
        targets.target_entities(c)
    assert "metal" in str(ei.value).lower() and "patch" in str(ei.value).lower()
    monkeypatch.setattr(targets, "_metal_patch_verified", lambda: True)
    ents, _md, hint = targets.target_entities(c)
    assert any(e["type"] == "ligand" for e in ents)
    assert hint == "metal_coordination"


def test_metal_feeds_ccd_residue_by_default(monkeypatch):
    """METAL-(b) STEP 1: the metal enters as a CCD residue (metal_ccd='ZN'), NOT SMILES — the
    cyclic preset's default '[Zn+2]' SMILES maps to the CCD code ZN so chai tokenizes a resolvable
    residue+atom for atom-aware coordination."""
    c = resolve_config("cyclic", target_type="metal")     # preset smiles '[Zn+2]'
    monkeypatch.setattr(targets, "_metal_patch_verified", lambda: True)
    ents, _md, hint = targets.target_entities(c)
    lig = next(e for e in ents if e["type"] == "ligand")
    assert lig.get("metal_ccd") == "ZN"
    assert "smiles" not in lig
    assert hint == "metal_coordination"


def test_metal_explicit_ccd_code_used(monkeypatch):
    """An explicit CCD metal code (target.ccd='FE') is fed as that CCD residue."""
    c = resolve_config("cyclic", target_type="metal")
    c.target.smiles = ""
    c.target.ccd = "FE"
    monkeypatch.setattr(targets, "_metal_patch_verified", lambda: True)
    ents, _md, _hint = targets.target_entities(c)
    lig = next(e for e in ents if e["type"] == "ligand")
    assert lig.get("metal_ccd") == "FE"


def test_metal_non_ccd_smiles_falls_back_to_smiles(monkeypatch):
    """A non-CCD metal/SMILES (an arbitrary SMILES with no CCD mapping) keeps the SMILES path."""
    c = resolve_config("cyclic", target_type="metal")
    c.target.smiles = "[Pt+2]"        # no CCD mapping in our metal table -> SMILES fallback
    c.target.ccd = ""
    monkeypatch.setattr(targets, "_metal_patch_verified", lambda: True)
    ents, _md, _hint = targets.target_entities(c)
    lig = next(e for e in ents if e["type"] == "ligand")
    assert lig.get("smiles") == "[Pt+2]"
    assert "metal_ccd" not in lig


def test_alpha_empty_fasta_falls_back_to_case_default(monkeypatch):
    """T10 fix 1: alpha with no explicit FASTA resolves the case default target record
    (alpha._TARGET_RECORD) instead of reading Path('') (IsADirectoryError)."""
    import dataclasses

    from xenodesign.benchmark import cases
    fake_case = dataclasses.replace(cases.get_case("alpha"), fasta_path="DEFAULT.fasta")
    monkeypatch.setattr(targets, "_case_for", lambda cfg: fake_case)
    captured = {}

    def _fake_read(path, name=None):
        captured["path"], captured["name"] = path, name
        return "TARGETSEQ"
    monkeypatch.setattr("xenodesign.seed.read_target_sequence", _fake_read)

    c = resolve_config("alpha", target_type="protein")   # preset fasta_path == ""
    assert c.target.fasta_path == ""
    ents, _md, _ = targets.target_entities(c)
    assert ents[0]["sequence"] == "TARGETSEQ" and ents[0]["chirality"] == "L"
    assert captured["path"] == "DEFAULT.fasta"
    assert captured["name"] == "trimer_DL_ABLE_B"      # alpha._TARGET_RECORD


def test_non_alpha_default_routes_to_multichain_ha_msa(monkeypatch):
    """T10 fix 2: non_alpha (preset msa=True, chains=()) routes to the 2-chain HA target via
    load_ha_entities + the default MSA dir without an explicit fasta."""
    calls = {}
    monkeypatch.setattr(targets, "load_ha_entities",
                        lambda p: (calls.setdefault("fasta", p),
                                   [{"type": "protein", "name": "HA1", "sequence": "AA",
                                     "chirality": "L"},
                                    {"type": "protein", "name": "HA2", "sequence": "CC",
                                     "chirality": "L"}])[1])
    c = resolve_config("non_alpha", target_type="protein")
    assert c.target.msa is True and c.target.chains == () and c.target.fasta_path == ""
    ents, msa_dir, _ = targets.target_entities(c)
    assert len(ents) == 2
    assert calls["fasta"] == targets._DEFAULT_HA_FASTA
    assert msa_dir == targets._DEFAULT_MSA_DIR


def test_none_target_is_binder_only():
    """target_type='none' -> no target entity (free cyclic/linear peptide; binder = chain A)."""
    c = resolve_config("cyclic", target_type="none")
    ents, msa_dir, hint = targets.target_entities(c)
    assert ents == [] and msa_dir is None and hint is None


def test_unknown_target_type_rejected():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        resolve_config("cyclic", target_type="bogus")


def test_rna_entity(tmp_path):
    fa = tmp_path / "r.fasta"
    fa.write_text(">R\nACGU\n")
    ents, _md, _ = targets.target_entities(_cfg(target_type="rna", fasta_path=str(fa), chains=("R",)))
    assert ents[0]["type"] == "rna" and ents[0]["sequence"] == "ACGU"
