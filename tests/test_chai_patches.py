"""CPU tests for the token_dist (CONTACT/COVALENT) restraint patch.

No GPU: a fake ``chai_lab.data.features.generators.token_dist_restraint`` module plus a fake
``chai_lab.utils.tensor_utils.tensorcode_to_string`` stand in for the container. The empirical
"restraint actually applies" check is the gpu-marked probe (see docs/results).

Models the verified container behaviour: D / non-canonical residues are tokenized PER-ATOM
(many atom-tokens sharing one residue_index + residue name); canonical residues are 1 token.
"""
import sys
import types

import numpy as np
import pytest


class _Names:
    """Minimal [n, 8] name table supporting boolean-mask + integer indexing."""
    def __init__(self, names):
        self._names = list(names)  # list[str], one per token

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._names[int(key)]            # tensorcode row -> decoded by fake
        picked = [self._names[i] for i, m in enumerate(np.asarray(key)) if m]
        return picked


def _install_fake_chai(monkeypatch):
    for full in ("chai_lab", "chai_lab.data", "chai_lab.data.features",
                 "chai_lab.data.features.generators", "chai_lab.utils"):
        monkeypatch.setitem(sys.modules, full,
                            sys.modules.get(full) or types.ModuleType(full))
    tu = types.ModuleType("chai_lab.utils.tensor_utils")
    # our fake "tensorcode" row IS already the decoded string.
    tu.tensorcode_to_string = lambda row: row
    monkeypatch.setitem(sys.modules, "chai_lab.utils.tensor_utils", tu)

    mod = types.ModuleType("chai_lab.data.features.generators.token_dist_restraint")

    class TokenDistanceRestraint:
        pass  # the patch installs add_distance_restraint onto this class

    mod.TokenDistanceRestraint = TokenDistanceRestraint
    monkeypatch.setitem(sys.modules,
                        "chai_lab.data.features.generators.token_dist_restraint", mod)
    return mod


class _BoolMask(np.ndarray):
    """ndarray bool that also supports .unsqueeze(dim) like a torch tensor (for outer product)."""
    def unsqueeze(self, dim):
        return np.expand_dims(np.asarray(self), dim).view(_BoolMask)

    def __and__(self, other):
        return np.logical_and(np.asarray(self), np.asarray(other)).view(_BoolMask)


class _IntVec(np.ndarray):
    """ndarray whose ``==`` yields a torch-tensor-like _BoolMask (supports .unsqueeze/&)."""
    def __eq__(self, other):
        return np.equal(np.asarray(self), other).view(_BoolMask)

    def __hash__(self):
        return id(self)


def _ivec(arr):
    return np.asarray(arr).view(_IntVec)


def test_patch_is_idempotent_and_flips_verified(monkeypatch):
    from xenodesign import chai_patches
    monkeypatch.setattr(chai_patches, "_RELAXED", False, raising=False)
    mod = _install_fake_chai(monkeypatch)
    chai_patches._patch_dist_restraint_match()
    chai_patches._patch_dist_restraint_match()  # idempotent: second call a no-op
    assert getattr(mod.TokenDistanceRestraint, "_xeno_dist_match_relaxed", False) is True
    assert chai_patches.dist_restraint_patch_verified() is True


def test_resolve_mask_canonical_single_token(monkeypatch):
    from xenodesign import chai_patches
    _install_fake_chai(monkeypatch)
    # L-peptide-like: 1 token per residue.
    asym = np.array([2, 2]); idx = np.array([1, 3]); names = _Names(["SER", "SER"])
    mask, name = chai_patches._resolve_residue_mask(asym, idx, names,
                                                    residue_asym_id=2, residue_index=1)
    assert name == "SER" and int(np.asarray(mask).sum()) == 1


def test_resolve_mask_d_residue_multi_token_same_name(monkeypatch):
    """The core repair: a per-atom D-His (10 tokens, one residue_index) resolves to a
    10-token mask with the single name DHI instead of raising 'found 10'."""
    from xenodesign import chai_patches
    _install_fake_chai(monkeypatch)
    asym = np.array([1] * 10 + [1]); idx = np.array([1] * 10 + [2])
    names = _Names(["DHI"] * 10 + ["GLY"])
    mask, name = chai_patches._resolve_residue_mask(asym, idx, names,
                                                    residue_asym_id=1, residue_index=1)
    assert name == "DHI"
    assert int(np.asarray(mask).sum()) == 10        # all atom-tokens, narrowed to one residue


def test_resolve_mask_raises_on_truly_ambiguous(monkeypatch):
    from xenodesign import chai_patches
    _install_fake_chai(monkeypatch)
    # two DIFFERENT residue names share the same (asym,index) -> genuinely ambiguous.
    asym = np.array([1, 1]); idx = np.array([1, 1]); names = _Names(["DHI", "DAL"])
    with pytest.raises(AssertionError) as ei:
        chai_patches._resolve_residue_mask(asym, idx, names,
                                           residue_asym_id=1, residue_index=1)
    assert "ambiguous" in str(ei.value)


def test_patched_add_distance_restraint_applies_d_coordinator(monkeypatch):
    """End-to-end on the fake: stock would assert 'found 10' for the D-His; the patched
    function instead writes the restraint over the residue's atom-tokens (matrix changes)."""
    from xenodesign import chai_patches
    monkeypatch.setattr(chai_patches, "_RELAXED", False, raising=False)
    mod = _install_fake_chai(monkeypatch)
    chai_patches._patch_dist_restraint_match()

    # 11 tokens: asym1 D-His (10 atom-tokens, idx1) + asym2 SER (1 token, idx1).
    asym = _ivec([1] * 10 + [2]); idx = _ivec([1] * 10 + [1])
    names = _Names(["DHI"] * 10 + ["SER"])
    mat = np.full((11, 11), -1.0)
    inst = mod.TokenDistanceRestraint()
    out = inst.add_distance_restraint(
        constraint_mat=mat, token_asym_id=asym, token_residue_index=idx,
        token_residue_names=names,
        left_residue_asym_id=1, right_residue_asym_id=2,
        left_residue_index=1, right_residue_index=1,
        left_residue_name="HIS", right_residue_name="SER",   # stock L-mapping; relaxed by patch
        distance_threshold=6.0)
    # the 10 D-His atom-tokens x the 1 SER token cell block is now set to the threshold.
    assert (out[:10, 10] == 6.0).all()
    assert (out[:10, :10] == -1.0).all()                 # nothing else touched


# ── METAL-(b) STEP 2: ATOM-AWARE token_dist (narrow a residue mask to one atom token) ──
#
# A residue tokenized PER-ATOM (a D residue, or a CCD metal/ligand) has one token PER ATOM, all
# sharing the residue's (asym, residue_index). An atom-aware contact (His ND1 <-> Zn) must narrow
# the residue mask down to the SINGLE token that IS that atom. _resolve_residue_mask gains an
# OPTIONAL (token_atom_names, atom_name) pair: when supplied, the mask is intersected with the
# atom-name match; when not, behaviour is unchanged (position/residue-only, back-compat).

class _AtomNames:
    """[n] atom-name table supporting boolean-mask + integer indexing (parallel to _Names)."""
    def __init__(self, names):
        self._names = list(names)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self._names[int(key)]
        return [self._names[i] for i, m in enumerate(np.asarray(key)) if m]


def test_resolve_mask_atom_narrows_to_single_atom_token(monkeypatch):
    """A per-atom His (10 atom-tokens, one residue_index) narrowed by atom 'ND1' selects the ONE
    ND1 atom-token, not all 10."""
    from xenodesign import chai_patches
    _install_fake_chai(monkeypatch)
    asym = np.array([1] * 10 + [2])
    idx = np.array([1] * 10 + [1])
    names = _Names(["DHI"] * 10 + ["LIG"])
    atom_names = _AtomNames(
        ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"] + ["ZN"])
    mask, name = chai_patches._resolve_residue_mask(
        asym, idx, names, residue_asym_id=1, residue_index=1,
        token_atom_names=atom_names, atom_name="ND1")
    assert name == "DHI"
    assert int(np.asarray(mask).sum()) == 1       # narrowed to the single ND1 atom-token
    # and it IS the ND1 token (index 6).
    assert bool(np.asarray(mask)[6]) is True


def test_resolve_mask_no_atom_keeps_residue_level(monkeypatch):
    """No atom supplied -> behaviour unchanged (whole-residue mask, all atom-tokens)."""
    from xenodesign import chai_patches
    _install_fake_chai(monkeypatch)
    asym = np.array([1] * 10 + [2])
    idx = np.array([1] * 10 + [1])
    names = _Names(["DHI"] * 10 + ["LIG"])
    mask, name = chai_patches._resolve_residue_mask(
        asym, idx, names, residue_asym_id=1, residue_index=1)
    assert name == "DHI" and int(np.asarray(mask).sum()) == 10


def test_resolve_mask_metal_atom_narrows_to_zn_token(monkeypatch):
    """The metal side: a CCD metal residue (atom ZN) narrowed by atom 'ZN' selects its token."""
    from xenodesign import chai_patches
    _install_fake_chai(monkeypatch)
    asym = np.array([1, 2])
    idx = np.array([1, 1])
    names = _Names(["DHI", "ZN"])
    atom_names = _AtomNames(["ND1", "ZN"])
    mask, name = chai_patches._resolve_residue_mask(
        asym, idx, names, residue_asym_id=2, residue_index=1,
        token_atom_names=atom_names, atom_name="ZN")
    assert name == "ZN" and int(np.asarray(mask).sum()) == 1
    assert bool(np.asarray(mask)[1]) is True


def test_resolve_mask_atom_not_found_raises(monkeypatch):
    """An atom name that matches no token in the residue fails loudly (genuine error)."""
    from xenodesign import chai_patches
    _install_fake_chai(monkeypatch)
    asym = np.array([1, 1]); idx = np.array([1, 1])
    names = _Names(["DHI", "DHI"]); atom_names = _AtomNames(["N", "CA"])
    with pytest.raises(AssertionError) as ei:
        chai_patches._resolve_residue_mask(
            asym, idx, names, residue_asym_id=1, residue_index=1,
            token_atom_names=atom_names, atom_name="ND1")
    assert "atom" in str(ei.value).lower()


def test_patched_add_distance_restraint_atom_aware_contact(monkeypatch):
    """End-to-end on the fake: an atom-aware contact (His ND1 <-> Zn) writes the restraint into
    the (ND1-token x ZN-token) cell only, not the whole residue block."""
    from xenodesign import chai_patches
    monkeypatch.setattr(chai_patches, "_RELAXED", False, raising=False)
    mod = _install_fake_chai(monkeypatch)
    chai_patches._patch_dist_restraint_match()

    # 11 tokens: asym1 D-His (10 atom-tokens), asym2 ZN (1 atom-token).
    asym = _ivec([1] * 10 + [2]); idx = _ivec([1] * 10 + [1])
    names = _Names(["DHI"] * 10 + ["ZN"])
    atom_names = _AtomNames(
        ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"] + ["ZN"])
    mat = np.full((11, 11), -1.0)
    inst = mod.TokenDistanceRestraint()
    out = inst.add_distance_restraint(
        constraint_mat=mat, token_asym_id=asym, token_residue_index=idx,
        token_residue_names=names,
        left_residue_asym_id=1, right_residue_asym_id=2,
        left_residue_index=1, right_residue_index=1,
        left_residue_name="HIS", right_residue_name="X",
        distance_threshold=2.6,
        token_atom_names=atom_names, left_atom_name="ND1", right_atom_name="ZN")
    # only the (ND1=token6, ZN=token10) cell is set.
    assert out[6, 10] == 2.6
    assert (np.delete(np.delete(out, 6, axis=0), 10, axis=1) == -1.0).all()


def _install_fake_residue_constants(monkeypatch):
    """Minimal chai_lab.data.residue_constants with the L one-letter -> 3-letter table."""
    import types
    for full in ("chai_lab", "chai_lab.data"):
        monkeypatch.setitem(sys.modules, full,
                            sys.modules.get(full) or types.ModuleType(full))
    rc = types.ModuleType("chai_lab.data.residue_constants")
    rc.restype_1to3 = {"C": "CYS", "A": "ALA", "G": "GLY", "H": "HIS"}
    monkeypatch.setitem(sys.modules, "chai_lab.data.residue_constants", rc)
    monkeypatch.setattr(sys.modules["chai_lab.data"], "residue_constants", rc, raising=False)


# ── METAL-(b) STEP 1: feed metals as CCD residues, not SMILES ligands ───────────
#
# WHY current fails: a metal entered as SMILES '[Zn+2]' becomes chai token residue name 'LIG',
# atom 'ZN1'; an atom-aware coordination restraint that references the metal's 'ZN' atom never
# resolves. BUT chai's conformer cache already holds metals as CCD residues (ZN -> atom ZN). If
# we feed the metal as a CCD Residue(name='ZN', smiles=None) the tokenizer hits the cached-conformer
# path and the atom 'ZN' is resolvable. The DECISION ("is this entity name a known CCD code?") is
# the only CPU-testable part — exercised here against a FAKE conformer cache, not real chai.

def test_is_ccd_ligand_code_true_for_cached_metal():
    """A ligand whose code IS present in the conformer cache -> CCD path (True)."""
    from xenodesign.chai_patches import _is_ccd_ligand_code
    cache = {"ZN", "FE", "CU", "NI", "MG", "MN", "CO"}
    fake_get = lambda name: object() if name in cache else None
    assert _is_ccd_ligand_code("ZN", conformer_get=fake_get) is True
    assert _is_ccd_ligand_code("FE", conformer_get=fake_get) is True


def test_is_ccd_ligand_code_false_for_smiles_and_unknown():
    """A real SMILES string (not a CCD code) and an unknown code -> SMILES path (False)."""
    from xenodesign.chai_patches import _is_ccd_ligand_code
    cache = {"ZN"}
    fake_get = lambda name: object() if name in cache else None
    assert _is_ccd_ligand_code("[Zn+2]", conformer_get=fake_get) is False
    assert _is_ccd_ligand_code("CC(=O)O", conformer_get=fake_get) is False  # acetic acid SMILES
    assert _is_ccd_ligand_code("BOGUS", conformer_get=fake_get) is False
    assert _is_ccd_ligand_code("", conformer_get=fake_get) is False
    assert _is_ccd_ligand_code(None, conformer_get=fake_get) is False


def test_is_ccd_ligand_code_case_insensitive_lookup():
    """The cache keys CCD codes uppercase; a lowercase 'zn' entity name still resolves."""
    from xenodesign.chai_patches import _is_ccd_ligand_code
    fake_get = lambda name: object() if name == "ZN" else None
    assert _is_ccd_ligand_code("zn", conformer_get=fake_get) is True


def test_ccd_decision_picks_ccd_residue_over_smiles():
    """The shared per-input decision: a CCD-coded ligand input yields a CCD residue spec
    (name=<CODE>, smiles=None); a genuine SMILES ligand input keeps the SMILES spec."""
    from xenodesign.chai_patches import _ccd_residue_spec_for_ligand
    fake_get = lambda name: object() if name == "ZN" else None
    # CCD-coded ligand (entity_name == sequence == 'ZN'): build a CCD residue.
    spec = _ccd_residue_spec_for_ligand(entity_name="zn", sequence="ZN", conformer_get=fake_get)
    assert spec == {"name": "ZN", "smiles": None}
    # genuine SMILES ligand: no CCD residue -> None (caller falls back to SMILES path).
    assert _ccd_residue_spec_for_ligand(entity_name="lig", sequence="[Zn+2]",
                                        conformer_get=fake_get) is None


def test_covalent_name_candidates_accepts_d_synonym(monkeypatch):
    """A COVALENT row's one-letter code must match BOTH the L 3-letter name AND its D-CCD form,
    so chai stops dropping D-Cys disulfides / head-to-tail D closures at the name guard."""
    _install_fake_residue_constants(monkeypatch)
    from xenodesign.chai_patches import _covalent_name_candidates

    assert _covalent_name_candidates("C") == {"CYS", "DCY"}
    assert _covalent_name_candidates("A") == {"ALA", "DAL"}
    # An unmapped code falls back to UNK (and UNK has no D synonym).
    assert _covalent_name_candidates("Z") == {"UNK"}


# ── FIX B8: head-to-tail CLOSURE resolves POSITION-ONLY (identity-independent) ──────
#
# The closure covalent bond <Cterm>@C -> <Nterm>@N targets BACKBONE atoms (the carbonyl C of
# the last residue, the amide N of the first), which exist for ANY residue identity. The row's
# one-letter codes come from the SEED termini; during the greedy loop MPNN changes residue 1's
# identity, so the seed-based name (e.g. 'R1') goes stale. A backbone-atom covalent bond must
# therefore resolve by POSITION only — the residue-name guard must be SKIPPED for it, so a stale
# terminus name does not drop/crash the closure.

def test_covalent_match_by_name_skips_backbone_atoms():
    """Backbone atoms (N/C/CA/O) exist for every residue identity -> a covalent bond on them
    resolves POSITION-ONLY (no residue-name guard). Side-chain atoms keep the name guard."""
    from xenodesign.chai_patches import _covalent_match_by_name

    # Head-to-tail closure backbone atoms: name guard OFF (position-only), even with a name.
    assert _covalent_match_by_name("C", "R") is False
    assert _covalent_match_by_name("N", "K") is False
    assert _covalent_match_by_name("CA", "H") is False
    assert _covalent_match_by_name("O", "H") is False
    # Side-chain liganding/disulfide atoms: name guard preserved when a name is present.
    assert _covalent_match_by_name("SG", "C") is True
    assert _covalent_match_by_name("ND1", "H") is True
    # No residue name at all -> nothing to match against (position-only) regardless of atom.
    assert _covalent_match_by_name("SG", "") is False
    assert _covalent_match_by_name("SG", None) is False


def test_closure_resolves_when_terminus_identity_drifts_from_seed():
    """The crux of FIX B8: a head-to-tail closure row built from the SEED termini (e.g. 'R1',
    'H12') must still resolve when MPNN has CHANGED residue 1's identity in the loop (so the
    seed-based name is now STALE). Because the closure uses the backbone N/C atoms,
    _covalent_match_by_name returns False for both ends -> the residue-name guard is skipped, so
    the stale name is irrelevant and the bond resolves by position. (Contrast: a side-chain
    disulfide WOULD still require the name to match.)"""
    from xenodesign.chai_patches import _covalent_match_by_name

    # Closure row built from a seed where residue 1 was 'R' (Arg); MPNN later made it 'D' (Asp).
    seed_n_term_one_letter = "R"   # stale seed identity baked into the closure row
    closure_atom_n = "N"           # backbone amide N of residue 1
    closure_atom_c = "C"           # backbone carbonyl C of residue L
    # Both ends are backbone atoms -> name guard skipped on BOTH ends regardless of the (stale)
    # one-letter code -> resolution does NOT depend on the seed identity matching the live one.
    assert _covalent_match_by_name(closure_atom_n, seed_n_term_one_letter) is False
    assert _covalent_match_by_name(closure_atom_c, "H") is False
    # Sanity: were the closure (wrongly) a side-chain bond, the stale name WOULD gate it.
    assert _covalent_match_by_name("SG", seed_n_term_one_letter) is True
