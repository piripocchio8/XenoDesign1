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
