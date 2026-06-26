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
