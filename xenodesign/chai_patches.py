"""Runtime monkeypatches for Chai-1 0.6.1 restraint generators.

Sibling of ``scripts.run_restrained_batch._patch_pocket_name_check`` (which relaxes the
POCKET residue-name assertion to position-only). This module repairs the CONTACT/COVALENT
path (``token_dist_restraint.TokenDistanceRestraint.add_distance_restraint``) so a
coordination restraint that names a D / non-canonical coordinator actually APPLIES instead
of being silently dropped.

ROOT CAUSE (verified on the metal probe, docs/results/2026-06-24-metal-restraint-probe.md):
Chai tokenizes D / non-canonical residues PER-ATOM — a D-His becomes 10 atom-tokens, a D-Ala
5, all sharing the SAME ``token_residue_index`` — whereas canonical residues (GLY, SER, the
whole L-peptide) get exactly ONE token each. Stock ``add_distance_restraint`` builds the
residue mask as ``(token_asym_id == asym) & (token_residue_index == index)`` and asserts
``torch.sum(mask) == 1``. For a D-coordinator that mask selects ALL of the residue's
atom-tokens (e.g. 10), so the assertion fires with ``Expected unique residue but found 10``
and the restraint is silently dropped (job finishes as a FREE predict, ipTM ~0.12). A second
name guard (``HIS`` from the L-mapping vs the tokenized ``DHI``) would also block it.

THE FIX. Replace ``add_distance_restraint`` with a version that (a) recognises a multi-token
residue (all matched tokens share one residue name) and NARROWS to the residue's atom-token
set instead of asserting a single token, and (b) uses the ACTUAL decoded residue name so the
chirality name-guard is moot. The constraint matrix is written over the (left-atoms x
right-atoms) outer product so the WHOLE coordinator residue is restrained toward the partner
token — chemically the intended "this residue contacts that residue". A genuinely ambiguous
restraint (matched tokens span >1 distinct residue name) still fails loudly.

The pure mask/name logic is factored into ``_resolve_residue_mask`` so it can be unit tested
on CPU against a fake module (tests/test_chai_patches.py) with no GPU.
"""
from __future__ import annotations

_RELAXED = False


def _resolve_residue_mask(token_asym_id, token_residue_index, token_residue_names,
                          residue_asym_id, residue_index,
                          token_atom_names=None, atom_name=None):
    """Return (mask, residue_name) for the residue at (asym, residue_index).

    Pure logic shared by the patched ``add_distance_restraint`` and the CPU unit test.
    ``token_*`` are 1-D tensors (or array-likes supporting ``==``/``&``/boolean indexing);
    ``token_residue_names`` is ``[n, 8]`` uint8 tensorcode (rows decodable by
    ``tensorcode_to_string``). The mask may select MULTIPLE tokens when the residue is a
    D / non-canonical residue (or a CCD metal/ligand) tokenized per-atom — that is fine AS LONG AS
    every matched token decodes to the SAME residue name (one residue, many atoms). Returns the
    (possibly multi-token) mask and the single decoded residue name. Raises ``AssertionError`` if no
    token matches, or if the matched tokens span more than one distinct residue name (a truly
    ambiguous restraint), so genuine errors still fail loudly rather than silently.

    ATOM-AWARE narrowing (METAL-(b) STEP 2): when both ``token_atom_names`` (a per-token atom-name
    table, parallel to the token arrays — ``token_atom_names[i]`` is the atom name of token i) AND
    ``atom_name`` are supplied, the residue mask is intersected with the atom-name match so it
    selects the SINGLE atom-token that IS that atom (e.g. His ND1, or the metal's ZN). With no atom
    supplied the mask stays whole-residue (back-compat). An atom name that matches no token within
    the residue fails loudly (a genuine wiring error)."""
    from chai_lab.utils.tensor_utils import tensorcode_to_string

    asym_mask = token_asym_id == residue_asym_id
    index_mask = token_residue_index == residue_index
    mask = asym_mask & index_mask
    n_hit = int(mask.sum())
    assert n_hit >= 1, (
        f"Expected >=1 residue token but found 0 "
        f"(residue_asym_id={residue_asym_id}, residue_index={residue_index}, "
        f"n_asym_match={int(asym_mask.sum())}, n_index_match={int(index_mask.sum())})"
    )
    names = {tensorcode_to_string(token_residue_names[i])
             for i in range(len(mask)) if bool(mask[i])}
    assert len(names) == 1, (
        f"Restraint residue (asym={residue_asym_id}, index={residue_index}) is ambiguous: "
        f"matched {n_hit} tokens spanning residue names {sorted(names)} — refusing to apply."
    )
    residue_name = next(iter(names))
    if token_atom_names is not None and atom_name:
        # Narrow the residue mask to the single token whose atom name IS ``atom_name``.
        atom_hit = [bool(mask[i]) and (token_atom_names[i] == atom_name)
                    for i in range(len(mask))]
        n_atom = sum(1 for h in atom_hit if h)
        assert n_atom >= 1, (
            f"Atom {atom_name!r} not found in residue {residue_name} "
            f"(asym={residue_asym_id}, index={residue_index}) — atom-aware restraint cannot apply."
        )
        # Rebuild the mask as an atom-narrowed mask preserving the original mask's type/ops.
        mask = mask & _atom_name_mask(token_atom_names, atom_name, len(mask), mask)
    return mask, residue_name


def _atom_name_mask(token_atom_names, atom_name, n, like_mask):
    """A boolean mask (same array type/ops as ``like_mask``) of tokens whose atom name == ``atom_name``.

    Built per-element so it works against BOTH the real tensor atom-name table and the fake list-
    backed table used in the CPU unit tests; the result is coerced to the same view as ``like_mask``
    so ``&`` / ``.unsqueeze`` keep working downstream."""
    import numpy as _np
    hits = _np.array([bool(token_atom_names[i] == atom_name) for i in range(n)])
    try:
        return hits.view(type(like_mask))
    except (TypeError, ValueError):
        return hits


def _patch_dist_restraint_match() -> None:  # pragma: no cover (gpu import)
    """Repair the token_dist (CONTACT/COVALENT) residue match for D/non-canonical coordinators.

    Idempotent. Replaces ``TokenDistanceRestraint.add_distance_restraint`` with a version that
    narrows a per-atom-tokenized residue to its atom-token set (instead of asserting a single
    token, which fails for D-coordinators) and writes the restraint over the left x right
    atom-token outer product, using the actual decoded residue names (so the chirality
    name-guard is moot). Mirrors the position-only intent of ``_patch_pocket_name_check``.

    METAL-(b) STEP 2: the replacement ALSO accepts an OPTIONAL per-token atom-name table
    (``token_atom_names``) + a per-side atom name (``left_atom_name``/``right_atom_name``). When
    supplied, the side's mask is narrowed to that single atom token (His ND1, metal ZN) for a real
    atom-specific coordination contact; when omitted, behaviour is unchanged (whole-residue mask).
    The narrowing logic is CPU-tested against fakes (tests/test_chai_patches.py).

    GPU-INTEGRATION BOUNDARY (flagged for Marco): the stock chai CONTACT path
    (``restraint_context.load_manual_restraints_for_chai1`` -> ``RestraintGroup`` ->
    ``generate_from_restraint``) DROPS the parsed ``atom_nameA/atom_nameB`` and never passes a
    per-token atom table into ``add_distance_restraint`` — so the new atom kwargs are not yet
    fed by the live batch. Wiring them (preserve atoms in the ContactRestraint; build the token
    atom-name table from the batch atom-level fields ``atom_ref_name``/``token_ref_atom_index``
    and forward them) touches chai batch internals not exercisable on CPU; it is left for GPU
    validation. Until wired, an atom-aware CONTACT file (STEP 3) still applies correctly at the
    RESIDUE level via the existing per-atom narrowing (the atom only TIGHTENS the contact).
    """
    global _RELAXED
    from chai_lab.data.features.generators import token_dist_restraint as M

    cls = M.TokenDistanceRestraint
    if getattr(cls, "_xeno_dist_match_relaxed", False):
        _RELAXED = True
        return

    def add_distance_restraint(self, *, constraint_mat, token_asym_id,
                               token_residue_index, token_residue_names,
                               left_residue_asym_id, right_residue_asym_id,
                               left_residue_index, right_residue_index,
                               right_residue_name, left_residue_name,
                               distance_threshold,
                               token_atom_names=None, left_atom_name=None,
                               right_atom_name=None):
        # ATOM-AWARE narrowing (METAL-(b) STEP 2): when a per-token atom-name table + a side's
        # atom name are supplied, narrow that side's mask to the single atom token (His ND1, metal
        # ZN, ...). Omitted -> whole-residue mask (back-compat, the D-coordinator repair path).
        left_mask, left_actual = _resolve_residue_mask(
            token_asym_id, token_residue_index, token_residue_names,
            left_residue_asym_id, left_residue_index,
            token_atom_names=token_atom_names, atom_name=left_atom_name)
        right_mask, right_actual = _resolve_residue_mask(
            token_asym_id, token_residue_index, token_residue_names,
            right_residue_asym_id, right_residue_index,
            token_atom_names=token_atom_names, atom_name=right_atom_name)
        n_left, n_right = int(left_mask.sum()), int(right_mask.sum())
        if n_left > 1 or n_right > 1 or left_actual != left_residue_name \
                or right_actual != right_residue_name:
            print(f"[patch] token_dist repaired: left {left_residue_name!r}->{left_actual!r}"
                  f"({n_left} atom-tokens), right {right_residue_name!r}->{right_actual!r}"
                  f"({n_right} atom-tokens)", flush=True)
        # Restrain the WHOLE coordinator residue's atom-tokens toward the partner residue's
        # atom-tokens (outer product). For a canonical 1-token residue this is identical to
        # the stock single-cell write; for a per-atom D residue it pins every atom-token.
        constraint_mat[left_mask.unsqueeze(-1) & right_mask.unsqueeze(0)] = distance_threshold
        return constraint_mat

    cls.add_distance_restraint = add_distance_restraint
    cls._xeno_dist_match_relaxed = True
    _RELAXED = True
    print("[patch] token_dist residue-match repaired for D/non-canonical coordinators "
          "(per-atom narrowing + name relaxed to position-only).", flush=True)


def dist_restraint_patch_verified() -> bool:
    """True once :func:`_patch_dist_restraint_match` has installed in this process.

    ``xenodesign.targets`` consults this to gate the ``metal`` target type closed until the
    coordination-restraint patch is active."""
    return _RELAXED


def _patch_pocket_name_check() -> None:  # pragma: no cover (gpu)
    """Disable chai's POCKET residue-NAME assertion so the restraint applies BY POSITION only.

    chai's ``add_pocket_restraint`` asserts the tokenized residue name at the named target
    position equals the one-letter->L-3-letter name from the .restraints file. Our pocket files
    are register-agnostic and name only TARGET positions, but a D-peptide target tokenizes to the
    D-CCD name (``DGL`` != ``GLU``) and a composition-preserving scramble reorders residues, so
    the name guard fires even though pinning the binder to those exact target POSITIONS is the
    intent. We rebind ``pocket_token_residue_name`` to the ACTUAL FASTA token before the guard so
    it passes trivially; the uniqueness/position assertion (the one that matters) is preserved.

    Extracted verbatim from ``scripts.run_restrained_batch`` so it can be installed off the shared
    :func:`ensure_patches` dispatch path; ``run_restrained_batch`` now delegates here. Idempotent.
    """
    from chai_lab.data.features.generators import token_pair_pocket_restraint as M

    cls = M.TokenPairPocketRestraint
    if getattr(cls, "_xeno_name_check_relaxed", False):
        return
    orig = cls.add_pocket_restraint

    def add_pocket_restraint(self, restraint_mat, token_asym_id, token_residue_index,
                             token_residue_names, *, pocket_chain_asym_id,
                             pocket_token_asym_id, pocket_token_residue_index,
                             pocket_token_residue_name, pocket_distance_threshold):
        from einops import rearrange

        from chai_lab.data.features.generators.token_pair_pocket_restraint import (
            tensorcode_to_string,
        )

        mask = (token_residue_index == pocket_token_residue_index)
        mask = mask & (token_asym_id == pocket_token_asym_id)
        actual = tensorcode_to_string(rearrange(token_residue_names[mask], "1 l -> l"))
        return orig(self, restraint_mat, token_asym_id, token_residue_index,
                    token_residue_names, pocket_chain_asym_id=pocket_chain_asym_id,
                    pocket_token_asym_id=pocket_token_asym_id,
                    pocket_token_residue_index=pocket_token_residue_index,
                    pocket_token_residue_name=actual,
                    pocket_distance_threshold=pocket_distance_threshold)

    cls.add_pocket_restraint = add_pocket_restraint
    cls._xeno_name_check_relaxed = True
    print("[patch] POCKET residue-name check relaxed to position-only.", flush=True)


def _covalent_name_candidates(one_letter):
    """Acceptable token residue names for a COVALENT row's one-letter code: the L 3-letter name
    AND its D-CCD counterpart (CYS->{CYS,DCY}, ALA->{ALA,DAL}).

    Pure logic shared by the patched bond builder and the CPU unit test. chai's
    ``get_atom_covalent_bond_pairs_from_constraints`` maps the one-letter via the L-only
    ``rc.restype_1to3`` and matches ``token_residue_name == <L name>`` exactly, so a D-residue
    (stored under its D-CCD name) never matches and the bond is dropped at the
    ``left_residue_idx.numel() > 0`` assert. Accepting the D synonym lifts that for D-Cys
    disulfides + head-to-tail mainchain closure (verified on GPU, docs/results 2026-06-24)."""
    from chai_lab.data import residue_constants as rc
    from xenodesign.mirror import L_TO_D

    l3 = rc.restype_1to3.get(one_letter, "UNK")
    candidates = {l3}
    if l3 in L_TO_D:
        candidates.add(L_TO_D[l3])
    return candidates


# Backbone heavy atoms present for EVERY residue identity, regardless of L/D or side chain.
# A covalent bond on these (the head-to-tail closure C->N) is identity-independent, so its
# residue-NAME guard must be skipped: it resolves by POSITION only.
_COVALENT_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


def _covalent_match_by_name(atom_name, residue_one_letter) -> bool:
    """Whether a COVALENT bond endpoint should still match the residue NAME, or resolve by
    POSITION only (FIX B8).

    Pure logic shared by the patched bond builder and the CPU unit test. The head-to-tail
    CLOSURE bond (``<Cterm>@C -> <Nterm>@N``) is built from the SEED termini one-letter codes;
    during the greedy loop MPNN changes residue 1's identity, so the seed-based name goes STALE.
    But the closure targets BACKBONE atoms (carbonyl C of the last residue, amide N of the
    first), which exist for ANY residue identity — so its resolution must NOT depend on the
    (now-stale) residue name. Returns False (skip the name guard, resolve by position) when the
    atom is a backbone atom, OR when no residue name is supplied. Side-chain atoms (SG, ND1, ...)
    keep the name guard (return True) so disulfides / side-chain bonds stay pinned to the right
    residue identity."""
    if not residue_one_letter:
        return False
    return atom_name not in _COVALENT_BACKBONE_ATOMS


def _patch_covalent_bond_match() -> None:  # pragma: no cover (gpu import)
    """Let chai 0.6.1 create COVALENT bonds whose endpoints are D-residues (idempotent).

    Replaces ``bond_utils.get_atom_covalent_bond_pairs_from_constraints`` (and re-points the
    by-name import in ``chai_lab.chai1``) with a version whose residue-name guard accepts the
    D-CCD synonym via :func:`_covalent_name_candidates`. Every position/atom-name assert is
    preserved, so the bond is still pinned to exactly one atom pair. Unblocks D-Cys SG-SG
    disulfides and N-to-C head-to-tail macrocyclization (GPU-verified: SG-SG closed to 1.77 A)."""
    import torch
    from einops import rearrange
    import chai_lab.data.dataset.structure.bond_utils as BU
    import chai_lab.chai1 as C
    from chai_lab.data.parsing.restraints import PairwiseInteractionType
    from chai_lab.model.utils import get_asym_id_from_subchain_id
    from chai_lab.utils.tensor_utils import string_to_tensorcode

    if getattr(BU, "_xeno_covalent_d_relaxed", False):
        return

    def _name_mask(one_letter, token_residue_name):
        width = token_residue_name.shape[-1]
        m = torch.zeros(token_residue_name.shape[0], dtype=torch.bool,
                        device=token_residue_name.device)
        for nm in _covalent_name_candidates(one_letter):
            tc = string_to_tensorcode(nm, pad_to_length=width).to(token_residue_name.device)
            m = m | (token_residue_name == rearrange(tc, "d -> 1 d")).all(dim=-1)
        return m

    def get_atom_covalent_bond_pairs_from_constraints(
        provided_constraints, token_residue_index, token_residue_name,
        token_subchain_id, token_asym_id, atom_token_index, atom_ref_name,
    ):
        ret_a, ret_b = [], []
        for constraint in provided_constraints:
            ctype = constraint.connection_type
            if ctype != PairwiseInteractionType.COVALENT:
                if ctype in (PairwiseInteractionType.CONTACT,
                             PairwiseInteractionType.POCKET):
                    continue
                raise ValueError(f"Unrecognized pariwise interaction: {ctype}")
            assert constraint.atom_nameA and constraint.atom_nameB, \
                "Atoms must be provided for covalent bonds"
            left_asym = get_asym_id_from_subchain_id(
                subchain_id=constraint.chainA, source_pdb_chain_id=token_subchain_id,
                token_asym_id=token_asym_id)
            right_asym = get_asym_id_from_subchain_id(
                subchain_id=constraint.chainB, source_pdb_chain_id=token_subchain_id,
                token_asym_id=token_asym_id)
            left_asym_mask = token_asym_id == left_asym
            right_asym_mask = token_asym_id == right_asym
            assert torch.any(left_asym_mask) and torch.any(right_asym_mask)
            left_idx_mask = token_residue_index == constraint.res_idxA_pos - 1
            right_idx_mask = token_residue_index == constraint.res_idxB_pos - 1
            assert torch.any(left_idx_mask) and torch.any(right_idx_mask)
            # FIX B8: for a BACKBONE-atom covalent bond (the head-to-tail closure C->N), resolve
            # by POSITION only — skip the residue-name guard. The closure row carries the SEED
            # termini one-letter codes, which go stale once the loop's MPNN changes a terminus
            # identity; backbone N/C exist for any identity, so the name must be irrelevant.
            # Side-chain bonds (SG disulfides, ND1 coordination) still match the name.
            left_residue_mask = left_asym_mask & left_idx_mask
            if _covalent_match_by_name(constraint.atom_nameA, constraint.res_idxA_name):
                left_residue_mask &= _name_mask(constraint.res_idxA_name, token_residue_name)
            right_residue_mask = right_asym_mask & right_idx_mask
            if _covalent_match_by_name(constraint.atom_nameB, constraint.res_idxB_name):
                right_residue_mask &= _name_mask(constraint.res_idxB_name, token_residue_name)
            left_residue_idx = torch.where(left_residue_mask)[0]
            right_residue_idx = torch.where(right_residue_mask)[0]
            # GRACEFUL DEGRADE (B6 sibling): a covalent endpoint that resolves to ZERO tokens must
            # NOT crash the whole predict. This happens when a DECLARED coordinator drifts during the
            # loop (e.g. the C-term His is rewritten to Gly), so the side-chain name guard filters out
            # every token and the row references a residue/atom that no longer exists. Instead of the
            # hard assert, SKIP that one bond (warn, naming the unresolved constraint) and keep the
            # remaining resolvable bonds — a single drifted coordinator degrades gracefully. Both
            # sides resolving is unchanged (the bond is built exactly as before).
            if left_residue_idx.numel() == 0 or right_residue_idx.numel() == 0:
                print(f"[patch] WARNING: covalent bond skipped — unresolved endpoint "
                      f"({constraint.chainA}/{constraint.res_idxA_pos}@{constraint.atom_nameA} -> "
                      f"{constraint.chainB}/{constraint.res_idxB_pos}@{constraint.atom_nameB}): "
                      f"left {int(left_residue_idx.numel())} token(s), "
                      f"right {int(right_residue_idx.numel())} token(s). "
                      f"A declared coordinator likely drifted; continuing with remaining bonds.",
                      flush=True)
                continue
            left_atoms_mask = torch.isin(atom_token_index, test_elements=left_residue_idx)
            right_atoms_mask = torch.isin(atom_token_index, test_elements=right_residue_idx)
            assert torch.any(left_atoms_mask) and torch.any(right_atoms_mask)
            left_name_mask = torch.tensor([n == constraint.atom_nameA for n in atom_ref_name],
                                          dtype=torch.bool)
            right_name_mask = torch.tensor([n == constraint.atom_nameB for n in atom_ref_name],
                                           dtype=torch.bool)
            left_atom_mask = left_atoms_mask & left_name_mask
            right_atom_mask = right_atoms_mask & right_name_mask
            assert torch.sum(left_atom_mask) == torch.sum(right_atom_mask) == 1, \
                f"Expect single atoms, got {torch.sum(left_atom_mask)}, {torch.sum(right_atom_mask)}"
            (la,) = torch.where(left_atom_mask)
            (rb,) = torch.where(right_atom_mask)
            ret_a.append(la.item())
            ret_b.append(rb.item())
        return (torch.tensor(ret_a, dtype=torch.long),
                torch.tensor(ret_b, dtype=torch.long))

    BU.get_atom_covalent_bond_pairs_from_constraints = get_atom_covalent_bond_pairs_from_constraints
    C.get_atom_covalent_bond_pairs_from_constraints = get_atom_covalent_bond_pairs_from_constraints
    BU._xeno_covalent_d_relaxed = True
    print("[patch] COVALENT bond residue-name match relaxed to accept D-CCD names "
          "(L name OR L_TO_D[L]).", flush=True)


# ── METAL-(b) STEP 1: CCD metal feeding (real atom-specific coordination) ───────
#
# A metal fed as SMILES '[Zn+2]' tokenizes to residue name 'LIG', atom 'ZN1' — neither of which a
# coordination restraint that names the metal atom 'ZN' can resolve. chai's conformer cache already
# holds metals (and many cofactors) as CCD residues whose code IS the residue name and whose atom is
# the bare element (ZN -> atom ZN). Feeding the metal as a CCD Residue(name='ZN', smiles=None) takes
# the tokenizer's cached-conformer path (_get_ref_conformer_data branch 1), giving a resolvable
# residue+atom. We detect "this ligand entity is a known CCD code" via the conformer cache and, when
# so, build a CCD residue instead of a SMILES ligand. Anything not in the cache (a real SMILES) keeps
# the SMILES path. Decision logic is pure + CPU-tested (test_chai_patches); the monkeypatch wiring is
# gpu-only (needs the real chai tokenizer / cache).


def _is_ccd_ligand_code(name, conformer_get) -> bool:
    """True iff ``name`` is a CCD residue code present in chai's conformer cache.

    Pure decision shared by the CCD-feeding patch and its CPU unit test. ``conformer_get`` is a
    callable ``code -> ConformerData | None`` (the real ``RefConformerGenerator.get``; a fake in
    tests). A real SMILES string (e.g. '[Zn+2]', 'CC(=O)O') is never a CCD code, so it returns
    False and the caller keeps the SMILES path. The CCD cache keys codes UPPERCASE, so the lookup
    is case-insensitive (a lowercase entity name 'zn' still resolves to 'ZN'). Empty/None -> False.
    """
    if not name:
        return False
    return conformer_get(str(name).upper()) is not None


def _ccd_residue_spec_for_ligand(entity_name, sequence, conformer_get) -> dict | None:
    """Return ``{'name': <CODE>, 'smiles': None}`` when a ligand input is a known CCD code, else None.

    A ligand is fed CCD-style when EITHER its sequence line OR its entity name is a CCD code in the
    conformer cache (io_spec emits ``>ligand|name=ZN`` with the sequence line also ``ZN``). The
    sequence is checked first (it is the SMILES slot for genuine SMILES ligands; a CCD code there is
    unambiguous), then the entity name. Returns None for a genuine SMILES ligand so the caller falls
    back to the stock SMILES path. The returned name is UPPERCASED to the canonical CCD code."""
    for candidate in (sequence, entity_name):
        if _is_ccd_ligand_code(candidate, conformer_get):
            return {"name": str(candidate).upper(), "smiles": None}
    return None


def _patch_ligand_ccd_feeding() -> None:  # pragma: no cover (gpu import)
    """Build a CCD-coded ligand as a cached-conformer CCD residue instead of a SMILES ligand.

    Idempotent. Wraps ``inference_dataset.raw_inputs_to_entitites_data`` so that, for each LIGAND
    input whose code resolves in the conformer cache (:func:`_ccd_residue_spec_for_ligand`), the
    input's ``sequence`` is rewritten to the CCD code AND a one-off tokenizer-friendly CCD residue
    is produced via the cached-conformer path (Residue(name=<CODE>, smiles=None)). The simplest safe
    interception: monkeypatch ``get_lig_residues`` is NOT enough (it has no access to the entity
    name), so we instead wrap the top-level function and post-process LIGAND entities whose single
    residue should be a CCD residue. Non-CCD ligands (real SMILES) are untouched.

    The decision logic is unit-tested on CPU against a fake cache; this wiring is GPU-validated by
    Marco (needs the real conformer cache + tokenizer)."""
    import chai_lab.data.dataset.inference_dataset as ID
    from chai_lab.data.parsing.structure.entity_type import EntityType
    from chai_lab.data.parsing.structure.residue import Residue, get_restype
    from chai_lab.data.residue_constants import residue_types_with_nucleotides_order
    from chai_lab.data.sources.rdkit import RefConformerGenerator

    if getattr(ID, "_xeno_ccd_ligand_feeding", False):
        return

    orig = ID.raw_inputs_to_entitites_data
    # One shared generator for the cache lookup (cheap; reused across calls).
    _gen = {"g": None}

    def _conformer_get(code):
        if _gen["g"] is None:
            _gen["g"] = RefConformerGenerator()
        return _gen["g"].get(code)

    def _ccd_lig_residues(code):
        """A single CCD ligand residue taking the cached-conformer tokenizer path."""
        return [Residue(
            name=code, label_seq=0,
            restype=residue_types_with_nucleotides_order.get("X", 0),
            residue_index=0, is_missing=False, b_factor_or_plddt=0.0,
            conformer_data=None, smiles=None,
        )]

    def raw_inputs_to_entitites_data(inputs, *args, **kwargs):
        # Rewrite any CCD-coded ligand input's sequence to its CCD code so the stock entity-id
        # bookkeeping (which keys ligands by their sequence) stays coherent, then let the original
        # build everything; finally swap the SMILES residue for a CCD residue on those entities.
        ccd_codes: dict[int, str] = {}
        for i, inp in enumerate(inputs):
            if inp.entity_type == EntityType.LIGAND.value:
                spec = _ccd_residue_spec_for_ligand(inp.entity_name, inp.sequence,
                                                    _conformer_get)
                if spec is not None:
                    ccd_codes[i] = spec["name"]
        entities = orig(inputs, *args, **kwargs)
        for i, code in ccd_codes.items():
            entities[i].residues[:] = _ccd_lig_residues(code)
            entities[i].full_sequence[:] = [code]
            print(f"[patch] ligand {entities[i].entity_name!r} fed as CCD residue "
                  f"{code!r} (cached conformer; atom-resolvable).", flush=True)
        return entities

    ID.raw_inputs_to_entitites_data = raw_inputs_to_entitites_data
    ID._xeno_ccd_ligand_feeding = True
    print("[patch] ligand CCD feeding installed (CCD-coded ligands -> cached-conformer residue).",
          flush=True)


def ensure_patches() -> None:  # pragma: no cover (gpu import)
    """Idempotently install the chai restraint patches needed by the design/restraint paths.

    Installs (1) the POCKET residue-name relaxation (:func:`_patch_pocket_name_check`),
    (2) the CONTACT/COVALENT token-dist residue-match repair (:func:`_patch_dist_restraint_match`,
    which also flips :func:`dist_restraint_patch_verified` so the ``metal`` target gate opens), and
    (3) the COVALENT bond-creation D-residue name match (:func:`_patch_covalent_bond_match`, which
    unblocks D-Cys disulfides + head-to-tail macrocyclization), and (4) CCD metal feeding
    (:func:`_patch_ligand_ccd_feeding`, so a metal entered as a CCD code tokenizes to a resolvable
    residue+atom for atom-aware coordination).

    Single entry point shared by ``xenodesign.dispatch.run_design`` (so cyclic's metal coordination
    restraint actually applies through the unified ``scripts/design.py`` path) and by
    ``scripts.run_restrained_batch`` (whose behaviour is unchanged — it installed the same patches
    before). Both calls are no-ops once the patches are in place."""
    _patch_pocket_name_check()
    _patch_dist_restraint_match()
    _patch_covalent_bond_match()
    _patch_ligand_ccd_feeding()
