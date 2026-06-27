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
                          residue_asym_id, residue_index):
    """Return (mask, residue_name) for the residue at (asym, residue_index).

    Pure logic shared by the patched ``add_distance_restraint`` and the CPU unit test.
    ``token_*`` are 1-D tensors (or array-likes supporting ``==``/``&``/boolean indexing);
    ``token_residue_names`` is ``[n, 8]`` uint8 tensorcode (rows decodable by
    ``tensorcode_to_string``). The mask may select MULTIPLE tokens when the residue is a
    D / non-canonical residue tokenized per-atom — that is fine AS LONG AS every matched
    token decodes to the SAME residue name (one residue, many atoms). Returns the (possibly
    multi-token) mask and the single decoded residue name. Raises ``AssertionError`` if no
    token matches, or if the matched tokens span more than one distinct residue name (a truly
    ambiguous restraint), so genuine errors still fail loudly rather than silently."""
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
    return mask, next(iter(names))


def _patch_dist_restraint_match() -> None:  # pragma: no cover (gpu import)
    """Repair the token_dist (CONTACT/COVALENT) residue match for D/non-canonical coordinators.

    Idempotent. Replaces ``TokenDistanceRestraint.add_distance_restraint`` with a version that
    narrows a per-atom-tokenized residue to its atom-token set (instead of asserting a single
    token, which fails for D-coordinators) and writes the restraint over the left x right
    atom-token outer product, using the actual decoded residue names (so the chirality
    name-guard is moot). Mirrors the position-only intent of ``_patch_pocket_name_check``.
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
                               distance_threshold):
        left_mask, left_actual = _resolve_residue_mask(
            token_asym_id, token_residue_index, token_residue_names,
            left_residue_asym_id, left_residue_index)
        right_mask, right_actual = _resolve_residue_mask(
            token_asym_id, token_residue_index, token_residue_names,
            right_residue_asym_id, right_residue_index)
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
            assert left_residue_idx.numel() > 0 and right_residue_idx.numel() > 0
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


def ensure_patches() -> None:  # pragma: no cover (gpu import)
    """Idempotently install the chai restraint patches needed by the design/restraint paths.

    Installs (1) the POCKET residue-name relaxation (:func:`_patch_pocket_name_check`),
    (2) the CONTACT/COVALENT token-dist residue-match repair (:func:`_patch_dist_restraint_match`,
    which also flips :func:`dist_restraint_patch_verified` so the ``metal`` target gate opens), and
    (3) the COVALENT bond-creation D-residue name match (:func:`_patch_covalent_bond_match`, which
    unblocks D-Cys disulfides + head-to-tail macrocyclization).

    Single entry point shared by ``xenodesign.dispatch.run_design`` (so cyclic's metal coordination
    restraint actually applies through the unified ``scripts/design.py`` path) and by
    ``scripts.run_restrained_batch`` (whose behaviour is unchanged — it installed the same patches
    before). Both calls are no-ops once the patches are in place."""
    _patch_pocket_name_check()
    _patch_dist_restraint_match()
    _patch_covalent_bond_match()
