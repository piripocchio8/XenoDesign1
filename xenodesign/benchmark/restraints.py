"""Chai-1 0.6.1 .restraints CSV builders (spec §7 #27).

Verified against the chai-lab 0.6.1 parser
(``chai_lab/data/parsing/restraints.py`` ::PairwiseConstraintDataframeModel) and the
bundled example files (``/chai-lab/examples/restraints/{contact,pocket}.restraints``).

Column set (header order matches chai's own example files; the parser reads by NAME via
``pd.read_csv`` so order is not load-bearing, but we keep it identical for clarity):

    chainA,res_idxA,chainB,res_idxB,connection_type,confidence,
    min_distance_angstrom,max_distance_angstrom,comment,restraint_id

Schema facts that drive the format (all from the 0.6.1 source):
  * ``restraint_id`` must be a UNIQUE non-null string; ``chainA``/``chainB`` non-null.
  * distances are ``min_distance_angstrom`` / ``max_distance_angstrom`` (NOT the old
    ``min_distance``/``max_distance`` — that was the SchemaError we are fixing).
  * ``confidence`` in [0, 1] (parser: ``ge=0.0, le=1.0``; nullable -> defaults to 1.0).
  * ``connection_type`` is one of the ``PairwiseInteractionType`` enum VALUES:
    ``'covalent'`` | ``'contact'`` | ``'pocket'``.
  * ``res_idx`` is parsed as ``res = res_idx[0:]`` split on '@'; for the no-atom form
    ``name = res_idx[0]`` (a SINGLE residue ONE-LETTER code) and ``pos = int(res_idx[1:])``
    (1-based). So a residue token is ``'<one-letter><pos>'`` e.g. ``'H3'`` (His at 3).
    The leading char is the residue IDENTITY, not the chain id — chai looks it up in
    ``residue_constants.restype_1to3_with_x`` (``KeyError`` for non-residue chars like a
    chain letter 'B'; ``''`` is also not a key). When the designed identity is unknown at
    build time we emit ``'X'`` (=> 'UNK', the only safe wildcard the table accepts).
    NOTE: for a CONTACT, the GPU run additionally asserts the one-letter code MATCHES the
    actual token residue name; 'X'/UNK is correct only for genuinely-unknown designed
    positions. Known identities (e.g. coordinating His) are emitted with their real code.
  * POCKET: A is the CHAIN-level side -> ``res_idxA`` MUST be empty (``''``); B is the
    TOKEN-level side -> ``res_idxB`` is a residue token. (Parser ``__post_init__`` asserts
    ``res_idxA == ''`` and ``res_idxB != ''`` for pocket.)
  * CONTACT: BOTH endpoints must specify a token (parser asserts ``res_idxA``/``res_idxB``
    are non-empty). Used for "two specific residues in contact".

Connection-type choice per builder:
  * metal_coordination / contact -> ``contact`` (two specific tokens; both identities must
    match the real structure residue names — only safe when both are KNOWN).
  * pin_polarity / pocket -> ``pocket`` (binder chain-level A, FIXED target token B). The
    pin uses a pocket precisely because the binder anchor identity is DESIGNED/unknown, so a
    contact's two-sided identity assertion would crash; the pocket only asserts the fixed
    target token's identity (#27 crash fix).

All restraints here are INTER-chain (Chai is not trained on intra-chain bonds; head-to-tail
/disulfide closure is cyclization #23, out of scope). These functions emit ROWS and write
FILES only; they NEVER call chai (CPU-testable round-trip).
"""
from __future__ import annotations

from pathlib import Path

RESTRAINT_HEADER = (
    'chainA,res_idxA,chainB,res_idxB,connection_type,'
    'confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id'
)

# Safe wildcard one-letter residue code: chai maps 'X' -> 'UNK' in restype_1to3_with_x.
# Used when the residue identity at a token is unknown at restraint-build time.
UNKNOWN_RES = 'X'


def _check_confidence(confidence: float) -> None:
    # chai schema: confidence in [0, 1]. We keep the original (0, 1] lower bound (a 0
    # restraint carries no signal) which is stricter than chai and still valid.
    if not (0.0 < float(confidence) <= 1.0):
        raise ValueError(f'confidence must be in (0, 1], got {confidence}')


def _res_idx(one_letter: str, resnum: int) -> str:
    """Build a chai residue-token index '<one-letter><1-based pos>' (e.g. 'H3').

    ``one_letter`` is the residue ONE-LETTER code (identity), NOT the chain id; use
    ``UNKNOWN_RES`` ('X' => UNK) when the identity is unknown at build time.
    """
    return f'{one_letter}{int(resnum)}'


def contact_row(chain_a: str, resnum_a: int, chain_b: str, resnum_b: int,
                confidence: float, max_distance: float, min_distance: float = 0.0,
                res_one_letter_a: str = UNKNOWN_RES, res_one_letter_b: str = UNKNOWN_RES,
                atom_a: str = '', atom_b: str = '',
                comment: str = '', restraint_id: str = '') -> str:
    """An inter-chain CONTACT restraint row: residue (chain_a, resnum_a) within
    [min_distance, max_distance] A of residue (chain_b, resnum_b). Both endpoints are
    residue tokens '<one-letter><pos>'. Rejects intra-chain (chain_a == chain_b) — Chai
    contact restraints are inter-chain only.

    ``res_one_letter_a``/``res_one_letter_b`` are the residue ONE-LETTER identities
    ('X' default => UNK when unknown at build time).

    ATOM-AWARE contact (METAL-(b) STEP 3): when ``atom_a``/``atom_b`` are given the endpoint is
    emitted as the chai token-atom form '<one-letter><pos>@<atom>' (e.g. 'H6@ND1', 'X1@ZN'). chai's
    parser (``_parse_res_idx``) reads the '@atom' into ``PairwiseInteraction.atom_nameA/B`` so the
    contact targets that specific atom. Omitted -> residue-level token (unchanged)."""
    if chain_a == chain_b:
        raise ValueError(
            f'contact restraints are INTER-chain only; got chainA==chainB=={chain_a!r} '
            f'(Chai is not trained on intra-chain bonds; use cyclization #23 instead)')
    _check_confidence(confidence)
    tok_a = _res_idx(res_one_letter_a, resnum_a) + (f'@{atom_a}' if atom_a else '')
    tok_b = _res_idx(res_one_letter_b, resnum_b) + (f'@{atom_b}' if atom_b else '')
    return ','.join([
        chain_a, tok_a, chain_b, tok_b,
        'contact', str(float(confidence)), str(float(min_distance)),
        str(float(max_distance)), comment, restraint_id,
    ])


def pocket_row(binder_chain: str, target_chain: str, target_resnum: int,
               confidence: float, max_distance: float, min_distance: float = 0.0,
               res_one_letter_target: str = UNKNOWN_RES,
               comment: str = '', restraint_id: str = '') -> str:
    """An inter-chain POCKET restraint row: ANY residue of binder_chain (the chain-level
    A side, res_idxA EMPTY) within max_distance A of target epitope residue
    (target_chain, target_resnum) (the token-level B side). Rejects
    binder_chain == target_chain (inter-chain only).

    chai pocket semantics: A = chain-level (empty res_idx), B = token-level epitope
    residue. ``res_one_letter_target`` is the epitope residue's one-letter identity
    ('X' default => UNK)."""
    if binder_chain == target_chain:
        raise ValueError(
            f'pocket restraints are INTER-chain only; got binder==target=={binder_chain!r}')
    _check_confidence(confidence)
    return ','.join([
        binder_chain, '', target_chain, _res_idx(res_one_letter_target, target_resnum),
        'pocket', str(float(confidence)), str(float(min_distance)),
        str(float(max_distance)), comment, restraint_id,
    ])


def write_restraints(path, rows) -> Path:
    """Write a Chai .restraints CSV (header + given rows) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RESTRAINT_HEADER + '\n' + '\n'.join(rows) + ('\n' if rows else ''))
    return path


def parse_restraints(path) -> list:
    """Parse a .restraints CSV back into a list of column->value dicts (round-trip helper)."""
    import csv

    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def pin_polarity_rows(params: dict) -> list:
    """ALPHA pin-polarity restraint (spec §2.1): a single inter-chain POCKET pinning the
    binder (chain-level side) into a pocket around a FIXED target anchor token near the HLH
    loop, locking the designed-for topology so per-iteration polarity checks can be skipped
    (validate it reproduces the GT polarity). Returns a 1-element row list.

    WHY POCKET, NOT CONTACT (#27 crash fix): a CONTACT restraint asserts BOTH endpoint
    one-letter codes match the REAL token residue names in the structure
    (token_dist_restraint.add_distance_restraint: ``expected_res_name == left/right_residue_name``).
    The binder is DESIGNED, so its anchor identity is unknown at build time — emitting 'X'/UNK
    there fails that assertion at the very first (L-seed) predict (the binder anchor is never
    actually UNK). A POCKET sidesteps per-residue identity matching on the VARIABLE binder: the
    binder is the CHAIN-level side A (``res_idxA == ''`` — no token, no identity check), while
    the only token-level identity asserted is the FIXED target anchor B, whose residue identity
    IS known (passed in as ``target_anchor_one_letter``; achiral Gly at the GT α target so the L
    and D 3-letter codes coincide). max_distance is reused as the pocket radius.

    ``target_anchor_one_letter`` MUST be the target anchor's REAL one-letter code (the pocket
    generator still asserts the token-side identity matches the structure); defaults to 'X'
    (UNK) only when the caller genuinely cannot supply it.
    """
    return [pocket_row(
        binder_chain=params['binder_chain'], target_chain=params['target_chain'],
        target_resnum=params['target_anchor_resnum'],
        confidence=params['confidence'], max_distance=params['max_distance'],
        res_one_letter_target=params.get('target_anchor_one_letter', UNKNOWN_RES),
        comment='pin-polarity', restraint_id='pin_polarity_0',
    )]


def metal_coordination_rows(params: dict) -> list:
    """Coordinator<->metal restraints (spec §2.2): one tight inter-chain CONTACT per
    coordinating residue, between the coordinator chain and the metal chain (the metal enters
    as a separate HETATM/ligand chain, so this is inter-chain and valid). Drives diffusion into
    the metal-coordinated subspace. The metal token has no standard one-letter code, so the
    metal side is always 'X' (UNK).

    Two coordinator sources (DECLARATIVE wins when present):
      * ``coord_residues`` — a list of (1-based pos, one-letter identity[, three_letter,
        chirality, atom]) tuples from the declarative ``--coord_residues`` flag. GENERALIZES
        beyond His/Zn: each coordinator emits its REAL one-letter identity (His 'H', Cys 'C',
        Asp 'D', ...). The chai CONTACT asserts the one-letter matches the structure token, so a
        correct identity is required.
      * ``his_resnums`` — the legacy His-only positions (each emitted with identity 'H'); used
        only when ``coord_residues`` is absent (the case default).

    ATOM-SPECIFIC coordination (METAL-(b) solution (b)): when a coordinator carries a liganding
    atom (the 5th tuple element, e.g. 'ND1'/'SG'/'OD1'), a SINGLE ATOM-AWARE CONTACT row is emitted
    carrying ``atom_a`` (the coordinator's liganding atom) and ``atom_b`` (the metal atom, default
    'ZN') — a real atom-specific dative coordination (His ND1 <-> Zn), NOT a covalent bond. The
    metal must be fed as a CCD residue (targets/io_spec) so 'ZN' resolves to the cached-conformer
    atom; with the SMILES form the atom is 'ZN1' and would not resolve. D coordinators are handled
    the same (the dist patch relaxes the residue-name guard to position-only).

    Coordination is DATIVE, so the covalent-to-metal path stays OFF by default
    (``metal_covalent_atoms=False``): a covalent-to-metal row makes chai's
    ``get_atom_covalent_bond_pairs_from_constraints`` assert (the metal one-letter 'X'@ZN does not
    resolve through the L-only name table), which CRASHES the run — GPU-confirmed. It is emitted
    ONLY when ``metal_covalent_atoms=True`` is passed explicitly. With NO atoms, behavior is
    unchanged (residue-level contact)."""
    metal_chain = params['metal_chain']
    metal_resnum = params['metal_resnum']
    coord_chain = params.get('his_chain') or params.get('coord_chain')
    metal_atom = params.get('metal_atom', 'ZN')
    coords = params.get('coord_residues')
    if coords:
        # DECLARATIVE path: (pos, one-letter[, three_letter, chirality, atom]); generic id/comment.
        # ``atom`` is the OPTIONAL 5th element (index 4); absent on legacy 2-tuples.
        items = [(int(t[0]), str(t[1]),
                  (t[4] if len(t) > 4 else None),
                  f'{t[1]}-metal', f'metal_coord_{int(t[0])}')
                 for t in coords]
    else:
        his_resnums = params.get('his_resnums')
        if not his_resnums:
            raise ValueError('metal_coordination_rows: no coord_residues or his_resnums provided')
        # Legacy His-only path: identity 'H', no atom, original 'His-Zn'/'zn_coord_<pos>'.
        items = [(int(hr), 'H', None, 'His-Zn', f'zn_coord_{int(hr)}') for hr in his_resnums]
    # Covalent-to-metal emission is OFF by default (it crashes Chai's covalent-bond builder — the
    # metal one-letter 'X'@ZN never resolves). Coordination is dative; the atom-aware CONTACT below
    # carries the liganding atom instead. The bond only emits when explicitly opted in.
    emit_covalent = bool(params.get('metal_covalent_atoms', False))
    rows = []
    for pos, one_letter, atom, comment, rid in items:
        # Atom-aware contact when the coordinator declares a liganding atom: narrows the contact to
        # <coord>@<atom> <-> <metal>@<metal_atom>. Residue-level otherwise (back-compat).
        rows.append(contact_row(
            chain_a=coord_chain, resnum_a=pos, chain_b=metal_chain, resnum_b=metal_resnum,
            confidence=params['confidence'], max_distance=params['max_distance'],
            res_one_letter_a=one_letter, res_one_letter_b=UNKNOWN_RES,
            atom_a=(atom or ''), atom_b=(metal_atom if atom else ''),
            comment=comment, restraint_id=rid))
        if emit_covalent and atom:
            rows.append(covalent_bond_row(
                chain_a=coord_chain, resnum_a=pos, atom_a=atom,
                chain_b=metal_chain, resnum_b=metal_resnum, atom_b=metal_atom,
                res_one_letter_a=one_letter, res_one_letter_b=UNKNOWN_RES,
                confidence=params['confidence'],
                comment=f'{comment}-covalent', restraint_id=f'{rid}_cov'))
    return rows


def covalent_bond_row(chain_a: str, resnum_a: int, atom_a: str,
                      chain_b: str, resnum_b: int, atom_b: str,
                      res_one_letter_a: str, res_one_letter_b: str,
                      confidence: float = 1.0, bond_length: float = 2.05,
                      comment: str = '', restraint_id: str = '') -> str:
    """A COVALENT bond row (atom-level), the chai primitive for disulfides + macrocyclization.

    UNLIKE contact/pocket, a COVALENT row may be INTRA-chain (chain_a == chain_b) — that is
    exactly the head-to-tail / disulfide closure case. chai consumes COVALENT rows via
    ``get_atom_covalent_bond_pairs_from_constraints`` (bond_utils) into real
    ``atom_covalent_bond_indices`` (NOT through the contact/pocket restraint path, which skips
    them). Requirements verified against the 0.6.1 source:
      * atom names MUST be given on BOTH ends (``assert atom_nameA and atom_nameB``);
        emitted as the ``<one-letter><pos>@<atom>`` token form.
      * the one-letter residue identity is matched against the token (rc.restype_1to3[name]),
        so ``res_one_letter_*`` MUST be the residue actually at that position. (CAVEAT: chai
        maps the one-letter via the canonical L table; a D-CCD token whose stored name differs
        from the L parent may fail this match — verified per-run, see design_cyclic/nonalpha.)
      * max/min distance are NOT used as a restraint for COVALENT (the bond is topological);
        ``bond_length`` is carried in max_distance_angstrom for documentation only.
    """
    _check_confidence(confidence)
    return ','.join([
        chain_a, f'{_res_idx(res_one_letter_a, resnum_a)}@{atom_a}',
        chain_b, f'{_res_idx(res_one_letter_b, resnum_b)}@{atom_b}',
        'covalent', str(float(confidence)), '0.0', str(float(bond_length)),
        comment, restraint_id,
    ])


def disulfide_rows(chain: str, cys_pairs, res_one_letters=None,
                   confidence: float = 1.0) -> list:
    """COVALENT SG-SG bond rows for a set of (i, j) 1-based Cys position pairs (a cystine
    knot has 3 disulfides). INTRA-chain by construction. ``res_one_letters`` optionally maps a
    1-based position -> its one-letter code (default 'C' for every Cys); pass the real codes
    when the binder uses a D-CCD Cys whose token identity differs."""
    rows = []
    rol = res_one_letters or {}
    for (i, j) in cys_pairs:
        if i == j:
            raise ValueError(f'disulfide pair must be distinct positions, got ({i}, {j})')
        rows.append(covalent_bond_row(
            chain, i, 'SG', chain, j, 'SG',
            res_one_letter_a=rol.get(i, 'C'), res_one_letter_b=rol.get(j, 'C'),
            confidence=confidence, bond_length=2.05,
            comment=f'disulfide_{i}_{j}', restraint_id=f'ss_{i}_{j}'))
    return rows


def head_to_tail_closure_row(chain: str, length: int,
                             n_term_one_letter: str, c_term_one_letter: str,
                             confidence: float = 1.0) -> str:
    """COVALENT backbone bond closing a head-to-tail macrocycle (#23): the C-terminal residue's
    carbonyl C bonded to the N-terminal residue's amide N (intra-chain). The peptide-bond length
    ~1.33 A is carried in max_distance for documentation. ``*_one_letter`` are the residue codes
    at positions 1 (N-term) and ``length`` (C-term)."""
    return covalent_bond_row(
        chain, length, 'C', chain, 1, 'N',
        res_one_letter_a=c_term_one_letter, res_one_letter_b=n_term_one_letter,
        confidence=confidence, bond_length=1.33,
        comment='cyclic_closure', restraint_id='head_to_tail')


def build_for_case(case) -> list:
    """Dispatch on case.restraint.kind and return the .restraints rows for that case (#27).

    Raises ValueError if the case has no restraint, or if its restraint is still a SHELL whose
    params are not yet usable (e.g. 9DXX's pocket with empty target_resnums, pending gate #29).
    """
    spec = case.restraint
    if spec is None:
        raise ValueError(f'case {case.case_id!r} has no restraint spec')
    p = dict(spec.params)
    if spec.kind == 'pin_polarity':
        return pin_polarity_rows(p)
    if spec.kind == 'metal_coordination':
        return metal_coordination_rows(p)
    if spec.kind == 'contact':
        return [contact_row(
            chain_a=p['binder_chain'], resnum_a=p['binder_resnum'],
            chain_b=p['target_chain'], resnum_b=p['target_resnum'],
            confidence=p['confidence'], max_distance=p['max_distance'],
            res_one_letter_a=p.get('binder_one_letter', UNKNOWN_RES),
            res_one_letter_b=p.get('target_one_letter', UNKNOWN_RES),
            comment=p.get('comment', ''), restraint_id=p.get('restraint_id', 'contact_0'),
        )]
    if spec.kind == 'pocket':
        target_resnums = p.get('target_resnums', ())
        if not target_resnums:
            raise ValueError(
                f'case {case.case_id!r} pocket restraint is a SHELL (no target_resnums); '
                f'target epitope is pending gate #{p.get("pending_gate", 29)} — cannot build.')
        return [pocket_row(
            binder_chain=p['binder_chain'], target_chain=p['target_chain'],
            target_resnum=target_resnums[0], confidence=p['confidence'],
            max_distance=p['max_distance'],
            res_one_letter_target=p.get('target_one_letter', UNKNOWN_RES),
            comment='pocket', restraint_id='pocket_0',
        )]
    raise ValueError(f'unknown restraint kind {spec.kind!r} for case {case.case_id!r}')
