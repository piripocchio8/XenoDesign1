# tests/test_restraints.py
import pytest

from xenodesign.benchmark.restraints import (
    RESTRAINT_HEADER, contact_row, pocket_row, write_restraints, parse_restraints,
)


def test_restraint_header_is_the_chai_061_columns():
    # chai 0.6.1 PairwiseConstraintDataframeModel columns (see the parser + the
    # bundled examples/restraints/*.restraints). Distances are *_angstrom.
    assert RESTRAINT_HEADER == (
        'chainA,res_idxA,chainB,res_idxB,connection_type,'
        'confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id')


def test_contact_row_inter_chain_format():
    # res_idx is '<one-letter-residue-code><1-based-pos>'. The leading char is the
    # residue IDENTITY (chai looks it up in restype_1to3_with_x), NOT the chain id.
    # Unknown identity at build time -> 'X' (=> UNK).
    row = contact_row(chain_a='A', resnum_a=1, chain_b='B', resnum_b=21,
                      confidence=0.5, max_distance=12.0, restraint_id='pin0')
    assert row == 'A,X1,B,X21,contact,0.5,0.0,12.0,,pin0'


def test_contact_row_honours_explicit_one_letter_codes():
    row = contact_row(chain_a='A', resnum_a=3, chain_b='B', resnum_b=1,
                      confidence=0.8, max_distance=2.6,
                      res_one_letter_a='H', res_one_letter_b='X',
                      comment='His-Zn', restraint_id='zn0')
    assert row == 'A,H3,B,X1,contact,0.8,0.0,2.6,His-Zn,zn0'


def test_pocket_row_leaves_res_idxA_empty():
    # chai pocket: A is chain-level (res_idxA MUST be empty), B is the token.
    row = pocket_row(binder_chain='A', target_chain='B', target_resnum=5,
                     confidence=0.3, max_distance=14.0, restraint_id='pkt0')
    assert row == 'A,,B,X5,pocket,0.3,0.0,14.0,,pkt0'


def test_contact_row_rejects_intra_chain():
    with pytest.raises(ValueError):
        contact_row(chain_a='A', resnum_a=1, chain_b='A', resnum_b=12,
                    confidence=0.5, max_distance=8.0, restraint_id='bad')


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValueError):
        contact_row('A', 1, 'B', 2, confidence=1.5, max_distance=8.0, restraint_id='x')


def test_write_and_parse_roundtrip(tmp_path):
    rows = [
        contact_row('A', 1, 'B', 21, confidence=0.5, max_distance=12.0, restraint_id='c0'),
        pocket_row('A', 'B', 5, confidence=0.3, max_distance=14.0, restraint_id='p0'),
    ]
    path = tmp_path / 'case.restraints'
    write_restraints(path, rows)
    text = path.read_text()
    assert text.splitlines()[0] == RESTRAINT_HEADER
    parsed = parse_restraints(path)
    assert len(parsed) == 2
    assert parsed[0]['chainA'] == 'A' and parsed[0]['res_idxA'] == 'X1'
    assert parsed[0]['connection_type'] == 'contact'
    assert parsed[1]['connection_type'] == 'pocket' and parsed[1]['res_idxA'] == ''
    assert parsed[1]['res_idxB'] == 'X5'
    assert parsed[1]['max_distance_angstrom'] == '14.0'


from xenodesign.benchmark.restraints import (
    pin_polarity_rows, metal_coordination_rows, build_for_case,
)
from xenodesign.benchmark.cases import get_case


def test_pin_polarity_single_pocket_row():
    # #27 crash fix: the pin is a POCKET, not a contact. The binder is the chain-level side
    # (chainA, res_idxA EMPTY -> NO identity assertion on the DESIGNED binder anchor); the
    # FIXED target anchor is the token side (chainB) with its REAL one-letter code. Here the
    # caller passed the real target anchor code 'G' (achiral Gly at the α target resnum 21).
    rows = pin_polarity_rows({
        'binder_chain': 'A', 'binder_anchor_resnum': 1,
        'target_chain': 'B', 'target_anchor_resnum': 21,
        'target_anchor_one_letter': 'G',
        'max_distance': 12.0, 'confidence': 0.5,
    })
    assert len(rows) == 1
    # pocket: chainA (binder) chain-level => res_idxA empty; chainB (target) token => 'G21'
    # with the REAL fixed-target one-letter code.
    assert rows[0] == 'A,,B,G21,pocket,0.5,0.0,12.0,pin-polarity,pin_polarity_0'


def test_pin_polarity_target_code_defaults_to_unknown():
    # Without an explicit target one-letter code, fall back to 'X' (UNK) — used only when the
    # caller genuinely cannot supply the fixed target anchor identity.
    rows = pin_polarity_rows({
        'binder_chain': 'A', 'binder_anchor_resnum': 1,
        'target_chain': 'B', 'target_anchor_resnum': 21,
        'max_distance': 12.0, 'confidence': 0.5,
    })
    assert rows[0] == 'A,,B,X21,pocket,0.5,0.0,12.0,pin-polarity,pin_polarity_0'


def test_metal_coordination_one_row_per_his():
    rows = metal_coordination_rows({
        'metal_chain': 'B', 'metal_resnum': 1, 'metal_atom': 'ZN',
        'his_chain': 'A', 'his_resnums': (3, 6, 8, 11),
        'max_distance': 2.6, 'confidence': 0.8,
    })
    assert len(rows) == 4
    # His side identity is known (H); metal token identity unknown to the 1-letter
    # table -> 'X'.
    assert rows[0] == 'A,H3,B,X1,contact,0.8,0.0,2.6,His-Zn,zn_coord_3'
    assert rows[3] == 'A,H11,B,X1,contact,0.8,0.0,2.6,His-Zn,zn_coord_11'


def test_metal_coordination_contact_only_by_default_even_with_atoms():
    # FIX A (stopgap): a covalent-to-metal row makes chai's
    # get_atom_covalent_bond_pairs_from_constraints assert (the metal one-letter 'X'@ZN does
    # not resolve), crashing the run. So even when coordinators carry liganding atoms, the
    # DEFAULT emission is residue-level CONTACT ONLY (which apply via the position-only
    # token_dist patch). The coordinator atom is RETAINED in the data for a future patch, but
    # NO covalent-to-metal row is emitted unless explicitly opted in.
    rows = metal_coordination_rows({
        'metal_chain': 'B', 'metal_resnum': 1, 'metal_atom': 'ZN',
        'coord_chain': 'A',
        'coord_residues': [(6, 'H', 'HIS', 'L', 'ND1'), (12, 'H', 'DHI', 'D', 'ND1')],
        'max_distance': 2.6, 'confidence': 0.8,
    })
    contacts = [r for r in rows if ',contact,' in r]
    covalents = [r for r in rows if ',covalent,' in r]
    assert len(contacts) == 2 and len(covalents) == 0
    # residue-level contact (robust distance bias): 'H6'/'X1'.
    assert contacts[0].split(',')[1] == 'H6' and contacts[0].split(',')[3] == 'X1'
    assert contacts[1].split(',')[1] == 'H12' and contacts[1].split(',')[3] == 'X1'


def test_metal_coordination_emits_covalent_only_when_flag_explicitly_on():
    # The covalent-to-metal path still exists but is OFF by default; it appears ONLY when
    # metal_covalent_atoms=True is explicitly passed (a future patch will make this safe).
    rows = metal_coordination_rows({
        'metal_chain': 'B', 'metal_resnum': 1, 'metal_atom': 'ZN',
        'coord_chain': 'A',
        'coord_residues': [(6, 'H', 'HIS', 'L', 'ND1'), (12, 'H', 'DHI', 'D', 'ND1')],
        'max_distance': 2.6, 'confidence': 0.8,
        'metal_covalent_atoms': True,
    })
    contacts = [r for r in rows if ',contact,' in r]
    covalents = [r for r in rows if ',covalent,' in r]
    assert len(contacts) == 2 and len(covalents) == 2
    # atom-level covalent: binder 'H6@ND1' <-> metal 'X1@ZN'.
    c0 = covalents[0].split(',')
    assert c0[0] == 'A' and c0[1] == 'H6@ND1'
    assert c0[2] == 'B' and c0[3] == 'X1@ZN'
    assert c0[4] == 'covalent'
    # D coordinator also produces a covalent row (DHI12, atom ND1).
    c1 = covalents[1].split(',')
    assert c1[1] == 'H12@ND1' and c1[3] == 'X1@ZN'


def test_metal_coordination_no_atoms_is_pure_contact_backcompat():
    # No atoms on coordinators -> only contact rows (existing behavior, unchanged).
    rows = metal_coordination_rows({
        'metal_chain': 'B', 'metal_resnum': 1,
        'coord_chain': 'A',
        'coord_residues': [(6, 'H'), (12, 'H')],
        'max_distance': 2.6, 'confidence': 0.8,
    })
    assert all(',contact,' in r for r in rows)
    assert not any(',covalent,' in r for r in rows)
    assert len(rows) == 2


def test_metal_coordination_his_resnums_still_pure_contact():
    # Legacy His-only path (his_resnums) carries no atoms -> pure contact, no covalent.
    rows = metal_coordination_rows({
        'metal_chain': 'B', 'metal_resnum': 1,
        'his_chain': 'A', 'his_resnums': (3, 6),
        'max_distance': 2.6, 'confidence': 0.8,
    })
    assert all(',contact,' in r for r in rows) and len(rows) == 2


def test_build_for_case_alpha_uses_pin_polarity():
    # The α pin is now a POCKET (#27 crash fix): chainA (binder) chain-level (res_idxA EMPTY),
    # chainB (target) the FIXED anchor token. The case-nominal params carry no explicit target
    # one-letter code here, so build_for_case falls back to 'X' (the run driver,
    # build_alpha_restraint, supplies the REAL target code read from the FASTA).
    rows = build_for_case(get_case('alpha'))
    assert len(rows) == 1 and rows[0].startswith('A,,B,X21,pocket,')


def test_build_for_case_cyclic_uses_metal_coordination():
    rows = build_for_case(get_case('cyclic'))
    # Full 24-mer 6UFA has 4 coordinating His (6/12/18/24, L/D/L/D) -> 4 His-Zn contact rows.
    assert len(rows) == 4 and all(',contact,' in r for r in rows)
    assert [r.split(',')[1] for r in rows] == ['H6', 'H12', 'H18', 'H24']


def test_build_cyclic_restraint_rows_preserves_atom_and_chirality():
    # WT-RESTRAINTS #1+#2: build_cyclic_restraint_rows must pass the FULL coord tuple
    # (pos, one_letter, three_letter, chirality, atom) through to metal_coordination_rows so
    # (a) atom-level COVALENT His-ND1->Zn rows emit, and (b) the D coordinator's identity is
    # not silently flattened to an L 2-tuple. Previously it truncated to (pos, one_letter).
    from xenodesign.classes._cyclic_internals import build_cyclic_restraint_rows
    case = get_case('cyclic')
    coords = [(6, 'H', 'HIS', 'L', 'ND1'), (12, 'H', 'DHI', 'D', 'ND1')]
    rows = build_cyclic_restraint_rows(case, his_chain='A', metal_chain='B',
                                       coord_residues=coords)
    contacts = [r for r in rows if ',contact,' in r]
    covalents = [r for r in rows if ',covalent,' in r]
    # FIX A: covalent-to-metal is OFF by default (it crashes Chai), so the builder emits
    # CONTACT rows only. The full tuple (incl. atom + chirality) is still threaded through
    # (a future patch will consume the retained atom), but no metal covalent rows emit here.
    assert len(contacts) == 2 and len(covalents) == 0
    # The declared D coordinator's identity is preserved in the contact rows.
    assert contacts[0].split(',')[1] == 'H6'
    assert contacts[1].split(',')[1] == 'H12'


def test_build_for_case_nonalpha_shell_raises_pending_gate():
    import pytest
    with pytest.raises(ValueError) as ei:
        build_for_case(get_case('nonalpha'))
    assert '#29' in str(ei.value) or 'pending' in str(ei.value).lower()


# ── P2b / P3b: COVALENT bond builders (disulfides + head-to-tail closure) ──────────

from xenodesign.benchmark.restraints import (
    covalent_bond_row, disulfide_rows, head_to_tail_closure_row,
)


def _cols(row):
    return row.split(',')


def test_covalent_bond_row_atom_level_format():
    row = covalent_bond_row('A', 5, 'SG', 'A', 19, 'SG',
                            res_one_letter_a='C', res_one_letter_b='C',
                            comment='disulfide_5_19', restraint_id='ss_5_19')
    c = _cols(row)
    assert c[0] == 'A' and c[1] == 'C5@SG'          # atom-level token <name><pos>@<atom>
    assert c[2] == 'A' and c[3] == 'C19@SG'
    assert c[4] == 'covalent'
    assert c[8] == 'disulfide_5_19' and c[9] == 'ss_5_19'


def test_covalent_bond_row_allows_intra_chain():
    # COVALENT is the ONE row type that may be intra-chain (closure/disulfide) — must NOT raise.
    row = covalent_bond_row('A', 1, 'N', 'A', 12, 'C',
                            res_one_letter_a='K', res_one_letter_b='H')
    assert row.split(',')[4] == 'covalent'


def test_disulfide_rows_three_pairs_cystine_knot():
    rows = disulfide_rows('A', [(3, 17), (7, 22), (12, 26)])
    assert len(rows) == 3
    assert all(',covalent,' in r and 'SG' in r for r in rows)
    assert _cols(rows[0])[1] == 'C3@SG' and _cols(rows[0])[3] == 'C17@SG'


def test_disulfide_rows_custom_one_letter_for_d_cys():
    # When a D-CCD Cys token differs from the L 'C' code, the caller can override per position.
    rows = disulfide_rows('A', [(3, 17)], res_one_letters={3: 'C', 17: 'C'})
    assert _cols(rows[0])[1] == 'C3@SG'


def test_disulfide_rows_rejects_self_pair():
    with pytest.raises(ValueError):
        disulfide_rows('A', [(5, 5)])


def test_head_to_tail_closure_row_backbone_n_c():
    row = head_to_tail_closure_row('A', length=12, n_term_one_letter='K', c_term_one_letter='H')
    c = _cols(row)
    # C-term residue's carbonyl C  <->  N-term residue's amide N (intra-chain closure).
    assert c[0] == 'A' and c[1] == 'H12@C'
    assert c[2] == 'A' and c[3] == 'K1@N'
    assert c[4] == 'covalent' and c[8] == 'cyclic_closure'


def test_covalent_rows_roundtrip_through_chai_columns(tmp_path):
    rows = disulfide_rows('A', [(3, 17)]) + [
        head_to_tail_closure_row('A', 12, 'K', 'H')]
    p = write_restraints(tmp_path / 'cov.restraints', rows)
    parsed = parse_restraints(p)
    assert len(parsed) == 2
    # every chai column present + connection_type == covalent
    for d in parsed:
        assert d['connection_type'] == 'covalent'
        assert '@' in d['res_idxA'] and '@' in d['res_idxB']
        assert set(d) >= {'chainA', 'res_idxA', 'chainB', 'res_idxB', 'connection_type',
                          'confidence', 'min_distance_angstrom', 'max_distance_angstrom',
                          'comment', 'restraint_id'}
