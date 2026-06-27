import pytest
from xenodesign.io_spec import to_d_fasta, build_fasta


def test_to_d_fasta_parenthesizes_each_d_residue():
    # ALA, CYS, SER -> their D CCD codes in parentheses (chai_lab modified-residue syntax).
    assert to_d_fasta("ACS") == "(DAL)(DCY)(DSN)"


def test_to_d_fasta_keeps_glycine_single_letter():
    assert to_d_fasta("AGS") == "(DAL)G(DSN)"


def test_to_d_fasta_rejects_unknown_letter():
    with pytest.raises(KeyError):
        to_d_fasta("AXS")


def test_build_fasta_l_target_and_d_binder():
    entities = [
        {"type": "protein", "name": "target", "sequence": "MAK", "chirality": "L"},
        {"type": "protein", "name": "binder", "sequence": "ACG", "chirality": "D"},
    ]
    out = build_fasta(entities)
    assert out == (
        ">protein|target\n"
        "MAK\n"
        ">protein|binder\n"
        "(DAL)(DCY)G\n"
    )


def test_build_fasta_defaults_to_L_chirality():
    entities = [{"type": "protein", "name": "t", "sequence": "AC"}]
    assert build_fasta(entities) == ">protein|t\nAC\n"


def test_build_fasta_ligand_smiles_entity():
    """Metal/ligand target entities (no 'sequence') emit a chai >ligand|name= + SMILES."""
    entities = [
        {"type": "protein", "name": "binder", "sequence": "GH", "chirality": "L"},
        {"type": "ligand", "name": "zn", "smiles": "[Zn+2]"},
    ]
    assert build_fasta(entities) == (
        ">protein|binder\nGH\n"
        ">ligand|name=zn\n[Zn+2]\n"
    )


def test_build_fasta_ligand_ccd_entity():
    out = build_fasta([{"type": "ligand", "name": "x", "ccd": "ZN"}])
    assert out == ">ligand|ccd=ZN\n"


def test_build_fasta_metal_ccd_emits_name_form_for_patch():
    """METAL-(b): a metal fed as a CCD residue is emitted as `>ligand|name=ZN` with the CCD code
    on the sequence line — the form chai_patches._patch_ligand_ccd_feeding recognizes (entity name
    AND sequence both the CCD code), so the metal tokenizes via the cached conformer (residue ZN,
    atom ZN) instead of SMILES `[Zn+2]` (residue LIG, atom ZN1)."""
    out = build_fasta([
        {"type": "protein", "name": "binder", "sequence": "GH", "chirality": "L"},
        {"type": "ligand", "name": "zn", "metal_ccd": "ZN"},
    ])
    assert out == (
        ">protein|binder\nGH\n"
        ">ligand|name=ZN\nZN\n"
    )


from xenodesign.io_spec import d_fasta_to_one_letter


def test_d_fasta_to_one_letter_inverts_to_d_fasta():
    assert d_fasta_to_one_letter("(DAL)(DCY)(DSN)") == "ACS"


def test_d_fasta_to_one_letter_keeps_glycine():
    assert d_fasta_to_one_letter("(DAL)G(DSN)") == "AGS"


def test_round_trip_l_to_d_to_l():
    seq = "ACGSEKLMFY"
    assert d_fasta_to_one_letter(to_d_fasta(seq)) == seq


def test_to_d_fasta_unknown_letter_reports_letter_and_position():
    with pytest.raises(KeyError) as exc:
        to_d_fasta("AXS")
    msg = str(exc.value)
    assert "X" in msg
    assert "2" in msg  # 1-based position of the offending letter


def test_d_fasta_to_one_letter_normalizes_selenomethionine():
    # MSE (selenomethionine) is a common CIF/Chai modified residue -> treat as MET.
    assert d_fasta_to_one_letter("(MSE)") == "M"


def test_build_fasta_rejects_fully_ncaa_chain():
    # All-D with no glycine -> fully parenthesized -> chai-1 cannot tokenize it.
    entities = [{"type": "protein", "name": "binder", "sequence": "AAA", "chirality": "D"}]
    with pytest.raises(ValueError, match="canonical"):
        build_fasta(entities)


def test_build_fasta_allows_all_D_with_one_glycine():
    # A single glycine (canonical, achiral) anchors tokenization -> allowed.
    entities = [{"type": "protein", "name": "binder", "sequence": "AAG", "chirality": "D"}]
    assert "(DAL)(DAL)G" in build_fasta(entities)


# --- P3 #8: glycine handling so all-D / cyclic chains pass Chai's >=1-canonical guard ---
def test_glycine_satisfy_guard_noncyclic_appends_trailing_gly():
    from xenodesign.io_spec import glycine_satisfy_guard
    out = glycine_satisfy_guard('AAAA', cyclic=False)
    assert out == 'AAAAG'


def test_glycine_satisfy_guard_idempotent():
    from xenodesign.io_spec import glycine_satisfy_guard
    once = glycine_satisfy_guard('AAAA', cyclic=False)
    twice = glycine_satisfy_guard(once, cyclic=False)
    assert once == twice == 'AAAAG'
    assert glycine_satisfy_guard('AAGAA', cyclic=False) == 'AAGAA'
    assert glycine_satisfy_guard('AAGAA', cyclic=True) == 'AAGAA'


def test_glycine_satisfy_guard_cyclic_premutates_first_SNP_to_gly():
    from xenodesign.io_spec import glycine_satisfy_guard
    assert glycine_satisfy_guard('AASAA', cyclic=True) == 'AAGAA'
    assert glycine_satisfy_guard('AANAA', cyclic=True) == 'AAGAA'
    assert glycine_satisfy_guard('AAPAA', cyclic=True) == 'AAGAA'
    assert len(glycine_satisfy_guard('AASAA', cyclic=True)) == 5


def test_glycine_satisfy_guard_cyclic_no_SNP_falls_back_to_append():
    from xenodesign.io_spec import glycine_satisfy_guard
    out = glycine_satisfy_guard('AAAA', cyclic=True)
    assert out == 'AAAAG'


def test_glycine_guard_composes_with_to_d_fasta_and_build_fasta():
    from xenodesign.io_spec import glycine_satisfy_guard, to_d_fasta, build_fasta
    safe = glycine_satisfy_guard('AAAA', cyclic=False)        # 'AAAAG'
    d = to_d_fasta(safe)                                       # '(DAL)(DAL)(DAL)(DAL)G'
    assert d.endswith('G')
    fasta = build_fasta([{'type': 'protein', 'name': 'binder',
                          'sequence': safe, 'chirality': 'D'}])
    assert '>protein|binder' in fasta and fasta.strip().endswith('G')
