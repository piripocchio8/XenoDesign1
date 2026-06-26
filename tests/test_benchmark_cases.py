# tests/test_benchmark_cases.py
import pytest

from xenodesign.benchmark.cases import (
    BaselineMetrics, BenchmarkCase, RestraintSpec, CASES, get_case,
)


def test_baseline_metrics_holds_alpha_measured_values():
    # BASELINE (#33 corrected): genuine PER-TOKEN GT — ipAE ~12.2 (token_asym split; the 8.59 was
    # an NA=21 chain-split artifact), Dunbrack ipsae_cut10 ~0.22. Optional field (None elsewhere).
    bm = BaselineMetrics(interface_iptm=0.44, ipae=12.2, ipsae_cut10=0.22, chirality=0.0)
    assert bm.interface_iptm == 0.44
    assert bm.ipae == pytest.approx(12.2, abs=0.1)
    assert bm.ipsae_cut10 == pytest.approx(0.22, abs=0.03)
    assert bm.chirality == 0.0


def test_baseline_metrics_ipsae_cut10_defaults_none():
    # New optional field must not perturb cases that never measured it.
    assert BaselineMetrics().ipsae_cut10 is None


def test_alpha_case_registered_and_shaped():
    case = get_case('alpha')
    assert isinstance(case, BenchmarkCase)
    assert case.case_id == 'alpha'
    assert case.chirality_class == 'all_D'
    assert case.binder_chain == 'A'
    assert case.target_chains == ('B',)
    assert case.binder_length == 21
    assert case.baseline.interface_iptm == 0.44
    # BASELINE (#33 corrected): genuine PER-TOKEN GT ipAE ~12.2 (token_asym split; matches
    # metrics.score_interface + spec §2.1). The 8.59 was an NA=21 chain-split artifact.
    assert case.baseline.ipae == pytest.approx(12.2, abs=0.2)
    # Dunbrack ipsae_cut10 baseline ~0.22 (per-token true split; 0.73 was the NA=21 artifact).
    assert case.baseline.ipsae_cut10 == pytest.approx(0.22, abs=0.03)
    assert case.baseline.chirality == 0.0
    assert case.restraint is not None
    assert case.restraint.kind == 'pin_polarity'
    assert case.knobs['binding_weight'] == pytest.approx(0.45)


def test_alpha_sequences_are_NOT_inlined_only_a_gitignored_path():
    case = get_case('alpha')
    assert case.fasta_path.endswith(
        'XenoDesign1_local_ref/dl_able_ground_truth/trimer_DL_ABLE.fasta')
    blob = repr(case).upper()
    for forbidden in ('MKTAYIAKQR', 'RKDES'):
        assert forbidden not in blob


def test_get_case_unknown_raises_with_known_ids():
    with pytest.raises(KeyError) as ei:
        get_case('nope')
    msg = str(ei.value)
    assert 'alpha' in msg


def test_cyclic_case_zn_cofactor_geometry_recovery():
    case = get_case('cyclic')
    assert case.case_id == 'cyclic'
    assert case.chirality_class == 'mixed'
    assert case.cofactor == 'Zn'
    assert case.binder_chain == 'A'
    assert case.target_chains == ()
    assert case.baseline.backbone_rmsd is not None and case.baseline.backbone_rmsd <= 1.0
    assert case.baseline.interface_iptm is None
    assert case.knobs['checkpoint_noise'] == pytest.approx(0.10)
    assert case.knobs['fix_first_shell'] is True
    assert case.restraint is not None and case.restraint.kind == 'metal_coordination'
    assert case.target_prep == 'fixed_input'
    # #33: new optional ipsae_cut10 field leaves non-alpha cases untouched.
    assert case.baseline.ipsae_cut10 is None


def test_cyclic_registered_in_cases():
    assert 'cyclic' in CASES


from xenodesign.benchmark.cases import target_gate_note


def test_nonalpha_case_gate_resolved_target_prep_msa():
    # Gate #29 resolved (spec §2.3, 2026-06-14): target uses precomputed MSA; binder MSA-free.
    case = get_case('nonalpha')
    assert case.case_id == 'nonalpha'
    assert case.chirality_class == 'all_D'
    assert case.binder_chain == 'A'
    assert case.target_chains == ('B',)
    assert case.binder_length == 31
    assert case.target_prep == 'msa'
    assert case.cofactor is None
    assert case.knobs['ss_bias'] == 'anti_alpha'
    assert case.restraint is not None and case.restraint.kind == 'pocket'
    # Binder baseline metrics stay None — separate future measurement
    assert case.baseline.interface_iptm is None
    assert case.baseline.ipsae_cut10 is None  # #33: optional field unaffected


def test_target_gate_note_resolved_for_nonalpha():
    # Gate #29 resolved: target_gate_note returns '' for nonalpha (no longer pending).
    assert target_gate_note('nonalpha') == ''
    assert target_gate_note('alpha') == ''
    assert target_gate_note('cyclic') == ''


def test_all_three_cases_registered():
    assert sorted(CASES) == ['alpha', 'cyclic', 'nonalpha']


def test_scaffold_composes_across_all_cases():
    from xenodesign.io_spec import glycine_satisfy_guard, to_d_fasta, build_fasta
    from xenodesign.benchmark.restraints import build_for_case
    for cid in ('alpha', 'cyclic', 'nonalpha'):
        case = get_case(cid)
        cyclic = (cid == 'cyclic')
        placeholder = 'A' * case.binder_length
        safe = glycine_satisfy_guard(placeholder, cyclic=cyclic)
        assert 'G' in to_d_fasta(safe)
        if case.target_chains:
            fasta = build_fasta([
                {'type': 'protein', 'name': 'binder', 'sequence': safe, 'chirality': 'D'},
                {'type': 'protein', 'name': 'target', 'sequence': 'AAAA', 'chirality': 'L'},
            ])
            assert '>protein|binder' in fasta and '>protein|target' in fasta
        if cid == 'nonalpha':
            import pytest
            with pytest.raises(ValueError):
                build_for_case(case)
        else:
            rows = build_for_case(case)
            # Per-case connection type (#27 crash fix): the α pin is a POCKET (binder is the
            # chain-level side -> no identity assertion on the DESIGNED binder anchor; a contact
            # would crash on the unknown 'X' binder code at the first predict); the cyclic
            # His<->Zn metal coordination stays a CONTACT (both identities known).
            expected_conn = ',pocket,' if cid == 'alpha' else ',contact,'
            assert rows and all(expected_conn in r for r in rows)


def test_nonalpha_target_prep_is_msa_after_gate_resolution():
    # Gate #29 resolved (spec §2.3, 2026-06-14): target_prep='msa', gate note cleared.
    case = get_case('nonalpha')
    assert case.target_prep == 'msa'
    assert target_gate_note('nonalpha') == ''
