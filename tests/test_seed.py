import pytest
from xenodesign.seed import retro_inverso, RandomSeedGenerator


def test_retro_inverso_reverses_sequence():
    # Pure reverse of the one-letter sequence (chirality handled downstream as D).
    assert retro_inverso("ACDEF") == "FEDCA"


def test_retro_inverso_no_reverse_keeps_order():
    assert retro_inverso("ACDEF", reverse=True) == "FEDCA"
    assert retro_inverso("ACDEF", reverse=False) == "ACDEF"


def test_random_seed_generator_length_and_alphabet():
    gen = RandomSeedGenerator(seed=0)
    s = gen.generate(length=20)
    assert len(s) == 20
    assert set(s) <= set("ARNDCQEGHILKMFPSTWYV")


def test_random_seed_generator_is_deterministic():
    assert RandomSeedGenerator(seed=7).generate(15) == RandomSeedGenerator(seed=7).generate(15)


from xenodesign.seed import PepMLMSeedGenerator


def test_pepmlm_generator_applies_retro_inverso_via_injected_fn():
    # Inject a fake "model" that ignores the target and returns a fixed peptide.
    fake = lambda target_seq, length: "ACDEF"
    gen = PepMLMSeedGenerator(generate_fn=fake)
    # default: retro-inverso reverses the generated peptide
    assert gen.generate(target_seq="MKLV", length=5) == "FEDCA"


def test_pepmlm_generator_can_disable_reverse():
    fake = lambda target_seq, length: "ACDEF"
    gen = PepMLMSeedGenerator(generate_fn=fake, reverse=False)
    assert gen.generate(target_seq="MKLV", length=5) == "ACDEF"


def test_pepmlm_generator_lazy_no_transformers_required():
    # Constructing with an injected fn must not import transformers/torch.
    gen = PepMLMSeedGenerator(generate_fn=lambda t, l: "AAAA")
    assert gen.generate("M", 4) == "AAAA"


import numpy as np
from xenodesign.seed import select_seeds_by_mirror_consistency


def test_select_seeds_keeps_low_discrepancy():
    rng = np.random.RandomState(0)
    a = rng.rand(10, 3)
    good_twin = a.copy(); good_twin[:, 0] *= -1.0           # exact mirror -> ~0 discrepancy
    bad_twin = rng.rand(10, 3)                              # unrelated -> high discrepancy
    seeds = [
        {"id": "good", "coords": a, "twin_coords": good_twin},
        {"id": "bad", "coords": a, "twin_coords": bad_twin},
    ]
    kept = select_seeds_by_mirror_consistency(seeds, axis=0, threshold=0.1)
    assert [s["id"] for s in kept] == ["good"]


def test_select_seeds_empty_when_all_above_threshold():
    rng = np.random.RandomState(1)
    seeds = [{"id": "x", "coords": rng.rand(8, 3), "twin_coords": rng.rand(8, 3)}]
    assert select_seeds_by_mirror_consistency(seeds, axis=0, threshold=0.01) == []


# ── double-flip D-correct seeding (spec §2.3/§4) ─────────────────────────────

def _reflect_binder_in_complex_numpy(coords, binder_mask, axis=0):
    """Pure-numpy core of reflect_binder_in_complex_from_cif (no gemmi, CPU-testable).

    ``coords`` is (n_atoms, 3), ``binder_mask`` is a boolean array of length n_atoms.
    Returns a new (n_atoms, 3) array with the binder atoms reflected along ``axis``
    and target atoms unchanged.
    """
    from xenodesign.mirror import reflect_coords
    reflected = reflect_coords(coords, axis=axis)
    out = coords.copy()
    out[binder_mask] = reflected[binder_mask]
    return out


def test_reflect_binder_leaves_target_unchanged():
    """Target atoms must not be touched by the binder reflection."""
    rng = np.random.RandomState(42)
    target_coords = rng.rand(20, 3).astype(np.float32)
    binder_coords = rng.rand(8, 3).astype(np.float32)
    combined = np.vstack([target_coords, binder_coords])
    binder_mask = np.array([False] * 20 + [True] * 8)

    out = _reflect_binder_in_complex_numpy(combined, binder_mask, axis=0)

    np.testing.assert_array_equal(out[:20], target_coords, err_msg="target atoms changed")
    np.testing.assert_array_almost_equal(
        out[20:, 0], -binder_coords[:, 0], decimal=6,
        err_msg="binder x-coords not reflected"
    )
    np.testing.assert_array_equal(out[20:, 1], binder_coords[:, 1], err_msg="binder y changed")
    np.testing.assert_array_equal(out[20:, 2], binder_coords[:, 2], err_msg="binder z changed")


def test_reflect_binder_produces_d_chirality():
    """Reflecting an ideal L Cα frame along x should give a chiral-volume sign consistent with D.

    We place N, CA, C, CB at ideal L-alanine geometry, confirm L sign, then reflect only the
    binder part (all four atoms) and confirm the sign flips to D.
    """
    from xenodesign.chirality import is_chirality_violation

    # Ideal L-alanine frame (same as _IDEAL_L in chirality.py)
    n  = np.array([-0.525,  1.363, 0.000])
    ca = np.array([ 0.000,  0.000, 0.000])
    c  = np.array([ 1.526,  0.000, 0.000])
    cb = np.array([-0.529, -0.774, -1.205])

    # Verify L sign is correct for L
    assert not is_chirality_violation(n, ca, c, cb, "L"), "ideal L-Ala should not be L-violation"
    # And it IS a violation for D labelling
    assert is_chirality_violation(n, ca, c, cb, "D"), "ideal L-Ala should be D-violation"

    # Build a mock complex: 2 target atoms, then 4 binder atoms (N, CA, C, CB).
    target_atoms = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    binder_atoms = np.array([n, ca, c, cb], dtype=np.float32)
    combined = np.vstack([target_atoms, binder_atoms])
    binder_mask = np.array([False, False, True, True, True, True])

    out = _reflect_binder_in_complex_numpy(combined, binder_mask, axis=0)

    # Extract reflected binder atoms
    rn, rca, rc, rcb = out[2], out[3], out[4], out[5]

    # After reflection: binder should now have D chirality
    assert not is_chirality_violation(rn, rca, rc, rcb, "D"), (
        "reflected L→D binder frame is a D-chirality violation — reflect_binder_in_complex logic is wrong"
    )
    # And the L label should be violated
    assert is_chirality_violation(rn, rca, rc, rcb, "L"), (
        "reflected binder is still L — reflection did not flip chirality"
    )


# ── P4: SeedResult value object + unconditioned generate_conditioned shim ─────
from xenodesign.seed import SeedResult


def test_seed_result_fields_default_safe():
    r = SeedResult(one_letter="ACDEF", length=5)
    assert r.one_letter == "ACDEF"
    assert r.length == 5
    assert r.reverse_applied is False
    assert r.conditioned is False
    assert r.fixed_chirality == {}


def test_seed_result_records_retro_and_conditioning():
    r = SeedResult(one_letter="FEDCA", length=5, reverse_applied=True,
                   conditioned=True, fixed_chirality={2: "D", 5: "L"})
    assert r.reverse_applied is True
    assert r.conditioned is True
    assert r.fixed_chirality == {2: "D", 5: "L"}


def test_random_generator_generate_conditioned_ignores_target():
    gen = RandomSeedGenerator(seed=3)
    s = gen.generate_conditioned(target_seq="MKLVANYTHING", length=12)
    assert s == RandomSeedGenerator(seed=3).generate(12)  # target is ignored
    assert len(s) == 12


# ── P4: target FASTA reader (gitignored sequences never inlined in tests) ─────
from xenodesign.seed import read_target_sequence


def test_read_target_sequence_first_record(tmp_path):
    f = tmp_path / "t.fasta"
    f.write_text(">protein|target\nMKLVAAAA\n>protein|binder\nGGGG\n")
    assert read_target_sequence(f) == "MKLVAAAA"


def test_read_target_sequence_named_record(tmp_path):
    f = tmp_path / "t.fasta"
    f.write_text(">protein|target\nMKLVAAAA\n>protein|binder\nGGGG\n")
    assert read_target_sequence(f, name="binder") == "GGGG"


def test_read_target_sequence_multiline_and_strips(tmp_path):
    f = tmp_path / "t.fasta"
    f.write_text(">target\nMKLV\nAAAA\n\n")  # sequence wrapped across two lines
    assert read_target_sequence(f) == "MKLVAAAA"


def test_read_target_sequence_missing_name_raises(tmp_path):
    f = tmp_path / "t.fasta"
    f.write_text(">target\nMKLV\n")
    import pytest
    with pytest.raises(KeyError):
        read_target_sequence(f, name="nope")


# ── P4: manual His chirality insertion for the cyclic seed ────────────────
from xenodesign.seed import insert_fixed_chirality


def test_insert_fixed_chirality_places_his_and_records_map():
    seq = "ACDEFGHIKLMN"
    out, fixed = insert_fixed_chirality(
        seq, positions={3: "L", 6: "D", 8: "L", 11: "D"}, residue="H"
    )
    assert len(out) == len(seq)          # ring size preserved
    assert out[2] == "H" and out[5] == "H" and out[7] == "H" and out[10] == "H"
    assert out[0] == "A" and out[11] == "N"  # other positions untouched
    assert fixed == {3: "L", 6: "D", 8: "L", 11: "D"}


def test_insert_fixed_chirality_out_of_range_raises():
    import pytest
    with pytest.raises(ValueError):
        insert_fixed_chirality("ACDEF", positions={6: "D"}, residue="H")  # 1-based > len


def test_insert_fixed_chirality_empty_is_identity():
    out, fixed = insert_fixed_chirality("ACDEF", positions={}, residue="H")
    assert out == "ACDEF"
    assert fixed == {}


# ── PepMLM per-run sampling wiring (correction: argmax made every run identical) ──

def test_pepmlm_generator_accepts_seed_and_temperature():
    from xenodesign.seed import PepMLMSeedGenerator
    # injected-fn path (no network) still works + the new seed/temperature params are accepted
    g = PepMLMSeedGenerator(generate_fn=lambda t, l: "A" * l, reverse=False,
                            seed=42, temperature=0.8)
    assert g.generate("MAK", 5) == "AAAAA"
    assert g._seed == 42 and g._temperature == 0.8
    # default (no seed) keeps deterministic-argmax semantics flagged
    assert PepMLMSeedGenerator()._seed is None
