# tests/test_controls.py
"""CPU tests for the within-Chai circularity controls (#35).

These pin the pure builders/analysers in xenodesign.eval.controls:
  - composition_matched_scramble: identical composition, scrambled order, deterministic.
  - off_target_helices: a small fixed decoy set of ~21-res L helices.
  - interface_footprint: which TARGET residues a binder contacts + heptad register.
The GT-reference loader test is skip-if-absent and asserts only aggregate length (never
inlines the unpublished D-binder sequence).
"""
from collections import Counter
from pathlib import Path

import pytest

from xenodesign.eval.controls import (
    composition_matched_scramble,
    off_target_helices,
    interface_footprint,
    heptad_register,
    register_shifted_target_decoy,
    aggregate_multiseed,
    GT_REFERENCE_FASTA,
)

# ---------------------------------------------------------------------------
# (1) composition_matched_scramble
# ---------------------------------------------------------------------------

def test_scramble_preserves_composition():
    seq = "GAKLVDEFWNQRSTYHILMP"
    out = composition_matched_scramble(seq, rng_seed=0)
    assert Counter(out) == Counter(seq)
    assert len(out) == len(seq)


def test_scramble_differs_in_order():
    # A sequence with enough entropy that a permutation almost surely reorders it.
    seq = "ACDEFGHIKLMNPQRSTVWY"
    out = composition_matched_scramble(seq, rng_seed=7)
    assert out != seq                       # order destroyed
    assert Counter(out) == Counter(seq)     # composition intact


def test_scramble_deterministic_given_seed():
    seq = "GAKLVDEFWNQRSTY"
    a = composition_matched_scramble(seq, rng_seed=123)
    b = composition_matched_scramble(seq, rng_seed=123)
    c = composition_matched_scramble(seq, rng_seed=124)
    assert a == b            # same seed -> identical
    assert a != c            # different seed -> (almost surely) different


def test_scramble_preserves_glycine_count():
    # Gly is the required canonical residue for chai tokenisation of an all-D chain;
    # a composition-matched scramble must keep every Gly (composition identity guarantees it).
    seq = "GGAKLVDEFWNQRST"
    out = composition_matched_scramble(seq, rng_seed=3)
    assert out.count("G") == seq.count("G") == 2


def test_scramble_homopolymer_is_stable():
    # All-identical residues: a permutation is necessarily identical (composition == order).
    seq = "AAAAAA"
    assert composition_matched_scramble(seq, rng_seed=9) == seq


# ---------------------------------------------------------------------------
# (2) off_target_helices
# ---------------------------------------------------------------------------

def test_off_target_set_shape():
    helices = off_target_helices()
    assert isinstance(helices, list)
    assert 3 <= len(helices) <= 5
    names = set()
    for item in helices:
        assert isinstance(item, tuple) and len(item) == 2
        name, seq = item
        assert isinstance(name, str) and name
        assert isinstance(seq, str)
        # ~21-res L helices: canonical one-letter, plausibly helical length.
        assert 18 <= len(seq) <= 24
        assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")
        names.add(name)
    assert len(names) == len(helices)   # unique names


# ---------------------------------------------------------------------------
# (3) interface_footprint — synthetic fixture CIF
# ---------------------------------------------------------------------------

def _write_two_chain_cif(path: Path, target_xyz, binder_xyz, target_chain="B",
                         binder_chain="A"):
    """Write a minimal Chai-style _atom_site CIF with CA atoms only.

    target_xyz / binder_xyz: list of (x, y, z) for consecutive CA atoms (resid 1..n).
    Column order matches the real Chai CIF header used in metrics._parse_cif_ca.
    """
    header = [
        "data_test",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_seq_id",
        "_atom_site.label_asym_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
    ]
    lines = list(header)
    aid = 1
    for chain, coords in ((target_chain, target_xyz), (binder_chain, binder_xyz)):
        for resid, (x, y, z) in enumerate(coords, start=1):
            lines.append(
                f"ATOM {aid} C CA . ALA {resid} {chain} "
                f"{x:.3f} {y:.3f} {z:.3f} 1.000 80.0"
            )
            aid += 1
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


def test_interface_footprint_classifies_contacts(tmp_path):
    # Target: 7 CA atoms spaced 20 A apart on x, so only ONE target residue (index 3)
    # is near the binder; others are far.
    target = [(i * 20.0, 0.0, 0.0) for i in range(7)]
    # Binder: a single CA placed 5 A from target residue 4 (resid index 3, 0-based).
    binder = [(3 * 20.0 + 5.0, 0.0, 0.0)]
    cif = tmp_path / "synthetic.cif"
    _write_two_chain_cif(cif, target, binder, target_chain="B", binder_chain="A")

    fp = interface_footprint(str(cif), target_chain="B", binder_chain="A")
    # Only target residue 4 (1-based) / index 3 is contacted (< 8 A).
    assert fp["contacted_resids"] == [4]
    assert fp["n_contacted"] == 1
    # Each contacted residue gets a heptad letter and a core/surface class.
    assert set(fp["registers"]) == {4}
    reg = fp["registers"][4]
    assert reg["heptad"] in "abcdefg"
    assert reg["face"] in ("core", "surface")
    # Aggregate: surface fraction in [0,1].
    assert 0.0 <= fp["surface_fraction"] <= 1.0


def test_interface_footprint_surface_vs_core(tmp_path):
    # Place the binder against a target residue whose heptad letter we control via offset.
    # Heptad register starts at 'a' on resid index 0 (resid 1). Contact target index 1
    # -> 'b' -> surface; contact index 0 -> 'a' -> core.
    target = [(i * 20.0, 0.0, 0.0) for i in range(7)]
    # Binder contacts target index 1 (resid 2) -> heptad 'b' -> surface face.
    binder_surface = [(1 * 20.0 + 4.0, 0.0, 0.0)]
    cif_s = tmp_path / "surface.cif"
    _write_two_chain_cif(cif_s, target, binder_surface, target_chain="B", binder_chain="A")
    fp_s = interface_footprint(str(cif_s), target_chain="B", binder_chain="A",
                               heptad_start="a")
    assert fp_s["contacted_resids"] == [2]
    assert fp_s["registers"][2]["heptad"] == "b"
    assert fp_s["registers"][2]["face"] == "surface"
    assert fp_s["surface_fraction"] == 1.0

    # Binder contacts target index 0 (resid 1) -> heptad 'a' -> core face.
    binder_core = [(0 * 20.0 + 4.0, 0.0, 0.0)]
    cif_c = tmp_path / "core.cif"
    _write_two_chain_cif(cif_c, target, binder_core, target_chain="B", binder_chain="A")
    fp_c = interface_footprint(str(cif_c), target_chain="B", binder_chain="A",
                               heptad_start="a")
    assert fp_c["contacted_resids"] == [1]
    assert fp_c["registers"][1]["heptad"] == "a"
    assert fp_c["registers"][1]["face"] == "core"
    assert fp_c["surface_fraction"] == 0.0


def test_heptad_register_cycle():
    # The heptad letters cycle a,b,c,d,e,f,g over residue index, core = a,d.
    letters = [heptad_register(i, start="a")[0] for i in range(8)]
    assert letters == ["a", "b", "c", "d", "e", "f", "g", "a"]
    faces = [heptad_register(i, start="a")[1] for i in range(7)]
    assert faces == ["core", "surface", "surface", "core",
                     "surface", "surface", "surface"]


# ---------------------------------------------------------------------------
# (4) GT reference loader — skip-if-absent; assert ONLY aggregate (length 21)
# ---------------------------------------------------------------------------

def test_gt_reference_loads_length_only():
    from xenodesign.eval.controls import load_gt_reference_binder

    if not GT_REFERENCE_FASTA.exists():
        pytest.skip(f"GT reference fasta absent: {GT_REFERENCE_FASTA}")
    seq = load_gt_reference_binder()
    # Assert ONLY the aggregate property: the reference D-binder is a 21-res helix.
    # NEVER inline or print the sequence itself.
    assert isinstance(seq, str)
    assert len(seq) == 21
    assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# (5) register_shifted_target_decoy — the coiled-coil register control (#Gate-B 2)
# ---------------------------------------------------------------------------

def test_register_shifted_decoy_is_composition_identical_rotation():
    # A synthetic 21-mer (3 heptads, varied composition so the rotation is observable).
    seq = "GAKLVDEFWNQRSTYHILMPC"  # 21 residues
    assert len(seq) == 21
    name, out = register_shifted_target_decoy(seq, shift=3)
    # Composition-identical (same bag of residues).
    assert Counter(out) == Counter(seq)
    assert len(out) == len(seq)
    # A pure circular rotation by 3: out == seq[3:] + seq[:3].
    assert out == seq[3:] + seq[:3]
    # Order is genuinely changed (shifted heptad register).
    assert out != seq
    # The decoy carries a descriptive, non-empty name.
    assert isinstance(name, str) and name


def test_register_shifted_decoy_default_shift_and_wrap():
    seq = "ACDEFGHIKLMNPQRSTVWYA"  # 21 residues
    name, out = register_shifted_target_decoy(seq)            # default shift=3
    assert out == seq[3:] + seq[:3]
    # A shift equal to the length is the identity rotation (degenerate, but composition-safe).
    _n, ident = register_shifted_target_decoy(seq, shift=len(seq))
    assert ident == seq
    # A shift > length wraps modulo length.
    _n2, wrapped = register_shifted_target_decoy(seq, shift=len(seq) + 3)
    assert wrapped == seq[3:] + seq[:3]


# ---------------------------------------------------------------------------
# (6) aggregate_multiseed — fold K per-seed verdict JSONs into mean +/- std (#Gate-B 3)
# ---------------------------------------------------------------------------

def test_aggregate_multiseed_means_and_spread(tmp_path):
    import json

    # Two synthetic verdict JSONs (the shape score_controls writes: {"verdict": {...}}).
    v1 = {"verdict": {"design_iptm": 0.70, "worst_offtarget_iptm": 0.50,
                      "mean_offtarget_iptm": 0.45, "SPECIFIC": True}}
    v2 = {"verdict": {"design_iptm": 0.80, "worst_offtarget_iptm": 0.60,
                      "mean_offtarget_iptm": 0.55, "SPECIFIC": False}}
    p1, p2 = tmp_path / "s42.json", tmp_path / "s7.json"
    p1.write_text(json.dumps(v1))
    p2.write_text(json.dumps(v2))

    agg = aggregate_multiseed([p1, p2])
    assert agg["n_runs"] == 2
    # Per-metric mean/std across the two runs.
    assert agg["design_iptm"]["mean"] == pytest.approx(0.75)
    assert agg["design_iptm"]["std"] == pytest.approx(0.05)   # population std of {.70,.80}
    assert agg["worst_offtarget_iptm"]["mean"] == pytest.approx(0.55)
    assert agg["mean_offtarget_iptm"]["mean"] == pytest.approx(0.50)
    # The boolean SPECIFIC is summarised as a pass fraction, not mean/std.
    assert agg["specific_fraction"] == pytest.approx(0.5)


def test_aggregate_multiseed_single_run_zero_std(tmp_path):
    import json

    p = tmp_path / "only.json"
    p.write_text(json.dumps({"verdict": {"design_iptm": 0.63, "worst_offtarget_iptm": 0.4,
                                         "SPECIFIC": True}}))
    agg = aggregate_multiseed([p])
    assert agg["n_runs"] == 1
    assert agg["design_iptm"]["mean"] == pytest.approx(0.63)
    assert agg["design_iptm"]["std"] == pytest.approx(0.0)
    assert agg["specific_fraction"] == pytest.approx(1.0)


def test_aggregate_multiseed_skips_missing_metrics(tmp_path):
    import json

    # One run is missing a metric; aggregation uses only the runs that report it.
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    p1.write_text(json.dumps({"verdict": {"design_iptm": 0.5, "SPECIFIC": True}}))
    p2.write_text(json.dumps({"verdict": {"design_iptm": 0.7, "worst_offtarget_iptm": 0.3,
                                          "SPECIFIC": True}}))
    agg = aggregate_multiseed([p1, p2])
    assert agg["design_iptm"]["mean"] == pytest.approx(0.6)
    assert agg["design_iptm"]["n"] == 2
    # worst_offtarget_iptm only present in one run.
    assert agg["worst_offtarget_iptm"]["mean"] == pytest.approx(0.3)
    assert agg["worst_offtarget_iptm"]["n"] == 1
