"""CPU tests for the 6UFA cyclic Zn-macrocycle design driver (task #9).

All tests here are pure-CPU: they exercise the seed construction, the mixed-chirality
FASTA builder, the Zn-ligand FASTA emission, the metal-coordination restraint wiring,
and the RMSD-to-deposit / Zn-N geometry scorers on SYNTHETIC coordinates. Nothing here
touches chai/torch (the GPU path is `run_cyclic_design`, marked elsewhere / not imported).

The unpublished design sequences are never inlined; the 6UFA DEPOSIT sequence is public
(RCSB 6UFA) but we still avoid hard-coding coords — synthetic arrays drive the geometry
tests so they are deterministic and deposit-file-independent.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.design_cyclic import (
    CYCLIC_HIS_CHIRALITY,
    ZN_SMILES,
    backbone_rmsd_to_deposit,
    build_closure_row,
    build_cyclic_input_fasta,
    build_cyclic_restraint_rows,
    build_cyclic_seed,
    mixed_chirality_fasta,
    write_cyclic_restraints,
    zn_coordination_geometry,
)
from xenodesign.benchmark.cases import get_case


# ── Seed construction (insert_fixed_chirality at the case His positions) ────────

def test_build_cyclic_seed_length_and_his_placement():
    case = get_case("cyclic")
    seed = build_cyclic_seed(case, seed_seq="ACDEFGHIKLMNACDEFGHIKLMN")  # len 24, deterministic
    # length preserved, His placed at the 4 deposit coordinating positions (1-based 6/12/18/24)
    assert seed.length == case.binder_length == 24
    assert len(seed.one_letter) == 24
    for pos in (6, 12, 18, 24):
        assert seed.one_letter[pos - 1] == "H"
    # fixed_chirality records the per-position handedness exactly as the policy table.
    assert seed.fixed_chirality == CYCLIC_HIS_CHIRALITY
    assert seed.conditioned is False  # cyclic is the unconditioned path (no metal-aware pLM)


def test_build_cyclic_seed_random_is_deterministic_with_seed():
    case = get_case("cyclic")
    a = build_cyclic_seed(case, rng_seed=7)
    b = build_cyclic_seed(case, rng_seed=7)
    assert a.one_letter == b.one_letter
    # His still pinned at the coordinating positions even for a random backbone seed.
    for pos in CYCLIC_HIS_CHIRALITY:
        assert a.one_letter[pos - 1] == "H"


def test_build_cyclic_seed_rejects_wrong_length_explicit_seed():
    case = get_case("cyclic")
    with pytest.raises(ValueError):
        build_cyclic_seed(case, seed_seq="TOOSHORT")  # != binder_length 12


# ── Mixed-chirality FASTA (per-position L vs D, NOT all-D) ──────────────────────

def test_mixed_chirality_fasta_places_d_only_at_d_positions():
    # 4-mer seq, His at all 4; mark positions 2 and 4 as D, 1 and 3 as L.
    seq = "HHHH"
    fixed = {1: "L", 2: "D", 3: "L", 4: "D"}
    out = mixed_chirality_fasta(seq, fixed)
    # L-His stays bare 'H'; D-His becomes the parenthesized D-CCD block (DHI).
    assert out == "H(DHI)H(DHI)"


def test_mixed_chirality_fasta_unmarked_positions_default_l():
    # Positions not in `fixed` are L (bare canonical) — this is the LINEAR phase-1 seed
    # where only the coordinating His are pinned to a handedness.
    seq = "ACDH"
    fixed = {4: "D"}  # only the His is D
    out = mixed_chirality_fasta(seq, fixed)
    assert out == "ACD(DHI)"


def test_mixed_chirality_fasta_glycine_stays_bare():
    # Glycine is achiral: stays a single 'G' regardless of any handedness request.
    out = mixed_chirality_fasta("GHG", {1: "D", 2: "L", 3: "D"})
    assert out == "GHG"


def test_mixed_chirality_fasta_passes_ncaa_blocks_through():
    # track #2: a Variant-B identity may already carry an ncAA as a (XXX) block; it must be
    # emitted verbatim (chai's modified-residue contract), not looked up as a 1-letter code.
    out = mixed_chirality_fasta("A(AIB)C", {1: "L", 3: "L"})
    assert out == "A(AIB)C"


def test_mixed_chirality_fasta_ncaa_block_unaffected_by_chirality_marks():
    # An ncAA block is one position; a D mark on a neighbouring canonical still applies, but the
    # (XXX) block itself is passed through unchanged (D-ncAA, if any, is already encoded in it).
    out = mixed_chirality_fasta("(NLE)H", {1: "L", 2: "D"})
    assert out == "(NLE)(DHI)"


# ── B5: per-coordinator L/D chirality applied through the seq-update loop ────────

def test_cyclic_seq_update_builds_per_coordinator_chirality_pattern():
    """The cyclic seq_update must thread a chirality_pattern pinning each coordinator's DECLARED
    handedness (L@6,D@12,L@18,D@24), not blanket-flip everything to D."""
    from types import SimpleNamespace
    from xenodesign.classes.cyclic import Cyclic
    from xenodesign.config import resolve_config

    captured = {}

    def fake_make(wrapper, *, num_seqs, backend, roles,
                  frozen_positions, coordinators, chirality_pattern):
        captured["frozen_positions"] = frozen_positions
        captured["coordinators"] = coordinators
        captured["chirality_pattern"] = chirality_pattern
        return lambda pred: "x"

    coords = [(6, "H", "HIS", "L"), (12, "H", "HIS", "D"),
              (18, "H", "HIS", "L"), (24, "H", "HIS", "D")]
    cfg = resolve_config("cyclic", target_type="metal",
                         cli_overrides={"use_pepmlm": False,
                                        "restraint.params": {"coord_residues": coords}})

    import xenodesign.classes.alpha as alpha_mod
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(alpha_mod, "make_alpha_seq_update_fn", fake_make)
        Cyclic().seq_update(cfg, SimpleNamespace(), SimpleNamespace())

    # 0-based coordinator positions 5/11/17/23 with their DECLARED handedness.
    assert captured["chirality_pattern"] == {5: "L", 11: "D", 17: "L", 23: "D"}
    assert captured["frozen_positions"] == {5, 11, 17, 23}


def test_cyclic_chirality_emits_bare_H_for_L_and_DHI_for_D():
    """End-to-end at the SequenceUpdater level with the cyclic coordinator wiring: L-His coords
    (6/18) emit bare 'H', D-His coords (12/24) emit '(DHI)' — NOT everything -> D."""
    import numpy as np
    from xenodesign.abc.moves import identity_tokens
    from xenodesign.sequence_update import SequenceUpdater

    # A backend that echoes known_seq at fixed positions (LigandMPNN's native fixed-pos behaviour).
    def echo_backend(bb, cc, ce, fm, t, n, known_seq=None):
        return ["".join(known_seq[i] if fm[i] else "A" for i in range(bb.shape[0]))
                for _ in range(n)]

    # Coordinators (1-based) L@6,D@12,L@18,D@24 -> 0-based 5/11/17/23.
    coord0 = {5: "L", 11: "D", 17: "L", 23: "D"}
    n = 24
    # design_codes carry His at coordinators (L-His -> HIS, D-His -> DHI); rest all-D Ala.
    design_codes = ["DAL"] * n
    for i, hand in coord0.items():
        design_codes[i] = "HIS" if hand == "L" else "DHI"
    chirality_pattern = {i: coord0.get(i, "D") for i in range(n)}

    upd = SequenceUpdater(design_fn=echo_backend, frozen_positions=set(coord0))
    r = upd.update(
        design_backbone=np.zeros((n, 4, 3)),
        design_codes=design_codes,
        context_coords=np.zeros((0, 3)), context_elements=[],
        chirality_pattern=chirality_pattern,
    )
    toks = identity_tokens(r.d_fasta)
    assert toks[5] == "H" and toks[17] == "H"          # L-His coords -> bare H
    assert toks[11] == "(DHI)" and toks[23] == "(DHI)"  # D-His coords -> D-CCD
    # The identity is His at all four coordinators (preserved natively, not Ala).
    for i in coord0:
        assert r.one_letter[i] == "H"


def test_cyclic_seq_update_d_fasta_through_loop_keeps_L_coordinators_bare_H():
    """INTEGRATION (the all-D regression): run the cyclic seq-update through the REAL
    make_alpha_seq_update_fn (with CPU fakes for the predictor extractor + base backend),
    then through the loop's _to_d_fasta_safe re-encode — exactly the greedy loop path
    (loop.py: new_seq = seq_update_fn(pred); state.d_fasta = _to_d_fasta_safe(new_seq)).

    The d_fasta that would be fed to Chai the NEXT iteration MUST carry bare 'H' at the L
    coordinators (6/18 -> 0-based 5/17) and '(DHI)' at the D coordinators (12/24 -> 11/23),
    NOT the all-D '(DHI)x24' that to_d_fasta produces. This reproduces the GPU-confirmed
    all-D regression (selected_d_fasta = (DHI)x23): coordinators 6/18 came out D."""
    import numpy as np
    from types import SimpleNamespace

    import xenodesign.classes.alpha as alpha_mod
    import xenodesign.classes._alpha_internals as ai
    from xenodesign.abc.moves import identity_tokens
    from xenodesign.loop import _to_d_fasta_safe

    n = 24
    # Coordinators (1-based) L@6,D@12,L@18,D@24 -> 0-based 5/11/17/23.
    coord_residues = [(6, "H", "HIS", "L"), (12, "H", "DHI", "D"),
                      (18, "H", "HIS", "L"), (24, "H", "DHI", "D")]
    frozen = {int(t[0]) - 1 for t in coord_residues}
    coordinators = [(int(t[0]) - 1, t[1]) for t in coord_residues]
    chirality_pattern = {int(t[0]) - 1: t[3] for t in coord_residues}

    # Base backend that echoes known_seq at fixed positions (LigandMPNN's native behaviour),
    # 'A' elsewhere — so the four His coordinators are preserved by identity.
    def echo_backend(bb, cc, ce, fm, t, num_seqs, known_seq=None):
        return ["".join(known_seq[i] if fm[i] else "A" for i in range(bb.shape[0]))
                for _ in range(num_seqs)]

    # A fake wrapper holding a CIF dir (its presence is all _extract needs since we fake the
    # cif readers). last_out_dir must be set or _extract raises.
    wrapper = SimpleNamespace(last_out_dir="/tmp/fake_iter")

    # design_codes the cyclic _extract would build: His CCD at coordinators (L->HIS, D->DHI),
    # all-D Ala elsewhere. We fake the backbone/context readers to return synthetic data; the
    # design_codes themselves are built INSIDE make_alpha_seq_update_fn from `coordinators` +
    # `chirality_pattern`, so we only need to fake the geometry readers.
    fake_bb = [{"N": (0.0, 0.0, 0.0), "CA": (1.0, 0.0, 0.0), "C": (2.0, 0.0, 0.0),
                "CB": (1.0, 1.0, 0.0)} for _ in range(n)]

    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(ai, "_make_base_backend", lambda backend: echo_backend)
        # _cterm_gly_anchor wraps the base; keep it (it forces a C-term Gly). The anchor is fine
        # here — position 23 (C-term) is a declared coordinator so it stays His via known_seq…
        # actually _cterm_gly_anchor force-sets fm[-1]=True and overwrites last char to 'G'. To
        # keep the test focused on chirality (not the anchor), neutralize the anchor.
        mp.setattr(ai, "_cterm_gly_anchor", lambda fn, *_, **__: fn)
        # Fake the CIF readers used by _extract (and _stage_extract on the S1.7 stage path).
        # binder_seq_from_cif is called by _stage_extract to supply prev_l_seq; return a
        # length-24 placeholder — the echo_backend will overwrite it via known_seq anyway.
        mp.setattr(ai, "_self", lambda: SimpleNamespace(
            _make_base_backend=lambda backend: echo_backend,
            _cterm_gly_anchor=lambda fn, *_, **__: fn,
            _best_cif_path=lambda d: "/tmp/fake.cif",
            _all_atoms_from_chain=lambda cif, ch: (np.zeros((0, 3)), []),
            binder_seq_from_cif=lambda cif, chain: "A" * n,
        ))
        # backbone_by_residue_from_cif is imported inside make_alpha_seq_update_fn from
        # eval.gate_tier0a; patch it there.
        import xenodesign.eval.gate_tier0a as g0
        mp.setattr(g0, "backbone_by_residue_from_cif",
                   lambda cif, chain: fake_bb)

        seq_update_fn = alpha_mod.make_alpha_seq_update_fn(
            wrapper, num_seqs=1, backend="ligandmpnn", roles=None,
            frozen_positions=frozen, coordinators=coordinators,
            chirality_pattern=chirality_pattern)

        pred = SimpleNamespace(coords=np.zeros((n, 3)), iptm=0.5)
        new_seq = seq_update_fn(pred)

    # The greedy loop re-encodes the seq_update output via _to_d_fasta_safe before feeding
    # Chai the next iteration. This is where the all-D override historically struck.
    d_fasta = _to_d_fasta_safe(new_seq)
    toks = identity_tokens(d_fasta)
    assert len(toks) == n
    # L coordinators (6/18 -> 5/17) MUST be bare 'H', NOT '(DHI)'.
    assert toks[5] == "H", f"L coord 6 came out {toks[5]!r} (all-D regression)"
    assert toks[17] == "H", f"L coord 18 came out {toks[17]!r} (all-D regression)"
    # D coordinators (12/24 -> 11/23) stay '(DHI)'.
    assert toks[11] == "(DHI)"
    assert toks[23] == "(DHI)"


def test_seed_d_fasta_keeps_L_coordinators_bare_for_mixed_seed():
    """The iter_000 seed encode (_seed_d_fasta) must honour a mixed L/D fixed_chirality: L
    coordinators stay bare canonical, every other position is D-CCD — NOT whole-chain all-D."""
    from types import SimpleNamespace
    from xenodesign.abc.moves import identity_tokens
    from xenodesign.dispatch import _seed_d_fasta

    # His at 1-based 6/12/18/24 (L/D/L/D), Ala elsewhere; cyclic-style 24-mer seed.
    one = list("A" * 24)
    for p in (6, 12, 18, 24):
        one[p - 1] = "H"
    seed = SimpleNamespace(one_letter="".join(one),
                           fixed_chirality={6: "L", 12: "D", 18: "L", 24: "D"})
    d_fasta = _seed_d_fasta(seed)
    toks = identity_tokens(d_fasta)
    assert toks[5] == "H" and toks[17] == "H"           # L coords -> bare H
    assert toks[11] == "(DHI)" and toks[23] == "(DHI)"  # D coords -> D-CCD
    assert toks[0] == "(DAL)"                            # non-coord backbone stays all-D


def test_seed_d_fasta_all_D_seed_unchanged():
    """A pure all-D seed (no 'L' pinned — the alpha/non_alpha default) is encoded byte-for-byte
    by the legacy to_d_fasta path."""
    from types import SimpleNamespace
    from xenodesign.dispatch import _seed_d_fasta
    from xenodesign.io_spec import to_d_fasta

    seed = SimpleNamespace(one_letter="ACDEFG", fixed_chirality={})
    assert _seed_d_fasta(seed) == to_d_fasta("ACDEFG")


# ── B6: Gly never clobbers a coordinator; only fires when truly no L present ─────

def test_ensure_canonical_anchor_excludes_coordinators():
    """A coordinator at the C-terminus is never overwritten by the Gly anchor."""
    from xenodesign.classes.cyclic import Cyclic

    # All-D non-coordinator backbone (would normally trigger the Gly anchor), His coordinator at
    # the C-TERMINUS (1-based 4). The anchor must place Gly at a NON-coordinator position, never
    # clobbering the C-terminal coordinator.
    one = "AAAH"
    fixed = {4: "D"}  # coordinator at the C-term (1-based)
    out = Cyclic._ensure_canonical_anchor(one, fixed, default_chirality="D")
    assert out[3] == "H"          # C-terminal coordinator preserved
    assert "G" in out             # Gly anchor placed somewhere among non-coordinators
    assert out.index("G") != 3    # NOT at the coordinator position


def test_ensure_canonical_anchor_no_forced_gly_when_backbone_mixed():
    """The metal loop emits a genuinely mixed L/D non-coordinator backbone (an L IS present), so
    the 'no L present' trigger must NOT fire -> no forced Gly clobbering the designed sequence."""
    from xenodesign.classes.cyclic import Cyclic

    # Coordinators at 1-based 1/3 (His), non-coordinators at 2 (L) and 4 (D) -> mixed.
    one = "HAHA"
    fixed = {1: "L", 3: "D"}
    chirality_map = {2: "L", 4: "D"}   # non-coordinator backbone is mixed L/D
    out = Cyclic._ensure_canonical_anchor(one, fixed, chirality_map=chirality_map,
                                          default_chirality="D")
    assert out == one            # unchanged: mixed backbone already tokenizable
    assert "G" not in out


# ── Zn-ligand FASTA emission (the metal/HETATM context) ─────────────────────────

def test_build_cyclic_input_fasta_has_protein_and_zn_ligand():
    # METAL-(b): by default the metal is fed as a CCD residue (`>ligand|name=ZN` + 'ZN'), the form
    # chai_patches._patch_ligand_ccd_feeding recognizes (resolvable residue ZN, atom ZN).
    fasta = build_cyclic_input_fasta(
        binder_mixed_seq="H(DHI)H(DHI)", binder_name="binder", zn_name="zn"
    )
    lines = fasta.strip().splitlines()
    # protein chain first (so chai labels it chain A), Zn ligand second (chain B).
    assert lines[0] == ">protein|binder"
    assert lines[1] == "H(DHI)H(DHI)"
    assert lines[2] == ">ligand|name=ZN"
    assert lines[3] == "ZN"        # the CCD code on the sequence line (cached-conformer path)


def test_build_cyclic_input_fasta_smiles_fallback():
    # zn_ccd=None falls back to the SMILES form, kept for non-CCD metals.
    fasta = build_cyclic_input_fasta(
        binder_mixed_seq="H(DHI)", binder_name="binder", zn_name="zn", zn_ccd=None
    )
    lines = fasta.strip().splitlines()
    assert lines[2] == ">ligand|name=zn"
    assert lines[3] == ZN_SMILES  # the zinc SMILES, e.g. '[Zn+2]'


def test_zn_smiles_is_zinc_ion():
    # The bare metal-cation SMILES, kept for the SMILES fallback path; zinc(II) is '[Zn+2]'.
    assert ZN_SMILES == "[Zn+2]"


# ── Metal-coordination restraint wiring (His<->Zn, via build_for_case) ──────────

def test_build_cyclic_restraint_rows_one_contact_per_his():
    case = get_case("cyclic")
    rows = build_cyclic_restraint_rows(case)
    # 4 coordinating His in the full 24-mer (6/12/18/24, L/D/L/D) -> 4 inter-chain contact rows
    # (His chain A <-> Zn chain B).
    assert len(rows) == 4
    his_resnums = case.restraint.params["his_resnums"]
    assert len(his_resnums) == 4
    for row, hr in zip(rows, his_resnums):
        cols = row.split(",")
        # chainA = His chain, res token '<H><resnum>'; chainB = Zn metal chain.
        assert cols[0] == case.restraint.params["his_chain"]
        assert cols[1] == f"H{hr}"
        assert cols[2] == case.restraint.params["metal_chain"]
        assert cols[4] == "contact"


# ── Backbone heavy-atom RMSD to the deposit (the RECALL metric) ─────────────────

def test_backbone_rmsd_zero_for_identical_coords():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(12, 3))
    assert backbone_rmsd_to_deposit(coords, coords) == pytest.approx(0.0, abs=1e-9)


def test_backbone_rmsd_invariant_under_rigid_motion():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(12, 3))
    # rotate + translate b; Kabsch-aligned RMSD must be ~0.
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    b = a @ rot.T + np.array([3.0, -2.0, 5.0])
    assert backbone_rmsd_to_deposit(a, b) == pytest.approx(0.0, abs=1e-6)


def test_backbone_rmsd_positive_for_perturbed_coords():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(12, 3))
    b = a + rng.normal(scale=0.5, size=(12, 3))
    assert backbone_rmsd_to_deposit(a, b) > 0.0


def test_backbone_rmsd_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        backbone_rmsd_to_deposit(np.zeros((12, 3)), np.zeros((10, 3)))


# ── Zn-N coordination geometry (secondary metric) ───────────────────────────────

def test_zn_coordination_geometry_tetrahedral_distances():
    # Put 4 N atoms at unit distance along tetrahedral directions; Zn at origin.
    zn = np.zeros(3)
    tetra = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True) * 2.0  # 2.0 A each
    geom = zn_coordination_geometry(zn, tetra)
    assert geom["n_coordinating"] == 4
    assert geom["mean_zn_n_distance"] == pytest.approx(2.0, abs=1e-6)
    assert geom["max_zn_n_distance"] == pytest.approx(2.0, abs=1e-6)
    # ideal tetrahedral angle ~109.47 deg
    assert geom["mean_n_zn_n_angle"] == pytest.approx(109.47, abs=0.5)


def test_zn_coordination_geometry_counts_within_cutoff():
    zn = np.zeros(3)
    # two N within 2.6 A, one beyond (4.0 A)
    ns = np.array([[2.0, 0, 0], [0, 2.5, 0], [0, 0, 4.0]], dtype=float)
    geom = zn_coordination_geometry(zn, ns, cutoff=2.6)
    assert geom["n_coordinating"] == 2  # the 4.0-A N is not counted as coordinating


def test_zn_coordination_geometry_empty_when_no_neighbors():
    zn = np.zeros(3)
    ns = np.array([[10.0, 0, 0]], dtype=float)
    geom = zn_coordination_geometry(zn, ns, cutoff=2.6)
    assert geom["n_coordinating"] == 0
    assert geom["mean_zn_n_distance"] is None  # nothing within cutoff to average


# ── His-position deposit-vs-case consistency guard (documents the discrepancy) ──

def test_case_his_positions_synced_to_deposit():
    # 2026-06-24: the registry models the FULL S2-symmetric 6UFA 24-mer. The four coordinating
    # His are 6/12/18/24 with chirality L/D/L/D (module docstring / DEPOSIT REALITY) — a single
    # 12-mer cannot make the 4-coordinate [Zn(His)4] site. This asserts the corrected positions
    # AND chirality, plus self-consistency between the seeding map and the restraint his_resnums
    # (so the two never drift apart again).
    case = get_case("cyclic")
    assert case.restraint.params["his_resnums"] == (6, 12, 18, 24)
    assert CYCLIC_HIS_CHIRALITY == {6: "L", 12: "D", 18: "L", 24: "D"}
    assert tuple(sorted(CYCLIC_HIS_CHIRALITY)) == case.restraint.params["his_resnums"]


# ── P2b: head-to-tail covalent closure (#23) ──────────────────────────────────────

def test_build_closure_row_is_covalent_n_to_c():
    case = get_case("cyclic")
    seed = build_cyclic_seed(case, seed_seq="ACDEFGHIKLMNACDEFGHIKLMN")   # His placed at 6/12/18/24
    row = build_closure_row(seed)
    cols = row.split(",")
    assert cols[4] == "covalent"
    # C-term residue (pos 24) carbonyl C  <->  N-term residue (pos 1) amide N.
    assert cols[1].endswith("24@C") and cols[3].endswith("1@N")
    assert cols[3].startswith(seed.one_letter[0])   # N-term residue one-letter


def test_write_cyclic_restraints_closure_appends_one_covalent(tmp_path):
    case = get_case("cyclic")
    seed = build_cyclic_seed(case, seed_seq="ACDEFGHIKLMNACDEFGHIKLMN")
    base = write_cyclic_restraints(case, tmp_path / "noclose", seed_result=seed, closure=False)
    closed = write_cyclic_restraints(case, tmp_path / "close", seed_result=seed, closure=True)
    n_base = sum(1 for _ in base.read_text().splitlines()[1:])     # minus header
    n_closed = sum(1 for _ in closed.read_text().splitlines()[1:])
    assert n_closed == n_base + 1                                  # exactly one closure bond
    assert "covalent" in closed.read_text() and "cyclic_closure" in closed.read_text()


# ── Part F: result provenance — recorded sequence must match the PANEL-selected step ──

def test_assemble_records_panel_selected_sequence_not_greedy(tmp_path):
    """The recorded selected_d_fasta must come from the PANEL-selected step (whose CIF is the
    deposited model), NOT the greedy highest-score step — mirroring the alpha path. Here the
    greedy best (highest score) is step 0, but the panel selects step 1."""
    from xenodesign.classes.cyclic import _assemble_cyclic_result
    from xenodesign.config import resolve_config
    from xenodesign.judges.panel import PanelResult, RefereeScore

    class _Pred:
        def __init__(self, iptm, ptm):
            self.iptm, self.ptm = iptm, ptm

    class _State:
        def __init__(self, d_fasta):
            self.d_fasta = d_fasta

    class _Step:
        def __init__(self, d_fasta, iptm, ptm, score):
            self.prediction = _Pred(iptm, ptm)
            self.state = _State(d_fasta)
            self.score = score

    # step 0 has the HIGHER score (greedy would pick it); step 1 is the panel pick.
    history = [_Step("(DHI)GREEDY", 0.9, 0.9, 0.99),
               _Step("(DHI)PANEL", 0.5, 0.5, 0.10)]
    raw = [RefereeScore(chirality_violation=0.0, iptm=0.9),
           RefereeScore(chirality_violation=0.0, iptm=0.5)]
    panel = PanelResult(selected_idx=1, composite_scores=[0.1, 0.9],
                        vetoed=[False, False], raw_scores=raw)

    cfg = resolve_config("cyclic", target_type="metal", out_dir=str(tmp_path))
    case = get_case("cyclic")
    result = _assemble_cyclic_result(cfg, history, panel_result=panel,
                                     case=case, out_dir=tmp_path)
    assert result["selected_d_fasta"] == "(DHI)PANEL"
    assert result["selected_iptm"] == 0.5  # ipTM read from the SAME (panel) step
