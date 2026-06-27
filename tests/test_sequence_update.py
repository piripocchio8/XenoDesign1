import numpy as np
import pytest
from xenodesign.sequence_update import SequenceUpdater


def _fake_design_fn(design_backbone, context_coords, context_elements, fixed_mask):
    # Return a fixed L one-letter sequence the length of the design chain.
    n = design_backbone.shape[0]
    return "A" * n


def test_update_returns_d_ccd_entity_for_all_D_chain():
    # 3-residue all-D peptide backbone (N,CA,C,CB per residue), arbitrary coords.
    design_bb = np.random.RandomState(0).rand(3, 4, 3)
    upd = SequenceUpdater(design_fn=_fake_design_fn)
    result = upd.update(
        design_backbone=design_bb,
        design_codes=["DAL", "DSN", "DLE"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    # The designed letters (AAA) are re-encoded as D-CCD for the next Chai cycle.
    assert result.d_fasta == "(DAL)(DAL)(DAL)"
    assert result.one_letter == "AAA"


def test_update_passes_designable_mask_to_design_fn():
    captured = {}

    def capture_fn(design_backbone, context_coords, context_elements, fixed_mask):
        captured["mask"] = list(fixed_mask)
        return "A" * design_backbone.shape[0]

    design_bb = np.random.RandomState(1).rand(3, 4, 3)
    upd = SequenceUpdater(design_fn=capture_fn)
    # One ncAA (SEP) in the middle -> not designable -> fixed True.
    upd.update(
        design_backbone=design_bb,
        design_codes=["DAL", "SEP", "DLE"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    # fixed_mask is True where the position is NOT designable (the ncAA position).
    assert captured["mask"] == [False, True, False]


def test_update_raises_if_no_designable_positions():
    design_bb = np.random.RandomState(2).rand(2, 4, 3)
    upd = SequenceUpdater(design_fn=_fake_design_fn)
    with pytest.raises(ValueError, match="no designable"):
        upd.update(
            design_backbone=design_bb,
            design_codes=["SEP", "AIB"],  # all ncAA
            context_coords=np.zeros((0, 3)),
            context_elements=[],
        )


def test_all_L_chain_not_reflected_geometry_matches_mask():
    captured = {}

    def capture_fn(design_backbone, context_coords, context_elements, fixed_mask):
        captured["bb"] = design_backbone.copy()
        return "A" * design_backbone.shape[0]

    design_bb = np.random.RandomState(5).rand(3, 4, 3)
    upd = SequenceUpdater(design_fn=capture_fn)
    # all-L codes -> choose_reflection False -> backbone must be passed UNREFLECTED.
    upd.update(
        design_backbone=design_bb,
        design_codes=["ALA", "SER", "LEU"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    assert np.allclose(captured["bb"], design_bb)


def test_all_D_chain_is_reflected():
    captured = {}

    def capture_fn(design_backbone, context_coords, context_elements, fixed_mask):
        captured["bb"] = design_backbone.copy()
        return "A" * design_backbone.shape[0]

    design_bb = np.random.RandomState(6).rand(3, 4, 3)
    upd = SequenceUpdater(design_fn=capture_fn)
    # all-D codes -> choose_reflection True -> backbone reflected (axis 0 negated).
    upd.update(
        design_backbone=design_bb,
        design_codes=["DAL", "DSN", "DLE"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    expected = design_bb.copy()
    expected[..., 0] *= -1.0
    assert np.allclose(captured["bb"], expected)


# --- P2: SequenceUpdater drives a new-protocol (6-arg, list-returning) backend ---
def test_update_accepts_new_protocol_backend():
    captured = {}

    def backend(design_backbone, context_coords, context_elements,
                fixed_mask, temperature, num_seqs):
        captured["temperature"] = temperature
        captured["num_seqs"] = num_seqs
        n = design_backbone.shape[0]
        return ["A" * n for _ in range(num_seqs)]

    design_bb = np.random.RandomState(7).rand(3, 4, 3)
    upd = SequenceUpdater(design_fn=backend, temperature=0.2)
    result = upd.update(
        design_backbone=design_bb,
        design_codes=["DAL", "DSN", "DLE"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    # SequenceUpdater takes the FIRST candidate (keep-best is MultiCandidate's job, Task 4).
    assert result.one_letter == "AAA"
    assert result.d_fasta == "(DAL)(DAL)(DAL)"
    # The configured temperature is forwarded; num_seqs defaults to 1 for a bare updater.
    assert captured["temperature"] == 0.2
    assert captured["num_seqs"] == 1


def test_update_legacy_four_arg_design_fn_still_works():
    # The 4-arg fakes used by the pre-P2 tests must keep working unchanged.
    def legacy(design_backbone, context_coords, context_elements, fixed_mask):
        return "A" * design_backbone.shape[0]

    design_bb = np.random.RandomState(8).rand(4, 4, 3)
    upd = SequenceUpdater(design_fn=legacy)
    result = upd.update(
        design_backbone=design_bb,
        design_codes=["DAL", "DSN", "DLE", "DAL"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    assert result.one_letter == "AAAA"


# --- P2: make_sequence_update_fn — loop-ready (pred -> D-fasta str) adapter ---
from types import SimpleNamespace
from xenodesign.sequence_update import make_sequence_update_fn, SequenceUpdater
from xenodesign.inverse_folding import MultiCandidate


def _extract(pred):
    # Map a prediction object to the four SequenceUpdater.update inputs.
    return dict(
        design_backbone=pred.design_backbone,
        design_codes=pred.design_codes,
        context_coords=pred.context_coords,
        context_elements=pred.context_elements,
    )


def test_make_sequence_update_fn_returns_d_fasta():
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["A" * bb.shape[0]] * num_seqs

    updater = SequenceUpdater(design_fn=backend)
    fn = make_sequence_update_fn(updater, _extract)
    pred = SimpleNamespace(
        design_backbone=np.random.RandomState(11).rand(3, 4, 3),
        design_codes=["DAL", "DSN", "DLE"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    # The loop expects a single one-letter L string (it maps to D-CCD itself).
    assert fn(pred) == "AAA"


def test_make_sequence_update_fn_with_multicandidate_keep_best():
    # The whole P2 stack composed: MultiCandidate(keep-best) inside a SequenceUpdater,
    # exposed as the loop's sequence_update_fn — loop.py never changes.
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["AAA", "CCC", "DDD"][:num_seqs]

    mc = MultiCandidate(backend, num_seqs=3, key_fn=lambda s: s.count("C"))
    updater = SequenceUpdater(design_fn=mc)
    fn = make_sequence_update_fn(updater, _extract)
    pred = SimpleNamespace(
        design_backbone=np.zeros((3, 4, 3)),
        design_codes=["DAL", "DSN", "DLE"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    assert fn(pred) == "CCC"  # keep-best by 'C' count


def test_make_sequence_update_fn_emit_d_fasta():
    # Optional emit='d_fasta' returns the D-CCD encoding instead of the L one-letter.
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["A" * bb.shape[0]] * num_seqs

    updater = SequenceUpdater(design_fn=backend)
    fn = make_sequence_update_fn(updater, _extract, emit="d_fasta")
    pred = SimpleNamespace(
        design_backbone=np.zeros((2, 4, 3)),
        design_codes=["DAL", "DSN"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    assert fn(pred) == "(DAL)(DAL)"


# --- ABC T1: per-position handedness mirror-back ---
def _fake_backend(bb, cc, ce, fm, temperature, num_seqs):
    return ["A" * bb.shape[0]] * num_seqs


def test_update_default_is_all_D_unchanged():
    # No chirality_pattern -> byte-identical to the existing all-D behaviour (no regression).
    upd = SequenceUpdater(design_fn=_fake_backend)
    r = upd.update(design_backbone=np.zeros((3, 4, 3)),
                   design_codes=["DAL", "DSN", "DLE"],
                   context_coords=np.zeros((0, 3)), context_elements=[])
    assert r.d_fasta == "(DAL)(DAL)(DAL)"   # whole chain D, as today


def test_update_per_position_handedness_designed_subset_only():
    # design_mask: positions 0,2 are designed (-> D); position 1 is fixed-L context (kept L).
    upd = SequenceUpdater(design_fn=_fake_backend)
    r = upd.update(
        design_backbone=np.zeros((3, 4, 3)),
        design_codes=["DAL", "ALA", "DLE"],         # mixed input: D, L, D
        context_coords=np.zeros((0, 3)), context_elements=[],
        chirality_pattern={0: "D", 1: "L", 2: "D"},  # per-position target handedness
    )
    # Designed positions (0,2) emit D-CCD; fixed position 1 stays canonical L 'A'.
    assert r.d_fasta == "(DAL)A(DAL)"


def test_update_chirality_pattern_keeps_glycine_achiral():
    # Gly stays achiral 'G' even when its position is marked 'D'.
    upd = SequenceUpdater(design_fn=lambda bb, cc, ce, fm, t, n: ["GAG"])
    r = upd.update(
        design_backbone=np.zeros((3, 4, 3)),
        design_codes=["GLY", "ALA", "GLY"],
        context_coords=np.zeros((0, 3)), context_elements=[],
        chirality_pattern={0: "D", 1: "D", 2: "D"},
    )
    assert r.d_fasta == "G(DAL)G"   # Gly achiral, only the Ala becomes D


def test_update_chirality_pattern_length_must_match():
    upd = SequenceUpdater(design_fn=_fake_backend)
    with pytest.raises(ValueError):
        upd.update(design_backbone=np.zeros((2, 4, 3)),
                   design_codes=["DAL", "DLE"],
                   context_coords=np.zeros((0, 3)), context_elements=[],
                   chirality_pattern={0: "D"})       # too short


# --- Part C: frozen_positions force coordinator positions fixed in the MPNN mask ---
def test_frozen_positions_force_fixed_mask_true():
    # Coordinator positions (0-based) must be True in fixed_mask even when they would
    # otherwise be designable (all-D codes -> every position designable after reflection).
    captured = {}

    def capture_fn(bb, cc, ce, fixed_mask, t, n):
        captured["mask"] = list(fixed_mask)
        return ["A" * bb.shape[0] for _ in range(n)]

    frozen = {5, 11, 17, 23}
    upd = SequenceUpdater(design_fn=capture_fn, frozen_positions=frozen)
    upd.update(
        design_backbone=np.zeros((24, 4, 3)),
        design_codes=["DAL"] * 24,           # all designable after reflection
        context_coords=np.zeros((0, 3)), context_elements=[],
    )
    for i in range(24):
        assert captured["mask"][i] is (i in frozen)


def test_frozen_positions_default_none_unchanged():
    # No frozen_positions -> mask is purely the designability mask (no regression).
    captured = {}

    def capture_fn(bb, cc, ce, fixed_mask, t, n):
        captured["mask"] = list(fixed_mask)
        return ["A" * bb.shape[0] for _ in range(n)]

    upd = SequenceUpdater(design_fn=capture_fn)
    upd.update(
        design_backbone=np.zeros((3, 4, 3)),
        design_codes=["DAL", "SEP", "DLE"],  # SEP ncAA -> fixed at index 1 only
        context_coords=np.zeros((0, 3)), context_elements=[],
    )
    assert captured["mask"] == [False, True, False]
