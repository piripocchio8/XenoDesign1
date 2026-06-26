import numpy as np
import pytest
from xenodesign.geometry import signed_chiral_volume
from xenodesign.chirality import L_REFERENCE_SIGN
from xenodesign.inverse_folding import prepare_inverse_folding_inputs
from tests.conftest import IDEAL_L_ALA

REFL_Z = np.diag([1.0, 1.0, -1.0])


def _d_residue_backbone():
    # Mirror an ideal L residue -> a D residue backbone (N, CA, C, CB).
    return np.stack([IDEAL_L_ALA[k] @ REFL_Z for k in ("N", "CA", "C", "CB")])


def _sign(bb_res):
    v = signed_chiral_volume(bb_res[0], bb_res[1], bb_res[2], bb_res[3])
    return 1 if v > 0 else -1


def test_design_chain_starts_D():
    d_bb = _d_residue_backbone()
    assert _sign(d_bb) == -L_REFERENCE_SIGN


def test_prepare_flips_design_chain_to_L():
    d_bb = _d_residue_backbone()[None, ...]  # (1 res, 4, 3)
    out = prepare_inverse_folding_inputs(
        d_bb, context_coords=np.zeros((0, 3)), context_elements=[], axis=2
    )
    assert _sign(out.design_backbone[0]) == L_REFERENCE_SIGN


def test_interface_distances_preserved():
    d_bb = _d_residue_backbone()[None, ...]
    ctx = np.array([[3.0, 1.0, 2.0], [-1.0, 0.5, 4.0]])
    out = prepare_inverse_folding_inputs(d_bb, ctx, ["C", "O"], axis=2)
    before = np.linalg.norm(d_bb[0, 1] - ctx[0])
    after = np.linalg.norm(out.design_backbone[0, 1] - out.context_coords[0])
    assert after == pytest.approx(before)
    assert out.context_elements == ["C", "O"]


def test_empty_context_is_handled():
    d_bb = _d_residue_backbone()[None, ...]
    out = prepare_inverse_folding_inputs(d_bb, np.zeros((0, 3)), [], axis=0)
    assert out.context_coords.shape == (0, 3)
    assert out.context_elements == []


# --- Designability logic: chirality-mixed / all-D / ncAA exceptions (spec §2.7) ---
from xenodesign.inverse_folding import (
    residue_class,
    is_designable,
    designable_positions,
    choose_reflection,
    can_use_ligandmpnn,
)


def test_residue_class_buckets():
    assert residue_class("ALA") == "L"
    assert residue_class("DAL") == "D"
    assert residue_class("GLY") == "achiral_canonical"
    assert residue_class("SEP") == "ncAA"   # phosphoserine: ncAA
    assert residue_class("AIB") == "ncAA"   # achiral ncAA, still outside MPNN alphabet


def test_all_D_chain_designable_after_flip():
    codes = ["DAL", "DSN", "DLE"]
    assert choose_reflection(codes) is True
    assert designable_positions(codes, flip=True) == [True, True, True]


def test_all_L_chain_designable_without_flip():
    codes = ["ALA", "SER", "LEU"]
    assert choose_reflection(codes) is False
    assert designable_positions(codes, flip=False) == [True, True, True]


def test_glycine_designable_in_either_frame():
    assert is_designable("GLY", flip=False) is True
    assert is_designable("GLY", flip=True) is True


def test_ncaa_never_designable():
    assert is_designable("SEP", flip=False) is False
    assert is_designable("SEP", flip=True) is False


def test_mixed_chirality_only_one_handedness_designable():
    codes = ["ALA", "DAL", "DSN"]  # 1 L, 2 D
    # flip=True makes the 2 D-> L designable (majority); the L becomes D (not designable).
    assert choose_reflection(codes) is True
    assert designable_positions(codes, flip=True) == [False, True, True]
    # without flip, only the single L is designable.
    assert designable_positions(codes, flip=False) == [True, False, False]


def test_mixed_with_ncaa_fixes_ncaa_designs_canonicals():
    codes = ["DAL", "SEP", "DLE"]  # 2 D canonicals + 1 ncAA
    assert designable_positions(codes, flip=True) == [True, False, True]


def test_all_ncaa_peptide_cannot_use_ligandmpnn():
    codes = ["SEP", "AIB", "PCA"]
    assert can_use_ligandmpnn(codes, flip=False) is False
    assert can_use_ligandmpnn(codes, flip=True) is False


# --- flip-gated reflection (review fix): geometry must match the chosen reflection ---
def test_prepare_no_flip_keeps_L_geometry():
    l_bb = np.stack([IDEAL_L_ALA[k] for k in ("N", "CA", "C", "CB")])[None, ...]  # L residue
    out = prepare_inverse_folding_inputs(
        l_bb, np.zeros((0, 3)), [], axis=2, flip=False
    )
    assert _sign(out.design_backbone[0]) == L_REFERENCE_SIGN      # stays L
    assert np.allclose(out.design_backbone, l_bb)                 # unchanged


def test_prepare_flip_reflects_to_L():
    d_bb = _d_residue_backbone()[None, ...]                       # D residue
    out = prepare_inverse_folding_inputs(
        d_bb, np.zeros((0, 3)), [], axis=2, flip=True
    )
    assert _sign(out.design_backbone[0]) == L_REFERENCE_SIGN      # D -> L


# --- P2: InverseFoldingBackend protocol (designed chain only) ---
from xenodesign.inverse_folding import InverseFoldingBackend, is_inverse_folding_backend


def _fake_backend(design_backbone, context_coords, context_elements,
                  fixed_mask, temperature, num_seqs):
    """Deterministic fake backend: returns num_seqs designed-chain L sequences,
    each the length of the design chain. NEVER returns the fixed/target chain."""
    n = design_backbone.shape[0]
    return ["A" * n for _ in range(num_seqs)]


def test_backend_returns_list_of_designed_chain_sequences():
    bb = np.zeros((5, 4, 3))
    out = _fake_backend(bb, np.zeros((0, 3)), [], [False] * 5,
                        temperature=0.1, num_seqs=3)
    assert isinstance(out, list)
    assert len(out) == 3
    # Each entry is the DESIGNED chain only, length == n_res of the design backbone.
    assert all(isinstance(s, str) and len(s) == 5 for s in out)


def test_is_inverse_folding_backend_accepts_six_arg_callable():
    # A callable with the 6-positional-arg protocol is recognised as a backend.
    assert is_inverse_folding_backend(_fake_backend) is True
    # A 4-arg legacy design_fn is NOT the new protocol.
    assert is_inverse_folding_backend(lambda a, b, c, d: "AAA") is False


def test_protocol_is_runtime_checkable():
    # InverseFoldingBackend is a runtime-checkable Protocol -> isinstance works on callables.
    assert isinstance(_fake_backend, InverseFoldingBackend)


# --- P2: output normalization — fixed chain from input, designed -> D-CCD ---
from xenodesign.inverse_folding import assemble_complex_fasta


def test_assemble_complex_fasta_designed_is_D_fixed_is_input():
    # Designed chain (3 L letters) -> D-CCD; fixed/target chain read from the known input.
    fasta = assemble_complex_fasta(
        fixed_chain_seq_from_input="ACE",   # the KNOWN fixed/target chain (L one-letter)
        designed_chain_letters="AGA",        # the designed chain (L one-letter from the backend)
    )
    # Designed chain A: D-CCD (G stays bare G, achiral); fixed chain B: kept as-is (L).
    assert fasta == (
        ">protein|design_A\n(DAL)G(DAL)\n"
        ">protein|target_B\nACE\n"
    )


def test_assemble_complex_fasta_fixed_chirality_D():
    # When the fixed/target chain is itself D (e.g. the mirror-image _L pairing), encode it D too.
    fasta = assemble_complex_fasta(
        fixed_chain_seq_from_input="AC",
        designed_chain_letters="GG",
        fixed_chirality="D",
    )
    assert fasta == (
        ">protein|design_A\nGG\n"
        ">protein|target_B\n(DAL)(DCY)\n"
    )


def test_assemble_complex_fasta_custom_names():
    fasta = assemble_complex_fasta(
        fixed_chain_seq_from_input="A",
        designed_chain_letters="A",
        design_name="binder",
        fixed_name="HLH",
    )
    assert fasta == (
        ">protein|binder\n(DAL)\n"
        ">protein|HLH\nA\n"
    )


def test_assemble_complex_fasta_never_trusts_model_for_fixed_chain():
    # The fixed chain in the output is byte-identical to the input arg — proving the
    # harness reconstructs it from the KNOWN input, not from any backend echo.
    fixed_in = "MKTAYIAKQR"
    fasta = assemble_complex_fasta(fixed_chain_seq_from_input=fixed_in,
                                   designed_chain_letters="GGG")
    assert f"\n{fixed_in}\n" in fasta


# --- P2: MultiCandidate — num_seqs sampling + keep-best ---
from xenodesign.inverse_folding import MultiCandidate


def test_multicandidate_default_keeps_first():
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        # distinct sequences so we can see which one is kept
        return ["A" * bb.shape[0], "C" * bb.shape[0], "D" * bb.shape[0]][:num_seqs]

    mc = MultiCandidate(backend, num_seqs=3)
    out = mc(np.zeros((2, 4, 3)), np.zeros((0, 3)), [], [False, False],
             temperature=0.1, num_seqs=1)
    # MultiCandidate returns a 1-element list (the winner); default key = first.
    assert out == ["AA"]


def test_multicandidate_keeps_best_by_key_fn():
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["AAAA", "CCCC", "DDDD"][:num_seqs]

    # key_fn ranks by count of 'C' (higher = better) -> "CCCC" wins.
    mc = MultiCandidate(backend, num_seqs=3, key_fn=lambda s: s.count("C"))
    out = mc(np.zeros((4, 4, 3)), np.zeros((0, 3)), [], [False] * 4,
             temperature=0.1, num_seqs=1)
    assert out == ["CCCC"]


def test_multicandidate_overrides_num_seqs_passed_by_caller():
    captured = {}

    def backend(bb, cc, ce, fm, temperature, num_seqs):
        captured["num_seqs"] = num_seqs
        return ["A" * bb.shape[0]] * num_seqs

    mc = MultiCandidate(backend, num_seqs=8)
    # Caller passes num_seqs=1 (the SequenceUpdater default); MultiCandidate forces its own 8.
    mc(np.zeros((3, 4, 3)), np.zeros((0, 3)), [], [False] * 3, temperature=0.1, num_seqs=1)
    assert captured["num_seqs"] == 8


def test_multicandidate_is_a_backend():
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["A" * bb.shape[0]] * num_seqs

    mc = MultiCandidate(backend, num_seqs=4)
    assert is_inverse_folding_backend(mc) is True
    assert isinstance(mc, InverseFoldingBackend)


# --- Stage-1: MultiCandidate top_k (ordered top-k slice; back-compat at top_k=1) ---
def test_multicandidate_top_k_returns_m_best_in_order():
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        # distinct 'C' counts so the key ordering is unambiguous: 0,1,2,3 Cs.
        return ["AAAA", "CAAA", "CCAA", "CCCA"][:num_seqs]

    # key_fn = count of 'C'; top_k=3 -> the 3 highest in descending order.
    mc = MultiCandidate(backend, num_seqs=4, key_fn=lambda s: s.count("C"), top_k=3)
    out = mc(np.zeros((4, 4, 3)), np.zeros((0, 3)), [], [False] * 4,
             temperature=0.1, num_seqs=1)
    assert out == ["CCCA", "CCAA", "CAAA"]


def test_multicandidate_top_k_one_is_byte_identical_to_old_behavior():
    # top_k=1 (the default) must reproduce the old single-winner behavior exactly.
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["AAAA", "CCCC", "DDDD"][:num_seqs]

    mc_default = MultiCandidate(backend, num_seqs=3, key_fn=lambda s: s.count("C"))
    mc_explicit = MultiCandidate(backend, num_seqs=3, key_fn=lambda s: s.count("C"), top_k=1)
    args = (np.zeros((4, 4, 3)), np.zeros((0, 3)), [], [False] * 4)
    out_default = mc_default(*args, temperature=0.1, num_seqs=1)
    out_explicit = mc_explicit(*args, temperature=0.1, num_seqs=1)
    assert out_default == ["CCCC"]
    assert out_explicit == ["CCCC"]


def test_multicandidate_top_k_one_preserves_model_order_when_no_key_fn():
    # With key_fn=None, top_k=1 keeps the first (model order), as before.
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["A" * bb.shape[0], "C" * bb.shape[0], "D" * bb.shape[0]][:num_seqs]

    mc = MultiCandidate(backend, num_seqs=3, top_k=1)
    out = mc(np.zeros((2, 4, 3)), np.zeros((0, 3)), [], [False, False],
             temperature=0.1, num_seqs=1)
    assert out == ["AA"]


def test_multicandidate_top_k_greater_than_num_seqs_raises():
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["A" * bb.shape[0]] * num_seqs

    with pytest.raises((AssertionError, ValueError)):
        MultiCandidate(backend, num_seqs=3, top_k=4)


def test_multicandidate_top_k_zero_raises():
    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["A" * bb.shape[0]] * num_seqs

    with pytest.raises((AssertionError, ValueError)):
        MultiCandidate(backend, num_seqs=3, top_k=0)


def test_multicandidate_drives_sequence_updater_keep_best():
    # End-to-end: MultiCandidate injected as a SequenceUpdater's design_fn -> loop seam.
    from xenodesign.sequence_update import SequenceUpdater

    def backend(bb, cc, ce, fm, temperature, num_seqs):
        return ["A" * bb.shape[0], "C" * bb.shape[0], "D" * bb.shape[0]][:num_seqs]

    # Keep the candidate with the most 'D' letters (toy "best" key).
    mc = MultiCandidate(backend, num_seqs=3, key_fn=lambda s: s.count("D"))
    upd = SequenceUpdater(design_fn=mc)
    result = upd.update(
        design_backbone=np.random.RandomState(3).rand(3, 4, 3),
        design_codes=["DAL", "DSN", "DLE"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    assert result.one_letter == "DDD"
    assert result.d_fasta == "(DAS)(DAS)(DAS)"
