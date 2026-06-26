"""GPU test: _ligandmpnn_design_fn produces a valid L-amino-acid sequence."""
import numpy as np
import pytest
from tests.gpu.conftest import require_cuda


@pytest.mark.gpu
def test_ligandmpnn_designs_full_length():
    require_cuda()
    from xenodesign.sequence_update import _ligandmpnn_design_fn

    rng = np.random.RandomState(0)
    bb = rng.rand(10, 4, 3) * 10.0   # 10-residue L-frame backbone
    seq = _ligandmpnn_design_fn(bb, np.zeros((0, 3)), [], [False] * 10)
    assert len(seq) == 10
    assert set(seq) <= set("ARNDCQEGHILKMFPSTWYV")


@pytest.mark.gpu
def test_ligandmpnn_fixed_mask_respected():
    """Positions marked fixed=True in fixed_mask must survive in the output.
    We seed the backbone with canonical Ala geometry so the sequence input
    to LigandMPNN is all-Ala; fixed positions must remain 'A'."""
    require_cuda()
    from xenodesign.sequence_update import _ligandmpnn_design_fn

    # Build a simple linear backbone (will not be geometrically perfect, but
    # enough for a smoke-test with fixed positions)
    n_res = 6
    rng = np.random.RandomState(42)
    bb = rng.rand(n_res, 4, 3) * 10.0
    # Fix positions 0 and 5
    fixed_mask = [True, False, False, False, False, True]
    seq = _ligandmpnn_design_fn(bb, np.zeros((0, 3)), [], fixed_mask)
    assert len(seq) == n_res
    assert set(seq) <= set("ARNDCQEGHILKMFPSTWYV")
    # Fixed positions should be 'A' (the placeholder we put in for fixed residues)
    assert seq[0] == "A", f"pos 0 fixed but got '{seq[0]}'"
    assert seq[5] == "A", f"pos 5 fixed but got '{seq[5]}'"


@pytest.mark.gpu
def test_ligandmpnn_context_changes_design():
    """Prove that context atoms actually condition the design (ligand_mpnn model).

    Design the same backbone with vs without a dense cluster of carbon atoms
    placed very close to the peptide (~3 Å).  The ligand_mpnn model consumes
    context via Y/Y_t/Y_m; protein_mpnn (old checkpoint) ignores them entirely.
    With a fixed seed, the two outputs must differ — demonstrating that the
    context signal propagates through the network rather than being silently
    discarded.
    """
    require_cuda()
    import torch
    from xenodesign.sequence_update import _ligandmpnn_design_fn

    rng = np.random.RandomState(7)
    n_res = 12
    # Place backbone atoms in a small cluster around the origin.
    bb = rng.rand(n_res, 4, 3) * 5.0  # coords in [0, 5] Å

    # ---- WITHOUT context ----
    torch.manual_seed(42)
    seq_no_ctx = _ligandmpnn_design_fn(bb, np.zeros((0, 3)), [], [False] * n_res)

    # ---- WITH context ----
    # Dense cloud of carbon atoms very close to the backbone (within cutoff_for_score=8 Å)
    # to maximize the context signal.  We use 30 atoms so even after heavy-element
    # filtering enough survive.
    ctx_coords = rng.rand(30, 3) * 4.0  # coords in [0, 4] Å, overlapping backbone
    ctx_elements = ["C"] * 30           # carbon (heavy element, passes Y_m filter)
    torch.manual_seed(42)
    seq_with_ctx = _ligandmpnn_design_fn(bb, ctx_coords, ctx_elements, [False] * n_res)

    # Both must be valid sequences.
    assert len(seq_no_ctx) == n_res, f"no-ctx length {len(seq_no_ctx)} != {n_res}"
    assert len(seq_with_ctx) == n_res, f"with-ctx length {len(seq_with_ctx)} != {n_res}"
    assert set(seq_no_ctx) <= set("ARNDCQEGHILKMFPSTWYV"), f"invalid chars: {seq_no_ctx}"
    assert set(seq_with_ctx) <= set("ARNDCQEGHILKMFPSTWYV"), f"invalid chars: {seq_with_ctx}"

    print(f"\nDesign WITHOUT context: {seq_no_ctx}")
    print(f"Design WITH    context: {seq_with_ctx}")

    # The sequences must differ — this is the key assertion: context is consumed.
    assert seq_no_ctx != seq_with_ctx, (
        "Context-conditioned and context-free designs are identical — "
        "the context tensors are NOT being consumed by the model. "
        f"Both: {seq_no_ctx!r}"
    )
