"""CPU contract tests for the CARBonAra inverse-folding backend (xenodesign/carbonara_backend.py).

CARBonAra is research-only (CC-BY-NC-SA) and runs in its own .venv; it is NEVER imported into
this process and NEVER called for real here. The subprocess seam (`_run_carbonara`) is patched
to return canned per-sample [design_seq, known_seq] pairs, so these tests touch no GPU, no
network, and no real CARBonAra/env. The adapter is chirality-AGNOSTIC (L-in/L-out): it mirrors
`_ligandmpnn_design_fn` so the upstream double-flip and downstream D-CCD re-encode are reused
unchanged (see inverse_folding.py / sequence_update.py).
"""
import os
import pathlib
import sys

import numpy as np
import pytest

import xenodesign.carbonara_backend as cb
from xenodesign.inverse_folding import (
    InverseFoldingBackend,
    is_inverse_folding_backend,
)

ALPHABET = set("ARNDCQEGHILKMFPSTWYV")


# --------------------------------------------------------------------------------------------
# Helpers: a canned _run_carbonara that records its call args and returns [design, known] pairs.
# --------------------------------------------------------------------------------------------
def _patch_run(monkeypatch, design_seqs, known_seq="MQIFVKTLTGKT"):
    """Patch the subprocess seam to return one [design_seq, known_seq] pair per sample.

    Returns the capture dict so tests can assert what the adapter forwarded (num_seqs,
    known_positions, known_chains, the temp PDB path, etc.).
    """
    captured = {}

    def fake_run(pdb_path, out_dir, num_seqs, known_positions, known_chains, temperature):
        captured["pdb_path"] = str(pdb_path)
        captured["out_dir"] = str(out_dir)
        captured["num_seqs"] = num_seqs
        captured["known_positions"] = list(known_positions)
        captured["known_chains"] = list(known_chains)
        captured["temperature"] = temperature
        return [[ds, known_seq] for ds in design_seqs[:num_seqs]]

    monkeypatch.setattr(cb, "_run_carbonara", fake_run)
    return captured


# --------------------------------------------------------------------------------------------
# Protocol conformance.
# --------------------------------------------------------------------------------------------
def test_design_fn_is_inverse_folding_backend():
    assert is_inverse_folding_backend(cb.carbonara_design_fn) is True
    assert isinstance(cb.carbonara_design_fn, InverseFoldingBackend)


def test_design_fn_has_six_required_plus_optional_known_seq():
    # B2: the contract now carries an OPTIONAL 7th positional param `known_seq` (the real
    # L-projected sequence). The first six are unchanged; known_seq has a default so legacy
    # 6-arg callers still work.
    import inspect

    params = [
        p for p in inspect.signature(cb.carbonara_design_fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    assert [p.name for p in params] == [
        "design_backbone", "context_coords", "context_elements",
        "fixed_mask", "temperature", "num_seqs", "known_seq",
    ]
    assert params[-1].default is None   # known_seq is optional


def test_returns_num_seqs_l_sequences_of_len_n_res(monkeypatch):
    n_res = 4
    _patch_run(monkeypatch, design_seqs=["GMDR", "AKMA", "YLMG"])
    bb = np.zeros((n_res, 4, 3))
    out = cb.carbonara_design_fn(bb, np.zeros((0, 3)), [], [False] * n_res,
                                 temperature=0.1, num_seqs=3)
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(s, str) and len(s) == n_res for s in out)


def test_returned_sequences_are_in_canonical_alphabet(monkeypatch):
    n_res = 4
    _patch_run(monkeypatch, design_seqs=["GMDR", "AKMA"])
    bb = np.zeros((n_res, 4, 3))
    out = cb.carbonara_design_fn(bb, np.zeros((0, 3)), [], [False] * n_res,
                                 temperature=0.1, num_seqs=2)
    for s in out:
        assert set(s) <= ALPHABET


def test_asserts_len_equals_design_backbone(monkeypatch):
    # A canned design seq of the WRONG length must trip the len==n_res guard.
    _patch_run(monkeypatch, design_seqs=["GMD"])  # 3 letters
    bb = np.zeros((4, 4, 3))                        # but 4 residues expected
    with pytest.raises((AssertionError, ValueError)):
        cb.carbonara_design_fn(bb, np.zeros((0, 3)), [], [False] * 4,
                               temperature=0.1, num_seqs=1)


def test_known_seq_wrong_length_raises(monkeypatch):
    # FAIL FAST: a known_seq whose length != n_res would otherwise 'A'-pad its tail, silently
    # dropping pinned coordinator identities. It must raise a clear ValueError instead.
    _patch_run(monkeypatch, design_seqs=["GMDR"])
    bb = np.zeros((4, 4, 3))                  # n_res == 4
    with pytest.raises(ValueError, match="known_seq length"):
        cb.carbonara_design_fn(bb, np.zeros((0, 3)), [], [False] * 4,
                               temperature=0.1, num_seqs=1, known_seq="GMD")  # len 3


def test_known_seq_correct_length_still_works(monkeypatch):
    # A correctly-sized known_seq is accepted; its identity is re-imposed at fixed positions.
    _patch_run(monkeypatch, design_seqs=["GMDR"])
    bb = np.zeros((4, 4, 3))
    out = cb.carbonara_design_fn(bb, np.zeros((0, 3)), [],
                                 [True, False, False, False], temperature=0.1,
                                 num_seqs=1, known_seq="CKLY")  # len 4 == n_res
    assert len(out) == 1 and len(out[0]) == 4
    assert out[0][0] == "C"  # fixed position 0 restored from known_seq


# --------------------------------------------------------------------------------------------
# fixed_mask (0-based) -> known_positions (1-based on the design chain) — OFF-BY-ONE.
# --------------------------------------------------------------------------------------------
def test_fixed_mask_maps_to_one_based_known_positions(monkeypatch):
    # fixed_mask True at 0-based indices 1 and 3 -> 1-based known_positions [2, 4].
    captured = _patch_run(monkeypatch, design_seqs=["GMDR"])
    bb = np.zeros((4, 4, 3))
    cb.carbonara_design_fn(bb, np.zeros((0, 3)), [],
                           [False, True, False, True], temperature=0.1, num_seqs=1)
    assert captured["known_positions"] == [2, 4]
    assert captured["known_chains"] == ["B"]


def test_no_fixed_positions_gives_empty_known_positions(monkeypatch):
    captured = _patch_run(monkeypatch, design_seqs=["GMDR"])
    bb = np.zeros((4, 4, 3))
    cb.carbonara_design_fn(bb, np.zeros((0, 3)), [], [False] * 4,
                           temperature=0.1, num_seqs=1)
    assert captured["known_positions"] == []


# --------------------------------------------------------------------------------------------
# Design-chain-only parse: discard the known chain B echoed after the colon.
# --------------------------------------------------------------------------------------------
def test_parses_design_chain_discards_known_chain(monkeypatch):
    # _run_carbonara returns [design, known] pairs; the adapter must keep ONLY the design seq.
    n_res = 4
    _patch_run(monkeypatch, design_seqs=["GMDR", "AKMA"],
               known_seq="MQIFVKTLTGKT")
    bb = np.zeros((n_res, 4, 3))
    out = cb.carbonara_design_fn(bb, np.zeros((0, 3)), [], [False] * n_res,
                                 temperature=0.1, num_seqs=2)
    assert out == ["GMDR", "AKMA"]
    # the known chain (ubiquitin) is never present in any returned sequence
    assert all("MQIFVKTLTGKT" not in s for s in out)


# --------------------------------------------------------------------------------------------
# Fixed positions -> deterministic placeholder ('A'), like sequence_update.py:412-414.
# --------------------------------------------------------------------------------------------
def test_fixed_positions_reapplied_as_placeholder_A(monkeypatch):
    # No known_seq (legacy) -> fixed positions fall back to the 'A' placeholder.
    _patch_run(monkeypatch, design_seqs=["GMDR"])
    bb = np.zeros((4, 4, 3))
    out = cb.carbonara_design_fn(bb, np.zeros((0, 3)), [],
                                 [True, False, True, False], temperature=0.1, num_seqs=1)
    # positions 0 and 2 forced to 'A'; designed positions 1,3 kept ('M','R').
    assert out == ["AMAR"]


def test_fixed_positions_restored_from_known_seq(monkeypatch):
    # B2: with a known_seq, fixed positions are restored to their REAL identity, not 'A'.
    _patch_run(monkeypatch, design_seqs=["GMDR"])
    bb = np.zeros((4, 4, 3))
    out = cb.carbonara_design_fn(bb, np.zeros((0, 3)), [],
                                 [True, False, True, False], temperature=0.1, num_seqs=1,
                                 known_seq="HKHE")
    # fixed positions 0,2 -> known_seq 'H','H'; designed positions 1,3 kept ('M','R').
    assert out == ["HMHR"]


# --------------------------------------------------------------------------------------------
# Lazy-import guard: importing the module must NOT import torch or carbonara.
# --------------------------------------------------------------------------------------------
def test_import_does_not_pull_torch_or_carbonara():
    # The module under test is already imported at top-of-file; assert it never imported these.
    for mod in ("torch", "carbonara"):
        assert mod not in sys.modules, f"{mod} must not be imported by carbonara_backend"


def test_fresh_subprocess_import_keeps_torch_and_carbonara_out():
    # Robust guard: import the module in a CLEAN interpreter and assert torch/carbonara are
    # absent from its sys.modules afterwards (the top-of-file import above can't see a process
    # that already loaded torch for some other reason).
    import subprocess

    code = (
        "import sys; import xenodesign.carbonara_backend; "
        "assert 'torch' not in sys.modules, 'torch leaked'; "
        "assert 'carbonara' not in sys.modules, 'carbonara leaked'; "
        "print('OK')"
    )
    repo = str(pathlib.Path(cb.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": repo}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


# --------------------------------------------------------------------------------------------
# Fail-fast when the CARBonAra .venv python is absent.
# --------------------------------------------------------------------------------------------
def test_real_run_fails_fast_when_venv_python_missing(monkeypatch, tmp_path):
    # Point the venv python at a non-existent path and call the REAL _run_carbonara (not patched).
    monkeypatch.setattr(cb, "_CARBONARA_VENV_PYTHON", tmp_path / "no" / "python")
    with pytest.raises(FileNotFoundError, match="CARBonAra"):
        cb._run_carbonara(tmp_path / "x.pdb", tmp_path / "out", num_seqs=1,
                          known_positions=[], known_chains=["B"], temperature=0.1)


# --------------------------------------------------------------------------------------------
# PDB-build unit: chain A backbone+CB and chain B context, parse back with gemmi.
# --------------------------------------------------------------------------------------------
def test_build_pdb_writes_chain_A_backbone_cb_and_chain_B_context(tmp_path):
    import gemmi

    # 2-residue design chain: N, CA, C, CB per residue (distinct coords).
    bb = np.array([
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, -1.0, 0.0]],
        [[3.0, 0.0, 0.0], [4.5, 0.0, 0.0], [5.0, 1.0, 0.0], [4.0, -1.0, 0.0]],
    ], dtype=float)
    ctx = np.array([[10.0, 0.0, 0.0], [11.0, 1.0, 0.0], [12.0, 2.0, 0.0]], dtype=float)
    ctx_elems = ["C", "N", "O"]

    pdb_path = tmp_path / "complex.pdb"
    cb._build_pdb(bb, ctx, ctx_elems, pdb_path)
    assert pdb_path.exists()

    st = gemmi.read_structure(str(pdb_path))
    model = st[0]
    chains = {ch.name: ch for ch in model}
    assert "A" in chains and "B" in chains

    # Chain A: 2 residues, each with N, CA, C, CB.
    chain_a = chains["A"]
    assert len(chain_a) == 2
    for res in chain_a:
        names = {a.name for a in res}
        assert {"N", "CA", "C", "CB"} <= names

    # Chain B: 3 context atoms, elements preserved.
    chain_b = chains["B"]
    ctx_atoms = [a for res in chain_b for a in res]
    assert len(ctx_atoms) == 3
    got_elems = [a.element.name.upper() for a in ctx_atoms]
    assert got_elems == ["C", "N", "O"]

    # Coordinates round-trip (PDB has 3-decimal precision).
    a_ca0 = chain_a[0].find_atom("CA", "*")
    assert a_ca0.pos.x == pytest.approx(1.5, abs=1e-2)


def test_build_pdb_empty_context_writes_only_chain_A(tmp_path):
    import gemmi

    bb = np.zeros((3, 4, 3))
    pdb_path = tmp_path / "nocontext.pdb"
    cb._build_pdb(bb, np.zeros((0, 3)), [], pdb_path)
    st = gemmi.read_structure(str(pdb_path))
    model = st[0]
    chain_names = {ch.name for ch in model}
    assert "A" in chain_names
    # No chain B (or an empty one) when there is no context.
    if "B" in chain_names:
        assert sum(1 for res in next(ch for ch in model if ch.name == "B") for _ in res) == 0


# --------------------------------------------------------------------------------------------
# End-to-end through a SequenceUpdater: the adapter is chirality-agnostic L-in/L-out and the
# downstream D-CCD re-encode is reused unchanged (mirrors the LigandMPNN integration).
# --------------------------------------------------------------------------------------------
def test_adapter_drives_sequence_updater_d_ccd_reencode(monkeypatch):
    from xenodesign.sequence_update import SequenceUpdater

    _patch_run(monkeypatch, design_seqs=["GGG"])
    upd = SequenceUpdater(design_fn=cb.carbonara_design_fn)
    result = upd.update(
        design_backbone=np.random.RandomState(0).rand(3, 4, 3),
        design_codes=["DAL", "DSN", "DLE"],
        context_coords=np.zeros((0, 3)),
        context_elements=[],
    )
    # L-in 'GGG' -> downstream re-encodes to D-CCD (Gly stays bare G, achiral).
    assert result.one_letter == "GGG"
    assert result.d_fasta == "GGG"
