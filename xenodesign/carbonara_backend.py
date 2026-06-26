"""CARBonAra inverse-folding backend (spec §2.7, §7; sibling of the LigandMPNN adapter).

CARBonAra (CC-BY-NC-SA, research-only) is a context-aware, all-atom inverse-folding model. It is
NOT D-native: like LigandMPNN it treats the structure as an all-atom graph and emits L sequences,
so it plugs into the SAME mirror-into-L double-flip used for LigandMPNN. This adapter is therefore
chirality-AGNOSTIC, L-in / L-out: the upstream reflection (`prepare_inverse_folding_inputs`) and the
downstream D-CCD re-encode (`SequenceUpdater` / `to_d_fasta`) are reused UNCHANGED. CARBonAra's role
is a comparison against LigandMPNN inside the double-flip — to test whether it drifts less toward L
over the loop — not as a D-native replacement.

`carbonara_design_fn` mirrors `sequence_update._ligandmpnn_design_fn`: it implements the
`InverseFoldingBackend` protocol (exactly 6 positional params -> list[str]) and designs the DESIGN
CHAIN ONLY, never echoing the fixed/target chain (the harness reconstructs that from the known input
via `assemble_complex_fasta`; CARBonAra cannot emit D-CCD codes anyway).

CARBonAra lives in its own .venv and is NEVER imported into this process: the only entry point is a
subprocess to that venv's python (`_run_carbonara`). Heavy deps (gemmi for PDB build/parse) are
imported lazily so importing this module loads neither torch nor carbonara — keeping the CPU test
suite green without a GPU/env.
"""
from __future__ import annotations

import os
import pathlib
from typing import List, Sequence

# CARBonAra checkout + its self-contained venv python (validated in-container today). The adapter
# only ever shells out to this interpreter; carbonara is never imported into the current process.
# Override the location via the CARBONARA_DIR env var; the default keeps ifrit unchanged.
_CARBONARA_DIR = pathlib.Path(os.environ.get("CARBONARA_DIR", "/home/user/CARBonAra"))
_CARBONARA_VENV_PYTHON = _CARBONARA_DIR / ".venv" / "bin" / "python"
_CARBONARA_SCRIPT = _CARBONARA_DIR / "carbonara.py"

# temperature has NO CARBonAra equivalent. We map it to sampling_method='sampled' with a FIXED
# imprint_ratio held constant across calls (documented; not a free knob). 0.5 is CARBonAra's own
# default and the value validated in-container.
_IMPRINT_RATIO = 0.5

# The design chain we write into the temp PDB; the fixed partner is chain B (echoed by CARBonAra
# after the colon and discarded). Matches the in-container invocation (--known_chains B).
_DESIGN_CHAIN = "A"
_KNOWN_CHAIN = "B"

# Canonical one-letter alphabet (the CARBonAra / LigandMPNN output alphabet).
_ALPHABET = set("ARNDCQEGHILKMFPSTWYV")


def carbonara_design_fn(
    design_backbone,
    context_coords,
    context_elements: Sequence[str],
    fixed_mask: Sequence[bool],
    temperature: float = 0.1,
    num_seqs: int = 1,
) -> List[str]:
    """CARBonAra inverse-folding backend (InverseFoldingBackend protocol).

    Designs the DESIGN CHAIN ONLY given its (already mirror-into-L) backbone + the fixed partner as
    all-atom context, returning `num_seqs` candidate one-letter L sequences (each len == n_res).

    Args:
        design_backbone: np.ndarray (n_res, 4, 3) — N, CA, C, CB in L-frame.
        context_coords: np.ndarray (n_ctx, 3) — partner atoms (may be empty).
        context_elements: list[str] — element symbols of the context atoms (may be empty).
        fixed_mask: list[bool] — True = keep this design position fixed (output 'A' placeholder).
        temperature: float — mapped to sampling_method='sampled' + a fixed imprint_ratio (no
            CARBonAra temperature exists). Held constant; forwarded only for parity/logging.
        num_seqs: int — number of candidate sequences to sample and return.

    Returns:
        list[str] of length num_seqs — designed-chain one-letter L sequences (the DESIGNED chain
        ONLY; the known chain B is parsed off and discarded), each len == n_res, chars ⊆ _ALPHABET.
    """
    import tempfile

    import numpy as np

    design_backbone = np.asarray(design_backbone, dtype=float)
    n_res = design_backbone.shape[0]
    fixed_mask = list(fixed_mask)

    # fixed_mask is 0-based; CARBonAra --known_positions is 1-based over the design chain (chain A
    # is first in PDB order, so residue indices 1..n_res map directly). OFF-BY-ONE lives here.
    known_positions = [i + 1 for i, fixed in enumerate(fixed_mask) if fixed]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        pdb_path = tmp / "complex.pdb"
        out_dir = tmp / "out"
        _build_pdb(design_backbone, context_coords, context_elements, pdb_path)

        # Subprocess seam (patched in tests): one [design_seq, known_seq] pair per sample.
        pairs = _run_carbonara(
            pdb_path, out_dir, num_seqs=int(num_seqs),
            known_positions=known_positions, known_chains=[_KNOWN_CHAIN],
            temperature=float(temperature),
        )

    results: List[str] = []
    for pair in pairs:
        # The design chain (chain A) is BEFORE the colon; the known chain B is echoed after and
        # discarded. _run_carbonara already split on ':', so pair[0] is the design chain.
        design_seq = pair[0]
        letters = list(design_seq)
        # Re-apply fixed positions to a deterministic placeholder ('A'), exactly like
        # sequence_update.py:412-414 (the model may have changed them; we pin them).
        for i, fixed in enumerate(fixed_mask):
            if fixed and i < len(letters):
                letters[i] = "A"
        one = "".join(letters)
        if len(one) != n_res:
            raise ValueError(
                f"CARBonAra returned {len(one)} design residues, expected {n_res}"
            )
        results.append(one)

    assert len(results) == num_seqs or len(results) == len(pairs)
    return results


def _build_pdb(design_backbone, context_coords, context_elements, path) -> None:
    """Write a temp PDB: design chain 'A' (N/CA/C/CB backbone) + partner chain 'B' (all-atom
    context as HETATM). CARBonAra then designs chain A while keeping chain B's full interface
    atoms as context. gemmi is imported lazily (kept out of module import).

    Args:
        design_backbone: np.ndarray (n_res, 4, 3) — N, CA, C, CB.
        context_coords: np.ndarray (n_ctx, 3) — partner atoms (may be empty).
        context_elements: list[str] — element symbols for the context atoms.
        path: output PDB path.
    """
    import gemmi
    import numpy as np

    design_backbone = np.asarray(design_backbone, dtype=float)
    context_coords = np.asarray(context_coords, dtype=float).reshape(-1, 3)
    context_elements = list(context_elements)

    structure = gemmi.Structure()
    model = gemmi.Model("1")

    # --- Design chain A: one ALA per residue with N, CA, C, CB. ---
    chain_a = gemmi.Chain(_DESIGN_CHAIN)
    atom_names = ("N", "CA", "C", "CB")
    for ri in range(design_backbone.shape[0]):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = gemmi.SeqId(ri + 1, " ")
        for ai, aname in enumerate(atom_names):
            atom = gemmi.Atom()
            atom.name = aname
            atom.element = gemmi.Element("C" if aname == "CB" else aname[0])
            x, y, z = design_backbone[ri, ai]
            atom.pos = gemmi.Position(float(x), float(y), float(z))
            res.add_atom(atom)
        chain_a.add_residue(res)
    model.add_chain(chain_a)

    # --- Known/partner chain B: all-atom context atoms (one per gemmi residue, HETATM-style). ---
    if context_coords.shape[0] > 0:
        chain_b = gemmi.Chain(_KNOWN_CHAIN)
        for ci in range(context_coords.shape[0]):
            res = gemmi.Residue()
            res.name = "LIG"
            res.het_flag = "H"  # HETATM context
            res.seqid = gemmi.SeqId(ci + 1, " ")
            atom = gemmi.Atom()
            el = context_elements[ci] if ci < len(context_elements) else "C"
            atom.name = el.upper()
            atom.element = gemmi.Element(el)
            x, y, z = context_coords[ci]
            atom.pos = gemmi.Position(float(x), float(y), float(z))
            res.add_atom(atom)
            chain_b.add_residue(res)
        model.add_chain(chain_b)

    structure.add_model(model)
    structure.setup_entities()
    structure.write_pdb(str(path))


def _run_carbonara(pdb_path, out_dir, num_seqs, known_positions, known_chains, temperature):
    """Subprocess seam: run the real CARBonAra CLI in its own .venv and parse the output fastas.

    This is the ONLY place CARBonAra is invoked, and it is the seam mocked in tests (no real
    CARBonAra/env touched). Mirrors the LigandMPNN FileNotFoundError fail-fast (sequence_update.py).

    Returns:
        list of per-sample [design_seq, known_seq, ...] pairs — chains in PDB order, ':'-separated
        in the fasta. The design chain (A) is first; the adapter keeps element [0] and discards the
        rest. Empty/missing known chain is tolerated (CARBonAra echoes whatever chains exist).
    """
    import subprocess

    out_dir = pathlib.Path(out_dir)
    pdb_path = pathlib.Path(pdb_path)

    if not pathlib.Path(_CARBONARA_VENV_PYTHON).exists():
        raise FileNotFoundError(
            f"CARBonAra venv python not found at {_CARBONARA_VENV_PYTHON}. "
            f"Expected a CARBonAra checkout with a .venv at {_CARBONARA_DIR}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(_CARBONARA_VENV_PYTHON), str(_CARBONARA_SCRIPT),
        str(pdb_path), str(out_dir),
        "--num_sequences", str(int(num_seqs)),
        "--sampling_method", "sampled",
        "--imprint_ratio", str(_IMPRINT_RATIO),
        "--device", "cpu",
    ]
    if known_chains:
        cmd += ["--known_chains", ",".join(known_chains)]
    if known_positions:
        cmd += ["--known_positions", ",".join(str(p) for p in known_positions)]

    subprocess.run(cmd, cwd=str(_CARBONARA_DIR), check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return _parse_output_fastas(out_dir, pdb_path.stem)


def _parse_output_fastas(out_dir, stem) -> List[List[str]]:
    """Parse CARBonAra's per-sample `<stem>_<i>.fasta` files into [chain0, chain1, ...] lists.

    Each fasta has a header line (">imprint_ratio=..., score=...") then ONE sequence line where
    chains are ':'-separated in PDB chain order. Returns one list of chain substrings per sample,
    sorted by sample index. Ignores `<stem>_pssm.csv` and `<stem>_scaffold.pdb`.
    """
    out_dir = pathlib.Path(out_dir)
    fastas = sorted(
        f for f in out_dir.glob(f"{stem}_*.fasta")
        if not f.name.endswith("_scaffold.pdb")
    )
    pairs: List[List[str]] = []
    for f in fastas:
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        seq_line = next((ln for ln in lines if not ln.startswith(">")), "")
        pairs.append(seq_line.split(":"))
    return pairs
