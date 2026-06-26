"""Fixed-geometry "thread + relax" register decoy builder (T03b).

The honest register test
------------------------
The free-re-predict register decoys (``register_decoys/``) let a composition-
preserving circular shift RE-DOCK: when you hand the shifted SEQUENCE to a
structure model it just finds an equally good interface in a new pose, so neither
Chai confidence (ADR-022) nor relaxed physics (T03) can tell real from shift.
That masks register by construction.

This script removes that escape hatch. It holds the DEPOSITED backbone GEOMETRY
of the 62-res helix ("binder") FIXED and only swaps residue IDENTITIES along it
by a circular shift. The 20-res partner chain is kept exactly as deposited. So a
"shifted" decoy is forced to present the rotated sequence ON THE SAME helix axis,
in the SAME interface pose as the real complex — it cannot re-dock away. We then
score it with the existing parity-safe relaxed dE_int (scripts/score_ddg.py),
identically for every condition, so real-vs-shift is a controlled comparison.

What "thread" means here
------------------------
For a circular shift ``s`` and binder length ``L`` (=62), the new residue at
position ``i`` (0-based, along the deposited chain order) takes the IDENTITY of
the deposited binder residue ``(i + s) mod L``, while KEEPING position ``i``'s
deposited backbone frame (N, CA, C, O coordinates). This matches the convention
in register_decoys/items.json: shift 3 rotates seq_a left by 3
("LPVEKII..." -> "EKII...LPV"), i.e. new[i] = old[(i+s) mod L].

After renaming, the old side-chain heavy atoms are dropped and PDBFixer rebuilds
the NEW identities' side chains onto the held backbone (findMissingAtoms ->
addMissingAtoms), then hydrogens are added. The partner chain is passed through
untouched.

Approximations (marked ``ponytail:`` throughout)
------------------------------------------------
- ponytail: backbone-held side-chain REBUILD, not a full rotamer repack. PDBFixer
  places each new side chain in a default/template conformation on the fixed
  backbone; we do NOT search rotamers for the best fit. This is fine for the
  COMPARISON because every condition (real and each shift) gets the identical
  build+relax pipeline, and score_ddg's CA-restrained minimization then relaxes
  local clashes the same way for all. We are measuring the RELATIVE register
  signal, not an absolute affinity.
- ponytail: D binders (8GQP) are emitted with their deposited D-residue names.
  score_ddg renames D->L for the force field on its own (parity-safe; only the
  inter-chain term is read), and the backbone geometry it threads onto is the
  deposited D geometry, so chirality is preserved end to end.
- ponytail: the partner is kept as the deposited ORDERED residues only (waters and
  any disordered tail are dropped by the gemmi clean), identically for every
  condition, so it cancels out of the real-vs-shift contrast.

Output
------
Writes a 2-chain complex PDB (partner chain first as "A", threaded binder second
as "B") so it drops straight into score_ddg's ``--target_chain A --binder_chain B``
relaxed-dE_int path. score_ddg re-runs PDBFixer (idempotent: nothing missing if we
already rebuilt) + the CA-restrained minimization + parity-safe pull-apart.

CLI
---
  micromamba run -n SE3nv python scripts/thread_register_decoy.py \
      --cif XenoDesign1_local_ref/benchmarks/8GQP.cif \
      --binder_chain A --partner_chain B --shift 3 --out /tmp/8GQP_shift3.pdb

Self-check (no GPU, builds 8GQP shift0 and shift3 in a tmp dir):
  micromamba run -n SE3nv python scripts/thread_register_decoy.py --selfcheck
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import gemmi

# Standard L 3-letter set and the D->L 3-letter map (shared convention with
# score_ddg.py). Used here to (a) recognise protein residues vs HOH/hetero and
# (b) get a one-letter code for the self-check sequence assertions.
_D2L = {"DAL": "ALA", "DAR": "ARG", "DSG": "ASN", "DAS": "ASP", "DCY": "CYS",
        "DCYS": "CYS", "DGN": "GLN", "DGL": "GLU", "DHI": "HIS", "DIL": "ILE",
        "DLE": "LEU", "DLY": "LYS", "MED": "MET", "DPN": "PHE", "DPR": "PRO",
        "DSN": "SER", "DTH": "THR", "DTR": "TRP", "DTY": "TYR", "DVA": "VAL",
        "DGY": "GLY"}
_STD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
_BACKBONE = ("N", "CA", "C", "O")


def _is_protein(resname):
    return resname in _STD or resname in _D2L


def _protein_residues(chain):
    """Ordered protein residues of a gemmi chain (drops HOH / hetero)."""
    return [r for r in chain if _is_protein(r.name)]


def _one_letter(resname):
    return gemmi.find_tabulated_residue(_D2L.get(resname, resname)).one_letter_code.upper()


def thread(cif, binder_chain, partner_chain, shift):
    """Build a threaded 2-chain gemmi.Structure for a circular ``shift``.

    Partner chain -> emitted as "A" (kept exactly as deposited protein residues).
    Binder chain  -> emitted as "B": each position i keeps its DEPOSITED backbone
    (N, CA, C, O) but takes the residue NAME of deposited binder residue
    (i+shift) mod L. Side-chain atoms are dropped here; PDBFixer (downstream)
    rebuilds them onto the held backbone.

    Returns (structure, threaded_names, deposited_binder_names).
    """
    st = gemmi.read_structure(str(cif))
    m = st[0]
    binder = _protein_residues(m[binder_chain])
    partner = _protein_residues(m[partner_chain])
    L = len(binder)
    deposited_names = [r.name for r in binder]

    ns = gemmi.Structure()
    ns.add_model(gemmi.Model("1"))

    # Partner first -> chain "A", verbatim (heavy atoms; H re-added downstream).
    pc = gemmi.Chain("A")
    for r in partner:
        nr = gemmi.Residue()
        nr.name = r.name
        nr.seqid = r.seqid
        for a in r:
            if a.element == gemmi.Element("H"):
                continue
            nr.add_atom(a)
        pc.add_residue(nr)
    ns[0].add_chain(pc)

    # Threaded binder -> chain "B": held backbone, shifted identity.
    bc = gemmi.Chain("B")
    threaded_names = []
    for i in range(L):
        src = binder[(i + shift) % L]          # identity donor (shifted)
        frame = binder[i]                       # backbone-frame donor (held, position i)
        nr = gemmi.Residue()
        nr.name = src.name                      # new identity
        nr.seqid = frame.seqid                  # keep position i's numbering
        # ponytail: copy ONLY backbone atoms from the held frame; PDBFixer rebuilds
        # the new identity's side chain. No rotamer search (see module docstring).
        for a in frame:
            if a.name in _BACKBONE:
                nr.add_atom(a)
        bc.add_residue(nr)
        threaded_names.append(src.name)
    ns[0].add_chain(bc)

    ns.setup_entities()
    return ns, threaded_names, deposited_names


def build(cif, binder_chain, partner_chain, shift, out_path):
    """Thread + PDBFixer side-chain rebuild -> write a ready-to-score PDB.

    The output is a 2-chain PDB (A=partner, B=threaded binder) with full heavy
    atoms + hydrogens, which score_ddg's relaxed dE_int path consumes directly.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    ns, threaded_names, deposited_names = thread(cif, binder_chain, partner_chain, shift)

    tmp = Path(tempfile.mkdtemp(prefix="thread_"))
    bb_pdb = tmp / "threaded_backbone.pdb"
    ns.write_pdb(str(bb_pdb))

    # PDBFixer rebuilds the new side chains onto the held backbone, then adds H.
    fixer = PDBFixer(filename=str(bb_pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}                  # never model unresolved loops
    fixer.findMissingAtoms()                    # finds the stripped side-chain atoms
    fixer.addMissingAtoms()                     # builds them on the held backbone
    fixer.addMissingHydrogens(7.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    return out_path, threaded_names, deposited_names


# --------------------------------------------------------------------------- #
def _selfcheck():
    """One runnable self-check (no GPU):

      (1) shift 0 reproduces the deposited binder sequence exactly;
      (2) a shift!=0 decoy carries the circularly-rotated sequence
          (new[i] == deposited[(i+shift) % L]); and
      (3) the built PDB parses back as a 2-chain complex.
    """
    cif = ("XenoDesign1_local_ref/benchmarks/8GQP.cif")
    if not Path(cif).exists():
        print(f"SELFCHECK SKIP: {cif} not found", file=sys.stderr)
        return 0
    binder_chain, partner_chain = "A", "B"

    # (1) shift 0 == deposited
    _, names0, dep = thread(cif, binder_chain, partner_chain, 0)
    seq0 = "".join(_one_letter(n) for n in names0)
    depseq = "".join(_one_letter(n) for n in dep)
    assert seq0 == depseq, f"shift0 seq mismatch:\n  {seq0}\n  {depseq}"

    # (2) shift 3 == circular rotation of deposited
    s = 3
    _, names3, _ = thread(cif, binder_chain, partner_chain, s)
    L = len(dep)
    expected = [dep[(i + s) % L] for i in range(L)]
    assert names3 == expected, "shift3 not a clean circular rotation"
    seq3 = "".join(_one_letter(n) for n in names3)
    assert seq3 != depseq, "shift3 sequence should differ from real"

    # (3) build + parse round-trip
    tmp = Path(tempfile.mkdtemp(prefix="thread_sc_"))
    out, tnames, _ = build(cif, binder_chain, partner_chain, s, tmp / "8GQP_shift3.pdb")
    st = gemmi.read_structure(str(out))
    nchains = len(st[0])
    nprot_b = len(_protein_residues(st[0]["B"]))
    assert nchains == 2, f"expected 2 chains in output, got {nchains}"
    assert nprot_b == L, f"binder length changed: {nprot_b} != {L}"

    print("SELFCHECK PASS", file=sys.stderr)
    print(f"  deposited (shift0) binder seq : {depseq}", file=sys.stderr)
    print(f"  shift3 threaded binder seq    : {seq3}", file=sys.stderr)
    print(f"  built+parsed: {nchains} chains, binder {nprot_b} res -> {out}", file=sys.stderr)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cif", help="deposited complex CIF")
    p.add_argument("--binder_chain", default="A", help="62-res helix chain to thread")
    p.add_argument("--partner_chain", default="B", help="20-res partner chain (held)")
    p.add_argument("--shift", type=int, default=0, help="circular shift (0=real)")
    p.add_argument("--out", help="output PDB path")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args(argv)
    if a.selfcheck:
        return _selfcheck()
    if not (a.cif and a.out):
        p.error("--cif and --out required (or use --selfcheck)")
    out, tnames, dep = build(a.cif, a.binder_chain, a.partner_chain, a.shift, a.out)
    seq = "".join(_one_letter(n) for n in tnames)
    print(f"wrote {out}")
    print(f"shift={a.shift} threaded binder seq: {seq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
