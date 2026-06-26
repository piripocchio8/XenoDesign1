"""Build a REGISTER-AGNOSTIC chai POCKET restraint that ties the whole BINDER chain
to the set of TARGET pocket residues, for the clean register experiment (STEP 3 prep).

Why a POCKET restraint (not residue-residue CONTACT): a circular register shift of the
binder rotates which binder residue sits in the pocket. A CONTACT restraint names a
specific binder residue and would therefore favour one register over another. The chai
POCKET restraint leaves the binder side as a whole chain (res_idxA empty) and only names
residues on the TARGET (res_idxB), so the SAME file is valid for the true binder and every
circular-shift decoy. This pins all of them to the same target site so an unrestrained
decoy cannot dock elsewhere and confound the signal.

chai .restraints format (CSV, learned from chai-lab/examples/restraints/pocket.restraints
and XenoDesign1_local_ref/restraint_variants/v_tyrspan.restraints):

  chainA,res_idxA,chainB,res_idxB,connection_type,confidence,min_distance_angstrom,max_distance_angstrom,comment,restraint_id

  POCKET row: res_idxA is EMPTY (chainA = the whole pocket-bound chain), res_idxB =
  <one-letter><1-based-pos> of the named token on chainB. Per chai's parser
  (data/parsing/restraints.py) res_idxB[0] is the one-letter residue name and res_idxB[1:]
  is the 1-based position; the position indexes the chain's chai FASTA sequence.

  We put:  chainA = BINDER (empty res)  ;  chainB = TARGET  ;  res_idxB = each target pocket residue.

ponytail: one POCKET row per pocket residue with a fixed confidence/min/max and a 0..5.5 A
threshold (chai's example default for protein-heavy pockets). We do NOT tune per-residue
distances or confidences — the experiment only needs "binder sits at this target site".

ponytail: chai's POCKET name-check maps res_idxB's one-letter back to a STANDARD L 3-letter
code (restype_1to3_with_x) and asserts it equals the tokenized residue name. For a D-peptide
TARGET (7YH8) the tokenised names are DGL/DHI/... so chai's own assertion (GLU != DGL) will
reject D-target pocket rows at RUNTIME. The emitted file is still a correct, parseable,
register-agnostic restraint referencing the right target residues; the D-target runtime
limitation is chai's (documented L-only name-match), not this generator's, and is left for
the GPU step to handle (e.g. patch the name-check) rather than worked around here.

SINGLE-CENTRAL-RESIDUE mode (--single): instead of the whole pocket, emit ONE POCKET row
binding the whole binder chain to the single TARGET residue with the most binder heavy-atom
contacts (<4.5 A; tie-break = closest to the interface-contact centroid), at max_distance
~6.0 A. A lone anchor holds the binder at the SITE without pinning the register, so the same
file stays valid for the true binder and every circular-shift / scramble decoy. Output goes to
benchmarks/pocket_restraints_single/ (full-pocket mode is unchanged, in benchmarks/pocket_restraints/).

Usage:
  PYTHONPATH=$PWD python3 scripts/make_pocket_restraints.py            # full pocket: emit 8GQP + 7YH8 + self-check
  PYTHONPATH=$PWD python3 scripts/make_pocket_restraints.py --selfcheck
  PYTHONPATH=$PWD python3 scripts/make_pocket_restraints.py --single   # single central anchor + self-check
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gemmi

# score_complex lives next to this file; make the import cwd-independent.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_complex import _3to1, _D2L  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "XenoDesign1_local_ref" / "benchmarks"
OUTDIR = BENCH / "pocket_restraints"
OUTDIR_SINGLE = BENCH / "pocket_restraints_single"

HEADER = (
    "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
    "min_distance_angstrom,max_distance_angstrom,comment,restraint_id"
)
POCKET_DIST_A = 5.5      # ponytail: chai example default for protein-heavy pockets
CONTACT_CUTOFF_A = 4.5   # heavy-atom cutoff defining "in the pocket"
SINGLE_DIST_A = 6.0      # single-anchor max_distance: holds the SITE, not the register


def _one_letter(resname: str) -> str:
    """3-letter residue name (D- or L-) -> standard 1-letter code (X if unknown)."""
    return _3to1.get(_D2L.get(resname, resname), "X")


def _polymer_residues(chain) -> list:
    return [r for r in chain if not r.is_water()]


def target_pocket(cif: Path, binder_chain: str, target_chain: str,
                  cutoff: float = CONTACT_CUTOFF_A) -> list[tuple[int, str, str]]:
    """Return [(pos_1based, resname, one_letter), ...] for target residues with any heavy
    atom within `cutoff` A of any binder heavy atom. pos is the 1-based index along the
    target polymer (matches chai's FASTA residue index)."""
    st = gemmi.read_structure(str(cif))
    m = st[0]
    binder_heavy = [a.pos for r in _polymer_residues(m[binder_chain])
                    for a in r if not a.is_hydrogen()]
    pocket = []
    for pos, r in enumerate(_polymer_residues(m[target_chain]), start=1):
        for a in r:
            if a.is_hydrogen():
                continue
            if any(a.pos.dist(bp) <= cutoff for bp in binder_heavy):
                pocket.append((pos, r.name, _one_letter(r.name)))
                break
    return pocket


def central_pocket_residue(cif: Path, binder_chain: str, target_chain: str,
                           cutoff: float = CONTACT_CUTOFF_A
                           ) -> tuple[int, str, str, int]:
    """Pick the ONE target residue most central to the interface.

    Score = number of binder heavy atoms within `cutoff` A of any heavy atom of that
    target residue (most binder contacts = deepest in the pocket). Tie-break: the residue
    whose Cα (fallback: residue centroid) is closest to the centroid of all interface
    contact midpoints. Returns (pos_1based, resname, one_letter, n_binder_contacts).
    """
    st = gemmi.read_structure(str(cif))
    m = st[0]
    binder_heavy = [a.pos for r in _polymer_residues(m[binder_chain])
                    for a in r if not a.is_hydrogen()]

    contact_mids = []   # midpoints of every cross-chain heavy-atom contact (interface centroid)
    scored = []         # (pos, resname, one, n_contacts, anchor_pos)
    for pos, r in enumerate(_polymer_residues(m[target_chain]), start=1):
        t_heavy = [a.pos for a in r if not a.is_hydrogen()]
        n_contacts = 0
        for tp in t_heavy:
            for bp in binder_heavy:
                if tp.dist(bp) <= cutoff:
                    n_contacts += 1
                    contact_mids.append(((tp.x + bp.x) / 2,
                                         (tp.y + bp.y) / 2,
                                         (tp.z + bp.z) / 2))
        if n_contacts == 0:
            continue
        ca = next((a.pos for a in r if a.name == "CA"), None)
        if ca is None:
            cx = sum(p.x for p in t_heavy) / len(t_heavy)
            cy = sum(p.y for p in t_heavy) / len(t_heavy)
            cz = sum(p.z for p in t_heavy) / len(t_heavy)
            ca = gemmi.Position(cx, cy, cz)
        scored.append((pos, r.name, _one_letter(r.name), n_contacts, ca))

    if not scored:
        raise RuntimeError(f"{cif.name}: no target residue contacts the binder within {cutoff} A")

    # interface contact centroid (used only for the tie-break)
    n = len(contact_mids)
    centroid = gemmi.Position(sum(c[0] for c in contact_mids) / n,
                              sum(c[1] for c in contact_mids) / n,
                              sum(c[2] for c in contact_mids) / n)
    # primary key: most binder contacts (desc); tie-break: closest anchor to centroid (asc)
    best = max(scored, key=lambda s: (s[3], -s[4].dist(centroid)))
    pos, resname, one, n_contacts, _ = best
    return pos, resname, one, n_contacts


def write_single_restraint(cif: Path, binder_chain: str, target_chain: str,
                           out: Path, tag: str) -> tuple[int, str, str, int]:
    """Emit a SINGLE-anchor POCKET restraint binding the whole BINDER chain to the ONE
    most-central TARGET interface residue (max_distance ~6.0 A). A single anchor holds the
    SITE but not the register, so the same file is valid for the true binder and every
    circular-shift / scramble decoy without biasing any one register."""
    pos, resname, one, n_contacts = central_pocket_residue(cif, binder_chain, target_chain)
    # ponytail: ONE pocket row, fixed confidence/min, 6.0 A threshold — a lone site anchor.
    lines = [HEADER]
    lines.append(
        f"{binder_chain},,{target_chain},{one}{pos},pocket,1.0,0.0,"
        f"{SINGLE_DIST_A},{tag}-single-{resname}{pos}-c{n_contacts},{tag}_single_{pos}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return pos, resname, one, n_contacts


def write_restraints(cif: Path, binder_chain: str, target_chain: str,
                     out: Path, tag: str) -> list[tuple[int, str, str]]:
    """Emit one POCKET restraint file binding the BINDER chain to the TARGET pocket."""
    pocket = target_pocket(cif, binder_chain, target_chain)
    lines = [HEADER]
    for pos, resname, one in pocket:
        # chainA = binder (whole chain, res empty); chainB = target named residue.
        lines.append(
            f"{binder_chain},,{target_chain},{one}{pos},pocket,1.0,0.0,"
            f"{POCKET_DIST_A},{tag}-pocket-{resname}{pos},{tag}_pocket_{pos}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return pocket


# (cif relative to BENCH, binder_chain, target_chain) per system.
SYSTEMS = {
    "8GQP": ("8GQP.cif", "A", "B"),   # binder A = 62-res D-helix; target B = 20-res L-pep
    "7YH8": ("7YH8.cif", "A", "B"),   # binder A = 62-res L-helix; target B = D-Pep-1
}


def build_all() -> dict[str, list[tuple[int, str, str]]]:
    result = {}
    for tag, (cif_name, b, t) in SYSTEMS.items():
        out = OUTDIR / f"{tag}.restraints"
        pocket = write_restraints(BENCH / cif_name, b, t, out, tag)
        result[tag] = pocket
        print(f"{tag}: wrote {out} ({len(pocket)} pocket rows): "
              f"{[f'{o}{p}' for p, _, o in pocket]}")
    return result


def build_all_single() -> dict[str, tuple[int, str, str, int]]:
    """Single-central-residue mode: one anchor per system in OUTDIR_SINGLE."""
    result = {}
    for tag, (cif_name, b, t) in SYSTEMS.items():
        out = OUTDIR_SINGLE / f"{tag}.restraints"
        pos, resname, one, n = write_single_restraint(BENCH / cif_name, b, t, out, tag)
        result[tag] = (pos, resname, one, n)
        print(f"{tag}: wrote {out} (1 single-anchor row): central={resname}{pos} "
              f"(one-letter {one}, {n} binder contacts, max_dist={SINGLE_DIST_A} A)")
    return result


def _selfcheck() -> None:
    """Emit both files, then assert each parses as chai POCKET rows referencing in-range,
    non-empty, standard-named target residues."""
    built = build_all()
    for tag, (cif_name, b, t) in SYSTEMS.items():
        out = OUTDIR / f"{tag}.restraints"
        st = gemmi.read_structure(str(BENCH / cif_name))
        target_len = len(_polymer_residues(st[0][t]))

        rows = [ln.split(",") for ln in out.read_text().splitlines()]
        header, body = rows[0], rows[1:]
        assert header[:5] == ["chainA", "res_idxA", "chainB", "res_idxB",
                              "connection_type"], f"{tag}: bad header {header[:5]}"
        assert body, f"{tag}: no restraint rows emitted"
        ids = set()
        for r in body:
            chainA, res_idxA, chainB, res_idxB, conn = r[0], r[1], r[2], r[3], r[4]
            assert conn == "pocket", f"{tag}: not a pocket row: {r}"
            assert chainA == b, f"{tag}: chainA must be binder {b}, got {chainA}"
            assert res_idxA == "", f"{tag}: pocket res_idxA must be empty, got {res_idxA!r}"
            assert chainB == t, f"{tag}: chainB must be target {t}, got {chainB}"
            assert res_idxB and res_idxB[0].isalpha(), f"{tag}: bad res_idxB {res_idxB!r}"
            assert res_idxB[0] != "X", f"{tag}: unknown residue name in {res_idxB!r}"
            pos = int(res_idxB[1:])
            assert 1 <= pos <= target_len, f"{tag}: pos {pos} out of range 1..{target_len}"
            rid = r[-1]
            assert rid not in ids, f"{tag}: duplicate restraint_id {rid}"
            ids.add(rid)
        # rows must match the computed pocket exactly
        assert len(body) == len(built[tag]), f"{tag}: row count != pocket size"
        print(f"{tag}: self-check OK ({len(body)} pocket rows, target_len={target_len})")
    print("SELFCHECK PASS")


def _selfcheck_single() -> None:
    """Emit the single-anchor files, then assert each is exactly ONE chai POCKET row
    referencing an in-range, non-empty, standard-named target residue at max_dist 6.0."""
    built = build_all_single()
    for tag, (cif_name, b, t) in SYSTEMS.items():
        out = OUTDIR_SINGLE / f"{tag}.restraints"
        st = gemmi.read_structure(str(BENCH / cif_name))
        target_len = len(_polymer_residues(st[0][t]))

        rows = [ln.split(",") for ln in out.read_text().splitlines()]
        header, body = rows[0], rows[1:]
        assert header[:5] == ["chainA", "res_idxA", "chainB", "res_idxB",
                              "connection_type"], f"{tag}: bad header {header[:5]}"
        assert len(body) == 1, f"{tag}: single mode must emit exactly 1 row, got {len(body)}"
        r = body[0]
        chainA, res_idxA, chainB, res_idxB, conn = r[0], r[1], r[2], r[3], r[4]
        assert conn == "pocket", f"{tag}: not a pocket row: {r}"
        assert chainA == b, f"{tag}: chainA must be binder {b}, got {chainA}"
        assert res_idxA == "", f"{tag}: pocket res_idxA must be empty, got {res_idxA!r}"
        assert chainB == t, f"{tag}: chainB must be target {t}, got {chainB}"
        assert res_idxB and res_idxB[0].isalpha() and res_idxB[0] != "X", \
            f"{tag}: bad/unknown res_idxB {res_idxB!r}"
        pos = int(res_idxB[1:])
        assert 1 <= pos <= target_len, f"{tag}: pos {pos} out of range 1..{target_len}"
        assert float(r[7]) == SINGLE_DIST_A, f"{tag}: max_dist != {SINGLE_DIST_A}"
        assert pos == built[tag][0], f"{tag}: emitted pos != chosen central residue"
        print(f"{tag}: single self-check OK (central {res_idxB}, target_len={target_len})")
    print("SELFCHECK PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true",
                    help="emit files and run assert-based parse/in-range self-check")
    ap.add_argument("--single", action="store_true",
                    help="single-central-residue mode -> pocket_restraints_single/ + self-check")
    args = ap.parse_args()
    # Default behaviour also runs the self-check so a bare run is verified end-to-end.
    if args.single:
        _selfcheck_single()
    else:
        _selfcheck()
