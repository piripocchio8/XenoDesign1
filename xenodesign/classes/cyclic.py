"""CYCLIC 6UFA binder class — single-chain Zn-macrocycle design + geometry RECALL (T7).

Migrated from ``scripts/design_cyclic.py`` (which is now a thin re-export shim). This
module owns the cyclic-class logic: the mixed-chirality seed (coordinating His pinned
L/D), the Zn-ligand FASTA emission, the His<->Zn metal-coordination restraints, the
opt-in head-to-tail COVALENT closure bond, and the RMSD-to-deposit / Zn-N geometry
scorers — plus the :class:`Cyclic` :class:`~xenodesign.classes.base.BinderClass`
adapter the dispatcher (T3) wires into the untouched :class:`HalluLoop`.

PRIMARY cyclization = MAINCHAIN head-to-tail closure (a real N-to-C backbone COVALENT
bond, consumed by chai's bond_utils, NOT a soft distance restraint). It is opt-in via
``restraint.params['closure']=True``; the default run is LINEAR + emergent-closure and
relies on the His<->Zn restraints + the deposit's intrinsic ring geometry, reporting the
N/C-terminus distance as a closure proxy. Disulfide closure is SECONDARY/not used for
this case (the cyclic 6UFA site is metal-coordinated, not disulfide-bonded).

METAL GATE (T8 merged): the cyclic target is ``metal`` (Zn ligand + His<->Zn
metal_coordination restraints). The token_dist_restraint repair lives in
``xenodesign.chai_patches`` and is now AVAILABLE, so the metallo-cyclic coordination
restraints can use the repaired ``token_dist`` path on GPU. ``targets.target_entities``
consults ``chai_patches.dist_restraint_patch_verified()`` as the metal gate; the
dispatcher applies the patch before any constrained predict. CPU tests here exercise
only seed/closure/restraint-builder/objective/report (never a predict).

6UFA DEPOSIT REALITY (verified on the RCSB mmCIF; load-bearing for interpreting results)
----------------------------------------------------------------------------------------
The 6UFA deposit is a SINGLE chain A of 24 residues = the 12-mer repeat unit TWICE.
One Zn(II) is chelated tetrahedrally by FOUR His ND1 atoms — residues 6, 12, 18, 24 —
i.e. the L-His (pos 6) and D-His (pos 12) of EACH repeat: [Zn(L-His)2(D-His)2], Zn-N
~2.02 A. The registry/seeding model the FULL 24-mer: his_resnums (6, 12, 18, 24),
_CYCLIC_HIS_CHIRALITY {6:'L', 12:'D', 18:'L', 24:'D'} — all FOUR coordinating His, the
true 4-coordinate site. (A single 12-mer carries only 2 His and CANNOT make the site.)

SEQUENCE APPROXIMATION (phase-1): the design seed is a canonical-residue stand-in for the
deposit's AIB / D-Gln / D-Glu / D-Lys / D-Leu macrocycle. We pin only the coordinating His
handedness (the chemistry that defines the site); the rest is designed/approximated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from xenodesign.benchmark.cases import get_case
from xenodesign.benchmark.restraints import build_for_case, write_restraints
from xenodesign.benchmark.seeding import _CYCLIC_HIS_CHIRALITY, build_seed_for_case
from xenodesign.classes.base import SeedSpec
from xenodesign.eval.metal_geometry_gate import metal_geometry_gate
from xenodesign.geometry import kabsch_rmsd
from xenodesign.io_spec import AA1_TO_AA3
from xenodesign.mirror import L_TO_D
from xenodesign.seed import RandomSeedGenerator, SeedResult, insert_fixed_chirality

# Re-export the seeding policy's His chirality map so the class + tests share one source.
CYCLIC_HIS_CHIRALITY: dict = dict(_CYCLIC_HIS_CHIRALITY)

# The Zn(II) cofactor enters Chai as a SMILES ligand entity (chai_lab encodes ligands as
# `>ligand|name=...` + SMILES). Zinc(II) is the bare metal cation SMILES.
ZN_SMILES = "[Zn+2]"

_DEFAULT_DEVICE = None  # unset -> resolve_device() (XENO_DEVICE / cuda:0 if avail / mps / cpu)
# 6UFA is a tetrahedral [Zn(His)4] site; Zn-N(His) bond ~2.0-2.2 A. 2.6 A cutoff (matches the
# case restraint max_distance) is generous enough to count a coordinating N without false hits.
_ZN_N_CUTOFF = 2.6


# ── Seed construction (insert_fixed_chirality at the case His positions) ────────

def build_cyclic_seed(case, seed_seq: str | None = None, rng_seed: int = 0):
    """Return the 12-res cyclic SeedResult with the coordinating His pinned (mixed L/D).

    Uses the EXISTING unconditioned cyclic seeding path (benchmark.seeding.build_seed_for_case),
    which calls seed.insert_fixed_chirality to place 'H' at the case's his_resnums and record
    their D/L handedness (_CYCLIC_HIS_CHIRALITY). PepMLM cannot condition on a metal, so the
    backbone is unconditioned (random by default, or a caller-supplied seed_seq).

    Args:
        case: the cyclic BenchmarkCase (drives length 12 + the His positions via seeding policy).
        seed_seq: optional explicit 12-res L backbone seed (His positions are overwritten with
            'H' regardless); must equal case.binder_length. None -> a deterministic random seed.
        rng_seed: RNG seed for the random backbone when seed_seq is None.

    Returns:
        seed.SeedResult(one_letter, length, reverse_applied=False, conditioned=False,
        fixed_chirality={pos: 'D'|'L'}).
    """
    if seed_seq is not None:
        if len(seed_seq) != case.binder_length:
            raise ValueError(
                f"seed_seq length {len(seed_seq)} != case.binder_length {case.binder_length}")
        # Place the coordinating His on the explicit backbone (insert_fixed_chirality validates
        # positions + records handedness); reuse the policy's chirality map.
        one_letter, fixed = insert_fixed_chirality(
            seed_seq.upper(), positions=dict(CYCLIC_HIS_CHIRALITY), residue="H")
        return SeedResult(one_letter=one_letter, length=case.binder_length,
                          reverse_applied=False, conditioned=False, fixed_chirality=fixed)

    # Unconditioned path: RandomSeedGenerator feeds build_seed_for_case, which applies the
    # cyclic policy (His placement + chirality recording). Deterministic via rng_seed.
    generator = RandomSeedGenerator(seed=rng_seed)
    return build_seed_for_case(case, generator=generator, target_seq=None)


# ── Mixed-chirality FASTA (per-position L vs D — NOT the all-D to_d_fasta) ───────

def mixed_chirality_fasta(seq_one_letter: str, fixed_chirality: dict) -> str:
    """Build a Chai sequence string with PER-POSITION L/D chirality (mixed-chirality case).

    io_spec.to_d_fasta makes the WHOLE chain D; the cyclic case is MIXED (L+D His, L backbone),
    so we emit a (DXX) D-CCD parenthesized block ONLY at positions explicitly marked 'D' in
    `fixed_chirality`. Every other position (including those marked 'L', and any unmarked
    position) is a bare canonical L residue. Glycine (achiral) always stays a single 'G'.

    Args:
        seq_one_letter: the 1-letter L sequence (His already placed at coordinating positions).
        fixed_chirality: {1-based pos: 'D'|'L'}; positions absent default to L.

    Returns:
        The Chai sequence string, e.g. 'H(DHI)H(DHI)' for HHHH with {2:'D',4:'D'}.

    Raises:
        KeyError on an unknown amino-acid letter (with its 1-based position).
    """
    out: list[str] = []
    for i, aa in enumerate(seq_one_letter):
        try:
            three = AA1_TO_AA3[aa]
        except KeyError:
            raise KeyError(f"unknown amino-acid letter {aa!r} at position {i + 1}") from None
        if three == "GLY":
            out.append("G")  # achiral — no D form
            continue
        if fixed_chirality.get(i + 1) == "D":
            out.append(f"({L_TO_D[three]})")  # D-CCD parenthesized block
        else:
            out.append(aa)  # bare canonical L residue
    return "".join(out)


# ── Zn-ligand FASTA emission (the metal/HETATM context) ─────────────────────────

def build_cyclic_input_fasta(binder_mixed_seq: str, binder_name: str = "binder",
                             zn_name: str = "zn") -> str:
    """Build the full Chai input FASTA: the mixed-chirality peptide + the Zn SMILES ligand.

    io_spec.build_fasta ONLY emits protein chains (it has no ligand path), so the Zn metal
    context is appended HERE as a chai `>ligand|name=...` SMILES entity. The protein chain is
    written FIRST so Chai labels the peptide chain A and the Zn ligand chain B — matching the
    case's metal_coordination restraint (his_chain='A', metal_chain='B').

    Args:
        binder_mixed_seq: the peptide's mixed-chirality Chai sequence (from mixed_chirality_fasta).
        binder_name: name for the protein entity header.
        zn_name: name for the Zn ligand entity header.

    Returns:
        A Chai FASTA string (trailing newline) with two entities: protein then ligand.
    """
    return (
        f">protein|{binder_name}\n{binder_mixed_seq}\n"
        f">ligand|name={zn_name}\n{ZN_SMILES}\n"
    )


# ── Metal-coordination restraint wiring (His<->Zn, via build_for_case) ──────────

def build_cyclic_restraint_rows(case, his_chain: str = "A", metal_chain: str = "B",
                                coord_residues=None) -> list:
    """Return the coordinator<->metal metal_coordination .restraints rows for the cyclic case.

    One inter-chain CONTACT per coordinating residue between the coordinator (peptide/binder)
    chain and the metal ligand chain; the metal token is 'X' (UNK). The chain letters default to
    the standalone driver's peptide=A / Zn=B order, but are OVERRIDDEN per the assembled entity
    order on the dispatcher path: the wrapper appends the binder LAST, so the metal ligand is
    chain A and the peptide is the last chain (B). Passing the wrong chains silently drops the
    restraint ("Expected >=1 residue token but found 0 ... HIS").

    ``coord_residues`` (DECLARATIVE ``--coord_residues``): a list of (pos, one_letter[, ...])
    tuples; when given they REPLACE the case's hardcoded His-only ``his_resnums`` and each
    coordinator emits its REAL one-letter identity (generalizing past His)."""
    from xenodesign.benchmark.restraints import metal_coordination_rows
    p = dict(case.restraint.params)
    p["his_chain"] = his_chain
    p["metal_chain"] = metal_chain
    if coord_residues:
        p["coord_residues"] = [(int(t[0]), str(t[1])) for t in coord_residues]
    return metal_coordination_rows(p)


def build_closure_row(seed_result, chain: str = "A"):
    """Head-to-tail COVALENT closure row (#23) for the cyclic 12-mer: C-term carbonyl C of the
    last residue bonded to the N-term amide N of residue 1 (intra-chain). The residue one-letter
    codes come from the seed (positions 1 and length). ``chain`` is the peptide/binder chain
    (default 'A' for the standalone driver; 'B' on the dispatcher path where the binder is last).
    chai consumes COVALENT rows as a real backbone bond (bond_utils), so this is true head-to-tail
    macrocyclization — NOT a soft distance restraint. NB chai matches the residue identity against
    the token; a D-CCD terminus may fail that match (verified on GPU; runs WITHOUT closure if so)."""
    from xenodesign.benchmark.restraints import head_to_tail_closure_row

    seq = seed_result.one_letter
    return head_to_tail_closure_row(
        chain, length=len(seq), n_term_one_letter=seq[0], c_term_one_letter=seq[-1])


def write_cyclic_restraints(case, out_dir, seed_result=None, closure: bool = False,
                            binder_chain: str = "A", zn_chain: str = "B",
                            metal: bool = True, coord_residues=None) -> Path:
    """Write the cyclic His<->Zn restraints CSV to out_dir/cyclic.restraints and return it.

    ``binder_chain`` / ``zn_chain`` are the chains the peptide (His) and the Zn ligand actually
    occupy in the assembled complex; defaults match the standalone driver (peptide=A, Zn=B), and
    the dispatcher passes (binder=B, Zn=A) since the wrapper appends the binder LAST.

    ``metal`` (default True): emit the His<->Zn metal_coordination rows. The NO-TARGET free-cyclic
    run (target_type='none') has NO Zn chain, so pass metal=False — the file then carries ONLY the
    opt-in closure row (no coordination rows that would reference a non-existent Zn chain).

    closure (#23): when True (and seed_result given), append a head-to-tail COVALENT backbone
    bond on the binder chain closing the macrocycle (N-to-C ring bond). Default False keeps the
    phase-1 LINEAR + emergent-closure behaviour."""
    rows = (build_cyclic_restraint_rows(case, his_chain=binder_chain, metal_chain=zn_chain,
                                        coord_residues=coord_residues)
            if metal else [])
    if closure and seed_result is not None:
        rows = rows + [build_closure_row(seed_result, chain=binder_chain)]
    return write_restraints(Path(out_dir) / "cyclic.restraints", rows)


# ── Backbone heavy-atom RMSD to the deposit (the RECALL metric) ─────────────────

def backbone_rmsd_to_deposit(design_coords, deposit_coords) -> float:
    """Kabsch backbone heavy-atom RMSD between a design and the 6UFA deposit (RECALL).

    Thin wrapper over geometry.kabsch_rmsd: both inputs are (n, 3) ordered, length-matched
    backbone heavy-atom coordinate arrays (e.g. N/CA/C/O of the 12-mer). The caller is
    responsible for extracting matched-order backbone atoms from each structure (see
    backbone_heavy_atoms_from_cif). Shape mismatch raises ValueError (from kabsch_rmsd).
    """
    return kabsch_rmsd(np.asarray(design_coords, dtype=float),
                       np.asarray(deposit_coords, dtype=float))


# ── Zn-N coordination geometry (secondary metric) ───────────────────────────────

def zn_coordination_geometry(zn_pos, nitrogen_positions, cutoff: float = _ZN_N_CUTOFF) -> dict:
    """Measure the Zn first-coordination-shell geometry from a Zn + candidate N positions.

    Counts the His-N atoms within `cutoff` A of the Zn as the coordinating shell and reports
    the Zn-N distance stats + the mean N-Zn-N angle (a tetrahedral [Zn(His)4] site is ~109.47
    deg). Pure geometry — the caller supplies the Zn position and the candidate coordinating-N
    positions (e.g. His ND1/NE2 atoms parsed from the predicted CIF).

    Args:
        zn_pos: (3,) Zn coordinate.
        nitrogen_positions: (k, 3) candidate coordinating-N coordinates.
        cutoff: max Zn-N distance (A) to count an N as coordinating (default 2.6).

    Returns a dict:
        n_coordinating       : int   — Ns within cutoff
        mean_zn_n_distance   : float|None — mean Zn-N distance over the shell (None if empty)
        max_zn_n_distance    : float|None
        mean_n_zn_n_angle    : float|None — mean of all distinct N-Zn-N angles (deg), None if <2
        ideal_tetrahedral    : float  — 109.47, for reference
    """
    zn = np.asarray(zn_pos, dtype=float).reshape(3)
    ns = np.asarray(nitrogen_positions, dtype=float).reshape(-1, 3)
    if ns.shape[0] == 0:
        return {"n_coordinating": 0, "mean_zn_n_distance": None,
                "max_zn_n_distance": None, "mean_n_zn_n_angle": None,
                "ideal_tetrahedral": 109.47}

    dists = np.linalg.norm(ns - zn, axis=1)
    shell = ns[dists <= cutoff]
    shell_d = dists[dists <= cutoff]
    n_coord = int(shell.shape[0])
    if n_coord == 0:
        return {"n_coordinating": 0, "mean_zn_n_distance": None,
                "max_zn_n_distance": None, "mean_n_zn_n_angle": None,
                "ideal_tetrahedral": 109.47}

    # All distinct N-Zn-N angles among the coordinating shell.
    angles: list[float] = []
    vecs = shell - zn
    norms = np.linalg.norm(vecs, axis=1)
    for i in range(n_coord):
        for j in range(i + 1, n_coord):
            denom = norms[i] * norms[j]
            if denom <= 0:
                continue
            cos = float(np.dot(vecs[i], vecs[j]) / denom)
            cos = max(-1.0, min(1.0, cos))
            angles.append(float(np.degrees(np.arccos(cos))))

    return {
        "n_coordinating": n_coord,
        "mean_zn_n_distance": float(shell_d.mean()),
        "max_zn_n_distance": float(shell_d.max()),
        "mean_n_zn_n_angle": float(np.mean(angles)) if angles else None,
        "ideal_tetrahedral": 109.47,
    }


# ── CIF parsing helpers for the GPU path (deposit + predicted structures) ───────

_BACKBONE_ATOMS = ("N", "CA", "C", "O")


def backbone_heavy_atoms_from_cif(cif_path, chain_name: str = "A",
                                  atoms=_BACKBONE_ATOMS):  # pragma: no cover (needs gemmi+CIF)
    """Extract ordered backbone heavy-atom coords (n_res*len(atoms), 3) for one chain.

    Skips waters/het residues with no backbone; returns atoms in (residue, atom) order so two
    length-matched chains overlay residue-by-residue for the RMSD. Used to build BOTH the
    deposit and the predicted-design coordinate arrays for backbone_rmsd_to_deposit.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    coords: list = []
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                if res.name in ("HOH", "ZN"):
                    continue
                if res.find_atom("CA", "*") is None:
                    continue
                for aname in atoms:
                    a = res.find_atom(aname, "*")
                    if a is not None:
                        coords.append([a.pos.x, a.pos.y, a.pos.z])
        break  # first model only
    return np.asarray(coords, dtype=float)


def zn_and_his_nitrogens_from_cif(cif_path, chain_name: str = "A"):  # pragma: no cover (gemmi)
    """Return (zn_pos|None, his_nitrogen_positions (k,3)) parsed from a CIF.

    Finds the Zn atom (any chain) and all His/D-His imidazole nitrogens (ND1/NE2) on
    `chain_name`. Feeds zn_coordination_geometry on the GPU path.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    zn_pos = None
    n_positions: list = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.name == "ZN":
                    a = res.find_atom("ZN", "*")
                    if a is not None:
                        zn_pos = np.array([a.pos.x, a.pos.y, a.pos.z], dtype=float)
                if chain.name == chain_name and res.name in ("HIS", "DHI"):
                    for aname in ("ND1", "NE2"):
                        a = res.find_atom(aname, "*")
                        if a is not None:
                            n_positions.append([a.pos.x, a.pos.y, a.pos.z])
        break
    return zn_pos, np.asarray(n_positions, dtype=float).reshape(-1, 3)


def termini_distance_from_cif(cif_path, chain_name: str = "A"):  # pragma: no cover (gemmi)
    """N(term CA) <-> C(term CA) distance for one chain — the EMERGENT-closure proxy (#23).

    Phase-1 predicts a LINEAR peptide; a small N-to-C CA distance indicates the ring closed
    emergently (the deposit is a macrocycle). Reported, not enforced. None if <2 residues.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    cas: list = []
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                a = res.find_atom("CA", "*")
                if a is not None:
                    cas.append([a.pos.x, a.pos.y, a.pos.z])
        break
    if len(cas) < 2:
        return None
    cas = np.asarray(cas, dtype=float)
    return float(np.linalg.norm(cas[0] - cas[-1]))


def head_to_tail_closure_geometry_from_cif(cif_path, chain_name: str = "A"):  # pragma: no cover (gemmi)
    """GROUND-TRUTH head-to-tail ring closure (#23 calibration): the C-terminal carbonyl C
    of the LAST residue bonded to the N-terminal amide N of residue 1.

    Returns a dict with the load-bearing closure observables (None when the chain has < 2
    residues or the required atoms are missing):
        cn_distance      : float  — |C(res L) - N(res 1)| in A. A closed peptide bond is
                                     ~1.33 A; a linear/open terminus is many A apart.
        closure_omega    : float|None — the closure-amide dihedral CA(L)-C(L)-N(1)-CA(1)
                                     (deg). A planar trans amide is ~180 (or ~0 cis).
        omega_planarity  : float|None — amide_omega_score of that omega in [0,1]
                                     (1 = flat/planar, 0 = maximally twisted).
        closed           : bool   — cn_distance <= ``closed_cutoff`` (1.6 A by default,
                                     a generous bound around the 1.33 A peptide bond).
    The N/C atoms are the REAL backbone amide N and carbonyl C (not CA), matching the
    head_to_tail_closure_row COVALENT bond (C of res L -> N of res 1)."""
    import gemmi
    from xenodesign.geometry import amide_omega_score

    structure = gemmi.read_structure(str(cif_path))
    residues = []
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                if res.name in ("HOH", "ZN"):
                    continue
                if res.find_atom("CA", "*") is None:
                    continue
                residues.append(res)
        break  # first model only
    if len(residues) < 2:
        return None

    def _xyz(res, name):
        a = res.find_atom(name, "*")
        return None if a is None else np.array([a.pos.x, a.pos.y, a.pos.z], dtype=float)

    first, last = residues[0], residues[-1]
    n1 = _xyz(first, "N")        # N-term amide N
    cL = _xyz(last, "C")         # C-term carbonyl C
    ca1 = _xyz(first, "CA")
    caL = _xyz(last, "CA")
    out: dict = {"cn_distance": None, "closure_omega": None,
                 "omega_planarity": None, "closed": False}
    if n1 is None or cL is None:
        return out
    cn = float(np.linalg.norm(cL - n1))
    out["cn_distance"] = cn
    out["closed"] = bool(cn <= 1.6)
    if ca1 is not None and caL is not None:
        from xenodesign.geometry import dihedral
        out["closure_omega"] = float(dihedral(caL, cL, n1, ca1))
        out["omega_planarity"] = float(amide_omega_score(caL, cL, n1, ca1))
    return out


# ── Loop objective (ipTM+pLDDT composite; CPU-clean, deferred scorer import) ─────

# ── INTRAMOLECULAR objective (NO-TARGET free cyclic/linear peptide, target_type='none') ──
#
# For a SINGLE-chain design there is no interface, so ipTM and a binder-chain index are
# UNDEFINED — the alpha _loop_score_fn does not apply. The single-chain spec replaces it with a 4-term
# intramolecular score, each term normalised to [0, 1], combined with explicit named weights:
#
#   (1) mainchain_plddt — mean pLDDT of the mainchain (N, CA, C, O) atoms of the two CYCLIZING
#       residues (the termini: residue 1 and residue L), divided by 100. This is the local
#       confidence right at the ring-closure seam, which is where a bad macrocycle shows first.
#   (2) chirality       — 1 - chirality_violation_frac over the cycle residues (reuse the design
#       metric is_chirality_violation against each residue's intended L/D label). 1.0 = every
#       stereocenter handed as designed.
#   (3) geometry        — closure + backbone geometry quality: the planarity of the head-to-tail
#       amide bond (omega = CA(L)-C(L)-N(1)-CA(1) ~ 0/180; amide_omega_score) AVERAGED with a
#       backbone valence-angle sanity term (N-CA-C ~ 111 deg; angle_deviation_score). One scalar.
#   (4) ptm             — prediction.ptm (already [0,1]). NOTE: chai UNDER-estimates pTM for
#       D / mixed-chirality peptides, so this term is deliberately the lowest-weighted.
#
# Weights are EXPLICIT and easy to tune (they sum to 1.0). Defaults below.
INTRAMOLECULAR_WEIGHTS: dict = {
    "mainchain_plddt": 0.30,   # confidence at the cyclisation seam (termini mainchain)
    "chirality":       0.25,   # handedness as designed (mixed L/D)
    "geometry":        0.25,   # closure-amide planarity + backbone valence-angle sanity
    "ptm":             0.20,   # global fold confidence (D-underestimated -> lowest weight)
}

# Ideal backbone N-CA-C valence angle (deg) and the tolerance past which credit decays to 0.
_IDEAL_N_CA_C = 111.0
_ANGLE_TOL_DEG = 12.0


def _mainchain_plddt_term(records) -> float:
    """Term (1): mean pLDDT of the (N,CA,C,O) mainchain atoms of the CYCLIZING termini (res 1, L).

    ``records`` are per-residue dicts carrying a ``plddt`` sub-dict {atom_name: pLDDT}. Atoms
    absent from a record are skipped. Returns the mean over the present mainchain atoms of the
    first and last residue, divided by 100 (-> [0,1]). Empty -> 0.0.
    """
    if len(records) < 1:
        return 0.0
    termini = [records[0]] if len(records) == 1 else [records[0], records[-1]]
    vals = []
    for rec in termini:
        pl = rec.get("plddt", {})
        for atom in ("N", "CA", "C", "O"):
            if atom in pl:
                vals.append(float(pl[atom]))
    if not vals:
        return 0.0
    return float(max(0.0, min(1.0, np.mean(vals) / 100.0)))


def _chirality_term(records) -> float:
    """Term (2): 1 - chirality_violation_frac over the cycle residues (reuse design metric).

    Each record needs N/CA/C/CB coords + a ``chirality`` label ('L'|'D'); GLY (no CB) is achiral
    and skipped. Returns 1.0 when no stereocenters (nothing to fault)."""
    from xenodesign.chirality import is_chirality_violation
    total = violations = 0
    for rec in records:
        if "CB" not in rec:
            continue
        total += 1
        if is_chirality_violation(rec["N"], rec["CA"], rec["C"], rec["CB"],
                                  rec.get("chirality", "L")):
            violations += 1
    if total == 0:
        return 1.0
    return float(1.0 - violations / total)


def _geometry_term(records) -> float:
    """Term (3): closure-amide planarity (omega at C(L)->N(1)) AVERAGED with backbone N-CA-C
    valence-angle sanity. Mean of two [0,1] sub-scores. Needs >=2 residues; else 1.0 (neutral)."""
    from xenodesign.geometry import amide_omega_score, valence_angle, angle_deviation_score
    if len(records) < 2:
        return 1.0
    first, last = records[0], records[-1]
    # Head-to-tail amide: CA(L)-C(L)-N(1)-CA(1) planarity (the closure bond C(L)->N(1)).
    omega_score = amide_omega_score(last["CA"], last["C"], first["N"], first["CA"])
    # Backbone valence-angle sanity: N-CA-C for every residue with all three atoms.
    angles, ideals = [], []
    for rec in records:
        if {"N", "CA", "C"} <= rec.keys():
            angles.append(valence_angle(rec["N"], rec["CA"], rec["C"]))
            ideals.append(_IDEAL_N_CA_C)
    angle_score = angle_deviation_score(angles, ideals, tol_deg=_ANGLE_TOL_DEG)
    return float(0.5 * (omega_score + angle_score))


def intramolecular_terms_from_records(records, ptm: float) -> dict:
    """Compute the 4 intramolecular terms from per-residue records + the prediction's pTM.

    Pure / CPU-testable: ``records`` is a list of per-residue dicts with N/CA/C[/O/CB] coords,
    an optional per-atom ``plddt`` sub-dict, and a ``chirality`` label. ``ptm`` is clamped to
    [0,1]. Returns {'mainchain_plddt','chirality','geometry','ptm'} each in [0,1]."""
    return {
        "mainchain_plddt": _mainchain_plddt_term(records),
        "chirality": _chirality_term(records),
        "geometry": _geometry_term(records),
        "ptm": float(max(0.0, min(1.0, ptm))),
    }


def combine_intramolecular_terms(terms: dict, weights: dict = INTRAMOLECULAR_WEIGHTS) -> float:
    """Weighted sum of the 4 intramolecular terms (weights sum to 1.0 -> result in [0,1])."""
    return float(sum(weights[k] * float(terms[k]) for k in weights))


def cyclic_records_from_cif(cif_path, chain_name: str = "A"):  # pragma: no cover (needs gemmi+CIF)
    """Parse one chain into the per-residue records the intramolecular objective consumes.

    Each record = {N,CA,C[,O,CB] coords, 'plddt': {atom: b_iso}, 'chirality': 'L'|'D'}. chai
    writes per-atom pLDDT into the CIF B-factor column, so b_iso IS the pLDDT. The chirality
    label is read from the residue name (a parenthesised D-CCD residue, e.g. 'DHI'/'DAL', is 'D';
    a canonical 3-letter name is 'L'; achiral GLY has no CB so it is skipped by the chirality
    term). Used on the GPU path; the pure terms above are CPU-tested with synthetic records.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    records: list = []
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                if res.name in ("HOH", "ZN"):
                    continue
                atoms = {a.name: a for a in res}
                if not {"N", "CA", "C"} <= atoms.keys():
                    continue
                rec: dict = {}
                plddt: dict = {}
                for name in ("N", "CA", "C", "O", "CB"):
                    a = atoms.get(name)
                    if a is not None:
                        rec[name] = np.array([a.pos.x, a.pos.y, a.pos.z], dtype=float)
                        plddt[name] = float(a.b_iso)
                rec["plddt"] = plddt
                # D-CCD residue names start with 'D' (DAL, DHI, ...) and are length 3; canonical
                # L names (ALA, HIS, ...) do not. Heuristic, matches the mixed_chirality_fasta emit.
                nm = res.name.upper()
                rec["chirality"] = "D" if (len(nm) == 3 and nm.startswith("D")
                                           and nm not in ("ASP", "ASN", "ASX")) else "L"
                records.append(rec)
        break  # first model only
    return records


def make_intramolecular_score_fn(wrapper):
    """Build the per-iteration loop score_fn for the NO-TARGET free-cyclic peptide objective.

    Mirrors ``make_mixed_loop_score_fn``: the HalluLoop calls ``score_fn(prediction)`` right after
    the structure step, so the candidate's scored CIF lives at ``wrapper.last_out_dir``. This
    closure parses the binder chain (A) from that CIF, computes the 4 intramolecular terms +
    combines them. GRACEFUL FALLBACK: any failure to find/parse the CIF (no wrapper, no last_out_dir,
    CPU mock, gemmi error) degrades to a pTM-only finite score (still well-defined for one chain),
    so a single bad iteration never crashes the loop.
    """
    from xenodesign.cif_io import _best_cif_path

    def _score(prediction) -> float:
        ptm = float(getattr(prediction, "ptm", 0.0) or 0.0)
        out_dir = getattr(wrapper, "last_out_dir", None)
        if out_dir is None:
            # No CIF available (CPU mock / first call): pTM-only fallback, weighted as in the
            # full combine so the scale is comparable.
            return float(INTRAMOLECULAR_WEIGHTS["ptm"] * max(0.0, min(1.0, ptm)))
        try:
            cif = _best_cif_path(out_dir)
            records = cyclic_records_from_cif(cif, chain_name="A")
            terms = intramolecular_terms_from_records(records, ptm=ptm)
            return combine_intramolecular_terms(terms)
        except Exception:
            return float(INTRAMOLECULAR_WEIGHTS["ptm"] * max(0.0, min(1.0, ptm)))

    return _score


# ── Result assembly (behaviour-preserving vs run_cyclic_design's result dict) ────

def _best_step(history):
    """Highest-score LoopStep in the trajectory (greedy selection; panel-agnostic here)."""
    return max(history, key=lambda h: getattr(h, "score", float("-inf")))


def _assemble_cyclic_result(cfg, history, panel_result, case, out_dir,
                            *, l_seed_iptm: float = 0.0, wall_time_s: float = 0.0) -> dict:
    """Build the cyclic result dict + write cyclic_result.json (RECALL-oriented).

    Selects the best trajectory step (ipTM/pTM drive), records the Zn-N geometry / RMSD-to-
    deposit / closure-proxy fields when available on the step's prediction, and writes the
    same cyclic_result.json the GPU driver emits. Geometry fields default to None on CPU
    (no CIF parsed); the GPU smoke (T10) populates them from the predicted structure."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PROVENANCE (Part F): record the PANEL-selected step (whose per-iteration CIF is the model
    # the dispatcher deposits as the result, ``loop/iter_{sel_idx:03d}``), mirroring the alpha
    # path's ``panel_result.selected_idx`` selection. Fall back to the greedy best() only when no
    # panel result with a valid index is available, so the recorded sequence always matches the
    # deposited CIF.
    sel_idx = None
    if panel_result is not None:
        idx = getattr(panel_result, "selected_idx", None)
        if idx is not None and history and 0 <= int(idx) < len(history):
            sel_idx = int(idx)
    if sel_idx is not None:
        best = history[sel_idx]
    else:
        best = _best_step(history) if history else None
    pred = getattr(best, "prediction", None) if best is not None else None

    result = {
        "case_id": "cyclic",
        "n_iters": len(history),
        "l_seed_iptm": float(l_seed_iptm),
        "wall_time_s": float(wall_time_s),
        "selected_iptm": (float(pred.iptm) if pred is not None and hasattr(pred, "iptm")
                          else None),
        "selected_ptm": (float(pred.ptm) if pred is not None and hasattr(pred, "ptm")
                         else None),
        "selected_d_fasta": (getattr(best.state, "d_fasta", None)
                             if best is not None else None),
        "backbone_rmsd_to_deposit": getattr(pred, "backbone_rmsd_to_deposit", None),
        "baseline_backbone_rmsd": case.baseline.backbone_rmsd,
        "zn_coordination_geometry": getattr(pred, "zn_coordination_geometry", None),
        "termini_distance_closure_proxy": getattr(pred, "termini_distance", None),
        "closure": bool(cfg.restraint.params.get("closure")),
        "restraints": bool(cfg.restraints_on),
        "phase": "linear+emergent-closure (mainchain head-to-tail closure opt-in)",
        "out_dir": str(out_dir),
    }

    # POST-SELECTION VERIFICATION (Part G): for a METAL target, run the EXISTING best-effort
    # metal-geometry gate on the SELECTED CIF and RECORD (never enforce) its verdict. Not a hard
    # loop gate — purely a recorded check. Skipped for non-metal / no-target (no metal site).
    if cfg.target.target_type == "metal":
        sel = sel_idx if sel_idx is not None else 0
        iter_dir = out_dir / "loop" / f"iter_{sel:03d}"
        try:
            from xenodesign.cif_io import _best_cif_path
            cif_path = _best_cif_path(iter_dir)
        except Exception:
            cif_path = iter_dir  # best-effort: the gate never raises (pass-through on a non-CIF)
        gate = metal_geometry_gate(cif_path)
        result["metal_geometry"] = {
            "geometry": gate.geometry,
            "perplexity": gate.perplexity,
            "passed": bool(gate.passed),
        }

    (out_dir / "cyclic_result.json").write_text(
        json.dumps(result, indent=2,
                   default=lambda o: getattr(o, "tolist", lambda: str(o))()))
    return result


# ── BinderClass adapter ──────────────────────────────────────────────────────────

class Cyclic:
    """Cyclic Zn-macrocycle binder class (single-chain, no interface; geometry RECALL).

    Implements the :class:`~xenodesign.classes.base.BinderClass` protocol. The dispatcher
    wires these hooks into the untouched HalluLoop: ``seed`` pins the coordinating His L/D,
    ``restraints`` writes the His<->Zn metal_coordination CSV (+ opt-in mainchain closure),
    ``objective`` drives on ipTM/pTM, and ``report`` assembles the RECALL result dict.
    """

    case_id = "cyclic"

    def seed(self, cfg, target_seq) -> SeedSpec:
        """FROM-SCRATCH unified UNCONDITIONAL seed (PepMLM cannot condition on a metal).

        Routes through the ONE ``seed.unified_seed`` path with ``target_seq=""`` (no protein
        target) at ``resolve_binder_length(cfg)`` (default 24 = the S2-symmetric 6UFA full length;
        16 for ``target_type='none'``; overridable via --binder_length). The seed NEVER inherits
        the deposit's sequence or length. The coordinating-His placement is OPT-IN: only when
        restraints are ON do we pin the case's ``his_resnums`` (with their L/D handedness) as fixed
        positions; otherwise the seed carries no mandatory His scaffold."""
        from xenodesign.config import resolve_binder_length
        from xenodesign.seed import make_configured_generator, unified_seed

        length = resolve_binder_length(cfg)
        gen = make_configured_generator(cfg)
        fixed = self._his_positions(cfg, length) if cfg.restraints_on else None
        result = unified_seed(gen, target_seq="", length=length, reverse=False,
                              fixed_positions=fixed, fixed_residue="H")
        one = self._place_declared_residues(cfg, result.one_letter, length)
        one = self._ensure_canonical_anchor(one, fixed)
        return SeedSpec(one_letter=one,
                        fixed_chirality=dict(result.fixed_chirality))

    def ss_bias(self, cfg, case):
        from xenodesign.benchmark.cases import ss_bias_config_for_case
        return ss_bias_config_for_case(case)  # anti_alpha -> 0.0

    def restraints(self, cfg, case, out_dir, target_ctx):
        if not cfg.restraints_on:
            return None
        # NO-TARGET (target_type='none'): there is NO Zn chain, so SKIP the His<->Zn metal
        # coordination rows; the binder is the sole chain A and only the opt-in head-to-tail
        # closure row (on chain A) is written. The discriminator is the TARGET TYPE, not the
        # entity list — the metal case legitimately passes target_ctx=None (legacy standalone
        # order) yet still needs the coordination rows.
        metal = cfg.target.target_type != "none"
        closure = bool(cfg.restraint.params.get("closure"))
        seed_result = self._seed_result(cfg, case) if closure else None

        if not metal:
            # Free peptide: binder=A, no Zn. Closure (if any) is on chain A.
            if not closure:
                return None   # no Zn + no closure -> unconstrained free peptide
            return write_cyclic_restraints(case, out_dir, seed_result=seed_result,
                                           closure=closure, binder_chain="A", zn_chain="B",
                                           metal=False)

        # Metal case: the wrapper appends the binder LAST, so when the assembled entity list is
        # given (target_ctx with the Zn ligand) the Zn is chain A and the peptide/His is the last
        # chain; no ctx -> legacy standalone order (peptide=A, Zn=B). Read the chains from the ONE
        # chain contract (``ChainRoles.from_entities``) instead of an inline chr(...), so this site
        # uses the SAME derivation as the seq-update extractor and the double-flip (the dispatcher
        # gap that this whole contract closes).
        entities = (target_ctx[0] if target_ctx else None) or []
        if entities:
            from xenodesign.targets import ChainRoles
            roles = ChainRoles.from_entities(entities)
            binder_chain = roles.binder      # Zn=A -> binder=B
            zn_chain = roles.targets[0]      # the Zn ligand chain (A)
        else:
            binder_chain, zn_chain = "A", "B"
        return write_cyclic_restraints(case, out_dir, seed_result=seed_result,
                                       closure=closure, binder_chain=binder_chain,
                                       zn_chain=zn_chain, metal=True,
                                       coord_residues=self._coord_residues(cfg))

    def closure(self, cfg, seed_spec) -> list:
        """PRIMARY cyclization: mainchain head-to-tail COVALENT closure (opt-in).

        Returns ``[]`` for the default LINEAR + emergent-closure run; when
        ``restraint.params['closure']`` is set, returns a single head-to-tail COVALENT
        backbone bond row (N-to-C ring bond) built from the seed's termini. Disulfide
        closure is secondary and not used for the metal-coordinated 6UFA site."""
        if not cfg.restraint.params.get("closure"):
            return []
        sr = SeedResult(one_letter=seed_spec.one_letter,
                        length=len(seed_spec.one_letter),
                        reverse_applied=False, conditioned=False,
                        fixed_chirality=dict(seed_spec.fixed_chirality))
        return [build_closure_row(sr)]

    def seq_update(self, cfg, wrapper, seed_spec, roles=None):
        """Per-iteration sequence-update fn. The cyclic recall case keeps the pinned seed;
        the dispatcher's fallback (identity to the seed one-letter) is sufficient on CPU, and
        the GPU path re-designs the non-coordinating backbone via the wrapper's MPNN.

        ``roles`` threads the dispatch chain contract: metal case (Zn=A) -> binder 'B';
        no-target free peptide -> binder/context BOTH 'A' (the bug that crashed iter_000)."""
        from xenodesign.classes.alpha import make_alpha_seq_update_fn
        # Freeze declared coordinators in the MPNN mask so pinned donors (e.g. His) never
        # drift. coord_residues tuple[0] is the 1-based position -> 0-based for the mask.
        frozen_positions = {int(t[0]) - 1 for t in self._coord_residues(cfg)}
        return make_alpha_seq_update_fn(wrapper, num_seqs=cfg.loop.num_seqs,
                                        backend=cfg.loop.backend, roles=roles,
                                        frozen_positions=frozen_positions or None)

    def accept_fns(self, cfg):
        from xenodesign.loop import compose_accept_fns
        return compose_accept_fns(None)

    def objective(self, cfg, wrapper):
        # NO-TARGET (target_type='none'): single free cyclic/linear peptide -> ipTM/binder-chain
        # index are undefined, so use the INTRAMOLECULAR 4-term objective (mainchain-pLDDT of the
        # cyclising termini + chirality goodness + closure/backbone geometry + pTM).
        if cfg.target.target_type == "none":
            return make_intramolecular_score_fn(wrapper)
        # Metal case: share alpha's ipTM + binder-chain pLDDT objective (DRY — was a byte-identical
        # local copy; cyclic already reuses alpha's seq-update). Lazy import mirrors seq_update.
        from xenodesign.classes.alpha import _loop_score_fn
        return _loop_score_fn  # recall case: ipTM/pTM drive; geometry scored in report

    def referee(self, cfg, loop_dir, esm_judge, roles=None):
        return lambda step, i: None  # no per-step referee for the recall case (no chain read)

    def report(self, cfg, history, panel_result, case, out_dir,
               *, l_seed_iptm: float = 0.0, wall_time_s: float = 0.0) -> dict:
        return _assemble_cyclic_result(cfg, history, panel_result, case, out_dir,
                                       l_seed_iptm=l_seed_iptm, wall_time_s=wall_time_s)

    # ── internal ────────────────────────────────────────────────────────────────

    def _coord_residues(self, cfg):
        """The DECLARATIVE coordinator list from ``cfg.restraint.params['coord_residues']``.

        Each entry is a (pos, one_letter, three_letter, chirality) tuple (as stored by the CLI
        flag wiring). Returns [] when the flag was absent — callers then fall back to the case's
        hardcoded His defaults."""
        params = cfg.restraint.params if cfg.restraint else {}
        return list(params.get("coord_residues") or [])

    def _his_positions(self, cfg, length) -> dict:
        """OPT-IN coordinating positions+chirality for the from-scratch cyclic seed.

        DECLARATIVE override (``--coord_residues``): when ``cfg.restraint.params['coord_residues']``
        is set, those (pos, chirality) pairs ARE the fixed positions (generalizing beyond His/Zn —
        any donor, any chirality). Absent -> the case's ``metal_coordination`` ``his_resnums`` +
        ``CYCLIC_HIS_CHIRALITY`` defaults. Positions outside the from-scratch ``length`` are
        dropped. Returns {} when neither source applies. NEVER mandatory: only consulted when
        restraints are ON."""
        coords = self._coord_residues(cfg)
        if coords:
            # Tuple is (pos, one_letter, three_letter, chirality[, atom]); index for back-compat.
            return {int(t[0]): t[3]
                    for t in coords if 1 <= int(t[0]) <= length}
        case = get_case("cyclic")
        spec = case.restraint
        if spec is None or spec.kind != "metal_coordination":
            return {}
        resnums = spec.params.get("his_resnums", ())
        return {int(p): CYCLIC_HIS_CHIRALITY.get(int(p), "L")
                for p in resnums if 1 <= int(p) <= length}

    def _seed_result(self, cfg, case):
        """Rebuild the cyclic SeedResult (for closure-row construction at restraint time).

        Mirrors :meth:`seed` exactly (unified UNCONDITIONAL from-scratch path + opt-in His), so the
        closure/restraint rows reference the same seed the loop starts from."""
        from xenodesign.config import resolve_binder_length
        from xenodesign.seed import make_configured_generator, unified_seed

        length = resolve_binder_length(cfg)
        gen = make_configured_generator(cfg)
        fixed = self._his_positions(cfg, length) if cfg.restraints_on else None
        result = unified_seed(gen, target_seq="", length=length, reverse=False,
                              fixed_positions=fixed, fixed_residue="H")
        one = self._place_declared_residues(cfg, result.one_letter, length)
        one = self._ensure_canonical_anchor(one, fixed)
        return SeedResult(one_letter=one, length=result.length,
                          reverse_applied=result.reverse_applied,
                          conditioned=result.conditioned,
                          fixed_chirality=dict(result.fixed_chirality))

    def _place_declared_residues(self, cfg, one_letter: str, length: int) -> str:
        """Overwrite the DECLARED coordinator positions with their REAL one-letter identities.

        ``unified_seed`` places a single ``fixed_residue`` ('H') at the fixed positions; the
        declarative ``--coord_residues`` flag generalizes that to ANY donor (His 'H', Cys 'C',
        Asp 'D', ...), so when coordinators are declared (and restraints ON) we overwrite each
        declared position with its real one-letter code. No-op when the flag is absent (the
        'H' His default already placed by unified_seed stands)."""
        if not cfg.restraints_on:
            return one_letter
        chars = list(one_letter)
        for t in self._coord_residues(cfg):
            # Tuple is (pos, one_letter, three_letter, chirality[, atom]); index for back-compat.
            pos, ol = int(t[0]), t[1]
            if 1 <= pos <= length:
                chars[pos - 1] = ol
        return "".join(chars)

    @staticmethod
    def _ensure_canonical_anchor(one_letter: str, fixed: "dict | None",
                                 chirality_map: "dict | None" = None,
                                 *, default_chirality: str = "D") -> str:
        """Gly-guard the from-scratch cyclic seed when it is otherwise all-one-handedness (#9, Part E).

        chai needs >=1 canonical residue per chain to tokenize; a chain that is ENTIRELY one
        handedness (e.g. all-D) is fully non-canonical and crashes at iter_000 (ADR-004). A
        genuinely MIXED L/D design already carries tokenizable L residues, so it needs no anchor.

        Rule (Part E): ensure an achiral Gly when, among the NON-coordinator positions, there is
        NO D-residue OR NO L-residue present (i.e. the design is otherwise all-one-handedness) AND
        no Gly already exists. The Gly is placed at the C-TERMINUS (last non-coordinator position),
        never overwriting a declared coordinator. ``fixed`` keys are the 1-based coordinator
        positions; non-coordinator handedness is read from ``chirality_map`` (positions absent
        default to ``default_chirality`` — the cyclic backbone is encoded uniformly all-D in the
        no-target path, L in the mixed-metal path)."""
        if "G" in one_letter:
            return one_letter
        n = len(one_letter)
        pinned = {int(p) for p in (fixed or {})}  # 1-based coordinator positions
        chir = dict(chirality_map or {})
        non_coord = [i for i in range(n) if (i + 1) not in pinned]
        hands = {chir.get(i + 1, default_chirality) for i in non_coord}
        # Mixed (both 'D' and 'L' present among non-coordinators): chai can tokenize the L
        # residues, so no forced Gly is needed.
        if "D" in hands and "L" in hands:
            return one_letter
        # All-one-handedness: place the achiral anchor at the C-TERMINAL non-coordinator position.
        if non_coord:
            i = non_coord[-1]
            return one_letter[:i] + "G" + one_letter[i + 1:]
        return one_letter  # pragma: no cover (every position pinned — impossible for real lengths)
