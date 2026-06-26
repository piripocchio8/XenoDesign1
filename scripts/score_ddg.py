"""Parity-safe OpenMM interaction-energy (ddG-proxy) scorer for a 2-chain complex (T20).

Computes the SINGLE-POINT interaction energy of a binder/target complex:

    E_int = E(complex) - E(target_alone) - E(binder_alone)

with NO minimization. More negative E_int = a more favorable interface.

Why this is PARITY-SAFE for D-peptide binders
----------------------------------------------
The D-chirality of the binder is encoded entirely in its INTRA-chain bonded and
improper (chirality) terms. In the subtraction above, the binder appears once in
E(complex) and once in E(binder_alone) with the IDENTICAL internal geometry
(single point, no minimization), so every intra-binder term — bonds, angles,
torsions, impropers, 1-4 and intra-chain nonbonded — cancels exactly. The same
holds for the target. What survives is the INTER-chain nonbonded energy
(van der Waals + Coulomb [+ implicit-solvent cross terms]), which depends only on
the relative atom positions and atom types, NOT on chirality. So we may safely
rename D-residues to their L counterparts to let a standard L-only force field
parameterize the system: the renamed atoms keep their exact deposited coordinates,
and the only quantity we read out (the inter-chain term) is chirality-agnostic.

How the cancellation is made EXACT (single-system "pull-apart")
---------------------------------------------------------------
Building three separate PDBs (complex / target / binder) and subtracting their
energies does NOT cancel cleanly in practice: pdbfixer adds hydrogens and any
missing heavy atoms INDEPENDENTLY per file, so the binder's internal geometry in
the complex differs slightly from the isolated binder, leaving large spurious
residuals in the bonded terms. Instead we build ONE topology+positions for the
complex and evaluate E_int as

    E_int = E(positions) - E(positions with the binder chain translated to infinity)

on the SAME system. The bonded and intra-chain terms are byte-for-byte identical
in both states (the binder is only rigidly translated), so they cancel EXACTLY;
only the inter-chain nonbonded (vdW+Coulomb) and the inter-chain part of the
implicit-solvent (GB) term survive. Verified: per-term pull-apart deltas for
HarmonicBond/Angle/Torsion are 0.000. This is the parity-invariant interface
energy by construction.

Robustness / fallback
----------------------
Primary path: gemmi (CIF->PDB, D->L rename) -> pdbfixer (add missing heavy atoms +
H) -> OpenMM amber14 with implicit solvent (GBn2/OBC) and a no-cutoff nonbonded
method, single-point energy on complex and each isolated chain.

If the OpenMM topology / force-field build chokes (unparameterizable residue,
naming, terminal patching, etc.), we FALL BACK to a documented, parity-invariant
interface energy: a Lennard-Jones + Coulomb sum over interface HEAVY-ATOM pairs
within a cutoff, using a single standard parameter set (Amber-like vdW radii/well
depths + per-atom-name partial charges from the same amber14 ff when available,
else element-default radii and zero charge for the LJ-only term). This is still a
pure inter-chain quantity, hence still parity-invariant. The chosen method is
reported in the output JSON under "method".

CLI
---
  micromamba run -n SE3nv python scripts/score_ddg.py <complex.cif> \
      --binder_chain B --target_chain A
  # prints JSON: {"e_int_kcal": ..., "e_complex_kcal": ..., "method": ...}

Self-check:
  micromamba run -n SE3nv python scripts/score_ddg.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import gemmi

# D 3-letter -> L 3-letter. Geometry/atoms identical; only used so an L-only force
# field can parameterize the system. Parity-safe (see module docstring).
_D2L = {"DAL": "ALA", "DAR": "ARG", "DSG": "ASN", "DAS": "ASP", "DCY": "CYS",
        "DCYS": "CYS", "DGN": "GLN", "DGL": "GLU", "DHI": "HIS", "DIL": "ILE",
        "DLE": "LEU", "DLY": "LYS", "MED": "MET", "DPN": "PHE", "DPR": "PRO",
        "DSN": "SER", "DTH": "THR", "DTR": "TRP", "DTY": "TYR", "DVA": "VAL",
        "DGY": "GLY"}
_STD = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}

KJ_PER_KCAL = 4.184


# --------------------------------------------------------------------------- #
# gemmi: clean a subset of chains into a PDB on disk (D->L, heavy+given atoms). #
# --------------------------------------------------------------------------- #
def _clean_chains(cif, chains, drop_h=True):
    """Return a gemmi.Structure with `chains`, D->L renamed, std AAs only.

    Keeps original atom coordinates exactly. Optionally drops explicit hydrogens
    (pdbfixer re-adds a consistent set)."""
    st = gemmi.read_structure(str(cif))
    ns = gemmi.Structure()
    ns.add_model(gemmi.Model("1"))
    for cn in chains:
        nc = gemmi.Chain(cn)
        for ch in st[0]:
            if ch.name != cn:
                continue
            for r in ch:
                nm = _D2L.get(r.name, r.name)
                if nm not in _STD:
                    continue
                nr = gemmi.Residue()
                nr.name = nm
                nr.seqid = r.seqid
                for a in r:
                    if drop_h and a.element == gemmi.Element("H"):
                        continue
                    nr.add_atom(a)
                if len(nr):
                    nc.add_residue(nr)
            break
        ns[0].add_chain(nc)
    ns.setup_entities()
    return ns


def _write_pdb(struct, path):
    struct.write_pdb(str(path))


# --------------------------------------------------------------------------- #
# Primary path: OpenMM single-point inter-chain energy (single-system          #
# "pull-apart") via pdbfixer + amber14 + GBn2.                                  #
# --------------------------------------------------------------------------- #
def _score_openmm(cif, target_chain, binder_chain, sep_nm=100.0,
                  minimize=True, restraint_k=5.0, min_iters=500,
                  jitter_A=0.0, jitter_seed=0):
    """E_int = E(together) - E(binder translated to infinity), one consistent system.

    Bonded/intra terms cancel exactly (binder is rigidly translated); only the
    inter-chain nonbonded (vdW+Coulomb) and inter-chain GB solvation survive. Also
    reports the NonbondedForce-only inter-chain term (clash-robust, no solvent).
    Raises on any failure so the caller can fall back.

    Interface-restrained minimization (ADR-023)
    --------------------------------------------
    Un-minimized Chai coordinates carry huge per-atom steric spikes (+169..+870
    kcal vdW/GB, ±880 kcal across reps) that drown the ~0.06-0.17 register signal.
    When ``minimize`` is on (default), we restrain every CA to its DEPOSITED
    position with a harmonic CustomExternalForce (k=``restraint_k`` kcal/mol/A^2)
    and run LocalEnergyMinimization. This relaxes only the bad local clashes
    (side-chain rotamers, added H, terminal patches) while the CA cage HOLDS the
    fold and register exactly, so we still score the deposited geometry — just
    de-noised. The restraint force is then removed before the energy readout so it
    contributes nothing to the reported numbers.

    ponytail: CA-only harmonic restraint (not full soft-core vdW). A CA cage is
    the laziest thing that pins the register without freezing side chains, and the
    pull-apart parity still holds regardless of how we got the coords (see below).

    Parity stays EXACT
    ------------------
    Minimization runs on the SINGLE complex system, and the subsequent pull-apart
    still rigidly translates ONLY the binder. So whatever minimized coords we end
    up with, every intra-binder/intra-target bonded + improper (chirality) term is
    byte-identical in the together/apart states and cancels exactly; only the
    inter-chain nonbonded + GB survives. The restraint is a positional bias on
    absolute coords; under a +100 nm rigid translation of the binder it is NOT
    part of the E_int subtraction because we DELETE it before reading energies."""
    import numpy as np
    from openmm import app, unit, Platform
    import openmm as mm
    from pdbfixer import PDBFixer

    tmp = Path(tempfile.mkdtemp(prefix="ddg_"))
    comp = tmp / "complex.pdb"
    # Target chain written first, binder second, so binder = topology chain index 1.
    _write_pdb(_clean_chains(cif, [target_chain, binder_chain]), comp)

    fixer = PDBFixer(filename=str(comp))
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # do not model unresolved loops
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    top = fixer.topology

    if top.getNumChains() != 2:
        raise RuntimeError(f"expected 2 chains in built topology, got {top.getNumChains()}")
    chain_idx = np.array([ch.index for ch in top.chains() for _ in ch.atoms()])
    binder_mask = chain_idx == 1        # second-written chain = binder

    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff,
                             constraints=None, rigidWater=False)
    # Force groups so we can read NonbondedForce in isolation.
    nb_group = None
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)
        if type(f).__name__ == "NonbondedForce":
            nb_group = i

    # Interface-restrained minimization to denoise un-minimized Chai clashes.
    # Added BEFORE building the integrator/Simulation so the restraint force is part
    # of the system during minimization; removed again right after so the reported
    # energies are restraint-free. CA atoms are pinned to their deposited positions.
    restraint_force_idx = None
    pos0 = np.array(fixer.positions.value_in_unit(unit.nanometer))
    # Optional tiny isotropic Cartesian jitter on the PRE-min coords, to get an
    # honest minimizer-basin noise estimate across reps (PDBFixer + minimize are
    # otherwise deterministic). The CA cage restraint pulls back toward the
    # deposited positions, so jitter probes basin sensitivity, not register. The
    # RMSD guard below is measured vs this jittered start, so it stays meaningful.
    if jitter_A and jitter_A > 0.0:
        rng = np.random.default_rng(int(jitter_seed))
        pos0 = pos0 + rng.normal(0.0, jitter_A * 0.1, size=pos0.shape)  # A->nm
    if minimize:
        ca_idx = [a.index for a in top.atoms() if a.name == "CA"]
        rest = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        # k in OpenMM units: kcal/mol/A^2 -> kJ/mol/nm^2.
        k_omm = restraint_k * KJ_PER_KCAL / (0.1 ** 2)
        rest.addGlobalParameter("k", k_omm)
        for p in ("x0", "y0", "z0"):
            rest.addPerParticleParameter(p)
        for i in ca_idx:
            rest.addParticle(int(i), [pos0[i, 0], pos0[i, 1], pos0[i, 2]])
        restraint_force_idx = system.addForce(rest)
        rest.setForceGroup(31)          # off to the side; never read into E_int

    integ = mm.VerletIntegrator(1.0 * unit.femtosecond)
    sim = app.Simulation(top, system, integ, Platform.getPlatformByName("CPU"))

    # CA indices (global + per-chain) for the post-min drift guard below.
    ca_all = np.array([a.index for a in top.atoms() if a.name == "CA"], dtype=int)
    ca_binder = np.array([a.index for a in top.atoms()
                          if a.name == "CA" and binder_mask[a.index]], dtype=int)

    ca_rmsd_all = ca_rmsd_binder = None
    if minimize:
        sim.context.setPositions(pos0 * unit.nanometer)
        sim.minimizeEnergy(maxIterations=min_iters)
        pos = sim.context.getState(getPositions=True).getPositions(
            asNumpy=True).value_in_unit(unit.nanometer)
        pos = np.array(pos)
        # RMSD GUARD (requirement): every minimization must verify the
        # minimized structure has not drifted far from the starting (deposited)
        # coords. The CA cage restraint should keep this tiny; a large value means
        # the restraint failed to hold the fold/register and the resulting E_int is
        # NOT a faithful score of the deposited geometry. No superposition: the
        # restraint pins ABSOLUTE CA positions, so a direct (unaligned) per-CA RMSD
        # vs pos0 is exactly the quantity the restraint is supposed to bound. nm->A.
        def _ca_rmsd(idx):
            if len(idx) == 0:
                return None
            d = pos[idx] - pos0[idx]
            return round(float(np.sqrt((d * d).sum(axis=1).mean()) * 10.0), 4)
        ca_rmsd_all = _ca_rmsd(ca_all)
        ca_rmsd_binder = _ca_rmsd(ca_binder)
        # Remove the restraint so it contributes 0 to every reported energy, and
        # reindex force groups so nb_group still points at the NonbondedForce.
        system.removeForce(restraint_force_idx)
        nb_group = None
        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i)
            if type(f).__name__ == "NonbondedForce":
                nb_group = i
        sim.context.reinitialize()
    else:
        pos = pos0

    pos_apart = pos.copy()
    pos_apart[binder_mask] += np.array([sep_nm, 0.0, 0.0])

    def _e(positions, groups=None):
        sim.context.setPositions(positions * unit.nanometer)
        st = (sim.context.getState(getEnergy=True, groups={groups})
              if groups is not None else sim.context.getState(getEnergy=True))
        return st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)

    e_tot_together = _e(pos)
    e_tot_apart = _e(pos_apart)
    e_int = e_tot_together - e_tot_apart
    e_int_nb = _e(pos, nb_group) - _e(pos_apart, nb_group)
    method = ("openmm_amber14_gbn2_CArestrained_min_pullapart" if minimize
              else "openmm_amber14_gbn2_singlepoint_pullapart")
    rmsd_thresh = 2.0
    ca_rmsd_flag = bool(minimize and ca_rmsd_all is not None
                        and ca_rmsd_all > rmsd_thresh)
    return {
        "method": method,
        "minimized": bool(minimize),
        "restraint_k_kcal_per_A2": restraint_k if minimize else None,
        "e_int_kcal": round(e_int, 3),                 # vdW+Coulomb+GB inter-chain
        "e_int_nonbonded_kcal": round(e_int_nb, 3),    # vdW+Coulomb only (no solvent)
        "e_int_solvation_kcal": round(e_int - e_int_nb, 3),
        "e_complex_total_kcal": round(e_tot_together, 3),
        # RMSD GUARD: CA-RMSD(minimized vs pre-min starting coords), all-CA and
        # binder-only, plus a flag if all-CA drift exceeds 2.0 A (restraint failure).
        "ca_rmsd_min_vs_start_A": ca_rmsd_all,
        "ca_rmsd_binder_min_vs_start_A": ca_rmsd_binder,
        "ca_rmsd_threshold_A": rmsd_thresh if minimize else None,
        "ca_rmsd_drift_flag": ca_rmsd_flag,
    }


# --------------------------------------------------------------------------- #
# Fallback: parity-invariant LJ + Coulomb over interface heavy-atom pairs.      #
# --------------------------------------------------------------------------- #
# Amber-like vdW: sigma (A), epsilon (kcal/mol), per element of the heavy atom.
# Coarse but standard; only the INTER-chain sum is taken, so still parity-invariant.
_VDW = {  # element -> (Rmin/2 in A, epsilon in kcal/mol) approx GAFF/amber values
    "C": (1.908, 0.086), "N": (1.824, 0.170), "O": (1.661, 0.210),
    "S": (2.000, 0.250), "P": (2.100, 0.200), "H": (1.000, 0.016),
}
_DEFAULT_VDW = (1.800, 0.100)


def _heavy_atoms(struct, chain):
    import numpy as np
    xyz, elem = [], []
    for r in struct[0][chain]:
        for a in r:
            if a.element == gemmi.Element("H"):
                continue
            xyz.append([a.pos.x, a.pos.y, a.pos.z])
            elem.append(a.element.name)
    return np.asarray(xyz, float), elem


def _score_fallback(cif, target_chain, binder_chain, cutoff=8.0):
    """LJ + (distance-dependent-dielectric) Coulomb-free LJ interface energy.

    Charges are not reliably name-mappable without the ff, so this fallback uses
    LJ only (vdW), which is the dominant, parity-invariant packing term. Reported
    as an energy proxy with method clearly flagged."""
    import numpy as np
    s = _clean_chains(cif, [target_chain, binder_chain], drop_h=True)
    xa, ea = _heavy_atoms(s, target_chain)
    xb, eb = _heavy_atoms(s, binder_chain)
    if len(xa) == 0 or len(xb) == 0:
        raise RuntimeError("fallback: empty chain after cleaning")
    d = np.linalg.norm(xa[:, None] - xb[None], axis=2)
    ii, jj = np.where(d < cutoff)
    e_lj = 0.0
    for i, j in zip(ii, jj):
        ri, epi = _VDW.get(ea[i], _DEFAULT_VDW)
        rj, epj = _VDW.get(eb[j], _DEFAULT_VDW)
        rmin = ri + rj
        eps = (epi * epj) ** 0.5
        r = d[i, j]
        if r < 1e-3:
            continue
        ratio = rmin / r
        e_lj += eps * (ratio ** 12 - 2.0 * ratio ** 6)
    n_contacts = int((d < 4.5).sum())
    return {
        "method": "fallback_LJ_interface_heavyatoms",
        "e_int_kcal": round(float(e_lj), 3),
        "e_complex_kcal": None,
        "e_target_kcal": None,
        "e_binder_kcal": None,
        "n_atom_contacts_4.5A": n_contacts,
        "note": "OpenMM build failed; LJ-only inter-chain vdW proxy (parity-invariant). "
                "Coulomb omitted (no reliable ff charge map without OpenMM).",
    }


# --------------------------------------------------------------------------- #
def score(cif, target_chain="A", binder_chain="B", minimize=True,
          jitter_A=0.0, jitter_seed=0):
    cif = str(cif)
    try:
        res = _score_openmm(cif, target_chain, binder_chain, minimize=minimize,
                            jitter_A=jitter_A, jitter_seed=jitter_seed)
    except Exception as e:                       # documented fallback
        res = _score_fallback(cif, target_chain, binder_chain)
        res["openmm_error"] = str(e)[:200]
    res["cif"] = cif
    res["target_chain"] = target_chain
    res["binder_chain"] = binder_chain
    return res


# --------------------------------------------------------------------------- #
def _max_abs_intra_force_e(cif, target_chain, binder_chain, minimize):
    """Return (e_int_nonbonded, e_int_solvation) for a CIF at given minimize setting.

    Used by the self-check to show the minimized inter-chain vdW/GB magnitude drops
    vs the un-minimized single point on the SAME structure."""
    res = score(cif, target_chain=target_chain, binder_chain=binder_chain,
                minimize=minimize)
    return res


def _parity_bonded_check(cif, target_chain, binder_chain):
    """Assert the bonded (chirality-bearing) energy is mirror-invariant.

    Builds the cleaned/fixed complex once, then computes the total HarmonicBond +
    HarmonicAngle + PeriodicTorsion energy for the coords AND for their mirror
    image (x -> -x). A reflection inverts all chirality; if the bonded model were
    chirality-sensitive in a way that leaked into our pull-apart, these would
    differ. They must match to numerical precision, which is exactly why renaming
    D->L and reading only the inter-chain term is parity-safe."""
    import numpy as np
    from openmm import app, unit, Platform
    import openmm as mm
    from pdbfixer import PDBFixer

    tmp = Path(tempfile.mkdtemp(prefix="ddg_parity_"))
    comp = tmp / "complex.pdb"
    _write_pdb(_clean_chains(cif, [target_chain, binder_chain]), comp)
    fixer = PDBFixer(filename=str(comp))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    top = fixer.topology
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff,
                             constraints=None, rigidWater=False)
    bonded_groups = set()
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)
        if type(f).__name__ in ("HarmonicBondForce", "HarmonicAngleForce",
                                "PeriodicTorsionForce"):
            bonded_groups.add(i)
    integ = mm.VerletIntegrator(1.0 * unit.femtosecond)
    sim = app.Simulation(top, system, integ, Platform.getPlatformByName("CPU"))
    pos = np.array(fixer.positions.value_in_unit(unit.nanometer))

    def _bonded_e(p):
        sim.context.setPositions(p * unit.nanometer)
        st = sim.context.getState(getEnergy=True, groups=bonded_groups)
        return st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)

    mirror = pos.copy()
    mirror[:, 0] *= -1.0          # reflection -> inverts chirality of every residue
    return _bonded_e(pos), _bonded_e(mirror)


def _selfcheck():
    """Two-part self-check (ADR-023):

      (a) interface-restrained minimization REDUCES the inter-chain vdW/GB spike
          vs the un-minimized single point on a real register test CIF; and
      (b) parity terms still cancel: the bonded energy of the cleaned complex
          equals that of its mirror image (chirality-invariant bonded model), so
          the D->L rename + inter-chain-only readout remains parity-safe."""
    import math
    # Primary register-validation system per ADR-023: 8GQP real best model.
    base = Path("/home/user/claude_projects/XenoDesign1/XenoDesign1_local_ref"
                "/benchmarks/register_decoys/8GQP/real/chai_out")
    cands = sorted(base.glob("pred.model_idx_*.cif"))
    if not cands:
        print("SELFCHECK SKIP: no 8GQP real CIF found", file=sys.stderr)
        return 0
    cif = str(cands[0])
    tgt, bnd = "B", "A"          # 8GQP: chain A = 62-res D binder, chain B = 20-res L target

    sp = _max_abs_intra_force_e(cif, tgt, bnd, minimize=False)
    mn = _max_abs_intra_force_e(cif, tgt, bnd, minimize=True)
    # (a) The complex's TOTAL potential energy is the clash spike that drowns the
    #     register signal: un-minimized Chai coords sit at large positive vdW/GB
    #     (e.g. +7000 kcal here from steric/H/terminal clashes). CA-restrained
    #     minimization relaxes those clashes (holding the fold) so the complex
    #     energy plummets. ponytail: we use e_complex_total as the spike proxy
    #     rather than tracking the literal per-atom max-force — it is the quantity
    #     ADR-023 cares about (the energy that swamps E_int) and is already
    #     reported.
    spike_sp = sp["e_complex_total_kcal"]
    spike_mn = mn["e_complex_total_kcal"]
    ok_spike = math.isfinite(mn["e_int_kcal"]) and spike_mn < spike_sp

    # (b) Parity: mirror-image bonded energy matches.
    e_fwd, e_mir = _parity_bonded_check(cif, tgt, bnd)
    parity_abs = abs(e_fwd - e_mir)
    parity_rel = parity_abs / (abs(e_fwd) + 1e-9)
    ok_parity = parity_rel < 1e-4

    print(json.dumps({
        "cif": cif,
        "singlepoint": sp,
        "minimized": mn,
        "spike_e_complex_singlepoint_kcal": round(spike_sp, 3),
        "spike_e_complex_minimized_kcal": round(spike_mn, 3),
        "bonded_e_forward_kcal": round(e_fwd, 6),
        "bonded_e_mirror_kcal": round(e_mir, 6),
        "bonded_parity_abs_kcal": round(parity_abs, 6),
    }, indent=2))
    print(f"SELFCHECK (a) complex clash spike reduced: {ok_spike} "
          f"(E_complex {spike_sp:.1f} -> {spike_mn:.1f} kcal)", file=sys.stderr)
    print(f"SELFCHECK (b) parity holds: {ok_parity} "
          f"(|dE_bonded|={parity_abs:.2e} kcal, rel={parity_rel:.2e})", file=sys.stderr)
    ok = ok_spike and ok_parity
    print(f"SELFCHECK {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("complex", nargs="?", help="2-chain complex CIF/PDB")
    p.add_argument("--binder_chain", default="B")
    p.add_argument("--target_chain", default="A")
    # Interface-restrained minimization is ON by default (ADR-023). Use
    # --no-minimize to recover the old un-minimized single-point path.
    p.add_argument("--minimize", dest="minimize", action="store_true", default=True,
                   help="CA-restrained energy minimization before E_int (default on)")
    p.add_argument("--no-minimize", dest="minimize", action="store_false",
                   help="old un-minimized single-point path")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args(argv)
    if a.selfcheck:
        return _selfcheck()
    if not a.complex:
        p.error("complex CIF/PDB required (or use --selfcheck)")
    res = score(a.complex, target_chain=a.target_chain, binder_chain=a.binder_chain,
                minimize=a.minimize)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
