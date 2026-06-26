"""Mixed-objective metric panel for a 2-chain complex (P4).

Structural (chirality-invariant; D-residues renamed to L for radii — identical atoms/geometry):
  - bsa_A2          : buried surface area = SASA(A)+SASA(B)-SASA(AB)  [freesasa, via gemmi-written PDBs]
  - n_residue_contacts / n_atom_contacts : cross-chain heavy-atom pairs < 4.5 A
  - iface_closest_A : mean nearest cross-chain heavy-atom distance over interface atoms
                      (lower = tighter packing; a geometry-complementarity proxy, NOT Lawrence Sc)
Confidence (only with --chai_dir; L-trained / parity-biased — report, don't over-trust):
  - iptm  : max off-diagonal per_chain_pair_iptm
  - ipae  : mean interface PAE over the A-B / B-A blocks
  - ipsae : xenodesign.metrics.ipsae

Usage:
  python scripts/score_complex.py --cif <complex.cif> --chain_a A --chain_b B [--chai_dir <dir> --na N] --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi
import numpy as np

_D2L = {"DAL": "ALA", "DAR": "ARG", "DSG": "ASN", "DAS": "ASP", "DCY": "CYS", "DGN": "GLN",
        "DGL": "GLU", "DHI": "HIS", "DIL": "ILE", "DLE": "LEU", "DLY": "LYS", "MED": "MET",
        "DPN": "PHE", "DPR": "PRO", "DSN": "SER", "DTH": "THR", "DTR": "TRP", "DTY": "TYR",
        "DVA": "VAL"}
_3to1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
         "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
         "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def _clean(cif, chains, keep_h=False):
    """gemmi Structure with the requested chains, D->L renamed, AA-only, no water/ligand.
    Hydrogens are dropped unless keep_h=True (the H-bond angle filter needs them retained)."""
    st = gemmi.read_structure(str(cif))
    ns = gemmi.Structure(); ns.add_model(gemmi.Model("1"))
    for cn in chains:
        nc = gemmi.Chain(cn)
        for ch in st[0]:
            if ch.name != cn:
                continue
            for r in ch:
                nm = _D2L.get(r.name, r.name)
                if _3to1.get(nm) is None:
                    continue
                nr = gemmi.Residue(); nr.name = nm; nr.seqid = r.seqid
                for a in r:
                    if a.element.name == "H" and not keep_h:
                        continue
                    nr.add_atom(a)
                nc.add_residue(nr)
            break
        ns[0].add_chain(nc)
    return ns


def _sasa(struct):
    """Total SASA (A^2) of a gemmi Structure via freesasa.

    freesasa is computed IN-PROCESS (import freesasa; calc on a temp PDB). Inside the chai
    container `sys.executable` is /opt/venv/bin/python, which lacks freesasa, so the old
    subprocess-via-sys.executable path raised ModuleNotFoundError and zeroed out bsa (the
    highest-weighted objective term). In-process import works wherever freesasa is importable
    (host python3 and any env running this scorer). The subprocess is kept only as a fallback
    for the rare case where freesasa cannot be imported in-process but a freesasa-capable
    interpreter is on PATH.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
        path = f.name
    struct.write_pdb(path)
    try:
        try:
            import freesasa
            freesasa.setVerbosity(freesasa.silent)
            return float(freesasa.calc(freesasa.Structure(path)).totalArea())
        except ImportError:
            # fallback: shell out to whatever interpreter is on PATH that has freesasa
            code = ("import freesasa,sys;freesasa.setVerbosity(freesasa.silent);"
                    "print(freesasa.calc(freesasa.Structure(sys.argv[1])).totalArea())")
            r = subprocess.run([sys.executable, "-c", code, path],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("freesasa: " + r.stderr[-200:])
            return float(r.stdout.strip().splitlines()[-1])
    finally:
        Path(path).unlink(missing_ok=True)


def _atoms(struct, chain):
    xyz, owner = [], []
    for ri, r in enumerate(struct[0][chain]):
        for a in r:
            xyz.append([a.pos.x, a.pos.y, a.pos.z]); owner.append(ri)
    return np.asarray(xyz, float), np.asarray(owner, int)


# --------------------------------------------------------------------------- H-bonds
# Side-chain polar atoms by residue + atom name, classed donor (D) / acceptor (A) / both (B).
# Backbone N (amide) is always a donor; backbone O (carbonyl) is always an acceptor.
# This is a standard heavy-atom heuristic (no protonation state inference): O atoms that can both
# donate and accept (OH/carboxylate) are tagged "B"; amide/guanidinium/amine N are donors; His ring
# N and most carbonyl/carboxyl/ether O are acceptors. Reference-free and register-sensitive: a
# correct register lines up specific donor->acceptor pairs that a register shift breaks.
_SC_POLAR = {
    "SER": {"OG": "B"}, "THR": {"OG1": "B"}, "TYR": {"OH": "B"},
    "ASN": {"OD1": "A", "ND2": "D"}, "GLN": {"OE1": "A", "NE2": "D"},
    "ASP": {"OD1": "A", "OD2": "A"}, "GLU": {"OE1": "A", "OE2": "A"},
    "LYS": {"NZ": "D"}, "ARG": {"NE": "D", "NH1": "D", "NH2": "D"},
    "HIS": {"ND1": "B", "NE2": "B"}, "TRP": {"NE1": "D"},
    "CYS": {"SG": "B"},  # thiol: weak donor/acceptor (kept; element S still allowed below)
}
# Elements eligible to participate in an H-bond as heavy donor/acceptor atoms.
_HBOND_ELEMENTS = {"N", "O", "S"}


def _polar_atoms(struct, chain):
    """Polar heavy atoms (potential H-bond donors/acceptors) for one chain.

    Returns a list of dicts: {pos:(x,y,z), elem, role, res_idx, res_name, atom_name}.
    role in {'D','A','B'} (donor / acceptor / both). Backbone N=donor, backbone O=acceptor;
    side-chain polar atoms from _SC_POLAR. Atoms not in the polar set are skipped.
    """
    out = []
    for ri, r in enumerate(struct[0][chain]):
        sc = _SC_POLAR.get(r.name, {})
        for a in r:
            el = a.element.name
            if el not in _HBOND_ELEMENTS:
                continue
            nm = a.name
            if nm == "N":            # backbone amide nitrogen
                role = "D"
            elif nm == "O" or nm == "OXT":   # backbone carbonyl / C-terminal carboxyl O
                role = "A"
            elif nm in sc:
                role = sc[nm]
            elif el == "O":          # any other O (e.g. unmodeled) -> acceptor-capable
                role = "A"
            else:
                continue
            out.append({"pos": (a.pos.x, a.pos.y, a.pos.z), "elem": el, "role": role,
                        "res_idx": ri, "res_name": r.name, "atom_name": nm})
    return out


def _has_explicit_h(struct):
    return any(a.element.name == "H" for ch in struct[0] for r in ch for a in r)


def interchain_hbonds(cif, ca, cb, dist=3.5, angle_min=90.0, contact=4.5):
    """Reference-free cross-chain hydrogen-bond count for a 2-chain complex.

    A donor heavy atom (N/O/S) on one chain paired with an acceptor heavy atom (O/N/S) on the
    other chain within `dist` A (heavy-atom distance) counts as one H-bond. 'B' (both) atoms can
    act as donor or acceptor. If explicit H atoms are present, a loose donor-H..acceptor angle
    filter (>= angle_min degrees) is applied; otherwise the count is distance-only.

    Register-SENSITIVE: a correct register forms specific donor->acceptor pairs; a shift breaks
    them. Reference-free: no ground-truth structure is used.

    Returns: {n_interchain_hbonds, hbond_density, hbond_angle_filtered (bool)}.
    hbond_density = n_interchain_hbonds / (interface residues on both chains), where an interface
    residue has any heavy atom within `contact` A of the other chain.
    """
    st = _clean(cif, [ca, cb], keep_h=True)   # retain H so the angle filter can engage if modeled
    return _hbonds_from_struct(st, ca, cb, dist, angle_min, contact)


def _hbonds_from_struct(st, ca, cb, dist=3.5, angle_min=90.0, contact=4.5):
    pa = _polar_atoms(st, ca); pb = _polar_atoms(st, cb)
    res = {"n_interchain_hbonds": 0, "hbond_density": None, "hbond_angle_filtered": False}
    if not pa or not pb:
        # still report interface-residue denominator as 0 -> density None
        return res

    use_angle = _has_explicit_h(st)
    res["hbond_angle_filtered"] = bool(use_angle)
    h_by_res = _h_positions_by_res(st, [ca, cb]) if use_angle else {}

    Xa = np.array([p["pos"] for p in pa], float)
    Xb = np.array([p["pos"] for p in pb], float)
    d = np.linalg.norm(Xa[:, None, :] - Xb[None, :, :], axis=2)
    ii, jj = np.where(d < dist)
    n = 0
    seen = set()  # dedupe: count one H-bond per (donor-atom, acceptor-atom) ORDERED pair max once
    for i, j in zip(ii.tolist(), jj.tolist()):
        A, B = pa[i], pb[j]
        # determine if a valid donor->acceptor assignment exists in either direction
        ok = False
        for don, acc, dchain in ((A, B, ca), (B, A, cb)):
            if don["role"] in ("D", "B") and acc["role"] in ("A", "B"):
                key = (dchain, don["res_idx"], don["atom_name"], acc["res_idx"], acc["atom_name"])
                if key in seen:
                    continue
                if use_angle and not _angle_ok(don, acc, h_by_res, dchain, angle_min):
                    continue
                seen.add(key)
                ok = True
        if ok:
            n += 1
    res["n_interchain_hbonds"] = int(n)

    # interface-residue denominator (HEAVY-atom contact within `contact` A; ignore any retained H)
    xa, oa = _heavy_atoms(st, ca); xb, ob = _heavy_atoms(st, cb)
    if len(xa) == 0 or len(xb) == 0:
        return res
    dd = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=2)
    n_if_a = len({int(oa[i]) for i in np.where(dd.min(1) < contact)[0]})
    n_if_b = len({int(ob[j]) for j in np.where(dd.min(0) < contact)[0]})
    denom = n_if_a + n_if_b
    res["hbond_density"] = round(n / denom, 3) if denom else None
    return res


def _heavy_atoms(struct, chain):
    """Like _atoms but heavy atoms only (drops H, which may be retained for the angle filter)."""
    xyz, owner = [], []
    for ri, r in enumerate(struct[0][chain]):
        for a in r:
            if a.element.name == "H":
                continue
            xyz.append([a.pos.x, a.pos.y, a.pos.z]); owner.append(ri)
    return np.asarray(xyz, float), np.asarray(owner, int)


def _h_positions_by_res(st, chains):
    """{(chain, res_idx): [H positions]} for explicit-H angle filtering."""
    out = {}
    for cn in chains:
        for ri, r in enumerate(st[0][cn]):
            hs = [(a.pos.x, a.pos.y, a.pos.z) for a in r if a.element.name == "H"]
            if hs:
                out[(cn, ri)] = hs
    return out


def _angle_ok(don, acc, h_by_res, dchain, angle_min):
    """Loose D-H..A angle filter: accept if ANY donor-bound H gives angle >= angle_min (deg).
    If no H is modeled on the donor residue, fall back to accept (distance-only for that pair)."""
    hs = h_by_res.get((dchain, don["res_idx"]))
    if not hs:
        return True
    D = np.array(don["pos"]); A = np.array(acc["pos"])
    for h in hs:
        H = np.array(h)
        if np.linalg.norm(H - D) > 1.3:   # only H covalently bound to this donor (~1.0 A)
            continue
        v1 = D - H; v2 = A - H
        nv = np.linalg.norm(v1) * np.linalg.norm(v2)
        if nv < 1e-6:
            continue
        ang = np.degrees(np.arccos(np.clip(float(np.dot(v1, v2)) / nv, -1.0, 1.0)))
        if ang >= angle_min:
            return True
    return False


def _outward_normals(X, r=8.0):
    """Per-atom outward surface normal = atom - centroid(same-chain neighbors within r), normalized.
    For interface (surface) atoms this points away from the local mass = the local surface normal."""
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    out = np.zeros_like(X)
    for i in range(len(X)):
        nb = X[d2[i] < r * r]
        v = X[i] - nb.mean(0)
        n = np.linalg.norm(v)
        out[i] = v / n if n > 1e-6 else 0.0
    return out


def shape_complementarity(xa, xb, d, contact=5.0):
    """Convex/concave complementarity: mean opposition (-n_a . n_b) of outward surface normals over
    cross-chain atom pairs within `contact` A. +1 = perfectly complementary (convex into concave); 0 = flat;
    <0 = clashing/parallel. A van-der-Waals-surface analogue of Lawrence-Colman Sc (which uses dot surfaces)."""
    na = _outward_normals(xa); nb = _outward_normals(xb)
    ii, jj = np.where(d < contact)
    if len(ii) == 0:
        return None
    opp = -(na[ii] * nb[jj]).sum(1)          # -n_a . n_b  per contacting pair
    w = np.exp(-(d[ii, jj] ** 2) / (2 * 2.0 ** 2))  # gentle distance weight (sigma 2 A)
    return round(float((opp * w).sum() / w.sum()), 3)


def structural(cif, ca, cb, contact=4.5):
    sa = _clean(cif, [ca]); sb = _clean(cif, [cb]); sab = _clean(cif, [ca, cb])
    seqa = "".join(_3to1[r.name] for r in sa[0][ca])
    seqb = "".join(_3to1[r.name] for r in sb[0][cb])
    res = {"chain_a": ca, "chain_b": cb, "n_res_a": len(seqa), "n_res_b": len(seqb),
           "seq_a": seqa, "seq_b": seqb}
    try:
        res["bsa_A2"] = round(_sasa(sa) + _sasa(sb) - _sasa(sab), 1)
    except Exception as e:
        res["bsa_A2"] = None; res["bsa_error"] = str(e)[:120]
    xa, oa = _atoms(sab, ca); xb, ob = _atoms(sab, cb)
    d = np.linalg.norm(xa[:, None] - xb[None], axis=2)
    hit = d < contact
    res["n_atom_contacts"] = int(hit.sum())
    pairs = {(int(oa[i]), int(ob[j])) for i, j in zip(*np.where(hit))}
    res["n_residue_contacts"] = len(pairs)
    res["n_iface_res_a"] = len({i for i, _ in pairs}); res["n_iface_res_b"] = len({j for _, j in pairs})
    ia = np.where(d.min(1) < contact)[0]; ib = np.where(d.min(0) < contact)[0]
    if len(ia):
        nn = np.concatenate([d[ia].min(1), d[:, ib].min(0)])
        res["iface_closest_mean_A"] = round(float(nn.mean()), 2)
    res["contact_density"] = round(res["n_atom_contacts"] / max(1, res["n_iface_res_a"] + res["n_iface_res_b"]), 1)
    res["sc_normal_opp"] = shape_complementarity(xa, xb, d)
    # reference-free, register-sensitive inter-chain H-bonds (reuse the already-cleaned sab)
    res.update(_hbonds_from_struct(sab, ca, cb, contact=contact))
    return res


def reexam(cif, ca, cb):
    """Quick re-exam helper: report sc_normal_opp (existing Sc) plus the H-bond panel for a complex,
    so a 'real vs shift' comparison of Sc and H-bonds can be done directly. Returns a small dict."""
    sab = _clean(cif, [ca, cb])
    xa, _ = _atoms(sab, ca); xb, _ = _atoms(sab, cb)
    d = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=2)
    out = {"chain_a": ca, "chain_b": cb, "sc_normal_opp": shape_complementarity(xa, xb, d)}
    out.update(_hbonds_from_struct(sab, ca, cb))
    return out


def confidence(chai_dir, na):
    out = {}; cd = Path(chai_dir)
    sf = sorted(cd.glob("scores.model_idx_*.npz"))
    if not sf:
        return out
    bi, bv = sf[0], -np.inf
    for f in sf:
        a = float(np.load(f)["aggregate_score"].reshape(-1)[0])
        if a > bv: bv, bi = a, f
    z = np.load(bi)
    if "per_chain_pair_iptm" in z:
        m = np.asarray(z["per_chain_pair_iptm"]); m = m.reshape(int(m.size ** 0.5), -1)
        out["iptm"] = round(float(max(m[0, 1], m[1, 0])), 3)
    idx = int(re.search(r"idx_(\d+)", bi.name).group(1))
    cf = cd / f"confidence.model_idx_{idx}.npz"
    if cf.exists():
        zc = np.load(cf)
        if "pae" in zc:
            pae = np.asarray(zc["pae"]); pae = pae[0] if pae.ndim == 3 else pae
            nt = pae.shape[-1]; A = list(range(na)); B = list(range(na, nt))
            inter = np.concatenate([pae[np.ix_(A, B)].reshape(-1), pae[np.ix_(B, A)].reshape(-1)])
            out["ipae"] = round(float(inter.mean()), 2)
            try:
                from xenodesign.metrics import ipsae
                asym = np.array([0] * na + [1] * (nt - na))
                out["ipsae"] = round(float(ipsae(pae, asym, 0, 1)), 3)
            except Exception as e:
                out["ipsae_error"] = str(e)[:100]
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cif", required=False)
    p.add_argument("--chain_a", default="A")
    p.add_argument("--chain_b", default="B")
    p.add_argument("--chai_dir", default=None)
    p.add_argument("--na", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--reexam", action="store_true",
                   help="only report sc_normal_opp + H-bond panel (fast Sc real-vs-shift compare)")
    p.add_argument("--selfcheck", action="store_true",
                   help="run the synthetic inter-chain H-bond TDD self-check and exit")
    a = p.parse_args(argv)
    if a.selfcheck:
        return _hbond_selfcheck()
    if not a.cif:
        p.error("--cif is required (or use --selfcheck)")
    if a.reexam:
        res = reexam(a.cif, a.chain_a, a.chain_b)
        print(json.dumps(res, indent=2))
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=2))
        return res
    res = structural(a.cif, a.chain_a, a.chain_b)
    if a.chai_dir:
        res.update(confidence(a.chai_dir, a.na or res["n_res_a"]))
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return res


# --------------------------------------------------------------------------- selfcheck
def _hbond_selfcheck():
    """Tiny 2-chain synthetic with a KNOWN cross-chain H-bond pair.

    Chain A: one SER (backbone N donor + OG donor/acceptor). Chain B: one ASP placed so its
    carboxyl O (acceptor) sits 2.7 A from SER OG (donor). That is exactly ONE register-correct
    donor->acceptor pair within 3.5 A. A 'shifted' copy translates chain B by +6 A so no polar
    atoms are within 3.5 A -> ZERO H-bonds. Asserts: real has >=1 H-bond, shift has 0, and real
    H-bond density > shift density (register sensitivity, reference-free).
    """
    def build(dx):
        st = gemmi.Structure(); st.add_model(gemmi.Model("1"))
        ca = gemmi.Chain("A")
        r = gemmi.Residue(); r.name = "SER"; r.seqid = gemmi.SeqId("1")
        for nm, el, xyz in [("N", "N", (0.0, 0.0, 0.0)), ("CA", "C", (1.5, 0.0, 0.0)),
                            ("C", "C", (2.0, 1.4, 0.0)), ("O", "O", (1.2, 2.3, 0.0)),
                            ("CB", "C", (2.0, -1.0, 1.0)), ("OG", "O", (3.0, -1.5, 1.5))]:
            at = gemmi.Atom(); at.name = nm; at.element = gemmi.Element(el)
            at.pos = gemmi.Position(*xyz); r.add_atom(at)
        ca.add_residue(r)
        cb = gemmi.Chain("B")
        r2 = gemmi.Residue(); r2.name = "ASP"; r2.seqid = gemmi.SeqId("1")
        # OD1 placed 2.7 A from SER OG (3.0,-1.5,1.5) when dx=0: put it at (5.6,-1.7,1.6) -> ~2.62 A
        for nm, el, xyz in [("N", "N", (7.0, 0.0, 0.0)), ("CA", "C", (6.5, 0.0, 0.0)),
                            ("C", "C", (6.0, 1.4, 0.0)), ("O", "O", (6.5, 2.3, 0.0)),
                            ("CB", "C", (6.0, -1.0, 1.0)), ("CG", "C", (5.8, -1.4, 1.3)),
                            ("OD1", "O", (5.6, -1.7, 1.6)), ("OD2", "O", (5.5, -1.4, 2.6))]:
            at = gemmi.Atom(); at.name = nm; at.element = gemmi.Element(el)
            at.pos = gemmi.Position(xyz[0] + dx, xyz[1], xyz[2]); r2.add_atom(at)
        cb.add_residue(r2)
        st[0].add_chain(ca); st[0].add_chain(cb)
        f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
        f.close(); st.make_mmcif_document().write_file(f.name)
        return Path(f.name)

    cif_real = build(0.0)
    cif_shift = build(6.0)
    real = interchain_hbonds(cif_real, "A", "B")
    shift = interchain_hbonds(cif_shift, "A", "B")
    rx = reexam(cif_real, "A", "B")
    cif_real.unlink(missing_ok=True); cif_shift.unlink(missing_ok=True)

    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    check("real has >=1 inter-chain H-bond", real["n_interchain_hbonds"] >= 1,
          f"got {real['n_interchain_hbonds']}")
    check("shift has 0 inter-chain H-bonds", shift["n_interchain_hbonds"] == 0,
          f"got {shift['n_interchain_hbonds']}")
    check("real hbond_density defined and > 0", (real["hbond_density"] or 0) > 0,
          f"got {real['hbond_density']}")
    check("real density > shift density",
          (real["hbond_density"] or 0) > (shift["hbond_density"] or 0),
          f"got {real['hbond_density']} vs {shift['hbond_density']}")
    check("angle filter off (no explicit H)", real["hbond_angle_filtered"] is False)
    check("reexam reports sc_normal_opp key", "sc_normal_opp" in rx)
    check("reexam reports n_interchain_hbonds key", rx["n_interchain_hbonds"] >= 1,
          f"got {rx.get('n_interchain_hbonds')}")

    print("\nREAL :", json.dumps(real))
    print("SHIFT:", json.dumps(shift))
    print("REEXAM:", json.dumps(rx))
    print("\nSELFCHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    r = main()
    sys.exit(r if isinstance(r, int) else 0)
