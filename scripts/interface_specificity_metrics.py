"""Interface SPECIFICITY metrics for a 2-chain complex — probes interface *correctness*, not just size
(BSA/contacts measure size and don't separate real binders from scrambles; ADR-016/P4b). Per ADR-018 this is
a standalone driver, not an edit to score_complex.py. Chirality-invariant (geometry only; D side-chain atom
names match L).

Metrics:
  - rosetta_coord  : Rosetta-like contact coordination — for each interface residue, count residues (CB, both
                     chains) within 7 A, averaged over interface residues.
  - n_hbond_pairs  : cross-chain N/O .. N/O pairs < 3.5 A (H-bond candidates).
  - n_saltbridges  : cross-chain (Arg/Lys/His +N) .. (Asp/Glu -O) pairs < 4.0 A.
  - pep_helix_frac : helix fraction of chain B (the peptide), handedness-agnostic.
  - binder_epitope : chain-A residues contacted by chain B (<4.5 A heavy-atom).
  - epitope_jaccard: Jaccard of binder_epitope vs a --ref_cif's binder_epitope (does it dock the SAME patch?).

Usage:
  python scripts/interface_specificity_metrics.py --cif <complex.cif> --chain_a A --chain_b B \
      [--ref_cif <real_complex.cif>] --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi
import numpy as np

from xenodesign.secondary_structure import helix_fraction

_D2L = {"DAL": "ALA", "DAR": "ARG", "DSG": "ASN", "DAS": "ASP", "DCY": "CYS", "DGN": "GLN", "DGL": "GLU",
        "DHI": "HIS", "DIL": "ILE", "DLE": "LEU", "DLY": "LYS", "MED": "MET", "DPN": "PHE", "DPR": "PRO",
        "DSN": "SER", "DTH": "THR", "DTR": "TRP", "DTY": "TYR", "DVA": "VAL"}
_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE",
       "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
_POS = {"ARG": {"NH1", "NH2", "NE"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"}}
_NEG = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}


def _chain(cif, c):
    st = gemmi.read_structure(str(cif))
    out = []
    for ch in st[0]:
        if ch.name != c:
            continue
        for ri, r in enumerate(ch):
            nm = _D2L.get(r.name, r.name)
            if nm not in _AA:
                continue
            atoms = [(a.name, a.element.name, np.array([a.pos.x, a.pos.y, a.pos.z]))
                     for a in r if a.element.name != "H"]
            out.append({"name": nm, "ridx": ri, "seqid": r.seqid.num, "atoms": atoms})
        break
    return out


def _ca_cb(res):
    ca = next((p for n, e, p in res["atoms"] if n == "CA"), None)
    cb = next((p for n, e, p in res["atoms"] if n == "CB"), ca)
    return ca, cb


def _binder_epitope(ra, rb, cut=4.5):
    """chain-A residue seqids with any heavy atom within cut of any chain-B heavy atom."""
    xb = np.array([p for r in rb for _, _, p in r["atoms"]])
    epi = set()
    for r in ra:
        xa = np.array([p for _, _, p in r["atoms"]])
        if len(xa) and len(xb) and np.linalg.norm(xa[:, None] - xb[None], axis=2).min() < cut:
            epi.add(r["seqid"])
    return epi


def metrics(cif, ca_id, cb_id, ref_cif=None):
    ra, rb = _chain(cif, ca_id), _chain(cif, cb_id)
    res = {"n_res_a": len(ra), "n_res_b": len(rb)}
    # interface residues (either chain, contacting the other within 4.5 A heavy-atom)
    xa_all = [np.array([p for _, _, p in r["atoms"]]) for r in ra]
    xb_all = [np.array([p for _, _, p in r["atoms"]]) for r in rb]
    iface_a = [i for i, xa in enumerate(xa_all) if len(xa) and min(np.linalg.norm(xa[:, None] - xb[None], axis=2).min() for xb in xb_all if len(xb)) < 4.5]
    iface_b = [j for j, xb in enumerate(xb_all) if len(xb) and min(np.linalg.norm(xb[:, None] - xa[None], axis=2).min() for xa in xa_all if len(xa)) < 4.5]
    # Rosetta-like coordination: CB of all residues (both chains); for each interface residue count CB within 7 A
    allres = ra + rb
    cbs = np.array([(_ca_cb(r)[1] if _ca_cb(r)[1] is not None else _ca_cb(r)[0]) for r in allres])
    iface_global = list(iface_a) + [len(ra) + j for j in iface_b]
    coords = []
    for gi in iface_global:
        d = np.linalg.norm(cbs - cbs[gi], axis=1)
        coords.append(int((d < 7.0).sum()) - 1)  # exclude self
    res["rosetta_coord"] = round(float(np.mean(coords)), 2) if coords else None
    res["n_iface_res"] = len(iface_global)
    # polar specificity: cross-chain N/O.. and salt bridges
    aN = [(r["name"], n, p) for r in ra for n, e, p in r["atoms"] if e in ("N", "O")]
    bN = [(r["name"], n, p) for r in rb for n, e, p in r["atoms"] if e in ("N", "O")]
    hb = sb = 0
    for _, _, pa in aN:
        for _, _, pb in bN:
            if np.linalg.norm(pa - pb) < 3.5:
                hb += 1
    for rn_a, an_a, pa in aN:
        a_pos = an_a in _POS.get(rn_a, ()); a_neg = an_a in _NEG.get(rn_a, ())
        if not (a_pos or a_neg):
            continue
        for rn_b, an_b, pb in bN:
            b_pos = an_b in _POS.get(rn_b, ()); b_neg = an_b in _NEG.get(rn_b, ())
            if ((a_pos and b_neg) or (a_neg and b_pos)) and np.linalg.norm(pa - pb) < 4.0:
                sb += 1
    res["n_hbond_pairs"] = hb
    res["n_saltbridges"] = sb
    # electrostatic complementarity: net (opposite - same) charged cross-chain side-chain pairs < 8 A
    def _charged(rlist):
        out = []
        for r in rlist:
            for n, e, p in r["atoms"]:
                if n in _POS.get(r["name"], ()):
                    out.append((+1, p))
                elif n in _NEG.get(r["name"], ()):
                    out.append((-1, p))
        return out
    qa, qb = _charged(ra), _charged(rb)
    opp = same = 0
    for sa, pa in qa:
        for sbq, pb in qb:
            if np.linalg.norm(pa - pb) < 8.0:
                if sa * sbq < 0:
                    opp += 1
                else:
                    same += 1
    res["elec_compl"] = opp - same  # >0 = net charge-complementary interface
    # interface contiguity: largest connected patch of chain-A interface residues (CB graph, 8 A) / total
    if len(iface_a) > 1:
        cbA = np.array([(_ca_cb(ra[i])[1] if _ca_cb(ra[i])[1] is not None else _ca_cb(ra[i])[0]) for i in iface_a])
        adj = np.linalg.norm(cbA[:, None] - cbA[None], axis=2) < 8.0
        seen, best = set(), 0
        for s in range(len(iface_a)):
            if s in seen:
                continue
            stack, comp = [s], 0
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u); comp += 1
                stack += [v for v in np.where(adj[u])[0] if v not in seen]
            best = max(best, comp)
        res["iface_contiguity"] = round(best / len(iface_a), 3)
    cab = np.array([_ca_cb(r)[0] for r in rb if _ca_cb(r)[0] is not None])
    res["pep_helix_frac"] = round(float(helix_fraction(cab)), 3) if len(cab) >= 4 else None
    epi = _binder_epitope(ra, rb)
    res["binder_epitope"] = sorted(epi)
    if ref_cif:
        rra, rrb = _chain(ref_cif, ca_id), _chain(ref_cif, cb_id)
        ref_epi = _binder_epitope(rra, rrb)
        inter = epi & ref_epi; union = epi | ref_epi
        res["epitope_jaccard_vs_ref"] = round(len(inter) / len(union), 3) if union else None
        res["ref_epitope_size"] = len(ref_epi)
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cif", required=True)
    p.add_argument("--chain_a", default="A")
    p.add_argument("--chain_b", default="B")
    p.add_argument("--ref_cif", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    r = metrics(a.cif, a.chain_a, a.chain_b, a.ref_cif)
    print(json.dumps(r, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
