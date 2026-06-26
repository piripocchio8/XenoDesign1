"""Register-SENSITIVE interface features for a 2-chain D-peptide/target complex (CPU-only).

Motivation (ADR-011 / ADR-022): the T20 panel (bsa_A2, n_residue_contacts, contact_density,
iface_closest_mean_A, sc_normal_opp, ipsae, iptm, ipae) is BULK / global. For a broad amphipathic
helix lying along a target groove, sliding the binder register by a few residues barely changes the
buried AREA or the contact COUNT - the same hydrophobic FACE still buries roughly the same amount of
surface - so those features are register-AGNOSTIC. The features here are built to change when the
binder register shifts, by looking at WHICH residues / WHICH positions actually face the target,
not how MUCH surface is buried.

Three families, all from a single 2-chain CIF (binder chain, target chain), reusing score_complex
helpers (gemmi parse + D->L rename + heavy-atom contact detection):

 (f1) interface RESIDUE-IDENTITY burial
      Interface binder residues = binder residues with a heavy atom < `contact` (default 4.5 A) of any
      target heavy atom. Report:
        - buried_residue_list      : (1-based index, 1-letter type) of each interface binder residue
        - buried_residue_profile   : count of each amino-acid TYPE buried
        - buried_hydrophobic_frac  : fraction of interface binder residues whose type is in FILMVWY(AC)
      WHY register-sensitive: a register shift rotates a DIFFERENT set of side chains into the
      interface, so both the multiset of buried types and the hydrophobic fraction change, even when
      the buried area is constant.

 (f2) interface POSITION fingerprint
      iface_position_set = the SET of binder residue INDICES (1-based) at the interface.
      Reference-free:
        - iface_pos_contiguity : longest run of consecutive interface indices / size of the set
                                 (a clean amphipathic face gives one long, periodic stripe; a shift
                                 keeps a stripe but at different indices)
        - iface_pos_period_3p6 : how helical-periodic the interface indices are - fraction of
                                 index gaps that are ~3 or ~4 (the i, i+3/i+4 spacing of one helix face)
      Reference-based (alpha-only, needs --gt_cif = GT/real complex):
        - iface_pos_jaccard_vs_gt : Jaccard overlap of this complex's interface-position set vs the GT's
        - iface_pos_shift_vs_gt   : integer offset (mode of pairwise index differences) that best aligns
                                    this set onto the GT set - 0 means same register as GT.
      WHY register-sensitive: the index SET slides with the register; Jaccard vs GT drops and the
      best-fit offset becomes non-zero. Reference-free contiguity/period stay high (it is still a
      helix face) but the absolute indices move, which the offset captures.

 (f3) anchor / key-residue packing
      Rank binder residues by burial depth (number of target heavy-atom contacts within `contact`).
      Report:
        - anchor_residues          : top-k most-buried binder residues (index, type, n_contacts)
        - anchor_hydrophobic_frac  : fraction of the top-k anchors that are large hydrophobics (FILMVWY)
        - core_vs_exposed_logodds  : log2( mean burial of FILMVWY residues / mean burial of polar/charged
                                     DEKRHNQST ) - >0 means big hydrophobics sit in the core (correct
                                     packing), <=0 means they are mis-packed / exposed (a wrong register).
      WHY register-sensitive: a correct register drives the large hydrophobics into the most-buried
      anchor slots; a shift puts polar/charged residues there instead, dropping the hydrophobic anchor
      fraction and the core-vs-exposed log-odds.

CLI:
  python scripts/register_features.py <cif> --binder_chain A --target_chain B [--gt_cif <GT real cif>] [--out json]
  python scripts/register_features.py --selfcheck

Note on chain order: in this dataset the D-BINDER is chain A (21 res) and the L-TARGET is chain B
(41 res) - the OPPOSITE of score_complex's A/B convention - so the defaults here are binder=A,
target=B. Pass --binder_chain / --target_chain explicitly for other layouts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Reuse the gemmi parse / D->L rename / contact helpers from score_complex.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_complex import _atoms, _clean, _3to1  # noqa: E402

# Large/aromatic hydrophobics that should pack the interface CORE of a correct register.
_BIG_HYDROPHOBIC = set("FILMVWY")
# Broader hydrophobic set (incl. Ala/Cys) for the buried-hydrophobic FRACTION (f1).
_HYDROPHOBIC = set("FILMVWYAC")
# Polar / charged that should stay solvent-exposed in a correct register (f3 denominator).
_POLAR_CHARGED = set("DEKRHNQST")


def _best_cif(path: Path) -> Path:
    """A CIF path, or the best (highest aggregate_score) pred.model_idx_*.cif under a chai_out dir."""
    path = Path(path)
    if path.is_file():
        return path
    sf = sorted(path.glob("scores.model_idx_*.npz"))
    if sf:
        bi, bv = 0, -np.inf
        for f in sf:
            agg = float(np.asarray(np.load(f)["aggregate_score"]).reshape(-1)[0])
            if agg > bv:
                bv, bi = agg, int(re.search(r"idx_(\d+)", f.name).group(1))
        c = path / f"pred.model_idx_{bi}.cif"
        if c.exists():
            return c
    cifs = sorted(path.glob("*.cif"))
    if cifs:
        return cifs[0]
    raise FileNotFoundError(f"no CIF under {path}")


def _contact_matrix(cif: Path, binder: str, target: str, contact: float):
    """Per binder-residue target-contact counts + binder 1-letter sequence.

    Returns (seq, per_res_contacts) where per_res_contacts[i] = number of target heavy atoms
    within `contact` A of any heavy atom of binder residue i (a burial-depth proxy).
    """
    sab = _clean(cif, [binder, target])
    seq = "".join(_3to1[r.name] for r in sab[0][binder])
    xb, ob = _atoms(sab, binder)   # binder atoms + owner residue index
    xt, _ = _atoms(sab, target)
    nres = len(seq)
    if len(xb) == 0 or len(xt) == 0:
        return seq, np.zeros(nres, int)
    # heavy-atom distance matrix (binder atoms x target atoms); count contacts per binder residue
    d = np.linalg.norm(xb[:, None, :] - xt[None, :, :], axis=2)
    atom_contacts = (d < contact).sum(1)            # per binder ATOM
    per_res = np.zeros(nres, int)
    np.add.at(per_res, ob, atom_contacts)           # accumulate onto owner residue
    return seq, per_res


def _iface_positions(per_res: np.ndarray):
    """1-based indices of binder residues that make >=1 target contact."""
    return [int(i + 1) for i in np.where(per_res > 0)[0]]


def f1_identity(seq: str, per_res: np.ndarray) -> dict:
    idx = _iface_positions(per_res)
    types = [seq[i - 1] for i in idx]
    n = len(types)
    hyd = sum(1 for t in types if t in _HYDROPHOBIC)
    return {
        "n_iface_binder_res": n,
        "buried_residue_list": [[i, seq[i - 1]] for i in idx],
        "buried_residue_profile": dict(sorted(Counter(types).items())),
        "buried_hydrophobic_frac": round(hyd / n, 3) if n else None,
    }


def _longest_run(sorted_idx) -> int:
    if not sorted_idx:
        return 0
    best = run = 1
    for a, b in zip(sorted_idx, sorted_idx[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


def _best_offset(a_set, b_set, span=12):
    """Integer offset k maximizing |{i in a : i+k in b}|; ties -> smallest |k|. (a onto b)"""
    if not a_set or not b_set:
        return None
    best_k, best_n = 0, -1
    for k in range(-span, span + 1):
        n = sum(1 for i in a_set if (i + k) in b_set)
        if n > best_n or (n == best_n and abs(k) < abs(best_k)):
            best_n, best_k = n, k
    return best_k


def f2_position(per_res: np.ndarray, gt_per_res: np.ndarray | None) -> dict:
    idx = sorted(_iface_positions(per_res))
    n = len(idx)
    longest = _longest_run(idx)
    # helical-periodicity: fraction of consecutive-index gaps that are ~3 or ~4 (one helix face)
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    period = round(sum(1 for g in gaps if g in (3, 4)) / len(gaps), 3) if gaps else None
    out = {
        "iface_position_set": idx,
        "iface_pos_contiguity": round(longest / n, 3) if n else None,
        "iface_pos_period_3p4": period,
    }
    if gt_per_res is not None:
        gt_idx = set(_iface_positions(gt_per_res))
        cur = set(idx)
        union = cur | gt_idx
        out["iface_pos_jaccard_vs_gt"] = round(len(cur & gt_idx) / len(union), 3) if union else None
        out["iface_pos_shift_vs_gt"] = _best_offset(idx, gt_idx)
    return out


def f3_anchor(seq: str, per_res: np.ndarray, topk: int = 3) -> dict:
    order = np.argsort(-per_res)                     # most-buried first
    anchors = [i for i in order if per_res[i] > 0][:topk]
    anchor_list = [[int(i + 1), seq[i], int(per_res[i])] for i in anchors]
    a_hyd = sum(1 for i in anchors if seq[i] in _BIG_HYDROPHOBIC)
    # core-vs-exposed log-odds: mean burial of big hydrophobics vs polar/charged (over ALL binder res)
    big = [per_res[i] for i in range(len(seq)) if seq[i] in _BIG_HYDROPHOBIC]
    pol = [per_res[i] for i in range(len(seq)) if seq[i] in _POLAR_CHARGED]
    logodds = None
    if big and pol:
        mb, mp = float(np.mean(big)), float(np.mean(pol))
        logodds = round(float(np.log2((mb + 1.0) / (mp + 1.0))), 3)  # +1 smoothing
    return {
        "anchor_residues": anchor_list,
        "anchor_hydrophobic_frac": round(a_hyd / len(anchors), 3) if anchors else None,
        "core_vs_exposed_logodds": logodds,
    }


def compute(cif: Path, binder: str, target: str, gt_cif: Path | None = None,
            contact: float = 4.5, topk: int = 3) -> dict:
    cif = _best_cif(Path(cif))
    seq, per_res = _contact_matrix(cif, binder, target, contact)
    gt_per_res = None
    if gt_cif is not None:
        gt_cif = _best_cif(Path(gt_cif))
        _, gt_per_res = _contact_matrix(gt_cif, binder, target, contact)
    res = {
        "cif": str(cif),
        "binder_chain": binder, "target_chain": target,
        "contact_A": contact, "binder_seq": seq, "n_binder_res": len(seq),
    }
    res.update(f1_identity(seq, per_res))
    res.update(f2_position(per_res, gt_per_res))
    res.update(f3_anchor(seq, per_res, topk))
    if gt_cif is not None:
        res["gt_cif"] = str(gt_cif)
    return res


# --------------------------------------------------------------------------- selfcheck
def _selfcheck() -> int:
    """Tiny synthetic 2-chain case: a binder bar whose register we shift by one residue.

    Binder = 7 CA-only residues spaced 4 A apart along x (so neighbours are resolvable). The target is
    a compact atom blob centred on binder residues 3-4 (1-based) at y=3.0, so ONLY positions 3 and 4
    fall within the 4.5 A contact cutoff - the interface POSITION set is fixed at {3,4} for both cases.
    The GT sequence packs large hydrophobics (W,L) into positions 3-4; the 'shift' uses the SAME
    geometry but rotates the SEQUENCE by one, so polar residues (S,T) now occupy 3-4 instead. This
    isolates the identity-based features: with positions held fixed, f1 (buried identities), f3 (anchor
    hydrophobicity / core-vs-exposed log-odds) MUST change while f2's position-set / Jaccard-vs-GT
    stays constant - exactly the register sensitivity we want.
    """
    import gemmi
    import tempfile

    def build(seq):
        st = gemmi.Structure(); st.add_model(gemmi.Model("1"))
        cb = gemmi.Chain("A")
        for i, aa in enumerate(seq):
            r = gemmi.Residue(); r.name = {v: k for k, v in _3to1.items()}[aa]; r.seqid = gemmi.SeqId(str(i + 1))
            a = gemmi.Atom(); a.name = "CA"; a.element = gemmi.Element("C")
            a.pos = gemmi.Position(4.0 * i, 0.0, 0.0); r.add_atom(a)   # 4 A spacing
            cb.add_residue(r)
        ct = gemmi.Chain("B")
        # compact target blob centred between binder residues 3 and 4 (x = 8 and 12 -> centre x=10), y=3
        r = gemmi.Residue(); r.name = "ALA"; r.seqid = gemmi.SeqId("1")
        for k, (xx, yy) in enumerate([(9.0, 3.0), (10.0, 3.0), (11.0, 3.0), (10.0, 3.2)]):
            a = gemmi.Atom(); a.name = f"C{k}"; a.element = gemmi.Element("C")
            a.pos = gemmi.Position(float(xx), float(yy), 0.0); r.add_atom(a)
        ct.add_residue(r)
        st[0].add_chain(cb); st[0].add_chain(ct)
        f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
        f.close(); st.make_mmcif_document().write_file(f.name)
        return Path(f.name)

    # GT: big hydrophobics (W,L) at positions 3-4 (the buried slots); polar S/T elsewhere -> correct pack
    gt_seq = "STWLSTS"     # 1-based: S T W L S T S ; positions 3,4 = W,L bury
    cif_gt = build(gt_seq)
    # shift: same residues rotated +1 so positions 3-4 are now polar (S,T); hydrophobics moved out
    sh_seq = "TSTWLST"     # 1-based: T S T W L S T ; positions 3,4 = T,W -> mis-packed
    cif_sh = build(sh_seq)

    rgt = compute(cif_gt, "A", "B", gt_cif=cif_gt)
    rsh = compute(cif_sh, "A", "B", gt_cif=cif_gt)
    cif_gt.unlink(missing_ok=True); cif_sh.unlink(missing_ok=True)

    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    # Interface POSITION set is fixed at {3,4} (geometry identical in both cases)...
    check("gt iface positions == {3,4}", rgt["iface_position_set"] == [3, 4],
          f"got {rgt['iface_position_set']}")
    check("shift iface positions == {3,4}", rsh["iface_position_set"] == [3, 4],
          f"got {rsh['iface_position_set']}")
    check("shift jaccard_vs_gt == 1.0 (same positions face target)",
          rsh["iface_pos_jaccard_vs_gt"] == 1.0, f"got {rsh['iface_pos_jaccard_vs_gt']}")
    check("shift offset_vs_gt == 0 (positions aligned)",
          rsh["iface_pos_shift_vs_gt"] == 0, f"got {rsh['iface_pos_shift_vs_gt']}")
    # ...but the buried IDENTITIES differ: GT buries W,L (both hydrophobic); shift buries T,W
    check("gt buried hydrophobic_frac == 1.0", rgt["buried_hydrophobic_frac"] == 1.0,
          f"got {rgt['buried_hydrophobic_frac']}")
    check("shift buried hydrophobic_frac < gt", rsh["buried_hydrophobic_frac"] < rgt["buried_hydrophobic_frac"],
          f"got {rsh['buried_hydrophobic_frac']} vs {rgt['buried_hydrophobic_frac']}")
    check("gt buried profile == {L:1,W:1}", rgt["buried_residue_profile"] == {"L": 1, "W": 1},
          f"got {rgt['buried_residue_profile']}")
    check("shift buried profile differs from gt", rsh["buried_residue_profile"] != rgt["buried_residue_profile"],
          f"got {rsh['buried_residue_profile']}")
    # anchor hydrophobic frac drops under the wrong register (the buried slots are now polar)
    check("gt anchor_hydrophobic_frac >= shift anchor_hydrophobic_frac",
          rgt["anchor_hydrophobic_frac"] >= rsh["anchor_hydrophobic_frac"],
          f"got {rgt['anchor_hydrophobic_frac']} vs {rsh['anchor_hydrophobic_frac']}")
    # core-vs-exposed log-odds: GT packs big hydrophobics into the core (>0), shift mis-packs (lower)
    check("gt core_vs_exposed_logodds > shift", rgt["core_vs_exposed_logodds"] > rsh["core_vs_exposed_logodds"],
          f"got {rgt['core_vs_exposed_logodds']} vs {rsh['core_vs_exposed_logodds']}")
    check("gt core_vs_exposed_logodds > 0 (correct pack)", rgt["core_vs_exposed_logodds"] > 0,
          f"got {rgt['core_vs_exposed_logodds']}")
    # reference-free position metrics are well-defined
    check("contiguity in (0,1]", 0 < (rgt["iface_pos_contiguity"] or 0) <= 1.0)

    print("\nGT  :", json.dumps({k: rgt[k] for k in
          ("buried_residue_profile", "buried_hydrophobic_frac", "anchor_residues",
           "core_vs_exposed_logodds", "iface_position_set")}))
    print("SHFT:", json.dumps({k: rsh[k] for k in
          ("buried_residue_profile", "buried_hydrophobic_frac", "anchor_residues",
           "core_vs_exposed_logodds", "iface_pos_jaccard_vs_gt", "iface_pos_shift_vs_gt")}))
    print("\nSELFCHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description="Register-sensitive interface features for a 2-chain complex")
    p.add_argument("cif", nargs="?", help="complex CIF or a chai_out dir")
    p.add_argument("--binder_chain", default="A", help="D-binder chain (default A in this dataset)")
    p.add_argument("--target_chain", default="B", help="L-target chain (default B in this dataset)")
    p.add_argument("--gt_cif", default=None, help="GT real complex CIF/dir for reference-based f2 (alpha-only)")
    p.add_argument("--contact", type=float, default=4.5)
    p.add_argument("--topk", type=int, default=3, help="number of anchor residues to report")
    p.add_argument("--out", default=None)
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args(argv)
    if a.selfcheck:
        return _selfcheck()
    if not a.cif:
        p.error("provide a CIF (or chai_out dir), or --selfcheck")
    res = compute(a.cif, a.binder_chain, a.target_chain, a.gt_cif, a.contact, a.topk)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = main()
    sys.exit(r if isinstance(r, int) else 0)
