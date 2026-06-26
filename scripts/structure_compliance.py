"""Goal 1 — structural compliance of the 11 Trues vs their deposits (DockQ + fnat + placement RMSD).

The question: is a 2-3 A binder placement RMSD acceptable, or does the Chai-predicted "real"
complex actually MISS the deposited interface? Eyeballed RMSD can't answer that; DockQ can.

For every system × seed × chai model we:
  1. Pick the COGNATE binder:target pair in the deposit (it has several copies) by interface size
     -- reusing gate_mdm2_restrained.pick_deposit_pair (largest contact count = the real pocket).
  2. Write a clean 2-chain native AND the prediction with binder->A, target->B, and RELABEL every
     D-residue to its canonical L 3-letter name (DLY->LYS, ...). Coordinates are untouched, so
     interface geometry / RMSD are exact; the relabel only lets DockQ (Biopython, drops unknown
     residues) align the chains. Without it the native D-binder collapses to its lone Gly -> fnat=0.
  3. Run DockQ --capri_peptide (peptide interface thresholds) with an explicit AB:AB mapping.
     DockQ's LRMSD == binder placement RMSD (target-superposed), verified to match the gate's
     Kabsch placement RMSD to <0.1 A; we report it as placement_rmsd plus DockQ/fnat/iRMSD.

DockQ bins: Incorrect <0.23, Acceptable 0.23-0.49, Medium 0.49-0.80, High >0.80. fnat>0.5 = the
native interface is well recovered.

CPU-only; gemmi + numpy + the DockQ CLI on PATH.
  python3 scripts/structure_compliance.py                 # all 11, all seeds/models -> JSON
  python3 scripts/structure_compliance.py --systems 3LNJ  # one system
  python3 scripts/structure_compliance.py --self-test     # deposit-vs-deposit -> DockQ 1.0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import gemmi
except ImportError:  # pragma: no cover
    gemmi = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_mdm2_restrained import _aa_residues, _chain, _one, pick_deposit_pair  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "XenoDesign1_local_ref" / "benchmarks"
R2 = BENCH / "gp41_mdm2_r2"
GP41_MDM2_DEPOSITS = BENCH / "gp41_mdm2" / "cifs"

# canonical L 3-letter name per one-letter code (chirality is irrelevant to interface geometry)
L3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU",
    "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}

# Two layouts / chain conventions:
#   gp41 + MDM2 (9): <sys>/{real}/seed{42,43,44}/chai_out, all 5 models; pred binder=A target=B.
#   8GQP / 7YH8v2 (2): <sys>_pred/chai_out (single), all 5 models;  pred binder=B target=A.
GP41 = ("3L35", "2R5B", "2R3C", "3MGN", "1CZQ", "2Q3I")
MDM2 = ("3LNJ", "7KJM", "3IWY")
SEEDS = (42, 43, 44)


FOCUSED_ROOT = BENCH / "focused"
FOCUSED_SYSTEMS = ("3L35", "3MGN", "2R3C", "8GQP")
FOCUSED_SEEDS = (42, 43, 44)


def focused_config(name):
    """Focused (<=3 restraint) re-predictions: uniform focused/<sys>/real/seed*/chai_out layout.
    Chain convention matches the original: gp41 trio binder=A/target=B; 8GQP binder=B/target=A."""
    if name in ("3L35", "3MGN", "2R3C"):
        dep = GP41_MDM2_DEPOSITS / f"{name}.cif"
        pb, pt = "A", "B"
    elif name == "8GQP":
        dep = BENCH / "8GQP.cif"
        pb, pt = "B", "A"
    else:
        raise ValueError(f"{name} not in focused set")
    dirs = [(f"seed{s}", FOCUSED_ROOT / name / "real" / f"seed{s}" / "chai_out") for s in FOCUSED_SEEDS]
    return dep, dirs, pb, pt


def system_config(name, focused=False):
    """Return (deposit_cif, [(label, chai_out_dir)], pred_binder_chain, pred_target_chain)."""
    if focused:
        return focused_config(name)
    if name in GP41 or name in MDM2:
        dep = GP41_MDM2_DEPOSITS / f"{name}.cif"
        dirs = [(f"seed{s}", R2 / name / "real" / f"seed{s}" / "chai_out") for s in SEEDS]
        return dep, dirs, "A", "B"
    if name in ("8GQP", "7YH8v2"):
        stem = "8GQP" if name == "8GQP" else "7YH8"
        dep = BENCH / f"{stem}.cif"
        dirs = [("single", BENCH / f"{stem}_pred" / "chai_out")]
        return dep, dirs, "B", "A"
    raise ValueError(f"unknown system {name}")


ALL_SYSTEMS = GP41 + MDM2 + ("8GQP", "7YH8v2")


def _write_pair_canon(binder_res, target_res, out_path):
    """Write a 2-chain PDB (binder->A, target->B), D-residues relabeled to canonical L names."""
    st = gemmi.Structure()
    m = gemmi.Model("1")
    for res_list, newname in ((binder_res, "A"), (target_res, "B")):
        ch = gemmi.Chain(newname)
        for i, r in enumerate(res_list):
            nr = gemmi.Residue()
            nr.name = L3[_one(r)]
            nr.seqid = gemmi.SeqId(i + 1, " ")
            for a in r:
                nr.add_atom(a)
            ch.add_residue(nr)
        m.add_chain(ch)
    st.add_model(m)
    st.setup_entities()
    st.write_pdb(str(out_path))


def _parse_dockq_short(stdout):
    """Parse the '--short' DockQ line into a dict of floats."""
    line = next((l for l in stdout.splitlines() if l.startswith("DockQ ")), None)
    if line is None:
        return None
    toks = line.split()
    out = {}
    for key in ("DockQ", "iRMSD", "LRMSD", "fnat", "fnonnat", "F1"):
        if key in toks:
            try:
                out[key.lower()] = float(toks[toks.index(key) + 1])
            except (ValueError, IndexError):
                pass
    return out or None


def _run_dockq(pred_pdb, nat_pdb):
    cmd = ["DockQ", str(pred_pdb), str(nat_pdb), "--mapping", "AB:AB",
           "--capri_peptide", "--allowed_mismatches", "2", "--short"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr.strip()[-200:] or "DockQ failed"}
    parsed = _parse_dockq_short(r.stdout)
    return parsed or {"error": "unparseable DockQ output"}


def _bin(dockq):
    if dockq is None:
        return None
    if dockq < 0.23:
        return "Incorrect"
    if dockq < 0.49:
        return "Acceptable"
    if dockq < 0.80:
        return "Medium"
    return "High"


def score_system(name, tmp, focused=False):
    dep_cif, dirs, pb, pt = system_config(name, focused=focused)
    dep_model = gemmi.read_structure(str(dep_cif))[0]

    # need a pred target seq to choose the cognate deposit target copy
    first_cif = None
    for _, d in dirs:
        cifs = sorted(d.glob("pred.model_idx_*.cif"))
        if cifs:
            first_cif = cifs[0]
            break
    if first_cif is None:
        return {"system": name, "status": "MISSING", "per_pred": []}

    pm0 = gemmi.read_structure(str(first_cif))[0]
    pred_target_seq = "".join(_one(r) for r in _aa_residues(_chain(pm0, pt)))
    dep_t, dep_b = pick_deposit_pair(dep_model, pred_target_seq)
    dep_binder_res = _aa_residues(dep_b)
    dep_target_res = _aa_residues(dep_t)
    nat_pdb = tmp / f"{name}_native.pdb"
    _write_pair_canon(dep_binder_res, dep_target_res, nat_pdb)

    rows = []
    for label, d in dirs:
        for cif in sorted(d.glob("pred.model_idx_*.cif")):
            midx = cif.name.split("model_idx_")[1].split(".")[0]
            pm = gemmi.read_structure(str(cif))[0]
            pred_pdb = tmp / f"{name}_{label}_m{midx}.pdb"
            _write_pair_canon(_aa_residues(_chain(pm, pb)), _aa_residues(_chain(pm, pt)), pred_pdb)
            res = _run_dockq(pred_pdb, nat_pdb)
            res.update({"label": label, "model": int(midx)})
            res["bin"] = _bin(res.get("dockq"))
            rows.append(res)

    return {
        "system": name,
        "status": "ok" if rows else "MISSING",
        "deposit_target_chain": dep_t.name,
        "deposit_binder_chain": dep_b.name,
        "n_pred": len(rows),
        "per_pred": rows,
        "summary": _summary(rows),
    }


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    s = sorted(vals)
    return {
        "n": n, "mean": round(mean, 3), "std": round(var ** 0.5, 3),
        "min": round(s[0], 3), "median": round(s[n // 2], 3), "max": round(s[-1], 3),
    }


def _summary(rows):
    ok = [r for r in rows if "dockq" in r]
    if not ok:
        return None
    bins = {}
    for r in ok:
        bins[r["bin"]] = bins.get(r["bin"], 0) + 1
    best = max(ok, key=lambda r: r["dockq"])
    return {
        "dockq": _stats([r.get("dockq") for r in ok]),
        "fnat": _stats([r.get("fnat") for r in ok]),
        "placement_rmsd_LRMSD": _stats([r.get("lrmsd") for r in ok]),
        "irmsd": _stats([r.get("irmsd") for r in ok]),
        "bins": bins,
        "best_dockq": round(best["dockq"], 3),
        "best_bin": best["bin"],
        "recapitulates_any_medium+": any(r["dockq"] >= 0.49 for r in ok),
    }


def self_test(systems):
    """deposit-vs-deposit per system: DockQ ~1.0, fnat ~1.0, RMSD ~0."""
    assert gemmi is not None, "gemmi required"
    allok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name in systems:
            dep_cif, dirs, pb, pt = system_config(name)
            dep_model = gemmi.read_structure(str(dep_cif))[0]
            first_cif = next((c for _, d in dirs for c in sorted(d.glob("pred.model_idx_*.cif"))), None)
            if first_cif is None:
                print(f"[self-test] {name}: MISSING preds, skip")
                continue
            pm0 = gemmi.read_structure(str(first_cif))[0]
            pts = "".join(_one(r) for r in _aa_residues(_chain(pm0, pt)))
            dep_t, dep_b = pick_deposit_pair(dep_model, pts)
            a = tmp / f"{name}_a.pdb"
            b = tmp / f"{name}_b.pdb"
            _write_pair_canon(_aa_residues(dep_b), _aa_residues(dep_t), a)
            _write_pair_canon(_aa_residues(dep_b), _aa_residues(dep_t), b)
            res = _run_dockq(a, b)
            ok = res.get("dockq", 0) > 0.99 and res.get("fnat", 0) > 0.99
            allok = allok and ok
            print(f"[self-test] {name}: DockQ={res.get('dockq')} fnat={res.get('fnat')} "
                  f"LRMSD={res.get('lrmsd')} {'PASS' if ok else 'FAIL'}")
    print("[self-test]", "PASS" if allok else "FAIL")
    return 0 if allok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=None)
    ap.add_argument("--focused", action="store_true",
                    help="score the focused (<=3 restraint) re-predictions under benchmarks/focused/")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if gemmi is None:
        print("ERROR: gemmi not importable.")
        return 2
    systems = args.systems or (list(FOCUSED_SYSTEMS) if args.focused else list(ALL_SYSTEMS))
    out_path = args.out or str(REPO / "docs" / "results" /
                               ("2026-06-23-structure-compliance-focused.json" if args.focused
                                else "2026-06-23-structure-compliance.json"))
    if args.self_test:
        return self_test(systems)

    out = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name in systems:
            print(f"[compliance] {name} ...", file=sys.stderr)
            out[name] = score_system(name, tmp, focused=args.focused)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))

    print(f"\n{'system':8} {'n':>3} {'best':>6} {'bin':>10} {'DockQ mean±std':>16} "
          f"{'fnat med':>9} {'LRMSD med':>10}")
    for name in systems:
        s = out[name].get("summary")
        if not s:
            print(f"{name:8} (no predictions)")
            continue
        dq, fn, lr = s["dockq"], s["fnat"], s["placement_rmsd_LRMSD"]
        print(f"{name:8} {s['dockq']['n']:>3} {s['best_dockq']:>6.3f} {s['best_bin']:>10} "
              f"{dq['mean']:>7.3f}±{dq['std']:<5.3f}  {fn['median']:>7.3f}  {lr['median']:>8.2f}")
    print(f"\nJSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
