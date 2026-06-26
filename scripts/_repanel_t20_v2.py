"""Driver: re-panel all available benchmark predictions with the current
score_complex.py (which now emits n_interchain_hbonds / hbond_density / sc_normal_opp
alongside iptm/ipae/ipsae). Writes one JSON per item to panels_t20_v2/.

Run: PYTHONPATH=$PWD python3 scripts/_repanel_t20_v2.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.score_complex import structural, confidence  # noqa: E402

REPO = Path("/home/user/claude_projects/XenoDesign1")
BM = REPO / "XenoDesign1_local_ref/benchmarks"
GATE = REPO / "XenoDesign1_local_ref/9dxx_target_gate"
OUT = BM / "panels_t20_v2"
OUT.mkdir(parents=True, exist_ok=True)


def best_model(chai_dir: Path):
    """Return (best_cif Path, best_idx int) = highest aggregate_score across the 5 scores npz."""
    sf = sorted(chai_dir.glob("scores.model_idx_*.npz"))
    if not sf:
        return None, None
    bi, bv = None, -np.inf
    for f in sf:
        a = float(np.load(f)["aggregate_score"].reshape(-1)[0])
        if a > bv:
            bv, bi = a, f
    idx = int(re.search(r"idx_(\d+)", bi.name).group(1))
    cif = chai_dir / f"pred.model_idx_{idx}.cif"
    return (cif if cif.exists() else None), idx


def confidence_pair(chai_dir: Path, na: int, ia: int, ib: int):
    """Like score_complex.confidence but for an arbitrary chain pair (ia,ib) in the
    per_chain_pair_iptm matrix and PAE blocks. Used for 9DXX (binder=C vs target=A).
    `na` = number of tokens BEFORE the second chain's block in the PAE (here: A+B length)
    -- but we want binder(C) vs target(A). We compute generic blocks from chain lengths.
    For 9DXX we instead pass explicit token ranges via confidence_9dxx().
    """
    return {}


def confidence_9dxx(chai_dir: Path):
    """Binder(C)-vs-target iptm/ipae/ipsae for the 3-chain 9DXX complex.
    Chains in token order: A=HA1, B=HA2, C=DP93 binder. We score the binder (chain index 2)
    against the target as a whole. iptm = max binder-target off-diagonal in per_chain_pair_iptm;
    ipae = mean interface PAE over binder<->target token blocks; ipsae via xenodesign.metrics
    with asym binder=1, target(A+B)=0."""
    out = {}
    sf = sorted(chai_dir.glob("scores.model_idx_*.npz"))
    if not sf:
        return out, None
    bi, bv = None, -np.inf
    for f in sf:
        a = float(np.load(f)["aggregate_score"].reshape(-1)[0])
        if a > bv:
            bv, bi = a, f
    idx = int(re.search(r"idx_(\d+)", bi.name).group(1))
    z = np.load(bi)
    # per_chain_pair_iptm is 3x3 (A,B,C). Binder=index 2. Target = {0,1}.
    if "per_chain_pair_iptm" in z:
        m = np.asarray(z["per_chain_pair_iptm"]).reshape(-1)
        k = int(round(m.size ** 0.5))
        m = m.reshape(k, k)
        if k >= 3:
            vals = [m[2, 0], m[0, 2], m[2, 1], m[1, 2]]
            out["iptm"] = round(float(max(vals)), 3)
        else:
            out["iptm"] = round(float(max(m[0, 1], m[1, 0])), 3)
    # chain token lengths from the best CIF
    cif = chai_dir / f"pred.model_idx_{idx}.cif"
    import gemmi
    st = gemmi.read_structure(str(cif))
    lens = {c.name: len(c) for c in st[0]}
    nA, nB, nC = lens.get("A", 0), lens.get("B", 0), lens.get("C", 0)
    cf = chai_dir / f"confidence.model_idx_{idx}.npz"
    pae = None
    if cf.exists():
        zc = np.load(cf)
        if "pae" in zc:
            pae = np.asarray(zc["pae"])
            pae = pae[0] if pae.ndim == 3 else pae
    if pae is None and "pae" in z:
        pae = np.asarray(z["pae"])
        pae = pae[0] if pae.ndim == 3 else pae
    if pae is not None:
        nt = pae.shape[-1]
        # token ranges (assume order A,B,C contiguous)
        A_idx = list(range(0, nA))
        B_idx = list(range(nA, nA + nB))
        C_idx = list(range(nA + nB, nA + nB + nC))
        target = A_idx + B_idx
        binder = C_idx
        # guard: clamp to nt
        target = [i for i in target if i < nt]
        binder = [i for i in binder if i < nt]
        if target and binder:
            inter = np.concatenate([
                pae[np.ix_(binder, target)].reshape(-1),
                pae[np.ix_(target, binder)].reshape(-1),
            ])
            out["ipae"] = round(float(inter.mean()), 2)
            try:
                from xenodesign.metrics import ipsae
                # asym: target=0, binder=1 across full token axis
                asym = np.zeros(nt, dtype=int)
                for i in binder:
                    asym[i] = 1
                out["ipsae"] = round(float(ipsae(pae, asym, 0, 1)), 3)
            except Exception as e:
                out["ipsae_error"] = str(e)[:100]
    return out, idx


def run_2chain(system, item, kind, chai_dir: Path, na: int, ca="A", cb="B", seed=None):
    cif, idx = best_model(chai_dir)
    rec = {"system": system, "item": item, "kind": kind, "seed": seed,
           "chai_dir": str(chai_dir), "best_model_idx": idx}
    if cif is None:
        rec["error"] = "no best model cif found"
        return rec
    res = structural(str(cif), ca, cb)
    res.update(confidence(str(chai_dir), na))
    rec.update(res)
    rec["best_model"] = cif.name
    panel_path = OUT / f"{system}__{item}.json"
    panel_path.write_text(json.dumps(rec, indent=2))
    rec["panel_path"] = str(panel_path)
    return rec


def run_9dxx(item, kind, chai_dir: Path, seed=None):
    rec = {"system": "9DXX", "item": item, "kind": kind, "seed": seed,
           "chai_dir": str(chai_dir)}
    conf, idx = confidence_9dxx(chai_dir)
    rec["best_model_idx"] = idx
    rec.update(conf)
    cif, _ = best_model(chai_dir)
    if cif is not None:
        rec["best_model"] = cif.name
        # best-effort 2-chain geometry: binder C vs target A (HA1)
        try:
            g = structural(str(cif), "C", "A")
            for k in ("sc_normal_opp", "n_interchain_hbonds", "hbond_density",
                      "n_atom_contacts", "n_residue_contacts", "bsa_A2"):
                if k in g:
                    rec[k] = g[k]
        except Exception as e:
            rec["geom_error"] = str(e)[:120]
    panel_path = OUT / f"9DXX__{item}.json"
    panel_path.write_text(json.dumps(rec, indent=2))
    rec["panel_path"] = str(panel_path)
    return rec


RESULTS = []
AVAIL = []


def note(d: Path, label):
    ok = d.exists() and any(d.glob("scores.model_idx_*.npz"))
    AVAIL.append(f"{label}: {'PRESENT' if ok else 'MISSING'} ({d})")
    return ok


# ---- 8GQP (na=62) ----
items_8gqp = [
    ("real",   "real",  BM / "8GQP_pred/chai_out"),
    ("scram1", "scram", BM / "decoys/8GQP_scram1/chai_out"),
    ("scram2", "scram", BM / "decoys/8GQP_scram2/chai_out"),
    ("reg_real",  "real",  BM / "register_decoys/8GQP/real/chai_out"),
    ("reg_shift3","shift", BM / "register_decoys/8GQP/shift3/chai_out"),
    ("reg_shift4","shift", BM / "register_decoys/8GQP/shift4/chai_out"),
    ("reg_shift7","shift", BM / "register_decoys/8GQP/shift7/chai_out"),
]
for item, kind, d in items_8gqp:
    if note(d, f"8GQP/{item}"):
        RESULTS.append(run_2chain("8GQP", item, kind, d, na=62))

# ---- 7YH8 (na=62) ----
items_7yh8 = [
    ("real",   "real",  BM / "7YH8_pred/chai_out"),
    ("scram1", "scram", BM / "decoys/7YH8_scram1/chai_out"),
    ("scram2", "scram", BM / "decoys/7YH8_scram2/chai_out"),
    ("reg_real",  "real",  BM / "register_decoys/7YH8/real/chai_out"),
    ("reg_shift3","shift", BM / "register_decoys/7YH8/shift3/chai_out"),
    ("reg_shift4","shift", BM / "register_decoys/7YH8/shift4/chai_out"),
    ("reg_shift7","shift", BM / "register_decoys/7YH8/shift7/chai_out"),
]
for item, kind, d in items_7yh8:
    if note(d, f"7YH8/{item}"):
        RESULTS.append(run_2chain("7YH8", item, kind, d, na=62))

# ---- 9DXX (3-chain; binder=C, target=A+B) ----
items_9dxx = [
    ("real_seed42", "real",  GATE / "chai_real_v2_seed42", "42"),
    ("real_seed43", "real",  GATE / "chai_real_v2_seed43", "43"),
    ("real_seed44", "real",  GATE / "chai_real_v2_seed44", "44"),
    ("scram1_seed42", "scram", GATE / "chai_scram1_v2_seed42", "42"),
    ("scram2_seed42", "scram", GATE / "chai_scram2_v2_seed42", "42"),
    ("scram3_seed42", "scram", GATE / "chai_scram3_v2_seed42", "42"),
]
for item, kind, d, seed in items_9dxx:
    if note(d, f"9DXX/{item}"):
        RESULTS.append(run_9dxx(item, kind, d, seed=seed))

print("=== AVAILABILITY ===")
for a in AVAIL:
    print(a)
print("\n=== RESULTS (compact) ===")
for r in RESULTS:
    print(json.dumps({k: r.get(k) for k in (
        "system", "item", "kind", "seed", "iptm", "ipae", "ipsae",
        "n_interchain_hbonds", "hbond_density", "sc_normal_opp",
        "best_model", "panel_path", "error", "geom_error", "ipsae_error")
        if r.get(k) is not None}))

# dump a combined summary for the agent to read back
(OUT / "_summary.json").write_text(json.dumps({"availability": AVAIL, "results": RESULTS}, indent=2))
print("\nWrote", OUT / "_summary.json")
