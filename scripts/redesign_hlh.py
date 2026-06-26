"""Redesign the L-HLH INSIDE face + loop with ProteinMPNN, keeping the OUTSIDE (Tyr-29) face
fixed at its real residues (non-designable). Tests the hypothesis: the HLH was carved from the
L-ABLE 4-helix bundle, so its sequence may be sub-optimal for a stand-alone HLH — a better sequence
may make the D-HLH fold properly (it currently folds to one long helix). Outputs the non-designable
residue list (for review) + the top-N sequences by MPNN score (real outside restored).

Face partition (from the predicted L-HLH structure + the GT binder footprint): the OUTSIDE face is
the side whose CB points the same way as the GT-contacted residues (helix-1 6/9/10/13,
helix-2 29/32/35/36); those + their half-cylinder are NON-designable. INSIDE-face residues + the
GDDDS loop (21-25) are designable. Backbone-only (no ligand context) -> proteinmpnn_v_48_020.

Run (GPU): python scripts/redesign_hlh.py --cif <L-HLH cif> --chain A --n_sample 200 --top 20 --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GT_OUT_H1, GT_OUT_H2 = [6, 9, 10, 13], [29, 32, 35, 36]
LOOP = list(range(21, 26))            # GDDDS — always designable
H1, H2 = range(1, 21), range(26, 42)
_AA = "ACDEFGHIKLMNPQRSTVWY"
_INT2STR = {0:"A",1:"C",2:"D",3:"E",4:"F",5:"G",6:"H",7:"I",8:"K",9:"L",10:"M",
            11:"N",12:"P",13:"Q",14:"R",15:"S",16:"T",17:"V",18:"W",19:"Y",20:"X"}
_STR2INT = {v: k for k, v in _INT2STR.items()}
_AA3to1 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
           "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
           "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}


def load_bb(cif, chain):
    import gemmi
    st = gemmi.read_structure(str(cif))
    res = []
    for m in st:
        for ch in m:
            if ch.name != chain:
                continue
            for r in ch:
                at = {n: r.find_atom(n, "*") for n in ("N", "CA", "C", "O")}
                if all(at[n] is not None for n in at):
                    d = {n: np.array([at[n].pos.x, at[n].pos.y, at[n].pos.z]) for n in at}
                    d["aa"] = _AA3to1.get(r.name.upper(), "A")
                    res.append(d)
        break
    return res


def _cbdir(r):
    b, c = r["CA"] - r["N"], r["C"] - r["CA"]
    a = np.cross(b, c)
    cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + r["CA"]
    v = cb - r["CA"]
    return v / (np.linalg.norm(v) + 1e-9)


def partition(gt_complex, binder_chain, target_chain, cutoff, n):
    """NON-designable = HLH (target) residues within `cutoff` A (CA-CA) of the GT D-binder in the
    GT complex = the binding Tyr-29 "outside" epitope. DESIGNABLE = everything else (the inside
    Met-14 face + the GDDDS loop). Tied to the real binder, so M14 stays designable / Y29 fixed."""
    bca = np.array([list(r["CA"]) for r in load_bb(gt_complex, binder_chain)])
    tca = np.array([list(r["CA"]) for r in load_bb(gt_complex, target_chain)])
    out = [j + 1 for j in range(min(len(tca), n))
           if np.linalg.norm(bca - tca[j], axis=1).min() < cutoff]
    designable = [i for i in range(1, n + 1) if i not in out]
    return sorted(out), designable


def redesign(cif, chain, gt_complex, binder_chain, target_chain, cutoff,
             n_sample, top, temperature, out_path):  # pragma: no cover (gpu)
    import sys as _sys
    import torch
    repo = Path(__file__).resolve().parent.parent
    _sys.path.insert(0, str(repo / "LigandMPNN"))
    from model_utils import ProteinMPNN

    res = load_bb(cif, chain)
    n = len(res)
    native = "".join(r["aa"] for r in res)
    outside, designable = partition(gt_complex, binder_chain, target_chain, cutoff, n)
    nd_codes = [f"{native[i-1]}{i}" for i in outside]   # e.g. T6, S9...
    print("=== NON-DESIGNABLE (outside/Tyr face) ===")
    print("  positions:", outside)
    print("  residues :", nd_codes)
    print("  DESIGNABLE (inside + loop):", designable, f"  (loop 21-25 = GDDDS)")
    print(f"  native HLH: {native}")

    X = np.stack([[r["N"], r["CA"], r["C"], r["O"]] for r in res]).astype(np.float32)  # (n,4,3)
    S = np.array([_STR2INT.get(a, 0) for a in native], np.int32)
    chain_mask = np.array([0.0 if (i + 1) in outside else 1.0 for i in range(n)], np.float32)  # 1=design
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(repo / "LigandMPNN/model_params/proteinmpnn_v_48_020.pt"), map_location=dev)
    model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                        num_encoder_layers=3, num_decoder_layers=3,
                        k_neighbors=ckpt["num_edges"], device=dev, model_type="protein_mpnn")
    model.load_state_dict(ckpt["model_state_dict"]); model.to(dev).eval()

    def t(a, dt): return torch.tensor(a, device=dev, dtype=dt)
    R_idx = np.arange(1, n + 1, dtype=np.int32)
    base = {"X": t(X, torch.float32), "S": t(S, torch.int32), "mask": t(np.ones(n, np.int32), torch.int32),
            "chain_mask": t(chain_mask, torch.float32), "R_idx": t(R_idx, torch.int32),
            "chain_labels": t(np.zeros(n, np.int32), torch.int32)}
    feat = {k: v[None] for k, v in base.items()}
    feat.update({"batch_size": 1, "temperature": float(temperature),
                 "bias": torch.zeros([1, n, 21], device=dev), "symmetry_residues": [[]],
                 "symmetry_weights": [[]], "R_idx_original": base["R_idx"][None]})

    cands = []
    with torch.no_grad():
        for _ in range(n_sample):
            feat["randn"] = torch.randn([1, n], device=dev)
            o = model.sample(feat)
            Ss = o["S"][0].cpu().numpy()
            lp = o.get("log_probs")
            seq = list(native)
            for i in range(n):
                if chain_mask[i] == 1.0:                       # designed inside/loop
                    seq[i] = _INT2STR.get(int(Ss[i]), "A")
            seq = "".join(seq).replace("X", "A")
            # score = mean log-prob of the DESIGNED residues (higher = better)
            sc = None
            if lp is not None:
                L = lp[0].cpu().numpy()
                idx = [i for i in range(n) if chain_mask[i] == 1.0]
                sc = float(np.mean([L[i, int(Ss[i])] for i in idx]))
            cands.append((sc if sc is not None else 0.0, seq))
    # dedupe, rank by score desc, take top
    seen, ranked = set(), []
    for sc, seq in sorted(cands, key=lambda x: -x[0]):
        if seq not in seen:
            seen.add(seq); ranked.append({"score": round(sc, 4), "seq": seq})
        if len(ranked) >= top:
            break
    result = {"chain": chain, "native": native, "nondesignable_positions": outside,
              "nondesignable_residues": nd_codes, "designable_positions": designable,
              "n_sampled": n_sample, "top": ranked}
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"\n=== TOP {len(ranked)} by ProteinMPNN score (outside restored) ===")
    for i, r in enumerate(ranked):
        print(f"  {i+1:>2} score {r['score']:+.3f}  {r['seq']}")
    print(f"-> {out_path}")
    return result


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cif", required=True, help="L-HLH-alone CIF (backbone to redesign)")
    p.add_argument("--chain", default="A")
    p.add_argument("--gt_complex", required=True, help="GT complex CIF (defines the binding epitope)")
    p.add_argument("--binder_chain", default="A")
    p.add_argument("--target_chain", default="B")
    p.add_argument("--cutoff", type=float, default=9.0, help="CA-CA epitope cutoff (A)")
    p.add_argument("--n_sample", type=int, default=200)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--out", default="XenoDesign1_local_ref/hlh_redesign/top.json")
    return p.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover (gpu)
    a = _parse()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    redesign(a.cif, a.chain, a.gt_complex, a.binder_chain, a.target_chain, a.cutoff,
             a.n_sample, a.top, a.temperature, a.out)
    sys.exit(0)
