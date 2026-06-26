"""De-game a ProteinMPNN/LigandMPNN redesign and emit a diverse top-N for D-fold screening.

Gamed = low-complexity (poly-Ala etc.) over the DESIGNED positions. We veto by max single-AA
fraction + Shannon entropy over the designed subset (the channel where MPNN games Chai's L optimism),
optionally re-rank by ESM-2 PLL (if a model_fn is wired), then dedupe by (loop, inside-face) and take
the top-N most MPNN-confident survivors. Output feeds predict_target_fold (--chirality D) screening.

Usage:
  python scripts/degame_select.py --in <redesign top.json> --n 12 --out <degamed.json> [--max_frac 0.45]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

LOOP = list(range(21, 26))  # 1-based GDDDS region


def shannon(seq):
    c = Counter(seq)
    n = len(seq)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--max_frac", type=float, default=0.45, help="veto if any AA > this frac of designed positions")
    p.add_argument("--min_entropy", type=float, default=2.5, help="veto if Shannon entropy (bits) over designed < this")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    d = json.loads(Path(a.inp).read_text())
    des = d["designable_positions"]
    native = d["native"]
    cands = []
    for e in d["top"]:
        seq = e["seq"]
        designed = "".join(seq[i - 1] for i in des)
        comp = Counter(designed)
        maxfrac = max(comp.values()) / len(designed)
        ent = shannon(designed)
        loop = "".join(seq[i - 1] for i in LOOP)
        gamed = maxfrac > a.max_frac or ent < a.min_entropy
        cands.append({"seq": seq, "mpnn_score": e["score"], "loop": loop,
                      "designed": designed, "max_aa_frac": round(maxfrac, 3),
                      "entropy_bits": round(ent, 2), "gamed": gamed,
                      "top_aa": comp.most_common(1)[0][0]})

    survivors = [c for c in cands if not c["gamed"]]
    # dedupe by (loop, full designed face); keep best MPNN score
    seen, dedup = set(), []
    for c in sorted(survivors, key=lambda x: -x["mpnn_score"]):
        key = (c["loop"], c["designed"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    chosen = dedup[: a.n]

    out = {
        "source": a.inp, "native": native, "designable_positions": des,
        "n_candidates": len(cands), "n_passed_degame": len(survivors),
        "veto": {"max_frac": a.max_frac, "min_entropy": a.min_entropy},
        "gamed_examples": [c["seq"] for c in cands if c["gamed"]][:3],
        "chosen": chosen,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"candidates={len(cands)} passed_degame={len(survivors)} chosen={len(chosen)}")
    for i, c in enumerate(chosen):
        print(f"  {i+1:>2} mpnn{c['mpnn_score']:+.3f} loop={c['loop']} maxAA={c['max_aa_frac']} H={c['entropy_bits']}  {c['seq']}")
    return out


if __name__ == "__main__":
    main()
