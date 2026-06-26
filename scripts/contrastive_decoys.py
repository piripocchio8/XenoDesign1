"""T01/T18 in-selection negative design — generate register-shift DECOYS for each design binder.

For every binder we emit the real complex plus binder-register-shifted decoys (circular shift of the binder
sequence by {3,4,7}; the TARGET is held fixed, per ADR-011 — register specificity is tested on the binder).
A register-specific binder binds well at its register but poorly when shifted; a generic amphipathic helix
binds equally. Output is a predict_batch JSON (binder = all-D, target = L). Score with contrastive_rank.py:
margin = score(real) - max_shift score(shift).

Usage:
  python scripts/contrastive_decoys.py --designs <dir1> <dir2> ... --target_fasta <t.fasta> \
      --out_root <root> --json <out.json> [--shifts 3,4,7]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _target_seq(fasta, record="trimer_DL_ABLE_B"):
    txt = Path(fasta).read_text()
    for block in txt.split(">"):
        if block.strip() and record in block.splitlines()[0]:
            return "".join(block.splitlines()[1:]).strip()
    # fallback: last record
    recs = [b for b in txt.split(">") if b.strip()]
    return "".join(recs[-1].splitlines()[1:]).strip()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--designs", nargs="+", required=True, help="design_alpha out_dirs (read selected_l_seq)")
    p.add_argument("--target_fasta", required=True)
    p.add_argument("--out_root", default="XenoDesign1_local_ref/contrastive")
    p.add_argument("--json", required=True)
    p.add_argument("--shifts", default="3,4,7")
    a = p.parse_args(argv)
    tgt = _target_seq(a.target_fasta)
    shifts = [int(s) for s in a.shifts.split(",")]
    items = []
    for d in a.designs:
        res = json.load(open(Path(d) / "alpha_result.json"))
        b = res["selected_l_seq"]            # binder letters; predicted all-D below
        tag = Path(d).name
        items.append({"name": f"{tag}__real", "design": tag, "kind": "real", "shift": 0,
                      "seq_a": b, "chir_a": "D", "seq_b": tgt, "chir_b": "L",
                      "out_dir": f"{a.out_root}/{tag}/real"})
        for s in shifts:
            bs = b[s:] + b[:s]               # circular shift
            items.append({"name": f"{tag}__shift{s}", "design": tag, "kind": "shift", "shift": s,
                          "seq_a": bs, "chir_a": "D", "seq_b": tgt, "chir_b": "L",
                          "out_dir": f"{a.out_root}/{tag}/shift{s}"})
    Path(a.json).write_text(json.dumps(items, indent=2))
    print(f"target ({len(tgt)} aa): {tgt}")
    print(f"{len(items)} complexes for {len(a.designs)} designs (real + shifts {shifts}) -> {a.json}")
    for it in items[:6]:
        print(f"  {it['name']:24} binder={it['seq_a']}")


if __name__ == "__main__":
    main()
