"""Run a batch of 2-chain complex predictions (positives + decoys) from a JSON list, on one GPU.
Each item: {name, seq_a, chir_a, seq_b, chir_b, out_dir}. Used for objective fitting (P4).

Usage: python scripts/predict_batch.py --json <list.json> --indices 0,1,2 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_complex import run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True)
    p.add_argument("--indices", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    items = json.load(open(a.json))
    for i in [int(x) for x in a.indices.split(",")]:
        e = items[i]
        print(f"=== [{i}] {e['name']} ===", flush=True)
        try:
            r = run(e["seq_a"], e["chir_a"], e["seq_b"], e["chir_b"], 3, 200, a.seed, a.device, e["out_dir"])
            print(f"    {e['name']} iptm={r['iptm']}", flush=True)
        except Exception as ex:
            print(f"    {e['name']} FAILED: {ex}", flush=True)


if __name__ == "__main__":
    main()
