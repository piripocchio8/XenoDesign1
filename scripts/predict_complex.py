"""Predict a 2-chain complex with per-chain chirality (Chai 0.6.1, MSA-free + ESM), saving scores +
per-token PAE so scripts/score_complex.py can read ipTM/ipAE/ipSAE. For benchmark calibration
(8GQP/7YH8) and for scoring designed D-binder / L-target complexes (P3/P4).

Run (GPU; chai container):
  python scripts/predict_complex.py --seq_a <BINDER> --chir_a D --seq_b <TARGET> --chir_b L \
      --recycles 3 --seed 42 --out_dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def run(seq_a, chir_a, seq_b, chir_b, recycles, steps, seed, device, out_dir):  # pragma: no cover (gpu)
    from chai_lab.chai1 import run_inference

    from xenodesign.backends.chai_backend import _save_confidence_npz, write_inputs
    ents = [{"type": "protein", "name": "A", "sequence": seq_a, "chirality": chir_a},
            {"type": "protein", "name": "B", "sequence": seq_b, "chirality": chir_b}]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fasta = write_inputs(ents, out_dir)
    chai_out = out_dir / "chai_out"; chai_out.mkdir(parents=True, exist_ok=True)
    cands = run_inference(fasta_file=fasta, output_dir=chai_out, device=device, seed=seed,
                          num_diffn_timesteps=steps, num_trunk_recycles=recycles,
                          use_esm_embeddings=True, use_msa_server=False)
    _save_confidence_npz(cands, chai_out)
    # best model ipTM
    bi, bv, iptm = 0, -np.inf, None
    for f in sorted(chai_out.glob("scores.model_idx_*.npz")):
        z = np.load(f); a = float(z["aggregate_score"].reshape(-1)[0])
        if a > bv:
            bv = a
            if "per_chain_pair_iptm" in z:
                m = np.asarray(z["per_chain_pair_iptm"]); m = m.reshape(int(m.size ** 0.5), -1)
                iptm = round(float(max(m[0, 1], m[1, 0])), 3)
    res = {"seq_a": seq_a, "chir_a": chir_a, "seq_b": seq_b, "chir_b": chir_b,
           "na": len(seq_a), "nb": len(seq_b), "seed": seed, "recycles": recycles, "iptm": iptm,
           "chai_out": str(chai_out)}
    (out_dir / "complex_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seq_a", required=True); p.add_argument("--chir_a", choices=["L", "D"], required=True)
    p.add_argument("--seq_b", required=True); p.add_argument("--chir_b", choices=["L", "D"], required=True)
    p.add_argument("--recycles", type=int, default=3); p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover (gpu)
    a = _parse()
    run(a.seq_a, a.chir_a, a.seq_b, a.chir_b, a.recycles, a.steps, a.seed, a.device, a.out_dir)
    sys.exit(0)
