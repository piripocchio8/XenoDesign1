"""Mixed parity-aware objective (T20 / ADR-014) — normalize each panel metric to [0,1] (good=1) and
combine with weights. Calibrated so the two REAL deposited heterochiral D-binder/L-target interfaces
(8GQP, GT D-ABLE) score high, encoding the empirical finding that Chai ipTM/ipAE under-rate real
heterochiral binders (8GQP ipTM 0.35, GT 0.44) while parity-invariant geometry (BSA, contacts, packing)
and ipSAE recognize them. Weights are a reasoned PROPOSAL pending decoy-based fitting (negatives needed).

Usage: python scripts/mixed_objective.py <panel.json> [<panel2.json> ...]
"""
from __future__ import annotations

import json
import sys

# EXPANDED DATA-DRIVEN REFIT (2026-06-23, fit_objective on 11 REAL heterochiral systems — 8GQP, 7YH8v2,
# 6 gp41 (3L35/2R5B/2R3C/3MGN/1CZQ/2Q3I), 3 MDM2 (3LNJ/7KJM/3IWY; C-ter-Gly + pocket restraint, all 3 gate) —
# SCRAMBLES ONLY (register-shifts dropped), all 8 terms re-evaluated FROM SCRATCH; fit_expanded.json +
# docs/results/2026-06-23-expanded-objective-fit.md). The discriminator is a COMBINATION (different terms carry
# different systems): Sc separates 9/11, hbond 7/11 (MDM2 + low-Sc gp41 — hbond FLIPPED POSITIVE on the larger
# set: the n=2 zeroing was exactly the carryover error to avoid), iptm carries 8GQP, ipsae 3LNJ;
# bsa/contacts/pack/ipae net-negative dead weight. Worst-case margin still negative (8GQP needs iptm and
# Sc-by-luck hurts it) → that residual is the case for in-selection negative design (T01), not a static tweak.
WEIGHTS = {"bsa": 0.0, "contacts": 0.0, "pack": 0.0, "sc": 0.84, "ipsae": 0.0, "iptm": 0.06, "ipae": 0.0, "hbond": 0.10}


def _clip(x):
    return max(0.0, min(1.0, x))


def normalize(p):
    return {
        "bsa": _clip(p.get("bsa_A2", 0) / 1400.0),                       # ~1400 A2 = excellent interface
        "contacts": _clip(p.get("n_residue_contacts", 0) / 30.0),
        "pack": _clip((5.0 - p.get("iface_closest_mean_A", 5.0)) / 1.5), # 3.5 A=1.0, 5.0 A=0 (lower gap better)
        "sc": _clip((p.get("sc_normal_opp") or 0.0) / 0.5),             # convex/concave normal-opposition Sc
        "ipsae": _clip(p.get("ipsae", 0.0)),
        "iptm": _clip(p.get("iptm", 0.0)),
        "ipae": _clip((20.0 - p.get("ipae", 20.0)) / 15.0),             # 5 A=1.0, 20 A=0 (lower PAE better)
        # register-SPECIFICITY term: reference-free cross-chain H-bond density (score_complex).
        # Real D-binder interfaces show ~0.06-0.11 H-bonds/interface-res; a register shift breaks
        # specific donor-acceptor pairs and drops it. 0.10 density -> 1.0. Optional (weight may be 0).
        "hbond": _clip((p.get("hbond_density") or 0.0) / 0.10),
    }


def score(p):
    n = normalize(p)
    s = sum(WEIGHTS[k] * n[k] for k in WEIGHTS)
    return s, n


if __name__ == "__main__":
    print(f"weights: {WEIGHTS}\n")
    for path in sys.argv[1:]:
        p = json.load(open(path))
        s, n = score(p)
        print(f"{path.split('/')[-1]:>28}  SCORE={s:.3f}  " +
              "  ".join(f"{k}={n[k]:.2f}" for k in WEIGHTS) +
              f"   (raw: bsa={p.get('bsa_A2')} con={p.get('n_residue_contacts')} "
              f"iptm={p.get('iptm')} ipsae={p.get('ipsae')} ipae={p.get('ipae')})")
