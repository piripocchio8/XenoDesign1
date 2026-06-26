#!/usr/bin/env python
"""Render the cyclization-calibration result tables from the per-lane JSON blobs.

Consumes the pos.json + neg.json written by ``run_cyclization_calibration.py`` and emits
the deliverable tables (CPU-only, no chai):

  * GROUND-TRUTH closure-vs-steps curve per (case, start): C<->N distance, omega planarity,
    closed? — and the MINIMUM steps (+best start) at which each KNOWN cycle (POS) closes.
  * Per-term objective table per (case, start, steps): mainchain_plddt / chirality /
    geometry / ptm / aggregate.
  * POS-vs-NEG discrimination: at each (start, steps), which TERM(s) separate the real
    mixed-chirality cycles from the strained full-L controls (mean POS - mean NEG).

Pure analysis; safe to run on the host (no GPU). Usage:
    python scripts/analyze_cyclization_calibration.py --pos .cyc_calib/pos.json \
        --neg .cyc_calib/neg.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_TERMS = ["mainchain_plddt", "chirality", "geometry", "ptm", "objective"]
_CLOSED_CUTOFF = 1.6  # A, matches head_to_tail_closure_geometry_from_cif


def _load(path) -> list[dict]:
    return json.loads(Path(path).read_text())["records"]


def _fmt(v, nd=3):
    if v is None:
        return "  —  "
    if isinstance(v, float) and v != v:  # NaN
        return " nan "
    return f"{v:.{nd}f}"


def closure_table(records: list[dict]) -> str:
    """Per (case,start,steps) ground-truth closure: C<->N dist, omega planarity, closed?."""
    lines = ["| case | start | steps | C–N dist (Å) | ω planarity | closed? |",
             "|------|-------|------:|-------------:|------------:|:-------:|"]
    for r in sorted(records, key=lambda r: (r["case_id"], r["start"], r["steps"])):
        lines.append(
            f"| {r['case_id']} | {r['start']} | {r['steps']} | "
            f"{_fmt(r.get('cn_distance'), 2)} | {_fmt(r.get('omega_planarity'))} | "
            f"{'✅' if r.get('closed') else '·'} |")
    return "\n".join(lines)


def min_steps_to_close(records: list[dict]) -> str:
    """Smallest steps (per start) at which each POS case closes (cn<=cutoff)."""
    lines = ["| case | start | min steps to close | C–N @ that step |",
             "|------|-------|-------------------:|----------------:|"]
    cases = sorted({r["case_id"] for r in records if r["is_good"]})
    for cid in cases:
        for start in sorted({r["start"] for r in records}):
            closed = sorted(
                (r for r in records
                 if r["case_id"] == cid and r["start"] == start and r.get("closed")),
                key=lambda r: r["steps"])
            if closed:
                r0 = closed[0]
                lines.append(f"| {cid} | {start} | {r0['steps']} | "
                             f"{_fmt(r0.get('cn_distance'), 2)} |")
            else:
                lines.append(f"| {cid} | {start} | never (≤max swept) | — |")
    return "\n".join(lines)


def per_term_table(records: list[dict]) -> str:
    lines = ["| case | start | steps | mc-pLDDT | chirality | geometry | pTM | aggregate |",
             "|------|-------|------:|---------:|----------:|---------:|----:|----------:|"]
    for r in sorted(records, key=lambda r: (r["case_id"], r["start"], r["steps"])):
        lines.append(
            f"| {r['case_id']} | {r['start']} | {r['steps']} | "
            f"{_fmt(r.get('mainchain_plddt'))} | {_fmt(r.get('chirality'))} | "
            f"{_fmt(r.get('geometry'))} | {_fmt(r.get('ptm'))} | "
            f"{_fmt(r.get('objective'))} |")
    return "\n".join(lines)


def discrimination_table(pos: list[dict], neg: list[dict]) -> str:
    """At each (start,steps): mean(POS term) - mean(NEG term) for each term + closure.

    A POSITIVE margin means the term ranks the real mixed-chirality cycles above the
    strained full-L controls (what can feed the bees). Also reports the closure-fraction
    margin (POS closed-frac - NEG closed-frac) as the ground-truth discriminator."""
    def _key(r):
        return (r["start"], r["steps"])

    starts_steps = sorted({_key(r) for r in pos} & {_key(r) for r in neg})
    cols = ["mainchain_plddt", "chirality", "geometry", "ptm", "objective"]
    head = ("| start | steps | " + " | ".join(
        {"mainchain_plddt": "Δmc-pLDDT", "chirality": "Δchir", "geometry": "Δgeom",
         "ptm": "ΔpTM", "objective": "Δaggregate"}[c] for c in cols)
        + " | Δclosed-frac | Δ(C–N) |")
    sep = "|" + "|".join(["---"] * (len(cols) + 4)) + "|"
    lines = [head, sep]
    for (start, steps) in starts_steps:
        ps = [r for r in pos if r["start"] == start and r["steps"] == steps]
        ns = [r for r in neg if r["start"] == start and r["steps"] == steps]

        def _mean(rows, k):
            vals = [r[k] for r in rows if r.get(k) is not None
                    and not (isinstance(r[k], float) and r[k] != r[k])]
            return sum(vals) / len(vals) if vals else None

        def _delta(k):
            mp, mn = _mean(ps, k), _mean(ns, k)
            return None if (mp is None or mn is None) else mp - mn

        deltas = [_delta(c) for c in cols]
        pc = sum(1 for r in ps if r.get("closed")) / len(ps) if ps else 0.0
        nc = sum(1 for r in ns if r.get("closed")) / len(ns) if ns else 0.0
        # closer C-N is better, so margin = mean(NEG cn) - mean(POS cn) (positive = POS closer)
        pcn, ncn = _mean(ps, "cn_distance"), _mean(ns, "cn_distance")
        dcn = None if (pcn is None or ncn is None) else ncn - pcn
        lines.append(
            f"| {start} | {steps} | " + " | ".join(
                ("+" if (d is not None and d >= 0) else "") + _fmt(d)
                for d in deltas)
            + f" | {pc - nc:+.2f} | {('+' if (dcn is not None and dcn >= 0) else '')}"
              f"{_fmt(dcn, 2)} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    args = ap.parse_args()
    pos, neg = _load(args.pos), _load(args.neg)
    print("## Ground-truth closure (POS lane)\n")
    print(closure_table(pos))
    print("\n## Ground-truth closure (NEG lane)\n")
    print(closure_table(neg))
    print("\n## Minimum steps to close (POS)\n")
    print(min_steps_to_close(pos))
    print("\n## Per-term objective (POS)\n")
    print(per_term_table(pos))
    print("\n## Per-term objective (NEG)\n")
    print(per_term_table(neg))
    print("\n## POS-vs-NEG discrimination (mean POS − mean NEG)\n")
    print(discrimination_table(pos, neg))


if __name__ == "__main__":
    main()
