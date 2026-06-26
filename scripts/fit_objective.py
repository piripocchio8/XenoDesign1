"""Fit the mixed-objective weights from labelled complexes (P4) — INTRA-SYSTEM edition.

Objective = AVERAGE OF WITHIN-SYSTEM (intra-PDB) true-vs-false separations, NOT the
pooled-across-systems separation.

WHY: the benchmark systems sit at very different absolute metric
baselines (8GQP real ipTM ~0.35, 7YH8 ~0.64, 9DXX likely ~0.9 with MSA). A POOLED fitter
that does sep[f] = mean(ALL positives) - mean(ALL negatives) and margin = min(all pos) -
max(all neg) rewards a metric for separating high-bindability systems from low ones — so a
7YH8 scramble can out-score the 8GQP real. WRONG: during design we only ever compare a
candidate to its OWN decoys (always intra-system). So the fit must maximize
mean_i( true_i - false_i ) computed WITHIN each system i.

Design philosophy: keep it simple and few-points-robust. The fitted weight for feature f is
just the AVERAGE over systems of the intra-system delta (clipped at 0, renormalized to sum 1)
— no 7-param logistic overfit on a handful of rows.

Systems are inferred from the panel filename prefix before the first '_'
(8GQP_pred_panel.json -> "8GQP", 7YH8_scram1_panel.json -> "7YH8"). Override with --group.
Negative kind (scramble / shift) is inferred from the filename when encoded and reported.

ipae is "lower=better" but mixed_objective.normalize already inverts it, so for every
normalized feature +Delta means the true binder is better.

Usage:
  python scripts/fit_objective.py --pos 8GQP_pred_panel.json 7YH8_pred_panel.json \\
        --neg 8GQP_scram1_panel.json 7YH8_scram1_panel.json [--out fit.json]
  python scripts/fit_objective.py --selfcheck          # synthetic pooled-vs-intra assert test
  # explicit grouping (parallel lists, one token per --pos/--neg entry):
  python scripts/fit_objective.py --pos a.json b.json --neg c.json d.json \\
        --group 8GQP 7YH8 --group_neg 8GQP 7YH8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mixed_objective import normalize  # noqa: E402

# Base features always present from normalize(); "hbond" is added only if a system actually
# carries a non-trivial hbond signal (kept optional per the register-specificity term).
BASE_FEATURES = ["bsa", "contacts", "pack", "sc", "ipsae", "iptm", "ipae"]
# Candidate single metrics ranked by intra-system dynamic range.
SINGLE_METRICS = ["iptm", "ipae", "ipsae"]

OBJECTIVE_TAG = "mean-of-intra-system-deltas (NOT pooled)"


def infer_system(name: str) -> str:
    """System id = filename prefix before the first '_' (8GQP_pred_panel.json -> 8GQP)."""
    base = Path(name).name
    return base.split("_", 1)[0]


def infer_neg_kind(name: str) -> str:
    """Best-effort negative kind from the filename: scramble / shift / unknown."""
    low = Path(name).name.lower()
    if "scram" in low:
        return "scramble"
    if "shift" in low or "register" in low or "reg" in low:
        return "shift"
    return "unknown"


def load_entry(path: str):
    """-> (system, name, normalized_dict, neg_kind)."""
    name = Path(path).name
    return infer_system(name), name, normalize(json.load(open(path))), infer_neg_kind(name)


def active_features(entries) -> list[str]:
    """BASE_FEATURES plus 'hbond' iff any entry carries a non-zero hbond term."""
    feats = list(BASE_FEATURES)
    if any(d.get("hbond", 0.0) for _, _, d, _ in entries):
        feats.append("hbond")
    return feats


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def fit(pos_entries, neg_entries, features):
    """Core intra-system fit. Returns the full output dict.

    pos_entries / neg_entries: lists of (system, name, normalized_dict, neg_kind).
    """
    systems = sorted({s for s, _, _, _ in pos_entries} | {s for s, _, _, _ in neg_entries})
    pos_by = defaultdict(list)
    neg_by = defaultdict(list)
    for s, n, d, _ in pos_entries:
        pos_by[s].append((n, d))
    for s, n, d, k in neg_entries:
        neg_by[s].append((n, d, k))

    # systems usable for separation = those with >=1 pos AND >=1 neg
    sep_systems = [s for s in systems if pos_by[s] and neg_by[s]]

    # ---- (2) per-feature intra-system delta, then AVERAGE over systems -> weights ----
    per_system_sep = {}      # system -> {feature: Delta_i,f}
    per_feature_deltas = defaultdict(list)  # feature -> [Delta_i,f over systems]
    for s in sep_systems:
        d_s = {}
        for f in features:
            mp = _mean(d[f] for _, d in pos_by[s])
            mn = _mean(d[f] for _, d, _ in neg_by[s])
            d_s[f] = mp - mn
            per_feature_deltas[f].append(d_s[f])
        per_system_sep[s] = {f: round(v, 4) for f, v in d_s.items()}

    per_feature_intra_mean = {f: _mean(per_feature_deltas[f]) for f in features}
    raw = {f: max(0.0, per_feature_intra_mean[f]) for f in features}
    tot = sum(raw.values()) or 1.0
    weights = {f: round(raw[f] / tot, 4) for f in features}

    def score(d):
        return round(sum(weights[f] * d[f] for f in features), 4)

    # ---- (4) PER-SYSTEM margins (within-system min(pos) - max(neg)) ----
    by_system_margin = {}
    for s in sep_systems:
        ps = [score(d) for _, d in pos_by[s]]
        ns = [score(d) for _, d, _ in neg_by[s]]
        by_system_margin[s] = round(min(ps) - max(ns), 4)
    margins = list(by_system_margin.values())
    per_system_margins = {
        "mean": round(_mean(margins), 4) if margins else None,
        "worst": round(min(margins), 4) if margins else None,
        "by_system": by_system_margin,
        "note": "margin_i = min(pos_i) - max(neg_i) WITHIN system i; pooled margin intentionally NOT reported.",
    }

    # ---- (3) SINGLE-METRIC selector: per-system Delta vs the HARDEST decoy ----
    # only rank metrics actually present in the data (so fit() stays usable on synthetic dicts)
    all_dicts = [d for s in sep_systems for _, d in pos_by[s]] + \
                [d for s in sep_systems for _, d, _ in neg_by[s]]
    present_metrics = [m for m in SINGLE_METRICS if all(m in d for d in all_dicts)]
    single_ranking = []
    for m in present_metrics:
        by_sys = {}
        for s in sep_systems:
            true_best = max(d[m] for _, d in pos_by[s])            # best true on this metric
            worst_neg = max(d[m] for _, d, _ in neg_by[s])         # hardest decoy = most favorable neg
            by_sys[s] = round(true_best - worst_neg, 4)
        deltas = list(by_sys.values())
        single_ranking.append({
            "metric": m,
            "mean_intra_delta": round(_mean(deltas), 4) if deltas else None,
            "worst_intra_delta": round(min(deltas), 4) if deltas else None,
            "by_system": by_sys,
        })
    # rank by mean intra delta desc; tiebreak: non-negative worst-case wins
    single_ranking.sort(
        key=lambda r: (r["mean_intra_delta"] if r["mean_intra_delta"] is not None else -1e9,
                       (r["worst_intra_delta"] or -1e9) >= 0,
                       r["worst_intra_delta"] if r["worst_intra_delta"] is not None else -1e9),
        reverse=True,
    )

    neg_kinds = sorted({k for _, _, _, k in neg_entries})
    out = {
        "objective": OBJECTIVE_TAG,
        "features": features,
        "systems": sep_systems,
        "neg_kinds_present": neg_kinds,
        "fitted_weights": weights,
        "per_feature_intra_mean": {f: round(per_feature_intra_mean[f], 4) for f in features},
        "per_system_separation": per_system_sep,
        "single_metric_ranking": single_ranking,
        "per_system_margins": per_system_margins,
        "scored": {
            "positives": {n: score(d) for s, n, d, _ in pos_entries},
            "negatives": {n: score(d) for s, n, d, _ in neg_entries},
        },
        "all_systems_separate": all(v > 0 for v in margins) if margins else False,
    }
    return out


def _attach_groups(paths, normd, groups):
    """Build (system, name, normd, neg_kind) entries, overriding system with explicit groups."""
    entries = []
    for i, (path, nd) in enumerate(zip(paths, normd)):
        name = Path(path).name
        sysid = groups[i] if groups else infer_system(name)
        entries.append((sysid, name, nd, infer_neg_kind(name)))
    return entries


# --------------------------------------------------------------------------- #
# SELFCHECK: synthetic 2-system case where POOLED picks WRONG, INTRA picks RIGHT
# --------------------------------------------------------------------------- #
def selfcheck() -> int:
    """Synthetic case: two systems at very different baselines.

    Feature 'good' separates true>false WITHIN both systems (the RIGHT metric).
    Feature 'bad'  does NOT separate within a system, but its huge cross-system baseline
    gap makes the POOLED delta favor it (the WRONG metric).

    Asserts: (a) pooled vs intra-mean margins give DIFFERENT verdicts,
             (b) intra objective identifies 'good' (separates within both systems),
             (c) per-system margins are computed within-system.
    """
    feats = ["good", "bad"]

    def E(system, name, good, bad, kind):
        # full normalized dict so fit()'s score works; only feats are used here
        return (system, name, {"good": good, "bad": bad}, kind)

    # System LOW: low absolute baseline. System HIGH: high absolute baseline on 'bad'.
    # 'good': true beats false by +0.2 within BOTH systems (consistent intra separation -> RIGHT metric).
    # 'bad' : within each system true is NOT better than false (intra delta <= 0), yet the pools are
    #         arranged (more positives in HIGH, more negatives in LOW) so the cross-system baseline gap
    #         makes the POOLED mean(pos.bad) - mean(neg.bad) come out POSITIVE -> pooled picks 'bad' (WRONG).
    pos = [
        E("LOW", "LOW_real", good=0.50, bad=0.05, kind="real"),
        E("HIGH", "HIGH_real1", good=0.50, bad=0.90, kind="real"),
        E("HIGH", "HIGH_real2", good=0.50, bad=0.90, kind="real"),
    ]
    neg = [
        E("LOW", "LOW_scram1", good=0.30, bad=0.07, kind="scramble"),   # bad: neg >= pos within LOW
        E("LOW", "LOW_scram2", good=0.30, bad=0.07, kind="scramble"),
        E("HIGH", "HIGH_scram", good=0.30, bad=0.92, kind="scramble"),  # bad: neg >= pos within HIGH
    ]

    # ---- POOLED objective (the OLD/wrong way) ----
    def pooled_delta(f):
        mp = _mean(d[f] for _, _, d, _ in pos)
        mn = _mean(d[f] for _, _, d, _ in neg)
        return mp - mn
    pooled = {f: pooled_delta(f) for f in feats}
    # pooled weights: clip>=0, renorm
    praw = {f: max(0.0, pooled[f]) for f in feats}
    ptot = sum(praw.values()) or 1.0
    pooled_w = {f: praw[f] / ptot for f in feats}

    def pooled_score(d):
        return sum(pooled_w[f] * d[f] for f in feats)
    pooled_margin = min(pooled_score(d) for _, _, d, _ in pos) - max(pooled_score(d) for _, _, d, _ in neg)

    # ---- INTRA objective (the NEW/right way) ----
    res = fit(pos, neg, feats)
    intra_w = res["fitted_weights"]
    intra_mean_margin = res["per_system_margins"]["mean"]
    intra_worst_margin = res["per_system_margins"]["worst"]

    # ---------- assertions ----------
    # 'bad' has a positive pooled delta (cross-system gap leaks in) ...
    assert pooled["bad"] > 0, f"setup invalid: pooled bad delta should be >0, got {pooled['bad']:.3f}"
    # ... while its INTRA mean delta is <= 0 (no within-system separation):
    assert per_feature_intra_mean_for(res, "bad") <= 0 + 1e-9, \
        f"intra 'bad' delta should be <=0, got {per_feature_intra_mean_for(res, 'bad'):.3f}"
    # 'good' separates within both systems -> positive intra mean delta:
    assert per_feature_intra_mean_for(res, "good") > 0, "intra 'good' delta should be >0"

    # (b) intra objective puts essentially ALL weight on 'good', pooled leaks weight onto 'bad':
    assert intra_w["good"] > intra_w["bad"], f"intra should favor 'good': {intra_w}"
    assert pooled_w["bad"] >= pooled_w["good"], f"pooled should favor 'bad': {pooled_w}"
    assert intra_w["good"] == 1.0 or intra_w["bad"] == 0.0, \
        f"intra 'bad' weight should clip to 0: {intra_w}"

    # (a) pooled vs intra-mean margins give DIFFERENT verdicts:
    pooled_sep = pooled_margin > 0
    intra_sep = intra_mean_margin > 0
    assert pooled_sep != intra_sep, (
        f"verdicts should DIFFER: pooled_margin={pooled_margin:.3f} (sep={pooled_sep}), "
        f"intra_mean_margin={intra_mean_margin:.3f} (sep={intra_sep})"
    )
    # specifically: pooled says NO-separate (true loses to a cross-system decoy), intra says YES:
    assert not pooled_sep and intra_sep, (
        f"expected pooled FAILS and intra SUCCEEDS: pooled={pooled_margin:.3f}, intra={intra_mean_margin:.3f}"
    )

    # (c) per-system margins are within-system and BOTH positive under intra weights:
    bysys = res["per_system_margins"]["by_system"]
    assert set(bysys) == {"LOW", "HIGH"}, f"per-system margins keys: {bysys}"
    assert all(v > 0 for v in bysys.values()), f"each within-system margin should be >0: {bysys}"
    assert intra_worst_margin > 0, f"worst-case intra margin should be >0: {intra_worst_margin}"

    # single-metric ranking is empty here ('good'/'bad' aren't iptm/ipae/ipsae) — that's expected.
    assert res["single_metric_ranking"] == [], "synthetic feats carry no SINGLE_METRICS"

    print(json.dumps({
        "selfcheck": "PASS",
        "objective": OBJECTIVE_TAG,
        "pooled_weights": {f: round(pooled_w[f], 3) for f in feats},
        "intra_weights": intra_w,
        "pooled_margin": round(pooled_margin, 4),
        "pooled_verdict_separates": pooled_sep,
        "intra_mean_margin": round(intra_mean_margin, 4),
        "intra_worst_margin": round(intra_worst_margin, 4),
        "intra_verdict_separates": intra_sep,
        "per_feature_intra_mean": res["per_feature_intra_mean"],
        "per_system_margins_by_system": bysys,
        "verdicts_differ": pooled_sep != intra_sep,
    }, indent=2))
    return 0


def per_feature_intra_mean_for(res, f):
    return res["per_feature_intra_mean"][f]


def main(argv=None):
    p = argparse.ArgumentParser(description="Fit mixed-objective weights as the MEAN of intra-system deltas.")
    p.add_argument("--pos", nargs="+", help="positive (real binder) panel JSONs")
    p.add_argument("--neg", nargs="+", help="negative (decoy) panel JSONs")
    p.add_argument("--group", nargs="+", default=None,
                   help="explicit system id per --pos entry (overrides filename inference)")
    p.add_argument("--group_neg", nargs="+", default=None,
                   help="explicit system id per --neg entry (overrides filename inference)")
    p.add_argument("--neg_kind", choices=["scramble", "shift", "all"], default="all",
                   help="note in output / filter negatives by encoded kind")
    p.add_argument("--out", default=None)
    p.add_argument("--selfcheck", action="store_true",
                   help="run synthetic pooled-vs-intra assert test and exit")
    a = p.parse_args(argv)

    if a.selfcheck:
        return selfcheck()

    if not a.pos or not a.neg:
        p.error("--pos and --neg are required (unless --selfcheck)")
    if a.group and len(a.group) != len(a.pos):
        p.error("--group must have one entry per --pos")
    if a.group_neg and len(a.group_neg) != len(a.neg):
        p.error("--group_neg must have one entry per --neg")

    pos_normd = [normalize(json.load(open(x))) for x in a.pos]
    neg_normd = [normalize(json.load(open(x))) for x in a.neg]
    pos_entries = _attach_groups(a.pos, pos_normd, a.group)
    neg_entries = _attach_groups(a.neg, neg_normd, a.group_neg)

    if a.neg_kind != "all":
        neg_entries = [e for e in neg_entries if e[3] == a.neg_kind]
        if not neg_entries:
            p.error(f"no negatives matched --neg_kind {a.neg_kind}")

    feats = active_features(pos_entries + neg_entries)
    out = fit(pos_entries, neg_entries, feats)
    out["neg_kind_filter"] = a.neg_kind
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
