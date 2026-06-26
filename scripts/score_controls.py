"""Within-Chai specificity controls for an α design (Gate-A #35; Gate-B upgrades; cross-model
deferred to IBiSCo).

Given a candidate binder sequence, score on Chai (UNRESTRAINED, identical pipeline + token split):
  - the DESIGN vs the real L-HLH target                       (headline)
  - K composition-matched SCRAMBLES of the design vs target   (negative: does sequence ORDER matter?)
  - the design vs each OFF-TARGET helix-as-target             (negative: is it specific to THIS target?)
  - the design vs the REGISTER-SHIFTED real target            (negative: register-specific? Gate-B #2)
  - the GENUINE reference D-binder vs target                  (positive control; should ≈ baseline 0.44)
  - [--binder_alone] the design ALONE, all-D, no target       (STRICT from-scratch chirality; Gate-B #4)
then apply the pre-registered WIN_MARGINS and report the interface footprint (heptad register).

WORST-DECOY is the honest specificity metric (Gate-B #1): the PASS gate uses the HARDEST single
off-target (max ipTM / min ipAE over the whole panel — generic helices AND the register-shifted
target), not the panel mean (which is reported only as a secondary number). A real, specific
binder must clear: design ≥ reference AND design − scramble ≥ iptm_gap (0.08) AND
design − WORST(off-target) ≥ off_target_gap (0.15), with ipAE gaps ≥ ipae_gap_A (2.0) and a
surface-biased footprint. If scrambles/off-targets also score high → generic amphipathic docking,
NOT a specific binder (an honest negative — the expected, valuable outcome to report either way).

MULTI-SEED (Gate-B #3): pass --seed to vary the ChaiBackend seed and run this script K times
(each writes its own controls_verdict.json), then fold the K verdicts into per-metric mean +/-
std with xenodesign.eval.controls.aggregate_multiseed([...verdict_paths]) (reports n_runs,
{mean,std,n} per numeric metric, and specific_fraction). Example aggregation::

    from xenodesign.eval.controls import aggregate_multiseed
    agg = aggregate_multiseed(["run_s42/controls_verdict.json",
                               "run_s7/controls_verdict.json",
                               "run_s13/controls_verdict.json"])
    print(agg["design_iptm"], agg["worst_offtarget_iptm"], agg["specific_fraction"])

Run inside the chai 0.6.1 container (PYTHONPATH=/work), e.g. a single seed:
    python scripts/score_controls.py --design MRRELLEALYGAVEEAREKVN --n_scramble 2 --device cuda:0 \
        --seed 42 --binder_alone --out_dir /work/XenoDesign1_local_ref/controls_run/s42
A full multi-seed + register-decoy + binder-alone pass over K=3 seeds:
    for S in 42 7 13; do
      python scripts/score_controls.py --design MRRELLEALYGAVEEAREKVN --n_scramble 2 \
        --device cuda:0 --seed $S --binder_alone \
        --out_dir /work/XenoDesign1_local_ref/controls_run/s$S
    done
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _score_complex(backend, target_seq, binder_seq, out_dir, case, n_steps):
    """Predict [target(L), binder(D)] on Chai and return the interface bundle (raw, no restraint)."""
    from xenodesign.benchmark.case_metrics import case_metrics
    entities = [
        {"type": "protein", "name": "target", "sequence": target_seq, "chirality": "L"},
        {"type": "protein", "name": "binder", "sequence": binder_seq, "chirality": "D"},
    ]
    out_dir = Path(out_dir)
    backend.predict(entities, out_dir, num_diffn_timesteps=n_steps)
    res = case_metrics(case, out_dir / "chai_out")
    m = res.get("metrics", {})
    return {
        "interface_iptm": m.get("interface_iptm"),
        "ipae_mean": m.get("ipae_mean"),
        "ipsae_cut10": m.get("ipsae_cut10"),
        "chai_out": str(out_dir / "chai_out"),
    }


def score_to_verdict(rows, margins, footprint=None):
    """Pure pre-registered verdict math over the control ``rows`` (no GPU/IO).

    ``rows`` is the list of per-case bundles produced by the GPU run, each with at
    least ``label`` (e.g. ``"design"``, ``"scramble1"``, ``"reference"``,
    ``"offtgt:foo"``), ``ok``, ``interface_iptm`` and ``ipae_mean``. ``margins`` is
    a WIN_MARGINS-shaped mapping (``iptm_gap``, ``off_target_gap``, ``ipae_gap_A``).

    **WORST-DECOY is the honest metric** (Gate-B #1). The specificity check uses the
    HARDEST single off-target, not the mean of the panel:

      * ipTM:  ``design_iptm - max(offtarget_iptm) >= off_target_gap`` — the worst decoy is
        the one with the HIGHEST ipTM (the most binder-like off-target).
      * ipAE:  ``min(offtarget_ipae) - design_ipae >= ipae_gap_A`` — the worst decoy here is
        the MOST CONFIDENT off-target (LOWEST ipAE).

    The panel mean (``mean_offtarget_iptm`` / ``mean_offtarget_ipae``) is still reported as a
    secondary number, but the PASS/FAIL gate is the worst decoy. ``worst_offtarget_iptm`` /
    ``worst_offtarget_label`` name the hardest decoy in the verdict.

    A design is SPECIFIC iff every evaluated check passes AND both negative-control
    channels (scramble + off-target) are actually present — a channel that errored
    out (its aggregate is ``None``) must NOT be allowed to count as a silent pass.

    Returns a verdict dict (also embeds the raw aggregates + reasons).
    """
    def g(label, key):
        for r in rows:
            if r.get("label") == label and r.get("ok"):
                return r.get(key)
        return None

    d_it, d_ip = g("design", "interface_iptm"), g("design", "ipae_mean")
    ref_it = g("reference", "interface_iptm")

    scr_its = [r["interface_iptm"] for r in rows
               if r.get("label", "").startswith("scramble") and r.get("ok")
               and r.get("interface_iptm") is not None]
    # off-target ROWS that carry an ipTM (keep label alongside the value so we can name the
    # worst decoy); the register-shifted target decoy (label "offtgt:register_shift") rides
    # in this same channel.
    off_it_rows = [(r["label"], r["interface_iptm"]) for r in rows
                   if r.get("label", "").startswith("offtgt:") and r.get("ok")
                   and r.get("interface_iptm") is not None]
    scr_ips = [r["ipae_mean"] for r in rows
               if r.get("label", "").startswith("scramble") and r.get("ok")
               and r.get("ipae_mean") is not None]
    off_ips = [r["ipae_mean"] for r in rows
               if r.get("label", "").startswith("offtgt:") and r.get("ok")
               and r.get("ipae_mean") is not None]

    max_scr = max(scr_its) if scr_its else None
    off_its = [v for _lbl, v in off_it_rows]
    mean_off = (sum(off_its) / len(off_its)) if off_its else None
    # WORST decoy = the most binder-like off-target = HIGHEST ipTM (hardest to beat).
    if off_it_rows:
        worst_off_label, worst_off = max(off_it_rows, key=lambda t: t[1])
    else:
        worst_off_label, worst_off = None, None
    # Lower ipAE = better, so the most-confident (worst-to-beat) scramble has the LOWEST ipAE.
    min_scr_ipae = min(scr_ips) if scr_ips else None
    mean_off_ipae = (sum(off_ips) / len(off_ips)) if off_ips else None
    # WORST decoy for the ipAE gap = the MOST CONFIDENT off-target = LOWEST ipAE.
    min_off_ipae = min(off_ips) if off_ips else None

    checks = {}
    reasons = []
    if d_it is not None:
        if max_scr is not None:
            checks["beats_scramble_by_margin"] = bool(d_it - max_scr >= margins["iptm_gap"])
        if worst_off is not None:
            # Honest specificity: beat the WORST (hardest) decoy, not the panel mean.
            checks["specific_vs_offtarget"] = bool(d_it - worst_off >= margins["off_target_gap"])
        if ref_it is not None:
            checks["beats_reference"] = bool(d_it >= ref_it)
    # ipAE gaps: the control must be LESS confident (HIGHER ipAE) by >= ipae_gap_A.
    if d_ip is not None:
        if min_scr_ipae is not None:
            checks["ipae_gap_vs_scramble"] = bool(min_scr_ipae - d_ip >= margins["ipae_gap_A"])
        if min_off_ipae is not None:
            # Honest ipAE gap: beat the MOST CONFIDENT (lowest-ipAE) decoy, not the panel mean.
            checks["ipae_gap_vs_offtarget"] = bool(min_off_ipae - d_ip >= margins["ipae_gap_A"])

    # Both negative-control channels must be PRESENT; a dropped channel is not a pass.
    scramble_present = max_scr is not None
    offtarget_present = worst_off is not None
    if not scramble_present:
        reasons.append("incomplete_controls: scramble channel missing")
    if not offtarget_present:
        reasons.append("incomplete_controls: offtarget channel missing")

    specific = bool(checks) and all(checks.values()) and scramble_present and offtarget_present

    return {
        "design_iptm": d_it, "design_ipae": d_ip, "reference_iptm": ref_it,
        "max_scramble_iptm": max_scr,
        "worst_offtarget_iptm": worst_off, "worst_offtarget_label": worst_off_label,
        "mean_offtarget_iptm": mean_off,
        "min_scramble_ipae": min_scr_ipae,
        "min_offtarget_ipae": min_off_ipae, "mean_offtarget_ipae": mean_off_ipae,
        "margins": dict(margins), "checks": checks, "footprint": footprint,
        "scramble_present": scramble_present, "offtarget_present": offtarget_present,
        "reasons": reasons, "SPECIFIC": specific,
    }


def _predict_binder_alone(backend, design, out_dir, n_steps):
    """Predict the design binder ALONE (all-D, no target, no seed, no restraint) — Gate-B #4.

    The STRICT chirality-real check: with no L-target context to template against, does Chai
    still fold the binder as a D-peptide from scratch? The entities dict is a single all-D
    protein chain (chirality 'D' triggers parenthesized-D conversion in ``build_fasta``); the
    sequence is glycine-guarded so the all-D chain stays tokenisable::

        entities = [{"type": "protein", "name": "binder",
                     "sequence": glycine_satisfy_guard(design), "chirality": "D"}]

    Because the binder is the only chain, Chai emits it as chain ``A`` — so the from-scratch
    D-chirality fraction is read off chain 'A' (NOT 'B' as in the complex). We also report the
    CA-geometry helix fraction of that chain when parseable.

    Returns a bundle: ``{label, ok, chirality_real_fraction, gly_fraction, helix_fraction,
    chai_out}`` (``chirality_real_fraction`` = 1 - violation fraction; higher is better).
    """
    from xenodesign.eval.chirality_reality import (
        backbone_chirality_fraction_from_cif, gly_fraction_from_cif,
    )
    from xenodesign.io_spec import glycine_satisfy_guard

    guarded = glycine_satisfy_guard(design)
    entities = [{"type": "protein", "name": "binder", "sequence": guarded, "chirality": "D"}]
    out_dir = Path(out_dir)
    backend.predict(entities, out_dir, num_diffn_timesteps=n_steps)
    chai_out = out_dir / "chai_out"
    from scripts.design_demo import _best_cif_path
    cif = _best_cif_path(chai_out)
    # Binder-alone => the lone chain is 'A'.
    viol = backbone_chirality_fraction_from_cif(cif, chain_name="A", chirality_label="D")
    gly = gly_fraction_from_cif(cif, chain_name="A")
    helix = None
    try:
        from xenodesign.metrics import _parse_cif_ca
        from xenodesign.secondary_structure import helix_fraction
        import numpy as np
        ca = np.array([[x, y, z] for (c, _r, x, y, z, _b) in _parse_cif_ca(cif) if c == "A"],
                      dtype=float)
        if len(ca) >= 4:
            helix = helix_fraction(ca)
    except Exception:  # pragma: no cover (gpu robustness)
        helix = None
    return {
        "label": "binder_alone", "ok": True,
        "chirality_real_fraction": 1.0 - viol,
        "chirality_violation_fraction": viol,
        "gly_fraction": gly,
        "helix_fraction": helix,
        "chai_out": str(chai_out),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="within-Chai specificity controls for an α design")
    p.add_argument("--design", required=True, help="the candidate binder one-letter L sequence")
    p.add_argument("--n_scramble", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42,
                   help="ChaiBackend seed; run K times with different --seed for multi-seed "
                        "stats, then aggregate the verdict JSONs via "
                        "controls.aggregate_multiseed (Gate-B #3)")
    p.add_argument("--binder_alone", action="store_true",
                   help="ALSO predict the design binder alone (all-D, no target, no restraint) "
                        "for the STRICT from-scratch chirality-real check (Gate-B #4)")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args(argv)

    import tempfile
    from xenodesign.backends.chai_backend import ChaiBackend
    from xenodesign.benchmark.cases import get_case
    from xenodesign.eval.controls import (
        WIN_MARGINS, composition_matched_scramble, interface_footprint,
        load_gt_reference_binder, off_target_helices, register_shifted_target_decoy,
    )
    from xenodesign.seed import read_target_sequence

    case = get_case("alpha")
    target_seq = read_target_sequence(case.fasta_path, name="trimer_DL_ABLE_B")
    out_dir = Path(args.out_dir or tempfile.mkdtemp(prefix="xd_controls_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    design = args.design.upper()

    print(f"\n{'='*78}\nWithin-Chai specificity controls (UNRESTRAINED)\n{'='*78}")
    print(f"  design   : {design} ({len(design)} aa)")
    print(f"  target   : {len(target_seq)} aa L-HLH | device {args.device} | {args.n_steps} steps")
    print(f"  seed     : {args.seed} | binder_alone {args.binder_alone}")
    print(f"  margins  : {WIN_MARGINS}\n  out_dir  : {out_dir}\n{'='*78}\n")

    t0 = time.time()
    backend = ChaiBackend(device=args.device, seed=args.seed)
    rows = []

    def run(label, target, binder, sub):
        try:
            r = _score_complex(backend, target, binder, out_dir / sub, case, args.n_steps)
            r.update(label=label, binder=binder, ok=True)
        except Exception as exc:  # pragma: no cover (gpu robustness)
            r = {"label": label, "binder": binder, "ok": False, "error": repr(exc)}
        rows.append(r)
        it = r.get("interface_iptm"); ip = r.get("ipae_mean"); sa = r.get("ipsae_cut10")
        print(f"  [{label:>16}] ipTM {it if it is None else round(it,4)}  "
              f"ipAE {ip if ip is None else round(ip,3)}  ipsae10 {sa if sa is None else round(sa,3)}")
        return r

    design_row = run("design", target_seq, design, "design")
    for k in range(args.n_scramble):
        run(f"scramble{k+1}", target_seq, composition_matched_scramble(design, rng_seed=k + 1), f"scramble{k+1}")
    try:
        run("reference", target_seq, load_gt_reference_binder(), "reference")
    except Exception as exc:
        print(f"  [reference] could not load GT reference: {exc!r}")
    # Generic off-target helices: design vs each unrelated helix-as-target.
    for name, helix in off_target_helices():
        run(f"offtgt:{name}", helix, design, f"offtgt_{name}")
    # Register-shifted TARGET decoy (Gate-B #2): the design vs a circular rotation of the REAL
    # target (composition-identical, register-shifted) — enters the worst-decoy panel so a
    # register-specific binder must NOT bind it.
    rs_name, rs_seq = register_shifted_target_decoy(target_seq, shift=3)
    run(f"offtgt:{rs_name}", rs_seq, design, f"offtgt_{rs_name}")

    # Binder-alone cold-start (Gate-B #4): STRICT from-scratch chirality-real check.
    if args.binder_alone:
        try:
            ba = _predict_binder_alone(backend, design, out_dir / "binder_alone", args.n_steps)
        except Exception as exc:  # pragma: no cover (gpu robustness)
            ba = {"label": "binder_alone", "ok": False, "error": repr(exc)}
        rows.append(ba)
        print(f"  [    binder_alone] chirality_real {ba.get('chirality_real_fraction')}  "
              f"gly_frac {ba.get('gly_fraction')}  helix {ba.get('helix_fraction')}")

    # Footprint of the design vs real target (which target residues, surface vs core).
    footprint = None
    if design_row.get("ok"):
        try:
            from scripts.design_demo import _best_cif_path
            cif = _best_cif_path(Path(design_row["chai_out"]))
            footprint = interface_footprint(cif, target_chain="A", binder_chain="B")
            print(f"\n  footprint: {footprint.get('n_contacted')}/{footprint.get('n_target')} target "
                  f"residues contacted; surface_fraction {round(footprint.get('surface_fraction', 0.0),3)}")
        except Exception as exc:
            print(f"  footprint failed: {exc!r}")

    # ── Pre-registered verdict (pure math, unit-tested off-GPU) ───────────────
    verdict = score_to_verdict(rows, WIN_MARGINS, footprint=footprint)
    verdict["wall_time_s"] = time.time() - t0
    verdict["seed"] = args.seed
    d_it, max_scr = verdict["design_iptm"], verdict["max_scramble_iptm"]
    worst_off, worst_lbl = verdict["worst_offtarget_iptm"], verdict["worst_offtarget_label"]
    mean_off, ref_it = verdict["mean_offtarget_iptm"], verdict["reference_iptm"]

    (out_dir / "controls_verdict.json").write_text(json.dumps({"verdict": verdict, "rows": rows}, indent=2))
    print(f"\n{'='*78}\nVERDICT: SPECIFIC={verdict['SPECIFIC']}  checks={verdict['checks']}")
    if verdict["reasons"]:
        print(f"  reasons: {verdict['reasons']}")
    print(f"  design {d_it} vs max-scramble {max_scr} vs WORST-offtarget {worst_off} "
          f"({worst_lbl}) [mean {mean_off}] vs reference {ref_it}")
    print(f"  → {out_dir/'controls_verdict.json'}  ({verdict['wall_time_s']/60:.1f} min)\n{'='*78}")
    return verdict


if __name__ == "__main__":
    main()
    sys.exit(0)
