"""T01 CONTRASTIVE SELECTOR — population-select register-specific D-binders by a
replicate-averaged contrastive margin (ADR-019 + ADR-021).

Given a POOL of fully-formed designs, for each design we build the REAL complex plus
binder-register-shift {3,4,7} decoys (circular shift of the BINDER; the L-target is held
FIXED — ADR-011/T18). Each (design, kind) complex is predicted R>=3 times with distinct
seeds and scored with the mixed parity-aware objective (T20) via contrastive_rank._score_dir.
We then average the objective over reps and compute:

    margin = mean_R(real_obj) − max_over_shifts[ mean_R(shift_obj) ]   (worst-case decoy, ADR-008)

with its std propagated over reps, and SELECT designs whose margin clears a NOISE-AWARE
threshold:  margin − k·margin_std > 0  (default k=1) — NOT the bare sign test, because the
single-run ipTM noise (std ~0.067) exceeds the GT margin (+0.087), so a single-shot margin
is ~50% noise (ADR-021).

This is the ADR-019 "generate a large pool and population-select by margin" alternative to
the GATED in-loop contrastive design; it does NOT rebuild decoy generation or the objective.

REUSE: contrastive_decoys (decoy/item build), predict_batch + predict_complex (GPU predict,
unchanged), contrastive_rank._score_dir (T20-scored panel per dir), mixed_objective.score.

The replicate-averaging + margin + noise-aware select is factored into PURE functions
(binder_shifts, select_by_margin) so it is importable and unit-tested on CPU with the
predictor/scorer mocked (tests/test_select_contrastive.py). GPU prediction is NOT run here.

CLI (GPU box; chai 0.6.1 container):
  python scripts/select_contrastive.py \
      --designs <d1> <d2> ... --target_fasta <t.fasta> \
      --out_root <root> --json <decoys.json> --out <ranked.json> \
      --reps 3 --seeds 42,43,44 --shifts 3,4,7 --na 21 \
      --threshold_k 1.0 --margin_axis obj --device cuda:0

Convention: D-binder seqs are reported lowercase (Gly=G). The replicate-averaged contrastive
margin is the selector — never absolute pLDDT/pTM.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SHIFTS_DEFAULT = (3, 4, 7)


# ── PURE: binder register shifts (shift the BINDER, target fixed) ──────────────

def binder_shifts(binder: str, shifts=SHIFTS_DEFAULT) -> dict[int, str]:
    """Return {shift: circularly-shifted binder} for each shift.

    Mirrors contrastive_decoys.py:48-50 — bs = b[s:] + b[:s]. The TARGET is never
    shifted; only the binder rotates (ADR-011 register specificity on the binder).
    """
    return {s: binder[s:] + binder[:s] for s in shifts}


# ── PURE: replicate-average + contrastive margin + noise-aware select ──────────

def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = statistics.fmean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def select_by_margin(
    per_dir_scores: dict,
    reps: int = 3,
    k: float = 1.0,
    shifts=SHIFTS_DEFAULT,
    axis: str = "obj",
) -> list[dict]:
    """Replicate-average per-(design,kind) scores, compute the contrastive margin
    ± std over reps, and population-select with a noise-aware threshold.

    Parameters
    ----------
    per_dir_scores : dict
        {(design, kind, rep): {"obj": float, "iptm": float}} where kind is
        "real" or "shift<n>". Produced on GPU by scoring each replicate dir with
        contrastive_rank._score_dir; MOCKED in CPU tests.
    reps : int
        Expected replicate count R (>=3 per ADR-021). Used only to flag rows whose
        available reps < R as incomplete; averaging always uses whatever is present.
    k : float
        Noise-aware threshold factor: select iff (margin − k·margin_std) > 0.
        k=0 collapses to the bare sign test (margin > 0).
    shifts : iterable[int]
        Register shifts applied to the binder (default {3,4,7}).
    axis : "obj" | "iptm"
        Which scored field drives the margin/selection. Default "obj" (T20 mixed
        objective, ADR-019 primary). The other axis is still reported.

    Returns
    -------
    list[dict]
        One row per design with a real + >=1 shift, sorted by margin descending.
        Designs missing the real kind or all shifts are dropped (flagged, not crashed).
    """
    shift_kinds = [f"shift{s}" for s in shifts]

    # group rep values per (design, kind, field)
    designs: dict = {}
    for (design, kind, _rep), sc in per_dir_scores.items():
        d = designs.setdefault(design, {})
        kd = d.setdefault(kind, {"obj": [], "iptm": []})
        kd["obj"].append(sc.get("obj"))
        kd["iptm"].append(sc.get("iptm"))

    def _field_margin(kinds, field, present_shifts):
        """mean±std(real) and worst-case (max-mean) shift margin±std on one field."""
        real_mean, real_std = _mean_std([v for v in kinds["real"][field] if v is not None])
        stats = {}
        for sk in present_shifts:
            m, s = _mean_std([v for v in kinds[sk][field] if v is not None])
            stats[sk] = (m, s)
        worst_kind = max(stats, key=lambda sk: stats[sk][0])  # ADR-008: worst = highest-binding shift
        w_mean, w_std = stats[worst_kind]
        margin = real_mean - w_mean
        margin_std = math.sqrt(real_std ** 2 + w_std ** 2)
        return {"real_mean": real_mean, "real_std": real_std, "worst_kind": worst_kind,
                "worst_mean": w_mean, "worst_std": w_std, "margin": margin,
                "margin_std": margin_std, "stats": stats}

    rows: list[dict] = []
    for design, kinds in designs.items():
        if "real" not in kinds:
            continue
        present_shifts = [sk for sk in shift_kinds if sk in kinds]
        if not present_shifts:
            continue

        incomplete = len([v for v in kinds["real"]["obj"] if v is not None]) < reps
        for sk in present_shifts:
            if len([v for v in kinds[sk]["obj"] if v is not None]) < reps:
                incomplete = True

        obj = _field_margin(kinds, "obj", present_shifts)
        ipt = _field_margin(kinds, "iptm", present_shifts)
        active = obj if axis == "obj" else ipt   # the SELECTOR field

        selected = (active["margin"] - k * active["margin_std"]) > 0

        per_shift = {
            sk: {"obj_mean": obj["stats"][sk][0], "obj_std": obj["stats"][sk][1],
                 "iptm_mean": ipt["stats"][sk][0], "iptm_std": ipt["stats"][sk][1]}
            for sk in present_shifts
        }

        row = {
            "design": design,
            "reps": reps,
            "incomplete": incomplete,
            "real_obj_mean": active["real_mean"],
            "real_obj_std": active["real_std"],
            "real_iptm_mean": ipt["real_mean"],
            "real_iptm_std": ipt["real_std"],
            "worst_shift": active["worst_kind"],
            "worst_shift_obj_mean": active["worst_mean"],
            "worst_shift_obj_std": active["worst_std"],
            # obj_margin = the active SELECTOR margin (obj by default; iptm if axis=iptm)
            "obj_margin": active["margin"],
            "obj_margin_std": active["margin_std"],
            # iptm_margin = the iptm-field margin, always reported (ADR-019 BAR)
            "iptm_margin": ipt["margin"],
            "iptm_margin_std": ipt["margin_std"],
            "selected": selected,
            "axis": axis,
            "k": k,
            "per_shift": per_shift,
        }
        rows.append(row)

    rows.sort(key=lambda r: -r["obj_margin"])
    return rows


# ── GPU orchestration (NOT run here; CPU-mocked in tests) ─────────────────────

def _build_decoy_items(designs, target_fasta, target_record, out_root, shifts,
                       seq_key="selected_l_seq"):  # pragma: no cover (io)
    """Reuse contrastive_decoys to build the base predict_batch item list (real + shifts).

    Each design dir must contain alpha_result.json with the binder under `seq_key`.
    Returns the flat item list (one shift0 'real' + one per shift per design).
    """
    import contrastive_decoys as cd

    tgt = cd._target_seq(target_fasta, record=target_record) if target_record \
        else cd._target_seq(target_fasta)
    items = []
    for d in designs:
        res = json.load(open(Path(d) / "alpha_result.json"))
        b = res[seq_key]
        tag = Path(d).name
        items.append({"name": f"{tag}__real", "design": tag, "kind": "real", "shift": 0,
                      "seq_a": b, "chir_a": "D", "seq_b": tgt, "chir_b": "L",
                      "out_dir": f"{out_root}/{tag}/real"})
        for s, bs in binder_shifts(b, shifts).items():
            items.append({"name": f"{tag}__shift{s}", "design": tag, "kind": "shift", "shift": s,
                          "seq_a": bs, "chir_a": "D", "seq_b": tgt, "chir_b": "L",
                          "out_dir": f"{out_root}/{tag}/shift{s}"})
    return items, tgt


def _rep_items(base_items, out_root, rep):
    """Rewrite each base item's out_dir to <out_root>/rep{r}/<tag>/<kind> (matches _score_pool).

    base out_dir is <out_root>/<tag>/<kind>; strip the FULL out_root prefix (it is multi-level —
    the old split("/",1) only stripped ONE component and re-prepended out_root, doubling the path)
    before inserting rep{r}. Chai 0.6.1 requires out_dir/chai_out empty/non-existent, so each rep
    gets its own dir. Pure string logic (testable; the doubled-path bug escaped because this was
    wrongly marked no-cover).
    """
    root = out_root.rstrip("/")
    out = []
    for it in base_items:
        sub = it["out_dir"]
        if sub.startswith(root + "/"):
            sub = sub[len(root) + 1:]
        ni = dict(it)
        ni["out_dir"] = f"{root}/rep{rep}/{sub}"
        out.append(ni)
    return out


def _score_pool(designs, out_root, reps, na, shifts):  # pragma: no cover (gpu/io)
    """Score every (design, kind, rep) replicate dir with contrastive_rank._score_dir.

    Returns {(design, kind, rep): {"obj", "iptm"}} ready for select_by_margin.
    Drops dirs that aren't scorable (missing chai_out) — flagged via incomplete in
    select_by_margin (fewer reps than `reps`).
    """
    from contrastive_rank import _score_dir

    shift_kinds = ["real"] + [f"shift{s}" for s in shifts]
    out = {}
    for d in designs:
        tag = Path(d).name
        for kind in shift_kinds:
            for r in range(reps):
                rep_dir = Path(out_root) / f"rep{r}" / tag / kind
                sc = _score_dir(str(rep_dir), na)
                if sc is None:
                    continue
                out[(tag, kind, r)] = {"obj": sc["obj"], "iptm": sc.get("iptm")}
    return out


def _print_table(rows, k, axis):
    print(f"\n{'design':18} {'real±std':>14} {'worstShift':>10} "
          f"{'OBJmargin±std':>18} {'iptm_m':>7} {'sel':>4}")
    print("─" * 78)
    for r in rows:
        sel = "Y" if r["selected"] else "."
        print(f"{r['design']:18} "
              f"{r['real_obj_mean']:.3f}±{r['real_obj_std']:.3f}  "
              f"{r['worst_shift']:>10} "
              f"{r['obj_margin']:+.3f}±{r['obj_margin_std']:.3f}   "
              f"{r['iptm_margin']:+.3f}  {sel:>4}")
    n_sel = sum(r["selected"] for r in rows)
    print(f"\nselected (margin − {k}·σ > 0, axis={axis}): {n_sel}/{len(rows)}")


def main(argv=None):  # pragma: no cover (cli/gpu)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--designs", nargs="+", required=True,
                   help="pool design dirs, each containing alpha_result.json")
    p.add_argument("--target_fasta", required=True)
    p.add_argument("--target_record", default=None,
                   help="FASTA record name for the L-target (default: contrastive_decoys default)")
    p.add_argument("--seq_key", default="selected_l_seq",
                   help="alpha_result.json key holding the binder letters")
    p.add_argument("--out_root", default="XenoDesign1_local_ref/contrastive_select")
    p.add_argument("--json", required=True, help="write the decoy item list here")
    p.add_argument("--out", default=None, help="write ranked rows JSON here")
    p.add_argument("--reps", type=int, default=3, help="replicates per (design,shift) (ADR-021 >=3)")
    p.add_argument("--seeds", default=None, help="comma seeds, one per rep (default base_seed..+reps)")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--shifts", default="3,4,7")
    p.add_argument("--na", type=int, default=21, help="binder token count for ipAE/ipSAE split")
    p.add_argument("--threshold_k", type=float, default=1.0,
                   help="noise-aware: select iff margin − k·std > 0 (k=0 = bare sign test)")
    p.add_argument("--margin_axis", choices=["obj", "iptm"], default="obj",
                   help="margin axis: T20 mixed objective (default) or raw ipTM")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(argv)

    shifts = tuple(int(s) for s in a.shifts.split(","))
    seeds = ([int(s) for s in a.seeds.split(",")] if a.seeds
             else [a.base_seed + r for r in range(a.reps)])
    if len(seeds) < a.reps:
        raise SystemExit(f"need >= {a.reps} seeds, got {len(seeds)}")

    # ensure output dirs exist (driver writes decoys.json / ranked.json / per-rep predict dirs)
    Path(a.out_root).mkdir(parents=True, exist_ok=True)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    # 1. POOL -> decoy items (real + binder shifts {3,4,7}; target fixed)
    base_items, tgt = _build_decoy_items(a.designs, a.target_fasta, a.target_record,
                                         a.out_root, shifts, a.seq_key)
    Path(a.json).write_text(json.dumps(base_items, indent=2))
    print(f"target ({len(tgt)} aa): {tgt}")
    print(f"{len(base_items)} base complexes for {len(a.designs)} designs "
          f"(real + shifts {list(shifts)}) -> {a.json}")

    # 2. R replicate predictions (GPU; one predict_batch run per rep, distinct seed + dir)
    from predict_complex import run as predict_run
    for r in range(a.reps):
        rep_items = _rep_items(base_items, a.out_root, r)
        print(f"\n[rep {r}] seed={seeds[r]} — {len(rep_items)} predictions")
        for it in rep_items:
            try:
                predict_run(it["seq_a"], it["chir_a"], it["seq_b"], it["chir_b"],
                            3, 200, seeds[r], a.device, it["out_dir"])
            except Exception as ex:
                print(f"    {it['name']} rep{r} FAILED: {ex}", flush=True)

    # 3. replicate-averaged scoring + 4/5. margin ± std + noise-aware select
    scores = _score_pool(a.designs, a.out_root, a.reps, a.na, shifts)
    rows = select_by_margin(scores, reps=a.reps, k=a.threshold_k,
                            shifts=shifts, axis=a.margin_axis)
    rows = [{**row, "seeds": seeds[:a.reps]} for row in rows]

    _print_table(rows, a.threshold_k, a.margin_axis)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nranked rows -> {a.out}")
    return rows


if __name__ == "__main__":  # pragma: no cover
    main()
