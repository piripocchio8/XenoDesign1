"""Extract per-design-loop trajectories of ipTM AND obj(T20) over design cycles, plus the winner's
final ipTM+obj, for both GREEDY and BEAM alpha-design runs.

For each run we emit an ordered list of epochs `[{epoch, iptm, obj, chirality?}]` and a `winner`
`{iptm, obj, dir}`, so the two optimizers can be compared on the SAME mixed objective (T20 / ADR-014)
rather than only on Chai ipTM (which under-rates heterochiral D-binders).

Run-type handling
-----------------
GREEDY (alpha_result.json with trajectory[]):
  - per-iter ipTM + chirality come straight from trajectory[] (cheap, already recorded);
  - obj(T20) per iter is RE-SCORED from that iter's CIFs:
        runs/<r>/loop/iter_NNN/chai_out  ->  contrastive_rank._score_dir(dir, na) -> obj
  - epoch == iter index.

BEAM (alpha_beam_result.json, NO trajectory[]):
  - reconstruct per-CYCLE best from the per-state CIFs under
        runs/<r>/beam/cycle_NNN/node_*/chai_out
    for each cycle take the best node (by obj from _score_dir; ipTM reported alongside).
  - anneal steps (runs/<r>/anneal/anneal_*/iter_*/chai_out) are treated as CONTINUED epochs
    AFTER the beam cycles: each anneal_* step contributes one epoch = its best iter (by obj).

WINNER (both):
  - the selected/best dir is re-scored with _score_dir -> final (iptm, obj).
  - greedy: trajectory selected_iter -> loop/iter_NNN.
  - beam: the dir (over all beam nodes + anneal iters) whose best-cif ipTM matches selected_iptm
    (falls back to global-max obj dir if no result json / no match).

T20 re-scoring (gemmi + freesasa) MUST run where both import — the gradio_design Docker (CPU) or a
host env with gemmi+freesasa. See `how_to_run` at the bottom. obj degrades gracefully (bsa=None) if
freesasa is missing, but DON'T trust obj in that case.

CLI:
  python scripts/analyze_trajectories.py \
      --runs runs/ab_greedy runs/greedy30b runs/ab_beam2 runs/beam_deep \
      --out runs/logs/trajectories.json [--na 21] [--selfcheck]

Self-check (no Chai, no GPU, no freesasa needed):
  python scripts/analyze_trajectories.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import contrastive_rank as cr  # noqa: E402  (reuse _score_dir / _best_cif)


# --------------------------------------------------------------------------------------
# cheap ipTM straight from a chai_out (no gemmi/freesasa) -- used to find the beam winner
# and as a sanity cross-check against _score_dir's iptm.
# --------------------------------------------------------------------------------------
def _cheap_best_iptm(chai_dir):
    chai = Path(chai_dir)
    sf = sorted(chai.glob("scores.model_idx_*.npz"))
    if not sf:
        return None
    bi = max(sf, key=lambda f: float(np.load(f)["aggregate_score"].reshape(-1)[0]))
    z = np.load(bi)
    if "per_chain_pair_iptm" not in z:
        return None
    m = np.asarray(z["per_chain_pair_iptm"])
    m = m.reshape(int(m.size ** 0.5), -1)
    return round(float(max(m[0, 1], m[1, 0])), 4)


def _score(dir_path, na):
    """_score_dir wrapper that never raises: returns {obj,iptm,bsa,sc} or None."""
    try:
        return cr._score_dir(str(dir_path), na)
    except Exception as e:  # missing CIF, freesasa explosion on a degenerate model, ...
        return {"error": str(e)[:160]}


def _iter_dirs(parent, pattern="iter_*"):
    return sorted(p for p in Path(parent).glob(pattern) if (p / "chai_out").is_dir())


# --------------------------------------------------------------------------------------
# GREEDY
# --------------------------------------------------------------------------------------
def analyze_greedy(run_dir, na):
    run = Path(run_dir)
    res = json.loads((run / "alpha_result.json").read_text())
    traj = res.get("trajectory", [])
    epochs = []
    for t in traj:
        i = t["iter"]
        d = run / "loop" / f"iter_{i:03d}"
        sc = _score(d, na) if (d / "chai_out").is_dir() else None
        obj = sc.get("obj") if sc else None
        rescored_iptm = sc.get("iptm") if sc else None
        epochs.append({
            "epoch": i,
            "iptm": round(float(t["iptm"]), 4) if t.get("iptm") is not None else rescored_iptm,
            "obj": obj,
            "chirality": t.get("chirality"),
            "rescored_iptm": rescored_iptm,
            "dir": str(d),
        })
    # winner = selected_iter
    sel = res.get("selected_iter")
    wdir = run / "loop" / f"iter_{sel:03d}" if sel is not None else None
    winner = None
    if wdir and (wdir / "chai_out").is_dir():
        wsc = _score(wdir, na)
        winner = {"iptm": wsc.get("iptm") if wsc else None,
                  "obj": wsc.get("obj") if wsc else None,
                  "selected_iter": sel, "dir": str(wdir)}
    return {"type": "greedy", "run": run.name, "n_epochs": len(epochs),
            "selected_iptm_reported": res.get("selected_iptm"),
            "epochs": epochs, "winner": winner}


# --------------------------------------------------------------------------------------
# BEAM
# --------------------------------------------------------------------------------------
def _best_node_of_cycle(cycle_dir, na):
    """Best node in one beam cycle by obj (ties broken by iptm). Returns an epoch dict."""
    best = None
    for node in sorted(Path(cycle_dir).glob("node_*")):
        if not (node / "chai_out").is_dir():
            continue
        sc = _score(node, na)
        if not sc or sc.get("obj") is None:
            continue
        key = (sc["obj"], sc.get("iptm") or 0.0)
        if best is None or key > best["_key"]:
            best = {"_key": key, "iptm": sc.get("iptm"), "obj": sc.get("obj"), "dir": str(node)}
    if best:
        best.pop("_key")
    return best


def _best_iter_of_anneal(anneal_step_dir, na):
    """Best iter in one anneal step by obj. Returns an epoch dict."""
    best = None
    for it in _iter_dirs(anneal_step_dir):
        sc = _score(it, na)
        if not sc or sc.get("obj") is None:
            continue
        key = (sc["obj"], sc.get("iptm") or 0.0)
        if best is None or key > best["_key"]:
            best = {"_key": key, "iptm": sc.get("iptm"), "obj": sc.get("obj"), "dir": str(it)}
    if best:
        best.pop("_key")
    return best


def analyze_beam(run_dir, na):
    run = Path(run_dir)
    rj = run / "alpha_beam_result.json"
    res = json.loads(rj.read_text()) if rj.exists() else {}
    epochs = []
    ep = 0
    # 1) beam cycles -> one epoch per cycle (best node)
    cycles = sorted((run / "beam").glob("cycle_*")) if (run / "beam").is_dir() else []
    for c in cycles:
        b = _best_node_of_cycle(c, na)
        if b is None:
            continue
        epochs.append({"epoch": ep, "phase": "beam", "cycle": c.name,
                       "iptm": b["iptm"], "obj": b["obj"], "dir": b["dir"]})
        ep += 1
    # 2) anneal steps -> continued epochs (best iter of each step)
    annesteps = sorted((run / "anneal").glob("anneal_*")) if (run / "anneal").is_dir() else []
    for a in annesteps:
        b = _best_iter_of_anneal(a, na)
        if b is None:
            continue
        epochs.append({"epoch": ep, "phase": "anneal", "step": a.name,
                       "iptm": b["iptm"], "obj": b["obj"], "dir": b["dir"]})
        ep += 1
    # winner: match selected_iptm across all scored dirs (beam nodes + anneal iters);
    # fall back to global-max obj.
    winner = _beam_winner(run, res, na)
    return {"type": "beam", "run": run.name, "n_epochs": len(epochs),
            "n_beam_cycles": len(cycles), "n_anneal_steps": len(annesteps),
            "selected_iptm_reported": res.get("selected_iptm"),
            "epochs": epochs, "winner": winner}


def _all_scored_dirs(run):
    dirs = []
    for node in sorted((run / "beam").glob("cycle_*/node_*")):
        if (node / "chai_out").is_dir():
            dirs.append(node)
    for it in sorted((run / "anneal").glob("anneal_*/iter_*")):
        if (it / "chai_out").is_dir():
            dirs.append(it)
    return dirs


def _beam_winner(run, res, na):
    dirs = _all_scored_dirs(run)
    if not dirs:
        return None
    sel_iptm = res.get("selected_iptm")
    wdir = None
    if sel_iptm is not None:
        # cheap ipTM match (no freesasa needed) -- robust + fast over many dirs
        best = None
        for d in dirs:
            it = _cheap_best_iptm(d / "chai_out")
            if it is None:
                continue
            diff = abs(it - sel_iptm)
            if best is None or diff < best[0]:
                best = (diff, d)
        if best and best[0] < 5e-3:   # ~exact match to the reported selected ipTM
            wdir = best[1]
    if wdir is None:  # fall back: global-max obj (requires scoring all -> heavier)
        best = None
        for d in dirs:
            sc = _score(d, na)
            if not sc or sc.get("obj") is None:
                continue
            if best is None or sc["obj"] > best[0]:
                best = (sc["obj"], d, sc)
        if best:
            return {"iptm": best[2].get("iptm"), "obj": best[2].get("obj"),
                    "dir": str(best[1]), "match": "global_max_obj"}
        return None
    wsc = _score(wdir, na)
    return {"iptm": wsc.get("iptm") if wsc else None,
            "obj": wsc.get("obj") if wsc else None,
            "dir": str(wdir), "match": "selected_iptm"}


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
def analyze_run(run_dir, na):
    run = Path(run_dir)
    if (run / "alpha_result.json").exists() and \
            json.loads((run / "alpha_result.json").read_text()).get("trajectory"):
        return analyze_greedy(run, na)
    if (run / "alpha_beam_result.json").exists() or (run / "beam").is_dir():
        return analyze_beam(run, na)
    raise ValueError(f"{run}: neither greedy trajectory nor beam layout found")


# --------------------------------------------------------------------------------------
# self-check on synthetic input -- no Chai/GPU/freesasa needed.
# Builds a fake greedy + fake beam run on disk with stub scores npz, monkeypatches
# _score_dir to a deterministic obj=f(iptm), and asserts the trajectory/winner shape.
# --------------------------------------------------------------------------------------
def _selfcheck():
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="trajcheck_"))

    def _mk_chai(dir_path, iptm, agg=None):
        chai = Path(dir_path) / "chai_out"
        chai.mkdir(parents=True, exist_ok=True)
        agg = iptm if agg is None else agg
        m = np.array([[0.0, iptm], [iptm, 0.0]], float)
        np.savez(chai / "scores.model_idx_0.npz",
                 aggregate_score=np.array([agg]),
                 per_chain_pair_iptm=m, iptm=np.array([iptm]),
                 ptm=np.array([0.9]))
        # a dummy cif so _best_cif resolves (real obj is monkeypatched, so contents don't matter)
        (chai / "pred.model_idx_0.cif").write_text("data_x\n")

    # fake greedy: 3 iters, selected_iter=2
    g = tmp / "fake_greedy"
    for i, ipt in enumerate([0.40, 0.55, 0.70]):
        _mk_chai(g / "loop" / f"iter_{i:03d}", ipt)
    (g / "alpha_result.json").write_text(json.dumps({
        "selected_iter": 2, "selected_iptm": 0.70,
        "trajectory": [{"iter": i, "iptm": ipt, "chirality": 0.0}
                       for i, ipt in enumerate([0.40, 0.55, 0.70])],
    }))

    # fake beam: 2 cycles (cycle_000 1 node, cycle_001 2 nodes) + 1 anneal step (2 iters)
    b = tmp / "fake_beam"
    _mk_chai(b / "beam" / "cycle_000" / "node_00000", 0.45)
    _mk_chai(b / "beam" / "cycle_001" / "node_00001", 0.50)
    _mk_chai(b / "beam" / "cycle_001" / "node_00002", 0.62)   # best in cycle_001
    _mk_chai(b / "anneal" / "anneal_00" / "iter_000", 0.66)
    _mk_chai(b / "anneal" / "anneal_00" / "iter_001", 0.75)   # best anneal iter & overall
    (b / "alpha_beam_result.json").write_text(json.dumps({"selected_iptm": 0.75}))

    # monkeypatch _score_dir: deterministic obj from the cheap ipTM (no freesasa needed)
    orig = cr._score_dir

    def fake_score_dir(d, na):
        it = _cheap_best_iptm(Path(d) / "chai_out")
        return {"obj": round((it or 0) * 0.9, 3), "iptm": it, "bsa": 999.0, "sc": 0.5}

    cr._score_dir = fake_score_dir
    try:
        gr = analyze_greedy(g, 21)
        br = analyze_beam(b, 21)
    finally:
        cr._score_dir = orig

    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + " " + msg)
        ok = ok and cond

    # greedy assertions
    chk(gr["n_epochs"] == 3, "greedy has 3 epochs")
    chk([e["epoch"] for e in gr["epochs"]] == [0, 1, 2], "greedy epochs ordered 0,1,2")
    chk(abs(gr["epochs"][2]["iptm"] - 0.70) < 1e-6, "greedy epoch2 iptm==0.70")
    chk(abs(gr["epochs"][2]["obj"] - 0.63) < 1e-6, "greedy epoch2 obj==0.63 (=0.70*0.9)")
    chk(gr["winner"]["selected_iter"] == 2 and abs(gr["winner"]["obj"] - 0.63) < 1e-6,
        "greedy winner = iter2, obj 0.63")

    # beam assertions
    chk(br["n_beam_cycles"] == 2 and br["n_anneal_steps"] == 1, "beam: 2 cycles + 1 anneal step")
    chk(br["n_epochs"] == 3, "beam epochs = 2 cycles + 1 anneal = 3")
    chk(br["epochs"][0]["phase"] == "beam" and br["epochs"][-1]["phase"] == "anneal",
        "beam epochs ordered beam->anneal")
    chk(abs(br["epochs"][1]["iptm"] - 0.62) < 1e-6, "beam cycle_001 best node iptm==0.62")
    chk(abs(br["epochs"][2]["iptm"] - 0.75) < 1e-6, "beam anneal-step best iter iptm==0.75")
    chk(br["winner"]["match"] == "selected_iptm" and abs(br["winner"]["iptm"] - 0.75) < 1e-6,
        "beam winner matched by selected_iptm == 0.75")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nSELFCHECK:", "OK" if ok else "FAILED")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", help="run dirs (greedy and/or beam)")
    p.add_argument("--na", type=int, default=21,
                   help="chain-A token count for ipAE/ipSAE split passed to _score_dir (default 21, "
                        "matches contrastive_rank). NOTE: in these runs chain A is the 41-res target; "
                        "iptm/bsa/sc are A/B-symmetric, only ipsae is mildly na-sensitive.")
    p.add_argument("--out", default=None, help="write per-run trajectory+winner JSON here")
    p.add_argument("--selfcheck", action="store_true", help="run synthetic self-check and exit")
    a = p.parse_args(argv)

    if a.selfcheck:
        return 0 if _selfcheck() else 1
    if not a.runs:
        p.error("--runs is required unless --selfcheck")

    out = {"na": a.na, "runs": {}}
    for r in a.runs:
        print(f"[analyze] {r} ...", file=sys.stderr)
        res = analyze_run(r, a.na)
        out["runs"][res["run"]] = res
        w = res.get("winner") or {}
        print(f"  {res['type']:6} epochs={res['n_epochs']:>2} "
              f"winner iptm={w.get('iptm')} obj={w.get('obj')}", file=sys.stderr)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"[analyze] wrote {a.out}", file=sys.stderr)
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
