"""Backend-agnostic inverse-folding SEQUENCE-quality analyzer (pure CPU; no GPU, no Chai).

Reads a glob of ``alpha_result.json`` files (the design_alpha output contract: each has a
``trajectory`` list of per-iter entries with ``l_seq``/``d_fasta``/``iptm``/``chirality`` plus the
top-level ``selected_l_seq``/``selected_iptm``) and scores the *sequence* — not the structure — of
every WINNER and every per-iter candidate. It answers "are the sequences the inverse-folder emits
actually good?", independent of which structure backend produced them, so the same script later
scores CARBonAra runs (pass ``--backend carbonara`` as a label).

For each sequence we report:

  * Ala%, Gly%, and the combined Ala+Gly fraction — the anti-poly-Ala readout. The composition
    floor ``_COMP_MAX_ALA_GLY_FRAC`` (0.40) is imported READ-ONLY from ``scripts.design_alpha`` so
    this analyzer can never drift from the design driver's veto; ``ala_gly_over_floor`` flags a seq.
  * Normalized Shannon entropy ``H / log2(20)`` in [0, 1] — REUSES
    :func:`xenodesign.scorer.sequence_complexity` (scorer.py:60-75), the same diversity measure the
    design driver re-ranks on.
  * Heptad / coiled-coil register quality: an ``a b c d e f g`` heptad is assigned along the helix
    (phaseable via ``--heptad_start``) using :func:`xenodesign.eval.controls.heptad_register`, and we
    score hydrophobic (A/I/L/M/F/V) occupancy at the buried a/d core vs the b/c/e/f/g surface. The
    headline ``heptad_match_fraction`` is the fraction of a/d core positions that ARE hydrophobic —
    a real amphipathic helix packs a hydrophobic seam at a/d, a poly-Ala blob or a polar helix does
    not. We also report the surface-hydrophobic fraction (lower is better) and a
    ``heptad_amphipathy`` = core_hydrophobic_frac - surface_hydrophobic_frac contrast.

Per RUN (not per sequence) we also report the chirality-clean rate = fraction of trajectory steps
with ``chirality <= --chir_max`` (default 0.10) — the survivorship-honest D-purity readout.

Sequences are reported in the project D-peptide convention: chiral D-residues LOWERCASE, Gly always 'G'.
The ``l_seq`` field stores L-letter residue *identities* of the designed all-D peptide, so for the α
(all-D) case the reported form is the whole string lowercased with Gly kept as 'G'. A ``--chirality``
override lets a future mixed/L backend report uppercase instead.

Run (CPU):
    PYTHONPATH=$PWD python scripts/analyze_backend_seq_quality.py \
        "XenoDesign1_local_ref/campaign/seed_*/alpha_result.json" --backend chai
    # multiple globs / explicit files both work; --out writes the JSON summary to a file too.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# Hydrophobic core alphabet for the coiled-coil a/d seam (one-letter, identity-only — chirality
# does not change which residues are hydrophobic).
_HYDROPHOBIC = frozenset("AILMFV")


def _to_dpeptide(seq: str, chirality: str = "D") -> str:
    """Report a sequence in the project convention: chiral D-residues lowercase, Gly always 'G'.

    ``l_seq`` stores L-letter residue identities of the designed peptide. For ``chirality='D'``
    (the α all-D case) every chiral residue is reported lowercase and Gly stays 'G'. For
    ``chirality='L'`` the string is returned uppercase unchanged. Empty input -> ''.
    """
    s = (seq or "").upper()
    if chirality == "L":
        return s
    return "".join(c if c == "G" else c.lower() for c in s)


def composition_stats(seq: str, ala_gly_floor: float) -> dict:
    """Ala%, Gly%, combined Ala+Gly fraction (the anti-poly-Ala readout) for a one-letter seq.

    Fractions are in [0, 1]. ``ala_gly_over_floor`` is True when (Ala+Gly)/n exceeds the design
    driver's composition floor (passed in so this analyzer tracks ``_COMP_MAX_ALA_GLY_FRAC``).
    Empty input -> all zeros, ``ala_gly_over_floor`` False.
    """
    from collections import Counter

    s = (seq or "").upper()
    n = len(s)
    if n == 0:
        return {"len": 0, "ala_frac": 0.0, "gly_frac": 0.0,
                "ala_gly_frac": 0.0, "ala_gly_over_floor": False}
    counts = Counter(s)
    ala = counts.get("A", 0) / n
    gly = counts.get("G", 0) / n
    ala_gly = (counts.get("A", 0) + counts.get("G", 0)) / n
    return {
        "len": n,
        "ala_frac": round(ala, 4),
        "gly_frac": round(gly, 4),
        "ala_gly_frac": round(ala_gly, 4),
        "ala_gly_over_floor": bool(ala_gly > ala_gly_floor),
    }


def heptad_quality(seq: str, heptad_start: str = "a") -> dict:
    """Coiled-coil register quality of a helix sequence (sequence-only; no structure).

    Assigns an ``a b c d e f g`` heptad along the sequence via
    :func:`xenodesign.eval.controls.heptad_register` (phaseable through ``heptad_start``), then
    scores hydrophobic (A/I/L/M/F/V) occupancy at the buried a/d CORE vs the b/c/e/f/g SURFACE.

    Returns a dict with:
      ``n_core`` / ``n_surface``: counts of a/d core and b/c/e/f/g surface positions.
      ``core_hydrophobic_frac``: fraction of core positions that are hydrophobic — the headline
          ``heptad_match_fraction`` mirrors this (a real amphipathic helix packs the a/d seam).
      ``surface_hydrophobic_frac``: fraction of surface positions that are hydrophobic (lower is
          better — a clean amphipathic helix keeps hydrophobics off the solvent face).
      ``heptad_match_fraction``: == ``core_hydrophobic_frac`` (named for the headline readout).
      ``heptad_amphipathy``: core_hydrophobic_frac - surface_hydrophobic_frac, in [-1, 1]; a
          well-formed amphipathic helix is strongly positive.
    Empty input -> zeros.
    """
    from xenodesign.eval.controls import heptad_register

    s = (seq or "").upper()
    n = len(s)
    n_core = n_surface = core_hyd = surf_hyd = 0
    for i, aa in enumerate(s):
        _letter, face = heptad_register(i, start=heptad_start)
        is_hyd = aa in _HYDROPHOBIC
        if face == "core":
            n_core += 1
            core_hyd += int(is_hyd)
        else:
            n_surface += 1
            surf_hyd += int(is_hyd)
    core_frac = (core_hyd / n_core) if n_core else 0.0
    surf_frac = (surf_hyd / n_surface) if n_surface else 0.0
    return {
        "n_core": n_core,
        "n_surface": n_surface,
        "core_hydrophobic_frac": round(core_frac, 4),
        "surface_hydrophobic_frac": round(surf_frac, 4),
        "heptad_match_fraction": round(core_frac, 4),
        "heptad_amphipathy": round(core_frac - surf_frac, 4),
    }


def score_sequence(seq: str, *, ala_gly_floor: float, heptad_start: str = "a",
                   chirality: str = "D", role: str = "", iter_idx=None,
                   iptm=None, chir=None) -> dict:
    """Full per-sequence quality record (composition + entropy + heptad register)."""
    from xenodesign.scorer import sequence_complexity

    rec = {
        "role": role,
        "iter": iter_idx,
        "seq": _to_dpeptide(seq, chirality),
        "iptm": iptm,
        "chirality": chir,
        "norm_entropy": round(float(sequence_complexity(seq)), 4),
    }
    rec.update(composition_stats(seq, ala_gly_floor))
    rec["heptad"] = heptad_quality(seq, heptad_start)
    return rec


def analyze_run(result_path: Path, *, backend: str, ala_gly_floor: float,
                heptad_start: str = "a", chirality: str = "D",
                chir_max: float = 0.10) -> dict:
    """Score the winner + every per-iter sequence of one alpha_result.json, plus run-level rates."""
    d = json.loads(Path(result_path).read_text())
    traj = d.get("trajectory", []) or []

    winner_seq = d.get("selected_l_seq", "")
    winner = score_sequence(
        winner_seq, ala_gly_floor=ala_gly_floor, heptad_start=heptad_start,
        chirality=chirality, role="winner",
        iter_idx=d.get("selected_iter"), iptm=d.get("selected_iptm"),
        chir=d.get("selected_chirality"),
    ) if winner_seq else None

    per_iter = []
    chir_vals = []
    for t in traj:
        c = t.get("chirality")
        if c is not None:
            chir_vals.append(c)
        seq = t.get("l_seq", "")
        if not seq:
            continue
        per_iter.append(score_sequence(
            seq, ala_gly_floor=ala_gly_floor, heptad_start=heptad_start,
            chirality=chirality, role="iter", iter_idx=t.get("iter"),
            iptm=t.get("iptm"), chir=c,
        ))

    n_clean = sum(1 for c in chir_vals if c <= chir_max)
    return {
        "run": str(Path(result_path).parent),
        "result_json": str(result_path),
        "backend": backend,
        "case_id": d.get("case_id"),
        "n_iters": len(traj),
        "chirality_clean_rate": (n_clean / len(chir_vals)) if chir_vals else None,
        "chir_max": chir_max,
        "winner": winner,
        "per_iter": per_iter,
    }


def _fmt_row(label: str, rec: dict) -> str:
    """One fixed-width table row for a per-sequence record."""
    h = rec.get("heptad", {})
    return (
        f"  {label:<22} "
        f"ipTM={_fmt_num(rec.get('iptm')):>6} "
        f"chir={_fmt_num(rec.get('chirality')):>5} "
        f"A%={rec['ala_frac']*100:5.1f} G%={rec['gly_frac']*100:5.1f} "
        f"A+G%={rec['ala_gly_frac']*100:5.1f}{'!' if rec['ala_gly_over_floor'] else ' '} "
        f"H={rec['norm_entropy']:.3f} "
        f"heptad={h.get('heptad_match_fraction', 0.0):.3f} "
        f"amph={h.get('heptad_amphipathy', 0.0):+.3f}"
    )


def _fmt_num(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def render_table(runs: list[dict]) -> str:
    """Readable per-run table: the winner row + the per-iter rows + the chirality-clean rate."""
    lines = []
    for r in runs:
        if "error" in r:
            lines.append(f"[{r.get('backend','?')}] {r['run']}  ERROR: {r['error']}")
            continue
        ccr = r["chirality_clean_rate"]
        ccr_s = f"{ccr*100:.0f}%" if ccr is not None else "n/a"
        lines.append(
            f"[{r['backend']}] {r['run']}  "
            f"(case={r.get('case_id')}, iters={r['n_iters']}, "
            f"chirality-clean={ccr_s})"
        )
        if r.get("winner"):
            w = r["winner"]
            lines.append(_fmt_row(f"WINNER {w['seq']}", w))
        for rec in r.get("per_iter", []):
            lines.append(_fmt_row(f"iter{rec['iter']:>3} {rec['seq']}", rec))
        lines.append("")
    return "\n".join(lines)


def aggregate(runs: list[dict]) -> dict:
    """Cross-run roll-up over the WINNER sequences (anti-survivorship is left to per-run rates)."""
    winners = [r["winner"] for r in runs if r.get("winner")]
    if not winners:
        return {"n_runs": len(runs), "n_winners": 0}

    def _mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    ccrs = [r["chirality_clean_rate"] for r in runs if r.get("chirality_clean_rate") is not None]
    return {
        "n_runs": len(runs),
        "n_winners": len(winners),
        "winner_mean_ala_gly_frac": _mean([w["ala_gly_frac"] for w in winners]),
        "winner_n_over_ala_gly_floor": sum(1 for w in winners if w["ala_gly_over_floor"]),
        "winner_mean_norm_entropy": _mean([w["norm_entropy"] for w in winners]),
        "winner_mean_heptad_match": _mean([w["heptad"]["heptad_match_fraction"] for w in winners]),
        "winner_mean_heptad_amphipathy": _mean([w["heptad"]["heptad_amphipathy"] for w in winners]),
        "mean_chirality_clean_rate": round(sum(ccrs) / len(ccrs), 4) if ccrs else None,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="globs and/or paths to alpha_result.json files (quote globs)")
    p.add_argument("--backend", default="chai",
                   help="backend label for these runs (e.g. chai, carbonara) — analyzer is "
                        "backend-agnostic, this only tags the output")
    p.add_argument("--ala_gly_floor", type=float, default=None,
                   help="Ala+Gly fraction floor (default: design_alpha._COMP_MAX_ALA_GLY_FRAC)")
    p.add_argument("--heptad_start", default="a",
                   help="heptad letter for residue 0 (phase the a/d core; default 'a')")
    p.add_argument("--chirality", choices=("D", "L"), default="D",
                   help="report sequences as all-D (lowercase, Gly=G) or L (uppercase)")
    p.add_argument("--chir_max", type=float, default=0.10,
                   help="chirality-clean threshold for the per-run clean rate (default 0.10)")
    p.add_argument("--out", default=None, help="also write the JSON summary to this path")
    p.add_argument("--no-table", action="store_true", help="suppress the readable table on stderr")
    args = p.parse_args(argv)

    if args.ala_gly_floor is None:
        from scripts.design_alpha import _COMP_MAX_ALA_GLY_FRAC
        ala_gly_floor = _COMP_MAX_ALA_GLY_FRAC
    else:
        ala_gly_floor = args.ala_gly_floor

    # Expand globs (and accept already-expanded literal paths); dedupe, keep order.
    paths: list[str] = []
    for pat in args.inputs:
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])
    seen = set()
    paths = [x for x in paths if not (x in seen or seen.add(x))]

    runs = []
    for pth in paths:
        try:
            runs.append(analyze_run(
                Path(pth), backend=args.backend, ala_gly_floor=ala_gly_floor,
                heptad_start=args.heptad_start, chirality=args.chirality,
                chir_max=args.chir_max,
            ))
        except Exception as exc:  # noqa: BLE001 — report-and-continue across a campaign glob
            runs.append({"run": str(Path(pth).parent), "result_json": pth,
                         "backend": args.backend,
                         "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "backend": args.backend,
        "ala_gly_floor": ala_gly_floor,
        "heptad_start": args.heptad_start,
        "chir_max": args.chir_max,
        "n_inputs": len(paths),
        "aggregate": aggregate([r for r in runs if "error" not in r]),
        "runs": runs,
    }

    if not args.no_table:
        sys.stderr.write(render_table(runs) + "\n")
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
    sys.exit(0)
