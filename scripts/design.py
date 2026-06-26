"""Multi-class hallucination design dispatcher CLI (T3).

Thin front end over ``xenodesign.dispatch.run_design``: parse the binder-class / target-chemistry
axes plus the per-knob override flags, resolve a ``DesignConfig`` (per-class PRESET ← config-file
← CLI flags), run the shared HalluLoop via the class's injected hooks, and print a one-line summary.

Examples::

    python scripts/design.py --binder_class alpha  --target_type protein --smoke
    python scripts/design.py --binder_class cyclic --target_type metal   --search greedy
    python scripts/design.py --binder_class non_alpha --config-file run.json --iters 30
"""
from __future__ import annotations

import argparse
import sys


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="multi-class hallucination design dispatcher")
    p.add_argument("--binder_class", choices=("alpha", "non_alpha", "cyclic"), required=True)
    p.add_argument("--target_type",
                   choices=("protein", "rna", "dna", "small_molecule", "metal", "none"),
                   default=None,
                   help="'none' = binder-only (free cyclic/linear peptide; intramolecular objective)")
    p.add_argument("--config-file", dest="config_file", default=None)
    p.add_argument("--fasta", default=None,
                   help="optional explicit target FASTA (overrides the per-class case default)")
    p.add_argument("--pdb", default=None,
                   help="optional explicit target PDB (overrides the per-class case default)")
    p.add_argument("--search", choices=("greedy", "beam", "abc"), default=None,
                   help="'abc' = mixed-chirality ABC search (cyclic + target_type none only)")
    p.add_argument("--abc_variant", choices=("a", "b"), default=None,
                   help="ABC axis split: 'a' = search chirality + MPNN identity (default); "
                        "'b' = search identity+chirality, MPNN warm-start only")
    p.add_argument("--abc_cycles", type=int, default=None, help="ABC employed/onlooker/scout cycles")
    p.add_argument("--colony_size", type=int, default=None, help="ABC colony size (food sources)")
    p.add_argument("--scout_limit", type=int, default=None,
                   help="ABC scout limit (cycles of stagnation before a source is re-seeded)")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--num_seqs", type=int, default=None)
    p.add_argument("--backend", choices=("ligandmpnn", "carbonara", "mixed"), default=None)
    p.add_argument("--objective", choices=("iptm", "mixed", "ipsae", "contrastive"), default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--binder_length", type=int, default=None,
                   help="FROM-SCRATCH binder length (clamped 6..50; 0/absent = per-class default)")
    p.add_argument("--cys_positions", default=None,
                   help="non_alpha: OPT-IN ICK Cys positions, e.g. '3,7,12,18,22,26' (placed at "
                        "those positions + drive the disulfide rows). Absent = no Cys scaffold.")
    p.add_argument("--coord_residues", default=None,
                   help="DECLARATIVE metal-coordinator residues for metallo design, e.g. "
                        "'H6,DHI12,H18,DHI24'. Each token = identity+position; a 1-letter code "
                        "(H) is an L-residue, a CCD code (DHI) a D-residue. Drives the seed's "
                        "fixed positions AND the metal-coordination restraint rows; overrides the "
                        "case default his_resnums/chirality when given.")
    p.add_argument("--length_sweep", action="store_true",
                   help="run a coarse length ladder and pick the best-by-objective design")
    p.add_argument("--chirality_gate", action="store_true")
    p.add_argument("--periodicity_gate", action="store_true")
    p.add_argument("--no_restraints", action="store_true")
    p.add_argument("--no_pepmlm", action="store_true")
    p.add_argument("--no_pll", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def _overrides(a) -> dict:
    """Map the present CLI flags to dotted DesignConfig override keys (absent flags are omitted so
    the per-class PRESET / config-file value wins)."""
    o: dict = {}
    if a.search is not None:
        o["loop.search"] = a.search
    if a.abc_variant is not None:
        o["abc.variant"] = a.abc_variant
    if a.abc_cycles is not None:
        o["abc.cycles"] = a.abc_cycles
    if a.colony_size is not None:
        o["abc.colony_size"] = a.colony_size
    if a.scout_limit is not None:
        o["abc.scout_limit"] = a.scout_limit
    if a.iters is not None:
        o["loop.iters"] = a.iters
    if a.num_seqs is not None:
        o["loop.num_seqs"] = a.num_seqs
    if a.backend is not None:
        o["loop.backend"] = a.backend
    if a.objective is not None:
        o["objective"] = a.objective
    if a.fasta is not None:
        o["target.fasta_path"] = a.fasta
    if a.pdb is not None:
        o["target.pdb_path"] = a.pdb
    if a.device is not None:
        o["device"] = a.device
    if a.seed is not None:
        o["seed"] = a.seed
    if a.binder_length is not None:
        o["binder_length"] = a.binder_length
    if a.chirality_gate:
        o["gates.chirality"] = True
    if a.periodicity_gate:
        o["gates.periodicity"] = True
    if a.no_restraints:
        o["restraints_on"] = False
    if a.no_pepmlm:
        o["use_pepmlm"] = False
    if a.no_pll:
        o["use_pll"] = False
    if a.smoke:
        o["loop.iters"], o["loop.num_seqs"] = 3, 2
    return o


def _parse_cys_positions(spec: str | None) -> tuple:
    """'3,7,12' -> (3, 7, 12); empty/None -> (). Rejects non-positive / non-int tokens."""
    if not spec or not spec.strip():
        return ()
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        p = int(tok)
        if p < 1:
            raise ValueError(f"--cys_positions must be >= 1, got {p}")
        out.append(p)
    return tuple(out)


def _apply_declarative_flags(cfg, a) -> None:
    """Thread the DECLARATIVE coordinator/scaffold flags into cfg.restraint.params (opt-in).

    Absent flags leave the per-class defaults untouched. Present flags populate
    restraint.params so the seed (opt-in fixed positions) and the metal restraint builder
    pick them up — replacing the hardcoded his_resnums/chirality for the metal case."""
    if a.cys_positions is not None:
        cfg.restraint.params["cys_positions"] = _parse_cys_positions(a.cys_positions)
    if a.coord_residues is not None:
        from xenodesign.coordinators import parse_coord_residues
        coords = parse_coord_residues(a.coord_residues)
        # Stored as plain tuples so the resolved-config dump is JSON-clean.
        cfg.restraint.params["coord_residues"] = [
            (c.pos, c.one_letter, c.three_letter, c.chirality) for c in coords]


def main(argv=None):
    from xenodesign.config import resolve_config
    from xenodesign.dispatch import run_design

    a = _parse_args(argv)
    out = a.out_dir or f"/home/tmp/xd_{a.binder_class}"
    cfg = resolve_config(a.binder_class, target_type=a.target_type,
                         config_file=a.config_file, cli_overrides=_overrides(a), out_dir=out)
    _apply_declarative_flags(cfg, a)
    if a.length_sweep:
        from xenodesign.dispatch import run_length_sweep
        result = run_length_sweep(cfg)
        print(f"SWEEP best binder_length {result.get('binder_length')}  "
              f"iptm {result.get('selected_iptm')}  -> {out}")
    else:
        result = run_design(cfg)
        if result.get("search") == "abc":
            print(f"SELECTED nectar {result.get('selected_nectar')}  "
                  f"variant {result.get('abc_variant')}  -> {out}")
        else:
            print(f"SELECTED iptm {result.get('selected_iptm')}  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
