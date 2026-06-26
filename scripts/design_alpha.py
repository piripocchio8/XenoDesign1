"""α-case (trimer D/L-ABLE) end-to-end design loop — thin shim over classes/alpha.

The α design logic now lives in :mod:`xenodesign.classes.alpha` (migrated in T5 of the
multi-class framework plan, behaviour-preserving). This module re-exports every public name
from there so existing callers (``scripts/design_alpha_beam.py``, ``tests/test_design_alpha*``)
keep importing the SAME objects from ``scripts.design_alpha``, and keeps the CLI
(``_parse_args`` + ``__main__``) that drives ``run_alpha_design``.

Monkeypatch contract (preserved): the migrated helpers resolve their patchable collaborators
(``_ligandmpnn_design_fn`` / ``carbonara_design_fn`` / ``_cterm_gly_anchor`` / ``_best_cif_path``
/ ``make_alpha_seq_update_fn``) through this module at call time, so tests that
``monkeypatch.setattr(scripts.design_alpha, ...)`` continue to steer behaviour exactly as before.

Usage (inside the gradio_design Docker container, PYTHONPATH=/work, --network host for PepMLM):
    python scripts/design_alpha.py --iters 30 --device cuda:0
    python scripts/design_alpha.py --smoke                  # 1-traj quick wiring smoke
    python scripts/design_alpha.py --chirality_gate         # gate the accepted path D-clean
    python scripts/design_alpha.py --seed_seq AC...G --no_pepmlm   # skip the network seed
"""
from __future__ import annotations

import argparse
import sys

import numpy as np  # noqa: F401  (kept: legacy tests do `np` lookups via this module's namespace)

# Re-export the migrated α logic so callers keep importing the same names from here. The
# inverse-folding base names (_ligandmpnn_design_fn / carbonara_design_fn), the demo plumbing
# (_best_cif_path / _LoopBackendWrapper), MultiCandidate, and every α helper are exposed so the
# legacy CPU tests can monkeypatch them on THIS module (the migrated helpers honour those patches
# via classes.alpha._shim()).
from scripts.design_demo import (  # noqa: F401
    _all_atoms_from_chain,
    _backbone_array_from_residues,
    _best_cif_path,
    _chirality_violation_frac_from_cif,
    _LoopBackendWrapper,
)
from xenodesign.carbonara_backend import carbonara_design_fn  # noqa: F401
from xenodesign.inverse_folding import MultiCandidate  # noqa: F401
from xenodesign.sequence_update import _ligandmpnn_design_fn  # noqa: F401

from xenodesign.classes.alpha import (  # noqa: F401
    _ALPHA_WEIGHTS,
    _COMP_MAX_ALA_GLY_FRAC,
    _COMP_MAX_HOMOPOLYMER_RUN,
    _COMP_MAX_SINGLE_AA_FRAC,
    _COMP_MIN_NORM_ENTROPY,
    _DEFAULT_DEVICE,
    _DEFAULT_N_ITERS,
    _DEFAULT_NUM_SEQS,
    _DEFAULT_REF_TIME_STEPS,
    _RUN_BINDER_CHAIN,
    _RUN_TARGET_CHAIN,
    _TARGET_RECORD,
    Alpha,
    _assemble_alpha_result,
    _binder_helix_fraction,
    _cterm_gly_anchor,
    _ensure_cterm_glycine,
    _loop_score_fn,
    _make_base_backend,
    _make_referee_fn,
    _MixedBackend,
    binder_seq_from_cif,
    build_alpha_restraint,
    build_alpha_seed,
    composition_violation,
    ipsae_objective_from_cif,
    make_alpha_seq_update_fn,
    make_ipsae_loop_score_fn,
    make_mixed_loop_score_fn,
    mixed_objective_from_cif,
    run_alpha_design,
)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="α (trimer D/L-ABLE) design loop")
    p.add_argument("--iters", type=int, default=_DEFAULT_N_ITERS)
    p.add_argument("--ref_time_steps", type=int, default=_DEFAULT_REF_TIME_STEPS)
    p.add_argument("--num_seqs", type=int, default=_DEFAULT_NUM_SEQS)
    p.add_argument("--device", default=_DEFAULT_DEVICE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--no_pepmlm", action="store_true", help="skip the network PepMLM seed")
    p.add_argument("--seed_seq", default=None, help="explicit 21-res L seed (offline/repeat)")
    p.add_argument("--chirality_gate", action="store_true",
                   help="gate the accepted trajectory to chirality ≤0.10")
    p.add_argument("--no_restraints", action="store_true",
                   help="disable the α pin-polarity restraint (revert to truncated_refine)")
    p.add_argument("--no_pll", action="store_true",
                   help="disable the ESM-2 pseudo-log-likelihood judge (impute the pll term)")
    p.add_argument("--esm_device", default=None,
                   help="device for the ESM-PLL judge (default cpu, to keep VRAM free)")
    p.add_argument("--restraint_file", default=None,
                   help="override the case-default pin with this .restraints CSV (topology "
                        "experiment; objective unchanged)")
    p.add_argument("--target_fasta", default=None,
                   help="override the L-HLH target FASTA (record trimer_DL_ABLE_B); for the GNGNS redesign")
    p.add_argument("--backend", choices=("ligandmpnn", "carbonara", "mixed"),
                   default="ligandmpnn",
                   help="inverse-folding base: ligandmpnn (default, baselines preserved), "
                        "carbonara (CARBonAra adapter), or mixed (per-iter round-robin)")
    p.add_argument("--objective", choices=("iptm", "mixed", "ipsae"), default="iptm",
                   help="selection objective: iptm (DEFAULT, reproducible baseline — ipTM+pLDDT), "
                        "mixed (parity-aware mixed_objective panel: bsa/contacts/pack/sc/"
                        "ipsae/iptm/ipae/hbond from a per-candidate score_complex panel), or "
                        "ipsae (rank by the candidate's raw ipSAE confidence axis alone)")
    p.add_argument("--periodicity_gate", action="store_true",
                   help="DESIGN-TIME register gate: reject heptad-periodic (register-UNachievable) "
                        "designs via seq_periodicity (the '-dep' variant)")
    p.add_argument("--heptad_thresh", type=float, default=0.35,
                   help="lag-7 hydropathy-autocorr threshold for the periodicity gate (default 0.35)")
    p.add_argument("--smoke", action="store_true",
                   help="quick wiring smoke: 3 iters, num_seqs 2")
    return p.parse_args(argv)


if __name__ == "__main__":
    import os
    from pathlib import Path

    args = _parse_args()
    iters, num_seqs = (3, 2) if args.smoke else (args.iters, args.num_seqs)
    out = Path(args.out_dir) if args.out_dir else Path(f"/home/tmp/xd_alpha_{os.getpid()}")
    run_alpha_design(
        n_iters=iters, ref_time_steps=args.ref_time_steps, num_seqs=num_seqs,
        device=args.device, seed=args.seed, out_dir=out,
        use_pepmlm=not args.no_pepmlm, seed_seq=args.seed_seq,
        chirality_gate=args.chirality_gate,
        restraints=not args.no_restraints, use_pll=not args.no_pll,
        esm_device=args.esm_device, restraint_file=args.restraint_file,
        target_fasta=args.target_fasta, backend=args.backend,
        objective=args.objective, periodicity_gate=args.periodicity_gate,
        heptad_thresh=args.heptad_thresh,
    )
    sys.exit(0)
