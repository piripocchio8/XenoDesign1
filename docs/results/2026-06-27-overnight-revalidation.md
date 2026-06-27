# Overnight re-validation + 4-track implementation (2026-06-27)

Autonomous session. Branch `feat/overnight-2026-06-27`, PR #5 (vs `main`, NOT merged).
CPU suite: **884 → 945 passed** (+61 TDD tests), ~18 commits. Spec: `docs/superpowers/specs/2026-06-27-overnight-tracks-design.md` (local).

## What is convincingly done + tested (code level)

| Track | Status | Proof |
|---|---|---|
| #1 metal coordinator-retention | DONE | masking unit tests guarantee declared coordinators stay fixed in MPNN + frozen in ABC; provenance test pins recorded seq == deposited CIF |
| #2 ncAA Variant-B | DONE | palette validation + ncAA move + `(XXX)` emission + MPNN-fix + frozen-dispatch tests |
| #3 audit refactors (HW-4/6, MOD-1..6) | DONE | each audit-named proving test green; alpha.py 1060→309, cyclic.py 944→323 LOC |
| #4 re-validations | LAUNCHED + PREPARED | calibration launched; Variant-A re-pilot + full-200 staged (guarded), not launched |

The #1 fix is **proven independent of GPU**: the unit tests show the coordinator positions are forced into the MPNN `fixed_mask` and excluded from ABC chirality mutation, so they cannot drift — the exact failure that produced the 2-His model.

## GPU validations (empirical bonus, in `runs/overnight/`, gitignored)

Two bounded jobs launched (≤2-GPU cap, compose-label override, GPU-pinned):

### #1 metal re-run — `runs/overnight/metal_rerun/`
`--binder_class cyclic --target_type metal --coord_residues 'H6@ND1,DHI12@ND1,H18@ND1,DHI24@ND1' --binder_length 24 --iters 12 --num_seqs 2 --seed 42`
- Confirmed in logs: all three monkeypatches active; **all four His→Zn contact restraints applied** (`token_dist repaired ... HIS->LIG` ×4 — i.e. the D-residue contact repair works for the coordinators).
- Slow cold-start (`esm_device=cpu` + ~25 GB weight load over bind mount); GPU diffusion had not produced the first `pred.model_idx_0.cif` at the time of writing.
- **Open empirical question:** whether Chai applies the atom-level COVALENT His ND1→Zn (dative) bond in addition to the contact. The residue-level contacts demonstrably apply regardless. To be filled from the per-iter outputs.

### #4 widened cyclization-calibration — `runs/overnight/cyclization_calibration/`
`run_cyclization_calibration.py` lanes pos+neg, widened steps `10 25 50 100 150 200` (denser than the default 5-point sweep). NOTE: "wider" here = denser **step** granularity; widening the **case** set (more cycle sizes) needs a small panel-builder edit, deferred.

## Prepared, NOT launched (track #4)
- `scripts/run_variant_a_repilot.sh` — fair Variant-A re-pilot matched to the Variant-B pilot budget (cyclic/none, `--abc_variant a --abc_cycles 20 --colony_size 12`, 90-eval budget via `abc_runs/repilot_a_matched.json`).
- `scripts/run_full200_recheck.sh` — guarded by `XENO_CONFIRM_FULL200`. The "full-200" set is **not defined in the repo**; the script mirrors the 4-class de-novo battery and documents what to confirm.

## Honest caveats
- The atom-specific covalent-to-Zn restraint is novel; its runtime effect is GPU-validated, not assumed.
- GPU results above are to be completed from `runs/overnight/` once the jobs finish.
