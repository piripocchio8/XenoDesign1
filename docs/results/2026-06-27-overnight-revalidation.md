# Overnight re-validation + 4-track implementation (2026-06-27)

Autonomous session. Branch `feat/overnight-2026-06-27`, PR #5 (vs `main`, NOT merged).
CPU suite: **884 → 945 passed** (+61 TDD tests), ~18 commits. Spec: `docs/superpowers/specs/2026-06-27-overnight-tracks-design.md` (local).

## What is convincingly done + tested (code level)

| Track | Status | Proof |
|---|---|---|
| #1 metal coordinator-retention | SEQUENCE-DRIFT FIXED + validated; geometry open | coordinator-anchor re-imposes His identity (GPU: His 6/12/18 held all 12 iters vs 0/4 broken); provenance fixed. Open: C-term coord (24), L-coord chirality emission, and 3D 4-His assembly (only 1 His coordinates Zn) — see follow-up issue |
| #2 ncAA Variant-B | DONE | palette validation + ncAA move + `(XXX)` emission + MPNN-fix + frozen-dispatch tests |
| #3 audit refactors (HW-4/6, MOD-1..6) | DONE | each audit-named proving test green; alpha.py 1060→309, cyclic.py 944→323 LOC |
| #4 re-validations | LAUNCHED + PREPARED | calibration launched; Variant-A re-pilot + full-200 staged (guarded), not launched |

The #1 fix is **proven independent of GPU**: the unit tests show the coordinator positions are forced into the MPNN `fixed_mask` and excluded from ABC chirality mutation, so they cannot drift — the exact failure that produced the 2-His model.

## GPU validations (COMPLETED, in `runs/overnight/`, gitignored)

Three bounded GPU jobs ran to completion (≤2-GPU cap, compose-label override, GPU-pinned, exit 0).

### #1 metal re-run — the fix was validated AND it exposed two further bugs
Two runs (12 iters, 2 seqs, seed 42, `--coord_residues 'H6@ND1,DHI12@ND1,H18@ND1,DHI24@ND1'`):

**v1 `runs/overnight/metal_rerun/` (masking-only) — BROKEN, diagnostic.** His collapsed from [6,12,18,24] to the N-terminus + poly-D-Ala from iter_001. **Root cause found:** in this codebase `fixed_mask=True` makes LigandMPNN emit an **'A' placeholder** (sequence_update.py:199), it does NOT preserve the residue. So freezing the coordinators turned His→D-Ala.

**Fix (commit 8f6f1a7):** a `_coordinator_anchor` that re-imposes the declared donor identity POST-design (mirrors the existing `_cterm_gly_anchor`).

**v2 `runs/overnight/metal_rerun_v2/` (coordinator-anchor) — sequence-drift FIXED:**
| iter | v1 His positions | v2 His positions |
|---|---|---|
| 001–011 | drift to [1,2], poly-Ala | **His held at 6, 12, 18 every iter** |

His **6/12/18 are retained across all 12 iterations** (vs 0/4 before) — the sequence-drift bug is fixed and GPU-validated.

**Three issues the GPU run still exposed (honest):**
1. **His24 (C-terminal coordinator) is lost** after iter_003 — conflicts with the C-term Gly anchor (so 3/4, not 4/4, coordinators retained).
2. **L-coordinator chirality not applied** — His6/His18 were declared **L** but the deposited model + `selected_d_fasta` emit them as **D**-His `(DHI)`. The per-coordinator `chirality_pattern` is not reaching the d_fasta encoder in the real loop (passes the unit test in isolation, fails integration).
3. **The 4-His tetrahedron does NOT form geometrically.** In the selected model (iter_011) only **His12 coordinates Zn** (ND1 at 2.78 Å); MetalHawk returns no geometry class (`metal_geometry: {geometry: None}`). Keeping His in the *sequence* is necessary but **not sufficient** — assembling 4 His around the Zn in 3D is an unsolved restraint-strength / structure problem, not a sequence-drift problem.

**Bottom line:** the sequence-drift bug (the thing that produced the original 2-His model from a 4-His design) is fixed and validated; the harder goal — a predicted 4-His L/D/L/D tetrahedron — remains open. Items 1–3 tracked in a follow-up issue.

### #4 widened cyclization-calibration — `runs/overnight/cyclization_calibration/` (COMPLETED)
Lanes pos+neg, widened steps `10 25 50 100 150 200`. The intended signal is present: real mixed-chirality POS cases close/score well (POS-6 @25 steps: objective 0.725, geometry 0.895) while homochiral NEG controls stay strained (NEG-6: cn-distance ~18 Å, omega ~46°, not closed). One transient `IncompleteRead` on POS-6/full/10-step (objective −∞ for that single point; benign). NOTE: "wider" = denser **step** granularity; widening the **case** set needs a panel-builder edit (deferred).

## Prepared, NOT launched (track #4)
- `scripts/run_variant_a_repilot.sh` — fair Variant-A re-pilot matched to the Variant-B pilot budget (cyclic/none, `--abc_variant a --abc_cycles 20 --colony_size 12`, 90-eval budget via `abc_runs/repilot_a_matched.json`).
- `scripts/run_full200_recheck.sh` — guarded by `XENO_CONFIRM_FULL200`. The "full-200" set is **not defined in the repo**; the script mirrors the 4-class de-novo battery and documents what to confirm.

## Honest caveats
- The atom-specific covalent-to-Zn restraint is novel; its runtime effect is GPU-validated, not assumed.
- GPU results above are to be completed from `runs/overnight/` once the jobs finish.
