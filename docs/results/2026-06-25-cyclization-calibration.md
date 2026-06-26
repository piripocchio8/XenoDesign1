# Cyclization calibration — diffusion (steps × start) vs RING CLOSURE (ABC gate)

**Date:** 2026-06-25  **Branch:** worktree off `feat/halludesign-chai-dpeptide` (tip `dbfe551`)
**Runtime:** chai_lab 0.6.1, image `gradio_design-gradio-design:latest`, 2× RTX A4500, ESM cache
`/home/user/chai_weights_cache`. Drivers: `scripts/run_cyclization_calibration.py` (GPU) +
`scripts/analyze_cyclization_calibration.py` (tables).
**GPU time:** POS lane (GPU 0) 987 s · NEG lane (GPU 1) 971 s, **run in parallel ≈ 16.5 min total**,
zero predict errors (40 predicts + 4 refine-seeds).

## Question (reframed from the 2026-06-24 GATE-FAILS run)

The signal the fast-cycle fitness must detect is **CYCLIZATION** (proper head-to-tail ring closure),
**NOT** real-vs-scramble register specificity. So this sweep uses **only the GENERAL head-to-tail
COVALENT closure restraint** (`benchmark.restraints.head_to_tail_closure_row`, == `cyclic.build_closure_row`)
— **NO Zn, NO coordination restraints** — so it generalizes to ANY mixed-chirality design.
`chai_patches.ensure_patches()` is installed so the D-residue covalent closure name-match works
(verified: the closure bond is **consumed**, atom pair built, not dropped — e.g. POS-6 C(D-Pro6)→N(Glu1)).

Is there a **(steps, start)** MUCH cheaper than full-200 where the KNOWN cycles are CLOSED **AND** a term
ranks proper cyclization ABOVE the strained full-L control?

## Panel

| case | lane / GPU | sequence (chai mixed-chirality) | role |
|---|---|---|---|
| **POS-24** | pos / GPU0 | `KL(DGN)(DGL)AH(DLY)(DLE)QEA(DHI)`×2 (AIB→ALA proxy) | real 6UFA 24-mer macrocycle |
| **POS-6**  | pos / GPU0 | `EP(DPR)KP(DPR)` (= EPpKPp) | real mixed-chirality hexamer |
| **NEG-24** | neg / GPU1 | full-L random 24-mer (closure bond applied) | strained homochiral control |
| **NEG-6**  | neg / GPU1 | full-L random 6-mer (closure bond applied)  | strained homochiral control |

Sweep (identical both lanes): steps **{10, 25, 50, 100, 200}** × start **{full, refine}**.
- **full** = full predict at K steps from pure noise **WITH the closure bond enforced** (constraint_path).
- **refine** = truncated refinement, trailing K steps from a less-noised 25-step seed. The vendored
  truncated sampler does **not** accept `constraint_path` (TODO #27), so closure here is **emergent**
  (measured, not enforced) — it isolates whether WHERE you start matters, not just how many steps.

Per (case × steps × start) we measured **(1)** ground-truth closure (C(resL)↔N(res1) distance + closure-
amide ω planarity) and **(2)** each objective TERM alone + the aggregate.

## 1. Ground-truth closure vs steps — the cycles essentially DO NOT close

`closed` = C–N ≤ 1.6 Å (around the 1.33 Å peptide bond). **Only 1 of 20 POS configs ever closed.**

| case | start | C–N (Å) by steps 10 / 25 / 50 / 100 / 200 | min steps to close |
|---|---|---|---|
| POS-6  | full   | 7.39 / 5.05 / 11.60 / 11.22 / 7.55 | **never** (best 5.05 @ 25) |
| POS-6  | refine | 11.35 / 11.21 / 8.96 / **1.47** / 7.78 | **100** (refine; 1.47 Å ✅) |
| POS-24 | full   | 40.10 / 36.44 / 36.08 / 35.95 / 36.23 | **never** (~36 Å, wide open) |
| POS-24 | refine | 36.48 / 36.38 / 35.94 / 36.49 / 36.50 | **never** |
| NEG-6  | full   | 17.82 / 18.92 / 18.86 / 18.93 / 18.80 | never (~19 Å) |
| NEG-24 | full   | 34.11 / 36.42 / 36.37 / 36.63 / 36.45 | never (~36 Å) |

**Finding:** the chai COVALENT closure bond is **soft conditioning, not a geometric clamp** — it is fed
into `atom_covalent_bond_indices` (a bond *feature* the network conditions on), so the predicted termini
are free to drift. The known cycles do **not** reliably close to a peptide-bond distance at any step count
or start; the 24-mers stay ~36 Å open (a fully extended chain) and the 6-mer hovers 5–12 Å. The single
✅ (POS-6 refine s100, 1.47 Å) is not reproducible across the neighbouring step counts, so **C–N "closed?"
is NOT a usable gate signal.** There is no "minimum steps to close" because closure does not robustly occur.

**BUT** the C–N distance still *separates POS from NEG monotonically*: POS-6 ≈ 5–12 Å vs NEG-6 ≈ 19 Å;
POS-24 ≈ 36 Å vs NEG-24 ≈ 36–38 Å (6-mer separation strong, 24-mer marginal). The real mixed-chirality
termini come closer than the strained full-L ones — see §3.

## 2. Per-term objective (each term alone + aggregate)

Representative rows (full tables in `scripts/analyze_cyclization_calibration.py` output):

| case | start | steps | mc-pLDDT | chirality | geometry | **pTM** | aggregate |
|---|---|---:|---:|---:|---:|---:|---:|
| POS-24 | full | 10 | 0.869 | 0.667 | 0.170 | **0.829** | 0.636 |
| POS-24 | full | 200 | 0.874 | 1.000 | 0.544 | **0.853** | 0.819 |
| NEG-24 | full | 10 | 0.761 | 0.625 | 0.032 | **0.377** | 0.468 |
| NEG-24 | full | 200 | 0.772 | 1.000 | 0.529 | **0.403** | 0.694 |
| POS-6  | full | 200 | 0.753 | 1.000 | 0.381 | **0.139** | 0.599 |
| NEG-6  | full | 200 | 0.851 | 1.000 | 0.405 | **0.159** | 0.638 |

- **chirality** saturates to 1.0 for BOTH POS and NEG at ≥25 steps (the network rarely violates the
  intended handedness once past the noisiest schedule) → **non-discriminating.**
- **mainchain-pLDDT** (cyclizing seam) is high for both (0.75–0.88) and actually *higher for NEG* on the
  6-mer → **non-discriminating / slightly anti-discriminating.**
- **geometry** (closure-ω planarity + backbone valence) is noisy and non-monotone in steps → **unreliable.**
- **pTM** is the one term that cleanly and stably separates the lanes — see §3.

Note the known chai pTM **under-estimate for short / D-rich peptides**: POS-6 pTM ≈ 0.14 (absolute), so on
the 6-mer pTM is depressed for both lanes and barely separates (Δ ≈ −0.02). pTM discrimination is carried
by the **24-mers**, where chai has enough tokens to score the fold.

## 3. POS-vs-NEG discrimination — WHICH term feeds the bees

mean(POS) − mean(NEG) per (start, steps). Positive = ranks real cyclization above the strained control.

| start | steps | Δmc-pLDDT | Δchir | Δgeom | **ΔpTM** | Δaggregate | Δ(C–N) Å |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 10 | +0.006 | −0.146 | +0.128 | **+0.215** | +0.040 | +2.23 |
| full | 25 | +0.004 | +0.000 | +0.150 | **+0.214** | +0.082 | +6.92 |
| full | 50 | +0.002 | −0.083 | +0.023 | **+0.216** | +0.029 | +3.77 |
| full | 100 | +0.004 | −0.083 | −0.105 | **+0.215** | −0.003 | +4.19 |
| full | 200 | +0.002 | +0.000 | −0.004 | **+0.215** | +0.043 | +5.73 |
| refine | 50 | −0.020 | +0.000 | +0.230 | **+0.199** | +0.091 | +5.90 |
| refine | 100 | −0.021 | +0.000 | +0.107 | **+0.197** | +0.060 | +8.61 |
| refine | 200 | −0.020 | +0.000 | −0.256 | **+0.198** | −0.030 | +6.04 |

- **pTM is the discriminating term.** ΔpTM ≈ **+0.20 to +0.22 at EVERY (steps, start)** — large, stable,
  and **already saturated at 10 steps** (it does not improve from 10→200). No other term is stable: Δchir
  flips sign, Δmc-pLDDT is ≈0 / negative, Δgeom swings −0.26…+0.23.
- **Δ(C–N) is a consistent secondary discriminator** (always positive, +2…+9 Å: real termini closer than
  strained), so the **ground-truth closure-distance proxy** agrees with pTM even though absolute closure
  never occurs.
- The **aggregate** is a weak, non-monotone discriminator (Δ −0.03…+0.09) precisely because its
  high-weight terms (mc-pLDDT 0.30, chirality 0.25) are the non-discriminating ones — it dilutes the pTM
  signal. A bee-fitness should **weight pTM (and the C–N proxy) up**, not use the current blend.

## VERDICT

**Is there a (steps, start) MUCH cheaper than full-200 where the known cycles are CLOSED AND a term ranks
proper cyclization above the strained full-L control?**

- **CLOSED: no.** The covalent closure bond is soft conditioning, not a clamp; the known cycles do not
  robustly close to a peptide-bond C–N distance at any step count or start (24-mers ~36 Å open; 6-mer
  5–12 Å; 1/20 fluke). "Steps-to-cyclize" is undefined — closure is not a function that crosses the
  threshold as steps rise, so there is nothing to scale vs tokens. (If hard closure is actually required,
  a post-hoc cyclic-relax / explicit ring-closure energy term is needed — chai's bond feature alone won't
  deliver it.)
- **DISCRIMINATES proper cyclization above the strained control: YES, via pTM — and CHEAPLY.** ΔpTM ≈
  +0.21 is large and **flat from K=10 to K=200** and across both starts. So for the term that works, the
  **MUCH-cheaper operating point is K≈10–25 steps, start=`full`** (≈10–20× cheaper than 200; `full`
  preferred over `refine` because `full` carries the enforced bond and gives the slightly cleaner ΔpTM and
  the strongest Δ(C–N) at 25). The 24-mer carries the pTM signal; the 6-mer's pTM is chai-underestimated so
  small cycles need the C–N proxy instead.

**VIABLE → recommended fast-cycle operating point: K* = 10–25 diffusion steps, start = full predict with
the head-to-tail closure bond. Feed the bees on pTM (primary) + C–N termini-distance proxy (secondary,
essential for short peptides); DO NOT drive on mainchain-pLDDT or chirality (both saturate for real and
fake alike) and DO NOT drive on the current equal-ish aggregate (it dilutes the only working signal).**

What is still needed before an ABC spend:
1. **Re-weight the intramolecular objective** to pTM-led (+ C–N proxy), or replace the aggregate — the
   current weights bury the discriminator. Cheap; re-run this calibration to confirm the re-weighted
   aggregate then separates ≥ ΔpTM.
2. **If hard ring closure is a hard requirement** (not just "real-looking fold"), add an explicit closure
   term/relax — the bond feature does not geometrically close the ring on its own.
3. **Larger panel** (more real macrocycles + more decoys per length) — 2 POS / 2 NEG is a thin gate; the
   pTM margin is encouraging but should be confirmed on more points and on a true mixed-chirality decoy
   (not only full-L), since some of the pTM gap is "L-homochiral vs mixed" not purely "open vs closed".

## Reproduce

```bash
# inside the chai 0.6.1 container (RUNBOOK §2); --label avoids the compose-reconcile kill.
# GPU 0 = real cycles (pos), GPU 1 = fake full-L (neg); identical sweep, run in parallel.
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 --label com.docker.compose.project=xenocalib \
  -e PYTHONPATH=/work -v $PWD:/work -v /home/user/chai_weights_cache:/chai-lab/downloads \
  -w /work --entrypoint bash gradio_design-gradio-design:latest -lc \
  'pip install -q gemmi; python scripts/run_cyclization_calibration.py --lane pos --device cuda:0 \
     --steps 10 25 50 100 200 --starts full refine \
     --out_root .cyc_calib/pos --out .cyc_calib/pos.json'
#  ... same with -e CUDA_VISIBLE_DEVICES=1 and --lane neg --out_root .cyc_calib/neg --out .cyc_calib/neg.json

python scripts/analyze_cyclization_calibration.py --pos .cyc_calib/pos.json --neg .cyc_calib/neg.json
```

CPU helpers (`intramolecular_per_term_fn`, `head_to_tail_closure_geometry_from_cif`, panel/closure-row
builders) are unit-tested in `tests/test_cyclization_calibration.py` (7 CPU tests; full suite 786 green).
Raw JSON + logs under `.cyc_calib/` (git-excluded).
