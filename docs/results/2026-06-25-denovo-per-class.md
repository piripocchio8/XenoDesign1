# De-novo per-class validation of the unified `scripts/design.py` dispatcher

**Date:** 2026-06-25
**Machine:** ifrit (2× RTX A4500, 20 GB each). Chai image `gradio_design-gradio-design:latest` (chai_lab 0.6.1).
**Branch:** `feat/halludesign-chai-dpeptide` tip (`cf3c1b1`) — the fixes under test:
ChainRoles contract (`0c0cc8b`), unified from-scratch PepMLM seeding (`e926ae2`), full S2-symmetric
24-mer 6UFA (`c2971c5`).
**Entry under test:** the unified `scripts/design.py --binder_class <C>` dispatch path
(`xenodesign/dispatch.run_design` → `targets.target_entities`/`ChainRoles` → class hooks → `HalluLoop`).
**Run scale:** NOT `--smoke`. 12 iters, `num_seqs 4`, ONE fixed seed for reproducibility.
**Seeding:** from-scratch ONLY (no reference-binder leakage). PepMLM target-conditioned masked-fill for the
protein-target classes; the no-target free-cyclic uses `--no_pepmlm` (random from-scratch seed) — see note.

CPU baseline on this worktree before the runs: **815 passed, 30 deselected** (`-m "not gpu and not network"`).

## Per-class results

| Class | CLI target | Iters | From-scratch seed (L, binder) | Selected design (D-FASTA) | ipTM | pTM | L-seed ipTM | Wall | PASS/FAIL |
|---|---|---|---|---|---|---|---|---|---|
| **alpha** | protein (case trimer L-HLH) | 12/12 | `TSKLVWNAKKDMMLALTFEEG` (21) | `(DAL)(DLY)(DAL)(DGL)(DGL)(DLE)(DAR)(DLY)(DAR)(DLE)(DGL)(DGL)(DIL)(DIL)(DLY)(DGL)(DLE)(DLE)(DGL)(DLY)G` (L = `AKAEELRKRLEEIIKELLEKG`) | **0.791** | — | 0.103 | 19.5 min | **PASS** |
| **non_alpha** | protein (9DXX 2-chain MSA'd HA) | 12/12 | `VLKCPDGGCTCKLVWNGKEFMNDALTFQEC` (30) | `(DAR)(DAR)(DGL)(DLE)(DLE)(DAL)(DAL)(DAL)(DAL)(DLE)(DPR)(DVA)(DAR)(DAR)(DAR)(DLE)G(DGL)(DLY)(DAL)(DLE)(DAL)(DAL)(DLE)(DAL)(DAL)(DIL)(DPR)(DAL)G` (30) | 0.167 | 0.380 | **0.869** | 38.5 min | **PASS** |
| **cyclic metallo** | metal ([Zn+2], His 6/12/18/24 L/D/L/D) | 12/12 | `LRRFTHANAMTHKANWVHKHAIGH` (24, His pinned 6/12/18/24) | `(DSN)(DLY)(DGL)(DGL)(DGL)(DGL)(DLY)(DLY)(DLY)(DGL)(DGL)(DGL)(DLY)(DLY)(DLY)(DGL)(DGL)(DGL)(DAS)(DLY)(DLY)(DLY)(DAS)G` (24) | 0.094 | **0.927** | 0.037 | 19.0 min | **PASS** |
| **free cyclic** | none (intramolecular, len 16) | 12/12 | `GYWCKHSYNVAHHWGE` (16, His pinned 6/12) | `G(DLE)GGGG(DAR)(DSG)(DVA)G(DGL)(DLY)(DTH)GGG` (16) | 0.0¹ | 0.656 | 0.0¹ | 9.1 min | **PASS** |

¹ ipTM is undefined for a single free chain (no inter-chain interface); the free-cyclic objective is the
intramolecular 4-term score, not ipTM. pTM is the structural-quality signal.

Total GPU wall ≈ **86 min** of design across the two A4500s (run 2-at-a-time, ≤2 GPU concurrent).

## Key validations

- **non_alpha stays its seed length — the ChainRoles fix.** The binder was **30 residues in EVERY iteration**
  (L-seed = 30; iter_000…iter_011 all = 30 D-CCD residues). NO 31→176 balloon, NO OOM. The 2-chain HA1+HA2
  MSA'd target (~530 aa) + binder fit in **18 GB on the 20 GB A4500** for all 12 iters. ChainRoles
  (`targets=('A','B')`, binder=`'C'`) routed the multi-chain MSA'd target and the seq-update extractor
  correctly, so the length never drifted.
- **cyclic metallo — the chain-order fix + Zn geometry.** The His↔Zn metal-coordination restraint applied on
  **every predict (52× = 4 His↔Zn rows × 13 predicts)** — log shows `[patch] token_dist repaired`, NOT the
  prior `Expected >=1 residue ... found 0` drop. The restraint file is ChainRoles-correct (His on chain **B**,
  Zn `X1` on chain **A** — the inversion that silently dropped the restraint in the 2026-06-24 smoke is gone).
  Best model (iter_011 model_idx_0): a Zn–His-N coordination shell at **1.9–2.3 Å** (chemically reasonable
  Zn–N). **MetalHawk verdict: geometry = `TET` (tetrahedral), perplexity 1.25 < 1.5 → PASS** (clean,
  confident [Zn(His)] tetrahedral assignment, exactly the 6UFA site class).
- **free cyclic — the no-target chain-A path completes.** Single-chain binder (chain A only, no Zn, no second
  chain — confirmed in the FASTA), 16-mer from-scratch, intramolecular objective, all 12 iters, exit 0.
- **alpha — the validated path still runs de-novo.** From-scratch seed → a clean amphipathic D-helix
  (`AKAEELRKRLEEIIKELLEKG`), ipTM 0.791, chirality 0.000, beats the documented baseline (0.44).

## Caveat / finding (free cyclic)

The free-cyclic run was FIRST attempted with the default PepMLM seed (seed 12345) and **crashed at iter_000's
`truncated_refine`**: that PepMLM 16-mer (`MERFTHALNIIHKANI`) contained no glycine, so its fully-D form is
un-tokenizable by chai-1 (`io_spec.build_fasta`: "protein chain 'binder' is fully non-canonical … chai-1 needs
>=1 canonical residue per chain to tokenize"). This is a **known constraint** (ADR-004: an all-D
peptide needs ≥1 canonical residue, e.g. a Gly, to tokenize), surfaced because the no-target free-cyclic seed
is not guaranteed a Gly. Re-running with `--no_pepmlm --seed 3` (a reproducible random from-scratch seed
`GYWCKHSYNVAHHWGE` that, after the cyclic class pins His at 6/12, both starts and ends with Gly) tokenizes and
completes all 12 iters cleanly. A small follow-up: enforce a canonical-residue (Gly-anchor) guard on the
no-target free-cyclic seed so any from-scratch seed tokenizes, as alpha/non_alpha already do via
`_ensure_cterm_glycine`. The protein-target classes (alpha/non_alpha/cyclic-metal) all kept their PepMLM seeds.

## Reproduce

Launchers are host-local helpers under `/tmp` (mirror the gitignored RUNBOOK pattern: docker label fix +
ESM cache mount + HF cache mount, plus the worktree's gitignored `XenoDesign1_local_ref/` and
`LigandMPNN/model_params/` bind-mounted so the relative-path targets/weights resolve inside the container).
All pinned to seed semantics that make the from-scratch seed reproducible (PepMLM temperature sampling keyed
by `--seed`; RandomSeedGenerator keyed by `--seed` for `--no_pepmlm`).

```bash
# 1. alpha
python scripts/design.py --binder_class alpha     --target_type protein \
  --iters 12 --num_seqs 4 --seed 12345 --device cuda:0 --out_dir <out>
# 2. non_alpha (9DXX 2-chain MSA'd HA)
python scripts/design.py --binder_class non_alpha --target_type protein \
  --iters 12 --num_seqs 4 --seed 12345 --device cuda:0 --out_dir <out>
# 3. cyclic metallo (24-mer 6UFA, His 6/12/18/24 L/D/L/D, 4 His<->Zn restraints)
python scripts/design.py --binder_class cyclic    --target_type metal \
  --coord_residues 'H6,DHI12,H18,DHI24' \
  --iters 12 --num_seqs 4 --seed 12345 --device cuda:0 --out_dir <out>
# 4. free cyclic peptide (no target, intramolecular, len 16) — Gly-containing from-scratch seed
python scripts/design.py --binder_class cyclic    --target_type none --binder_length 16 \
  --no_pepmlm --iters 12 --num_seqs 4 --seed 3 --device cuda:0 --out_dir <out>
```

MetalHawk geometry gate on the best cyclic-metallo model:
```bash
METALHAWK_DIR=/home/user/tools/MetalHawk python - <<'PY'
from xenodesign.eval.metal_geometry_gate import metal_geometry_gate
print(metal_geometry_gate('<best>.cif', metal_element='ZN', threshold=1.5,
      metalhawk_dir='/home/user/tools/MetalHawk', metalhawk_env='metalhawk').as_dict())
PY
# -> {'geometry': 'TET', 'perplexity': 1.25, 'passed': True, 'ok': True}
```

Run outputs live under the gitignored `XenoDesign1_local_ref/denovo_<class>/` (per-run `*_result.json`,
`resolved_config.json`, `cyclic.restraints`, `p0_l_seed/`, `loop/iter_000..011/`).
