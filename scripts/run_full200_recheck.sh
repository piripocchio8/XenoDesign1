#!/usr/bin/env bash
# =============================================================================
# Full de-novo set re-check — with the chain-role / OOM fixes in place.
# =============================================================================
# HEAVY: this re-runs the full de-novo design battery on GPU. It is gated and
# will REFUSE to run without an explicit opt-in (see guard below). It is safe to
# leave un-run; nothing executes on `bash -n` or on a plain source.
#
# -----------------------------------------------------------------------------
# !!! WHAT MARCO MUST CONFIRM BEFORE TRUSTING THIS AS "THE" FULL-200 RE-CHECK !!!
# -----------------------------------------------------------------------------
# There is NO "200-design" battery spec committed in this repo. A grep for
# "full-200 / 200 designs / battery" finds ONLY `num_diffn_timesteps=200`
# (full diffusion-step count), which is unrelated to a design count. The only
# VERIFIABLE "full de-novo set" is the 4-class per-class validation documented in
# docs/results/2026-06-25-denovo-per-class.md. This script mirrors THAT battery
# exactly (the 4 classes + their exact flags), driven per-class via design.py.
#
# CONFIRM / ADJUST:
#   1. SET DEFINITION: is "full-200" the 4-class de-novo battery below, or a
#      larger N-design campaign? If the latter, point this at a cases_json and
#      run via scripts/run_parallel.py (--cases_json cases.json --n_gpus N) — the
#      multi-GPU persistent-worker runner. (Template stub at the bottom, commented.)
#   2. N / REPLICATES: the documented battery is 1 fixed seed, iters=12,
#      num_seqs=4 per class (4 runs total). If "200" means 200 designs, raise
#      XENO_ITERS / XENO_NUM_SEQS or fan out many seeds via run_parallel.py.
#   3. CONFIGS: the per-class flags below are copied verbatim from the
#      2026-06-25-denovo-per-class.md "Reproduce" block. Confirm the non_alpha
#      target (9DXX 2-chain MSA'd HA) and the cyclic-metal coord spec are still
#      the intended ones for this re-check.
#   4. SEED: SEED=12345 reproduces alpha/non_alpha/cyclic-metal; free-cyclic uses
#      --no_pepmlm --seed 3 (Gly-anchored from-scratch seed; see the doc's caveat).
#
# WHY RE-CHECK: validates the chain-role contract + the 9DXX OOM fix
# (binder=last chain; non_alpha stays its seed length, no 31->176 balloon) end to
# end across all classes on the current branch.
#
# RUN IT (explicit opt-in required):
#   XENO_CONFIRM_FULL200=1 bash scripts/run_full200_recheck.sh
#   XENO_CONFIRM_FULL200=1 XENO_GPU=1 bash scripts/run_full200_recheck.sh
#
# Host-agnostic via env (defaults mirror scripts/run_design_smoke.sh):
#   XENO_REPO_ROOT     repo checkout to mount as /work  (default: this script's repo root)
#   CHAI_IMAGE         container image:tag              (default: xenodesign:test)
#   CHAI_WEIGHTS       chai weights cache dir           (default: $HOME/chai_weights_cache)
#   XENO_GPU           GPU device index                 (default: 0)
#   XENO_ITERS         design iters per class           (default: 12)
#   XENO_NUM_SEQS      num_seqs per iter                (default: 4)
#   XENO_SEED          reproducible seed                (default: 12345)
#   XENO_COMPOSE_GUARD =1 to emit docker-compose oneoff labels (dev-box reconcile workaround)
# =============================================================================
set -euo pipefail

# --- Heavy-run guard: refuse to run without explicit opt-in -------------------
: "${XENO_CONFIRM_FULL200:?Set XENO_CONFIRM_FULL200=1 to run the heavy full-200 re-check}"

REPO_ROOT="${XENO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHAI_IMAGE="${CHAI_IMAGE:-xenodesign:test}"
CHAI_WEIGHTS="${CHAI_WEIGHTS:-$HOME/chai_weights_cache}"
XENO_GPU="${XENO_GPU:-0}"
ITERS="${XENO_ITERS:-12}"
NUM_SEQS="${XENO_NUM_SEQS:-4}"
SEED="${XENO_SEED:-12345}"
cd "$REPO_ROOT"

OUT_ROOT="runs/overnight/full200_recheck"
echo "=== full de-novo re-check: GPU=$XENO_GPU image=$CHAI_IMAGE out=$OUT_ROOT iters=$ITERS seqs=$NUM_SEQS seed=$SEED ==="

COMPOSE_LABELS=()
if [ "${XENO_COMPOSE_GUARD:-0}" = "1" ]; then
  COMPOSE_LABELS+=(--label com.docker.compose.project=xeno-overnight)
  COMPOSE_LABELS+=(--label com.docker.compose.oneoff=True)
fi

# Run one design.py invocation inside the chai container.
#   $1 = sub-out-dir name under $OUT_ROOT ; $2.. = design.py flags
run_case() {
  local name="$1"; shift
  local out="$OUT_ROOT/$name"
  echo "--- case: $name -> $out  flags: $* ---"
  docker run --rm "${COMPOSE_LABELS[@]+"${COMPOSE_LABELS[@]}"}" \
    --gpus "\"device=$XENO_GPU\"" --network host -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 \
    -v "$PWD":/work -v "$CHAI_WEIGHTS":/chai-lab/downloads \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -w /work --entrypoint bash "$CHAI_IMAGE" -lc \
    "pip install -q freesasa gemmi 2>/dev/null; \
     python scripts/design.py --device cuda:0 --out_dir $out $*"
}

# --- The 4-class de-novo battery (verbatim from 2026-06-25-denovo-per-class.md) ---
# 1. alpha (protein target — trimer L-HLH)
run_case alpha \
  --binder_class alpha --target_type protein \
  --iters "$ITERS" --num_seqs "$NUM_SEQS" --seed "$SEED"

# 2. non_alpha (9DXX 2-chain MSA'd HA) — validates the ChainRoles / OOM fix
run_case non_alpha \
  --binder_class non_alpha --target_type protein \
  --iters "$ITERS" --num_seqs "$NUM_SEQS" --seed "$SEED"

# 3. cyclic metallo (24-mer 6UFA, His 6/12/18/24 L/D/L/D, 4 His<->Zn restraints)
run_case cyclic_metal \
  --binder_class cyclic --target_type metal \
  --coord_residues 'H6,DHI12,H18,DHI24' \
  --iters "$ITERS" --num_seqs "$NUM_SEQS" --seed "$SEED"

# 4. free cyclic peptide (no target, intramolecular, len 16) — Gly-anchored seed
run_case free_cyclic \
  --binder_class cyclic --target_type none --binder_length 16 \
  --no_pepmlm --iters "$ITERS" --num_seqs "$NUM_SEQS" --seed 3

echo "=== full de-novo re-check complete -> $OUT_ROOT ==="

# -----------------------------------------------------------------------------
# OPTIONAL — if "full-200" means a large N-design campaign (NOT the 4-class set):
# build a cases.json and drive the multi-GPU persistent-worker runner instead:
#
#   docker run --rm --gpus all -e PYTHONPATH=/work -v "$PWD":/work \
#     -v "$CHAI_WEIGHTS":/chai-lab/downloads -w /work \
#     --entrypoint bash "$CHAI_IMAGE" -lc \
#     'python scripts/run_parallel.py --cases_json cases.json \
#        --out_dir runs/overnight/full200_recheck --n_gpus 2'
#
# (Confirm the cases.json contents with Marco before using this path.)
# -----------------------------------------------------------------------------
