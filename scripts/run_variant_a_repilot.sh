#!/usr/bin/env bash
# =============================================================================
# Variant-A ABC re-pilot — apples-to-apples vs the 2026-06-25 Variant-B pilot.
# =============================================================================
# WHY THIS SCRIPT EXISTS
#   The 2026-06-25 ABC pilot (docs/results/2026-06-25-abc-pilot.md) was NOT a fair
#   A-vs-B comparison: Variant A ran at a 30-eval budget while Variant B ran at 90.
#   (Budgets were originally sized to per-eval cost: A pays an MPNN forward per move.)
#   On top of that, the *first* A pilot was broken by the double-encode -inf bug and
#   left only 2 usable eval dirs. Both A bugs are now fixed (return result.one_letter;
#   _cif_path wired so the termini term is live).
#
#   This re-pilot runs Variant A at the SAME budget Variant B got, so any A-vs-B
#   verdict is on equal GPU-eval footing rather than confounded by budget.
#
# EXACT BUDGET MATCHED TO THE VARIANT-B PILOT (see the doc + abc_runs/pilot_b.json):
#   class/target  : --binder_class cyclic --target_type none   (free macrocycle, NO target)
#   length        : --binder_length 12
#   search        : --search abc --abc_variant a               (ONLY difference vs B)
#   abc_cycles    : 20      (B pilot: --abc_cycles 20)
#   colony_size   : 12      (B pilot: --colony_size 12)
#   scout_limit   : (unset -> config default 5; B pilot did NOT pass it either)
#   eval budget   : 90      (B pilot: abc.chai_eval_budget=90 via --config-file; ~87 on disk)
#                   -> abc_runs/repilot_a_matched.json {"abc":{"chai_eval_budget":90}}
#   K* fast steps : config default fitness_steps=15 (the calibrated cheap point; B used the same)
#   --no_pll      : yes (B pilot passed --no_pll)
#   restraint     : closure only (head-to-tail covalent), NO Zn / NO coordination — same class as B
#   seed          : warm-start seed is internal/fixed (B pilot did NOT pass --seed)
#
# RUN IT (single command):
#   bash scripts/run_variant_a_repilot.sh
#   XENO_GPU=1 bash scripts/run_variant_a_repilot.sh      # pin to GPU 1
#
# Host-agnostic via env (override per box; defaults below mirror scripts/run_design_smoke.sh):
#   XENO_REPO_ROOT     repo checkout to mount as /work  (default: this script's repo root)
#   CHAI_IMAGE         container image:tag              (default: xenodesign:test)
#   CHAI_WEIGHTS       chai weights cache dir           (default: $HOME/chai_weights_cache)
#   XENO_GPU           GPU device index                 (default: 0)
#   XENO_COMPOSE_GUARD =1 to emit docker-compose oneoff labels (dev-box reconcile workaround;
#                      default off — see memory: ifrit gradio compose reconcile gotcha)
# =============================================================================
set -euo pipefail

REPO_ROOT="${XENO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHAI_IMAGE="${CHAI_IMAGE:-xenodesign:test}"
CHAI_WEIGHTS="${CHAI_WEIGHTS:-$HOME/chai_weights_cache}"
XENO_GPU="${XENO_GPU:-0}"
cd "$REPO_ROOT"

OUT="runs/overnight/variant_a_repilot"
CONFIG="abc_runs/repilot_a_matched.json"
echo "=== Variant-A re-pilot (matched to B): GPU=$XENO_GPU image=$CHAI_IMAGE out=$OUT ==="

COMPOSE_LABELS=()
if [ "${XENO_COMPOSE_GUARD:-0}" = "1" ]; then
  COMPOSE_LABELS+=(--label com.docker.compose.project=xeno-overnight)
  COMPOSE_LABELS+=(--label com.docker.compose.oneoff=True)
fi

docker run --rm "${COMPOSE_LABELS[@]+"${COMPOSE_LABELS[@]}"}" \
  --gpus "\"device=$XENO_GPU\"" --network host -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/work -v "$CHAI_WEIGHTS":/chai-lab/downloads \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /work --entrypoint bash "$CHAI_IMAGE" -lc \
  "pip install -q freesasa gemmi 2>/dev/null; \
   python scripts/design.py \
     --binder_class cyclic --target_type none --binder_length 12 \
     --search abc --abc_variant a --abc_cycles 20 --colony_size 12 \
     --config-file $CONFIG --no_pll --device cuda:0 --out_dir $OUT"
