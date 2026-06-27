#!/usr/bin/env bash
# T10 per-class --smoke launcher for the unified scripts/design.py entry.
# Mirrors /tmp/run_alpha.sh (label fix + HF-cache mount + ESM on GPU) but drives design.py.
#   bash scripts/run_design_smoke.sh <GPU> <binder_class> <out_dir> [extra design.py flags...]
# e.g. bash scripts/run_design_smoke.sh 0 alpha XenoDesign1_local_ref/smoke_alpha --target_type protein
#
# Host-agnostic via env (override per box; sane defaults below):
#   XENO_REPO_ROOT     repo checkout to mount as /work  (default: this script's repo root)
#   CHAI_IMAGE         container image:tag              (default: xenodesign:latest)
#   CHAI_WEIGHTS       chai weights cache dir           (default: $HOME/chai_weights_cache)
#   XENO_COMPOSE_GUARD =1 to emit the docker-compose oneoff labels (dev-box reconcile workaround;
#                      default off — see memory: ifrit gradio compose reconcile gotcha)
set -euo pipefail
GPU="$1"; CLASS="$2"; OUT="$3"; shift 3

REPO_ROOT="${XENO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHAI_IMAGE="${CHAI_IMAGE:-xenodesign:latest}"
CHAI_WEIGHTS="${CHAI_WEIGHTS:-$HOME/chai_weights_cache}"
cd "$REPO_ROOT"
echo "=== run_design: GPU=$GPU class=$CLASS out=$OUT image=$CHAI_IMAGE flags: $* ==="

COMPOSE_LABELS=()
if [ "${XENO_COMPOSE_GUARD:-0}" = "1" ]; then
  COMPOSE_LABELS+=(--label com.docker.compose.project=xeno_oneoff)
  COMPOSE_LABELS+=(--label com.docker.compose.oneoff=True)
fi

docker run --rm "${COMPOSE_LABELS[@]+"${COMPOSE_LABELS[@]}"}" \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPU" --network host -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/work -v "$CHAI_WEIGHTS":/chai-lab/downloads \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /work --entrypoint bash "$CHAI_IMAGE" -lc \
  "pip install -q freesasa gemmi 2>/dev/null; python scripts/design.py --binder_class $CLASS --smoke --device cuda:0 --out_dir $OUT $*"
