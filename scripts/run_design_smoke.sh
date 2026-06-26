#!/usr/bin/env bash
# T10 per-class --smoke launcher for the unified scripts/design.py entry.
# Mirrors /tmp/run_alpha.sh (label fix + HF-cache mount + ESM on GPU) but drives design.py.
#   bash /tmp/run_design.sh <GPU> <binder_class> <out_dir> [extra design.py flags...]
# e.g. bash /tmp/run_design.sh 0 alpha XenoDesign1_local_ref/smoke_alpha --target_type protein
set -euo pipefail
GPU="$1"; CLASS="$2"; OUT="$3"; shift 3
cd /home/user/claude_projects/XenoDesign1
echo "=== run_design: GPU=$GPU class=$CLASS out=$OUT flags: $* ==="
docker run --rm --label com.docker.compose.project=xeno_oneoff --label com.docker.compose.oneoff=True \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPU" --network host -e PYTHONPATH=/work -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/work -v /home/user/chai_weights_cache:/chai-lab/downloads \
  -v /home/user/.cache/huggingface:/root/.cache/huggingface \
  -w /work --entrypoint bash gradio_design-gradio-design:latest -lc \
  "pip install -q freesasa gemmi 2>/dev/null; python scripts/design.py --binder_class $CLASS --smoke --device cuda:0 --out_dir $OUT $*"
