# Running XenoDesign1 in a clean container

This directory ships **self-contained** container recipes so anyone can run the
`xenodesign` D-peptide design workflow without the ifrit-specific
`gradio_design` image.

- [`../Dockerfile`](../Dockerfile) — Docker / Podman build.
- [`xenodesign.def`](xenodesign.def) — Singularity / Apptainer build (HPC).

## The one thing to understand first: there is NO patched chai image

XenoDesign1 uses **stock `chai_lab==0.6.1`**, unmodified. Our Chai-1 changes
(D-coordinator token-distance match, POCKET name relaxation, COVALENT bonds on
D-residues) are **runtime monkeypatches** in
[`xenodesign/chai_patches.py`](../xenodesign/chai_patches.py), installed via
`ensure_patches()` at import/runtime by `xenodesign.dispatch.run_design`. They
are **not a fork** of `chai_lab`. So the container just needs
`pip install chai_lab==0.6.1` + this repo — the patches apply themselves when the
design code runs. **Do not look for, or try to publish, a "patched chai" image.**

The `0.6.1` pin is load-bearing: the patches target exact private functions in
chai 0.6.1 (`token_dist_restraint`, `token_pair_pocket_restraint`,
`bond_utils.get_atom_covalent_bond_pairs_from_constraints`). A different chai
version can move/rename these and silently break the patches. Keep it pinned.

## Dependencies (what's explicit vs. transitive)

The base is `nvidia/cuda:12.1.1-runtime-ubuntu22.04` because chai's pinned
`torch==2.3.1` is a CUDA-12.1 build. We `pip install` only these explicitly:

| Package            | Why                                                              |
|--------------------|-----------------------------------------------------------------|
| `chai_lab==0.6.1`  | structure model the patches target (pinned, ADR-004).           |
| `transformers`     | PepMLM seeding (`xenodesign/seed.py`). **Not** a chai dep.       |
| `freesasa`         | buried-surface-area term (`scripts/score_complex.py`). Not a chai dep, and was *not* baked into the old gradio image (it pip-installed it inline). |
| `gemmi`            | structure I/O for scoring. Already a chai dep; named for safety. |
| `pytest`           | CPU test suite only.                                             |

Everything else — **torch 2.3.1+cu121, einops, numpy, numba, rdkit, biopython,
pandas, matplotlib, modelcif, tmtools, typer** — is pulled **transitively by
`chai_lab==0.6.1`**. Don't re-pin it.

## What must be mounted (weights are NOT baked in)

Two caches must be volume-mounted; both are large and download on first use:

1. **Chai weights** → container `CHAI_DOWNLOADS_DIR` (`/chai_downloads`).
   chai downloads the **traced ESM-2 model (~5.7 GB)**, the conformer library, and
   the chai model weights here on first inference. Mount a persistent host dir so
   this happens once.
2. **HuggingFace cache** → container `HF_HOME` (`/hf_cache`). PepMLM weights for
   the seeding path download here on first use.

Neither is in the image — the image stays small and license-clean.

## Build

```bash
# Docker (from repo root)
docker build -t xenodesign:test .

# Apptainer / Singularity (from repo root). --fakeroot builds unprivileged on an
# HPC login node (it is implied for a non-root user; pass it explicitly to be safe):
apptainer build --fakeroot xenodesign.sif docker/xenodesign.def
# ...or convert the Docker image you just built on a workstation (no %post rebuild):
apptainer build --fakeroot xenodesign.sif docker-daemon://xenodesign:test
```

> `singularity` (SingularityCE) accepts the same `build --fakeroot`, `run --nv`,
> `-B`/`--bind`, and `docker-daemon://` syntax — swap the binary name.

## Run

### CPU tests (no GPU, no weights)

```bash
docker run --rm -e PYTHONPATH=/work -w /work --entrypoint python xenodesign:test \
  -m pytest -m "not gpu and not network" -q
```

```bash
apptainer exec xenodesign.sif python -m pytest -m "not gpu and not network" -q
```

### A GPU design run (smoke)

Mount the two caches and pick a GPU. The entrypoint is `scripts/design.py`.

```bash
docker run --rm --gpus all \
  -v /host/chai_cache:/chai_downloads \
  -v /host/hf_cache:/hf_cache \
  -v "$PWD/runs":/work/runs \
  xenodesign:test \
  --binder_class alpha --target_type protein --smoke \
  --device cuda:0 --out_dir /work/runs/smoke_alpha
```

For Apptainer the SIF is **read-only**, so `--out_dir` must be a **bind-mounted host
path** (not a path inside the image). `--nv` makes the host NVIDIA libs visible:

```bash
apptainer run --nv \
  -B /host/chai_cache:/chai_downloads \
  -B /host/hf_cache:/hf_cache \
  -B "$PWD/runs":/runs \
  xenodesign.sif \
  --binder_class alpha --target_type protein --smoke \
  --device cuda:0 --out_dir /runs/smoke_alpha
```

First GPU run downloads the ESM-2 / chai weights into the mounted chai cache
(~6 GB, one-time) and the PepMLM weights into the HF cache.

## HPC / SLURM (Apptainer-only clusters)

On clusters with no Docker daemon and no user root (e.g. V100-class HPC: 4× V100
32 GB, `sm_70`), this image runs unmodified. `torch
2.3.1+cu121` covers `sm_70`→`sm_90`, so the V100s need **no rebuild**.

**One-time, on a login node** (build, or convert a workstation Docker image, then
warm the caches so compute nodes never hit the network):

```bash
# Build the SIF without root (or scp a SIF built elsewhere):
apptainer build --fakeroot xenodesign.sif docker/xenodesign.def
# Point Apptainer's own cache at scratch (build cache can be large):
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer_cache
```

**`sbatch` job — 1 GPU per task** (`run.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=xeno-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # one V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00

module load apptainer 2>/dev/null || true   # site-specific; skip if always available

SIF=$SCRATCH/xenodesign.sif
CHAI_CACHE=$SCRATCH/chai_cache               # persistent across jobs
HF_CACHE=$SCRATCH/hf_cache
RUNS=$SCRATCH/runs/$SLURM_JOB_ID
mkdir -p "$CHAI_CACHE" "$HF_CACHE" "$RUNS"

srun apptainer run --nv \
  -B "$CHAI_CACHE":/chai_downloads \
  -B "$HF_CACHE":/hf_cache \
  -B "$RUNS":/runs \
  "$SIF" \
  --binder_class alpha --target_type protein --smoke \
  --device cuda:0 --out_dir /runs/smoke_alpha
```

Submit with `sbatch run.slurm`. Quick interactive check first:
`salloc --gres=gpu:1 --time=00:30:00`, then run the `srun apptainer run --nv ...`
line above.

**Multi-GPU on one allocation.** `scripts/run_parallel.py` runs a persistent-worker
pool: one long-lived process per GPU, each pinned with `CUDA_VISIBLE_DEVICES` (sees
its GPU as `cuda:0`), chai weights loaded once per worker, cases pulled from a shared
queue. With `--gres=gpu:4` run it inside one container so all 4 GPUs are visible:

```bash
#SBATCH --gres=gpu:4
srun apptainer exec --nv \
  -B "$CHAI_CACHE":/chai_downloads -B "$HF_CACHE":/hf_cache -B "$RUNS":/runs \
  "$SIF" python scripts/run_parallel.py \
    --cases_json /runs/cases.json --out_dir /runs/out --n_gpus 4
```

Keep `--workers_per_gpu 1` for heavy chai (one worker saturates a GPU). To instead
spread work over **separate** single-GPU tasks, submit a job array (`--array`,
`--gres=gpu:1` each) with one case per task — simpler scheduling, no shared queue.

### GPU pytest

```bash
docker run --rm --gpus all \
  -v /host/chai_cache:/chai_downloads -v /host/hf_cache:/hf_cache \
  -e PYTHONPATH=/work -w /work --entrypoint python xenodesign:test \
  -m pytest -m gpu -v -s -p no:cacheprovider
```

(`pytest`, `gemmi`, and `freesasa` are already baked into this clean image, so
unlike the old gradio flow you do **not** need to `pip install` them inline.)

## NOT needed in a clean image (ifrit-gradio-specific)

The ifrit run scripts (e.g. `scripts/run_design_smoke.sh`) add
`--label com.docker.compose.project=... --label com.docker.compose.oneoff=True`.
That is a **workaround for the `gradio_design` image only** — its baked-in compose
label makes Docker's reconcile loop intermittently kill ad-hoc containers. This
clean image has no such label, so **the `--label` flags are not needed here.**
The same scripts also `pip install -q freesasa gemmi` inline because the gradio
image lacked them; this image bakes them in, so drop that too.
