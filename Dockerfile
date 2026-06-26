# XenoDesign1 — clean, self-contained container.
#
# Runs the xenodesign HalluDesign-on-Chai-1 D-peptide workflow with NOTHING
# ifrit-specific: stock chai_lab 0.6.1 + this repo. Our Chai-1 modifications are
# RUNTIME MONKEYPATCHES (xenodesign/chai_patches.py, applied via ensure_patches());
# they are NOT a fork of chai_lab, so there is no patched-chai image to publish.
# The patches install at import/runtime on top of an unmodified chai_lab==0.6.1.
#
# torch 2.3.1 (chai's pin) is built for CUDA 12.1, so we use a CUDA 12.1 runtime base.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # All chai weights (ESM-2 traced ~5.7 GB, conformers, model weights) land here.
    # Mount a host cache over this path so they download once, not every run.
    CHAI_DOWNLOADS_DIR=/chai_downloads \
    # HuggingFace cache for PepMLM (transformers) seeding weights.
    HF_HOME=/hf_cache \
    PYTHONPATH=/work

# gcc (build-essential) + python3.11-dev are needed to compile the freesasa C
# extension: freesasa ships no cp311 manylinux wheel, so pip builds it from sdist.
# Without a C compiler + Python headers that build fails (gcc: No such file).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# --- Python deps -----------------------------------------------------------
# Explicit set ONLY. Everything else (torch 2.3.1+cu121, gemmi, einops, numpy,
# numba, rdkit, biopython, pandas, matplotlib, ...) is pulled TRANSITIVELY by
# chai_lab==0.6.1 — do not re-pin them here.
#   chai_lab==0.6.1  : the structure model the patches target (pinned, ADR-004).
#   transformers     : PepMLM seeding (xenodesign/seed.py, lazy import). NOT a
#                      chai dependency, so it must be named explicitly. pyproject
#                      [seed] asks transformers>=4.40 (gradio image shipped 4.44.2).
#   freesasa         : buried-surface-area term in scripts/score_complex.py.
#                      NOT a chai dependency and NOT in the old gradio image
#                      (which pip-installed it inline at runtime) — bake it in.
#   gemmi            : already a chai dep; named here so the scoring path is
#                      guaranteed present even if chai's tree ever drops it.
#   pytest           : CPU test suite (-m "not gpu and not network").
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "chai_lab==0.6.1" \
        "transformers>=4.40,<5" \
        freesasa \
        gemmi \
        pytest

# --- Repo ------------------------------------------------------------------
WORKDIR /work
COPY . /work

RUN mkdir -p /chai_downloads /hf_cache

# Default: the unified multi-class design CLI. ensure_patches() runs inside
# xenodesign.dispatch.run_design, so the chai monkeypatches apply automatically.
ENTRYPOINT ["python", "scripts/design.py"]
CMD ["--help"]
