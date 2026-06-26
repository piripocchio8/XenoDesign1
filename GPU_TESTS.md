# GPU / Network Test Package — run on your local GPU

This package validates the parts of `xenodesign` that need a GPU (Chai-1) or model downloads
(PepMLM). They are **deselected by default** so the CPU suite stays fast and hermetic.

> **Runtime environments (read this before trusting any "no network / no GPU" note below).**
> This repo is developed/run in two very different contexts — don't conflate them:
> - **Cloud authoring env** (where parts of this branch were first written): **no GPU**, and
>   outbound network **reaches GitHub only** (RCSB / MONDE·T return 403). Any "403 / GitHub-only /
>   no GPU" caveat in these docs describes *this* env only.
> - **Local GPU box**: **full GPUs and open outbound network** (RCSB / HuggingFace / GitHub all
>   reachable). GPU tests and runtime RCSB fetches work here. Chai-1 runs via the
>   **chai 0.6.1 container image** (exact tag in `RUNBOOK.local.md`; model weights baked in).
>   **This is not a sandbox — run GPUs, Docker, and network freely.**

> **Status:** `ChaiBackend.predict` and the Tier-0a chirality gate (`gate_tier0a.run_gate`)
> are **VERIFIED** on chai_lab 0.6.1 (GPU A4500, 2026-06-14). Chirality violation fraction =
> 0.000. `PepMLMSeedGenerator._real_generate` is not yet verified on GPU.

## 1. Environment

GPU tests run inside the **chai 0.6.1 container image** (exact image tag and host weights cache
path are in `RUNBOOK.local.md`):

```bash
# First time: pre-cache ESM traced model (~5.7 GB) to avoid re-download each run
# (URL and target path: see RUNBOOK.local.md §1 and §7)
wget "<ESM_URL>" -O <host_weights_cache>/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt

# Run GPU tests (from repo root) — image tag and host_weights_cache path in RUNBOOK.local.md:
docker run --rm --gpus all -e PYTHONPATH=/work -e PYTHONDONTWRITEBYTECODE=1 \
  -v $PWD:/work -v <host_weights_cache>:/chai-lab/downloads \
  -w /work --entrypoint bash <chai-image> -lc \
  'pip install -q pytest gemmi 2>/dev/null; python -m pytest -m gpu -v -s -p no:cacheprovider'
```

`pytest` and `gemmi` are not baked into the container image and are installed inline each run.
Chai-1 model weights (512-dim) are baked in at `/opt/venv/lib/python3.11/site-packages/downloads/`.
ESM traced model is volume-mounted from the host weights cache. A CUDA GPU is required; tests skip
if `torch.cuda.is_available()` is False.

## 2. Run

```bash
# the full CPU suite (no GPU, no downloads):
PYTHONPATH=$PWD python -m pytest -m "not gpu and not network" -q

# GPU tests (Chai-1) — run inside the chai 0.6.1 container (see §1 above, image tag + paths in RUNBOOK.local.md):
docker run --rm --gpus all -e PYTHONPATH=/work -e PYTHONDONTWRITEBYTECODE=1 \
  -v $PWD:/work -v <host_weights_cache>:/chai-lab/downloads \
  -w /work --entrypoint bash <chai-image> -lc \
  'pip install -q pytest gemmi 2>/dev/null; python -m pytest -m gpu -v -s -p no:cacheprovider'

# Network test (PepMLM weights download) — can run outside Docker:
pytest -m network -v
```

Tests skip with a clear reason if a dependency or the GPU is missing.

## 3. What each test checks

| Test | Marker | Checks |
|---|---|---|
| `tests/gpu/test_chai_predict_gpu.py` | `gpu` | `ChaiBackend.predict` runs and returns a well-formed `Prediction` (coords Nx3, per-token pLDDT, 0≤ipTM≤1). Validates the `StructureCandidates → Prediction` parsing. |
| `tests/gpu/test_chai_chirality_gate_gpu.py` | `gpu` | **The Tier-0a go/no-go** (spec §3): predicts an all-D peptide and asserts the predicted residues keep D chirality (violation fraction ≪ 0.51). |
| `tests/gpu/test_pepmlm_network.py` | `network` | `PepMLMSeedGenerator` (real weights) produces a valid-alphabet peptide of the requested length. |

## 3b. Real-PDB chirality benchmark (recommended go/no-go)

`tests/gpu/test_chirality_benchmark_gpu.py` (markers: `gpu` **and** `network`) is the
real-data Tier-0a gate. For each curated PDB id it **downloads the CIF from RCSB at runtime**,
finds the chain(s) containing D-amino acids (data-driven, via chai's `D_partners`), rebuilds
the Chai input from the experimental residue codes, predicts with Chai, and asserts the
predicted residues keep their D/L chirality.

```bash
pytest tests/gpu/test_chirality_benchmark_gpu.py -m "gpu and network" -v
```

> **Why not bundled with real data:** the *cloud authoring env* (see Runtime environments above)
> reaches **GitHub only** — RCSB and MONDE·T returned 403 *there* — so the CIFs are fetched at test
> time instead. On a **local GPU box (e.g. ifrit), RCSB is reachable** and this benchmark runs
> directly. Extend `D_CONTAINING_PDBS` from MONDE-T
> (`xenodesign.catalog.chai_supported_d_codes(<mondet.csv>)` → pick PDB ids with those codes).
> Caveat: if Chai relabels chains in its predicted CIF, map the design chain by entity order.

## 4. Making the gate rigorous (recommended next step)

The bundled gate case is a minimal all-D smoke signal. For a real go/no-go, add
`GateCase` entries built from experimental MONDE-T structures that contain D residues, with
`ref_backbone` filled from the deposited coordinates so the φ/ψ diagnostic (±25°) is active:

1. Use `xenodesign.catalog.chai_supported_d_codes(<mondet.csv>)` to enumerate the D codes
   Chai supports, and pick PDB entries containing them (e.g. **7QDI** D-AIB-310 coiled coil,
   polytheonamide B).
2. For each: build `entities` with the correct D-CCD sequence (`io_spec`), set `design_labels`
   to the per-position L/D pattern, and parse the experimental backbone into `ref_backbone`
   (list of `{'N','CA','C'}` per residue) — `gate_tier0a.backbone_by_residue_from_cif` does
   this for any CIF.
3. Run `run_gate(cases, ChaiBackend(...), out_dir)` and read `overall.passed`.

## 5. Decision

- **PASS** (chirality violation ≪ 51%, φ/ψ within ±25°) → proceed to the design-loop plan
  (`HalluLoop` + `SequenceUpdater` + `truncated_refine`).
- **FAIL** → Chai does not preserve D chirality in prediction; the forward-only route won't
  recover it. Reconsider per spec §3 (low-σ-only refinement, or pivot to AFI on Boltz).
