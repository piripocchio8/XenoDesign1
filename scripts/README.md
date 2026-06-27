# `scripts/` — driver & analysis tiers

This directory mixes the supported entry points with one-off analysis/diagnostic tools.
The tiers below tell contributors what is a stable interface, what is example code, and
what depends on dev-box-local reference data.

## Tier 1 — core drivers (supported entry points)

The main design / prediction / scoring drivers. `design.py` is the single unified entry
point; everything else is a focused driver behind it or alongside it.

| Script | Role |
|---|---|
| `design.py` | Unified multi-class hallucination design **dispatcher CLI** — thin front end over `xenodesign.dispatch.run_design`. Start here. |
| `run_parallel.py` | Multi-GPU persistent-worker batch runner (one worker per GPU, weights loaded once). |
| `design_alpha.py` | α-case (trimer D/L-ABLE) design loop driver (thin shim over `classes/alpha`). |
| `design_cyclic.py` | Cyclic (Zn-macrocycle) single-chain design + geometry-recall driver. |
| `design_nonalpha.py` | Non-alpha (D-knottin : protein-target) 2-chain design driver. |
| `generate_and_select.py` | Generate-and-Select harvester across N independent trajectories. |
| `predict_complex.py` | Predict a 2-chain complex with per-chain chirality; save scores + structure. |
| `score_complex.py` | Mixed-objective metric panel for a 2-chain complex. |

## Tier 2 — examples (not a library)

| Script | Note |
|---|---|
| `design_demo.py` | Worked **example** of wiring the loop/predict backends end-to-end. Treat it as a demo, **not** an importable library. Shared plumbing it demonstrates is being migrated into the `xenodesign` package (see audit MOD-1). |

## Tier 3 — analysis / diagnostics (need `XENO_LOCAL_REF`)

The remaining `analyze_*`, `score_*`, `gate_*`, `run_*`, `make_*`, `plot_*`, `fit_*`,
`rank_*`, `diagnose_*`, and underscore-prefixed scripts are exploratory analysis or
diagnostic tools. Many read fold/structure **reference data that is not committed** and
lives only on the dev box.

Route every access to that data through the single resolver:

```python
from scripts._local_ref import local_ref, LocalRefMissing
p = local_ref("some_subdir", "file.cif")   # joins under $XENO_LOCAL_REF
```

`local_ref()` resolves the `XENO_LOCAL_REF` env var (default `./XenoDesign1_local_ref`)
and raises `LocalRefMissing` — naming the env var — when the checkout is absent, so a
missing-data run fails with one clear message instead of a `FileNotFoundError` deep in a
script.
