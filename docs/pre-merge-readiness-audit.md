# Pre-merge readiness audit — hardware-agnosticity + readability/modularity

Read-only audit (no code edited) of the `feat/halludesign-chai-dpeptide` tip
(`28df281`) ahead of the pre-merge cleanup. Scope: make the repo safe to hand to
**external contributors + beta-testers** on a machine that is **not** `ifrit`.
Two axes: (1) hardware/host agnosticity, (2) readability/modularity. Section 3 is a
dispatchable list of TDD refactor tasks (not implemented here).

Code surveyed: `xenodesign/` package (~12.2 k LOC, 50 modules), `scripts/`
(~13.6 k LOC, 50 scripts), `tests/` (63 CPU + 13 GPU test files), `config.py`,
`README.md`, `setup.sh`, `pyproject.toml`, `tests/conftest.py`.

Headline: the **package core is in good shape** — a clean Protocol-based
`BinderClass` registry, a never-touched `HalluLoop`, lazy GPU imports so the CPU
suite is green without a GPU. The agnosticity problems are concentrated in (a)
hardcoded `"cuda:0"` defaults, (b) two hardcoded absolute install paths, (c) the
ifrit-specific Docker invocation in `run_design_smoke.sh`, and (d) ~40 host-bound
analysis scripts. The modularity problems are concentrated in (a) the inverted
`xenodesign → scripts/design_demo.py` import, (b) the two ~900–1050-line
`alpha.py` / `cyclic.py` god-modules, and (c) the legacy-driver vs. unified-dispatch
duplication.

---

## Section 1 — Hardware-agnosticity findings + fixes (prioritized)

Severity: **P0** = a fresh-clone external user hits it immediately; **P1** = breaks
on any non-2×A4500 / non-ifrit host; **P2** = cosmetic / doc-string only.

### P0-1 — Hardcoded absolute install paths in the package

| Where | Value | Why host-specific |
|---|---|---|
| `xenodesign/carbonara_backend.py:28-30` | `_CARBONARA_DIR = pathlib.Path("/home/user/CARBonAra")` (+ `.venv/bin/python`, `carbonara.py`) | The CARBonAra checkout + its venv only exist at that path on ifrit. The module imports fine (paths are module constants, lazily used), but **any** call to `carbonara_design_fn` shells out to a python that won't exist elsewhere. |
| `xenodesign/eval/metal_geometry_gate.py:276` | `["micromamba", "run", "-n", metalhawk_env, ...]` | Assumes `micromamba` on PATH and an env literally named `metalhawk`. `metalhawk_dir`/`metalhawk_env` are already parameters with defaults — but the default invocation is ifrit-shaped. |

**Fix:** read `_CARBONARA_DIR` from `XENO_CARBONARA_DIR` env (fallback to a config
field), and likewise let `metalhawk_dir` come from `XENO_METALHAWK_DIR` /
`metalhawk_env` from `XENO_METALHAWK_ENV`. Both backends already fail-soft, so the
fix is just: env-var → config field → documented "feature unavailable" message.

### P0-2 — `scripts/run_design_smoke.sh` bakes in ifrit's Docker layout + the compose-reconcile `--label` workaround

`scripts/run_design_smoke.sh` (the only committed smoke entry-point) hardcodes:
- `cd /home/user/claude_projects/XenoDesign1` (line 8),
- `-v /home/user/chai_weights_cache:/chai-lab/downloads` and
  `-v /home/user/.cache/huggingface:...` (lines 12-13),
- `gradio_design-gradio-design:latest` image tag (line 14),
- the **compose-reconcile workaround** labels
  `--label com.docker.compose.project=xeno_oneoff --label com.docker.compose.oneoff=True`
  (line 10) — this exists only to stop ifrit's gradio compose stack from
  reaping ad-hoc containers (per memory: the chai image's baked compose label).

Why host-specific: an external user has a different image tag, different cache
paths, a different repo root, and **no gradio compose stack at all** — the labels
are dead weight (at best) or confusing (at worst). The README already abstracts the
image/cache as placeholders pointing at the gitignored `RUNBOOK.local.md`; the
committed `.sh` contradicts that by hardcoding ifrit's values.

**Fix:** parameterize via env with documented defaults:
`XENO_REPO_ROOT="${XENO_REPO_ROOT:-$PWD}"`,
`CHAI_IMAGE="${CHAI_IMAGE:?set to your chai 0.6.1 image}"`,
`CHAI_WEIGHTS="${CHAI_WEIGHTS:?host weights cache}"`. Gate the compose `--label`
lines behind `XENO_COMPOSE_GUARD=1` (default off) with a one-line comment that they
are an ifrit-only workaround for a co-located compose stack.

### P1-1 — `"cuda:0"` is the default device in ~12 places (no `cuda.is_available()` fallback at the config layer)

`config.py:17` (`LoopConfig.device`), `config.py:151` (`DesignConfig.device`),
`classes/alpha.py:51`, `classes/cyclic.py:62`, `backends/chai_backend.py:130`,
`backends/chai_truncated.py:94`, `eval/controls.py:433`,
`eval/chirality_reality.py:56`, `abc/calibration.py:231`, plus every CLI default in
`scripts/*` (`--device cuda:0`).

Why host-specific: hard-fails on a CPU-only contributor box, an Apple-silicon
(`mps`) box, or a single-GPU box where the only device is addressed differently
under `CUDA_VISIBLE_DEVICES`. Two modules already do it right —
`sequence_update.py:317` and `judges/plm_judge.py:94` use
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`. The fix is to make
the rest consistent with those two.

**Fix:** add one helper `xenodesign.device.default_device()` returning
`os.environ.get("XENO_DEVICE")` → else `"cuda" if torch.cuda.is_available() else
"cpu"`, and make every `device="cuda:0"` default call it (lazily, so CPU import
stays torch-free where it is today). Keep `"cuda:0"` only inside the
single-visible-GPU worker subprocess (`run_parallel.py`, `generate_and_select.py`)
where it is correct-by-construction (parent pins `CUDA_VISIBLE_DEVICES`).

### P1-2 — `n_gpus=2` and "20 GB VRAM, 1 worker saturates it" baked into the parallel runners

`scripts/run_parallel.py` (`n_gpus: int = 2` defaults at lines 338, 548; docstrings
"default 2 for ifrit's A4500s", "20 GB VRAM") and
`scripts/generate_and_select.py` (`n_gpus: int = 2` at line 321, `--gpus` default).

Why host-specific: `2` is ifrit's GPU count; `workers_per_gpu=1` is tuned to the
A4500's 20 GB. On a 1-GPU box the default over-subscribes; on a 48 GB box it
under-subscribes. `plan_worker_slots` is already pure and parameterized — the only
problem is the **default value** and the doc-string framing.

**Fix:** default `n_gpus` to `torch.cuda.device_count()` (fallback 1) instead of the
literal `2`; reword the 20 GB comment to "tune `workers_per_gpu` to your VRAM". No
logic change — just don't hardcode ifrit's topology as the default.

### P1-3 — `ResourceWatchdog` default `disk_path="/scratch"`

`xenodesign/eval/watchdog.py:27`. `/scratch` is an HPC/ifrit convention; on a laptop
it doesn't exist, so the disk sample silently degrades. **Good news:** the VRAM
threshold is a **fraction** (`vram_frac_warn=0.92`), not a hardcoded `20480` MB — so
the watchdog is otherwise hardware-agnostic (the `20480` only appears in a docstring
doctest). **Fix:** default `disk_path="."` (already the documented fallback in the
docstring) and let `/scratch` be an explicit opt-in.

### P1-4 — `XenoDesign1_local_ref/` referenced 60+ times across scripts (gitignored data dir + symlink reliance)

`scripts/select_contrastive.py`, `run_cyclization_calibration.py`,
`build_9dxx_complex_fastas.py`, `thread_register_decoy.py`, `score_ddg.py:448`,
`run_ddg_confirmation.py:25`, `_repanel_t20_v2.py:15`,
`validate_loop_hits_register.py:46`, and ~10 more all default `--out_root` /
read inputs under `XenoDesign1_local_ref/...` (a gitignored local benchmark/ref
data tree, partly a symlink on ifrit). `xenodesign/inverse_folding.py:122` only
*mentions* it in a comment (clean). External users have none of this data, so these
scripts fail at runtime.

**Fix:** these are mostly **Tier-3 analysis throwaways** (see §2). Make the path a
single `XENO_LOCAL_REF` env var (default `./XenoDesign1_local_ref`) resolved in one
helper, and have scripts emit a clear "local reference data not found — this is an
ifrit-internal analysis script" message instead of a raw `FileNotFoundError`.
Document the Tier-3 status in `scripts/README.md`.

### P2-1 — `setup.sh` is stale upstream BoltzDesign1 setup, not the xenodesign workflow

`setup.sh` builds a `conda create -n boltz_design` environment, installs
PyRosetta/Boltz weights, and never mentions chai, the Docker image, or the
`xenodesign` package. It is the **inherited BoltzDesign1 installer** and will
mislead a new contributor into the wrong setup path.

**Fix:** either retire it (the real path is the chai container, already in the
README/RUNBOOK) or rename to `setup_boltzdesign_legacy.sh` and add a top comment
that xenodesign uses the chai container, not this.

### P2-2 — `/tmp/beam*` default out-dirs + `PYTHONPATH=/work` doc-strings

`xenodesign/beam.py:201,260,334` default `out_dir="/tmp/beam"` (fine on Linux,
breaks on Windows; minor). Many script docstrings say `PYTHONPATH=/work` (the
in-container repo mount) — harmless, but a contributor running outside the container
needs `PYTHONPATH=$PWD`. **Fix:** doc-only; mention both invocations.

### Non-issues confirmed (so reviewers don't re-flag them)

- **No `tests/__init__.py` shim exists** in the repo. (The memory note about a
  `tests` package conflict is about ifrit's stray `~/.local` site-package + a
  micromamba→system-python mismatch — a *runner-env* gotcha, not a repo file. It
  should be documented in `scripts/README.md` as a "running the tests on ifrit"
  note, but there is nothing in-repo to fix.)
- **VRAM gating is fractional, not a hardcoded 20 GB** (`watchdog.py`).
- **Worker subprocesses correctly use `cuda:0`** because the parent pins
  `CUDA_VISIBLE_DEVICES` — that is correct, not a bug.
- **GPU imports are lazy** throughout the package, so `import xenodesign` works on a
  CPU box — the agnosticity work is about *runtime defaults*, not import-time.

---

## Section 2 — Readability / modularity findings

### 2.1 — Inverted dependency: the `xenodesign` package imports from `scripts/design_demo.py` (the single biggest smell)

`scripts/design_demo.py` (538 LOC) is nominally a *demo CLI* (`if __name__ ==
"__main__"` + argparse), but it is the **de-facto home of core plumbing** imported
by the package:

| Importer (package) | Symbols pulled from `scripts.design_demo` |
|---|---|
| `xenodesign/classes/alpha.py` | `_all_atoms_from_chain`, `_backbone_array_from_residues`, `_best_cif_path`, `_chirality_violation_frac_from_cif`, `_LoopBackendWrapper`, (`_PredictBackendWrapper` cond.) |
| `xenodesign/dispatch.py` | `_best_cif_path`, `_LoopBackendWrapper`, `_PredictBackendWrapper` |
| `xenodesign/classes/cyclic.py` | `_best_cif_path` (conditional) |
| `xenodesign/abc/calibration.py` | `_best_cif_path` (conditional) |

A package importing private (`_`-prefixed) names from a `scripts/` CLI is a
dependency inversion: it means the package cannot be installed/used without the
`scripts/` dir on `sys.path`, and a new contributor reading `dispatch.py` has to go
spelunking into a "demo" to find the backend wrapper. These 6 helpers are pure
CIF/entity plumbing — they belong in the package.

**This is the #1 readability/modularity blocker for external contributors.**

### 2.2 — Two god-modules: `classes/alpha.py` (1054 LOC) and `classes/cyclic.py` (889 LOC)

`alpha.py` is the canonical α driver *and* the `Alpha` BinderClass adapter *and* the
standalone CLI's helper home *and* the monkeypatch-compat surface for the legacy
`scripts/design_alpha.py` shim. It mixes: restraint building, seed building, seq-update
factory, three different objective factories (`mixed`/`ipsae`/`_loop_score_fn`),
referee construction, the full `run_alpha_design` driver, and result assembly — ~30
top-level defs. `cyclic.py` is similar (intramolecular objective, no-target seed,
Zn-restraint, S2-symmetry handling, report). A new contributor wanting to add a
binder class has a 1000-line file as the only worked example, and it is hard to tell
which parts are the *contract* (the 9 `BinderClass` hooks at the bottom) vs the
*α-specific internals*.

The `BinderClass` Protocol + `CLASS_REGISTRY` design (in `classes/base.py`) is
**genuinely good** — the extension seam is clean and documented. The problem is the
*reference implementations* are monolithic, so the clean seam is buried.

### 2.3 — Legacy-driver vs. unified-dispatch duplication

There are now **two** ways to run α: `run_alpha_design` (in `alpha.py`, the validated
legacy driver) and `dispatch.run_design` (the unified, class-parameterized path that
"reproduces the wiring" of `run_alpha_design`). `dispatch.py` carries careful comments
about byte-for-byte preserving the legacy behaviour (`_PredictAdapter`,
`l_seed_iptm`/`wall_time_s` overwrite). This is fine as a *migration* state but is a
trap for a new contributor: which path is canonical? The monkeypatch-compat shim
(`scripts/design_alpha.py` re-exports + `alpha._shim()` call-time lookups) adds a
third indirection that exists purely to keep old tests green.

**Recommendation:** once dispatch is the single path, retire `run_alpha_design` (or
make it a thin wrapper over `dispatch.run_design`) and drop the `_shim()` machinery —
but only behind tests that pin dispatch == legacy output first.

### 2.4 — `scripts/` is a 50-file dumping ground with no tiering

8 core drivers (`design*.py`, `run_parallel.py`, `generate_and_select.py`,
~3150 LOC) are mixed in with ~40 one-off analysis/diagnostic scripts (`analyze_*`,
`score_*`, `diagnose_*`, `gate_*`, `make_*`, `_*`, calibration runners; ~10.5 k LOC),
most of which hardcode `XenoDesign1_local_ref/` or `/work` and will not run for an
external user. There is no `scripts/README.md` saying which are which.

### 2.5 — Unclear interfaces / naming traps a contributor will hit

- **`non_alpha` (CLI axis) vs `nonalpha` (benchmark case_id)** — the underscore
  mismatch is real and documented in `base.py`, but it is a foot-gun.
- **Binder chain varies by class** (`B` for alpha, `C` for non_alpha, `B`/`A` for
  cyclic) — handled correctly via the `ChainRoles` contract threaded from dispatch
  (good!), but the α module still carries `_RUN_BINDER_CHAIN` fallback constants with
  long explanatory comments about inverted chains; a contributor must read all of it.
- **`config.py` resolution order** (PRESET → `--config-file` → CLI dotted overrides →
  `resolve_binder_length` clamp) is correct and well-commented, but the dotted-key
  overlay (`_apply_dotted` vs `_overlay`) has two near-duplicate code paths.

### 2.6 — The genuinely good parts (keep / hold up as the model)

- `classes/base.py` — the `BinderClass` Protocol + `SeedSpec` + `CLASS_REGISTRY` is a
  textbook extension seam; stub classes raise loudly so the registry is testable
  before real bodies land.
- `xenodesign/loop.py` (421 LOC) — `HalluLoop` is small, the accept-gate combinators
  (`compose_accept_fns`, `chirality_gated_accept`, `periodicity_gated_accept`,
  `greedy_iptm_accept`) are clean, composable, and never class-specific.
- **Lazy GPU imports + pure-helper split** — `chai_backend.write_inputs`/`parse_*`
  are CPU-testable; heavy deps deferred to call time. 63 CPU test files run with no
  GPU/Docker.
- `config.py` dataclass tree with `PRESETS` + provenance dump
  (`resolved_config.json`) is a solid, discoverable config story.
- `watchdog.py` — fractional VRAM thresholds, `pynvml`→`nvidia-smi` fallback,
  pure `parse_nvidia_smi_csv` / `evaluate_thresholds` helpers with doctests.

---

## Section 3 — Recommended TDD refactor tasks (dispatchable; NOT implemented here)

Each task = title + what + the test that proves it. Ordered by leverage. They are
independent unless noted; **none** require touching `HalluLoop`'s control flow.

### Hardware-agnosticity tasks

**HW-1 — Central `default_device()` helper; kill scattered `"cuda:0"` defaults.**
*What:* add `xenodesign/device.py::default_device()` = `XENO_DEVICE` env → else
`"cuda" if torch.cuda.is_available() else "cpu"`; route `config.py` (both dataclasses),
`alpha.py`, `cyclic.py`, `chai_backend.py`, `chai_truncated.py`,
`controls.py`, `chirality_reality.py`, `abc/calibration.py` defaults through it.
*Test:* `test_default_device` — monkeypatch `torch.cuda.is_available` False → returns
`"cpu"`; set `XENO_DEVICE=mps` → returns `"mps"`; assert `LoopConfig()`/`DesignConfig()`
device == `default_device()` (not the literal `"cuda:0"`). CPU-only, no GPU.

**HW-2 — Externalize the CARBonAra + MetalHawk paths.**
*What:* `carbonara_backend._CARBONARA_DIR` ← `XENO_CARBONARA_DIR` env (fallback const);
`metal_geometry_gate` reads `XENO_METALHAWK_DIR` / `XENO_METALHAWK_ENV`.
*Test:* `test_carbonara_path_from_env` — set `XENO_CARBONARA_DIR=/tmp/x`, reload module,
assert `_CARBONARA_DIR == Path("/tmp/x")`; `test_metalhawk_unavailable_failsoft` — unset
env + nonexistent dir → `metal_geometry_gate` returns a pass-through `GateResult`
(`passed=True, ok=False, error=...`), never raises. CPU-only (subprocess never spawned).

**HW-3 — `n_gpus` defaults to detected device count, not literal 2.**
*What:* `run_parallel.run_cases` + `generate_and_select` default
`n_gpus = torch.cuda.device_count() or 1`; reword the 20 GB doc-strings.
*Test:* `test_n_gpus_default_from_devicecount` — monkeypatch `torch.cuda.device_count`
→ 1, assert `plan_worker_slots` default planning yields 1 slot; → 4 yields 4. Pure,
CPU-only (these planners are already importable without torch).

**HW-4 — Parameterize `run_design_smoke.sh` + gate the compose `--label` workaround.**
*What:* env-driven `XENO_REPO_ROOT`/`CHAI_IMAGE`/`CHAI_WEIGHTS`; wrap the two
`com.docker.compose.*` labels behind `XENO_COMPOSE_GUARD=1` (default off).
*Test:* `test_smoke_script_no_hardcoded_host` (shell/grep test) — assert the script
contains no literal `/home/user`, no literal `gradio_design-gradio-design`, and the
`--label` lines appear only inside the guard branch. Static, CPU-only.

**HW-5 — `watchdog` default `disk_path="."`; `/scratch` opt-in.**
*What:* flip the default. *Test:* extend `test_watchdog` — `ResourceWatchdog()` default
samples cwd, not `/scratch`; assert no degraded sample on a box without `/scratch`.

**HW-6 — `XENO_LOCAL_REF` single resolver for analysis scripts + clear "data missing" error.**
*What:* one `scripts/_local_ref.py::local_ref(*parts)` resolving
`XENO_LOCAL_REF` (default `./XenoDesign1_local_ref`); the Tier-3 scripts call it and
raise a friendly message if absent.
*Test:* `test_local_ref_resolver` — env set → joins correctly; absent dir → raises
`LocalRefMissing` with a message naming the env var. CPU-only.

### Readability / modularity tasks

**MOD-1 — Move shared CIF/backend plumbing out of `scripts/design_demo.py` into the package.**
*(Highest leverage.)* *What:* create `xenodesign/demo_runtime.py` (or
`xenodesign/cif_io.py` + `xenodesign/backends/wrappers.py`) holding
`best_cif_path`, `all_atoms_from_chain`, `backbone_array_from_residues`,
`chirality_violation_frac_from_cif`, `LoopBackendWrapper`, `PredictBackendWrapper`;
have `alpha.py`/`cyclic.py`/`dispatch.py`/`abc/calibration.py` import from the package;
leave `scripts/design_demo.py` importing from the package (demo only).
*Test:* `test_no_package_imports_from_scripts` — walk `xenodesign/**.py`, assert no
`from scripts` / `import scripts` anywhere; plus existing alpha/dispatch tests stay green
(they already exercise these symbols). CPU-only, no behaviour change.

**MOD-2 — Pin `dispatch.run_design` == legacy `run_alpha_design` output, then thin the legacy driver.**
*What:* first add a characterization test asserting both paths return the same report
dict for a fixed CPU-faked case (reuse the dispatch unit-test fakes); *then* refactor
`run_alpha_design` into a thin wrapper over `dispatch.run_design` and delete the
`_shim()`/re-export compat once tests confirm parity.
*Test:* `test_dispatch_matches_legacy_alpha` — same seed/case/CPU-fake backend →
`run_design(cfg)` report == `run_alpha_design(...)` report (key-by-key). CPU-only.
*(MOD-2's refactor step is gated on this test passing first.)*

**MOD-3 — Split `classes/alpha.py` into contract vs. internals.**
*What:* extract α restraint/seed/objective/referee internals into
`classes/_alpha_internals.py` (or a small `classes/alpha/` package), leaving
`classes/alpha.py` as: the `Alpha` BinderClass adapter (the 9 hooks) + `run_alpha_design`.
Same for `cyclic.py`. No behaviour change.
*Test:* existing `test_design_alpha.py` / `test_classes_cyclic_notarget.py` stay green
verbatim (they import by name — keep re-exports); add `test_alpha_module_under_400_loc`
asserting the public `alpha.py` shrank (a guardrail, not a hard rule).

**MOD-4 — Add `scripts/README.md` tiering + a `scripts/_local_ref` guard.**
*What:* document Tier-1 (core drivers), Tier-2 (`design_demo.py` = example, not a
library), Tier-3 (analysis/diagnostic, ifrit-local data). Pairs with HW-6.
*Test:* `test_scripts_readme_lists_core_drivers` — assert the 8 core driver filenames
appear in `scripts/README.md` (cheap guard against drift). CPU-only.

**MOD-5 — De-duplicate `config._apply_dotted` / `_overlay`.**
*What:* fold the dotted-key override and the nested-dict overlay into one recursive
setter. *Test:* `test_config_overlay_and_dotted_equivalence` — applying
`{"loop": {"iters": 9}}` (overlay) and `"loop.iters"=9` (dotted) produce identical
configs; existing `resolve_config` tests stay green.

**MOD-6 — Retire or rename the stale `setup.sh`.**
*What:* rename to `setup_boltzdesign_legacy.sh` (or delete) and add a one-line README
pointer to the chai-container workflow.
*Test:* `test_no_boltz_design_setup_in_readme_path` (static) — assert README's quick-start
does not point at `conda create -n boltz_design`. Low-tech guard.

**Total proposed TDD tasks: 12** (6 hardware-agnosticity HW-1…HW-6, 6
readability/modularity MOD-1…MOD-6). All have a CPU-only proving test; none touch
`HalluLoop`. Suggested dispatch order: **HW-1, HW-4, MOD-1** first (highest
contributor-facing leverage), then HW-2/HW-3/HW-5/HW-6 (host-agnostic defaults),
then MOD-2→MOD-3 (sequence MOD-2 before MOD-3), then MOD-4/MOD-5/MOD-6 (cleanup).
