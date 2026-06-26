# XenoDesign1 — Architecture

This document is the contributor-facing map of the `xenodesign/` package: the module layout, the
core data flow, and the two places you plug in new behaviour. For the *why* behind specific design
choices, see the architecture-decision record (ADR-lite). For runtime/Chai-input contracts, see `GPU_TESTS.md`.

The central design principle: **the loop control flow never changes per binder class.** A single
dispatcher wires per-class *hooks* into one untouched `HalluLoop`. Only *which callables* a class
injects differs.

---

## Module map

### Entry point & dispatch

- **`scripts/design.py`** — CLI front end. Parses the `--binder_class` / `--target_type` / `--search`
  axes plus knob flags, maps present flags to dotted `DesignConfig` override keys, resolves the
  config, threads the declarative coordinator/Cys flags, and calls `run_design` (or `run_length_sweep`).

- **`xenodesign/dispatch.py`** — the heart. `run_design(cfg)` is the single entry behind the CLI:
  resolve class → build target entities → seed → (restraints) → L-seed predict + double-flip → wrap
  → `HalluLoop` (greedy) or beam/anneal or ABC → referee → `JudgePanel` → `cls.report`. Three GPU/IO
  seams (`_registry`, `_make_predictor`, `targets.target_entities`) are the monkeypatch points the CPU
  dispatch test stubs out. Also hosts `run_length_sweep`, the `_run_beam` and `_run_abc` branches, and
  the `_is_mixed_chirality` guard (ABC ⇔ `cyclic` + `target_type=none`).

- **`xenodesign/config.py`** — `DesignConfig` and its sub-dataclasses (`TargetSpec`, `RestraintConfig`,
  `LoopKnobs`, `AbcKnobs`, `GateConfig`), the per-class `PRESETS`, `resolve_config` (preset ←
  config-file ← CLI overlay, dotted-key application), `resolve_binder_length` (clamp [6,50]), and
  `dump_config` (writes `resolved_config.json` for provenance).

### The binder-class contract

- **`xenodesign/classes/base.py`** — defines `SeedSpec`, the `BinderClass` **Protocol** (the only
  surface the dispatcher knows: `seed` / `ss_bias` / `restraints` / `closure` / `seq_update` /
  `accept_fns` / `objective` / `referee` / `report`), and `CLASS_REGISTRY` mapping the CLI axis
  (`alpha`/`non_alpha`/`cyclic`) to a class instance. Note the axis→case-id mapping: CLI `non_alpha`
  → benchmark `case_id 'nonalpha'`.

- **`xenodesign/classes/alpha.py`** — the **canonical** α-helical design logic (validated trimer
  D/L-ABLE loop; `scripts/design_alpha.py` is now a thin shim re-exporting it). Owns the PepMLM helix
  seed, the inverse-folding backend selector (`ligandmpnn`/`carbonara`/`mixed`), the loop objectives
  (`iptm` / `mixed` / `ipsae`), and the referee. The other two classes reuse its seq-update / objective
  / referee helpers.

- **`xenodesign/classes/non_alpha.py`** — non-α **ICK cystine-knot** binder against the 2-chain MSA'd
  HA receptor. Differs from α only in: ICK Cys-scaffold seed, **anti-α** SS-bias (knottins are
  non-helical), and the multi-chain MSA'd target. `closure` returns `[]` by default (a documented
  Chai 0.6.1 limitation rejects D-Cys disulfide COVALENT bonds — see the in-file header + memory).

- **`xenodesign/classes/cyclic.py`** — cyclic Zn-**macrocycle** design (6UFA) + geometry recall. Owns
  the mixed-chirality seed (coordinating His pinned L/D), the Zn-ligand FASTA, the His↔Zn
  metal-coordination restraints, the opt-in head-to-tail COVALENT closure bond, and the
  RMSD/Zn-N geometry scorers. Also the home of the no-target (`target_type=none`) free-peptide path.

### Targets & the chain contract

- **`xenodesign/targets.py`** — `target_entities(cfg)` builds the **fixed-context** Chai entity list
  per target chemistry (protein / multi-chain MSA'd protein / rna / dna / small_molecule / metal /
  none); the binder is appended downstream, never here. The **`metal`** branch is *gated*: it refuses
  unless the D-residue dist-restraint patch is verified-applied (else the coordination restraint
  silently drops). Defines **`ChainRoles`** — the *one* authoritative binder/target chain-letter
  assignment, derived once from entity order (`binder = chr('A'+len(entities))`) and threaded
  everywhere a consumer would otherwise hardcode a chain letter. This makes chain-misidentification
  structurally impossible across all classes (alpha→B, non_alpha→C, cyclic-metal→B, no-target→A).

### Seeding, the loop, sequence update

- **`xenodesign/seed.py`** — seed generators (`RandomSeedGenerator`, `PepMLMSeedGenerator`),
  retro-inverso, `insert_fixed_chirality`, and the **double-flip** (`reflect_binder_in_complex_from_cif`):
  predict the complex in L (in-manifold for Chai), then reflect the binder coords to produce a
  chirality-correct D seed.

- **`xenodesign/loop.py`** — the untouched **`HalluLoop`**: per-iteration `refine → sequence-update →
  score → accept`. `refine_fn` is injectable (default `truncated_refine`; a predict-wrapper for
  restrained full predicts). `accept_fn` is an optional gate; factories here build the
  `chirality_gated_accept`, `periodicity_gated_accept`, and `greedy_iptm_accept` gates, plus
  `compose_accept_fns` to AND-stack them. `LoopState` / `LoopStep` are the data carriers.

- **`xenodesign/sequence_update.py`** — the inverse-folding round-trip: mirror-out (reflect so the
  design chain is L; partner → all-atom context) → `design_fn` (LigandMPNN / CARBonAra over designable
  positions only) → mirror-back (re-encode designed L letters as D-CCD). The `_ligandmpnn_design_fn`
  adapter is the default backend.

- **`xenodesign/beam.py`** — beam-search + anneal machinery (`beam_search`, `anneal_best`,
  `BeamState`, `CostAccount`) used by the `--search beam` branch through the same per-class hooks.

### Gates & evaluation (`eval/`)

- **`eval/gate_tier0a.py`** — Tier-0a chirality gate: chiral-volume sign + φ/ψ agreement vs D-PDB
  references (MONDE-T). GPU `run_gate`; pure `aggregate_gate_report`.
- **`eval/metal_geometry_gate.py`** — the **MetalHawk** metal-coordination-geometry gate. Builds a
  "sphere PDB" from a predicted CIF, runs MetalHawk (isolated env via subprocess; it pins
  scikit-learn 1.0.2 pickles), reads the assigned geometry class (LIN/TRI/TET/…) and softmax entropy,
  and gates on `perplexity = exp(entropy)` (≈1.0 = a confident, clean geometry = pass). Best-effort /
  guarded: a gate that cannot run never vetoes. Off by default (`gates.metal_geometry`).
- **`eval/controls.py`** — within-Chai negative controls (composition-matched scramble, off-target
  helix panel) + interface-footprint/register analysis. CPU-pure half; GPU scoring is a documented recipe.
- **`eval/chirality_reality.py`** — anti-survivorship harness: full per-iteration chirality
  distribution across a run dir, mirror self-consistency, cold-start D-fold protocol.
- **`eval/watchdog.py`** — advisory GPU/RAM/disk watchdog for long campaigns (warn + flag, never kill).

### The ABC / evolutionary designer (`abc/`)

- **`abc/engine.py`** — the **fitness-agnostic** Artificial Bee Colony engine over `FoodSource`
  (identity + chirality pattern + cached nectar). Three phases per cycle (employed / onlooker / scout),
  greedy-keep, `scout_limit` stagnation re-seed, hard `eval_budget`. Imports nothing heavy; all
  randomness flows through one injected `random.Random`.
- **`abc/moves.py`** — structured chirality moves + stereochemistry priors (alternating L/D, D at
  β-turn apices, single L→D boundary) and `seed_chirality_pattern`. Honours `required` (fixed)
  handedness. Pure/deterministic.
- **`abc/fitness.py`** — the fast-cycle fitness adapter: a short (K*≈10–25 step) Chai predict of the
  single mixed-chirality peptide scored as `w_ptm·pTM + w_termini·termini_proximity` with the
  head-to-tail closure restraint only (no target-specific coordination). Guarded: a failed eval
  returns `-inf`.
- **`abc/variants.py`** — the A/B axis split: **A** = ABC owns chirality, MPNN fills identity
  per-pattern; **B** = ABC searches identity (point mutations over 20 AAs) + chirality, MPNN warm-start
  only.
- **`abc/calibration.py`** — the diffusion-steps ↔ fitness-fidelity calibration (Spearman vs the
  200-step reference) that selects the lowest faithful `K*`. Pure helpers; GPU body wired in dispatch.

### Judges (`judges/`)

- **`judges/panel.py`** — the adversarial `JudgePanel`: Layer 1 hard chirality veto, Layer 2 weighted
  composite (chirality / binding / ESM-2 PLL / optional mirror), Layer 3 `select`. `RefereeScore` is
  the per-step record.
- **`judges/plm_judge.py`** — `ESMPseudoLogLikelihood`: the fixed pretrained ESM-2 pLM scoring sequence
  naturalness (PLL), the chirality-blind naturalness term + veto input.

### The Chai backend & patches

- **`xenodesign/backends/chai_backend.py`** — `ChaiBackend` wrapping `chai_lab`: `write_inputs` (pure,
  CPU-testable), `predict` / `truncated_refine` (GPU, lazily imported). `Prediction` carries
  coords / pLDDT / iptm.
- **`xenodesign/backends/chai_truncated.py`** — the low-σ truncated-refine entry that preserves
  chirality.
- **`xenodesign/chai_patches.py`** — runtime monkeypatches for Chai 0.6.1 restraint generators. The
  **D-residue fix**: stock `add_distance_restraint` asserts a residue maps to exactly one token, but
  Chai tokenizes D / non-canonical residues *per-atom* (a D-His → 10 tokens), so the assertion fires
  and the restraint is silently dropped. The patch narrows to the residue's atom-token set and uses
  the actual decoded residue name (defeating the L-vs-D name guard). `dist_restraint_patch_verified()`
  is the predicate the `metal` target gate consults; `ensure_patches()` installs them idempotently.
- **`xenodesign/coordinators.py`** — `parse_coord_residues` for the declarative `--coord_residues`
  flag: each token is identity+position+chirality (1-letter = L, CCD code = D), generalizing beyond
  His/Zn to any donor/metal. Drives both the seed's opt-in fixed positions and the restraint rows.

### Supporting modules

`io_spec.py` (D-CCD encode/decode, FASTA build), `mirror.py` (L↔D maps, reflection),
`chirality.py` / `geometry.py` (chiral-volume, Kabsch RMSD), `scorer.py` / `metrics.py`
(design scores, sequence-quality key), `secondary_structure.py`, `ncaa_proxy.py`, `catalog.py`
(MONDE-T ncAA catalog), `pdb_extract.py`, `parallel.py` / `schedule.py`, `carbonara_backend.py`
(the alternative inverse-folding backend), `inverse_folding.py` (the `InverseFoldingBackend`
interface + `MultiCandidate`), and `benchmark/` (cases, restraint builders, seeding policies).

---

## Data flow: seed → (restraints) → predict → score → select

```
DesignConfig (preset ← config-file ← CLI)            config.py / scripts/design.py
        │
        ▼  dump resolved_config.json (provenance, BEFORE any predict)
target_entities(cfg)  ──►  ChainRoles.from_entities(...)        targets.py
        │  (fixed target context; binder appended last)         (one chain contract, threaded)
        ▼
cls.seed(cfg, target_seq)  ──►  SeedSpec (one-letter L)         classes/*.py + seed.py
        │
        ▼  [if restraints_on]  cls.restraints(...) ──► .restraints CSV   (+ chai_patches for D)
        ▼
L-seed predict (full 200-step) ──► double-flip reflect ──► D-correct seed coords    seed.py
        │  (real l_seed_iptm measured by the dispatcher)
        ▼
HalluLoop.run(init, accept_fn=cls.accept_fns(cfg))             loop.py
   per iter:  refine_fn ─► sequence_update_fn ─► score_fn (cls.objective) ─► accept
        │                  (mirror→L→design→D)    (iptm/mixed/ipsae/contrastive)
        │
        ├─ --search beam ─►  beam_search + anneal_best          dispatch._run_beam / beam.py
        └─ --search abc  ─►  abc_search (cyclic + none only)     dispatch._run_abc / abc/*
        ▼
referee (cls.referee) per step ──► RefereeScore[]              classes/*.py + judges/
        ▼
JudgePanel.combine/select  (chirality veto + composite)        judges/panel.py
        ▼
cls.report(...)  ──►  result dict + *_result.json              classes/*.py
   (dispatcher overwrites l_seed_iptm + wall_time_s with measured values)
```

The dispatcher owns two values the per-class `report` hooks cannot measure (the L-seed ipTM and the
end-to-end wall time) and threads the real values into `report` so the returned dict and the on-disk
JSON match the legacy single-class drivers byte-for-byte.

---

## Extending

### Add a new binder class

1. Implement the **`BinderClass`** protocol (`classes/base.py`): provide `case_id` and the hooks
   `seed` / `ss_bias` / `restraints` / `closure` / `seq_update` / `accept_fns` / `objective` /
   `referee` / `report`. Reuse α's helpers where the chemistry overlaps (non_alpha and cyclic do).
2. Register the instance in `CLASS_REGISTRY`, add a `PRESET` in `config.py`, and (if needed) add the
   axis value to `scripts/design.py`'s `--binder_class` choices.
3. If your class needs a new target chemistry, extend `targets.target_entities`. **Never hardcode a
   chain letter** — read `ChainRoles` (built once in `run_design`) and thread it through your
   `seq_update` / `referee`. The dispatcher already passes `roles` to hooks that accept it.
4. Add CPU tests exercising every hook with fakes (no predict). The dispatcher's three GPU seams are
   monkeypatchable, so an end-to-end dispatch test runs on CPU.

### Add a new objective

Implement a `score_fn(prediction) -> float` and surface it from a class's `objective` hook (and add
the name to `--objective` choices if it should be CLI-selectable). The α class is the reference for
`iptm` / `mixed` / `ipsae`; objectives that need a decoy (`contrastive`) compute
`score(design·target) − score(design·decoy)` — see ADR-019 in the architecture-decision record.

### Add a new gate

Acceptance gates are `accept_fn(candidate_step, current_step) -> bool` factories in `loop.py`; build
one and AND-compose it with `compose_accept_fns`. Post-hoc / selection gates (like the MetalHawk
metal-geometry gate) live in `eval/`, are best-effort/guarded (never crash a run, never veto when they
can't run), and are consulted in a class's `report`/selection rather than wired into the loop.
