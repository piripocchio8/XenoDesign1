# XenoDesign1 — A general hallucination framework for mixed-chirality, D-peptide and metallopeptide binder design on Chai-1

**Draft white paper · 2026-06-25**
**Status: DRAFT (for internal review / eventual publication). TODOs flag numbers not yet measured.**

The XenoDesign1 project and collaborators.
Software, runs and figures generated on a dual-GPU workstation (2× NVIDIA RTX A4500, 20 GB each).

> **Scope of this draft.** This paper reports a *software-and-method* contribution — a general,
> multi-class hallucination designer for non-canonical (mixed L/D, all-D, metallopeptide and cyclic)
> binders built on the Chai-1 structure model — together with end-to-end GPU validation on a small set of
> test cases. It does **not** yet report wet-lab validation; every structural number is a Chai-1 / MetalHawk
> in silico quantity and should be read as such. Where a number is missing it is marked **TODO**.

---

## 1. Motivation

Almost all of contemporary machine-learning protein design (RFdiffusion, ProteinMPNN/LigandMPNN,
AlphaFold/Boltz/Chai-based hallucination) is trained on, and therefore confined to, the natural L-amino-acid
chemical space. Yet some of the most useful and pharmacologically interesting peptides live *outside* that
space:

- **Proteolytic stability.** D-amino-acid and mixed-chirality peptides are largely invisible to natural
  proteases, so a D- or heterochiral binder can retain activity in serum / gut where an all-L analogue is
  degraded. This is the established rationale behind the gp41 D-peptide inhibitor family (PIE7/PIE12, etc.)
  and the MDM2 DPMI series.
- **Expanded chemical and topological space.** Mixed L/D backbones, head-to-tail macrocycles, and
  non-canonical side chains reach folds and surfaces that the 20 L-amino-acid alphabet cannot. Cyclization
  pre-organizes the binder, trading conformational entropy for affinity and stability.
- **Metalloprotein / metallopeptide design.** De novo metal-coordinated peptides — the AMBRA group's core
  research line — require *placing* a metal site (e.g. a tetrahedral [Zn(His)₄]) with the correct donor
  geometry, often using a mix of L- and D-His to satisfy the coordination polyhedron in a short scaffold.
  No general ML designer handles metal geometry as a first-class objective.

The combination — a designer that can (i) seed and optimize a binder of free chirality, (ii) close it into a
macrocycle, and (iii) build and *score* a metal-coordination geometry — is exactly the gap XenoDesign1
targets.

---

## 2. State of the art and the gap

**Mixed-chirality macrocycles, the Rosetta line.** The only mature, validated route to designed
mixed-chirality macrocycles is the Rosetta `GenKIC` (generalized kinematic closure) family:
Bhardwaj et al. (*Nature* 2016) designed and crystallographically validated hyperstable constrained
peptides; Hosseinzadeh et al. (*Science* 2017) mapped the accessible landscape of L/D macrocycles; and
Mulligan et al. (PNAS 2021) scaled mixed-chirality macrocycle design computationally. These are
**physics/sampling-based**, depend on the Rosetta energy function and explicit kinematic closure, and are
not learned models — they are accurate but heavy and bespoke per target.

**Learned inverse folding.** ProteinMPNN and LigandMPNN are the dominant learned sequence designers, but
they are trained on natural L-proteins; at a D position they emit L-biased, wrong-handed preferences. (A key
XenoDesign1 enabling observation, §4, is that the *coordinate-only* MPNN adapter — which reads only N/CA/C/CB
coordinates, never residue names — accepts D/ncAA backbones as pure geometry, sidestepping this for the
*backbone-conditioned* case.)

**Hallucination / ML structure models.** BoltzDesign1 and analogous hallucination loops optimize a sequence
against a folding model's own confidence. Chai-1 is an open, AlphaFold3-class all-atom model that natively
tokenizes non-canonical residues (parenthesized `(DAL)` codes) and ligands, and accepts geometric restraints
— making it, in principle, a single oracle for D-peptides, macrocycles and metal sites.

**The gap.** There is **no general, learned mixed-chirality designer.** Rosetta GenKIC is physics-only and
per-target; MPNN-family tools are L-only; hallucination frameworks are L-only and single-class. Nobody ties
together (a) from-scratch seeding of free-chirality binders, (b) per-class structure objectives (helix /
non-helix / cyclic / metal), (c) D-aware restraints (covalent closure, metal coordination) and (d) an
explicit metal-geometry gate, inside one ML loop. XenoDesign1 is an attempt at that synthesis, using Chai-1
as the structure model (no Boltz weights required — a deliberate fork choice; ADR-001..009).

---

## 3. Design and specification

### 3.1 One entry point, two orthogonal axes

XenoDesign1 consolidates three independently-grown driver scripts into **one parameterised dispatcher over
one shared hallucination loop** (`scripts/design.py → xenodesign/dispatch.run_design`). The design exposes
two deliberately orthogonal axes:

```
scripts/design.py --binder_class {alpha, non_alpha, cyclic}
                  --target_type  {protein, rna, dna, small_molecule, metal, none}
                  [--search {greedy, beam, abc}] [overrides…] [--config-file FILE]
```

- **`binder_class`** = *what is designed* (the binder's seeding, secondary-structure bias, closure, accept
  gates): `alpha` (single amphipathic D-helix), `non_alpha` (non-helical / cystine-knot-style all-D binder),
  `cyclic` (single-chain mixed-chirality macrocycle, with or without a metal).
- **`target_type`** = *what it is designed against* (the fixed context entities + the binder↔target
  restraint): a protein (single- or multi-chain, MSA'd), a nucleic acid, a small molecule, a **metal** ion
  (as a Chai ligand + coordination restraint), or **none** (intramolecular / free cyclic peptide).

The two axes are separable: multi-chain MSA'd protein targets, for example, fall out of the `protein` builder
and benefit *every* binder class, not just `non_alpha`.

### 3.2 Shared core + per-class hooks (the dispatcher contract)

The single `HalluLoop` (`xenodesign/loop.py`, untouched by the refactor) already takes its structure step,
sequence-update, score function and accept gate as **injected callables**. Each binder class implements a
small `BinderClass` protocol (`xenodesign/classes/base.py`) supplying those callables —
`seed / ss_bias / restraints / closure / accept_fns / objective / referee / report` — and the dispatcher
wires them. A new binder class is a new file under `classes/`; no edit to the dispatcher, config or loop.
`config.py` resolves a `DesignConfig` by **PRESET → `--config-file` → CLI flags** precedence and dumps the
fully-resolved config to `out_dir/resolved_config.json` for provenance (every run is reproducible from that
file alone).

### 3.3 From-scratch unified seeding (no reference-binder leakage)

A core scientific-integrity principle (memory: *seeding from scratch*): **never seed from the real binder**
(not its sequence, scaffold or length). One unified PepMLM seeding path serves all classes — *target-conditioned*
masked-fill when a target is present, *unconditional* (`target_seq=""`) when there is none — with the binder
length a free parameter (`--binder_length`, 6–50; per-class default; `--length_sweep` ladder). Scaffolding
(e.g. His coordinators for a metal site, Cys for a knot) is **opt-in and declared**, never baked in. This is
what lets the per-class GPU validation in §6 be an honest *de novo* test rather than a recapitulation.

### 3.4 The ChainRoles contract

Chains are **declared once** in the dispatcher and threaded everywhere, rather than assumed. The binder chain
*varies by class* (alpha → chain B; non_alpha → chain C behind a 2-chain target; cyclic-metal → chain B
behind the Zn ligand on chain A; free cyclic → chain A alone). Hardcoding "binder = chain B" was the root of
two distinct bugs (§5). `ChainRoles` names `targets=(…)` and `binder=…` once; the seq-update extractor, the
restraint builder and the MSA router all read it, so chain order can never silently drift.

---

## 4. Implementation and scope

### 4.1 The loop and the gates

The shared loop is, per ADR-021, a **noise-dominated sampler**: the best design is the cheapest per-predict
sampler drawn many times and max-selected; depth/breadth only buy more lottery tickets. Selection therefore
leans on *gates* (categorical filters) more than on a single continuous objective:

- **Chirality gate** — rejects designs whose predicted backbone drifts off the intended handedness
  (`loop.chirality_gated_accept`).
- **Periodicity gate** — heptad autocorrelation at lag-7 (`loop.periodicity_gated_accept`); the decisive,
  noise-independent lever for *register-capability* in coiled-coil-like α designs (ADR-025). Register decoys
  are only meaningful at ≥7-residue circular shifts (3/4-shifts do not move residues off the interface face;
  ADR-011/§7 of the multiclass design).
- **MetalHawk metal-geometry gate** (`xenodesign/eval/metal_geometry_gate.py`) — see §4.2.
- A composition floor (anti poly-Ala) + an ESM-2 pseudo-log-likelihood (PLL) veto, applied via the panel
  referee.

### 4.2 The MetalHawk metal-geometry gate

This is the metal analogue of the periodicity gate: instead of hardcoding ideal donor-atom geometries
(brittle), we let **MetalHawk** (Sgueglia & Vrettas et al., *JCIM* 2024) — a trained ANN — *assign* a
coordination class and read its confidence.

- **Wiring.** From a predicted CIF we build a "sphere PDB" (the central metal + its first-shell donors,
  auto-detected: N/O/S/Se/Cl/Br/I, not only His-N) and call MetalHawk in **its own isolated env via
  subprocess** (it pins scikit-learn 1.0.2 pickles, so it is never imported into the repo interpreter).
  MetalHawk returns `(class_index, entropy)` over its 7 geometry classes
  `(LIN, TRI, TET, SPL, SQP, TBP, OCT)`.
- **Perplexity threshold.** MetalHawk has no "perplexity" field; we derive **perplexity = exp(entropy)** =
  the *effective number of competing geometry classes* (1.0 = one class certain; ln 7 → ~7 = maximally
  confused). The gate passes when **perplexity ≤ threshold**, default `DEFAULT_PERPLEXITY_THRESH = 1.5`,
  tunable via `cfg.gates.metal_perplexity_thresh`.
- **Safety/altitude.** Best-effort and guarded: every entry point catches its own failure and returns a
  pass-through (`passed=True, ok=False`) so a gate that *cannot* run never vetoes a design. Off by default
  (`cfg.gates.metal_geometry`); the sphere-PDB build + parsing + threshold logic is pure CPU and unit-tested
  without MetalHawk installed.

### 4.3 D-residue Chai patches (runtime monkeypatches)

Chai-1 0.6.1 is L-centric in two restraint code paths; XenoDesign1 installs idempotent runtime patches via
`xenodesign/chai_patches.ensure_patches()`:

1. **`token_dist` residue match for D coordinators** (`_patch_dist_restraint_match`). CONTACT/COVALENT
   restraints go through `token_dist_restraint`, which builds a residue mask and asserts a unique residue,
   then asserts the tokenized name equals the restraint name. On a D coordinator (e.g. `DHI`) the
   index-match *did not narrow* — the mask collapsed to all residues and the restraint was **silently
   dropped** (the job completed as a *free* predict, ipTM ~0.12, no error). The patch repairs the residue
   match so a D/non-canonical token at the named index is uniquely selected, and relaxes the name assertion
   to the actual tokenized name. Verified live: `[patch] token_dist repaired: left 'HIS'->'DHI'…`.
2. **Covalent bonds on D-residues** (`_patch_covalent_bond_match`). Chai maps the one-letter via the L-only
   `restype_1to3` and matches `token_residue_name == <L name>` exactly, so a head-to-tail closure or
   disulfide whose endpoint is a D-residue is rejected. The patch supplies the D-CCD synonym so the covalent
   bond is *consumed* (atom pair built, not dropped). **Carried limitation:** even patched, certain D-Cys
   disulfide / D-His closure cases remain blocked upstream (memory: *chai D-residue covalent limitation*);
   those classes default to linear/emergent closure and the builder is kept for the L-terminus / future-chai
   path.

A separate ADR-004 hard constraint remains: an all-D peptide needs **≥1 canonical residue** (e.g. a Gly) to
tokenize at all — this surfaced as the free-cyclic seeding caveat in §6.4.

### 4.4 Declarative coordinator flags

Coordinators are declared, not hardcoded: `--coord_residues 'H6,DHI12,H18,DHI24'` (and `--cys_positions`)
pin which residues are His/Cys and at what handedness, feeding both the seed (`insert_fixed_chirality`) and
the restraint builder. This is how the 6UFA Zn site (His 6/12/18/24, L/D/L/D) is specified on the command
line.

### 4.5 The ABC / EA mixed-chirality designer

Inverse folding on a *mixed* L/D backbone has no clean solution: the "design over L by double reflection"
trick works only for a *homochiral* all-D binder (the exact global mirror of an all-L one). A mixed backbone
has no global reflection onto an all-L structure. So for mixed-chirality classes XenoDesign1 adds an
**Artificial Bee Colony (ABC) / evolutionary search** (`xenodesign/abc/`) whose fitness oracle is a *fast,
low-diffusion-step Chai cycle* + a parity-aware / intramolecular objective.

- **Algorithm.** A *food source* = `(identity, chirality_pattern, last_structure, nectar)`. Employed bees
  perturb (chirality move + identity move in Variant B); onlookers roulette-select high-nectar sources and
  exploit; scouts replace sources stagnant for `scout_limit` cycles with a fresh PepMLM draw + a new
  chirality prior. Chirality is a *structured* search, not blind: priors are alternating L/D, D-at-turn-apex,
  and metal-coordinator required handedness. A hard Chai-eval budget bounds cost. The engine is pure
  Python/numpy and CPU-tested with a synthetic fitness.
- **Fast-cycle fitness.** From the cyclization calibration (§6, and `2026-06-25-cyclization-calibration.md`),
  the operating point is **K\* = 10–25 diffusion steps, start = full predict** with the head-to-tail covalent
  closure bond. The objective is **`0.7·pTM + 0.3·termini_proximity`** at K\*: pTM is the one term that
  cleanly and stably separates real cyclization from a strained homochiral control (ΔpTM ≈ +0.21, flat from
  K=10 to 200), and the C–N termini-distance proxy is the essential secondary signal for short peptides where
  Chai's pTM is depressed. (Crucially, the covalent closure bond is **soft conditioning, not a geometric
  clamp** — the cycles do *not* reliably close to a peptide-bond distance, so absolute "closed?" is not a
  usable gate; pTM + the C–N proxy are.)
- **Variant A vs B.** *Variant A* — ABC searches the **chirality pattern**; LigandMPNN (coordinate-only
  subset-reflection) fills identity per pattern (identity has a cheap prior, don't search it blindly).
  *Variant B* — ABC searches **identity + chirality**; MPNN is warm-start only. The plan was to **pilot both
  and let convergence-per-GPU-hour decide** (§6.5).

### 4.6 Scope and deferrals

In scope and validated: the dispatcher + DesignConfig + BinderClass + all three class modules (alpha
migrated, non_alpha promoted to a real loop, cyclic migrated), the full `protein` target builder
(single/multi-chain + MSA), the `metal` target type behind the patched restraint, and the ABC engine for
mixed-chirality cases. Deferred (named, not designed in detail): full rna/dna/small_molecule validation, a
metal-design *campaign*, in-loop contrastive negative design (T01 — the standing top scientific priority for
register-specificity), and hard ring closure (the bond feature does not geometrically close a ring; a
post-hoc cyclic-relax term would be needed).

---

## 5. Bug fixes

- **Wrapper / chain-indexing bug (the 31→176 non_alpha balloon).** The shared loop's sequence-update
  extractor read the wrong chain (it picked up HA2, a 176-residue target chain, instead of the 31-residue
  binder), so the binder ballooned 31→176 residues each iteration and OOM'd on the 20 GB A4500. The bug was
  **not** a memory ceiling — the real complex fits in ~18 GB (§6.2). Fix: declare the binder as the **last
  chain** in the entity list and read it via `ChainRoles` everywhere
  (`make_alpha_seq_update_fn` / dispatcher). Memory: *9DXX OOM = seq-update chain bug* and
  *declare chains, don't assume*.
- **ABC dead termini term (`_cif_path` never populated).** The ABC fitness reads `pred._cif_path` to compute
  the C–N termini distance, but the real `ChaiBackend.predict` returned a `Prediction` with no such
  attribute, so `getattr(pred, "_cif_path", None)` was always `None` and the fitness silently collapsed to
  `0.7·pTM` only — the whole 0.3 termini (cyclization) term was **dead on the GPU path**. Fix: populate
  `_cif_path` in `load_prediction`.
- **Variant A double-encoded identity → permanent `-inf`.** `abc_variant_a_design_fn` returned the
  parenthesized `(DXX)` mixed-chirality FASTA, but the engine carries that string as the *plain* identity and
  the fitness re-applies `mixed_chirality_fasta` → `KeyError "unknown amino-acid letter '('"` → fitness
  returned `-inf` for *every* MPNN candidate, freezing Variant A on its seed. Fix: return `result.one_letter`
  (the fitness owns the per-position emit); regression test added.
- **ABC convergence history not persisted.** `_run_abc` captured the per-cycle history but only saved
  `n_cycles`. Fix: persist `"history"` in `abc_result.json` so convergence curves are recoverable.
- **Cyclic His↔Zn restraint chain inversion.** The metal restraint referenced His/Zn on the wrong chains in
  the 2026-06-24 smoke and was dropped; fixed to the `[Zn, binder]` ordering (Zn = chain A, His = chain B)
  and now applied on every predict.

The CPU test suite stays green across all of this (815 passed / 30 deselected at the per-class validation;
860 passed at the ABC pilot; 59 ABC tests green).

---

## 6. Results on test cases

> All numbers are Chai-1 0.6.1 (image `gradio_design-gradio-design:latest`) on 2× RTX A4500, 20 GB.
> The per-class runs (§6.1–6.4) are the **de novo per-class validation**
> (`2026-06-25-denovo-per-class.md`, 12 iters, num_seqs 4, one fixed seed, from-scratch seeding only).
> Earlier `--smoke` wiring checks are in `2026-06-24-multiclass-resmoke.md`. ABC numbers are from
> `2026-06-25-abc-pilot.md` + `abc_runs/*.json`.

### 6.1 Alpha — single D-helix vs an L protein trimer

De novo, from-scratch seed `TSKLVWNAKKDMMLALTFEEG` (21 mer) → selected design
`AKAEELRKRLEEIIKELLEKG` (chai-emitted all-D), a **clean amphipathic D-helix**. **ipTM 0.791**, chirality
drift 0.000, 12/12 iters, **19.5 min** wall. Beats the documented baseline (0.44) and the L-seed control
(0.103). The `--smoke` regression guard independently gave ipTM 0.6992 (in the ~0.69 band, `beats_baseline`).

### 6.2 Non_alpha — all-D binder vs the 9DXX 2-chain MSA'd HA

The target is the **true 2-chain MSA'd HA receptor** (HA1 328 aa + HA2 176 aa), binder appended as the last
chain (chain C). De novo seed `VLKCPDGGCTCKLVWNGKEFMNDALTFQEC` (30 mer) → 30-residue all-D design, **30
residues in every one of 12 iters** — the ChainRoles fix holds: **no 31→176 balloon, no OOM**, the ~535-token
complex fits in ~18 GB. ipTM 0.167 / pTM 0.380 for the binder interface (poor, as expected for a from-scratch
knot scaffold with no negative design); the **L-seed predict reaches ipTM 0.869**, confirming the target/MSA
wiring is correct. **38.5 min** wall, 12/12 iters, PASS (wiring). (The 2026-06-24 smoke surfaced the OOM that
this fix resolves.)

### 6.3 Cyclic metallopeptide — 6UFA 24-mer, Zn site + MetalHawk

The 6UFA Zn macrocycle is specified declaratively: `--coord_residues 'H6,DHI12,H18,DHI24'` (His at
6/12/18/24, alternating L/D/L/D), Zn as a `[Zn+2]` Chai ligand on chain A, peptide on chain B, **4 His↔Zn
coordination restraints**. (Note: 6UFA is a 24-mer = 12-mer × 2 under S2 symmetry; the deposited asymmetric
12-mer loses 2 of the 4 coordinating His and cannot form the tetrahedron — the full symmetric 24-mer sequence
is required. Memory: *6UFA S2 symmetry*.)

De novo seed `LRRFTHANAMTHKANWVHKHAIGH` (24 mer, His pinned) → 24-residue mixed-chirality design,
**pTM 0.927**, 12/12 iters, **19.0 min**. The His↔Zn restraint applied on **every predict** (52× = 4 rows ×
13 predicts) — `[patch] token_dist repaired`, not the prior dropped-restraint failure. Best model: a Zn–His-N
coordination shell at **1.9–2.3 Å** (chemically reasonable Zn–N). **MetalHawk verdict: geometry = TET
(tetrahedral), perplexity 1.25 < 1.5 → PASS** — a clean, confident [Zn(His)] tetrahedral assignment, exactly
the 6UFA site class. (ipTM is not the headline metric for a single peptide + metal; pTM and the geometry gate
are.)

### 6.4 Free cyclic peptide — no target, intramolecular

No target, length-16 intramolecular cyclic, from-scratch seed `GYWCKHSYNVAHHWGE` (His pinned 6/12) → design
`G(DLE)GGGG(DAR)(DSG)(DVA)G(DGL)(DLY)(DTH)GGG`, **pTM 0.656**, 12/12 iters, **9.1 min**, PASS. ipTM is
undefined for a single free chain; the objective is the intramolecular 4-term score, pTM the structural
signal. **Caveat / finding:** the default PepMLM 16-mer seed contained no Gly, so its fully-D form was
un-tokenizable (ADR-004 ≥1-canonical rule) and crashed at `truncated_refine`; re-running with a reproducible
Gly-anchored random seed completes cleanly. Follow-up: enforce a Gly-anchor guard on the no-target free-cyclic
seed (alpha/non_alpha already do via `_ensure_cterm_glycine`).

> Total de novo per-class wall ≈ **86 min** across the two A4500s (run ≤2 GPU concurrent).

### 6.5 ABC-designed macrocycles — Variant A vs B

Case: free mixed-chirality cyclic peptide, no target, closure restraint only, objective `0.7·pTM +
0.3·termini_proximity` at K\* = 15 steps; Variant A on GPU 0, Variant B on GPU 1, in parallel.

| Variant | Search | Budget | Evals | Wall | Best nectar | Outcome |
|---|---|---|---|---|---|---|
| **A** (fixed) | chirality (ABC) + identity (MPNN/move) | 30 | 30 | **16.9 min** | **0.735** | stalls at the warm-start seed |
| **B** | identity + chirality (ABC), MPNN warm-start only | 90 | 90 (87 on disk) | **41.0 min** | **0.841** | genuine monotone convergence |

- **Variant B converges:** running-best climbs monotonically 0.718 → 0.741 → 0.766 → 0.789 → 0.807 → 0.820 →
  **0.841** (last improvement at eval 80). Best design `MGVRIFQQFGAP`, chirality `LLDDDDDDLDDD` →
  `MG(DVA)(DAR)(DIL)(DPN)(DGN)(DGN)FG(DAL)(DPR)`: pTM **0.773**, C–N **1.21 Å** (closed ring),
  proximity 1.00, nectar **0.841**. A genuinely high-pTM, closed mixed-chirality 12-mer macrocycle.
- **Variant A stalls:** best = the curated alternating-L/D seed `MQALEHQQQIGH` (pTM 0.62, C–N 1.30 Å, nectar
  0.735); the 30 MPNN-designed candidates never beat it, because Variant A's MPNN currently designs on a
  **zero backbone** (the T9 "inject the last predicted structure" step is unwired) — the identity oracle is
  structure-blind. The search *machinery* is correct (post-fix it genuinely varies identity + chirality every
  move); the oracle is starved.
- **Verdict:** the fast-cycle ABC optimizes the objective end-to-end on GPU *once the two fitness/encoding
  bugs (§5) are fixed*. Variant B is decisively better **per GPU-hour** (~132 evals/GPU-h, +0.124 nectar over
  the seed) and is the recommended default ABC arm; Variant A is deferred until the backbone-injection step is
  wired.

**Scaled Variant-B showcase** (`abc_runs/scaled_B_*.json`): a 12-mer reaches nectar **0.858**
(`FGYFIEFITPGH`, chirality `LDDDDDDDDDDD`) and a 20-mer reaches **0.806** (`PVGEEHKNVIKVMPGSGNGI`, 6 L + 14 D)
— both at K\* = 15 steps. **TODO:** full-step (200) re-validation of these scaled hits and the pilot winners
(the fast oracle ranks; it is not final truth) is still to be reported.

### 6.6 Cyclization calibration (the K\* result underpinning §6.5)

On a 2-POS / 2-NEG panel (real mixed-chirality hexamer `EPpKPp` and a 6UFA-derived 24-mer vs strained full-L
controls), swept over steps {10,25,50,100,200} × start {full, refine}: **pTM is the discriminating term**,
ΔpTM ≈ +0.20…+0.22 at *every* (steps, start) and **already saturated at 10 steps**. Chirality and
mainchain-pLDDT saturate for real and fake alike (non-discriminating); geometry is noisy. Hence
**K\* = 10–25 steps, start = full**, ~10–20× cheaper than full-200, feeding the bees on pTM (primary) + the
C–N proxy (secondary, essential for short peptides where Chai underestimates pTM). 16.5 min total GPU,
zero predict errors. **Caveat:** the panel is thin (2 POS / 2 NEG) and some of the pTM gap is "L-homochiral vs
mixed" not purely "open vs closed"; a larger panel with true mixed-chirality decoys is flagged.

---

## 7. Usage recommendations

**Binder length** (`--binder_length`): 6–50, from-scratch and unconditional when no target. Short peptides
(≤6–8) suffer Chai's pTM under-estimate — rely on the C–N termini proxy, not absolute pTM, to rank them.

**Which search:**
- Homochiral α / all-D classes → keep the **cheap MPNN + Chai greedy loop** (ADR-021: noise-dominated, a
  cheap sampler × N draws × max-select is best; do not retire it). Beam is opt-in only and does not beat
  greedy's ipTM-over-seed.
- Mixed-chirality (cyclic, free cyclic, metallopeptide) → **`--search abc`**, because inverse folding on a
  mixed backbone has no double-reflection trick.

**ABC knobs:**
- `--abc_variant`: **B** is the validated default (it converges; A is structure-blind until backbone
  injection is wired).
- `--colony_size` ~12–40, `--abc_cycles` ~20–50, `--scout_limit` ~5, `chirality_move_rate` ~0.3, bounded by a
  hard `chai_eval_budget` (pilot used 30 for A, 90 for B; scaled showcase used ~146 evals over 4 cycles).
- `fitness_steps = K* = 10–25` (15 in all GPU runs to date).
- **Objective weights:** `0.7·pTM + 0.3·termini_proximity` for the no-target cyclization objective. Do **not**
  drive on mainchain-pLDDT or chirality (both saturate) or on the legacy equal-ish aggregate (it dilutes the
  one working signal).

**Gates:** turn on the **periodicity gate** for register-capable α designs; turn on the **MetalHawk
metal-geometry gate** (perplexity ≤ 1.5) for any metal target to certify the coordination polyhedron; keep
the chirality gate + composition floor + PLL veto on by default.

**Metal sites:** declare coordinators with `--coord_residues` (handedness explicit); for symmetric sites use
the *full* symmetric sequence (e.g. 6UFA needs the 24-mer, His 6/12/18/24, L/D/L/D), not the deposited
asymmetric unit. `ensure_patches()` must be installed or the D-His↔Zn restraint is silently dropped.

**Register / specificity:** only ≥7-residue circular shifts are valid decoys. In-loop contrastive negative
design (T01) is the open priority for register-specificity; until then, treat register as handled by the
design-time periodicity gate + physics (relaxed ΔΔG) validation, not by any static objective term.

---

## 8. Usage statistics over time and hardware

All runs on `ifrit`: 2× NVIDIA RTX A4500 (20 GB each), Docker `--gpus all`, Chai-1 0.6.1, ESM-2 traced model
(5.7 GB) host-cached and volume-mounted (never re-downloaded). Heavy jobs are pinned one per GPU
(`CUDA_VISIBLE_DEVICES`), never two on one card.

**Per-run GPU times** (12 iters, num_seqs 4, de novo per-class):

| Run | Class / case | Iters | Wall | Notes |
|---|---|---|---|---|
| Alpha | D-helix vs L trimer | 12 | 19.5 min | ipTM 0.791 |
| Non_alpha | all-D vs 9DXX 2-chain MSA'd HA (~535 tokens) | 12 | 38.5 min | fits ~18 GB; largest token count |
| Cyclic metallo | 6UFA 24-mer + Zn, 4 restraints | 12 | 19.0 min | 52 restraint applications |
| Free cyclic | 16-mer intramolecular | 12 | 9.1 min | cheapest |
| **Per-class total** | | | **≈ 86 min** | across 2 GPUs, ≤2 concurrent |
| ABC Variant A | 30 evals | — | 16.9 min | ~34 s/eval (incl. MPNN) |
| ABC Variant B | 90 evals | — | 41.0 min | ~27 s/eval → **~132 evals/GPU-hour** |
| Cyclization calibration | 40 predicts + 4 refine-seeds | — | 16.5 min | 2 lanes in parallel |

**20 GB A4500 considerations.**
- The non_alpha 2-chain MSA'd HA complex (~535 tokens) sits at ~18 GB — the practical ceiling on this card.
  A previous OOM at this scale was a **chain-indexing bug** (binder ballooned 31→176; §5), *not* a true memory
  limit; the real complex fits. Larger genuine complexes need a bigger GPU, fewer diffusion timesteps, or
  capped iters.
- **Per-atom D-tokenization cost:** D / non-canonical residues are *atom-tokenized* by Chai (each non-canonical
  residue expands to multiple atom tokens), so a D-rich or metallopeptide design carries more tokens per
  residue than an all-L chain of the same length — the dominant per-eval cost driver for short peptides.
- The fast-cycle K\* = 10–25 step operating point is the key throughput lever: ~10–20× cheaper per eval than
  full-200, which is what makes a 90-eval ABC run fit in ~41 min.

**Throughput (evals / GPU-hour).** Variant B: ~132 evals/GPU-hour at K\* = 15. Variant A is ~3× costlier per
eval (a LigandMPNN forward per move) at ~34 s/eval. **TODO:** a consolidated evals/GPU-hour figure for the
restrained (metal) ABC route — the pilot measured the unrestrained no-target route; the restrained route pays
a full-predict per eval and is expected to be more expensive (**TODO: measure**).

---

## Appendix — provenance and open TODOs

**Primary source documents** (in this repository):
`docs/results/2026-06-25-denovo-per-class.md`, `docs/results/2026-06-25-abc-pilot.md`,
`docs/results/2026-06-25-cyclization-calibration.md`, the architecture-decision record (ADR-001..026),
and the implementation modules `xenodesign/eval/metal_geometry_gate.py` and `xenodesign/chai_patches.py`.

**Open TODOs flagged in this draft:**
1. Full-step (200) re-validation of the ABC pilot winners and the scaled Variant-B 12/20-mer showcases.
2. evals/GPU-hour for the restrained (metal) ABC route.
3. Larger cyclization-calibration panel with true mixed-chirality decoys (current panel is 2 POS / 2 NEG).
4. Wire Variant A's per-move backbone injection (T9) before any A-vs-B production comparison.
5. T01 in-loop contrastive register-decoy negative design — the standing scientific priority.
6. Wet-lab validation — none of the structural numbers here are experimentally confirmed.
7. Literature citations to be completed with full bibliographic details (Bhardwaj 2016, Hosseinzadeh 2017,
   Mulligan 2021, ProteinMPNN/LigandMPNN, Chai-1, MetalHawk/Sgueglia-Vrettas JCIM 2024).
