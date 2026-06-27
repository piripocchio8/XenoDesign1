"""Loop configuration (spec §2.6): JSON -> validated dataclass with defaults."""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_VALID_ACCEPT = {"greedy"}  # spec §2.4: greedy only (no Metropolis/SA)
_VALID_SEED_SOURCES = {"pepmlm_retroinverso", "protgpt2_retroinverso", "random", "mirror_of_L"}

# Sentinel for "device not pinned by the user" — resolve_device() then auto-selects.
# (Empty string keeps the field a plain `str`, JSON-serializable, and falsy.)
DEVICE_AUTO = ""


def resolve_device(cfg: object | None = None) -> str:
    """The single device-resolution point (audit HW-1).

    Resolution order (first that applies wins):
      1. ``XENO_DEVICE`` env var (verbatim — lets a user force ``cpu`` / ``mps`` / ``cuda:N``);
      2. an explicit, non-sentinel ``cfg.device`` (e.g. a ``--device`` CLI override);
      3. auto: ``"cuda:0"`` if CUDA is available, else ``"mps"`` on Apple silicon, else ``"cpu"``.

    torch is imported lazily *inside* this function so importing ``config`` stays
    CPU/torch-free. If torch cannot be imported, the resolver degrades to ``"cpu"``.
    On a GPU box (e.g. ifrit) the auto result is still ``"cuda:0"`` — no behaviour change.
    """
    env = os.environ.get("XENO_DEVICE")
    if env:
        return env
    pinned = getattr(cfg, "device", None)
    if pinned:  # non-sentinel, user-pinned device
        return pinned
    try:
        import torch  # lazy: never on the CPU-import path
    except Exception:
        return "cpu"
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def local_ref(*parts: str) -> Path:
    """Resolve a path under the local reference-data tree (audit HW-5).

    Honours ``XENO_LOCAL_REF`` (absolute or relative); defaults to the repo-relative
    ``./XenoDesign1_local_ref`` so an external user gets a sensible relative path instead
    of a hard-coded ifrit absolute. The tree is gitignored, so callers must still fail-soft
    when the resolved path is absent.
    """
    root = Path(os.environ.get("XENO_LOCAL_REF", "XenoDesign1_local_ref"))
    return root.joinpath(*parts) if parts else root


@dataclass
class LoopConfig:
    engine: str = "chai"
    device: str = DEVICE_AUTO  # unset sentinel; resolve_device() auto-selects (HW-1)
    seed: int = 42
    ref_time_steps: int = 50
    num_diffn_timesteps: int = 200
    iterations: int = 30
    num_seqs: int = 8
    design_epoch_begin: int = 1
    seed_source: str = "pepmlm_retroinverso"
    accept: str = "greedy"
    select_topk_frac: float = 0.1


def load_config(path: str | Path) -> LoopConfig:
    """Load and validate a loop config JSON. Unknown keys are ignored; values validated."""
    data = json.loads(Path(path).read_text())
    cfg = LoopConfig(**{k: v for k, v in data.items() if k in LoopConfig.__dataclass_fields__})
    if cfg.accept not in _VALID_ACCEPT:
        raise ValueError(f"accept must be one of {_VALID_ACCEPT}, got {cfg.accept!r}")
    if cfg.seed_source not in _VALID_SEED_SOURCES:
        raise ValueError(f"seed_source must be one of {_VALID_SEED_SOURCES}, got {cfg.seed_source!r}")
    return cfg


# --- Multi-class design framework runtime config (spec §4) ---
# DesignConfig is the runtime layer ON TOP of the CASES registry: it starts from a per-class
# PRESET, references the benchmark case for fixed facts, and carries the run knobs. It is
# resolved PRESET -> --config-file -> CLI flags, then dumped to out_dir/resolved_config.json
# for provenance before the first predict. LoopConfig/load_config above are left untouched.


# The target-chemistry axis. 'none' = binder-only (no target entity): a free cyclic/linear
# peptide whose objective is INTRAMOLECULAR (see classes.cyclic.intramolecular_score_fn).
VALID_TARGET_TYPES = ("protein", "rna", "dna", "small_molecule", "metal", "none")

# FROM-SCRATCH binder lengths. These are sensible per-class DEFAULTS (overridable via
# cfg.binder_length / --binder_length), NOT locked to the reference binder — the seed must
# NEVER inherit the real binder's length (non-negotiable design principle). The benchmark cases.py
# binder_length is now used only for benchmark scoring, never to pin the seed.
DEFAULT_BINDER_LENGTH = {"alpha": 21, "non_alpha": 30, "cyclic": 24}
NO_TARGET_BINDER_LENGTH = 16          # free cyclic/linear peptide (target_type='none')
BINDER_LENGTH_MIN, BINDER_LENGTH_MAX = 6, 50


def resolve_binder_length(cfg) -> int:
    """Resolve the FROM-SCRATCH binder length for a DesignConfig, clamped to [6, 50].

    cfg.binder_length > 0 is an explicit override (used verbatim, then clamped). 0 means
    "use the sensible default": NO_TARGET_BINDER_LENGTH when target_type='none', else the
    per-class DEFAULT_BINDER_LENGTH (alpha 21, non_alpha 30, cyclic 24). Never reads the
    benchmark case's reference-binder length (design-from-scratch principle)."""
    n = int(getattr(cfg, "binder_length", 0) or 0)
    if n <= 0:
        if cfg.target.target_type == "none":
            n = NO_TARGET_BINDER_LENGTH
        else:
            n = DEFAULT_BINDER_LENGTH.get(cfg.binder_class, NO_TARGET_BINDER_LENGTH)
    return max(BINDER_LENGTH_MIN, min(BINDER_LENGTH_MAX, n))


@dataclass
class TargetSpec:
    target_type: str = "protein"      # protein|rna|dna|small_molecule|metal|none
    fasta_path: str = ""
    pdb_path: str = ""
    chains: tuple = ()
    msa: bool = False
    msa_dir: str = ""
    smiles: str = ""
    ccd: str = ""
    modifications: tuple = ()


@dataclass
class RestraintConfig:
    kind: str = ""                    # ''|pin_polarity|pocket|contact|metal_coordination|cyclization
    params: dict = field(default_factory=dict)


@dataclass
class LoopKnobs:
    iters: int = 30
    ref_time_steps: int = 50
    num_seqs: int = 8
    search: str = "greedy"            # greedy|beam
    beam_width: int = 2
    beam_cycles: int = 2
    backend: str = "ligandmpnn"       # ligandmpnn|carbonara|mixed


@dataclass
class AbcKnobs:
    """ABC mixed-chirality search tunables (spec §5.4; objective re-decided 2026-06-25).

    Consumed by the ``--search abc`` dispatch branch for MIXED-CHIRALITY cases (cyclic +
    target_type=none). ``variant`` picks the axis split (A: ABC chirality + MPNN identity;
    B: ABC identity+chirality, MPNN warm-start only). ``fitness_steps`` is the fast K* (10-25)
    diffusion-step count; ``w_ptm`` / ``w_termini`` are the pTM-primary + termini-secondary
    objective weights (the cyclization-calibration driver). All user-tunable.
    """

    colony_size: int = 24
    cycles: int = 50
    scout_limit: int = 5
    chirality_move_rate: float = 0.3
    variant: str = "a"                # "a" | "b"
    chai_eval_budget: int = 2000
    fitness_steps: int = 15           # K* fast diffusion steps (10-25; calibrated cheap point)
    w_ptm: float = 0.7                # pTM weight (primary discriminator)
    w_termini: float = 0.3            # C-N termini-proximity weight (secondary; short-peptide-critical)
    # Variant-B ncAA palette (track #2): CCD 3-letter codes the identity search may propose,
    # emitted as ``(XXX)`` in the FASTA. EMPTY = ncAA OFF (default → existing behaviour). A
    # non-empty list opts in; codes are validated (abc.ncaa.validate_palette) before use.
    ncaa_palette: list = field(default_factory=list)


@dataclass
class GateConfig:
    chirality: bool = False
    periodicity: bool = False
    heptad_thresh: float = 0.35
    entropy: bool = True
    pll_veto: bool = True
    esm_device: str = "cpu"
    # MetalHawk coordination-geometry gate (cyclic metallopeptides). OFF by default;
    # consulted by the cyclic class's report/selection, not hard-wired into the loop.
    metal_geometry: bool = False
    metal_perplexity_thresh: float = 1.5


@dataclass
class DesignConfig:
    binder_class: str = "alpha"
    target: TargetSpec = field(default_factory=TargetSpec)
    restraint: RestraintConfig = field(default_factory=RestraintConfig)
    loop: LoopKnobs = field(default_factory=LoopKnobs)
    abc: AbcKnobs = field(default_factory=AbcKnobs)
    gates: GateConfig = field(default_factory=GateConfig)
    objective: str = "iptm"
    device: str = DEVICE_AUTO  # unset sentinel; resolve_device() auto-selects (HW-1)
    seed: int = 42
    out_dir: str = ""
    use_pepmlm: bool = True
    use_pll: bool = True
    restraints_on: bool = True
    binder_length: int = 0   # 0 = per-class default (resolve_binder_length); else clamp 6..50


PRESETS: dict[str, DesignConfig] = {
    "alpha": DesignConfig(
        binder_class="alpha",
        target=TargetSpec(target_type="protein"),
        restraint=RestraintConfig(kind="pin_polarity"),
        gates=GateConfig(entropy=True, pll_veto=True),
        objective="iptm"),
    "non_alpha": DesignConfig(
        binder_class="non_alpha",
        target=TargetSpec(target_type="protein", msa=True),
        restraint=RestraintConfig(kind="pocket"),
        gates=GateConfig(entropy=True, pll_veto=True),
        objective="iptm"),
    "cyclic": DesignConfig(
        binder_class="cyclic",
        target=TargetSpec(target_type="metal", smiles="[Zn+2]"),
        restraint=RestraintConfig(kind="metal_coordination"),
        gates=GateConfig(chirality=False, entropy=False, pll_veto=False),
        objective="iptm"),
}


def _set_recursive(cfg: DesignConfig, key: str, value):
    """Recursive setter shared by the dotted-key and nested-dict override paths.

    `key` may be dotted ('loop.iters') to descend the dataclass tree; a dict `value`
    on a nested dataclass attribute is overlaid recursively (key-by-key).
    """
    if "." in key:
        head, tail = key.split(".", 1)
        _set_recursive(getattr(cfg, head), tail, value)
    elif (isinstance(value, dict) and hasattr(cfg, key)
          and dataclasses.is_dataclass(getattr(cfg, key))):
        child = getattr(cfg, key)
        for k, v in value.items():
            _set_recursive(child, k, v)
    else:
        setattr(cfg, key, value)


def _apply_dotted(cfg: DesignConfig, key: str, value):
    """Set 'loop.iters'/'target.msa'/'seed' style overrides on the dataclass tree."""
    _set_recursive(cfg, key, value)


def _overlay(cfg: DesignConfig, data: dict):
    for k, v in data.items():
        _set_recursive(cfg, k, v)


def resolve_config(binder_class: str, target_type: str | None = None,
                   config_file: str | None = None,
                   cli_overrides: dict | None = None,
                   out_dir: str = "") -> DesignConfig:
    if binder_class not in PRESETS:
        raise KeyError(f"unknown binder_class {binder_class!r}; registered: {sorted(PRESETS)}")
    cfg = copy.deepcopy(PRESETS[binder_class])
    if target_type is not None:
        if target_type not in VALID_TARGET_TYPES:
            raise ValueError(
                f"unknown target_type {target_type!r}; valid: {VALID_TARGET_TYPES}")
        cfg.target.target_type = target_type
    if out_dir:
        cfg.out_dir = out_dir
    if config_file:
        _overlay(cfg, json.loads(Path(config_file).read_text()))
    for k, v in (cli_overrides or {}).items():
        _apply_dotted(cfg, k, v)
    # binder_length is an int knob (0 = per-class default, else clamped to [6,50] at use time
    # by resolve_binder_length). Validate the TYPE here; the clamp itself is non-fatal.
    if not isinstance(cfg.binder_length, int) or isinstance(cfg.binder_length, bool):
        raise ValueError(
            f"binder_length must be an int (0 = per-class default), got {cfg.binder_length!r}")
    if cfg.binder_length < 0:
        raise ValueError(f"binder_length must be >= 0, got {cfg.binder_length}")
    # NOTE: cfg.device is left as the unset sentinel here so config resolution stays
    # torch-free; the actual device is resolved at point-of-use (dispatch._make_predictor)
    # and baked into the provenance by dump_config().
    return cfg


def dump_config(cfg: DesignConfig, out_dir) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / "resolved_config.json"
    p.write_text(json.dumps(dataclasses.asdict(cfg), indent=2,
                            default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    return p
