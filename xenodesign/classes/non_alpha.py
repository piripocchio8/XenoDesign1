"""NON-ALPHA 9DXX (D-knottin : influenza HA) binder class — a REAL design loop (T6).

Migrated + UPGRADED from ``scripts/design_nonalpha.py``, which is now a thin
re-export shim keeping its predict-only ``run_nonalpha_design`` CLI. The old
driver was a one-shot *predict* wiring validator (it called ``backend.predict``
ONCE, no loop). This module promotes it to a genuine HalluLoop design over the
**multi-chain MSA'd HA target**, by reusing the validated α loop wiring:

* ``seq_update`` reuses ``classes.alpha.make_alpha_seq_update_fn`` (the
  ``SequenceUpdater`` over ``MultiCandidate`` of the C-term-Gly-anchored base
  inverse-folding backend — the P2 drift fix);
* ``objective`` reuses ``classes.alpha._loop_score_fn`` /
  ``make_mixed_loop_score_fn`` / ``make_ipsae_loop_score_fn``;
* ``referee`` reuses ``classes.alpha._make_referee_fn`` (chirality + helix +
  composition + ESM-PLL from the per-iter scored CIF).

Only what makes this class *non-α* differs from α:
* the seed is an **ICK cystine-knot** scaffold (6 Cys + Gly anchor), not a PepMLM
  helix seed (``build_binder_seed`` / ``knottin_cys_positions``);
* the SS-bias is **anti-α** (``target_helix_frac=0.0`` — knottins are non-helical),
  via ``ss_bias_config_for_case`` reading ``case.knobs['ss_bias']='anti_alpha'``;
* the target is the **2-chain MSA'd HA receptor** (the ex-Goal-3 Route A): the
  ``non_alpha`` PRESET sets ``target.target_type='protein'`` + ``target.msa=True``;
  the dispatcher builds it via ``targets.target_entities`` (T4), which calls
  ``load_ha_entities`` for ``len(chains)>1`` and returns the cached MSA dir so the
  predict runs ``ChaiBackend.predict(msa_directory=...)`` (gate #29 resolved).

CLOSURE (D-Cys disulfides) — CARRIED-FORWARD LIMITATION, NOT re-solved here.
Chai 0.6.1 matches a COVALENT row's one-letter code against the token name via the
L-only ``rc.restype_1to3``; an all-D Cys token is stored as the D-CCD name (DCY),
which does NOT match the L 'CYS' code, so the disulfide COVALENT bond is REJECTED
(``bond_utils`` assert ``left_residue_idx.numel() > 0`` fails — verified on GPU,
see the ``chai D-residue covalent limitation`` memory + ``design_nonalpha.py``
header). Therefore ``closure`` returns ``[]`` by default and the binder is designed
under the anti-α SS-bias + the MSA'd target only. The disulfide-row builder is kept
(correct for L-Cys / future chai) for that path.

RESTRAINTS — the ``nonalpha`` case carries a pocket ``RestraintSpec`` whose target
epitope ``target_resnums`` is still EMPTY (pending gate #27); ``build_for_case``
refuses such a SHELL pocket. ``restraints`` therefore returns ``None`` (graceful —
the default GPU run is restraint-free, exactly what ``run_nonalpha_design`` does),
rather than crashing. Once the HA-stem epitope resnums are pinned, the same hook
will emit the pocket.

CPU tests here exercise seed / ss_bias / closure / restraints / objective /
seq_update / referee / report (never a predict). The dispatcher (T3) wires these
hooks into the untouched HalluLoop just as it does for α / cyclic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from xenodesign.benchmark.cases import get_case, ss_bias_config_for_case
from xenodesign.benchmark.restraints import disulfide_rows, write_restraints
from xenodesign.config import local_ref

# NOTE: SeedSpec is imported lazily inside NonAlpha.seed (not at module top) to avoid a circular
# import — base.py re-points its registry at this module (`from ...non_alpha import NonAlpha`)
# BEFORE the other classes, so a top-level `from ...base import SeedSpec` here would fail when
# non_alpha is imported first (base only partially initialised). Mirrors classes/alpha.py.

# Reuse the α loop machinery (same callables the validated α loop uses). Importing the
# NAMES is CPU-clean: alpha defers all torch/chai/gemmi imports to call time.
from xenodesign.classes.alpha import (
    _loop_score_fn,
    _make_referee_fn,
    make_alpha_seq_update_fn,
    make_ipsae_loop_score_fn,
    make_mixed_loop_score_fn,
)

# Cached MSA for the HA target (chai .aligned.pqt, keyed by sequence hash); gate #29 prep.
# Resolve under the local-reference tree (XENO_LOCAL_REF env, default ./XenoDesign1_local_ref).
_DEFAULT_MSA_DIR = str(local_ref("9dxx_target_gate", "chai_pred_msa", "msas"))
_DEFAULT_HA_FASTA = str(local_ref("9dxx_target_gate", "ha_target.fasta"))
_DEFAULT_BINDER_LEN = 31

# Chai labels FASTA chains A, B, C... in order. Entities are written [HA1, HA2, binder], so the
# designed binder is chain C and the disulfides/SS-bias act on chain C.
_BINDER_CHAIN = "C"


# ── HA target entities (the two L chains, MSA-backed) ──────────────────────────────

def load_ha_entities(fasta_path: str | Path) -> list:
    """Parse ha_target.fasta into [HA1, HA2] protein entities (both L). The sequences MUST be
    byte-identical to the ones the cached MSA was built from (the .aligned.pqt are keyed by
    sequence hash), so we read them straight from the prepared FASTA, never inline them."""
    text = Path(fasta_path).read_text().splitlines()
    entities, name, seq = [], None, []
    for line in text:
        line = line.strip()
        if line.startswith(">"):
            if name is not None and seq:
                entities.append({"type": "protein", "name": name,
                                 "sequence": "".join(seq), "chirality": "L"})
            name = line.split("|")[-1] if "|" in line else line[1:]
            seq = []
        elif line:
            seq.append(line)
    if name is not None and seq:
        entities.append({"type": "protein", "name": name,
                         "sequence": "".join(seq), "chirality": "L"})
    return entities


# ── Cystine-knot (ICK) scaffold ────────────────────────────────────────────────────

def knottin_cys_positions(length: int = _DEFAULT_BINDER_LEN) -> list:
    """Six 1-based Cys positions for an inhibitor-cystine-knot scaffold on a `length`-mer.

    A canonical ICK packs 6 Cys with a short III-IV loop and the knot threaded by the I-IV /
    II-V / III-VI bonds. We use an evenly-spread default (scaled to `length`) so it is a valid,
    well-separated 6-Cys layout for any binder length; an exact scaffold can be supplied via
    --cys_positions. Positions are clamped to [1, length] and de-duplicated."""
    if length < 6:
        raise ValueError(f"need >= 6 residues for a cystine knot, got {length}")
    # Fractional anchors of a typical ICK (I..VI) along the chain.
    fracs = [0.10, 0.26, 0.42, 0.55, 0.74, 0.90]
    pos = sorted({max(1, min(length, round(f * (length - 1)) + 1)) for f in fracs})
    # If rounding collided, spread them out to 6 distinct positions.
    i = 1
    while len(pos) < 6:
        cand = max(1, min(length, i))
        if cand not in pos:
            pos.append(cand)
            pos.sort()
        i += 1
    return pos[:6]


def ick_disulfide_pairs(cys_positions) -> list:
    """ICK connectivity over the 6 ordered Cys: I-IV, II-V, III-VI (1-4, 2-5, 3-6)."""
    c = list(cys_positions)
    if len(c) != 6:
        raise ValueError(f"ICK needs exactly 6 Cys, got {len(c)}")
    return [(c[0], c[3]), (c[1], c[4]), (c[2], c[5])]


def place_cys(seed_one_letter: str, cys_positions) -> str:
    """Overwrite the 1-based `cys_positions` of an L seed with 'C' (the scaffold Cys)."""
    s = list(seed_one_letter)
    for p in cys_positions:
        s[p - 1] = "C"
    return "".join(s)


def _ensure_glycine(seq: str, avoid) -> str:
    """Force >=1 canonical residue: chai cannot tokenize a FULLY non-canonical chain (an all-D
    peptide is all D-CCD = non-canonical), so it needs >=1 canonical anchor (achiral Gly works in
    either hand). Insert a 'G' at the most central position NOT in `avoid` (the Cys scaffold) if
    the seq has no glycine. Mirrors design_alpha._ensure_glycine — a tokenization requirement,
    not a softening of the design."""
    if "G" in seq:
        return seq
    n = len(seq)
    order = sorted(range(n), key=lambda i: abs(i - n // 2))  # central-out
    for i in order:
        if (i + 1) not in set(avoid):
            return seq[:i] + "G" + seq[i + 1:]
    return seq  # pragma: no cover (every position was a Cys — impossible for len>6)


def build_binder_seed(length: int, cys_positions, seed_seq: str | None = None,
                      rng_seed: int = 0) -> str:
    """A `length`-res L binder seed with Cys placed at the ICK positions and >=1 glycine anchor.
    Explicit seed_seq (right length) is used as the backbone; else a deterministic diverse random
    seed. The chain is encoded all-D downstream (write_inputs chirality='D'); the glycine anchor
    keeps chai able to tokenize the otherwise fully-non-canonical D chain."""
    if seed_seq is not None:
        if len(seed_seq) != length:
            raise ValueError(f"seed_seq length {len(seed_seq)} != binder length {length}")
        base = seed_seq.upper()
    else:
        import random
        rng = random.Random(rng_seed)
        # avoid extra Cys / Gly-Ala flooding so the only Cys are the scaffold ones
        alphabet = "DEFHIKLMNPQRSTVWY"
        base = "".join(rng.choice(alphabet) for _ in range(length))
    return _ensure_glycine(place_cys(base, cys_positions), cys_positions)


def build_nonalpha_disulfide_rows(cys_positions, binder_chain: str = _BINDER_CHAIN,
                                  confidence: float = 1.0) -> list:
    """The 3 ICK disulfide COVALENT rows on the binder chain (SG-SG, name 'C').

    Correct for L-Cys / future chai; REJECTED on an all-D binder by chai's L-only covalent
    name-match (see module header). Kept for the L-Cys path; ``closure`` returns [] by default."""
    pairs = ick_disulfide_pairs(cys_positions)
    return disulfide_rows(binder_chain, pairs, confidence=confidence)


# ── Binder-chain metrics from the predicted CIF ────────────────────────────────────

def disulfide_geometry_from_cif(cif_path, chain_name: str = _BINDER_CHAIN,
                                cys_positions=None):  # pragma: no cover (gemmi/CIF)
    """SG-SG distances for each ICK pair on the binder chain (a closed disulfide is ~2.05 A).

    Reads the binder chain's per-residue SG atoms (Cys/DCY) and returns the SG-SG distance for
    each ICK pair plus how many are within a 2.5-A bonded cutoff. Reports geometry only — chai
    forms the bonds via the COVALENT restraint; this checks they materialised."""
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    sg = {}
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for i, res in enumerate(chain, start=1):
                if res.name in ("CYS", "DCY"):
                    a = res.find_atom("SG", "*")
                    if a is not None:
                        sg[i] = np.array([a.pos.x, a.pos.y, a.pos.z], dtype=float)
        break
    pairs = ick_disulfide_pairs(cys_positions) if cys_positions else []
    out = []
    for (i, j) in pairs:
        if i in sg and j in sg:
            d = float(np.linalg.norm(sg[i] - sg[j]))
            out.append({"pair": [i, j], "sg_sg_dist": d, "bonded": d <= 2.5})
        else:
            out.append({"pair": [i, j], "sg_sg_dist": None, "bonded": False})
    return {"pairs": out, "n_bonded": sum(1 for p in out if p["bonded"])}


# ── Result assembly (real-loop trajectory + select; behaviour aligned with the
#    predict-only run_nonalpha_design result dict) ──────────────────────────────────

def _best_step(history):
    """Highest-score LoopStep in the trajectory (greedy selection; panel-agnostic here)."""
    return max(history, key=lambda h: getattr(h, "score", float("-inf")))


def _assemble_nonalpha_result(cfg, history, panel_result, case, out_dir,
                              *, l_seed_iptm: float = 0.0, wall_time_s: float = 0.0) -> dict:
    """Build the non-α result dict + write nonalpha_result.json (real-loop selection).

    Selects the best trajectory step (ipTM drives), records the binder seed length / Cys
    scaffold / ICK connectivity, the anti-α SS-bias target, and the disulfide geometry when
    available on the step's prediction. Geometry/chirality fields default to None on CPU (no
    CIF parsed); the GPU smoke (T10) populates them from the predicted structure. The schema
    mirrors the predict-only ``run_nonalpha_design`` result (case_id / iptm / cys_positions /
    ick_disulfide_pairs / disulfides / msa_dir / ss_bias_target_helix_frac), adding the loop's
    trajectory shape (n_iters / selected_*)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binder_length = case.binder_length
    cys = knottin_cys_positions(binder_length)
    ss = ss_bias_config_for_case(case)

    best = _best_step(history) if history else None
    pred = getattr(best, "prediction", None) if best is not None else None

    result = {
        "case_id": "nonalpha",
        "n_iters": len(history),
        "l_seed_iptm": float(l_seed_iptm),
        "wall_time_s": float(wall_time_s),
        "binder_length": binder_length,
        "cys_positions": list(cys),
        "ick_disulfide_pairs": ick_disulfide_pairs(cys),
        "selected_iptm": (float(pred.iptm) if pred is not None and hasattr(pred, "iptm")
                          else None),
        "selected_ptm": (float(pred.ptm) if pred is not None and hasattr(pred, "ptm")
                         else None),
        "selected_d_fasta": (getattr(best.state, "d_fasta", None)
                             if best is not None else None),
        "binder_chirality": getattr(pred, "binder_chirality", None),
        "disulfide_geometry": getattr(pred, "disulfide_geometry", None),
        "disulfides": False,  # D-Cys covalent rejected by chai → default off (carried-forward)
        "msa_dir": (cfg.target.msa_dir or _DEFAULT_MSA_DIR) if cfg.target.msa else None,
        "ss_bias_target_helix_frac": ss.target_helix_frac,
        "phase": "real HalluLoop design (anti-α SS-bias + MSA'd 2-chain HA target)",
        "restraints": bool(cfg.restraints_on),
        "out_dir": str(out_dir),
    }
    (out_dir / "nonalpha_result.json").write_text(
        json.dumps(result, indent=2,
                   default=lambda o: getattr(o, "tolist", lambda: str(o))()))
    return result


# ── BinderClass adapter ──────────────────────────────────────────────────────────

class NonAlpha:
    """Non-α (ICK D-knottin) binder class implementing
    :class:`xenodesign.classes.base.BinderClass`.

    ``case_id == 'nonalpha'`` (the benchmark registry key; the CLI axis is ``non_alpha``).
    Reuses the α loop machinery for seq_update / objective / referee; only the seed (ICK Cys
    scaffold), the SS-bias (anti-α), and the target (2-chain MSA'd HA) make it non-α.
    """

    case_id = "nonalpha"

    def seed(self, cfg, target_seq) -> "SeedSpec":
        """FROM-SCRATCH unified PepMLM seed conditioned on the HA target (no forced knottin).

        The seed is the unified ``generate_conditioned(target_seq, binder_length)`` peptide — it
        NEVER inherits the reference DP93 sequence, the ICK Cys scaffold, OR its length (the old
        forced-knottin path is removed). The Cys/disulfide scaffold is now OPT-IN: declare a set
        of fixed Cys positions via ``cfg.restraint.params['cys_positions']`` to pin them (those
        positions are overwritten with 'C' and reported as ``cys_positions``); absent -> no Cys.
        """
        from xenodesign.classes.base import SeedSpec
        from xenodesign.config import resolve_binder_length
        from xenodesign.seed import make_configured_generator, unified_seed

        length = resolve_binder_length(cfg)
        gen = make_configured_generator(cfg)
        # OPT-IN Cys scaffold (never default): a config-declared set of 1-based Cys positions.
        declared = cfg.restraint.params.get("cys_positions") if cfg.restraint else None
        cys = tuple(p for p in (declared or ()) if 1 <= int(p) <= length)
        result = unified_seed(gen, target_seq=target_seq or "", length=length, reverse=True)
        one = result.one_letter
        if cys:
            chars = list(one)
            for p in cys:
                chars[int(p) - 1] = "C"
            one = "".join(chars)
        return SeedSpec(one_letter=one, cys_positions=cys)

    def ss_bias(self, cfg, case):
        return ss_bias_config_for_case(case)  # anti_alpha -> target_helix_frac 0.0

    def restraints(self, cfg, case, out_dir, target_ctx):
        """Emit the HA-stem pocket restraint if buildable, else None.

        The ``nonalpha`` case pocket is currently a SHELL (empty ``target_resnums``, pending
        gate #27); ``build_for_case`` refuses such a pocket. We therefore return None (the
        default GPU run is restraint-free — exactly what ``run_nonalpha_design`` does), never
        crashing. When the epitope resnums are pinned this hook emits the pocket automatically."""
        if not cfg.restraints_on:
            return None
        from xenodesign.benchmark.restraints import build_for_case
        try:
            rows = build_for_case(case)
        except ValueError:
            return None  # shell pocket (pending gate #27) — restraint-free default
        return write_restraints(Path(out_dir) / "nonalpha.restraints", rows)

    def closure(self, cfg, seed_spec) -> list:
        """No closure rows: the ICK disulfide COVALENT bonds are REJECTED by chai on an all-D
        binder (token DCY != L 'CYS'; bond_utils assert). Carried-forward limitation, not
        re-solved here — see module header + the ``chai D-residue covalent limitation`` memory."""
        return []

    def seq_update(self, cfg, wrapper, seed_spec, roles=None):
        # Reuse the α drift-fixed sequence-update closure (MultiCandidate over the C-term-Gly-
        # anchored inverse-folding base). anti-α only changes the panel ss_bias, not the loop.
        # ``roles`` (the dispatch chain contract) is threaded through so a multi-chain target
        # (HA1/HA2 -> binder chain 'C') reads the RIGHT binder chain, not a hardcoded 'B'.
        return make_alpha_seq_update_fn(wrapper, num_seqs=cfg.loop.num_seqs,
                                        backend=cfg.loop.backend, roles=roles)

    def accept_fns(self, cfg):
        from xenodesign.loop import compose_accept_fns
        return compose_accept_fns(None)

    def objective(self, cfg, wrapper):
        if cfg.objective == "mixed":
            return make_mixed_loop_score_fn(wrapper)
        if cfg.objective == "ipsae":
            return make_ipsae_loop_score_fn(wrapper)
        return _loop_score_fn

    def referee(self, cfg, loop_dir, esm_judge, roles=None):
        # roles threads the chain contract: a 2-chain HA target -> binder chain 'C', so the
        # reused alpha referee reads the RIGHT binder chain, not HA2 ('B').
        return _make_referee_fn(loop_dir, esm_judge=esm_judge, roles=roles)

    def report(self, cfg, history, panel_result, case, out_dir,
               *, l_seed_iptm: float = 0.0, wall_time_s: float = 0.0) -> dict:
        return _assemble_nonalpha_result(cfg, history, panel_result, case, out_dir,
                                         l_seed_iptm=l_seed_iptm, wall_time_s=wall_time_s)
