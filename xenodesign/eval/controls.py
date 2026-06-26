"""Within-Chai circularity controls for the D-binder benchmark (tracker #35).

A real Chai-1 ipTM/ipAE win for a designed D-binder is only meaningful if it survives
*within-model* negative controls — otherwise Chai may simply reward "a peptide of this
composition near this target" rather than the designed interface. This module is the
**CPU-pure** half: it builds the control inputs (composition-matched scrambles, a fixed
off-target helix panel) and analyses *which* target residues a binder actually contacts
(the interface footprint + heptad register). The GPU scoring half is a documented recipe
(see :func:`gpu_control_protocol`) that the campaign driver executes; nothing here imports
chai/torch.

Cross-model controls (re-scoring with AF3/Boltz) are DEFERRED; this is the within-Chai set.

The three negative controls, in increasing subtlety:

1. **composition-matched scramble** (:func:`composition_matched_scramble`) — the KEY control.
   A random permutation of the *same* residues: identical amino-acid composition (so ESM/Chai
   sees the same "bag of residues"), but destroyed sequence order and therefore destroyed
   designed interface. A genuine binder must beat its own scramble.
2. **off-target helices** (:func:`off_target_helices`) — a small fixed panel of unrelated
   ~21-res L helices (canonical amphipathic / coiled-coil decoys). Scoring these against the
   target measures the *specificity gap*: the binder should beat generic helices.
3. **restraint-off** — re-score the binder with the pocket restraint removed (handled in the
   GPU recipe, not a builder): a restraint-dependent ipTM is a circularity red flag.

And one structural read-out:

4. **interface footprint** (:func:`interface_footprint`) — from a predicted complex CIF, which
   TARGET residues the binder contacts and their heptad register. A real helix-on-helix /
   helix-on-bundle binder should dock against the SURFACE (heptad b/c/e/f/g) of the target,
   not its buried a/d core; a footprint dominated by core positions is physically suspect.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

import os

from xenodesign.config import local_ref
from xenodesign.metrics import _parse_cif_ca

# Path to the reference trimer FASTA holding the published D-binder (chain-A record) and the
# L-target (chain-B record). Gitignored real-data; NEVER inline the binder sequence — load its
# aggregate length via load_gt_reference_binder(). The campaign driver scores THIS fixed binder
# (restraint on/off) as the positive anchor for every control comparison.
# Honours XENO_LOCAL_REF; when unset, anchors at the repo root (../../) so it is cwd-independent.
if os.environ.get("XENO_LOCAL_REF"):
    GT_REFERENCE_FASTA = local_ref("dl_able_ground_truth", "trimer_DL_ABLE.fasta")
else:
    GT_REFERENCE_FASTA = (
        Path(__file__).resolve().parents[2]
        / "XenoDesign1_local_ref" / "dl_able_ground_truth" / "trimer_DL_ABLE.fasta"
    )

# Heavy-atom / CA contact distance for "the binder touches this target residue" (Angstrom).
CONTACT_DIST_A = 8.0

# Pre-registered win margins (declare BEFORE scoring to avoid post-hoc goalpost-moving). A
# designed D-binder "passes the circularity controls" iff it clears EVERY margin below.
WIN_MARGINS = {
    "iptm_gap": 0.08,        # binder ipTM - control ipTM  >= 0.08 (vs scramble AND off-target)
    "ipae_gap_A": 2.0,       # control ipAE - binder ipAE  >= 2.0 A (lower ipAE is better)
    "off_target_gap": 0.15,  # binder ipTM - best off-target ipTM >= 0.15 (specificity gap)
}


# ---------------------------------------------------------------------------
# (1) composition-matched scramble — the key negative control
# ---------------------------------------------------------------------------

def composition_matched_scramble(seq: str, rng_seed: int) -> str:
    """Return a random permutation of ``seq`` with IDENTICAL amino-acid composition.

    This is the primary negative control: the scramble has the same residues (so the same
    ESM "bag of amino acids" and the same chai tokenisation budget, including any required
    glycine) but a destroyed sequence order and therefore a destroyed designed interface. A
    genuine binder must out-score its own composition-matched scramble.

    Deterministic given ``rng_seed`` (seeded :class:`random.Random`, so it is independent of
    global RNG state and reproducible across processes). Any glycine — the canonical residue
    that keeps an all-D chain chai-tokenisable — is preserved automatically because the
    multiset of residues is preserved exactly.

    Args:
        seq: one-letter L sequence to scramble (the binder, pre-``to_d_fasta``).
        rng_seed: integer seed for the permutation.

    Returns:
        A string that is a permutation of ``seq`` (``Counter(out) == Counter(seq)``). For a
        homopolymer (or any sequence where every permutation is identical) it equals ``seq``.
    """
    rng = random.Random(rng_seed)
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


# ---------------------------------------------------------------------------
# (2) off-target helix panel — the specificity control
# ---------------------------------------------------------------------------

# A small FIXED panel of unrelated ~21-res L helices used as off-target decoys. These are
# canonical / published helix sequences, deliberately NOT the design target, chosen to span
# the common "generic helix" archetypes so a non-specific Chai response (an ipTM driven by
# "any helix near this target") is exposed. One-letter L codes, ~21 residues each.
_OFF_TARGET_HELICES: tuple[tuple[str, str], ...] = (
    # Classic de novo amphipathic / coiled-coil heptad-repeat decoys (Hodges-style "ABCDEF"
    # GELEALEK... peptides): textbook left-/right-handed coiled-coil reference helices.
    ("coiledcoil_EK", "GELEALEKELEALEKELEALEK"),
    # Glu/Lys + Leu amphipathic helix on the opposite heptad phasing (LEKE... start).
    ("amphipathic_LEK", "LEKELEALEKELEALEKELEAL"),
    # MAX/leucine-zipper-style hydrophobic-seam helix (Leu every 7) — coiled-coil decoy.
    ("leucine_zipper", "MKQLEDKVEELLSKNYHLENEV"),
    # Generic Ala-rich solvated helix (high helix propensity, no designed seam) — a "boring"
    # surface helix control with minimal interface character.
    ("alanine_rich", "AEAAAKEAAAKEAAAKEAAAKA"),
)


def off_target_helices() -> list[tuple[str, str]]:
    """Return the fixed off-target helix panel as a list of ``(name, sequence)`` tuples.

    A small (3-5) set of unrelated ~21-res L helices (generic amphipathic / coiled-coil
    decoys; see :data:`_OFF_TARGET_HELICES`), used to measure the binder's specificity gap:
    each is scored against the SAME target with the SAME token split, and a genuine binder
    must clear :data:`WIN_MARGINS`\\ ``["off_target_gap"]`` over the best of them.

    Names are unique; sequences are one-letter L canonical residues. Returns a fresh list so
    callers cannot mutate the module-level panel.
    """
    return list(_OFF_TARGET_HELICES)


# ---------------------------------------------------------------------------
# (2b) register-shifted TARGET decoy — the missing coiled-coil control (Gate-B #2)
# ---------------------------------------------------------------------------

def register_shifted_target_decoy(target_seq: str, shift: int = 3) -> tuple[str, str]:
    """A circular rotation of the REAL target sequence — the heptad-register decoy.

    The generic :func:`off_target_helices` panel tests "is the binder specific vs *unrelated*
    helices?". It does NOT test the subtler coiled-coil failure mode: a binder that latches
    onto "a helix of THIS amino-acid composition" rather than onto THIS target's specific
    heptad register. This decoy closes that gap. Rotating the target sequence circularly by
    ``shift`` residues yields a decoy that is:

      * **composition-identical** to the real target (same bag of residues, so same ESM/Chai
        "amino-acid budget" and the same overall helix propensity), yet
      * **register-shifted** — the heptad ``a b c d e f g`` phase is rotated by ``shift``,
        so the buried-core (a/d) seam now falls on different residues.

    A binder that is truly specific to the target's register should NOT bind this decoy: it
    therefore belongs in the worst-decoy panel (scored by the driver under the label
    ``offtgt:register_shift``). This is a TARGET-DERIVED decoy and is deliberately distinct
    from the generic, target-agnostic :func:`off_target_helices`.

    Args:
        target_seq: the REAL target one-letter sequence (load via
            :func:`xenodesign.seed.read_target_sequence`; never inline it).
        shift: number of residues to rotate left, modulo ``len(target_seq)`` (default 3 — a
            non-heptad-multiple so the register genuinely changes for a 7-residue repeat).

    Returns:
        ``(name, decoy_seq)`` where ``name`` is ``"register_shift"`` and ``decoy_seq`` is the
        circular rotation ``target_seq[shift:] + target_seq[:shift]`` (composition-identical to
        the input; equal to the input only when ``shift % len`` is 0).
    """
    n = len(target_seq)
    s = (shift % n) if n else 0
    decoy = target_seq[s:] + target_seq[:s]
    return "register_shift", decoy


# ---------------------------------------------------------------------------
# (3) interface footprint + heptad register
# ---------------------------------------------------------------------------

# Heptad positions a..g; the buried coiled-coil core is a and d, the rest are surface.
_HEPTAD = "abcdefg"
_CORE_POSITIONS = frozenset("ad")


def heptad_register(res_index: int, start: str = "a") -> tuple[str, str]:
    """Heptad-register letter and core/surface face for a 0-based helix residue index.

    The coiled-coil heptad repeat assigns positions ``a b c d e f g`` cyclically along a
    helix; ``a`` and ``d`` form the buried hydrophobic core, the rest (``b c e f g``) face
    solvent. A real binder should contact the SURFACE positions of a helical target.

    Args:
        res_index: 0-based residue index along the target helix (resid 1 -> index 0).
        start: heptad letter assigned to ``res_index == 0`` (default 'a'); lets a caller phase
            the register to a known core position from a structure/sequence analysis.

    Returns:
        ``(heptad_letter, face)`` where ``face`` is ``"core"`` (a/d) or ``"surface"``.
    """
    if start not in _HEPTAD:
        raise ValueError(f"start must be one of {_HEPTAD!r}, got {start!r}")
    offset = _HEPTAD.index(start)
    letter = _HEPTAD[(res_index + offset) % 7]
    face = "core" if letter in _CORE_POSITIONS else "surface"
    return letter, face


def interface_footprint(cif_path, target_chain, binder_chain, *,
                        contact_dist: float = CONTACT_DIST_A,
                        heptad_start: str = "a") -> dict:
    """Which TARGET residues the binder contacts, with a heptad-register classification.

    Parses CA coordinates from a Chai-style complex CIF (reusing
    :func:`xenodesign.metrics._parse_cif_ca`) and reports, for the ``target_chain``, every
    residue whose CA lies within ``contact_dist`` Angstrom of any ``binder_chain`` CA — i.e.
    the binder's footprint on the target. Each contacted target residue is assigned a heptad
    register letter (a..g) and a core/surface face via :func:`heptad_register`, phased so that
    the first residue of the target chain (in CIF order) gets ``heptad_start``.

    NOTE on the contact metric: this uses CA-CA distance (the CIF reliably has CA atoms for
    every residue, including chai's atom-tokenized D residues). A heavy-atom <8 A contact is a
    stricter physical definition; CA-CA <8 A is the documented, parse-robust proxy used here.
    A real surface-docking binder yields a footprint dominated by b/c/e/f/g (high
    ``surface_fraction``); a footprint sitting on the buried a/d core is physically suspect.

    Args:
        cif_path: path to a 2-chain (target + binder) predicted complex CIF.
        target_chain: label_asym_id of the fixed target chain (e.g. 'B').
        binder_chain: label_asym_id of the designed binder chain (e.g. 'A').
        contact_dist: CA-CA contact cutoff in Angstrom (default 8.0).
        heptad_start: heptad letter for the first target residue (default 'a').

    Returns:
        dict with:
          ``contacted_resids``: sorted list of contacted target resids (label_seq_id, ints).
          ``n_contacted``: number of contacted target residues.
          ``n_target``: total target residues parsed.
          ``registers``: ``{resid: {"heptad": str, "face": "core"|"surface", "index": int}}``
              for each contacted target residue.
          ``surface_fraction``: fraction of contacted residues on the surface (b/c/e/f/g);
              0.0 when nothing is contacted.
          ``contact_dist``: the cutoff used.
    """
    rows = list(_parse_cif_ca(cif_path))
    target_rows = [r for r in rows if r[0] == target_chain]
    binder_rows = [r for r in rows if r[0] == binder_chain]
    if not target_rows:
        raise ValueError(f"no CA atoms for target_chain {target_chain!r} in {cif_path}")
    if not binder_rows:
        raise ValueError(f"no CA atoms for binder_chain {binder_chain!r} in {cif_path}")

    binder_xyz = np.array([[x, y, z] for (_c, _r, x, y, z, _b) in binder_rows], dtype=float)

    # CIF residue order -> 0-based index for heptad phasing; resid (label_seq_id) -> int.
    contacted_resids: list[int] = []
    registers: dict[int, dict] = {}
    for idx, (_c, resid_str, x, y, z, _b) in enumerate(target_rows):
        ca = np.array([x, y, z], dtype=float)
        if float(np.min(np.linalg.norm(binder_xyz - ca, axis=1))) < contact_dist:
            try:
                resid = int(resid_str)
            except (TypeError, ValueError):
                resid = idx + 1  # fall back to 1-based position if label_seq_id is non-numeric
            letter, face = heptad_register(idx, start=heptad_start)
            contacted_resids.append(resid)
            registers[resid] = {"heptad": letter, "face": face, "index": idx}

    contacted_resids.sort()
    n_contacted = len(contacted_resids)
    n_surface = sum(1 for r in contacted_resids if registers[r]["face"] == "surface")
    surface_fraction = (n_surface / n_contacted) if n_contacted else 0.0

    return {
        "contacted_resids": contacted_resids,
        "n_contacted": n_contacted,
        "n_target": len(target_rows),
        "registers": registers,
        "surface_fraction": surface_fraction,
        "contact_dist": contact_dist,
    }


# ---------------------------------------------------------------------------
# GT reference binder loader (aggregate-only; never inlines the sequence)
# ---------------------------------------------------------------------------

def load_gt_reference_binder(fasta_path: "str | Path | None" = None) -> str:
    """Load the published reference D-binder as a one-letter L sequence (chain-A record).

    Reads the FIRST protein record of the reference trimer FASTA (the published D-binder,
    written in chai parenthesized D form) and decodes it to one-letter L codes via
    :func:`xenodesign.io_spec.d_fasta_to_one_letter`. The campaign driver uses this fixed
    binder as the POSITIVE anchor for every control comparison.

    This is gitignored real data; callers/tests must assert only AGGREGATE properties (the
    binder is a 21-res helix) and must NEVER inline, log, or commit the returned sequence.

    Args:
        fasta_path: override path; defaults to :data:`GT_REFERENCE_FASTA`.

    Returns:
        The reference binder as a one-letter L sequence (length 21 for the trimer D/L-ABLE
        reference).
    """
    from xenodesign.io_spec import d_fasta_to_one_letter

    path = Path(fasta_path) if fasta_path is not None else GT_REFERENCE_FASTA
    name, raw = _first_fasta_record(path)
    return d_fasta_to_one_letter(raw)


def _first_fasta_record(path: Path) -> tuple[str, str]:
    """Return (header_without_'>', concatenated_sequence) for the FIRST record in a FASTA."""
    header: str | None = None
    seq_lines: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:]
            elif header is not None:
                seq_lines.append(line.strip())
    if header is None:
        raise ValueError(f"no FASTA record found in {path}")
    return header, "".join(seq_lines)


# ---------------------------------------------------------------------------
# (3b) multi-seed aggregation — fold K per-seed verdict JSONs (Gate-B #3)
# ---------------------------------------------------------------------------

# Numeric verdict metrics worth aggregating across seeds as mean +/- spread.
_MULTISEED_METRICS: tuple[str, ...] = (
    "design_iptm", "design_ipae", "reference_iptm",
    "max_scramble_iptm", "min_scramble_ipae",
    "worst_offtarget_iptm", "mean_offtarget_iptm",
    "min_offtarget_ipae", "mean_offtarget_ipae",
)


def aggregate_multiseed(verdict_paths) -> dict:
    """Fold K per-seed ``controls_verdict.json`` files into per-metric mean +/- std.

    The multi-seed protocol (Gate-B #3) runs :mod:`scripts.score_controls` K times with
    different ``--seed`` values; each run writes a ``controls_verdict.json`` of the shape
    ``{"verdict": {...}, "rows": [...]}``. This helper reads those K files and reports, for
    every numeric verdict metric present (see :data:`_MULTISEED_METRICS`), the mean and
    population standard deviation ACROSS runs, plus the fraction of runs that called the design
    SPECIFIC. A metric is averaged only over the runs that actually report it (a run missing a
    metric — e.g. a dropped channel — is skipped for THAT metric, not counted as zero).

    Args:
        verdict_paths: iterable of paths to per-seed verdict JSONs (each a dict with a
            top-level ``"verdict"`` mapping, or the verdict mapping itself).

    Returns:
        dict with:
          ``n_runs``: number of verdict files read.
          ``<metric>``: ``{"mean": float, "std": float, "n": int}`` for each numeric metric
              present in >= 1 run (``std`` is the population std; 0.0 for a single value).
          ``specific_fraction``: fraction of runs whose ``SPECIFIC`` is truthy (None if no run
              reports SPECIFIC).
    """
    import json
    import math

    verdicts: list[dict] = []
    for p in verdict_paths:
        data = json.loads(Path(p).read_text())
        v = data.get("verdict", data) if isinstance(data, dict) else {}
        verdicts.append(v)

    out: dict = {"n_runs": len(verdicts)}
    for metric in _MULTISEED_METRICS:
        vals = [v[metric] for v in verdicts
                if isinstance(v, dict) and v.get(metric) is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)   # population variance
        out[metric] = {"mean": mean, "std": math.sqrt(var), "n": len(vals)}

    specifics = [bool(v.get("SPECIFIC")) for v in verdicts
                 if isinstance(v, dict) and "SPECIFIC" in v]
    out["specific_fraction"] = (sum(specifics) / len(specifics)) if specifics else None
    return out


# ---------------------------------------------------------------------------
# (4) GPU control protocol — documented recipe (executed by the campaign driver)
# ---------------------------------------------------------------------------

def gpu_control_protocol() -> str:
    r"""The exact GPU scoring recipe for the within-Chai circularity controls (DOC ONLY).

    This module provides the CPU builders + footprint analysis; the *campaign driver* runs the
    GPU scoring below. The protocol scores, against the SAME L-target and with the SAME token
    split (binder = chain 0, target = chain 1), each of:

      * the FIXED reference D-binder (positive anchor; load its length via
        :func:`load_gt_reference_binder` — NEVER inline the sequence),
      * each composition-matched scramble (:func:`composition_matched_scramble`, several seeds),
      * each off-target helix (:func:`off_target_helices`),

    and does so BOTH with the pocket restraint ON and OFF (the restraint-off control). For
    every prediction it computes :func:`xenodesign.benchmark.case_metrics.case_metrics`
    (ipAE, Dunbrack ipSAE, interface ipTM) on the best model.

    Token split (CRITICAL): keep the binder as chain ``A`` and the target as chain ``B`` in
    the entities list so that ``case_metrics`` / ``score_interface`` use ``chain_a=0,
    chain_b=1`` consistently across the anchor, scrambles and off-targets. The binder must be
    glycine-safe (see :func:`xenodesign.io_spec.glycine_satisfy_guard`) before ``to_d_fasta``.

    Entities dict (one prediction). ``BINDER_SEQ`` is a one-letter L sequence supplied at
    runtime — the reference binder (from :func:`load_gt_reference_binder`), a scramble, or an
    off-target helix; ``TARGET_SEQ`` is the fixed L-target (chain-B record of
    :data:`GT_REFERENCE_FASTA`). Chirality 'D' triggers parenthesized-D conversion in
    ``build_fasta`` for the binder; the target stays 'L'::

        entities = [
            {"type": "protein", "name": "binder", "sequence": BINDER_SEQ, "chirality": "D"},
            {"type": "protein", "name": "target", "sequence": TARGET_SEQ, "chirality": "L"},
        ]

    Command (per condition; run once with the restraint and once without)::

        from pathlib import Path
        from xenodesign.backends.chai_backend import ChaiBackend
        from xenodesign.benchmark.case_metrics import case_metrics
        from xenodesign.benchmark.cases import get_case
        from xenodesign.eval.controls import (
            composition_matched_scramble, off_target_helices, load_gt_reference_binder,
            interface_footprint, WIN_MARGINS,
        )

        case = get_case("alpha")                       # fixed L-target context
        backend = ChaiBackend(seed=42)  # device auto-resolves (XENO_DEVICE / cuda:0 / mps / cpu)
        target_seq = TARGET_SEQ                         # chain-B record of the GT FASTA

        def score(binder_seq, out_dir, restraint):
            ents = [
                {"type": "protein", "name": "binder", "sequence": binder_seq, "chirality": "D"},
                {"type": "protein", "name": "target", "sequence": target_seq, "chirality": "L"},
            ]
            backend.predict(ents, out_dir,
                            constraint_path=(case.restraint_path if restraint else None))
            return case_metrics(case, Path(out_dir) / "chai_out")

        anchor = load_gt_reference_binder()             # 21-res reference D-binder (NEVER print)
        conditions = {"anchor": anchor}
        for s in (0, 1, 2):
            conditions[f"scramble_{s}"] = composition_matched_scramble(anchor, rng_seed=s)
        for name, seq in off_target_helices():
            conditions[f"offtarget_{name}"] = seq

        results = {}
        for restraint in (True, False):
            tag = "restr_on" if restraint else "restr_off"
            for cond, seq in conditions.items():
                results[(cond, tag)] = score(seq, f"runs/controls/{tag}/{cond}", restraint)

        # Footprint of the anchor complex (restraint on): expect SURFACE-dominated contacts.
        fp = interface_footprint("runs/controls/restr_on/anchor/chai_out/pred.model_idx_0.cif",
                                 target_chain="B", binder_chain="A")

    Decision (PRE-REGISTERED margins, :data:`WIN_MARGINS`). The anchor passes the within-Chai
    circularity controls iff ALL hold (restraint ON unless noted):

      * vs EVERY scramble:        anchor.ipTM - scramble.ipTM   >= 0.08  (``iptm_gap``)
                                  scramble.ipAE - anchor.ipAE   >= 2.0 A (``ipae_gap_A``)
      * vs BEST off-target:       anchor.ipTM - best_offtarget.ipTM >= 0.15 (``off_target_gap``)
      * restraint robustness:     anchor.ipTM(restraint OFF) still beats the scramble ipTM
                                  (a win that vanishes without the restraint is circular).
      * footprint:                ``fp["surface_fraction"]`` high (binder docks b/c/e/f/g,
                                  not the buried a/d core).

    Returns:
        This docstring as a string (so a driver can echo the protocol into a run log).
    """
    return gpu_control_protocol.__doc__
