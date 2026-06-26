"""Seed sequence generation for D-peptide design.

A seed is a one-letter L sequence; chirality (D) is applied downstream by
`io_spec.to_d_fasta`. `retro_inverso` performs the sequence reversal half of the
retro-inverso transform (the all-D half is the bracketed-CCD encoding in io_spec).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

AA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"


@dataclass(frozen=True)
class SeedResult:
    """A built seed + the per-case provenance flags the loop/harness need.

    one_letter: the L one-letter seed (chirality applied downstream by io_spec.to_d_fasta).
    length: the matched reference-binder length (== len(one_letter) unless a glycine-satisfy
        guard grew/edited it; callers should treat one_letter as authoritative).
    reverse_applied: True iff retro-inverso reversal was applied at the seed (spec §2.1b).
    conditioned: True iff the generator conditioned on a target sequence (PepMLM); False for
        the cyclic unconditioned path.
    fixed_chirality: {1-based position: 'D'|'L'} positions pinned by a manual chirality prior
        (e.g. the cyclic His coordinating positions); empty for the pure all-D cases.
    """
    one_letter: str
    length: int
    reverse_applied: bool = False
    conditioned: bool = False
    fixed_chirality: dict = field(default_factory=dict)


def retro_inverso(seq_one_letter: str, reverse: bool = True) -> str:
    """Return the (optionally reversed) sequence. Reverse is meaningful when mimicking
    a reference L-binder; for pure de novo it may be disabled (spec §2.1b)."""
    return seq_one_letter[::-1] if reverse else seq_one_letter


class RandomSeedGenerator:
    """Baseline seed generator: uniform random L peptide (for clean benchmarking)."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def generate(self, length: int) -> str:
        return "".join(self._rng.choice(AA_ALPHABET) for _ in range(length))

    def generate_conditioned(self, target_seq: str, length: int) -> str:
        """Target-tolerant alias for `generate` (target is IGNORED — this generator is
        unconditioned). Lets the seeding dispatcher call conditioned (PepMLM) and
        unconditioned generators through one `(target_seq, length)` signature."""
        return self.generate(length)


class PepMLMSeedGenerator:
    """Target-conditioned peptide seed via PepMLM, with retro-inverso applied.

    `generate_fn(target_seq, length) -> peptide` lets tests inject a fake and lets
    callers swap the backend. If not provided, the real PepMLM model is loaded lazily
    on first use (requires `transformers` + `torch`, and downloads the weights).
    """

    HF_MODEL = "ChatterjeeLab/PepMLM-650M"

    def __init__(
        self,
        generate_fn: Optional[Callable[[str, int], str]] = None,
        reverse: bool = True,
        seed: Optional[int] = None,
        temperature: float = 1.0,
    ):
        self._generate_fn = generate_fn
        self._reverse = reverse
        # seed != None -> per-run TEMPERATURE SAMPLING (so each campaign run gets a DIFFERENT
        # PepMLM seed); seed None -> the old deterministic argmax (back-compat).
        self._seed = seed
        self._temperature = float(temperature)

    def _real_generate(self, target_seq: str, length: int) -> str:  # pragma: no cover (network)
        """Generate a target-conditioned peptide via PepMLM masked-fill.

        BEST-EFFORT (simplified single-pass masked-fill). For the exact PepMLM protocol
        (iterative decoding, top-k sampling, pseudo-perplexity ranking) see
        https://github.com/programmablebio/pepmlm. VERIFY ON YOUR HARDWARE.
        Downloads weights on first use → mark callers with the 'network' pytest marker.
        """
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        # ponytail: PepMLM on GPU when available — CPU forward stalls the loop while the GPU idles.
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(self, "_model"):
            self._tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL)
            self._model = AutoModelForMaskedLM.from_pretrained(self.HF_MODEL).eval().to(dev)

        tok = self._tokenizer
        # PepMLM input: target sequence followed by the masked peptide region.
        masked = target_seq + tok.mask_token * length
        enc = {k: v.to(dev) for k, v in tok(masked, return_tensors="pt").items()}
        with torch.no_grad():
            logits = self._model(**enc).logits[0].cpu()

        mask_positions = (enc["input_ids"][0].cpu() == tok.mask_token_id).nonzero(as_tuple=True)[0]
        sel = logits[mask_positions]   # (length, vocab)

        if self._seed is None:
            # Deterministic argmax (back-compat — the OLD behaviour that made every run identical).
            pred_ids = sel.argmax(dim=-1)
            peptide = "".join(tok.convert_ids_to_tokens(pred_ids.tolist()))
            return "".join(ch for ch in peptide if ch in AA_ALPHABET)[:length]

        # Per-run TEMPERATURE SAMPLING over the 20 canonical AAs ONLY (restricting to AA columns
        # guarantees exactly `length` valid residues — argmax could be stripped to a short seq).
        aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in AA_ALPHABET])
        aa_logits = sel[:, aa_ids] / max(self._temperature, 1e-6)
        probs = torch.softmax(aa_logits, dim=-1)
        gen = torch.Generator().manual_seed(int(self._seed))
        idx = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)  # (length,)
        return "".join(AA_ALPHABET[int(i)] for i in idx.tolist())

    def generate(self, target_seq: str, length: int) -> str:
        fn = self._generate_fn or self._real_generate
        peptide = fn(target_seq, length)
        return retro_inverso(peptide, reverse=self._reverse)

    def generate_conditioned(self, target_seq: str, length: int) -> str:
        """Unified-seed alias for :meth:`generate` (same target-conditioned signature).

        Lets the single ``seed.unified_seed`` path drive PepMLM and the unconditioned generators
        through ONE ``generate_conditioned(target_seq, length)`` signature. ``target_seq=""``
        yields an UNCONDITIONAL masked-fill (PepMLM still produces a length-N peptide — the mask
        block has no target prefix to condition on)."""
        return self.generate(target_seq=target_seq, length=length)


def make_configured_generator(cfg):
    """Build the from-scratch seed generator the dispatcher configured (the SINGLE PepMLM path).

    ``cfg.use_pepmlm`` True (default) -> the real target-conditioned PepMLM masked-fill
    (network: weights download on first use; per-run temperature sampling keyed by ``cfg.seed``).
    ``use_pepmlm`` False -> an OFFLINE deterministic stand-in routed through the SAME
    ``generate_conditioned(target_seq, length)`` signature, so the unified seed path is identical
    on CPU/offline runs (it just ignores the target). Both paths apply NO retro-inverso here
    (``reverse=False``): the unified-seed caller owns reversal so the provenance flag is explicit.
    """
    if cfg.use_pepmlm:
        return PepMLMSeedGenerator(reverse=False, seed=cfg.seed)
    # Offline deterministic stand-in (ignores the target) on the SAME conditioned signature —
    # RandomSeedGenerator.generate_conditioned already does exactly this.
    return RandomSeedGenerator(seed=cfg.seed)


def unified_seed(generator, target_seq: str, length: int, *, reverse: bool = False,
                 fixed_positions: Optional[dict] = None,
                 fixed_residue: str = "H") -> SeedResult:
    """ONE from-scratch seed path for ALL binder classes (the unified-seeding principle).

    Every class — including the validated alpha — seeds through this single primitive: it
    ALWAYS calls ``generator.generate_conditioned(target_seq, length)``. When there is a protein
    target (alpha / non_alpha) pass that target sequence; when there is NO protein target
    (cyclic / metal, no-target) pass ``target_seq=""`` and the PepMLM generator degrades to an
    UNCONDITIONAL fill. The seed therefore NEVER inherits the real binder's sequence, scaffold,
    or length — only the from-scratch generator + the configured ``length`` shape it.

    Args:
        generator: any object exposing ``generate_conditioned(target_seq, length) -> str``
            (PepMLMSeedGenerator for the unified PepMLM path; RandomSeedGenerator / an injected
            fake in CPU tests). PepMLM already applies its OWN retro-inverso per its ``reverse``
            flag, so pass ``reverse=False`` here when the generator reverses — ``reverse`` below
            is only for generators that do NOT.
        target_seq: the protein target sequence to condition on, or ``""`` for unconditional.
        length: the from-scratch binder length (clamp 6..50 is the caller's responsibility).
        reverse: apply retro-inverso reversal HERE (only for generators that don't reverse).
        fixed_positions: OPT-IN scaffolding only — a {1-based pos: 'D'|'L'} map of residues to
            pin (e.g. cyclic coordinating His). ``None``/``{}`` (the default) places nothing;
            the seed carries no mandatory scaffold.
        fixed_residue: the one-letter code placed at ``fixed_positions`` (default 'H' = His).

    Returns:
        SeedResult(one_letter, length, reverse_applied, conditioned, fixed_chirality).
    """
    raw = generator.generate_conditioned(target_seq=target_seq, length=length)
    if reverse:
        raw = retro_inverso(raw, reverse=True)
    fixed: dict = {}
    if fixed_positions:
        raw, fixed = insert_fixed_chirality(raw, positions=dict(fixed_positions),
                                            residue=fixed_residue)
    return SeedResult(
        one_letter=raw, length=length, reverse_applied=bool(reverse),
        conditioned=bool(target_seq), fixed_chirality=fixed,
    )


def select_seeds_by_mirror_consistency(seeds, axis: int = 0, threshold: float = 0.1) -> list:
    """Keep seeds whose mirror twin matches within `threshold` (forward-only Tier-1, spec §2.2).

    Each seed is a dict with 'coords' and 'twin_coords' (the predicted mirror-twin complex).
    """
    from xenodesign.mirror import mirror_discrepancy
    kept = []
    for s in seeds:
        if mirror_discrepancy(s["coords"], s["twin_coords"], axis=axis) <= threshold:
            kept.append(s)
    return kept


def reflect_binder_in_complex_from_cif(
    cif_path,
    binder_chain: str = "B",
    axis: int = 0,
):
    """Return a chirality-correct D-seed by reflecting only the binder chain in a complex CIF.

    This implements the double-flip seeding strategy (spec §2.3/§4 "mirror of an L design"):

      1. Caller predicts L-binder + L-target with chai (in-manifold for L → clean 0 chirality).
      2. This function reads the resulting CIF and reflects ONLY the binder chain atoms
         along ``axis`` (default: x).  Target atoms remain at their original L-geometry.
      3. The returned (n_atoms, 3) array is a chirality-correct D-seed for the loop:
         the binder portion has D geometry while the target is still L-compatible,
         so ``truncated_refine`` at low σ (50 steps) can polish both chains without
         needing to re-fold from scratch.

    Why reflect only the binder (not the full complex)?
    At ref_time_steps=50 the sigma noise level is ~0.15 Å (barely any perturbation).
    If we reflected the full complex, the target chain would be in mirror-L geometry and
    50 low-sigma steps cannot re-fold it back to L — we would get a garbled target.
    By leaving the target in L-coordinates (the frame chai expects for an L chain with
    ESM embeddings), only the binder needs to be refined from its D-correct seed.

    Args:
        cif_path: path to the CIF output from the L-seed predict.
        binder_chain: chain letter for the binder in that CIF (default 'B', which is
            chai's default for the second entity).
        axis: reflection axis (0 = x, default); must match the axis used everywhere
            else in the pipeline (``mirror.reflect_coords`` / ``SequenceUpdater``).

    Returns:
        np.ndarray of shape (n_atoms, 3): complex coordinates with binder reflected.
    """
    import gemmi
    import numpy as np
    from xenodesign.mirror import reflect_coords

    structure = gemmi.read_structure(str(cif_path))
    all_coords: list = []
    for model in structure:
        for chain in model:
            in_binder = (chain.name == binder_chain)
            for res in chain:
                for atom in res:
                    coord = [atom.pos.x, atom.pos.y, atom.pos.z]
                    all_coords.append((coord, in_binder))
        break  # first model only

    if not all_coords:
        raise ValueError(f"No atoms found in CIF {cif_path}")

    coords = np.array([c for c, _ in all_coords], dtype=np.float32)
    binder_mask = np.array([in_b for _, in_b in all_coords], dtype=bool)

    if not binder_mask.any():
        raise ValueError(
            f"Binder chain '{binder_chain}' not found in CIF {cif_path}. "
            "Check the chain letter (Chai default: 'B' for the second entity)."
        )

    # Reflect ONLY the binder chain atoms along the chosen axis.
    reflected = reflect_coords(coords, axis=axis)
    coords[binder_mask] = reflected[binder_mask]
    return coords


def insert_fixed_chirality(seq_one_letter: str, positions: dict, residue: str = "H"):
    """Overwrite 1-based `positions` with `residue` and record their handedness.

    Used by the cyclic seed: PepMLM can't condition on a metal, so we generate
    unconditioned and MANUALLY place the coordinating His (D/L) at the metal-coordinating
    positions (spec §2.2/§2.8). The one-letter alphabet is chirality-agnostic — the per-
    position D/L is RECORDED in the returned map and applied downstream by the mixed-chirality
    encoder, never here. Ring/length is preserved (in-place overwrite).

    Args:
        seq_one_letter: the unconditioned L one-letter seed.
        positions: {1-based position: 'D'|'L'} for each coordinating residue.
        residue: the one-letter code to place (default 'H' = His).

    Returns:
        (new_seq, fixed_chirality) where fixed_chirality == positions (validated).

    Raises:
        ValueError if any position is out of [1, len(seq)] or any value not in {'D','L'}.
    """
    chars = list(seq_one_letter)
    for pos, hand in positions.items():
        if not (1 <= pos <= len(chars)):
            raise ValueError(
                f"position {pos} out of range 1..{len(chars)} for seq of len {len(chars)}")
        if hand not in ("D", "L"):
            raise ValueError(f"chirality at position {pos} must be 'D' or 'L', got {hand!r}")
        chars[pos - 1] = residue
    return "".join(chars), dict(positions)


def read_target_sequence(fasta_path, name: Optional[str] = None) -> str:
    """Read one record's sequence from a (possibly gitignored) Chai-style FASTA.

    Header form `>protein|<name>` or `>` + free text; the name match is a substring of the
    header text after `>`. Concatenates wrapped sequence lines and strips whitespace.

    Args:
        fasta_path: path to the FASTA (e.g. case.fasta_path; gitignored for alpha).
        name: substring to match in a header; if None, returns the FIRST record's sequence.

    Returns:
        The selected record's one-letter sequence (header line stripped).

    Raises:
        KeyError if `name` is given but no header contains it.
        ValueError if the file has no records.
    """
    records: list[tuple[str, str]] = []
    header, chunks = None, []
    for line in Path(fasta_path).read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line[1:].strip(), []
        elif header is not None:
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise ValueError(f"no FASTA records in {fasta_path}")
    if name is None:
        return records[0][1]
    for hdr, seq in records:
        if name in hdr:
            return seq
    raise KeyError(f"no FASTA record whose header contains {name!r} in {fasta_path}")
