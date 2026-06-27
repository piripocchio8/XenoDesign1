"""Mirror-into-L preprocessing for inverse folding (spec §2.7).

Neither LigandMPNN (MIT, default) nor CARBonAra (CC-BY-NC-SA, research-only) is D-native:
both treat the structure as an all-atom context-aware graph and produce L sequences, so both
require this double-flip preprocessing. We reflect the whole complex (one rigid reflection
preserves the interface): the D design chain becomes L (designable as protein), the fixed
partner becomes all-atom context (coords+elements). The designed letters apply directly to the
D-peptide; chirality is re-applied downstream via io_spec CCD codes. CARBonAra's role is as a
comparison against LigandMPNN inside the double-flip, to test whether it drifts less toward L
over the loop — not as a D-native replacement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence, runtime_checkable

import numpy as np

from xenodesign.io_spec import AA1_TO_AA3
from xenodesign.mirror import D_TO_L, reflect_coords

# Canonical L three-letter codes (the LigandMPNN / CARBonAra output alphabet), incl. GLY.
CANONICAL_L = set(AA1_TO_AA3.values())

# Three-letter L code -> one-letter (the inverse-folding alphabet). Built from AA1_TO_AA3.
_AA3_TO_AA1 = {three: one for one, three in AA1_TO_AA3.items()}


def l_projected_known_seq(codes: Sequence[str]) -> str:
    """Project each per-position residue code to its L-parent ONE-LETTER identity (spec §2.8, B2).

    This is the REAL known sequence fed to the inverse-folding backend so it natively preserves
    fixed positions and conditions free positions on real context — instead of an all-Ala
    placeholder. The projection is chirality-agnostic (a D residue maps to the SAME letter as its
    L parent), matching the mirror-into-L frame the backbone is reflected into:

      * L-canonical (HIS) / achiral GLY -> its own one-letter (H / G).
      * D-canonical (DHI) -> its L parent's one-letter (DHI -> HIS -> 'H').
      * ncAA -> nearest canonical via ``ncaa_proxy.proxy_for`` (e.g. SEP -> SER -> 'S'); if the
        ncAA has no proxy it is GLY ('G', a neutral backbone-only placeholder).
    """
    from xenodesign.ncaa_proxy import proxy_for

    out: list[str] = []
    for code in codes:
        c = str(code).upper()
        three = D_TO_L.get(c, c)        # D-canonical -> L parent; L/achiral pass through
        if three in _AA3_TO_AA1:
            out.append(_AA3_TO_AA1[three])
            continue
        proxy = proxy_for(c)            # genuine ncAA -> conformational proxy
        if proxy and proxy in _AA3_TO_AA1:
            out.append(_AA3_TO_AA1[proxy])
        else:
            out.append("G")            # unrepresentable -> neutral GLY placeholder
    return "".join(out)


@dataclass
class InverseFoldingInputs:
    design_backbone: np.ndarray   # (n_res, 4, 3) reflected to L, order N, CA, C, CB
    context_coords: np.ndarray    # (m_atoms, 3) reflected partner atoms
    context_elements: list[str]   # element symbols of the context atoms


def prepare_inverse_folding_inputs(
    design_backbone,
    context_coords,
    context_elements: Sequence[str],
    axis: int = 0,
    flip: bool = True,
) -> InverseFoldingInputs:
    """Prepare inverse-folding inputs, reflecting the complex iff `flip` is True.

    `flip` must come from `choose_reflection(design_codes)` so the geometry handed to the
    inverse-folding model agrees with `designable_positions(codes, flip)`: when flip is True
    the (D) design chain is reflected to L; when False the chain is already L and is left
    unchanged. A single shared reflection of design + context preserves interface distances.
    """
    design_backbone = np.asarray(design_backbone, dtype=float)
    context_coords = np.asarray(context_coords, dtype=float).reshape(-1, 3)
    if flip:
        design_backbone = reflect_coords(design_backbone, axis)
        context_coords = reflect_coords(context_coords, axis)
    return InverseFoldingInputs(
        design_backbone=design_backbone,
        context_coords=context_coords,
        context_elements=list(context_elements),
    )


def residue_class(code: str) -> str:
    """Classify a three-letter residue code for inverse-folding designability.

    Returns one of: 'achiral_canonical' (GLY), 'L' (standard L), 'D' (standard D),
    'ncAA' (anything else, e.g. SEP/AIB/UAA). ncAA are outside the LigandMPNN/CARBonAra
    output alphabet and are never designable (spec §2.7 exceptions).
    """
    c = code.upper()
    if c == "GLY":
        return "achiral_canonical"
    if c in CANONICAL_L:
        return "L"
    if c in D_TO_L:
        return "D"
    return "ncAA"


def is_designable(code: str, flip: bool) -> bool:
    """Whether a position can be sequence-designed after a global reflection.

    `flip` = whether the whole complex is reflected (swaps L<->D handedness). A position
    is designable iff, after the reflection, it is a canonical L amino acid. GLY (achiral)
    is designable in any frame; ncAA are never designable.
    """
    cls = residue_class(code)
    if cls == "ncAA":
        return False
    if cls == "achiral_canonical":
        return True
    effective_L = (cls == "L") != flip  # XOR: a reflection swaps handedness
    return effective_L


def designable_positions(codes: Sequence[str], flip: bool) -> list[bool]:
    """Boolean mask of positions LigandMPNN/CARBonAra may design after reflection `flip`.
    Non-designable positions (wrong-handed canonicals, ncAA) become all-atom context."""
    return [is_designable(c, flip) for c in codes]


def choose_reflection(codes: Sequence[str]) -> bool:
    """Pick the reflection (flip True/False) that maximizes the number of designable
    positions. Ties resolve to no-flip (False). For an all-D chain this returns True
    (reflect to L); for all-L, False; for mixed chirality, the majority-handedness frame.
    """
    n_no_flip = sum(designable_positions(codes, False))
    n_flip = sum(designable_positions(codes, True))
    return n_flip > n_no_flip


def can_use_ligandmpnn(codes: Sequence[str], flip: bool) -> bool:
    """True iff at least one position is designable in this frame (else LigandMPNN is
    unusable — e.g. an all-ncAA peptide; fall back to fixing the chain + Chai scoring)."""
    return any(designable_positions(codes, flip))


# ---------------------------------------------------------------------------
# Inverse-folding backend protocol (spec §7, task #17).
#
# A backend is a callable that designs the sequence of the DESIGN CHAIN ONLY,
# given its (already mirror-into-L) backbone + the fixed partner as all-atom
# context. It returns a LIST of `num_seqs` candidate one-letter L sequences,
# each the length of the design chain. It NEVER returns the fixed/target chain:
# the harness reconstructs that from the known input (see assemble_complex_fasta,
# spec §4 + XenoDesign1_local_ref/select_carbonara_for_chai.py). LigandMPNN
# (default) and CARBonAra (#18, later) both implement this protocol.
# ---------------------------------------------------------------------------
@runtime_checkable
class InverseFoldingBackend(Protocol):
    def __call__(
        self,
        design_backbone: "np.ndarray",      # (n_res, 4, 3) N,CA,C,CB in L-frame
        context_coords: "np.ndarray",        # (m_atoms, 3) fixed-partner atoms
        context_elements: Sequence[str],     # element symbols of the context atoms
        fixed_mask: Sequence[bool],          # True = keep this design position fixed
        temperature: float,                  # sampling temperature
        num_seqs: int,                       # number of candidate sequences to return
        known_seq: "str | None" = None,      # L-projected real sequence (B2); fixed positions
                                             # are kept FROM this (chain_mask=0), free designed
    ) -> List[str]:                          # designed-chain one-letter L seqs (len n_res)
        ...


def _accepts_known_seq(fn) -> bool:
    """True iff ``fn`` declares a ``known_seq`` parameter (or an **kwargs sink), so the B2 known
    sequence can be forwarded; legacy 6-arg backends/fakes (no such param) are called without it."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "known_seq" in params:
        return True
    return any(p.kind == p.VAR_KEYWORD for p in params.values())


def call_backend(fn, design_backbone, context_coords, context_elements,
                 fixed_mask, temperature, num_seqs, *, known_seq=None):
    """Invoke an InverseFoldingBackend, threading ``known_seq`` (B2) only when it opts in.

    Centralises the known_seq-aware call so every seam routed through it (SequenceUpdater,
    MultiCandidate, the C-term Gly anchor) preserves the L-projected real sequence while staying
    compatible with legacy 6-arg backends and test fakes.

    Known gap: beam.py bypasses this helper, so beam search does NOT yet thread known_seq through
    call_backend; this is subsumed by the planned composer refactor."""
    if known_seq is not None and _accepts_known_seq(fn):
        return fn(design_backbone, context_coords, context_elements,
                  fixed_mask, temperature, num_seqs, known_seq=known_seq)
    return fn(design_backbone, context_coords, context_elements,
              fixed_mask, temperature, num_seqs)


def is_inverse_folding_backend(fn) -> bool:
    """True iff `fn` matches the InverseFoldingBackend call protocol (6 positional
    parameters: design_backbone, context_coords, context_elements, fixed_mask,
    temperature, num_seqs). Used to distinguish a new-protocol backend from a legacy
    4-arg design_fn so SequenceUpdater can adapt either (Task 3)."""
    import inspect

    if not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values())
    # The protocol passes 6 positional args plus the OPTIONAL `known_seq` (B2). Accept a callable
    # that takes the 6 required positional args (the 7th, known_seq, may be positional-with-default
    # or keyword-only) or an *args sink.
    n_required_positional = sum(
        1 for p in positional if p.default is p.empty
    )
    return (
        (6 <= len(positional) <= 7 and n_required_positional <= 6)
        or (has_var_positional and n_required_positional <= 6)
    )


def assemble_complex_fasta(
    fixed_chain_seq_from_input: str,
    designed_chain_letters: str,
    design_name: str = "design_A",
    fixed_name: str = "target_B",
    design_chirality: str = "D",
    fixed_chirality: str = "L",
) -> str:
    """Assemble the heterochiral complex FASTA for the next Chai cycle (output-normalization,
    spec §4).

    The inverse-folding backend returns ONLY the designed chain. We reconstruct the
    fixed/target chain from the KNOWN input (`fixed_chain_seq_from_input`, a one-letter L
    sequence the caller already has) and seq-only-mirror the designed chain to D-CCD. We NEVER
    trust the backend to echo the fixed chain: CARBonAra cannot emit D-CCD codes (see
    XenoDesign1_local_ref/select_carbonara_for_chai.py::get_fixed_seq_from_input), and our
    LigandMPNN call passes the target as ligand-context atoms, so it is never a protein chain
    in the output either.

    Args:
        fixed_chain_seq_from_input: the known fixed/target chain as a one-letter L sequence.
        designed_chain_letters: the designed chain as a one-letter L sequence (from the backend).
        design_name / fixed_name: chai FASTA entity names.
        design_chirality / fixed_chirality: 'D' -> parenthesized D-CCD, 'L' -> kept as L letters.

    Returns:
        A Chai FASTA string with the design chain first, then the fixed/target chain.
    """
    from xenodesign.io_spec import to_d_fasta

    def _encode(seq: str, chirality: str) -> str:
        return to_d_fasta(seq) if str(chirality).upper() == "D" else seq

    design_seq = _encode(designed_chain_letters, design_chirality)
    fixed_seq = _encode(fixed_chain_seq_from_input, fixed_chirality)
    return (
        f">protein|{design_name}\n{design_seq}\n"
        f">protein|{fixed_name}\n{fixed_seq}\n"
    )


class MultiCandidate:
    """Wrap an InverseFoldingBackend to oversample `num_seqs` candidates and keep the best
    `top_k` by `key_fn` (spec §5 drift lever: oversample -> filter). Itself an
    InverseFoldingBackend (6-arg call, returns a `top_k`-element list, best first), so it
    injects wherever a backend does — e.g. as a SequenceUpdater's design_fn — keeping loop.py
    untouched. With the default `top_k=1` it returns a 1-element list = the single winner.

    Args:
        backend: the wrapped InverseFoldingBackend.
        num_seqs: how many candidates to sample (forced, overriding any caller value).
        key_fn: candidate -> sortable score; the MAX-scoring candidate wins. Default keeps
            the first (model order), i.e. no re-ranking. Wire an orthogonal Chai re-score
            (chirality-clean, then ipTM) here in P3/P7; this layer only needs the key fn.
        reverse: if True (default) higher key is better; set False to keep the minimum.
        top_k: how many top candidates to return, best first (beam expansion lever).
            Must satisfy 1 <= top_k <= num_seqs. top_k=1 (default) is byte-identical to the
            single-winner behavior.
    """

    def __init__(self, backend, num_seqs: int = 8, key_fn=None, reverse: bool = True,
                 top_k: int = 1):
        if not is_inverse_folding_backend(backend):
            raise TypeError("backend must implement the InverseFoldingBackend protocol "
                            "(6 positional args -> list[str])")
        if int(num_seqs) < 1:
            raise ValueError(f"num_seqs must be >= 1, got {num_seqs}")
        assert 1 <= int(top_k) <= int(num_seqs), \
            f"top_k must satisfy 1 <= top_k <= num_seqs ({num_seqs}), got {top_k}"
        self._backend = backend
        self._num_seqs = int(num_seqs)
        self._key_fn = key_fn
        self._reverse = reverse
        self._top_k = int(top_k)

    def __call__(self, design_backbone, context_coords, context_elements,
                 fixed_mask, temperature, num_seqs, known_seq=None):
        # The wrapper's own num_seqs governs oversampling; the caller's value is ignored
        # (a bare SequenceUpdater passes 1 — MultiCandidate forces the real sample count).
        # known_seq (B2) is forwarded only when the wrapped backend accepts it, so a legacy
        # 6-arg fake (no known_seq param) still works unchanged.
        candidates = call_backend(
            self._backend, design_backbone, context_coords, context_elements,
            fixed_mask, temperature, self._num_seqs, known_seq=known_seq,
        )
        if not candidates:
            raise ValueError("backend returned no candidates")
        # key_fn=None preserves model order (stable sort on a constant key); top_k=1 then
        # reproduces the old [candidates[0]]. With a key_fn, sorted(reverse=...) on a stable
        # sort keeps max/min first, matching the prior single-winner pick byte-for-byte.
        ordered = sorted(candidates, key=self._key_fn or (lambda c: 0),
                         reverse=self._reverse)
        return ordered[: self._top_k]
