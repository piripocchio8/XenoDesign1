"""SequenceUpdater: inverse-folding round-trip for D-peptide design (spec §2.7, §2.8).

mirror-out (reflect complex so the design chain is L; partner -> all-atom context) ->
design_fn (LigandMPNN/CARBonAra; injectable) over designable positions only ->
mirror-back (re-encode designed L letters as D-CCD for the next Chai cycle).

The design backend implements `inverse_folding.InverseFoldingBackend`:
`design_fn(design_backbone, context_coords, context_elements, fixed_mask, temperature,
num_seqs) -> list[str]` returns `num_seqs` one-letter L sequences (the DESIGNED chain only,
each the length of the design chain). The real LigandMPNN adapter is the default; tests
inject a fake. A legacy 4-arg `design_fn` returning a single str is accepted and wrapped.
Non-designable positions (wrong-handed canonicals, ncAA) are marked True in `fixed_mask`
and must be kept fixed by the design backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from xenodesign.inverse_folding import (
    call_backend,
    can_use_ligandmpnn,
    choose_reflection,
    designable_positions,
    is_inverse_folding_backend,
    l_projected_known_seq,
    prepare_inverse_folding_inputs,
)
from xenodesign.io_spec import to_d_fasta

# A backend may be the new InverseFoldingBackend protocol (6 args -> list[str]) or a
# legacy 4-arg design_fn (-> str); SequenceUpdater._as_backend normalises either form.
DesignFn = Callable[..., object]


@dataclass
class SequenceUpdateResult:
    one_letter: str   # designed L one-letter sequence (mirror frame)
    d_fasta: str      # re-encoded D-CCD sequence for the next Chai cycle
    flip: bool        # whether a global reflection was applied


class SequenceUpdater:
    """Drives one inverse-folding round-trip. Accepts EITHER a new-protocol
    InverseFoldingBackend (6 args, returns a list of designed-chain seqs) or a legacy
    4-arg design_fn (returns a single str) — the legacy form is wrapped so old call sites
    and tests keep working. `update` returns the FIRST candidate; multi-candidate keep-best
    is handled by `MultiCandidate` in inverse_folding.py (injected at the sequence_update_fn
    seam), so loop.py is never modified."""

    def __init__(
        self,
        design_fn: Optional[Callable] = None,
        axis: int = 0,
        temperature: float = 0.1,
        frozen_positions: Optional[set] = None,
    ):
        self._design_fn = self._as_backend(design_fn or _ligandmpnn_design_fn)
        self._axis = axis
        self._temperature = temperature
        # 0-based positions (e.g. declared metal coordinators) forced fixed in the MPNN
        # sequence-update REGARDLESS of designability, so pinned donors never drift.
        self._frozen_positions = set(frozen_positions or ())

    @staticmethod
    def _as_backend(fn: Callable):
        """Normalise `fn` to the InverseFoldingBackend protocol. A legacy 4-arg design_fn
        (returns a single str) is wrapped to the 6-arg, list-returning protocol."""
        if is_inverse_folding_backend(fn):
            return fn

        def _adapter(design_backbone, context_coords, context_elements,
                     fixed_mask, temperature, num_seqs, known_seq=None):
            one = fn(design_backbone, context_coords, context_elements, fixed_mask)
            return [one for _ in range(num_seqs)]

        return _adapter

    def update(
        self,
        design_backbone,
        design_codes: Sequence[str],
        context_coords,
        context_elements: Sequence[str],
        chirality_pattern: Optional[dict] = None,
    ) -> SequenceUpdateResult:
        design_backbone = np.asarray(design_backbone, dtype=float)
        flip = choose_reflection(design_codes)
        if not can_use_ligandmpnn(design_codes, flip):
            raise ValueError(
                "no designable (canonical-L-after-reflection) positions; "
                "use fixed-chain + Chai scoring instead (spec §2.8)"
            )
        prepared = prepare_inverse_folding_inputs(
            design_backbone, context_coords, context_elements, axis=self._axis, flip=flip
        )
        designable = designable_positions(design_codes, flip)
        fixed_mask = [not d for d in designable]  # True where position must stay fixed
        # OR in the declared frozen positions (coordinators): force them fixed even when
        # designability would mark them designable, so pinned donors are never mutated.
        if self._frozen_positions:
            fixed_mask = [
                fixed or (i in self._frozen_positions) for i, fixed in enumerate(fixed_mask)
            ]

        # B2: feed the backend the REAL (L-projected) sequence as `known_seq`, so it natively
        # KEEPS the fixed positions (chain_mask=0 there) with their declared identity (a D-His
        # coordinator -> 'H', NOT the old all-Ala placeholder) and conditions free positions on
        # real context. The projection is chirality-agnostic, matching the mirror-into-L frame.
        known_seq = l_projected_known_seq(design_codes)

        candidates = call_backend(
            self._design_fn,
            prepared.design_backbone,
            prepared.context_coords,
            prepared.context_elements,
            fixed_mask,
            self._temperature,
            1,
            known_seq=known_seq,
        )
        one_letter = candidates[0]
        if len(one_letter) != design_backbone.shape[0]:
            raise ValueError(
                f"design_fn returned {len(one_letter)} residues, "
                f"expected {design_backbone.shape[0]}"
            )
        # Mirror-back: re-encode the designed L letters as D-CCD for the next Chai cycle.
        if chirality_pattern is None:
            # Default (all-D) path — byte-identical to the historical behaviour. The whole
            # design chain is treated as D (the primary all-D case).
            d_fasta = to_d_fasta(one_letter)
        else:
            # Per-position handedness (ABC mixed-chirality, spec §5.2): re-encode ONLY the
            # positions marked 'D' as D-CCD and keep the rest at their own handedness, reusing
            # the verified `mixed_chirality_fasta` emit. `chirality_pattern` is keyed 0-based
            # (matching design_codes indexing); `mixed_chirality_fasta` wants 1-based keys, so
            # we translate. Gly stays achiral 'G' regardless of its mark.
            n = design_backbone.shape[0]
            if len(chirality_pattern) != n:
                raise ValueError(
                    f"chirality_pattern has {len(chirality_pattern)} entries, "
                    f"expected {n} (one per design position)"
                )
            # Import `base` before `cyclic` to prime the classes.base<->classes.cyclic
            # import cycle (base defines SeedSpec before re-importing cyclic.Cyclic);
            # importing cyclic first hits a partially-initialised-module ImportError.
            import xenodesign.classes.base  # noqa: F401
            from xenodesign.classes.cyclic import mixed_chirality_fasta

            fixed_chirality = {pos + 1: hand for pos, hand in chirality_pattern.items()}
            d_fasta = mixed_chirality_fasta(one_letter, fixed_chirality=fixed_chirality)
        return SequenceUpdateResult(
            one_letter=one_letter, d_fasta=d_fasta, flip=flip
        )


def make_sequence_update_fn(updater: "SequenceUpdater", extract_fn: Callable, emit: str = "one_letter") -> Callable:
    """Adapt a SequenceUpdater to the loop's ``sequence_update_fn(prediction) -> str`` seam.

    This is the injection point for the whole P2 stack: build the SequenceUpdater with a
    MultiCandidate-wrapped backend (num_seqs + keep-best, spec §5) and pass the result of
    this factory as ``HalluLoop(sequence_update_fn=...)`` — ``loop.py`` is never modified.

    Args:
        updater: a configured SequenceUpdater (its design_fn may be a MultiCandidate).
        extract_fn: ``prediction -> dict`` with keys design_backbone, design_codes,
            context_coords, context_elements (the four SequenceUpdater.update inputs).
        emit: 'one_letter' (default; the loop maps to D-CCD itself, matching the current
            loop.py contract) or 'd_fasta' (return the D-CCD encoding directly).

    Returns:
        ``fn(prediction) -> str``.
    """
    if emit not in ("one_letter", "d_fasta"):
        raise ValueError(f"emit must be 'one_letter' or 'd_fasta', got {emit!r}")

    def _fn(prediction) -> str:
        kw = extract_fn(prediction)
        result = updater.update(**kw)
        return result.one_letter if emit == "one_letter" else result.d_fasta

    return _fn


def _ligandmpnn_design_fn(  # pragma: no cover (gpu/integration)
    design_backbone, context_coords, context_elements,
    fixed_mask, temperature=0.1, num_seqs=1, known_seq=None
) -> list:
    """Default backend (InverseFoldingBackend protocol): LigandMPNN (vendored at
    repo-root LigandMPNN/, MIT).

    Calls the LigandMPNN Python API directly (no prody / no subprocess), constructing
    the protein_dict from the numpy backbone array.  Uses model_type='ligand_mpnn' with
    the ligandmpnn_v_32_010_25 checkpoint so context atoms (Y/Y_t/Y_m tensors) are
    actually consumed by the model — making sequence design context-aware of the target
    interface.  Switching from 'protein_mpnn' (backbone-only) to 'ligand_mpnn' is the
    key fix: the latter conditions on nearby non-protein atoms (the mirrored partner).

    Heavy deps (torch, model_utils) are imported lazily so they do NOT load at module
    import time — the CPU test suite must stay green without GPU/torch available.

    Args:
        design_backbone: np.ndarray (n_res, 4, 3) — N, CA, C, CB in L-frame.
        context_coords: np.ndarray (n_ctx, 3) — partner atoms (may be empty).
        context_elements: list[str] — element symbols (may be empty).
        fixed_mask: list[bool] — True = keep position fixed (chain_mask=0; LigandMPNN keeps the
            identity from `known_seq` there NATIVELY — no force-'A' overwrite, see B2).
        temperature: float — LigandMPNN sampling temperature.
        num_seqs: int — number of candidate sequences to sample and return.
        known_seq: str | None — the REAL L-projected design-chain sequence (one-letter, len n_res).
            Written into protein_dict["S"] so fixed positions are preserved with their declared
            identity and free positions are designed in real sequence context. None -> all-Ala S
            (legacy fallback; only the free positions then carry meaning).

    Returns:
        list[str] of length num_seqs — designed-chain one-letter L sequences (the DESIGNED
        chain ONLY; the target enters as ligand-context atoms and is never echoed), each
        len == n_res, chars ⊆ ARNDCQEGHILKMFPSTWYV.
    """
    # Lazy heavy imports — only executed when this function is actually called.
    import sys
    import pathlib

    import numpy as np
    import torch

    # ------------------------------------------------------------------
    # Locate the vendored LigandMPNN package and weights.
    # Repo layout: repo_root/xenodesign/sequence_update.py → repo_root/LigandMPNN/
    # ------------------------------------------------------------------
    _repo_root = pathlib.Path(__file__).parent.parent
    _lmpnn_dir = _repo_root / "LigandMPNN"
    _ckpt_path = _lmpnn_dir / "model_params" / "ligandmpnn_v_32_010_25.pt"

    if not _ckpt_path.exists():
        raise FileNotFoundError(
            f"LigandMPNN weights not found at {_ckpt_path}. "
            "Run: bash LigandMPNN/get_model_params.sh LigandMPNN/model_params"
        )

    # Add LigandMPNN dir to sys.path so we can import model_utils.
    # We do NOT import data_utils here because it has a top-level `from prody import *`
    # which would fail in environments without prody installed.  Instead we inline
    # only the three small pure-torch/numpy helpers we actually need.
    _lmpnn_str = str(_lmpnn_dir)
    if _lmpnn_str not in sys.path:
        sys.path.insert(0, _lmpnn_str)

    from model_utils import ProteinMPNN  # vendored; no prody dependency

    # ------------------------------------------------------------------
    # Inline minimal helpers from data_utils (avoids the prody import).
    # ------------------------------------------------------------------
    _restype_int_to_str = {
        0: "A", 1: "C", 2: "D", 3: "E", 4: "F", 5: "G", 6: "H",
        7: "I", 8: "K", 9: "L", 10: "M", 11: "N", 12: "P", 13: "Q",
        14: "R", 15: "S", 16: "T", 17: "V", 18: "W", 19: "Y", 20: "X",
    }

    def _get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
        device = CB.device
        mask_CBY = mask[:, None] * Y_m[None, :]
        L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
        L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0
        nn_idx = torch.argsort(L2_AB, -1)[:, :number_of_ligand_atoms]
        L2_AB_nn = torch.gather(L2_AB, 1, nn_idx)
        D_AB_closest = torch.sqrt(L2_AB_nn[:, 0])
        Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1)
        Y_t_r = Y_t[None, :].repeat(CB.shape[0], 1)
        Y_m_r = Y_m[None, :].repeat(CB.shape[0], 1)
        Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
        Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
        Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)
        Y_out = torch.zeros([CB.shape[0], number_of_ligand_atoms, 3],
                            dtype=torch.float32, device=device)
        Y_t_out = torch.zeros([CB.shape[0], number_of_ligand_atoms],
                              dtype=torch.int32, device=device)
        Y_m_out = torch.zeros([CB.shape[0], number_of_ligand_atoms],
                              dtype=torch.int32, device=device)
        n_nn = Y_tmp.shape[1]
        Y_out[:, :n_nn] = Y_tmp
        Y_t_out[:, :n_nn] = Y_t_tmp
        Y_m_out[:, :n_nn] = Y_m_tmp
        return Y_out, Y_t_out, Y_m_out, D_AB_closest

    def _featurize(input_dict, cutoff_for_score=8.0, use_atom_context=True,
                   number_of_ligand_atoms=16, model_type="protein_mpnn"):
        """Minimal reimplementation of data_utils.featurize (no prody)."""
        output_dict = {}
        if model_type == "ligand_mpnn":
            mask = input_dict["mask"]
            Y = input_dict["Y"]
            Y_t = input_dict["Y_t"]
            Y_m = input_dict["Y_m"]
            N  = input_dict["X"][:, 0, :]
            CA = input_dict["X"][:, 1, :]
            C  = input_dict["X"][:, 2, :]
            b = CA - N
            c = C - CA
            a = torch.linalg.cross(b, c)
            CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
            Y, Y_t, Y_m, D_XY = _get_nearest_neighbours(
                CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms)
            mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
            output_dict["mask_XY"] = mask_XY[None,]
            output_dict["Y"]   = Y[None,]
            output_dict["Y_t"] = Y_t[None,]
            output_dict["Y_m"] = Y_m[None,]
            if not use_atom_context:
                output_dict["Y_m"] = 0.0 * output_dict["Y_m"]

        # Renumber R_idx to avoid duplicates (featurize contract from data_utils)
        R_idx_list = []
        count = 0
        R_idx_prev = -100000
        for R_idx_val in list(input_dict["R_idx"]):
            val = int(R_idx_val)
            if R_idx_prev == val:
                count += 1
            R_idx_list.append(val + count)
            R_idx_prev = val
        R_idx_renumbered = torch.tensor(R_idx_list, device=input_dict["R_idx"].device)

        output_dict["R_idx"]          = R_idx_renumbered[None,]
        output_dict["R_idx_original"] = input_dict["R_idx"][None,]
        output_dict["chain_labels"]   = input_dict["chain_labels"][None,]
        output_dict["S"]              = input_dict["S"][None,]
        output_dict["chain_mask"]     = input_dict["chain_mask"][None,]
        output_dict["mask"]           = input_dict["mask"][None,]
        output_dict["X"]              = input_dict["X"][None,]
        return output_dict

    # ------------------------------------------------------------------
    # Build the protein_dict from design_backbone (n_res, 4, 3): N, CA, C, CB.
    # LigandMPNN's encode() uses X[:, :4, :] = N, CA, C, O.
    # We put CB in the O slot (slot 3); the encoder recomputes CB from N/CA/C
    # so the O slot is only used for certain side-chain models (not protein_mpnn).
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    design_backbone = np.asarray(design_backbone, dtype=np.float32)
    n_res = design_backbone.shape[0]

    X = design_backbone.copy()  # (n_res, 4, 3): N, CA, C, CB (CB in O slot)

    # B2: S is the REAL (L-projected) input sequence so fixed positions (chain_mask=0) keep their
    # declared identity and free positions are designed in real context. None -> legacy all-Ala.
    _aa_str_to_int = {v: k for k, v in _restype_int_to_str.items()}
    mask           = np.ones(n_res, dtype=np.int32)
    if known_seq is not None and len(known_seq) == n_res:
        S_input = np.array(
            [_aa_str_to_int.get(aa.upper(), 0) for aa in known_seq], dtype=np.int32
        )
    else:
        S_input = np.zeros(n_res, dtype=np.int32)  # legacy all-Ala fallback
    R_idx          = np.arange(1, n_res + 1, dtype=np.int32)
    chain_labels   = np.zeros(n_res, dtype=np.int32)
    chain_mask_arr = np.array([0.0 if fix else 1.0 for fix in fixed_mask],
                              dtype=np.float32)

    # ------------------------------------------------------------------
    # Context atoms Y/Y_t/Y_m (stub for empty context; element→integer table).
    # ------------------------------------------------------------------
    _element_list_uc = [
        "H","HE","LI","BE","B","C","N","O","F","NE","NA","MG","AL","SI","P","S",
        "CL","AR","K","CA","SC","TI","V","CR","MN","FE","CO","NI","CU","ZN","GA",
        "GE","AS","SE","BR","KR","RB","SR","Y","ZR","NB","MB","TC","RU","RH","PD",
        "AG","CD","IN","SN","SB","TE","I","XE","CS","BA","LA","CE","PR","ND","PM",
        "SM","EU","GD","TB","DY","HO","ER","TM","YB","LU","HF","TA","W","RE","OS",
        "IR","PT","AU","HG","TL","PB","BI","PO","AT","RN","FR","RA","AC","TH","PA",
        "U","NP","PU","AM","CM","BK","CF","ES","FM","MD","NO","LR","RF","DB","SG",
        "BH","HS","MT","DS","RG","CN","UUT","FL","UUP","LV","UUS","UUO",
    ]
    _element_dict = {e: i + 1 for i, e in enumerate(_element_list_uc)}

    context_coords = np.asarray(context_coords, dtype=np.float32)
    n_ctx = context_coords.shape[0]

    if n_ctx > 0:
        Y_t_raw = np.array(
            [_element_dict.get(el.upper(), 0) for el in context_elements], dtype=np.int32
        )
        Y_m_raw = ((Y_t_raw != 1) & (Y_t_raw != 0)).astype(bool)
        Y = context_coords[Y_m_raw]
        Y_t = Y_t_raw[Y_m_raw]
        Y_m = np.ones(len(Y), dtype=np.int32)
        if len(Y) == 0:
            Y = np.zeros([1, 3], np.float32)
            Y_t = np.zeros([1], np.int32)
            Y_m = np.zeros([1], np.int32)
    else:
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)

    # ------------------------------------------------------------------
    # Build protein_dict tensors.
    # ------------------------------------------------------------------
    protein_dict = {
        "X":            torch.tensor(X, device=device, dtype=torch.float32),
        "mask":         torch.tensor(mask, device=device, dtype=torch.int32),
        "S":            torch.tensor(S_input, device=device, dtype=torch.int32),
        "R_idx":        torch.tensor(R_idx, device=device, dtype=torch.int32),
        "chain_labels": torch.tensor(chain_labels, device=device, dtype=torch.int32),
        "chain_mask":   torch.tensor(chain_mask_arr, device=device, dtype=torch.float32),
        "Y":            torch.tensor(Y, device=device, dtype=torch.float32),
        "Y_t":          torch.tensor(Y_t, device=device, dtype=torch.int32),
        "Y_m":          torch.tensor(Y_m, device=device, dtype=torch.int32),
    }

    # ------------------------------------------------------------------
    # Load LigandMPNN model (context-aware checkpoint).
    # ligandmpnn_v_32_010_25.pt has atom_context_num=25, num_edges=32.
    # Using model_type="ligand_mpnn" causes the model to actually consume the
    # Y/Y_t/Y_m context tensors; "protein_mpnn" would silently ignore them.
    # ------------------------------------------------------------------
    _model_type = "ligand_mpnn"
    checkpoint = torch.load(str(_ckpt_path), map_location=device)
    k_neighbors = checkpoint["num_edges"]
    atom_context_num = checkpoint["atom_context_num"]
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=_model_type,
        ligand_mpnn_use_side_chain_context=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Featurize and sample.
    # ------------------------------------------------------------------
    feature_dict = _featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=atom_context_num,
        model_type=_model_type,
    )
    feature_dict["batch_size"]          = 1
    feature_dict["temperature"]         = float(temperature)
    feature_dict["bias"]                = torch.zeros(
        [1, n_res, 21], device=device, dtype=torch.float32
    )
    feature_dict["symmetry_residues"]   = [[]]
    feature_dict["symmetry_weights"]    = [[]]

    results: list = []
    with torch.no_grad():
        for _ in range(int(num_seqs)):
            feature_dict["randn"] = torch.randn(
                [1, feature_dict["mask"].shape[1]], device=device
            )
            output_dict = model.sample(feature_dict)
            S_sampled = output_dict["S"][0].cpu().numpy()  # (n_res,) int
            # B2: NO force-'A' overwrite. LigandMPNN keeps fixed positions (chain_mask=0) from the
            # real `known_seq` S natively, so the sampled S already carries the declared identity at
            # fixed positions and freshly designed residues elsewhere.
            one_letter_list = [_restype_int_to_str[int(aa)] for aa in S_sampled]
            results.append("".join(one_letter_list).replace("X", "A"))
    return results
