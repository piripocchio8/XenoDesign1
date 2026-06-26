"""Chai-1 backend wrapper.

`write_inputs` (pure, CPU-testable) builds the Chai FASTA. `predict` / `truncated_refine`
run on GPU via chai_lab and are imported lazily so this module loads without chai_lab.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from xenodesign.io_spec import build_fasta


@dataclass
class Prediction:
    """Structure-prediction result extracted from Chai outputs."""
    coords: np.ndarray          # (n_atoms, 3)
    plddt: np.ndarray           # per-residue pLDDT
    iptm: float                 # interface pTM
    token_index: np.ndarray     # per-residue chain index (chain bookkeeping)
    ptm: float = 0.0            # global pTM
    aggregate_score: float = 0.0  # chai's ranking score of the selected model
    has_inter_chain_clashes: bool = False
    _cif_path: "str | None" = None  # path to the selected model's CIF (for downstream geometry reads)


def write_inputs(entities: Sequence[Mapping[str, object]], out_dir: str | Path) -> Path:
    """Write the Chai FASTA for the given entities; return the FASTA path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "input.fasta"
    fasta_path.write_text(build_fasta(entities))
    return fasta_path


def load_prediction(out_dir: str | Path) -> "Prediction":
    """Parse a Chai-1 output directory into a Prediction (CPU-only, no chai_lab).

    Chai writes `scores.model_idx_N.npz` (aggregate_score, ptm, iptm, has_inter_chain_clashes,
    ...) and `pred.model_idx_N.cif` (coords + per-residue pLDDT in the B-factor column) per
    sample. We select the model with the highest `aggregate_score` and extract the contract in
    docs/benchmark-prior-art-map.md §2. This is the offline twin of `_to_prediction`, so the
    GPU `predict` path can be pinned by a CPU test against the committed fixture.
    """
    import re

    import gemmi

    out_dir = Path(out_dir)
    score_files = sorted(out_dir.glob("scores.model_idx_*.npz"))
    if not score_files:
        raise FileNotFoundError(f"no scores.model_idx_*.npz under {out_dir}")

    best_idx, best_agg, best_npz = None, -np.inf, None
    for f in score_files:
        d = np.load(f)
        agg = _first_scalar(d["aggregate_score"])
        if agg > best_agg:
            best_agg = agg
            best_idx = int(re.search(r"idx_(\d+)", f.name).group(1))
            best_npz = d

    iptm = _first_scalar(best_npz["iptm"])
    ptm = _first_scalar(best_npz["ptm"])
    has_clash = bool(np.asarray(best_npz["has_inter_chain_clashes"]).reshape(-1)[0])

    cif_path = out_dir / f"pred.model_idx_{best_idx}.cif"
    structure = gemmi.read_structure(str(cif_path))
    model = structure[0]
    coords = np.array(
        [[a.pos.x, a.pos.y, a.pos.z] for chain in model for res in chain for a in res],
        dtype=float,
    )
    plddt, chain_index = [], []
    for ci, chain in enumerate(model):
        for res in chain:
            ca = res.find_atom("CA", "*")
            plddt.append(ca.b_iso if ca is not None else float(np.mean([a.b_iso for a in res])))
            chain_index.append(ci)

    return Prediction(
        coords=coords,
        plddt=np.asarray(plddt, dtype=float),
        iptm=iptm,
        token_index=np.asarray(chain_index, dtype=int),
        ptm=ptm,
        aggregate_score=best_agg,
        has_inter_chain_clashes=has_clash,
        _cif_path=str(cif_path),
    )


def _save_confidence_npz(candidates, out_dir) -> None:  # pragma: no cover (gpu)
    """Persist per-token pae/pde/plddt per model — chai doesn't auto-save these.

    Mirrors the proven gradio chai_runner idiom; best-effort (warn, don't fail) so a
    StructureCandidates without these attrs never breaks a prediction.
    """
    import warnings

    try:
        pae, pde, plddt = candidates.pae, candidates.pde, candidates.plddt
        for idx in range(_to_numpy(pae).shape[0]):
            np.savez(
                Path(out_dir) / f"confidence.model_idx_{idx}.npz",
                pae=_to_numpy(pae[idx]),
                pde=_to_numpy(pde[idx]),
                plddt=_to_numpy(plddt[idx]),
            )
    except Exception as e:
        warnings.warn(f"could not save confidence npz: {e}")


def per_chain_plddt(prediction: "Prediction") -> dict[int, float]:
    """Mean pLDDT per chain, keyed by chain index (from `token_index` bookkeeping).

    Enables interface / per-chain pLDDT gating (spec §5: protein and peptide pLDDT each
    evaluated separately) without re-parsing the CIF.
    """
    ti = np.asarray(prediction.token_index)
    return {int(c): float(prediction.plddt[ti == c].mean()) for c in np.unique(ti)}


class ChaiBackend:
    """GPU-backed Chai-1 engine. Heavy deps imported lazily inside methods."""

    def __init__(self, device: "str | None" = None, seed: int = 42):
        from xenodesign.config import resolve_device
        # device=None -> resolve (XENO_DEVICE / cuda:0 if available / mps / cpu); an explicit
        # string is honoured verbatim. On a GPU box the resolved default is still cuda:0.
        self.device = device if device else resolve_device()
        self.seed = seed

    def predict(self, entities, out_dir, num_diffn_timesteps: int = 200,
                constraint_path: "str | Path | None" = None,
                msa_directory: "str | Path | None" = None) -> Prediction:  # pragma: no cover (gpu)
        """Run a full Chai-1 forward prediction and return the best candidate.

        Verified on chai_lab 0.6.1 (gradio_design-gradio-design:latest image):
        - run_inference signature: (fasta_file, *, output_dir, use_esm_embeddings,
          num_trunk_recycles, num_diffn_timesteps, num_diffn_samples, seed,
          device: str|None, low_memory, constraint_path, msa_directory, ...)
          → StructureCandidates.
        - device MUST be a str (or None for "cuda:0") in 0.6.1; run_inference converts
          internally via torch.device(). Do NOT pass a torch.device object.
        - output_dir must be empty or non-existent on entry (assertion in run_inference).
          We therefore write the FASTA to out_dir and run chai into out_dir/chai_out.
        - scores.model_idx_*.npz auto-saved with keys: aggregate_score, ptm, iptm,
          per_chain_ptm, per_chain_pair_iptm, has_inter_chain_clashes.
        - confidence.model_idx_*.npz (pae/pde/plddt) NOT auto-saved; persisted below.

        constraint_path (#27): optional Path to a chai .restraints CSV. chai 0.6.1's
        run_inference accepts constraint_path natively; None (default) = unconstrained
        (byte-for-byte the old behaviour).

        msa_directory (#29, P3a): optional Path to a directory of precomputed MSAs
        (chai .aligned.pqt / .a3m), passed straight to run_inference(msa_directory=...).
        Used by the non-α 9DXX case to give the FIXED HA target its precomputed MSA (gate
        #29: MSA-free Chai fails the HA fold; with an MSA it reproduces it near-natively).
        None (default) = MSA-free, unchanged behaviour. When set, ``use_msa_server`` stays
        False so chai reads the local MSAs only (never the network).
        """
        from chai_lab.chai1 import run_inference  # lazy import

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = write_inputs(entities, out_dir)
        # 0.6.1 asserts output_dir is empty; FASTA is in out_dir so use a subdir.
        chai_out = out_dir / "chai_out"
        chai_out.mkdir(parents=True, exist_ok=True)
        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=chai_out,
            device=self.device,  # 0.6.1: str | None; converted internally to torch.device
            seed=self.seed,
            num_diffn_timesteps=num_diffn_timesteps,
            use_esm_embeddings=True,
            use_msa_server=False,
            constraint_path=Path(constraint_path) if constraint_path is not None else None,
            msa_directory=Path(msa_directory) if msa_directory is not None else None,
        )
        # chai auto-writes pred.model_idx_*.cif + scores.model_idx_*.npz to chai_out,
        # but per-token pae/pde/plddt are NOT auto-saved — persist them like the gradio
        # chai_runner so interface/PAE scoring can read them later.
        _save_confidence_npz(candidates, chai_out)
        # Parse on-disk (CPU-tested via load_prediction) from chai_out where chai wrote.
        return load_prediction(chai_out)

    def truncated_refine(self, structure, ref_time_steps, out_dir) -> Prediction:  # pragma: no cover (gpu)
        """Run structure-conditioned ("truncated") diffusion refinement.

        Starts from a prior structure perturbed by sigma[ref_idx] and runs only the
        trailing `ref_time_steps` of diffusion (HalluDesign §2.1). Uses the vendored
        sampler in xenodesign/backends/chai_truncated.py.

        Args:
            structure: dict with keys:
                - "entities": list of entity dicts (same format as predict())
                - "coords": np.ndarray of shape (n_atoms, 3) — the prior structure
            ref_time_steps: number of trailing diffusion steps to run (e.g. 50 of 200)
            out_dir: Path-like; Chai writes pred.model_idx_*.cif + scores here

        Returns:
            Prediction parsed from the refined structure output.
        """
        import torch

        from xenodesign.backends.chai_truncated import run_inference_truncated

        out_dir = Path(out_dir)
        entities = structure["entities"]
        coords = structure["coords"]

        fasta_path = write_inputs(entities, out_dir)
        # Mirror predict(): write FASTA to out_dir, route chai into out_dir/chai_out/
        # so that FASTA and CIF outputs never collide (chai 0.6.1 asserts empty output_dir).
        chai_out = out_dir / "chai_out"
        chai_out.mkdir(parents=True, exist_ok=True)

        # Convert coords to tensor: (n_atoms, 3)
        initial_coords = torch.tensor(
            np.asarray(coords, dtype=np.float32), dtype=torch.float32
        )

        candidates = run_inference_truncated(
            fasta_file=fasta_path,
            output_dir=chai_out,
            device=torch.device(self.device),
            seed=self.seed,
            use_esm_embeddings=True,
            initial_coords=initial_coords,
            ref_time_steps=ref_time_steps,
        )

        _save_confidence_npz(candidates, chai_out)
        return load_prediction(chai_out)

    @staticmethod
    def _to_prediction(candidates) -> Prediction:  # pragma: no cover (gpu)
        """Extract the top-ranked candidate from chai's StructureCandidates into a Prediction.

        BEST-EFFORT — verify on GPU. coords are parsed from the top CIF via gemmi; iptm comes
        from the per-sample ranking (interface_ptm); plddt is the per-token confidence.

        PREFER `load_prediction(out_dir)` (CPU-tested against the committed fixture) once GPU
        wiring lands: chai writes scores/pred files to out_dir, so parsing from disk avoids the
        fragile in-memory attribute access here and gives real per-chain bookkeeping + clash flag.
        """
        import gemmi
        import numpy as np

        # Rank candidates: chai exposes per-sample ranking_data with an aggregate score.
        ranking = list(getattr(candidates, "ranking_data", []) or [])
        if ranking:
            scores = [float(_first_scalar(getattr(r, "aggregate_score", 0.0))) for r in ranking]
            best = int(np.argmax(scores))
        else:
            best = 0

        cif_path = candidates.cif_paths[best]
        structure = gemmi.read_structure(str(cif_path))
        coords = np.array(
            [[a.pos.x, a.pos.y, a.pos.z]
             for model in structure for chain in model for res in chain for a in res],
            dtype=float,
        )

        plddt = np.asarray(_to_numpy(candidates.plddt)[best], dtype=float)

        iptm = 0.0
        if ranking:
            ptm = getattr(ranking[best], "ptm_scores", None)
            if ptm is not None:
                iptm = float(_first_scalar(getattr(ptm, "interface_ptm", 0.0)))

        return Prediction(
            coords=coords,
            plddt=plddt,
            iptm=iptm,
            token_index=np.arange(plddt.shape[0]),
        )


def _to_numpy(x):  # pragma: no cover (gpu)
    """Convert a torch tensor (or array-like) to a numpy array."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _first_scalar(x):  # pragma: no cover (gpu)
    """Coerce a possibly-tensor score to a python float (taking the first element if 1-D)."""
    arr = _to_numpy(x).reshape(-1)
    return float(arr[0]) if arr.size else 0.0
