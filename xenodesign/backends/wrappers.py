"""LoopState ↔ ChaiBackend wrappers shared by the design loop.

``_LoopBackendWrapper`` (truncated-refine) and ``_PredictBackendWrapper`` (full
predict, restraint-capable) were historically defined in ``scripts/design_demo.py``
and imported back into the package (``classes/alpha.py``, ``dispatch.py``). They are
pure backend-adapter plumbing, so they live in the package now and
``scripts/design_demo.py`` re-exports them for its CLI (MOD-1).

Behaviour is byte-for-byte the same as the previous ``scripts.design_demo`` defs;
this was a move, not a rewrite.
"""
from __future__ import annotations

from pathlib import Path


def _as_target_list(target_entity):
    """Normalise the target arg to a chain-ordered entity list (the binder is appended LAST).

    Accepts a single dict (legacy single-target callers — α), a list of dicts (multi-chain /
    ligand+metal targets), or ``None``/empty (no-target free-peptide mode). Keeping a single
    dict == ``[dict]`` keeps the α path byte-identical (binder still becomes chain B)."""
    if target_entity is None:
        return []
    if isinstance(target_entity, dict):
        return [target_entity]
    return list(target_entity)


def _build_entities(target_entities, binder_l_seq):
    """[*target chains, binder] — binder is ALWAYS the LAST chain (Chai labels A,B,C… in order),
    so multi-chain protein / DNA-RNA / ligand+metal targets all order correctly and the binder's
    chain letter is ``chr(ord('A') + len(target_entities))``."""
    return [
        *target_entities,
        {"type": "protein", "name": "binder",
         "sequence": binder_l_seq, "chirality": "D"},
    ]


class _LoopBackendWrapper:
    """Translates LoopState API → ChaiBackend.truncated_refine."""

    def __init__(self, chai_backend, target_entity=None, *, msa_directory=None):
        self._backend = chai_backend
        self._target_entities = _as_target_list(target_entity)
        self._msa_directory = msa_directory
        self.last_out_dir: Path | None = None

    def truncated_refine(self, state, ref_time_steps, out_dir):
        from xenodesign.io_spec import d_fasta_to_one_letter

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        binder_l_seq = d_fasta_to_one_letter(state.d_fasta)
        entities = _build_entities(self._target_entities, binder_l_seq)

        self.last_out_dir = out_dir
        return self._backend.truncated_refine(
            structure={"entities": entities, "coords": state.coords},
            ref_time_steps=ref_time_steps,
            out_dir=out_dir,
        )


class _PredictBackendWrapper:
    """Per-iteration structure step that runs a FULL ChaiBackend.predict (not truncated).

    Drop-in ``refine_fn`` for ``HalluLoop`` (signature ``(state, ref_time_steps, out_dir) ->
    Prediction``). Unlike ``_LoopBackendWrapper.truncated_refine``, this re-folds the binder
    sequence from scratch (200-step predict) each iteration: slower, but (a) it preserves
    D-chirality better and (b) it supports ``constraint_path`` — the vendored truncated sampler
    does NOT (see chai_truncated.py TODO #27). Used by the RESTRAINED α run so the pin-polarity
    restraint is honoured on every iteration. ``ref_time_steps`` is ignored (predict is full).

    Like ``_LoopBackendWrapper`` it builds entities=[target, binder] so Chai labels
    TARGET=chain A, BINDER=chain B, and records ``last_out_dir`` for the sequence-update step.
    The binder is emitted with chirality 'D' (same as the truncated path).
    """

    def __init__(self, chai_backend, target_entity=None,
                 constraint_path: "Path | str | None" = None,
                 num_diffn_timesteps: int = 200, *, msa_directory=None):
        self._backend = chai_backend
        self._target_entities = _as_target_list(target_entity)
        self._constraint_path = Path(constraint_path) if constraint_path is not None else None
        self._num_diffn_timesteps = num_diffn_timesteps
        self._msa_directory = msa_directory
        self.last_out_dir: Path | None = None

    def __call__(self, state, ref_time_steps, out_dir):
        from xenodesign.io_spec import d_fasta_to_one_letter

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        binder_l_seq = d_fasta_to_one_letter(state.d_fasta)
        entities = _build_entities(self._target_entities, binder_l_seq)

        self.last_out_dir = out_dir
        return self._backend.predict(
            entities,
            out_dir,
            num_diffn_timesteps=self._num_diffn_timesteps,
            constraint_path=self._constraint_path,
            msa_directory=self._msa_directory,
        )
