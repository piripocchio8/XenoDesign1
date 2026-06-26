"""BinderClass hook contract + SeedSpec + CLASS_REGISTRY.

This module defines the ONLY surface the dispatcher (``xenodesign/dispatch.py``,
landed in T3) knows about. Each binder class (alpha / non_alpha / cyclic) supplies
a set of injected callables — seed / ss_bias / restraints / closure / accept_fns /
objective / referee / report / seq_update — that ``run_design`` wires into the
untouched :class:`xenodesign.loop.HalluLoop`. The loop control flow never changes;
only *which* callables are injected differs per class.

T2 ships the Protocol + ``SeedSpec`` + a ``CLASS_REGISTRY`` populated with no-op
*stub* classes so the CPU suite stays green before the real class bodies exist.
T5/T6/T7 each REPLACE one stub by re-pointing its registry entry at the real class
(``from xenodesign.classes.alpha import Alpha`` etc.).

The ``binder_class`` CLI axis uses underscores (``non_alpha``); the benchmark case
registry key (``cases.py``) has no underscore (``nonalpha``). The mapping lives here:
``CLASS_REGISTRY['non_alpha'].case_id == 'nonalpha'``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

__all__ = ["SeedSpec", "BinderClass", "CLASS_REGISTRY"]


@dataclass
class SeedSpec:
    """Initial binder sequence + per-residue design hints handed to the loop.

    Attributes:
        one_letter: The seed binder sequence as one-letter L codes.
        fixed_chirality: Map of 1-based position -> 'D' | 'L' for residues whose
            chirality is pinned (empty by default; alpha pins none, cyclic may).
        cys_positions: 1-based Cys positions for ICK disulfide topology
            (non_alpha); ``()`` for classes without an ICK scaffold.
    """

    one_letter: str
    fixed_chirality: dict = field(default_factory=dict)   # {1-based pos: 'D'|'L'}
    cys_positions: tuple = ()                             # non_alpha ICK; else ()


@runtime_checkable
class BinderClass(Protocol):
    """The per-class hook contract the dispatcher wires into ``HalluLoop``.

    Every method is an injected seam: the dispatcher calls them to assemble the
    loop for a given ``DesignConfig`` without knowing the binder chemistry.
    """

    case_id: str

    def seed(self, cfg, target_seq) -> SeedSpec: ...
    def ss_bias(self, cfg, case): ...
    def restraints(self, cfg, case, out_dir, target_ctx) -> "Path | None": ...
    def closure(self, cfg, seed_spec) -> list: ...
    def seq_update(self, cfg, wrapper, seed_spec, roles=None) -> Callable: ...
    def accept_fns(self, cfg) -> "Callable | None": ...
    def objective(self, cfg, wrapper) -> Callable: ...
    def referee(self, cfg, loop_dir, esm_judge, roles=None): ...
    def report(self, cfg, history, panel_result, case, out_dir,
               *, l_seed_iptm: float = 0.0, wall_time_s: float = 0.0) -> dict: ...


class _Stub:
    """No-op reference implementation: defines every hook, raising on use.

    Satisfies the ``BinderClass`` method-presence contract so the registry is
    usable (and testable) before T5/T6/T7 land the real class bodies. Each hook
    raises ``NotImplementedError`` rather than silently no-op'ing, so any code
    that actually *invokes* a stub fails loudly.
    """

    case_id = ""

    def seed(self, *a): raise NotImplementedError
    def ss_bias(self, *a): raise NotImplementedError
    def restraints(self, *a): raise NotImplementedError
    def closure(self, *a): raise NotImplementedError
    def seq_update(self, *a): raise NotImplementedError
    def accept_fns(self, *a): raise NotImplementedError
    def objective(self, *a): raise NotImplementedError
    def referee(self, *a): raise NotImplementedError
    def report(self, *a, **k): raise NotImplementedError


# T6: the real NonAlpha class lives in xenodesign.classes.non_alpha (real loop, promoted from
# the predict-only stub). CLI axis 'non_alpha' -> benchmark case_id 'nonalpha'.
from xenodesign.classes.non_alpha import NonAlpha  # noqa: E402

# T5: the real Alpha class lives in xenodesign.classes.alpha (migrated, behaviour-preserving).
from xenodesign.classes.alpha import Alpha  # noqa: E402
# T7: the real Cyclic class lives in xenodesign.classes.cyclic and is re-pointed below.
from xenodesign.classes.cyclic import Cyclic  # noqa: E402


# CLI binder_class axis -> class instance. T5/T6/T7 replace each stub with the
# real class (re-pointing the import target); the keys stay fixed.
CLASS_REGISTRY: dict = {
    "alpha": Alpha(),
    "non_alpha": NonAlpha(),
    "cyclic": Cyclic(),
}
