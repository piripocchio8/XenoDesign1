"""S2.2: ABC Variant-A routed through SequenceUpdate (XENO_SEQ_STAGE=1) feeds the REAL evolving
identity as known_seq (not all-Ala design_codes) and threads real context from last_structure;
flag off keeps the legacy all-Ala/empty-context body byte-identical."""
from __future__ import annotations

import numpy as np

from xenodesign.abc.variants import abc_variant_a_design_fn


def _capturing_backend(seen):
    def backend(design_backbone, context_coords, context_elements,
                fixed_mask, temperature, num_seqs, known_seq=None):
        seen["known_seq"] = known_seq
        seen["n_ctx"] = np.asarray(context_coords).shape[0]
        n = np.asarray(design_backbone).shape[0]
        return [(known_seq or "A" * n)[:n].ljust(n, "A") for _ in range(num_seqs)]
    return backend


def test_variant_a_legacy_uses_all_ala_known_seq(monkeypatch):
    """Flag OFF (default): known_seq is the L-projection of all-Ala design_codes (legacy)."""
    monkeypatch.delenv("XENO_SEQ_STAGE", raising=False)
    seen = {}
    fn = abc_variant_a_design_fn(_capturing_backend(seen))
    pattern = {0: "D", 1: "L", 2: "D", 3: "L"}
    out = fn("MKWV", pattern)
    assert seen["known_seq"] == "AAAA"               # legacy all-Ala (identity discarded)
    assert seen["n_ctx"] == 0                         # legacy empty context
    assert isinstance(out, str) and len(out) == 4


def test_variant_a_routed_uses_real_identity(monkeypatch):
    """Flag ON: known_seq is the REAL identity; non-frozen positions carry it (invariant #1)."""
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    seen = {}
    fn = abc_variant_a_design_fn(_capturing_backend(seen), roles=None, frozen=set())
    pattern = {0: "D", 1: "L", 2: "D", 3: "L"}
    out = fn("MKWV", pattern)
    assert seen["known_seq"] == "MKWV"               # REAL evolving identity (invariant #1)
    assert "A" * 4 != seen["known_seq"]
    assert isinstance(out, str) and len(out) == 4
