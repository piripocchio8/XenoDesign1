"""S2.2 intentional-improvement pin: ABC Variant-A routed through SequenceUpdate feeds the REAL
evolving identity as known_seq (not the legacy all-Ala design_codes). Pins the invariant-#1 fix for
the ABC identity-fill path (spec §1: abc/variants.py:104-106 all-Ala starvation)."""
from __future__ import annotations

import numpy as np

from xenodesign.abc.variants import abc_variant_a_design_fn


def test_routed_variant_a_known_seq_is_real_identity(monkeypatch):
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    seen = {}

    def backend(design_backbone, context_coords, context_elements,
                fixed_mask, temperature, num_seqs, known_seq=None):
        seen["known_seq"] = known_seq
        n = np.asarray(design_backbone).shape[0]
        return [(known_seq or "A" * n)[:n].ljust(n, "A") for _ in range(num_seqs)]

    fn = abc_variant_a_design_fn(backend, roles=None, frozen=set())
    fn("WYFMK", {0: "D", 1: "L", 2: "D", 3: "L", 4: "D"})
    assert seen["known_seq"] == "WYFMK"              # real identity, not all-Ala
    assert "A" * 5 != seen["known_seq"]
