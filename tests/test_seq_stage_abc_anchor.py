"""S2.3: the ABC fitness emit applies SequenceUpdate.ensure_canonical_anchor (invariant #3) so an
all-D chain gets a C-terminal Gly anchor (no all-D Chai-tokenization crash) for BOTH variants. An
L-bearing chain is unchanged. Flag off keeps the legacy emit byte-identical."""
from __future__ import annotations

from xenodesign.abc.fitness import make_abc_fitness


class _CaptureBackend:
    """Records the d_fasta the fitness emits into the predict entities."""
    def __init__(self):
        self.seen = None

    def predict(self, entities, out_dir, num_diffn_timesteps=None, constraint_path=None):
        self.seen = entities[0]["sequence"]
        class _P:
            ptm = 0.5
            _cif_path = None
        return _P()


def test_fitness_anchors_all_d_chain_when_routed(monkeypatch, tmp_path):
    monkeypatch.setenv("XENO_SEQ_STAGE", "1")
    be = _CaptureBackend()
    fitness = make_abc_fitness(be, k_star=10, closure=False, out_root=tmp_path)
    # all-D pattern over a chain with NO Gly -> the anchor must add a C-terminal Gly.
    fitness("ACDE", {0: "D", 1: "D", 2: "D", 3: "D"})
    assert be.seen == "(DAL)(DCY)(DAS)G"             # last position anchored to Gly, rest D-CCD


def test_fitness_legacy_no_anchor(monkeypatch, tmp_path):
    monkeypatch.delenv("XENO_SEQ_STAGE", raising=False)
    be = _CaptureBackend()
    fitness = make_abc_fitness(be, k_star=10, closure=False, out_root=tmp_path)
    fitness("ACDE", {0: "D", 1: "D", 2: "D", 3: "D"})
    assert be.seen == "(DAL)(DCY)(DAS)(DGL)"         # legacy: all-D, no anchor (the crash source)
