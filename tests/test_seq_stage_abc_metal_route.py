"""S3a.5a: make_abc_fitness, when given coord_restraint_rows, writes coordination + closure rows per
eval (the metal coordination ABC currently lacks). Absent -> the legacy closure-only file."""
from __future__ import annotations

from pathlib import Path

from xenodesign.abc.fitness import make_abc_fitness


class _FakeBackend:
    def predict(self, entities, out_dir, num_diffn_timesteps=15, constraint_path=None):
        # Capture the restraint file content for assertion; return a minimal prediction.
        _FakeBackend.last_constraint = Path(constraint_path).read_text() if constraint_path else None
        return type("P", (), {"ptm": 0.5, "_cif_path": None})()


def test_fitness_with_coord_rows_writes_coordination(tmp_path):
    coord_rows = ["A,H6@ND1,A,@ZN,covalent,1.0,0.0,0.0,His-metal,metal_coord_6"]
    fit = make_abc_fitness(_FakeBackend(), out_root=tmp_path, coord_restraint_rows=coord_rows)
    fit("HHHHHH", {i: "D" for i in range(6)})
    assert "H6@ND1" in _FakeBackend.last_constraint and "covalent" in _FakeBackend.last_constraint


def test_fitness_without_coord_rows_is_closure_only(tmp_path):
    fit = make_abc_fitness(_FakeBackend(), out_root=tmp_path)
    fit("HHHHHH", {i: "D" for i in range(6)})
    assert "H6@ND1" not in (_FakeBackend.last_constraint or "")
