"""HW-6: scripts/_local_ref.py single XENO_LOCAL_REF resolver with friendly missing-data error."""
import importlib
from pathlib import Path

import pytest

local_ref_mod = importlib.import_module("scripts._local_ref")


def test_local_ref_resolver(tmp_path, monkeypatch):
    base = tmp_path / "ref_root"
    base.mkdir()
    monkeypatch.setenv("XENO_LOCAL_REF", str(base))

    # joins env base with parts
    got = local_ref_mod.local_ref("sub", "file.txt")
    assert got == base / "sub" / "file.txt"

    # base alone (no parts)
    assert local_ref_mod.local_ref() == base

    # absent base dir -> friendly LocalRefMissing naming the env var
    monkeypatch.setenv("XENO_LOCAL_REF", str(tmp_path / "does_not_exist"))
    with pytest.raises(local_ref_mod.LocalRefMissing) as exc:
        local_ref_mod.local_ref("anything")
    assert "XENO_LOCAL_REF" in str(exc.value)
