"""Shared skip logic for GPU / network test packages.

These helpers let `pytest -m gpu` / `-m network` run gracefully on machines that lack a
GPU or the heavy optional dependencies (chai_lab, transformers, torch): the test skips
with a clear reason instead of erroring at collection time.
"""
from __future__ import annotations

import importlib.util

import pytest


def require_cuda():
    """Skip unless torch + a visible CUDA device are available."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")


def require_chai():
    """Skip unless chai_lab is importable."""
    if importlib.util.find_spec("chai_lab") is None:
        pytest.skip("chai_lab not installed (pip install chai_lab)")


def require_transformers():
    """Skip unless transformers + torch are importable."""
    for mod in ("torch", "transformers"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} not installed (pip install transformers torch)")
