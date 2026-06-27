"""Single resolver for ifrit-local reference data used by the Tier-3 analysis scripts.

The diagnostic/analysis scripts under ``scripts/`` read fold/structure reference data
that is not committed to the repo and lives only on the dev box. Route every such
access through :func:`local_ref` so the base directory is configurable via the
``XENO_LOCAL_REF`` env var and a missing checkout raises one clear error.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = "./XenoDesign1_local_ref"


class LocalRefMissing(FileNotFoundError):
    """Raised when the XENO_LOCAL_REF base directory is absent."""


def local_ref(*parts: str) -> Path:
    """Return ``$XENO_LOCAL_REF`` (default ``./XenoDesign1_local_ref``) joined with ``parts``.

    Raises :class:`LocalRefMissing` if the base directory does not exist.
    """
    base = Path(os.environ.get("XENO_LOCAL_REF", _DEFAULT))
    if not base.is_dir():
        raise LocalRefMissing(
            f"local reference data dir not found: {base} "
            f"(set the XENO_LOCAL_REF env var to your local-ref checkout)"
        )
    return base.joinpath(*parts)
