"""MONDE-T catalog loader for the mixed-chirality ncAA palette (CPU-only).

Reads the committed MONDE-T export (``data/benchmark/mondet/mondet.csv``: columns
``component_id,parent,entity_count,smiles``) and exposes a ranked, de-duplicated view
used to SCOPE the ncAA palette (``--ncaa_dict {d_only,d_common,all}``). The catalog is the
MONDE-T database (Waldherr, Freimann et al. 2025; CC-BY 4.0); see
``data/benchmark/mondet/PROVENANCE.md``.

The CSV has one row PER parent-class, so a given ``component_id`` can repeat: e.g. ``DTR``
carries both a canonical ``TRP`` parent row and a ``D-amino acid`` class row. We de-dup to one
entry per code and KEEP the canonical 3-letter parent (the one that resolves to a real residue
code), discarding the descriptive class labels (``D-amino acid``, ``β-amino acid``, ...).
``entity_count`` is the PDB occurrence frequency, i.e. the commonness ranking.

No GPU / no Chai. Path resolution is repo-relative (no hardcoded absolute paths).
"""
from __future__ import annotations

import csv
import functools
from pathlib import Path
from typing import Optional

from xenodesign.io_spec import AA1_TO_AA3

# Canonical three-letter residue codes (the 20 standard parents). A parent value that is in
# this set is the real chemistry parent; anything else (e.g. "D-amino acid") is a class label.
_CANONICAL_3 = set(AA1_TO_AA3.values())


def default_csv_path() -> Path:
    """Repo-relative path to the committed MONDE-T catalog (no hardcoded /home).

    Resolves relative to this file's location: ``<repo>/xenodesign/mondet.py`` ->
    ``<repo>/data/benchmark/mondet/mondet.csv``.
    """
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "data" / "benchmark" / "mondet" / "mondet.csv"


@functools.lru_cache(maxsize=8)
def _load_ranked(csv_path: str) -> tuple[tuple[str, str, int], ...]:
    """Cached parse: de-dup to one (code, canonical-parent, count) per code, ranked desc."""
    by_code: dict[str, tuple[str, int]] = {}  # code -> (best_parent, count)
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("component_id") or "").strip().upper()
            if not code:
                continue
            parent = (row.get("parent") or "").strip().upper()
            try:
                count = int(row.get("entity_count") or 0)
            except (TypeError, ValueError):
                count = 0
            prev = by_code.get(code)
            # Prefer a canonical 3-letter parent over a class label; keep the max count seen.
            if prev is None:
                by_code[code] = (parent, count)
            else:
                prev_parent, prev_count = prev
                best_parent = prev_parent
                if parent in _CANONICAL_3 and prev_parent not in _CANONICAL_3:
                    best_parent = parent
                by_code[code] = (best_parent, max(prev_count, count))
    ranked = sorted(
        ((code, parent, count) for code, (parent, count) in by_code.items()),
        key=lambda t: t[2],
        reverse=True,
    )
    return tuple(ranked)


def load_mondet(csv_path: str | Path | None = None) -> list[tuple[str, str, int]]:
    """Return ``[(component_id, parent, entity_count)]`` de-duped and ranked by count desc.

    ``csv_path`` defaults to the committed catalog (``default_csv_path()``).
    """
    path = str(Path(csv_path) if csv_path is not None else default_csv_path())
    return list(_load_ranked(path))


def top_ncaa(n: int, csv_path: str | Path | None = None) -> list[str]:
    """The ``n`` most common ncAA ``component_id``s by entity_count (highest first)."""
    return [code for code, _parent, _count in load_mondet(csv_path)[: max(0, int(n))]]


def mondet_parent(code: str, csv_path: str | Path | None = None) -> Optional[str]:
    """Canonical 3-letter parent for a MONDE-T code, or None if the code is not in the catalog."""
    want = (code or "").strip().upper()
    for c, parent, _count in load_mondet(csv_path):
        if c == want:
            return parent or None
    return None
