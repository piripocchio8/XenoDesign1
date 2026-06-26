"""Parse the MONDE-T ncAA CSV export and select Chai-supported D-amino-acid codes."""
from __future__ import annotations

import csv
from pathlib import Path

from xenodesign.mirror import D_TO_L


def load_mondet(csv_path: str | Path) -> list[dict[str, str]]:
    """Load a MONDE-T CSV export into a list of row dicts."""
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def chai_supported_d_codes(csv_path: str | Path) -> set[str]:
    """Return the set of CCD component IDs in the CSV that are Chai D-amino acids."""
    rows = load_mondet(csv_path)
    return {r["component_id"] for r in rows if r["component_id"] in D_TO_L}


from xenodesign.io_spec import AA1_TO_AA3  # noqa: E402 (append-style import)

_CANONICAL_3 = set(AA1_TO_AA3.values())


def ncaa_codes(csv_path: str | Path) -> set[str]:
    """All non-canonical component IDs in the MONDE-T CSV (any ncAA, incl. D and PTMs)."""
    rows = load_mondet(csv_path)
    return {r["component_id"] for r in rows if r["component_id"].upper() not in _CANONICAL_3}
