"""CPU tests for the MONDE-T catalog loader (xenodesign.mondet).

The loader reads the MONDE-T export (``data/benchmark/mondet/mondet.csv``: columns
``component_id,parent,entity_count,smiles``; one row PER parent-class so codes repeat)
and exposes a ranked, de-duplicated view used to scope the mixed-chirality ncAA palette.
``entity_count`` is the PDB occurrence frequency (commonness ranking); ``parent`` is the
canonical 3-letter parent (e.g. DTR->TRP). CPU-only — no GPU / Chai.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from xenodesign import mondet

FIXTURE = Path(__file__).parent / "fixtures" / "mondet_real_sample.csv"


def test_load_ranks_by_entity_count_desc():
    rows = mondet.load_mondet(FIXTURE)
    counts = [c for _code, _parent, c in rows]
    assert counts == sorted(counts, reverse=True)
    # Highest-count row first (MSE at 10214 in the fixture).
    assert rows[0][0] == "MSE"
    assert rows[0][2] == 10214


def test_load_dedupes_repeated_component_ids_to_canonical_3letter_parent():
    rows = mondet.load_mondet(FIXTURE)
    codes = [c for c, _p, _n in rows]
    # DTR/DAL appear twice in the CSV (real + 'D-amino acid' parent rows) -> collapse to one.
    assert codes.count("DTR") == 1
    assert codes.count("DAL") == 1
    parent = {c: p for c, p, _n in rows}
    # The canonical 3-letter parent is kept, NOT the 'D-amino acid' class label.
    assert parent["DTR"] == "TRP"
    assert parent["DAL"] == "ALA"


def test_top_ncaa_returns_n_codes_highest_first():
    top = mondet.top_ncaa(3, csv_path=FIXTURE)
    assert len(top) == 3
    # MSE (10214) is the most common in the fixture.
    assert top[0] == "MSE"


def test_mondet_parent_lookup():
    assert mondet.mondet_parent("DTR", csv_path=FIXTURE) == "TRP"
    assert mondet.mondet_parent("dtr", csv_path=FIXTURE) == "TRP"  # case-insensitive
    assert mondet.mondet_parent("NOSUCH", csv_path=FIXTURE) is None


def test_default_csv_path_resolves_to_repo_root():
    # No hardcoded /home: the default path resolves relative to the repo and exists.
    p = mondet.default_csv_path()
    assert p.exists()
    assert p.name == "mondet.csv"
    # The real catalog: 2562 CSV rows collapse to ~1435 unique component_ids after de-dup.
    rows = mondet.load_mondet()
    assert len(rows) > 1400


def test_real_catalog_top_is_most_common():
    # Sanity on the real catalog: ranking is by frequency, top is a very common entry.
    rows = mondet.load_mondet()
    assert rows[0][2] >= rows[-1][2]
