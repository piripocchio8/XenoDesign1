"""CPU tests for the --ncaa_dict palette-scope feature (extends track #2).

``--ncaa_dict {d_only,d_common,all}`` chooses the SCOPE of the mixed-chirality ncAA palette,
derived from the MONDE-T catalog instead of the old hardcoded 4-code default:
  - d_only  -> the canonical D-amino-acid CCD set (mixed-chirality, no exotic ncAA).
  - d_common-> D-canonical + top-``ncaa_top_x`` MONDE-T ncAA by entity_count.
  - all     -> D-canonical + ALL MONDE-T ncAA (no cap; count logged).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from xenodesign.abc.ncaa import (
    D_CANONICAL,
    build_palette,
    validate_palette,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mondet_real_sample.csv"


def test_d_canonical_set_matches_spec():
    expected = {
        "DAL", "DAR", "DSG", "DAS", "DCY", "DGN", "DGL", "DHI", "DIL", "DLE",
        "DLY", "MED", "DPN", "DPR", "DSN", "DTH", "DTR", "DTY", "DVA",
    }
    assert set(D_CANONICAL) == expected
    assert len(D_CANONICAL) == 19  # 19 chiral canonicals (Gly is achiral)


def test_d_only_palette_is_the_d_canonical_set():
    pal = build_palette("d_only", ncaa_top_x=20, csv_path=FIXTURE)
    assert set(pal) == set(D_CANONICAL)


def test_d_common_is_d_canonical_union_top_n():
    pal = build_palette("d_common", ncaa_top_x=2, csv_path=FIXTURE)
    s = set(pal)
    # Must contain every D-canonical code...
    assert set(D_CANONICAL) <= s
    # ...plus the top-2 MONDE-T codes by entity_count (MSE 10214, DAL 395).
    assert "MSE" in s
    # And must NOT contain the rare low-frequency code (RAR, count 2).
    assert "RAR" not in s


def test_all_includes_low_frequency_code_that_d_common_excludes():
    common = set(build_palette("d_common", ncaa_top_x=2, csv_path=FIXTURE))
    allp = set(build_palette("all", ncaa_top_x=2, csv_path=FIXTURE))
    assert "RAR" not in common
    assert "RAR" in allp  # 'all' = no cap, so the long tail is included


def test_invalid_ncaa_dict_value_errors():
    with pytest.raises(ValueError):
        build_palette("bogus", ncaa_top_x=20, csv_path=FIXTURE)


def test_validate_palette_accepts_mondet_code_via_parent():
    # DAL is a canonical D code: NOT in CONFORMATIONAL_PROXY, but resolvable via MONDE-T parent.
    assert validate_palette(["DAL"], csv_path=FIXTURE) == ["DAL"]
    # A genuinely unknown code is still dropped.
    assert validate_palette(["ZZZ"], csv_path=FIXTURE) == []


def test_validate_palette_still_accepts_conformational_proxy_codes():
    # Back-compat: the old proxy-table path still validates with no csv needed.
    assert validate_palette(["AIB", "ORN", "NLE", "HYP"]) == ["AIB", "ORN", "NLE", "HYP"]
