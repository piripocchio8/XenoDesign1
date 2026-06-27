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
    canonical_exclusion_set,
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
    # ...plus the top-2 TRUE ncAA by entity_count, EXCLUDING the canonical L+D codes.
    # Canonicals (DAL 395, DTR 121) are NOT counted toward top_x -> top-2 true ncAA = MSE, AIB.
    assert "MSE" in s
    assert "AIB" in s
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


# --- bug fix: top_x must rank only TRUE ncAA (exclude the 40 canonical L+D codes) ---

def test_canonical_exclusion_set_covers_all_l_and_d_canonicals():
    """The exclusion set = the 20 standard L codes + every D-canonical code."""
    from xenodesign.io_spec import AA1_TO_AA3

    excl = canonical_exclusion_set()
    # All 20 standard L three-letter codes are excluded...
    assert set(AA1_TO_AA3.values()) <= excl
    # ...and every D-canonical code is excluded.
    assert set(D_CANONICAL) <= excl
    # No true ncAA leaks into the exclusion set.
    for code in ("MSE", "AIB", "NLE", "ORN", "HYP", "RAR"):
        assert code not in excl


def test_d_common_top_x_counts_only_true_ncaa_not_d_canonicals():
    """top_x=3 -> all D-canonicals ALWAYS in, PLUS exactly 3 TRUE ncAA, none of them canonical."""
    pal = build_palette("d_common", ncaa_top_x=3, csv_path=FIXTURE)
    s = set(pal)
    # Every D-canonical is always-in regardless of top_x.
    assert set(D_CANONICAL) <= s
    # The non-canonical additions: whatever is in the palette beyond the D-canonical set.
    extras = s - set(D_CANONICAL)
    # Exactly 3 true ncAA were added (top_x=3); D-canonicals do NOT consume the budget.
    assert len(extras) == 3
    # None of the 3 extras is a canonical (L or D) code.
    excl = canonical_exclusion_set()
    assert not (extras & excl)
    # Concretely, the top-3 true ncAA by entity_count in the fixture: MSE, AIB, NLE.
    assert extras == {"MSE", "AIB", "NLE"}
    # And DTR (a D-canonical, count 121) is present as always-in, NOT as one of the 3.
    assert "DTR" in s
    assert "DTR" not in extras


def test_d_common_includes_highest_frequency_true_ncaa():
    """The most common TRUE ncAA (MSE, 10214) is present even at small top_x."""
    pal = build_palette("d_common", ncaa_top_x=1, csv_path=FIXTURE)
    s = set(pal)
    assert "MSE" in s
    assert (s - set(D_CANONICAL)) == {"MSE"}


def test_all_includes_low_freq_true_ncaa_absent_from_small_d_common():
    """A low-frequency TRUE ncAA (RAR, 2) is in `all` but not in a small-N d_common."""
    common = set(build_palette("d_common", ncaa_top_x=2, csv_path=FIXTURE))
    allp = set(build_palette("all", ncaa_top_x=2, csv_path=FIXTURE))
    assert "RAR" not in common
    assert "RAR" in allp


def test_all_excludes_no_true_ncaa_but_keeps_d_canonicals():
    """`all` = D-canonicals (always-in) + every TRUE ncAA; canonicals not double-ranked."""
    allp = set(build_palette("all", ncaa_top_x=2, csv_path=FIXTURE))
    assert set(D_CANONICAL) <= allp
    # Fixture true ncAA: MSE, AIB, NLE, RAR -> all present.
    assert {"MSE", "AIB", "NLE", "RAR"} <= allp
