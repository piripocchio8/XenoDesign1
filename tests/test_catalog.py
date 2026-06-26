from pathlib import Path
from xenodesign.catalog import load_mondet, chai_supported_d_codes

FIXTURE = Path(__file__).parent / "fixtures" / "mondet_sample.csv"


def test_load_mondet_returns_all_rows():
    rows = load_mondet(FIXTURE)
    assert len(rows) == 6
    assert rows[0]["component_id"] == "DAL"


def test_chai_supported_d_codes_filters_to_known_d_partners():
    codes = chai_supported_d_codes(FIXTURE)
    # Only D codes present in chai's D_partners survive; HIP/MSE/XXX dropped.
    assert codes == {"DAL", "DSN", "DTY"}


from xenodesign.catalog import ncaa_codes


def test_ncaa_codes_excludes_canonical_and_includes_noncanonical():
    codes = ncaa_codes(FIXTURE)
    # Canonical (none in fixture by 3-letter) excluded; D + other ncAA included.
    assert "DAL" in codes and "HIP" in codes and "MSE" in codes and "XXX" in codes
    # The canonical 20 three-letter codes must never appear.
    assert "ALA" not in codes and "GLY" not in codes
