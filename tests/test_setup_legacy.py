"""MOD-6: README quick-start must not instruct the legacy boltz_design conda setup."""
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def test_no_boltz_design_setup_in_readme_path():
    text = README.read_text()
    # No instruction to create/activate a boltz_design conda env in the quick-start path.
    assert not re.search(r"conda\s+(create|activate)[^\n]*boltz_design", text), \
        "README must not instruct the legacy boltz_design conda setup"
    assert "boltz_design" not in text, "README must not reference the legacy boltz_design env"
