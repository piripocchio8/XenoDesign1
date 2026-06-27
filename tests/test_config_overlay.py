"""MOD-5: dotted-key override and nested-dict overlay must be equivalent (one recursive setter)."""
import copy

from xenodesign import config


def test_config_overlay_and_dotted_equivalence():
    base = config.PRESETS["alpha"]

    via_overlay = copy.deepcopy(base)
    config._overlay(via_overlay, {"loop": {"iters": 9}})

    via_dotted = copy.deepcopy(base)
    config._apply_dotted(via_dotted, "loop.iters", 9)

    assert via_overlay.loop.iters == 9
    assert via_dotted.loop.iters == 9
    assert via_overlay == via_dotted

    # also equivalent for a top-level (non-nested) key
    o2 = copy.deepcopy(base)
    config._overlay(o2, {"binder_length": 17})
    d2 = copy.deepcopy(base)
    config._apply_dotted(d2, "binder_length", 17)
    assert o2 == d2
    assert o2.binder_length == 17
