import json
import pytest
from xenodesign.config import load_config, LoopConfig
from xenodesign.config import (
    DesignConfig, TargetSpec, RestraintConfig, LoopKnobs, GateConfig,
    PRESETS, resolve_config, dump_config,
)


def test_load_config_defaults(tmp_path):
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps({}))
    cfg = load_config(cfg_path)
    assert isinstance(cfg, LoopConfig)
    assert cfg.engine == "chai"
    assert cfg.ref_time_steps == 50
    assert cfg.iterations == 30
    assert cfg.seed_source == "pepmlm_retroinverso"
    assert cfg.accept == "greedy"
    assert cfg.select_topk_frac == 0.1


def test_load_config_overrides(tmp_path):
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps({
        "ref_time_steps": 150, "iterations": 7, "seed_source": "random",
        "select_topk_frac": 0.2,
    }))
    cfg = load_config(cfg_path)
    assert cfg.ref_time_steps == 150
    assert cfg.iterations == 7
    assert cfg.seed_source == "random"
    assert cfg.select_topk_frac == 0.2


def test_load_config_rejects_unknown_accept(tmp_path):
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps({"accept": "annealing"}))
    with pytest.raises(ValueError, match="accept"):
        load_config(cfg_path)


def test_presets_mirror_per_class_defaults():
    a = PRESETS["alpha"]
    assert a.binder_class == "alpha"
    assert a.objective == "iptm"
    assert a.restraint.kind == "pin_polarity"
    assert a.gates.entropy is True
    c = PRESETS["cyclic"]
    assert c.target.target_type == "metal"
    assert c.restraint.kind == "metal_coordination"
    assert c.gates.chirality is False
    n = PRESETS["non_alpha"]
    assert n.binder_class == "non_alpha"
    assert n.target.target_type == "protein"
    assert n.target.msa is True


def test_resolve_precedence_preset_then_file_then_cli(tmp_path):
    cfgfile = tmp_path / "ov.json"
    cfgfile.write_text(json.dumps({"loop": {"iters": 11}, "seed": 7}))
    cfg = resolve_config("alpha", target_type="protein",
                         config_file=str(cfgfile),
                         cli_overrides={"seed": 99, "loop.num_seqs": 4})
    assert cfg.binder_class == "alpha"
    assert cfg.loop.iters == 11       # from file (preset default differs)
    assert cfg.seed == 99             # CLI beats file
    assert cfg.loop.num_seqs == 4     # CLI dotted-path override
    assert cfg.target.target_type == "protein"


def test_resolve_unknown_binder_class_lists_keys():
    with pytest.raises(KeyError) as ei:
        resolve_config("banana", target_type="protein")
    assert "alpha" in str(ei.value) and "non_alpha" in str(ei.value)


def test_mixed_chirality_default_none_for_homochiral():
    # alpha / non_alpha default to homochiral greedy/beam: mixed_chirality stays "none".
    assert resolve_config("alpha", target_type="protein").mixed_chirality == "none"
    assert resolve_config("non_alpha", target_type="protein").mixed_chirality == "none"


def test_cyclic_preset_defaults_mixed_chirality_A():
    # cyclic is mixed-chirality by definition -> default Variant A, for EVERY target chemistry.
    assert PRESETS["cyclic"].mixed_chirality == "A"
    assert resolve_config("cyclic", target_type="metal").mixed_chirality == "A"
    assert resolve_config("cyclic", target_type="none").mixed_chirality == "A"


def test_mixed_chirality_validates_allowed_values():
    cfg = resolve_config("cyclic", target_type="none",
                         cli_overrides={"mixed_chirality": "B"})
    assert cfg.mixed_chirality == "B"
    with pytest.raises(ValueError, match="mixed_chirality"):
        resolve_config("cyclic", target_type="none",
                       cli_overrides={"mixed_chirality": "bogus"})


def test_loop_search_still_greedy_default_choices_unchanged():
    # The launcher axis is untouched: greedy default, beam still allowed.
    cfg = resolve_config("alpha", target_type="protein")
    assert cfg.loop.search == "greedy"
    cfg2 = resolve_config("alpha", target_type="protein",
                          cli_overrides={"loop.search": "beam"})
    assert cfg2.loop.search == "beam"


def test_dump_config_roundtrips(tmp_path):
    cfg = resolve_config("alpha", target_type="protein", out_dir=str(tmp_path))
    p = dump_config(cfg, tmp_path)
    assert p.name == "resolved_config.json"
    data = json.loads(p.read_text())
    assert data["binder_class"] == "alpha"
    assert data["loop"]["iters"] == cfg.loop.iters
