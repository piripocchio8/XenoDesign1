"""ChainRoles contract: the ONE authoritative binder/target chain assignment.

Two layers under test:
  1. ``ChainRoles.from_entities`` — pure derivation from the assembled target entity list.
  2. ``make_alpha_seq_update_fn``'s ``_extract`` — reads the RIGHT binder chain for all four
     dispatch cases (alpha 'B', non_alpha 3-chain 'C', cyclic metal 'B', no-target 'A'),
     proven by stubbing ``backbone_by_residue_from_cif`` so the returned per-chain length
     WITNESSES which chain letter the extractor actually asked for. A hardcoded "B" would
     read the wrong chain (or none) for the non-'B' cases and the witness would mismatch.
"""
import numpy as np
import pytest

import xenodesign.classes.alpha as alpha_mod
from xenodesign.targets import ChainRoles


# ── 1. ChainRoles.from_entities ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "entities, exp_binder, exp_targets",
    [
        ([{"x": 1}], "B", ("A",)),                       # alpha: 1 target -> binder B
        ([{"x": 1}, {"x": 2}], "C", ("A", "B")),         # non_alpha HA1/HA2 -> binder C
        ([{"x": 1}], "B", ("A",)),                       # cyclic metal: Zn -> binder B
        ([], "A", ()),                                   # no-target cyclic: binder is sole chain A
    ],
)
def test_from_entities_assigns_chains_by_order(entities, exp_binder, exp_targets):
    roles = ChainRoles.from_entities(entities)
    assert roles.binder == exp_binder
    assert roles.targets == exp_targets


def test_from_entities_none_is_empty():
    roles = ChainRoles.from_entities(None)
    assert roles.binder == "A"
    assert roles.targets == ()


def test_context_is_first_target_else_binder():
    assert ChainRoles.from_entities([{"x": 1}]).context == "A"          # target A is context
    assert ChainRoles.from_entities([{}, {}]).context == "A"            # first of two targets
    assert ChainRoles.from_entities([]).context == "A"                  # no-target: binder IS context


def test_chainroles_is_frozen():
    roles = ChainRoles.from_entities([{"x": 1}])
    with pytest.raises(Exception):
        roles.binder = "Z"  # type: ignore[misc]


# ── 2. seq-update extractor reads the chain the contract declares ────────────────

def _make_chain_witness(binder_chain, binder_len, context_chain):
    """A fake backbone_by_residue_from_cif: returns ``binder_len`` residues ONLY for the
    binder chain, a sentinel single residue for the context chain, and [] for everything else.

    A per-residue backbone dict needs N/CA/C arrays for the downstream backbone-array builder.
    """
    def _res():
        return {"N": np.zeros(3), "CA": np.zeros(3), "C": np.zeros(3)}

    seen = {"chains": []}

    def fake(cif_path, chain_name):
        seen["chains"].append(chain_name)
        if chain_name == binder_chain:
            return [_res() for _ in range(binder_len)]
        if chain_name == context_chain:
            return [_res()]
        return []

    return fake, seen


class _FakeWrapper:
    def __init__(self, out_dir):
        self.last_out_dir = out_dir


def _run_extract(monkeypatch, roles, binder_len):
    """Build the seq-update fn for ``roles`` and drive its ``_extract`` once; return the
    design_backbone length the extractor produced + the chains it queried.
    """
    fake_bb, seen = _make_chain_witness(roles.binder, binder_len, roles.context)
    # backbone_by_residue_from_cif is imported INTO the alpha module namespace inside the fn;
    # patch it at its source so the lazy import inside make_alpha_seq_update_fn picks up the fake.
    monkeypatch.setattr("xenodesign.eval.gate_tier0a.backbone_by_residue_from_cif", fake_bb)
    # _best_cif_path is resolved through the design_alpha shim; stub it + the all-atom context.
    monkeypatch.setattr(alpha_mod, "_best_cif_path", lambda out_dir: "/fake/best.cif",
                        raising=False)
    shim = alpha_mod._shim()
    monkeypatch.setattr(shim, "_best_cif_path", lambda out_dir: "/fake/best.cif")
    monkeypatch.setattr(alpha_mod, "_all_atoms_from_chain",
                        lambda cif, chain: (np.zeros((1, 3)), ["C"]))

    captured = {}

    def fake_make_seq_update_fn(updater, extract, emit="one_letter"):
        def _fn(prediction):
            captured["extracted"] = extract(prediction)
            return "G"
        return _fn

    monkeypatch.setattr("xenodesign.sequence_update.make_sequence_update_fn",
                        fake_make_seq_update_fn)
    # MultiCandidate / base backend must not actually run inverse folding.
    monkeypatch.setattr(shim, "_make_base_backend", lambda backend: (lambda *a, **k: "G"))
    monkeypatch.setattr(shim, "_cterm_gly_anchor", lambda base: base)

    fn = alpha_mod.make_alpha_seq_update_fn(_FakeWrapper("/fake/out"), num_seqs=1, roles=roles)
    fn(object())  # drive _extract
    return captured["extracted"], seen["chains"]


def test_extract_alpha_reads_chain_B(monkeypatch):
    roles = ChainRoles.from_entities([{"x": 1}])  # binder B
    extracted, chains = _run_extract(monkeypatch, roles, binder_len=17)
    assert extracted["design_backbone"].shape[0] == 17     # read B's length, not A's
    assert "B" in chains


def test_extract_nonalpha_3chain_reads_chain_C(monkeypatch):
    roles = ChainRoles.from_entities([{"x": 1}, {"x": 2}])  # HA1/HA2 -> binder C
    # binder_len 23 differs from any target length the witness gives -> proves C (not B=HA2) read.
    extracted, chains = _run_extract(monkeypatch, roles, binder_len=23)
    assert extracted["design_backbone"].shape[0] == 23
    assert "C" in chains
    assert roles.binder == "C"


def test_extract_cyclic_metal_reads_chain_B(monkeypatch):
    roles = ChainRoles.from_entities([{"name": "Zn"}])  # Zn=A -> binder B
    extracted, chains = _run_extract(monkeypatch, roles, binder_len=11)
    assert extracted["design_backbone"].shape[0] == 11
    assert "B" in chains


def test_extract_no_target_reads_chain_A(monkeypatch):
    roles = ChainRoles.from_entities([])  # binder is the sole chain A
    extracted, chains = _run_extract(monkeypatch, roles, binder_len=9)
    assert extracted["design_backbone"].shape[0] == 9
    assert "A" in chains
    assert roles.binder == "A"


def test_extract_default_roles_is_alpha_B(monkeypatch):
    """No roles kwarg -> byte-identical alpha behaviour: binder chain 'B'."""
    # Witness keyed to alpha's hardcoded ('B' binder, 'A' context) default.
    fake_bb, seen = _make_chain_witness("B", 13, "A")
    monkeypatch.setattr("xenodesign.eval.gate_tier0a.backbone_by_residue_from_cif", fake_bb)
    shim = alpha_mod._shim()
    monkeypatch.setattr(shim, "_best_cif_path", lambda out_dir: "/fake/best.cif")
    monkeypatch.setattr(alpha_mod, "_all_atoms_from_chain",
                        lambda cif, chain: (np.zeros((1, 3)), ["C"]))
    monkeypatch.setattr(shim, "_make_base_backend", lambda backend: (lambda *a, **k: "G"))
    monkeypatch.setattr(shim, "_cterm_gly_anchor", lambda base: base)

    captured = {}

    def fake_make_seq_update_fn(updater, extract, emit="one_letter"):
        def _fn(prediction):
            captured["extracted"] = extract(prediction)
            return "G"
        return _fn

    monkeypatch.setattr("xenodesign.sequence_update.make_sequence_update_fn",
                        fake_make_seq_update_fn)

    fn = alpha_mod.make_alpha_seq_update_fn(_FakeWrapper("/fake/out"), num_seqs=1)  # no roles
    fn(object())
    assert captured["extracted"]["design_backbone"].shape[0] == 13
    assert "B" in seen["chains"]
