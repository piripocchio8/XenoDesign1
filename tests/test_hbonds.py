# tests/test_hbonds.py
"""CPU tests for the reference-free inter-chain H-bond feature in scripts/score_complex.py.

interchain_hbonds() counts cross-chain donor->acceptor heavy-atom pairs within 3.5 A (distance-only
when no H is modeled; with a loose donor-H..acceptor angle filter when H is present). It is
register-SENSITIVE (a correct register lines up specific donor-acceptor pairs that a translation
breaks) and reference-free (no ground-truth structure). These tests pin: (a) a known H-bond pair is
counted, (b) translating one chain out of range drops the count to 0, (c) the panel keys exist, and
(d) the explicit-H angle filter accepts a well-oriented donor and rejects a back-pointing one.
"""
import tempfile
from pathlib import Path

import gemmi
import pytest

from scripts.score_complex import interchain_hbonds, reexam, structural


def _write(st):
    f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    f.close()
    st.make_mmcif_document().write_file(f.name)
    return Path(f.name)


def _ser_asp(dx=0.0, h=None):
    """SER (chain A) + ASP (chain B); ASP OD1 ~2.6 A from SER OG -> one donor->acceptor pair.
    dx translates the ASP along x (dx>=6 breaks the contact). h optionally adds an explicit OG H."""
    st = gemmi.Structure()
    st.add_model(gemmi.Model("1"))
    ca = gemmi.Chain("A")
    r = gemmi.Residue()
    r.name = "SER"
    r.seqid = gemmi.SeqId("1")
    for nm, el, xyz in [("N", "N", (0.0, 0.0, 0.0)), ("CA", "C", (1.5, 0.0, 0.0)),
                        ("C", "C", (2.0, 1.4, 0.0)), ("O", "O", (1.2, 2.3, 0.0)),
                        ("CB", "C", (2.0, -1.0, 1.0)), ("OG", "O", (3.0, -1.5, 1.5))]:
        a = gemmi.Atom()
        a.name = nm
        a.element = gemmi.Element(el)
        a.pos = gemmi.Position(*xyz)
        r.add_atom(a)
    if h is not None:
        a = gemmi.Atom()
        a.name = "HG"
        a.element = gemmi.Element("H")
        a.pos = gemmi.Position(*h)
        r.add_atom(a)
    ca.add_residue(r)
    cb = gemmi.Chain("B")
    r2 = gemmi.Residue()
    r2.name = "ASP"
    r2.seqid = gemmi.SeqId("1")
    for nm, el, xyz in [("N", "N", (7.0, 0.0, 0.0)), ("CA", "C", (6.5, 0.0, 0.0)),
                        ("C", "C", (6.0, 1.4, 0.0)), ("O", "O", (6.5, 2.3, 0.0)),
                        ("CB", "C", (6.0, -1.0, 1.0)), ("CG", "C", (5.8, -1.4, 1.3)),
                        ("OD1", "O", (5.6, -1.7, 1.6)), ("OD2", "O", (5.5, -1.4, 2.6))]:
        a = gemmi.Atom()
        a.name = nm
        a.element = gemmi.Element(el)
        a.pos = gemmi.Position(xyz[0] + dx, xyz[1], xyz[2])
        r2.add_atom(a)
    cb.add_residue(r2)
    st[0].add_chain(ca)
    st[0].add_chain(cb)
    return _write(st)


def test_known_pair_is_counted():
    cif = _ser_asp(dx=0.0)
    hb = interchain_hbonds(cif, "A", "B")
    cif.unlink(missing_ok=True)
    assert hb["n_interchain_hbonds"] >= 1
    assert hb["hbond_density"] is not None and hb["hbond_density"] > 0
    assert hb["hbond_angle_filtered"] is False  # no explicit H -> distance-only


def test_translation_breaks_hbonds():
    real = _ser_asp(dx=0.0)
    shift = _ser_asp(dx=6.0)
    hr = interchain_hbonds(real, "A", "B")
    hs = interchain_hbonds(shift, "A", "B")
    real.unlink(missing_ok=True)
    shift.unlink(missing_ok=True)
    assert hs["n_interchain_hbonds"] == 0
    assert hr["n_interchain_hbonds"] > hs["n_interchain_hbonds"]
    assert (hr["hbond_density"] or 0) > (hs["hbond_density"] or 0)


def test_panel_keys_present_in_structural_and_reexam():
    cif = _ser_asp(dx=0.0)
    rx = reexam(cif, "A", "B")
    cif.unlink(missing_ok=True)
    for k in ("n_interchain_hbonds", "hbond_density", "hbond_angle_filtered", "sc_normal_opp"):
        assert k in rx


def test_structural_panel_carries_hbond_keys():
    # structural() needs freesasa for bsa_A2 (may be absent on host) but must always emit the H-bond
    # keys regardless of whether SASA succeeds.
    cif = _ser_asp(dx=0.0)
    res = structural(cif, "A", "B")
    cif.unlink(missing_ok=True)
    for k in ("n_interchain_hbonds", "hbond_density", "hbond_angle_filtered"):
        assert k in res


def test_angle_filter_accepts_good_and_rejects_back_pointing():
    # H between OG and the acceptor -> good angle (accept). H on the far side -> bad angle (reject).
    good = _ser_asp(dx=0.0, h=(3.9, -1.6, 1.55))
    bad = _ser_asp(dx=0.0, h=(2.1, -1.4, 1.45))
    hg = interchain_hbonds(good, "A", "B")
    hb = interchain_hbonds(bad, "A", "B")
    good.unlink(missing_ok=True)
    bad.unlink(missing_ok=True)
    assert hg["hbond_angle_filtered"] is True
    assert hb["hbond_angle_filtered"] is True
    assert hg["n_interchain_hbonds"] >= 1
    assert hb["n_interchain_hbonds"] == 0
