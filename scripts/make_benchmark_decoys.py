"""Register-shift DECOY generator for the REAL benchmark systems 8GQP, 7YH8, 9DXX (CPU, no GPU).

Negative-design control (ADR-011, mirrors scripts/contrastive_decoys.py): for each benchmark binder we
emit the real complex (shift0) plus binder-register-shifted decoys (shifts {3,4,7}). The TARGET is held
FIXED; only the BINDER register slides. A register-SPECIFIC binder scores well at shift0 and poorly when
shifted; a generic / heptad-periodic binder scores equally at every shift (its face is reproduced) - so
the shift-margin is exactly the register-achievability signal. Pair with scripts/contrastive_rank.py
(margin = score(shift0) - max_shift score(shift)) after a GPU predict_batch pass.

Systems and chiralities (the BINDER is the LONGER designed chain; TARGET is held fixed):
  8GQP : D-binder 62mer (chain A, chir D)  vs  L-pep1 20mer (chain B, chir L)        [helix]
  7YH8 : L-binder 62mer (chain A, chir L)  vs  D-Pep-1 19mer+anchorG (chain B, chir D) [helix]
         -- the added C-terminal anchor Gly is part of the TARGET and stays FIXED (it is not the binder).
  9DXX : D-peptide DP93 knottin 31mer (chain E, chir D)  vs  HA target (HA1 chain A + HA2 chain B, chir L)

Output is a predict_batch JSON list (each item: name, seq_a, chir_a, seq_b, chir_b, out_dir, + metadata).
seq_a / chir_a = BINDER (the chain whose register is shifted); seq_b / chir_b = TARGET (fixed). This is
the schema scripts/predict_batch.py / predict_complex.run consume directly. We do NOT run any GPU here.

REGISTER-SHIFT DEFINITION
  Helical binders (8GQP, 7YH8): a CIRCULAR shift of the binder sequence by k (b[k:] + b[:k]) - the same
  amino-acid composition presented in a different register, exactly as contrastive_decoys.py.

  9DXX knottin (CAVEAT): the DP93 binder is a disulfide-knotted cystine knot (3 intramolecular SS bonds:
  Cys4-Cys21, Cys11-Cys23, Cys17-Cys29 in auth numbering). A CIRCULAR register-shift is ILL-DEFINED for
  a knot - rotating the sequence scrambles which cysteines pair and destroys the knot topology, and a
  knottin has no single amphipathic "face" that a heptad shift would reproduce. We therefore FLAG the
  knottin and emit a BEST-EFFORT LINEAR shift instead: the binder is shifted within a FIXED-LENGTH window
  by deleting k residues from the N-terminus and padding k Gly at the C-terminus (length preserved so the
  complex still builds), keeping chirality D. These 9DXX decoys are documented as a weak / caveated
  control, NOT a clean register test; the helical systems (8GQP, 7YH8) are the trustworthy ones.

9DXX TARGET (two chains) vs the 2-chain predict schema
  DP93 contacts BOTH HA1 (chain A) and HA2 (chain B); it binds the HA stem at the HA1/HA2 interface
  (heavy-atom contacts <4.5 A: 82 to HA1, 108 to HA2). predict_complex.run is 2-chain (seq_a + seq_b),
  so we put the PRIMARY-contact stem chain HA2 in seq_b/chir_b (works with the stock predict_batch), and
  ALSO emit the full target as target_chains[] (HA1 + HA2) so a multi-chain predictor can use both.
  Flagged in each 9DXX item via target_note.

Usage:
  python scripts/make_benchmark_decoys.py --out_root XenoDesign1_local_ref/benchmarks/register_decoys \
      --json XenoDesign1_local_ref/benchmarks/register_decoys/items.json [--shifts 3,4,7]
  python scripts/make_benchmark_decoys.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the D->L name map + 1-letter table from score_complex; extend with the 9DXX modified residues.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_complex import _D2L, _3to1  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "XenoDesign1_local_ref" / "benchmarks"
NINEDXX = REPO / "XenoDesign1_local_ref" / "9dxx_target_gate" / "9dxx.cif"

# 9DXX chain-E non-standard / modified residues -> nearest standard parent (1-letter), from the CIF
# struct_mod / chem_comp records:
#   7YO = (2R)-5-oxo-pyrrolidine-2-carboxylic acid, "Oxidation" of PRO            -> P (proline-like)
#   KW4 = 5-methyl-D-norleucine, "Norleucine", parent DLE                          -> L (leucine-like)
#   HMF = D-beta-homophenylalanine (2-amino-4-phenyl-butyric acid)                 -> F (phenylalanine-like)
#   F9D = D-propargylglycine (2-aminopent-4-ynoic acid); no clean standard parent  -> X (non-standard)
# (DCY/DAR/DPN/DPR/DSN/DIL/DLY/DAS/DSG/DTY are already in score_complex._D2L.)
_NSAA_9DXX = {"7YO": "P", "KW4": "L", "HMF": "F", "F9D": "X"}


def _resname_to_1(name: str) -> str:
    """3-letter residue name -> 1-letter, handling L, D (via _D2L), Gly, and 9DXX modified residues."""
    if name in _NSAA_9DXX:
        return _NSAA_9DXX[name]
    std = _D2L.get(name, name)          # D->L for chirality, passthrough for L
    return _3to1.get(std, "X")


def _fasta_records(path: Path):
    txt = path.read_text()
    recs = []
    for block in txt.split(">"):
        if not block.strip():
            continue
        lines = block.splitlines()
        recs.append((lines[0].strip(), "".join(lines[1:]).strip()))
    return recs


def binder_target_8gqp():
    """8GQP: D-binder 62mer (chain A) vs L-pep1 20mer (chain B). From the deposited FASTA."""
    recs = dict((h, s) for h, s in _fasta_records(BENCH / "8GQP.fasta"))
    binder = next(s for h, s in recs.items() if "D-binder" in h)
    target = next(s for h, s in recs.items() if "L-pep1" in h)
    return {
        "system": "8GQP", "kind": "helix",
        "binder_seq": binder, "binder_chir": "D",
        "target_seq": target, "target_chir": "L",
        "fixed_tail": 0, "knottin": False,
        "target_chains": [{"seq": target, "chir": "L", "src": "L-pep1 chain B"}],
        "target_note": "single-chain L-pep1 target, fixed",
    }


def binder_target_7yh8():
    """7YH8: L-binder 62mer (chain A) vs D-Pep-1 (chain B). The D-Pep-1 carries an added C-terminal
    anchor Gly (panel seq_b 'DEHELLETAARWFYEIAKRG' = 19mer + G); that anchor is TARGET and stays FIXED.
    """
    recs = dict((h, s) for h, s in _fasta_records(BENCH / "7YH8.fasta"))
    binder = next(s for h, s in recs.items() if "L-19437" in h)
    pep = next(s for h, s in recs.items() if "D-Pep-1" in h)        # 19mer, no anchor in the fasta
    # The prediction panel used the D-Pep-1 with a C-terminal anchor Gly; mirror that (anchor fixed).
    panel = BENCH / "7YH8_pred_panel.json"
    target = pep + "G"
    if panel.exists():
        target = json.loads(panel.read_text()).get("seq_b", target)
    return {
        "system": "7YH8", "kind": "helix",
        "binder_seq": binder, "binder_chir": "L",
        "target_seq": target, "target_chir": "D",
        "fixed_tail": 0, "knottin": False,
        "target_chains": [{"seq": target, "chir": "D", "src": "D-Pep-1 + C-term anchor Gly (anchor FIXED)"}],
        "target_note": "D-Pep-1 target with C-terminal anchor Gly held fixed (target chain, not binder)",
    }


def binder_target_9dxx():
    """9DXX: D-peptide DP93 knottin (chain E, 31mer, all-D + 3-SS knot) vs HA target (HA1+HA2, L).

    Binder sequence is read from the CIF chain E (auth seqid 0..30; modified residues mapped via
    _NSAA_9DXX). The HA target is HA1 (chain A) + HA2 (chain B); DP93 binds the stem with more contacts
    to HA2, so HA2 is the 2-chain seq_b and both are emitted in target_chains.
    """
    import gemmi
    if not NINEDXX.exists():
        raise SystemExit(f"missing 9DXX CIF: {NINEDXX}  (run 9dxx_target_gate/fetch_and_parse_9dxx.py)")
    st = gemmi.read_structure(str(NINEDXX))
    m = st[0]
    # binder = chain E polymer residues (skip waters and the free K ligand at seqid 101)
    binder_letters = []
    cys_seqids = []
    for r in m["E"]:
        if r.is_water() or r.seqid.num > 30:   # 0..30 are the knottin; 101 is a free Lys ligand
            continue
        one = _resname_to_1(r.name)
        binder_letters.append(one)
        if _D2L.get(r.name, r.name) == "CYS":
            cys_seqids.append(r.seqid.num)
    binder = "".join(binder_letters)
    # HA target chains (L) - canonical sequences from the prep FASTA (atom-record waters/sugars excluded)
    ha = dict((h, s) for h, s in _fasta_records(REPO / "XenoDesign1_local_ref" / "9dxx_target_gate" / "ha_target.fasta"))
    ha1 = next(s for h, s in ha.items() if "HA1" in h)
    ha2 = next(s for h, s in ha.items() if "HA2" in h)
    return {
        "system": "9DXX", "kind": "knottin",
        "binder_seq": binder, "binder_chir": "D",
        # 2-chain schema: primary-contact stem chain HA2 goes in seq_b
        "target_seq": ha2, "target_chir": "L",
        "fixed_tail": 0, "knottin": True,
        "cys_seqids": cys_seqids,
        "target_chains": [
            {"seq": ha1, "chir": "L", "src": "HA1 chain A"},
            {"seq": ha2, "chir": "L", "src": "HA2 chain B (primary stem contact)"},
        ],
        "target_note": ("HA stem target spans HA1+HA2 (DP93 contacts both, more to HA2). 2-chain seq_b=HA2; "
                        "full target in target_chains[]. Use a multi-chain predictor for both."),
        "knottin_caveat": ("DP93 is a cystine-knot (3 SS: Cys4-Cys21, Cys11-Cys23, Cys17-Cys29 auth). "
                           "Circular register-shift is ill-defined for a knot; emitting BEST-EFFORT LINEAR "
                           "N-trim + C-Gly-pad shifts (length preserved). Treat 9DXX decoys as a caveated, "
                           "weak control - the helical systems 8GQP/7YH8 are the trustworthy register tests."),
    }


def _shift_helix(seq: str, k: int) -> str:
    """Circular register shift of a helical binder (composition preserved, register rotated)."""
    k = k % len(seq)
    return seq[k:] + seq[:k]


def _shift_knottin_linear(seq: str, k: int) -> str:
    """Best-effort LINEAR shift for a knottin (knot topology is destroyed; documented caveat):
    delete k residues from the N-terminus, pad k Gly at the C-terminus (length preserved)."""
    k = min(k, len(seq))
    return seq[k:] + "G" * k


def build_items(systems, shifts, out_root):
    items = []
    for s in systems:
        sysname = s["system"]
        binder, bchir = s["binder_seq"], s["binder_chir"]
        tgt, tchir = s["target_seq"], s["target_chir"]
        knottin = s["knottin"]
        shift_fn = _shift_knottin_linear if knottin else _shift_helix

        def make(name, kind, k, bseq):
            it = {
                "name": name, "system": sysname, "kind": kind, "shift": k,
                "seq_a": bseq, "chir_a": bchir,              # BINDER (register shifted)
                "seq_b": tgt, "chir_b": tchir,               # TARGET (fixed)
                "out_dir": f"{out_root}/{sysname}/{kind}{'' if k == 0 else k}",
                "binder_is_knottin": knottin,
                "target_chains": s["target_chains"],
                "target_note": s["target_note"],
            }
            if knottin:
                it["knottin_caveat"] = s["knottin_caveat"]
                it["shift_mode"] = "linear_Ntrim_Cglypad"
            else:
                it["shift_mode"] = "circular"
            return it

        items.append(make(f"{sysname}__real", "real", 0, binder))
        for k in shifts:
            items.append(make(f"{sysname}__shift{k}", "shift", k, shift_fn(binder, k)))
    return items


def all_systems():
    return [binder_target_8gqp(), binder_target_7yh8(), binder_target_9dxx()]


# --------------------------------------------------------------------------- selfcheck
def _selfcheck() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    g = binder_target_8gqp()
    y = binder_target_7yh8()
    check("8GQP binder is the 62mer D chain",
          len(g["binder_seq"]) == 62 and g["binder_chir"] == "D", f"len={len(g['binder_seq'])}")
    check("8GQP target is the 20mer L-pep",
          len(g["target_seq"]) == 20 and g["target_chir"] == "L", f"len={len(g['target_seq'])}")
    check("7YH8 binder is the 62mer L chain",
          len(y["binder_seq"]) == 62 and y["binder_chir"] == "L", f"len={len(y['binder_seq'])}")
    check("7YH8 target = D-Pep-1 (19) + anchor Gly => 20mer ending G, chir D",
          len(y["target_seq"]) == 20 and y["target_seq"].endswith("G") and y["target_chir"] == "D",
          f"target={y['target_seq']}")

    # circular shift preserves composition and length, and shift7 of an aperiodic binder differs from real
    b = g["binder_seq"]
    for k in (3, 4, 7):
        bs = _shift_helix(b, k)
        check(f"8GQP circular shift{k} preserves length & composition",
              len(bs) == len(b) and sorted(bs) == sorted(b))
        check(f"8GQP shift{k} differs from real (aperiodic binder)", bs != b)

    # 9DXX (only if CIF present)
    if NINEDXX.exists():
        d = binder_target_9dxx()
        check("9DXX binder read from CIF, 31mer all-D, flagged knottin",
              len(d["binder_seq"]) == 31 and d["binder_chir"] == "D" and d["knottin"] is True,
              f"len={len(d['binder_seq'])} seq={d['binder_seq']}")
        check("9DXX binder has 6 cysteines (3-SS knot)",
              d["binder_seq"].count("C") == 6 and len(d["cys_seqids"]) == 6,
              f"nCys={d['binder_seq'].count('C')} seqids={d['cys_seqids']}")
        check("9DXX has 2-chain HA target (HA1+HA2) + knottin caveat",
              len(d["target_chains"]) == 2 and "caveat" in d["knottin_caveat"].lower())
        # linear knottin shift preserves length, trims N, pads C with Gly
        ks = _shift_knottin_linear(d["binder_seq"], 3)
        check("9DXX linear shift3 preserves length and pads 3 C-term Gly",
              len(ks) == len(d["binder_seq"]) and ks.endswith("GGG") and ks[:-3] == d["binder_seq"][3:],
              f"shift3={ks}")
        items = build_items([d], [3, 4, 7], "X")
        check("9DXX items: knottin shift_mode is linear", all(
            it["shift_mode"] == "linear_Ntrim_Cglypad" for it in items if it["kind"] == "shift"))
        print("\n9DXX binder seq (lowercase D; Gly=G):", d["binder_seq"].replace("C", "c").lower())
    else:
        print("  [SKIP] 9DXX CIF absent; skipping 9DXX checks")

    # full item list shape mirrors contrastive_decoys (predict_batch schema)
    systems = [g, y] + ([binder_target_9dxx()] if NINEDXX.exists() else [])
    items = build_items(systems, [3, 4, 7], "ROOT")
    n_expected = 4 * len(systems)  # real + 3 shifts per system
    check("item count == 4 per system", len(items) == n_expected, f"got {len(items)}")
    keys = {"name", "system", "kind", "shift", "seq_a", "chir_a", "seq_b", "chir_b", "out_dir"}
    check("each item carries the predict_batch schema keys",
          all(keys <= set(it) for it in items))
    check("every real item has shift 0 and binder==deposited",
          all(it["shift"] == 0 for it in items if it["kind"] == "real"))

    print("\nSELFCHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description="Register-shift decoy generator for 8GQP / 7YH8 / 9DXX")
    p.add_argument("--out_root", default="XenoDesign1_local_ref/benchmarks/register_decoys")
    p.add_argument("--json", default=None, help="write the predict_batch items JSON here")
    p.add_argument("--shifts", default="3,4,7")
    p.add_argument("--systems", default="8GQP,7YH8,9DXX",
                   help="comma list subset of 8GQP,7YH8,9DXX")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args(argv)
    if a.selfcheck:
        return _selfcheck()
    shifts = [int(s) for s in a.shifts.split(",")]
    want = set(s.strip() for s in a.systems.split(","))
    builders = {"8GQP": binder_target_8gqp, "7YH8": binder_target_7yh8, "9DXX": binder_target_9dxx}
    systems = [builders[k]() for k in ("8GQP", "7YH8", "9DXX") if k in want]
    items = build_items(systems, shifts, a.out_root)
    out_json = a.json or f"{a.out_root}/items.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(items, indent=2))
    print(f"{len(items)} items for {len(systems)} systems (real + shifts {shifts}) -> {out_json}")
    for s in systems:
        flag = "  [KNOTTIN: linear best-effort shift, see caveat]" if s["knottin"] else ""
        print(f"  {s['system']:5} binder({s['binder_chir']},{len(s['binder_seq'])}aa) "
              f"vs target({s['target_chir']},{len(s['target_seq'])}aa){flag}")
    return items


if __name__ == "__main__":
    r = main()
    sys.exit(r if isinstance(r, int) else 0)
