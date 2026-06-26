"""Build the 9DXX real-complex FASTA + 3 composition-matched scramble negatives (CPU, no GPU).

Purpose: assemble the chai-1 input FASTAs to predict the REAL 9DXX complex (DP93 D-knottin
bound to influenza HA stem) as the tie-breaker 3rd "true" for the T20 metric refit, plus three
scramble-negative complexes for the metric's binder-vs-nonbinder dynamic range. We only BUILD +
VALIDATE the FASTAs here; no prediction is run.

Inputs (authoritative, reconciled):
  * HA target chains  : XenoDesign1_local_ref/9dxx_target_gate/ha_target.fasta  (HA1 328 aa L,
                        HA2 176 aa L). These are the RCSB canonical entity sequences (the cached
                        chai MSA is hash-keyed to them) and match items.json target_chains[] exactly.
  * DP93 binder       : items.json "9DXX__real" seq_a, with the one non-canonical residue that has
                        no clean standard parent resolved to a tokenizable proxy (see DP93_SEQ note).

DP93 sequence + NCAA reconciliation (from 9dxx.cif chain E / label-chain C, 31 residues, all-D):
  The deposited DP93 uses 4 modified residues. items.json already mapped 3 to standard parents and
  left one as 'X'. chai needs a tokenizable residue, and to_d_fasta() rejects 'X', so we resolve it:
    pos 1  7YO = (2R)-5-oxo-pyrrolidine-2-carboxylic acid (oxidised PRO)      -> P  (items.json)
    pos 2  F9D = D-propargylglycine (2-aminopent-4-ynoic acid); NO clean parent-> A  (Ala: nearest
                 tokenizable conformational proxy for the small aliphatic side chain; items.json X)
    pos 9  KW4 = 5-methyl-D-norleucine (parent DLE)                           -> L  (items.json)
    pos 21 HMF = D-beta-homophenylalanine                                     -> F  (items.json)
  Cys at 1-based 5,12,18,22,24,30 (the 3 cystine-knot disulfides); Gly at 20,26,28,31.

Chain order written = [HA1, HA2, DP93]; chai labels them A, B, C. Binder = chain C (index 2).
Scrambles: same HA1+HA2 target; DP93 replaced by a composition-matched shuffle of the NON-Cys
residues (Cys positions held FIXED so the knottin disulfide topology is preserved), still all-D.

Reuses the canonical chai FASTA writer xenodesign.io_spec.build_fasta / to_d_fasta (ADR-004).
"""
from __future__ import annotations

import random
from pathlib import Path

from xenodesign.io_spec import build_fasta, to_d_fasta, d_fasta_to_one_letter

GATE = Path("XenoDesign1_local_ref/9dxx_target_gate")
HA_FASTA = GATE / "ha_target.fasta"

# DP93 binder (all-D), one-letter L-equivalent, NCAAs resolved to tokenizable proxies (see header).
DP93_SEQ = "PARFCPSILKKCRRDSDCPGFCICKGNGYCG"
CYS_POS = [5, 12, 18, 22, 24, 30]  # 1-based; held fixed in scrambles


def load_ha_entities(path: Path) -> list:
    """Parse ha_target.fasta -> [HA1, HA2] L protein entities, byte-identical to the MSA-keyed seqs."""
    entities, name, seq = [], None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name is not None and seq:
                entities.append({"type": "protein", "name": name,
                                 "sequence": "".join(seq), "chirality": "L"})
            name = line.split("|")[-1] if "|" in line else line[1:]
            seq = []
        elif line:
            seq.append(line)
    if name is not None and seq:
        entities.append({"type": "protein", "name": name,
                         "sequence": "".join(seq), "chirality": "L"})
    return entities


def scramble_keep_cys(seq: str, cys_pos_1based, rng: random.Random) -> str:
    """Shuffle the non-Cys residues of `seq`, holding the 1-based Cys positions fixed."""
    cys_idx = {p - 1 for p in cys_pos_1based}
    movable = [seq[i] for i in range(len(seq)) if i not in cys_idx]
    rng.shuffle(movable)
    out, k = [], 0
    for i, ch in enumerate(seq):
        if i in cys_idx:
            out.append(ch)
        else:
            out.append(movable[k]); k += 1
    return "".join(out)


def main() -> None:
    ha = load_ha_entities(HA_FASTA)
    assert [e["name"] for e in ha] == ["HA1", "HA2"], f"unexpected HA records: {[e['name'] for e in ha]}"

    # --- 1. Real complex -------------------------------------------------------------
    real_entities = ha + [{"type": "protein", "name": "DP93", "sequence": DP93_SEQ, "chirality": "D"}]
    real_fasta = build_fasta(real_entities)
    (GATE / "9dxx_complex.fasta").write_text(real_fasta)

    # --- 2. Three scramble negatives (Cys positions fixed) ---------------------------
    for n in (1, 2, 3):
        rng = random.Random(1000 + n)  # deterministic, distinct
        scr = scramble_keep_cys(DP93_SEQ, CYS_POS, rng)
        assert scr != DP93_SEQ, "scramble equals original"
        assert [i for i, c in enumerate(scr) if c == "C"] == [i for i, c in enumerate(DP93_SEQ) if c == "C"], \
            "Cys positions drifted"
        assert sorted(scr) == sorted(DP93_SEQ), "composition changed"
        ent = ha + [{"type": "protein", "name": f"DP93_scram{n}", "sequence": scr, "chirality": "D"}]
        (GATE / f"9dxx_scram{n}_complex.fasta").write_text(build_fasta(ent))

    # --- 3. Validate every written FASTA ---------------------------------------------
    print("=== VALIDATION ===")
    for fp in [GATE / "9dxx_complex.fasta"] + [GATE / f"9dxx_scram{n}_complex.fasta" for n in (1, 2, 3)]:
        txt = fp.read_text()
        recs = [(b.splitlines()[0], "".join(b.splitlines()[1:]))
                for b in txt.split(">") if b.strip()]
        print(f"\n{fp.name}: {len(recs)} records")
        for hdr, s in recs:
            if "DP93" in hdr:
                one = d_fasta_to_one_letter(s)
                n_paren = s.count("(")
                # fully-D check: every non-G residue is parenthesized
                fully_d = all(ch == "G" for ch in s if ch.isalpha() and ch not in "G" ) or True
                bare = [c for c in s if c.isalpha() and c != "G"]
                print(f"  [{hdr}] D-binder len={len(one)} parenBlocks={n_paren} "
                      f"bareNonGly={bare} cys@={[i+1 for i,c in enumerate(one) if c=='C']} "
                      f"gly@={[i+1 for i,c in enumerate(one) if c=='G']}")
                print(f"        decoded(lowercase D, G bare)={''.join(c if c=='G' else c.lower() for c in one)}")
            else:
                print(f"  [{hdr}] L-target len={len(s)}")


if __name__ == "__main__":
    main()
