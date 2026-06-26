"""CPU prep for the gp41/IQN17 (D-peptide:L-target) + MDM2 DPMI prediction batch (n>2 fit set).

NEW real heterochiral systems (D-peptide BINDER : L-protein TARGET). For each system we emit:
  - the chai input FASTA (D-binder in parens via xenodesign.io_spec, Gly bare; L-target normal),
  - a SINGLE-residue chai POCKET restraint on the L-TARGET (rule: target residue with the
    most binder heavy-atom contacts <4.5 A; tie-break -> more hydrophobic + lower avg contact dist),
  - 2 composition-preserving SCRAMBLES of the D-binder (target fixed) -- NEGATIVES (no register shifts),
and adds {real, scram1, scram2} x seeds {42,43,44} jobs to a manifest the run_restrained_batch driver reads.

Pocket anchor is on the L-target -> tokenizes cleanly (no D-residue name-check issue). gp41 binders carry
intramolecular disulfides; those run UNCONSTRAINED (chai rejects COVALENT bonds on D-Cys -- see memory).
MDM2 DPMI binders are Gly-free all-D 12-mers -> we APPEND ONE C-TERMINAL Gly so chai tokenizes the chain
directly (avoiding the fully-NCAA mirror path), per the task.

CPU/network PREP ONLY: fetches nothing here (CIFs already in cifs/); does NOT run any GPU.

Usage:
  PYTHONPATH=$PWD python3 scripts/prep_gp41_mdm2_batch.py            # build everything + self-check
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_complex import _3to1, _D2L  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xenodesign.io_spec import build_fasta  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "XenoDesign1_local_ref" / "benchmarks" / "gp41_mdm2"
CIFS = ROOT / "cifs"
FASTA_DIR = ROOT / "fastas"
RESTR_DIR = ROOT / "restraints"

# residues that are not part of the peptide polymer (caps / ions / buffers / solvents)
SKIP = {"ACE", "NH2", "SO4", "CL", "YT3", "GOL", "IPA", "PEG"}
# modified D residues with no entry in our D->L map -> nearest standard D parent (documented caveat).
#   D0C = 4-chloro-D-phenylalanine (7KJM)  -> D-Phe (one-letter f)
MOD_D = {"D0C": "PHE"}
CONTACT_CUTOFF_A = 4.5
SINGLE_DIST_A = 6.0
HEADER = ("chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
          "min_distance_angstrom,max_distance_angstrom,comment,restraint_id")

# Kyte-Doolittle hydropathy (higher = more hydrophobic) for the pocket-residue tie-break.
KD = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4,
      "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
      "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5, "X": 0.0}

# system -> (pdb_id, binder_chain, target_chain, mdm2). binder/target chains chosen by max
# heavy-atom contacts to target chain A in the reconnaissance pass.
SYSTEMS = [
    # gp41 / IQN17 family: D-peptide binder (has a Gly anchor) vs L IQN17 coiled-coil (chain A).
    ("3L35", "H", "A", False),   # PIE12
    ("2R5B", "H", "A", False),   # PIE7
    ("2R3C", "D", "A", False),   # PIE1  (D = kG... has Gly + ACE-stripped)
    ("3MGN", "K", "A", False),   # PIE71 (K = kGfvc... most contacts, has Gly)
    ("1CZQ", "D", "A", False),   # D10-P1
    ("2Q3I", "D", "A", False),   # D10-P3
    # MDM2 DPMI family: Gly-free all-D 12-mer -> APPEND C-terminal Gly; vs MDM2 (~83-85aa, chain A).
    ("3LNJ", "B", "A", True),    # DPMI-alpha
    ("7KJM", "B", "A", True),    # DPMI variant (has D0C = 4-Cl-D-Phe at pos 7)
    ("3IWY", "B", "A", True),    # DPMI-gamma
]


def _polymer(chain):
    return [r for r in chain if not r.is_water() and r.name not in SKIP]


def _one_letter_D(resname: str) -> str:
    """D (or modified D) 3-letter -> one-letter L code. Gly -> G."""
    if resname in MOD_D:
        three = MOD_D[resname]
    else:
        three = _D2L.get(resname, resname)
    return "G" if three == "GLY" else _3to1.get(three, "X")


def binder_one_letter(chain) -> str:
    """All-D binder chain -> one-letter L sequence (D residues map D->L; Gly stays G)."""
    return "".join(_one_letter_D(r.name) for r in _polymer(chain))


def target_one_letter(chain) -> str:
    """L target chain -> one-letter sequence."""
    return "".join(_3to1.get(r.name, "X") for r in _polymer(chain))


def heavy_pos(chain):
    return [a.pos for r in _polymer(chain) for a in r if not a.is_hydrogen()]


def central_target_residue(cif: Path, binder_chain: str, target_chain: str):
    """Rule: TARGET residue with the MOST binder heavy-atom contacts (<4.5 A).
    Tie-break: more hydrophobic (Kyte-Doolittle), then lower average contact distance.
    Returns (pos_1based, resname, one_letter, n_contacts, avg_dist)."""
    st = gemmi.read_structure(str(cif))
    m = st[0]
    binder_heavy = heavy_pos(m[binder_chain])
    scored = []
    for pos, r in enumerate(_polymer(m[target_chain]), start=1):
        dists = []
        for a in r:
            if a.is_hydrogen():
                continue
            for bp in binder_heavy:
                d = a.pos.dist(bp)
                if d <= CONTACT_CUTOFF_A:
                    dists.append(d)
        if not dists:
            continue
        one = _3to1.get(r.name, "X")
        avg = sum(dists) / len(dists)
        scored.append((pos, r.name, one, len(dists), avg))
    if not scored:
        raise RuntimeError(f"{cif.name}: no target residue contacts the binder")
    # primary: most contacts (desc); tie: more hydrophobic (desc); then lower avg dist (asc)
    best = max(scored, key=lambda s: (s[3], KD.get(s[2], 0.0), -s[4]))
    return best


def write_restraint(out: Path, pos: int, one: str, resname: str, n: int, tag: str) -> None:
    """Single POCKET restraint in chai FASTA-order chains.

    chai 0.6.1 labels chains by FASTA RECORD ORDER, not deposit letters. Our FASTAs are
    [record1 = D-BINDER -> chai chain A, record2 = L-TARGET -> chai chain B]. POCKET semantics:
      chainA = chain-level partner (res_idxA EMPTY)  -> the whole BINDER (chai chain A)
      chainB = token-level anchor (res_idxB = <one><pos>) -> the TARGET residue (chai chain B),
               `pos` indexing into the TARGET FASTA sequence (== _polymer(target) order).
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    line = (f"A,,B,{one}{pos},pocket,1.0,0.0,"
            f"{SINGLE_DIST_A},{tag}-single-{resname}{pos}-c{n},{tag}_single_{pos}")
    out.write_text(HEADER + "\n" + line + "\n")


def scramble(seq: str, rng: random.Random, keep_idx: set[int]) -> str:
    """Composition-preserving scramble. Positions in keep_idx (Gly anchor / Cys) stay fixed;
    the remaining residues are permuted. Mirrors scripts/build_9dxx scramble_keep_cys."""
    movable = [c for i, c in enumerate(seq) if i not in keep_idx]
    rng.shuffle(movable)
    out, k = [], 0
    for i, c in enumerate(seq):
        if i in keep_idx:
            out.append(c)
        else:
            out.append(movable[k]); k += 1
    return "".join(out)


def make_binder_fasta(name: str, seq_one_letter: str) -> str:
    """chai FASTA for the all-D binder (parens, Gly bare) via io_spec.build_fasta."""
    return build_fasta([{"type": "protein", "name": name,
                         "sequence": seq_one_letter, "chirality": "D"}])


def build():
    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    RESTR_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    report = []
    for pid, bchain, tchain, mdm2 in SYSTEMS:
        cif = CIFS / f"{pid}.cif"
        st = gemmi.read_structure(str(cif))
        m = st[0]
        binder = binder_one_letter(m[bchain])
        target = target_one_letter(m[tchain])
        ntarget = len(target)  # L-target length (chai chain B, FASTA record 2)

        # MDM2 DPMI: Gly-free all-D -> append ONE C-terminal Gly so chai tokenizes directly.
        appended_gly = False
        if mdm2 and "G" not in binder:
            binder = binder + "G"
            appended_gly = True

        # keep the Gly anchor position(s) and any Cys fixed in scrambles
        keep = {i for i, c in enumerate(binder) if c in ("G", "C")}

        # build the three binder variants (target fixed)
        variants = {"real": binder}
        for n in (1, 2):
            rng = random.Random(7000 + 13 * len(jobs) + n)  # deterministic, distinct per system+n
            scr = scramble(binder, rng, keep)
            # ensure a genuine reorder (retry a few seeds if degenerate / equal)
            tries = 0
            while (scr == binder) and tries < 20:
                rng = random.Random(8000 + tries)
                scr = scramble(binder, rng, keep)
                tries += 1
            assert sorted(scr) == sorted(binder), f"{pid} scram{n} composition changed"
            variants[f"scram{n}"] = scr

        # write FASTAs (one per variant): D-binder + L-target
        target_fasta_block = build_fasta([{"type": "protein", "name": f"{pid}_target_{tchain}",
                                            "sequence": target, "chirality": "L"}])
        na = len(binder)   # chai chain A length = BINDER (FASTA record 1, incl. any appended Gly)
        nb = ntarget       # chai chain B length = TARGET (FASTA record 2)
        fasta_paths = {}
        tokenizes = True
        for item, bseq in variants.items():
            bname = f"{pid}_binder_{bchain}_{item}"
            try:
                bfasta = make_binder_fasta(bname, bseq)
            except ValueError:
                tokenizes = False
                raise
            full = bfasta + target_fasta_block
            fp = FASTA_DIR / f"{pid}_{item}.fasta"
            fp.write_text(full)
            fasta_paths[item] = fp.relative_to(REPO).as_posix()

        # POCKET restraint: anchor on the L-target (chai chain B), whole binder = chai chain A
        pos, resname, one, ncont, avg = central_target_residue(cif, bchain, tchain)
        rpath = RESTR_DIR / f"{pid}.restraints"
        write_restraint(rpath, pos, one, resname, ncont, pid)
        rrel = rpath.relative_to(REPO).as_posix()

        # MSA flag: gp41 IQN17 is a short designed coiled-coil -> try MSA-free; MDM2 ~85aa may benefit.
        use_msa = bool(mdm2)

        # manifest jobs: {real, scram1, scram2} x {42,43,44}
        lane = 0 if not mdm2 else 1
        for item in ("real", "scram1", "scram2"):
            for seed in (42, 43, 44):
                jobs.append({
                    "system": pid,
                    "item": item,
                    "seed": seed,
                    "lane": lane,
                    "fasta": fasta_paths[item],
                    "restraint": rrel,
                    "out_dir": f"XenoDesign1_local_ref/benchmarks/gp41_mdm2/{pid}/{item}/seed{seed}",
                    "na": na,  # chai chain A (binder) length
                    "nb": nb,  # chai chain B (target) length
                    "use_msa": use_msa,
                })

        report.append({
            "pdb": pid, "family": "MDM2-DPMI" if mdm2 else "gp41-IQN17",
            "binder_chain": bchain, "binder_len": na, "binder_seq": binder,
            "appended_cterm_gly": appended_gly,
            "target_chain": tchain, "target_len": nb,
            "pocket_residue": f"{resname}{pos}", "pocket_one": one,
            "pocket_contacts": ncont, "pocket_avg_dist": round(avg, 2),
            "tokenizes": tokenizes, "use_msa": use_msa,
            "disulfide_caveat": ("C" in binder and not mdm2),
        })

    manifest = ROOT / "manifest.json"
    manifest.write_text(json.dumps(jobs, indent=2))
    return manifest, jobs, report


def _selfcheck(jobs, report):
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    chk("9 systems", len(report) == 9)
    chk("9*3*3 = 81 jobs", len(jobs) == 81, f"got {len(jobs)}")
    # each FASTA tokenizes (binder not fully-NCAA: build_fasta already raised otherwise)
    for r in report:
        chk(f"{r['pdb']} tokenizes", r["tokenizes"])
        chk(f"{r['pdb']} binder has a bare Gly anchor",
            "G" in r["binder_seq"], f"seq={r['binder_seq']}")
        if r["family"] == "MDM2-DPMI":
            chk(f"{r['pdb']} MDM2 appended C-term Gly", r["appended_cterm_gly"])
    # restraint files exist + are single POCKET rows in chai FASTA-order chains (A=binder, B=target)
    job_na = {j["system"]: j["na"] for j in jobs}
    job_nb = {j["system"]: j["nb"] for j in jobs}
    for r in report:
        rp = RESTR_DIR / f"{r['pdb']}.restraints"
        rows = rp.read_text().splitlines()
        chk(f"{r['pdb']} restraint = header+1 row", len(rows) == 2, f"{len(rows)} lines")
        body = rows[1].split(",")
        # chai chains: A (binder, chain-level, empty res_idxA) -> B (target, token-level anchor)
        chk(f"{r['pdb']} pocket row: chainA=A empty res, chainB=B anchor",
            body[0] == "A" and body[1] == "" and body[2] == "B"
            and body[4] == "pocket" and body[3][0].isalpha(),
            f"row={body[:5]}")
        # anchor one-letter must match the TARGET FASTA position (chai chain B)
        tgt_fasta = (FASTA_DIR / f"{r['pdb']}_real.fasta").read_text()
        blocks, cur = [], ""
        for ln in tgt_fasta.splitlines():
            if ln.startswith(">"):
                if cur:
                    blocks.append(cur); cur = ""
            else:
                cur += ln.strip()
        if cur:
            blocks.append(cur)
        tgt_seq = blocks[1]  # record 2 = target = chai chain B
        anchor_one, anchor_pos = body[3][0], int(body[3][1:])
        chk(f"{r['pdb']} anchor {body[3]} matches target FASTA pos",
            1 <= anchor_pos <= len(tgt_seq) and tgt_seq[anchor_pos - 1] == anchor_one,
            f"FASTA[{anchor_pos}]={tgt_seq[anchor_pos-1] if 0<anchor_pos<=len(tgt_seq) else '?'}")
        # manifest na = binder (chain A) length; nb = target (chain B) length
        chk(f"{r['pdb']} na == binder len ({r['binder_len']})",
            job_na[r["pdb"]] == r["binder_len"], f"na={job_na[r['pdb']]}")
        chk(f"{r['pdb']} nb == target len ({r['target_len']})",
            job_nb[r["pdb"]] == r["target_len"], f"nb={job_nb[r['pdb']]}")
    # scrambles preserve composition and differ from real
    print("SELFCHECK:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    manifest, jobs, report = build()
    print(f"\nWrote manifest: {manifest}  ({len(jobs)} jobs)\n")
    print(f"{'PDB':5} {'fam':12} {'bchain':6} {'blen':4} {'tlen':4} "
          f"{'pocket':9} {'cont':4} {'avgd':5} {'msa':4} {'tok':4} caveat")
    for r in report:
        cav = "SS-unconstrained" if r["disulfide_caveat"] else ("C-Gly-added" if r["appended_cterm_gly"] else "")
        print(f"{r['pdb']:5} {r['family']:12} {r['binder_chain']:6} {r['binder_len']:<4} "
              f"{r['target_len']:<4} {r['pocket_residue']:9} {r['pocket_contacts']:<4} "
              f"{r['pocket_avg_dist']:<5} {('Y' if r['use_msa'] else 'N'):4} "
              f"{('Y' if r['tokenizes'] else 'N'):4} {cav}")
        print(f"        binder(D,lowercase reporting): {r['binder_seq']}")
    print()
    ok = _selfcheck(jobs, report)
    sys.exit(0 if ok else 1)
