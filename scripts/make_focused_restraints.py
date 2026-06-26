"""Build FOCUSED (<=3) chai POCKET restraints for the structurally-FAILING Trues (Goal 1 follow-up).

Goal 1 (structure_compliance.py) showed 4 systems whose Chai-predicted "real" complex does NOT
recapitulate the deposited interface under the existing SINGLE-anchor pocket restraint (binder docks
20-36 A away): 3L35, 3MGN, 2R3C (gp41 family) and 8GQP. Re-predict these with BETTER FOCUSED
contact restraints, NO MORE THAN 3 each; keep the predictions aside and only fold them into the
objective re-fit if they actually come back good by DockQ.

Approach: instead of one loose pocket anchor, emit the TOP-3 TARGET residues most contacted by the
binder in the DEPOSITED cognate interface, each as a chai POCKET row (whole binder chain -> that target
residue, max 6.0 A). Three anchors spread across the real pocket localize the binder tightly to the
correct site while staying register-agnostic (binder side is the whole chain, res_idxA empty) and
anchoring only on the TARGET -- which is L for all four systems, so no D-residue name-check issue.

chai .restraints CSV (FASTA-RECORD-ORDER chains): chainA = the pocket-bound chain (res_idxA EMPTY),
chainB = target, res_idxB = <one-letter><1-based pos along the target FASTA/polymer>.

CPU-only; gemmi. Writes restraints + a real-only manifest the run_restrained_batch driver reads.
  PYTHONPATH=$PWD python3 scripts/make_focused_restraints.py            # build restraints + manifest + self-check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gemmi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from score_complex import _3to1, _D2L  # noqa: E402

BENCH = REPO / "XenoDesign1_local_ref" / "benchmarks"
GP41 = BENCH / "gp41_mdm2"
OUT_RESTR = BENCH / "focused" / "restraints"
OUT_ROOT = BENCH / "focused"            # predictions land under focused/<sys>/real/seed*
MANIFEST = BENCH / "focused" / "manifest_focused.json"

SKIP = {"ACE", "NH2", "SO4", "CL", "YT3", "GOL", "IPA", "PEG", "HOH"}
CONTACT_CUTOFF_A = 4.5
POCKET_DIST_A = 6.0      # same as the working MDM2 single anchors; 3 of them = focused localization
TOPN = 3
SEEDS = (42, 43, 44)
HEADER = ("chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
          "min_distance_angstrom,max_distance_angstrom,comment,restraint_id")

# system -> (deposit_cif, bound_chain, target_chain, reuse_fasta, na, nb)
#   bound_chain = the chain the POCKET ties (chai chain A / FASTA record 1)
#   target_chain = the chain whose residues we anchor on (chai chain B / FASTA record 2)
# gp41 trio: D-binder (record1=A) onto L IQN17 target chain A-of-deposit (record2=B). Reuse the
#   existing per-system real FASTA. na=binder len, nb=target len (from gp41_mdm2 manifest).
# 8GQP: the 62-mer "D-binder" (chai A) onto the 20-mer L target (chai B); reuse the register_decoys fasta.
SYSTEMS = {
    "3L35": (GP41 / "cifs/3L35.cif", "H", "A", GP41 / "fastas/3L35_real.fasta", 16, 45),
    "3MGN": (GP41 / "cifs/3MGN.cif", "K", "A", GP41 / "fastas/3MGN_real.fasta", None, None),
    "2R3C": (GP41 / "cifs/2R3C.cif", "D", "A", GP41 / "fastas/2R3C_real.fasta", None, None),
    "8GQP": (BENCH / "8GQP.cif", "A", "B", BENCH / "register_decoys/8GQP/real/input.fasta", 62, 20),
}


def _polymer(chain):
    return [r for r in chain if not r.is_water() and r.name not in SKIP]


def _one(resname):
    return _3to1.get(_D2L.get(resname, resname), "X")


def heavy(chain):
    return [a.pos for r in _polymer(chain) for a in r if not a.is_hydrogen()]


def top_target_residues(cif: Path, bound_chain: str, target_chain: str, n=TOPN):
    """Top-n target residues by # binder heavy-atom contacts (<4.5 A). Returns
    [(pos_1based_along_target_polymer, resname, one_letter, n_contacts), ...] sorted by contacts."""
    m = gemmi.read_structure(str(cif))[0]
    bheavy = heavy(m[bound_chain])
    scored = []
    for pos, r in enumerate(_polymer(m[target_chain]), start=1):
        nc = 0
        for a in r:
            if a.is_hydrogen():
                continue
            for bp in bheavy:
                if a.pos.dist(bp) <= CONTACT_CUTOFF_A:
                    nc += 1
        if nc:
            scored.append((pos, r.name, _one(r.name), nc))
    scored.sort(key=lambda s: -s[3])
    if not scored:
        raise RuntimeError(f"{cif.name}: no target contacts")
    return scored[:n]


def write_restraint(out: Path, sys_name: str, residues):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]
    for k, (pos, resname, one, nc) in enumerate(residues, start=1):
        lines.append(f"A,,B,{one}{pos},pocket,1.0,0.0,{POCKET_DIST_A},"
                     f"{sys_name}-focused-{resname}{pos}-c{nc},{sys_name}_focused_{k}")
    out.write_text("\n".join(lines) + "\n")


def build():
    OUT_RESTR.mkdir(parents=True, exist_ok=True)
    jobs = []
    report = {}
    for name, (cif, bchain, tchain, fasta, na, nb) in SYSTEMS.items():
        residues = top_target_residues(cif, bchain, tchain)
        rfile = OUT_RESTR / f"{name}.restraints"
        write_restraint(rfile, name, residues)
        report[name] = [f"{one}{pos}(c{nc})" for pos, _, one, nc in residues]
        # infer na/nb from fasta if not given (count polymer residues per record)
        if na is None or nb is None:
            recs = [l for l in fasta.read_text().splitlines() if not l.startswith(">")]
            # crude: count residues in record1 (binder, parens) and record2 (target)
            seqs = []
            cur = ""
            for line in fasta.read_text().splitlines():
                if line.startswith(">"):
                    if cur:
                        seqs.append(cur)
                    cur = ""
                else:
                    cur += line.strip()
            if cur:
                seqs.append(cur)
            na = seqs[0].count("(") + sum(1 for c in seqs[0] if c.isalpha() and seqs[0][max(0, seqs[0].find(c) - 1)] != "(")
            # simpler robust count: parens-tokens + bare letters
            na = _count_tokens(seqs[0])
            nb = _count_tokens(seqs[1])
        for seed in SEEDS:
            jobs.append({
                "system": name, "item": "real", "seed": seed,
                "lane": len(jobs) % 2,               # global round-robin -> even GPU split
                "fasta": str(fasta.relative_to(REPO)),
                "restraint": str(rfile.relative_to(REPO)),
                "out_dir": str((OUT_ROOT / name / "real" / f"seed{seed}").relative_to(REPO)),
                "na": na, "nb": nb, "use_msa": False,
            })
    MANIFEST.write_text(json.dumps(jobs, indent=1))
    return report, jobs


def _count_tokens(seq: str) -> int:
    """Count residues in a chai FASTA sequence string: each (XXX) paren group is one residue,
    each bare letter outside parens is one residue."""
    n, i = 0, 0
    while i < len(seq):
        if seq[i] == "(":
            j = seq.index(")", i)
            n += 1
            i = j + 1
        elif seq[i].isalpha():
            n += 1
            i += 1
        else:
            i += 1
    return n


def main():
    report, jobs = build()
    print("=== focused (<=3) pocket restraints (top target residues by binder contacts) ===")
    for s, rs in report.items():
        print(f"  {s}: {rs}")
    print(f"\n{len(jobs)} jobs ({len(SYSTEMS)} systems x {len(SEEDS)} seeds, real-only) -> {MANIFEST}")
    print("lane 0:", sum(1 for j in jobs if j["lane"] == 0), " lane 1:", sum(1 for j in jobs if j["lane"] == 1))
    # self-check: each restraint file has <=3 pocket rows + header
    for s in SYSTEMS:
        rows = (OUT_RESTR / f"{s}.restraints").read_text().strip().splitlines()
        assert 1 < len(rows) <= 1 + TOPN, f"{s}: {len(rows)-1} restraint rows (>3)"
    print("self-check: all restraint files have <=3 rows  PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
