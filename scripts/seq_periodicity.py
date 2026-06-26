"""Reference-free SEQUENCE-PERIODICITY / register-achievability metric (CPU, sequence-only).

Motivation (empirically confirmed): register-SPECIFICITY is only a meaningful design objective for
binders whose amphipathic face is NOT cleanly heptad-periodic. A perfectly heptad-periodic helix
(hydrophobic i, i+7, i+14 ...) reproduces the SAME interface face under a 7-residue register shift, so no
score - structural or sequence - can prefer the native register over the shifted one: register
specificity is UNACHIEVABLE. The real solution binders are amphipathic but LESS periodic (8GQP
hydropathy autocorr lag4 +0.35, lag7 +0.14), so a shift presents a different face and register IS
scorable; a periodic seed (e.g. vtyr seed_50: lag7 +0.52) is not.

This metric is REFERENCE-FREE (no GT, no structure): it reads only the binder amino-acid SEQUENCE and
quantifies how periodic its hydrophobicity is, via the autocorrelation of the Kyte-Doolittle hydropathy
profile. It reports:

  - kd_profile_len            : number of residues scored (= binder length)
  - autocorr                  : autocorrelation of the (mean-centred) KD hydropathy at lags 1..max_lag
                                (Pearson-style, normalized by lag-0 variance; in [-1, 1])
  - autocorr_lag3 / _lag4 / _lag7 : the three diagnostic lags (3-4 = one alpha-helix face spacing i,i+3/i+4;
                                7 = the heptad repeat a..g of a coiled-coil / amphipathic helix)
  - peak_lag                  : the lag in [min_lag .. max_lag] with the LARGEST positive autocorr
  - peak_autocorr             : the autocorrelation at peak_lag
  - register_achievable       : boolean. FALSE when the helix is strongly heptad-periodic
                                (autocorr_lag7 >= heptad_thresh) AND lag7 is the (near-)dominant peak,
                                because a 7-shift then reproduces the face. TRUE otherwise.
  - register_achievable_reason: short human-readable justification.

WHY autocorrelation (not an FFT power spectrum): the sequences are short (~20-62 aa). A direct lagged
autocorrelation is robust at short length, needs no windowing, and gives a value AT each chemically
meaningful lag (3, 4, 7) directly, which is exactly what the register-achievability decision needs.

CLI:
  python scripts/seq_periodicity.py SEQUENCE [--max_lag 10] [--heptad_thresh 0.35] [--out json]
  python scripts/seq_periodicity.py --fasta f.fasta [--record NAME] ...
  python scripts/seq_periodicity.py --selfcheck

Sequence case is ignored for the hydropathy lookup (D-residues are reported lowercase elsewhere in this
repo, but chirality does not change a side chain's hydrophobicity, so the KD profile is chirality-blind).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Kyte & Doolittle (1982) hydropathy index. Higher = more hydrophobic.
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def kd_profile(seq: str) -> np.ndarray:
    """Kyte-Doolittle hydropathy value per residue (0.0 for unknown/non-standard letters)."""
    return np.array([_KD.get(c.upper(), 0.0) for c in seq], float)


def autocorrelation(x: np.ndarray, max_lag: int) -> list[float]:
    """Normalized autocorrelation of x (mean-centred) at lags 1..max_lag.

    r(k) = sum_i (x_i - xbar)(x_{i+k} - xbar) / sum_i (x_i - xbar)^2  (lag-0 normalization).
    Returned values are in [-1, 1]; r(k)=None-equivalent (0.0) is avoided by gating on length.
    """
    n = len(x)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    out = []
    for k in range(1, max_lag + 1):
        if k >= n or denom <= 0.0:
            out.append(0.0)
            continue
        num = float((xc[:-k] * xc[k:]).sum())
        out.append(num / denom)
    return out


def compute(seq: str, max_lag: int = 10, heptad_thresh: float = 0.35,
            min_lag: int = 2, dominance: float = 0.9) -> dict:
    """Reference-free register-achievability features for a binder sequence.

    register_achievable is FALSE when the hydropathy is strongly heptad-periodic: lag-7 autocorr is
    >= heptad_thresh AND lag-7 is the dominant peak (its autocorr >= `dominance` * the global peak).
    In that regime a 7-residue register shift reproduces the same hydrophobic face, so no register
    metric can discriminate the native register. Otherwise register specificity is achievable.
    """
    seq = seq.strip()
    kd = kd_profile(seq)
    ac = autocorrelation(kd, max_lag)

    def at(lag: int) -> float:
        return round(ac[lag - 1], 4) if 1 <= lag <= len(ac) else 0.0

    # peak over the structurally meaningful range (skip lag 1: trivial neighbour correlation)
    if len(ac) >= min_lag:
        rng = list(range(min_lag, min(max_lag, len(ac)) + 1))
        peak_lag = max(rng, key=lambda k: ac[k - 1])
        peak_ac = round(ac[peak_lag - 1], 4)
    else:
        peak_lag, peak_ac = 0, 0.0

    lag7 = at(7)
    lag7_dominant = peak_ac > 0 and lag7 >= dominance * peak_ac
    heptad_periodic = lag7 >= heptad_thresh and lag7_dominant
    achievable = not heptad_periodic

    if heptad_periodic:
        reason = (f"heptad-periodic: lag7 autocorr {lag7:+.3f} >= {heptad_thresh} and is the dominant "
                  f"peak (peak lag {peak_lag}={peak_ac:+.3f}); a 7-shift reproduces the face -> "
                  f"register NOT scorable")
    else:
        reason = (f"not heptad-locked: lag7 autocorr {lag7:+.3f} (peak lag {peak_lag}={peak_ac:+.3f}); "
                  f"a register shift presents a different face -> register scorable")

    return {
        "seq": seq,
        "kd_profile_len": len(seq),
        "max_lag": max_lag,
        "autocorr": [round(v, 4) for v in ac],
        "autocorr_lag3": at(3),
        "autocorr_lag4": at(4),
        "autocorr_lag7": at(7),
        "peak_lag": peak_lag,
        "peak_autocorr": peak_ac,
        "heptad_thresh": heptad_thresh,
        "register_achievable": bool(achievable),
        "register_achievable_reason": reason,
    }


def _read_fasta(path: str, record: str | None) -> str:
    txt = Path(path).read_text()
    blocks = [b for b in txt.split(">") if b.strip()]
    if record is not None:
        for b in blocks:
            if record in b.splitlines()[0]:
                return "".join(b.splitlines()[1:]).strip()
        raise SystemExit(f"record {record!r} not found in {path}")
    # default: longest record (the designed binder convention in this repo)
    seqs = ["".join(b.splitlines()[1:]).strip() for b in blocks]
    return max(seqs, key=len)


# --------------------------------------------------------------------------- selfcheck
def _selfcheck() -> int:
    """Clean heptad (periodic, register NOT achievable) vs an irregular amphipath (achievable).

    PERIODIC: a textbook coiled-coil heptad (abcdefg)_n with the canonical 'a' and 'd' positions
    occupied by large hydrophobics (L/I/V) and b,c,e,f,g by polar/charged residues. Its hydropathy
    repeats every 7 residues, so lag-7 autocorrelation is large and dominant -> register_achievable
    must be FALSE.

    IRREGULAR: a scramble with no 7-periodic hydrophobic pattern (hydrophobics placed at aperiodic
    indices). Lag-7 autocorr is low / not dominant -> register_achievable must be TRUE.
    """
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    # heptad: positions a(0) and d(3) hydrophobic (L,I), rest polar/charged (E,K,Q,S,N) -> period 7
    heptad_unit = "LEKIQSN"  # a=L b=E c=K d=I e=Q f=S g=N
    periodic = heptad_unit * 6  # 42 aa, cleanly 7-periodic
    r_per = compute(periodic)
    check("periodic: lag7 autocorr >= heptad_thresh", r_per["autocorr_lag7"] >= r_per["heptad_thresh"],
          f"lag7={r_per['autocorr_lag7']}")
    check("periodic: peak_lag == 7", r_per["peak_lag"] == 7, f"peak_lag={r_per['peak_lag']}")
    check("periodic: register_achievable is False", r_per["register_achievable"] is False)

    # irregular: same composition-ish but hydrophobics at aperiodic positions, no 7-repeat
    irregular = "LEEKILESNKQELIKSEEQLNKILEESKNQELIKSEQLNKEIESLKQNE"
    r_irr = compute(irregular)
    check("irregular: lag7 autocorr < periodic lag7",
          r_irr["autocorr_lag7"] < r_per["autocorr_lag7"],
          f"{r_irr['autocorr_lag7']} vs {r_per['autocorr_lag7']}")
    check("irregular: register_achievable is True", r_irr["register_achievable"] is True,
          f"lag7={r_irr['autocorr_lag7']} peak_lag={r_irr['peak_lag']}")

    # A real solution binder (8GQP D-binder 62mer) should be register-ACHIEVABLE (lag7 +0.14)
    gqp = "LPVEKIIREAKKILDELLKRGLIDPELARIAREVLERARKLGNEEAARFVLELIERLRRELS"
    r_gqp = compute(gqp)
    check("8GQP binder: register_achievable is True", r_gqp["register_achievable"] is True,
          f"lag3={r_gqp['autocorr_lag3']} lag4={r_gqp['autocorr_lag4']} lag7={r_gqp['autocorr_lag7']} "
          f"peak_lag={r_gqp['peak_lag']}")

    print()
    for tag, r in (("PERIODIC", r_per), ("IRREGULAR", r_irr), ("8GQP", r_gqp)):
        print(f"{tag:9}", json.dumps({k: r[k] for k in
              ("autocorr_lag3", "autocorr_lag4", "autocorr_lag7", "peak_lag", "peak_autocorr",
               "register_achievable")}))
    print("\nSELFCHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description="Reference-free hydropathy-periodicity / register-achievability")
    p.add_argument("seq", nargs="?", help="binder amino-acid sequence (1-letter; case-insensitive)")
    p.add_argument("--fasta", help="read sequence from a FASTA instead of positional arg")
    p.add_argument("--record", help="FASTA record substring to pick (default: longest record)")
    p.add_argument("--max_lag", type=int, default=10)
    p.add_argument("--heptad_thresh", type=float, default=0.35,
                   help="lag-7 autocorr at/above which (when dominant) register is NOT achievable")
    p.add_argument("--out", default=None)
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args(argv)
    if a.selfcheck:
        return _selfcheck()
    if a.fasta:
        seq = _read_fasta(a.fasta, a.record)
    elif a.seq:
        seq = a.seq
    else:
        p.error("provide a SEQUENCE, --fasta, or --selfcheck")
    res = compute(seq, a.max_lag, a.heptad_thresh)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = main()
    sys.exit(r if isinstance(r, int) else 0)
