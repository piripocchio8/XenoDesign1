# xenodesign/metrics.py
"""Interface metrics for Chai-1 predictions, robust to Chai's atom-tokenization of
D/modified residues. Pure numpy; no GPU / chai import. (spec section 8, task #28)"""
from __future__ import annotations

import numpy as np

from xenodesign.mirror import D_TO_L as _D_TO_L


def aggregate_tokens_to_residues(pae, token_residue_index, token_asym_id, reduce="mean"):
    """Collapse the per-token PAE to a per-residue PAE matrix by grouping tokens that share
    (asym_id, residue_index) — needed because Chai atom-tokenizes D/modified residues.
    Returns (res_pae[n_res, n_res], res_asym_id[n_res]); residue order = first-seen."""
    pae = np.asarray(pae, dtype=float)
    aid = np.asarray(token_asym_id)
    rid = np.asarray(token_residue_index)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"pae must be square 2-D; got shape {pae.shape}")
    if len(aid) != pae.shape[0] or len(rid) != pae.shape[0]:
        raise ValueError("token_asym_id / token_residue_index length must match pae")
    keys = list(dict.fromkeys(zip(aid.tolist(), rid.tolist())))  # ordered unique (asym, resid)
    groups = [np.where((aid == a) & (rid == r))[0] for (a, r) in keys]
    if reduce not in ("mean", "min"):
        raise ValueError(f"reduce must be 'mean' or 'min', got {reduce!r}")
    reducer = np.mean if reduce == "mean" else np.min
    n = len(keys)
    R = np.zeros((n, n), dtype=float)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            R[i, j] = float(reducer(pae[np.ix_(gi, gj)]))
    return R, np.array([a for (a, _) in keys])


def _d0(n: int) -> float:
    n = max(int(n), 1)
    # d0 stays 1.0 until n≈22 (formula < 1 there); Zhang & Skolnick 2004 TM-score d0.
    return 1.0 if n <= 15 else max(1.0, 1.24 * ((n - 15) ** (1.0 / 3.0)) - 1.8)


def _ptm_term(pae: float, d0: float) -> float:
    return 1.0 / (1.0 + (pae / d0) ** 2)


def _ipsae_asym(res_pae, src, tgt, cutoff: float) -> float:
    best = 0.0
    for i in src:
        valid = [j for j in tgt if res_pae[i, j] < cutoff]
        if valid:
            d0 = _d0(len(valid))
            best = max(best, float(np.mean([_ptm_term(res_pae[i, j], d0) for j in valid])))
    return best


def ipsae(res_pae, res_asym_id, chain_a, chain_b, pae_cutoff: float = 10.0) -> float:
    """Residue-correct ipSAE (Dunbrack-style): pTM-form score restricted to inter-chain
    residue pairs with PAE < cutoff, d0 from the per-residue interface count; asymmetric ->
    max over both chain directions. Operates on a RESIDUE-level PAE (see
    aggregate_tokens_to_residues). Cross-check vs the Dunbrack ipsae.py reference.
    Returns 0.0 when NO inter-chain residue pair is below `pae_cutoff` (Dunbrack convention)
    — a 0.0 means either a true zero score or no valid pairs; callers wanting to distinguish
    should inspect the PAE.

    NOTE (task #32): this residue-AGGREGATED form collapses the d0 normalization for
    atom-tokenized D/modified residues — the per-residue valid-partner *count* is the number
    of residues (≤ n_res), so d0 stays clamped near 1.0 and every ptm-term shrinks. The
    canonical Dunbrack ipSAE counts per-TOKEN partners; use :func:`ipsae_token` for the
    Dunbrack-scale value. Kept here for backward compatibility / the residue-level unit tests."""
    aid = np.asarray(res_asym_id)
    A = np.where(aid == chain_a)[0]
    B = np.where(aid == chain_b)[0]
    return max(_ipsae_asym(res_pae, A, B, pae_cutoff), _ipsae_asym(res_pae, B, A, pae_cutoff))


def ipsae_token(pae, token_asym_id, chain_a, chain_b, pae_cutoff: float = 10.0) -> float:
    """Canonical Dunbrack ipSAE on the raw per-TOKEN PAE (the form the reference ipsae.py
    computes). For each source token i, take its inter-chain partner tokens j (PAE < cutoff),
    derive d0 from THAT per-token partner count, average the pTM-term over the valid partners,
    and report the best token; asymmetric -> max over both chain directions.

    This is the correct-scale ipSAE: because Chai atom-tokenizes D/modified residues, a token
    on the L-target can see many partner tokens across the atom-tokenized D-chain, so the
    valid-partner count (and hence d0) is large and the ptm-terms do not collapse the way they
    do in the residue-aggregated :func:`ipsae`. Chains are identified by `token_asym_id`, so
    the chain split is the TRUE asym boundary (not an atom-offset one).

    Returns 0.0 when no inter-chain token pair is below `pae_cutoff` (Dunbrack convention)."""
    pae = np.asarray(pae, dtype=float)
    aid = np.asarray(token_asym_id)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"pae must be square 2-D; got shape {pae.shape}")
    if len(aid) != pae.shape[0]:
        raise ValueError("token_asym_id length must match pae")
    A = np.where(aid == chain_a)[0]
    B = np.where(aid == chain_b)[0]
    return max(_ipsae_asym(pae, A, B, pae_cutoff), _ipsae_asym(pae, B, A, pae_cutoff))


def _parse_cif_ca(cif_path):
    """Yield (chain, resid, x, y, z, bfactor) for CA atoms — stdlib mmCIF _atom_site reader.
    Assumes Chai-style CIF: unquoted single-token fields, no multi-line (`;`) values."""
    col, in_site, collecting, ci = {}, False, False, 0
    with open(cif_path) as fh:
        for line in fh:
            s = line.strip()
            if s == "loop_":
                in_site = collecting = False; col = {}; ci = 0; continue
            if s.startswith("_atom_site."):
                if not collecting: collecting = True; in_site = True
                col[s.split(".", 1)[1].split()[0]] = ci; ci += 1; continue
            if in_site and collecting: collecting = False
            if in_site and not collecting and s and not s.startswith("_"):
                if s.startswith("#"): in_site = False; continue
                p = s.split()
                if len(p) < ci: continue
                def g(name, d="."):
                    i = col.get(name); return p[i] if i is not None and i < len(p) else d
                if g("group_PDB", "ATOM") not in ("ATOM", "HETATM"): continue
                if g("label_atom_id") != "CA": continue
                if g("type_symbol", "C").upper() != "C":   # exclude calcium ('CA'/element CA)
                    continue
                if g("label_alt_id", ".") not in (".", "A", ""):
                    continue
                try:
                    yield (g("label_asym_id"), g("label_seq_id"),
                           float(g("Cartn_x")), float(g("Cartn_y")), float(g("Cartn_z")),
                           float(g("B_iso_or_equiv", "0")))
                except ValueError:
                    continue


def interface_plddt_from_cif(cif_path, contact_dist: float = 10.0) -> dict:
    """Per chain: mean CA-pLDDT (B-factor) over all residues and over interface residues
    (CA within contact_dist of any other-chain CA). Chai writes pLDDT into the B-factor column."""
    rows = list(_parse_cif_ca(cif_path))
    chains: dict[str, list] = {}
    for ch, _resid, x, y, z, b in rows:
        chains.setdefault(ch, []).append((np.array([x, y, z]), b))
    out = {}
    for ch, residues in chains.items():
        others = np.array([p for oc, rs in chains.items() if oc != ch for (p, _b) in rs])
        bs = [b for (_p, b) in residues]
        iface = []
        for (p, b) in residues:
            if others.size and float(np.min(np.linalg.norm(others - p, axis=1))) < contact_dist:
                iface.append(b)
        out[ch] = {
            "n_residues": len(residues),
            "chain_plddt": float(np.mean(bs)) if bs else 0.0,
            "n_interface": len(iface),
            "interface_plddt": float(np.mean(iface)) if iface else 0.0,
        }
    return out


def interface_pae(pae, token_asym_id, chain_a, chain_b) -> dict:
    """ipAE / Bennett pae_interaction: mean & min PAE over the inter-chain token block
    (both directions). PAE is per-TOKEN; chains are identified by token_asym_id, so this
    is correct even when chain_a is atom-tokenized (D-residues)."""
    pae = np.asarray(pae, dtype=float)
    aid = np.asarray(token_asym_id)
    A = np.where(aid == chain_a)[0]
    B = np.where(aid == chain_b)[0]
    if A.size == 0 or B.size == 0:
        raise ValueError(f"empty chain: |A|={A.size} |B|={B.size}")
    # Chai PAE is asymmetric (source≠target) → pool A→B and B→A.
    inter = np.concatenate([pae[np.ix_(A, B)].ravel(), pae[np.ix_(B, A)].ravel()])
    return {"ipae_mean": float(inter.mean()), "ipae_min": float(inter.min())}


def token_maps_from_cif(cif_path):
    """Derive (token_asym_id, token_residue_index) from a Chai-style CIF.

    Chai atom-tokenizes modified / D residues: a residue whose CCD comp_id is in
    ``xenodesign.mirror.D_TO_L`` contributes one token per heavy atom; all other
    residues (standard L + GLY) contribute exactly 1 token.  Token order equals
    CIF residue order (first-seen per (label_asym_id, label_seq_id) pair).

    Returns
    -------
    token_asym_id : np.ndarray[int]
        Integer chain id for each token (first-seen chain → 0, 1, …).
    token_residue_index : np.ndarray[int]
        Residue index (0-based, global across chains) for each token.
    """
    Dcodes = set(_D_TO_L)

    # --- parse _atom_site, collecting (chain, resid, comp, n_heavy_atoms) ---
    col, in_site, collecting, ci = {}, False, False, 0
    residues: dict = {}   # (chain, resid) -> [comp, n_atoms]
    order: list = []      # first-seen (chain, resid) pairs, CIF order

    with open(cif_path) as fh:
        for line in fh:
            s = line.strip()
            if s == "loop_":
                in_site = collecting = False; col = {}; ci = 0; continue
            if s.startswith("_atom_site."):
                if not collecting:
                    collecting = True; in_site = True
                col[s.split(".", 1)[1].split()[0]] = ci; ci += 1; continue
            if in_site and collecting:
                collecting = False
            if in_site and not collecting and s and not s.startswith("_"):
                if s.startswith("#"):
                    in_site = False; continue
                p = s.split()
                if len(p) < ci:
                    continue
                def g(name, d="."):
                    i = col.get(name); return p[i] if i is not None and i < len(p) else d
                if g("group_PDB", "ATOM") not in ("ATOM", "HETATM"):
                    continue
                chain = g("label_asym_id")
                resid = g("label_seq_id")
                comp  = g("label_comp_id")
                key = (chain, resid)
                if key not in residues:
                    residues[key] = [comp, 0]
                    order.append(key)
                residues[key][1] += 1   # count every atom (CIF is heavy-atom only)

    # --- build token arrays ---
    chain_ids: dict = {}   # chain label -> integer id (first-seen)
    asym:  list = []
    residx: list = []
    ridx = 0
    for key in order:
        chain, _resid = key
        comp, n_atoms = residues[key]
        a = chain_ids.setdefault(chain, len(chain_ids))
        ntok = n_atoms if comp in Dcodes else 1
        asym  += [a]    * ntok
        residx += [ridx] * ntok
        ridx += 1

    return np.asarray(asym, dtype=int), np.asarray(residx, dtype=int)


def load_confidence(npz_path) -> dict:
    """Load a saved confidence npz (pae [+ token_asym_id, token_residue_index if present])."""
    d = np.load(npz_path)
    out = {"pae": np.asarray(d["pae"], dtype=float)}
    for k in ("token_asym_id", "token_residue_index", "plddt", "pde"):
        if k in d.files:
            out[k] = np.asarray(d[k])
    return out


def score_interface(npz_path, cif_path, chain_a=0, chain_b=1, pae_cutoff=10.0) -> dict:
    """Full interface metric bundle for one model: ipAE + residue-correct ipSAE
    (+ interface pLDDT from the CIF).

    Token maps (token_asym_id, token_residue_index) are read from the npz when
    present.  When absent (e.g. raw Chai outputs whose npz only contains 'pae'),
    they are derived from the CIF via :func:`token_maps_from_cif`.  A
    ``ValueError`` is raised if the derived token count does not match the PAE
    matrix dimension (prevents silent dimension mismatches).
    """
    c = load_confidence(npz_path)
    aid, rid = c.get("token_asym_id"), c.get("token_residue_index")
    if aid is None or rid is None:
        aid, rid = token_maps_from_cif(cif_path)
        n_tok = len(aid)
        pae_dim = c["pae"].shape[0]
        if n_tok != pae_dim:
            raise ValueError(
                f"CIF-derived token count ({n_tok}) does not match PAE dimension "
                f"({pae_dim}); check that cif_path corresponds to this npz."
            )
    res_pae, res_asym = aggregate_tokens_to_residues(c["pae"], rid, aid, reduce="mean")
    m = interface_pae(c["pae"], aid, chain_a, chain_b)
    # Canonical Dunbrack ipSAE on the raw per-TOKEN PAE (task #32): cut10 + cut15. These are the
    # Dunbrack-scale values (per-token partner count drives d0, so the score does not collapse the
    # way the residue-aggregated form does). 'ipsae' is kept as an alias of 'ipsae_cut10'.
    m["ipsae_cut10"] = ipsae_token(c["pae"], aid, chain_a, chain_b, pae_cutoff=10.0)
    m["ipsae_cut15"] = ipsae_token(c["pae"], aid, chain_a, chain_b, pae_cutoff=15.0)
    m["ipsae"] = ipsae_token(c["pae"], aid, chain_a, chain_b, pae_cutoff=pae_cutoff)
    # Residue-aggregated form (the previous 'ipsae'); kept for provenance/debugging.
    m["ipsae_resagg"] = ipsae(res_pae, res_asym, chain_a, chain_b, pae_cutoff)
    for ch, vals in interface_plddt_from_cif(cif_path).items():
        for k, v in vals.items():
            m[f"chain{ch}_{k}"] = v
    return m
