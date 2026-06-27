"""CYCLIC 6UFA binder class — single-chain Zn-macrocycle design + geometry RECALL (T7).

Migrated from ``scripts/design_cyclic.py`` (which is now a thin re-export shim). This
module owns the cyclic-class logic: the mixed-chirality seed (coordinating His pinned
L/D), the Zn-ligand FASTA emission, the His<->Zn metal-coordination restraints, the
opt-in head-to-tail COVALENT closure bond, and the RMSD-to-deposit / Zn-N geometry
scorers — plus the :class:`Cyclic` :class:`~xenodesign.classes.base.BinderClass`
adapter the dispatcher (T3) wires into the untouched :class:`HalluLoop`.

PRIMARY cyclization = MAINCHAIN head-to-tail closure (a real N-to-C backbone COVALENT
bond, consumed by chai's bond_utils, NOT a soft distance restraint). It is opt-in via
``restraint.params['closure']=True``; the default run is LINEAR + emergent-closure and
relies on the His<->Zn restraints + the deposit's intrinsic ring geometry, reporting the
N/C-terminus distance as a closure proxy. Disulfide closure is SECONDARY/not used for
this case (the cyclic 6UFA site is metal-coordinated, not disulfide-bonded).

METAL GATE (T8 merged): the cyclic target is ``metal`` (Zn ligand + His<->Zn
metal_coordination restraints). The token_dist_restraint repair lives in
``xenodesign.chai_patches`` and is now AVAILABLE, so the metallo-cyclic coordination
restraints can use the repaired ``token_dist`` path on GPU. ``targets.target_entities``
consults ``chai_patches.dist_restraint_patch_verified()`` as the metal gate; the
dispatcher applies the patch before any constrained predict. CPU tests here exercise
only seed/closure/restraint-builder/objective/report (never a predict).

6UFA DEPOSIT REALITY (verified on the RCSB mmCIF; load-bearing for interpreting results)
----------------------------------------------------------------------------------------
The 6UFA deposit is a SINGLE chain A of 24 residues = the 12-mer repeat unit TWICE.
One Zn(II) is chelated tetrahedrally by FOUR His ND1 atoms — residues 6, 12, 18, 24 —
i.e. the L-His (pos 6) and D-His (pos 12) of EACH repeat: [Zn(L-His)2(D-His)2], Zn-N
~2.02 A. The registry/seeding model the FULL 24-mer: his_resnums (6, 12, 18, 24),
_CYCLIC_HIS_CHIRALITY {6:'L', 12:'D', 18:'L', 24:'D'} — all FOUR coordinating His, the
true 4-coordinate site. (A single 12-mer carries only 2 His and CANNOT make the site.)

SEQUENCE APPROXIMATION (phase-1): the design seed is a canonical-residue stand-in for the
deposit's AIB / D-Gln / D-Glu / D-Lys / D-Leu macrocycle. We pin only the coordinating His
handedness (the chemistry that defines the site); the rest is designed/approximated.
"""
from __future__ import annotations

# Names the Cyclic adapter (below) references at module scope. SeedSpec is imported LAZILY inside
# the hooks (not here) to avoid the import cycle base -> cyclic -> base (base imports Cyclic from
# this module): the same pattern Alpha uses. The annotation ``-> SeedSpec`` is a string under
# ``from __future__ import annotations`` so it needs no module-level binding.
from xenodesign.benchmark.cases import get_case
from xenodesign.seed import SeedResult

# MOD-3 split: the seed / mixed-chirality / Zn-restraint / geometry / intramolecular-objective /
# result-assembly INTERNALS live in ``_cyclic_internals`` (incl. the track-#1 coordinator-masking,
# Gly-anchor, provenance and metal-verify code). Re-export every name here so existing imports
# (``from xenodesign.classes.cyclic import X``) keep working, AND so the call-time
# ``_self().metal_geometry_gate`` lookup in ``_cyclic_internals`` resolves against THIS module
# (the monkeypatch surface the cyclic CPU tests patch).
from xenodesign.classes._cyclic_internals import (  # noqa: F401
    CYCLIC_HIS_CHIRALITY,
    INTRAMOLECULAR_WEIGHTS,
    ZN_SMILES,
    _ANGLE_TOL_DEG,
    _BACKBONE_ATOMS,
    _DEFAULT_DEVICE,
    _IDEAL_N_CA_C,
    _ZN_N_CUTOFF,
    _assemble_cyclic_result,
    _best_step,
    _chirality_term,
    _geometry_term,
    _mainchain_plddt_term,
    _self,
    backbone_heavy_atoms_from_cif,
    backbone_rmsd_to_deposit,
    build_closure_row,
    build_cyclic_input_fasta,
    build_cyclic_restraint_rows,
    build_cyclic_seed,
    combine_intramolecular_terms,
    cyclic_records_from_cif,
    head_to_tail_closure_geometry_from_cif,
    intramolecular_terms_from_records,
    make_intramolecular_score_fn,
    metal_geometry_gate,
    mixed_chirality_fasta,
    termini_distance_from_cif,
    write_cyclic_restraints,
    zn_and_his_nitrogens_from_cif,
    zn_coordination_geometry,
)


# ── BinderClass adapter ──────────────────────────────────────────────────────────

class Cyclic:
    """Cyclic Zn-macrocycle binder class (single-chain, no interface; geometry RECALL).

    Implements the :class:`~xenodesign.classes.base.BinderClass` protocol. The dispatcher
    wires these hooks into the untouched HalluLoop: ``seed`` pins the coordinating His L/D,
    ``restraints`` writes the His<->Zn metal_coordination CSV (+ opt-in mainchain closure),
    ``objective`` drives on ipTM/pTM, and ``report`` assembles the RECALL result dict.
    """

    case_id = "cyclic"

    def seed(self, cfg, target_seq) -> SeedSpec:
        """FROM-SCRATCH unified UNCONDITIONAL seed (PepMLM cannot condition on a metal).

        Routes through the ONE ``seed.unified_seed`` path with ``target_seq=""`` (no protein
        target) at ``resolve_binder_length(cfg)`` (default 24 = the S2-symmetric 6UFA full length;
        16 for ``target_type='none'``; overridable via --binder_length). The seed NEVER inherits
        the deposit's sequence or length. The coordinating-His placement is OPT-IN: only when
        restraints are ON do we pin the case's ``his_resnums`` (with their L/D handedness) as fixed
        positions; otherwise the seed carries no mandatory His scaffold."""
        from xenodesign.classes.base import SeedSpec
        from xenodesign.config import resolve_binder_length
        from xenodesign.seed import make_configured_generator, unified_seed

        length = resolve_binder_length(cfg)
        gen = make_configured_generator(cfg)
        fixed = self._his_positions(cfg, length) if cfg.restraints_on else None
        result = unified_seed(gen, target_seq="", length=length, reverse=False,
                              fixed_positions=fixed, fixed_residue="H")
        one = self._place_declared_residues(cfg, result.one_letter, length)
        one = self._ensure_canonical_anchor(one, fixed)
        return SeedSpec(one_letter=one,
                        fixed_chirality=dict(result.fixed_chirality))

    def ss_bias(self, cfg, case):
        from xenodesign.benchmark.cases import ss_bias_config_for_case
        return ss_bias_config_for_case(case)  # anti_alpha -> 0.0

    def restraints(self, cfg, case, out_dir, target_ctx):
        if not cfg.restraints_on:
            return None
        # NO-TARGET (target_type='none'): there is NO Zn chain, so SKIP the His<->Zn metal
        # coordination rows; the binder is the sole chain A and only the opt-in head-to-tail
        # closure row (on chain A) is written. The discriminator is the TARGET TYPE, not the
        # entity list — the metal case legitimately passes target_ctx=None (legacy standalone
        # order) yet still needs the coordination rows.
        metal = cfg.target.target_type != "none"
        closure = bool(cfg.restraint.params.get("closure"))
        seed_result = self._seed_result(cfg, case) if closure else None

        if not metal:
            # Free peptide: binder=A, no Zn. Closure (if any) is on chain A.
            if not closure:
                return None   # no Zn + no closure -> unconstrained free peptide
            return write_cyclic_restraints(case, out_dir, seed_result=seed_result,
                                           closure=closure, binder_chain="A", zn_chain="B",
                                           metal=False)

        # Metal case: the wrapper appends the binder LAST, so when the assembled entity list is
        # given (target_ctx with the Zn ligand) the Zn is chain A and the peptide/His is the last
        # chain; no ctx -> legacy standalone order (peptide=A, Zn=B). Read the chains from the ONE
        # chain contract (``ChainRoles.from_entities``) instead of an inline chr(...), so this site
        # uses the SAME derivation as the seq-update extractor and the double-flip (the dispatcher
        # gap that this whole contract closes).
        entities = (target_ctx[0] if target_ctx else None) or []
        if entities:
            from xenodesign.targets import ChainRoles
            roles = ChainRoles.from_entities(entities)
            binder_chain = roles.binder      # Zn=A -> binder=B
            zn_chain = roles.targets[0]      # the Zn ligand chain (A)
        else:
            binder_chain, zn_chain = "A", "B"
        return write_cyclic_restraints(case, out_dir, seed_result=seed_result,
                                       closure=closure, binder_chain=binder_chain,
                                       zn_chain=zn_chain, metal=True,
                                       coord_residues=self._coord_residues(cfg))

    def closure(self, cfg, seed_spec) -> list:
        """PRIMARY cyclization: mainchain head-to-tail COVALENT closure (opt-in).

        Returns ``[]`` for the default LINEAR + emergent-closure run; when
        ``restraint.params['closure']`` is set, returns a single head-to-tail COVALENT
        backbone bond row (N-to-C ring bond) built from the seed's termini. Disulfide
        closure is secondary and not used for the metal-coordinated 6UFA site."""
        if not cfg.restraint.params.get("closure"):
            return []
        sr = SeedResult(one_letter=seed_spec.one_letter,
                        length=len(seed_spec.one_letter),
                        reverse_applied=False, conditioned=False,
                        fixed_chirality=dict(seed_spec.fixed_chirality))
        return [build_closure_row(sr)]

    def seq_update(self, cfg, wrapper, seed_spec, roles=None):
        """Per-iteration sequence-update fn. The cyclic recall case keeps the pinned seed;
        the dispatcher's fallback (identity to the seed one-letter) is sufficient on CPU, and
        the GPU path re-designs the non-coordinating backbone via the wrapper's MPNN.

        ``roles`` threads the dispatch chain contract: metal case (Zn=A) -> binder 'B';
        no-target free peptide -> binder/context BOTH 'A' (the bug that crashed iter_000)."""
        from xenodesign.classes.alpha import make_alpha_seq_update_fn
        # Freeze declared coordinators in the MPNN mask so pinned donors (e.g. His) never
        # drift. coord_residues tuple[0] is the 1-based position -> 0-based for the mask.
        frozen_positions = {int(t[0]) - 1 for t in self._coord_residues(cfg)}
        return make_alpha_seq_update_fn(wrapper, num_seqs=cfg.loop.num_seqs,
                                        backend=cfg.loop.backend, roles=roles,
                                        frozen_positions=frozen_positions or None)

    def accept_fns(self, cfg):
        from xenodesign.loop import compose_accept_fns
        return compose_accept_fns(None)

    def objective(self, cfg, wrapper):
        # NO-TARGET (target_type='none'): single free cyclic/linear peptide -> ipTM/binder-chain
        # index are undefined, so use the INTRAMOLECULAR 4-term objective (mainchain-pLDDT of the
        # cyclising termini + chirality goodness + closure/backbone geometry + pTM).
        if cfg.target.target_type == "none":
            return make_intramolecular_score_fn(wrapper)
        # Metal case: share alpha's ipTM + binder-chain pLDDT objective (DRY — was a byte-identical
        # local copy; cyclic already reuses alpha's seq-update). Lazy import mirrors seq_update.
        from xenodesign.classes.alpha import _loop_score_fn
        return _loop_score_fn  # recall case: ipTM/pTM drive; geometry scored in report

    def referee(self, cfg, loop_dir, esm_judge, roles=None):
        return lambda step, i: None  # no per-step referee for the recall case (no chain read)

    def report(self, cfg, history, panel_result, case, out_dir,
               *, l_seed_iptm: float = 0.0, wall_time_s: float = 0.0) -> dict:
        return _assemble_cyclic_result(cfg, history, panel_result, case, out_dir,
                                       l_seed_iptm=l_seed_iptm, wall_time_s=wall_time_s)

    # ── internal ────────────────────────────────────────────────────────────────

    def _coord_residues(self, cfg):
        """The DECLARATIVE coordinator list from ``cfg.restraint.params['coord_residues']``.

        Each entry is a (pos, one_letter, three_letter, chirality) tuple (as stored by the CLI
        flag wiring). Returns [] when the flag was absent — callers then fall back to the case's
        hardcoded His defaults."""
        params = cfg.restraint.params if cfg.restraint else {}
        return list(params.get("coord_residues") or [])

    def _his_positions(self, cfg, length) -> dict:
        """OPT-IN coordinating positions+chirality for the from-scratch cyclic seed.

        DECLARATIVE override (``--coord_residues``): when ``cfg.restraint.params['coord_residues']``
        is set, those (pos, chirality) pairs ARE the fixed positions (generalizing beyond His/Zn —
        any donor, any chirality). Absent -> the case's ``metal_coordination`` ``his_resnums`` +
        ``CYCLIC_HIS_CHIRALITY`` defaults. Positions outside the from-scratch ``length`` are
        dropped. Returns {} when neither source applies. NEVER mandatory: only consulted when
        restraints are ON."""
        coords = self._coord_residues(cfg)
        if coords:
            # Tuple is (pos, one_letter, three_letter, chirality[, atom]); index for back-compat.
            return {int(t[0]): t[3]
                    for t in coords if 1 <= int(t[0]) <= length}
        case = get_case("cyclic")
        spec = case.restraint
        if spec is None or spec.kind != "metal_coordination":
            return {}
        resnums = spec.params.get("his_resnums", ())
        return {int(p): CYCLIC_HIS_CHIRALITY.get(int(p), "L")
                for p in resnums if 1 <= int(p) <= length}

    def _seed_result(self, cfg, case):
        """Rebuild the cyclic SeedResult (for closure-row construction at restraint time).

        Mirrors :meth:`seed` exactly (unified UNCONDITIONAL from-scratch path + opt-in His), so the
        closure/restraint rows reference the same seed the loop starts from."""
        from xenodesign.config import resolve_binder_length
        from xenodesign.seed import make_configured_generator, unified_seed

        length = resolve_binder_length(cfg)
        gen = make_configured_generator(cfg)
        fixed = self._his_positions(cfg, length) if cfg.restraints_on else None
        result = unified_seed(gen, target_seq="", length=length, reverse=False,
                              fixed_positions=fixed, fixed_residue="H")
        one = self._place_declared_residues(cfg, result.one_letter, length)
        one = self._ensure_canonical_anchor(one, fixed)
        return SeedResult(one_letter=one, length=result.length,
                          reverse_applied=result.reverse_applied,
                          conditioned=result.conditioned,
                          fixed_chirality=dict(result.fixed_chirality))

    def _place_declared_residues(self, cfg, one_letter: str, length: int) -> str:
        """Overwrite the DECLARED coordinator positions with their REAL one-letter identities.

        ``unified_seed`` places a single ``fixed_residue`` ('H') at the fixed positions; the
        declarative ``--coord_residues`` flag generalizes that to ANY donor (His 'H', Cys 'C',
        Asp 'D', ...), so when coordinators are declared (and restraints ON) we overwrite each
        declared position with its real one-letter code. No-op when the flag is absent (the
        'H' His default already placed by unified_seed stands)."""
        if not cfg.restraints_on:
            return one_letter
        chars = list(one_letter)
        for t in self._coord_residues(cfg):
            # Tuple is (pos, one_letter, three_letter, chirality[, atom]); index for back-compat.
            pos, ol = int(t[0]), t[1]
            if 1 <= pos <= length:
                chars[pos - 1] = ol
        return "".join(chars)

    @staticmethod
    def _ensure_canonical_anchor(one_letter: str, fixed: "dict | None",
                                 chirality_map: "dict | None" = None,
                                 *, default_chirality: str = "D") -> str:
        """Gly-guard the from-scratch cyclic seed when it is otherwise all-one-handedness (#9, Part E).

        chai needs >=1 canonical residue per chain to tokenize; a chain that is ENTIRELY one
        handedness (e.g. all-D) is fully non-canonical and crashes at iter_000 (ADR-004). A
        genuinely MIXED L/D design already carries tokenizable L residues, so it needs no anchor.

        Rule (Part E): ensure an achiral Gly when, among the NON-coordinator positions, there is
        NO D-residue OR NO L-residue present (i.e. the design is otherwise all-one-handedness) AND
        no Gly already exists. The Gly is placed at the C-TERMINUS (last non-coordinator position),
        never overwriting a declared coordinator. ``fixed`` keys are the 1-based coordinator
        positions; non-coordinator handedness is read from ``chirality_map`` (positions absent
        default to ``default_chirality`` — the cyclic backbone is encoded uniformly all-D in the
        no-target path, L in the mixed-metal path)."""
        if "G" in one_letter:
            return one_letter
        n = len(one_letter)
        pinned = {int(p) for p in (fixed or {})}  # 1-based coordinator positions
        chir = dict(chirality_map or {})
        non_coord = [i for i in range(n) if (i + 1) not in pinned]
        hands = {chir.get(i + 1, default_chirality) for i in non_coord}
        # Mixed (both 'D' and 'L' present among non-coordinators): chai can tokenize the L
        # residues, so no forced Gly is needed.
        if "D" in hands and "L" in hands:
            return one_letter
        # All-one-handedness: place the achiral anchor at the C-TERMINAL non-coordinator position.
        if non_coord:
            i = non_coord[-1]
            return one_letter[:i] + "G" + one_letter[i + 1:]
        return one_letter  # pragma: no cover (every position pinned — impossible for real lengths)
