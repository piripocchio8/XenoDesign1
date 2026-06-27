"""target_entities(cfg) — build the FIXED-context chai entity list per target chemistry (T4).

Subsumes the ex-"Goal-3 Route A" multi-chain + MSA'd receptor target: a protein target with
>1 chain reuses the verified ``design_nonalpha.load_ha_entities`` builder (byte-identical HA1/HA2
sequences keyed to the cached MSA) so ``non_alpha`` can target a 2-chain MSA'd receptor. The
binder chain is NOT built here (the binder-class hooks add it); this only assembles the target.

Returns ``(entities, msa_dir, restraint_hint)``:
  * entities      — chai entity dicts for the target chains/ligands (binder appended downstream).
  * msa_dir       — the cached MSA directory (protein + ``target.msa``) or None.
  * restraint_hint — a RestraintConfig.kind hint the class may honour ('metal_coordination'), else None.

The ``metal`` target is GATED (spec §6): it refuses unless the ``token_dist_restraint`` patch
(``chai_patches._patch_dist_restraint_match``) is verified-applied in-process — the coordinator
D/non-canonical residue match otherwise silently drops the coordination restraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Re-exported for reuse + monkeypatchability (the 2-chain MSA'd HA target builder).
from scripts.design_nonalpha import (  # noqa: F401
    load_ha_entities, _DEFAULT_MSA_DIR, _DEFAULT_HA_FASTA,
)


@dataclass(frozen=True)
class ChainRoles:
    """The ONE authoritative binder/target chain-letter assignment for a dispatch run.

    Chai labels chains A, B, C… by ENTITY ORDER, and the binder-class wrapper appends the
    binder entity LAST. So from the assembled target ``entities`` list (everything BEFORE the
    binder) the chain letters are fully determined — no consumer needs to guess. Build this
    ONCE at entity assembly (``from_entities``) and THREAD it to every site that would
    otherwise hardcode a chain letter (the seq-update extractor, the double-flip reflection,
    restraint/metric builders). This makes chain-misidentification structurally impossible for
    ANY binder class / target chemistry.

    Witnessed cases:
      * alpha (1 protein target)      -> targets=('A',),       binder='B'
      * non_alpha 2-chain HA target   -> targets=('A','B'),    binder='C'
      * cyclic metal (Zn ligand)      -> targets=('A',),       binder='B'
      * no-target cyclic (free)       -> targets=(),           binder='A'
    """

    binder: str
    targets: tuple[str, ...]

    @classmethod
    def from_entities(cls, entities) -> "ChainRoles":
        """Derive the contract from the assembled TARGET entity list (binder NOT included).

        ``binder = chr('A' + len(entities))`` (binder appended last); the targets occupy the
        leading ``len(entities)`` letters.
        """
        n = len(entities or [])
        targets = tuple(chr(ord("A") + i) for i in range(n))
        binder = chr(ord("A") + n)
        return cls(binder=binder, targets=targets)

    @property
    def context(self) -> str:
        """The chain whose all-atom coords seed the inverse-folding CONTEXT.

        The first target chain when a target exists; for the no-target case (binder is the sole
        chain A) the binder IS its own context, so this collapses to the binder chain. Either
        way it is derived from the contract, never hardcoded.
        """
        return self.targets[0] if self.targets else self.binder


def _case_for(cfg):
    """The benchmark case backing this run's binder class (for default-target fallback).

    Maps the CLI ``binder_class`` axis to its registered ``BinderClass.case_id`` and returns
    ``get_case(case_id)`` — so a run with an EMPTY ``target.fasta_path`` falls back to the same
    validated case target the legacy single-class drivers (``run_alpha_design`` /
    ``run_nonalpha_design``) use, instead of reading ``Path("")`` (the IsADirectoryError gap).
    """
    from xenodesign.benchmark.cases import get_case
    cls = _registry()[cfg.binder_class]
    return get_case(cls.case_id)


def _registry():
    from xenodesign.classes.base import CLASS_REGISTRY
    return CLASS_REGISTRY


def _alpha_target_record(cfg):
    """The FASTA record-name to select from the case default target, or None for the first.

    The α case FASTA holds BOTH the binder (record A) and the fixed L-HLH target (record B);
    ``run_alpha_design`` reads the target by name ``alpha._TARGET_RECORD`` ("trimer_DL_ABLE_B").
    Other classes' default target FASTAs are single-record, so None (first record) is right.
    """
    if cfg.binder_class == "alpha":
        from xenodesign.classes.alpha import _TARGET_RECORD
        return _TARGET_RECORD
    return None


def _read_fasta_records(path):
    recs, name, seq = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                recs[name] = "".join(seq)
            name, seq = line[1:].split("|")[0].strip(), []
        elif line.strip():
            seq.append(line.strip())
    if name is not None:
        recs[name] = "".join(seq)
    return recs


def _metal_patch_verified() -> bool:
    """True once the metal coordination dist-restraint patch is active (gates ``metal`` open)."""
    from xenodesign.chai_patches import dist_restraint_patch_verified
    return dist_restraint_patch_verified()


# METAL-(b): bare-metal-cation SMILES -> CCD residue code. A metal fed as a CCD residue tokenizes
# to a resolvable residue+atom (e.g. ZN -> atom ZN) for atom-aware coordination, whereas the SMILES
# form gives residue 'LIG' / atom 'ZN1' (unresolvable). Codes match chai's conformer cache.
_METAL_SMILES_TO_CCD = {
    "[ZN+2]": "ZN", "[FE+2]": "FE", "[FE+3]": "FE", "[CU+2]": "CU", "[CU+]": "CU",
    "[NI+2]": "NI", "[MN+2]": "MN", "[CO+2]": "CO", "[MG+2]": "MG", "[CA+2]": "CA",
}


def _metal_ligand_entity(target):
    """Build the metal ligand entity dict for the ``metal`` target (METAL-(b)).

    Prefers a CCD residue (``metal_ccd``) so the metal tokenizes to a resolvable residue+atom:
      * an explicit ``target.ccd`` is used verbatim as the CCD code;
      * else a recognised bare-metal-cation ``target.smiles`` (e.g. '[Zn+2]') maps to its CCD code.
    Falls back to the SMILES path only for a non-CCD metal/SMILES (no mapping, no ccd)."""
    lig = {"type": "ligand", "name": "zn"}
    if target.ccd:
        lig["metal_ccd"] = str(target.ccd).upper()
        return lig
    code = _METAL_SMILES_TO_CCD.get(str(target.smiles).upper().strip())
    if code:
        lig["metal_ccd"] = code
        return lig
    # Non-CCD metal / arbitrary SMILES: keep the SMILES path (still a valid ligand, just not
    # atom-resolvable for coordination — documented fallback).
    lig["smiles"] = target.smiles
    return lig


def target_entities(cfg):
    """Return (entities, msa_dir, restraint_hint) for the FIXED context (binder added later)."""
    t = cfg.target
    if t.target_type == "none":
        # Binder-only (free cyclic/linear peptide): NO target entity. The loop wrapper appends
        # the binder as the sole chain (chain A), and the class uses an INTRAMOLECULAR objective
        # (no ipTM / no interface). No MSA, no restraint hint.
        return ([], None, None)
    if t.target_type == "protein":
        # Multi-chain MSA'd HA receptor (ex-Goal-3 Route A). Taken when the preset asks for >1
        # chain OR — the non_alpha DEFAULT — when an MSA'd target is requested without an explicit
        # FASTA (its preset has msa=True, chains=()). Falls back to the case's built-in HA fasta +
        # cached MSA dir, mirroring run_nonalpha_design's load_ha_entities(_DEFAULT_HA_FASTA).
        if len(t.chains) > 1 or (t.msa and not t.fasta_path and not t.chains):
            ha_fasta = t.fasta_path or _DEFAULT_HA_FASTA
            ents = load_ha_entities(ha_fasta)
            msa_dir = (t.msa_dir or _DEFAULT_MSA_DIR) if t.msa else None
            return ents, msa_dir, None
        if t.fasta_path:
            recs = _read_fasta_records(t.fasta_path)
            name = t.chains[0] if t.chains else next(iter(recs))
            seq = recs[name]
        else:
            # No explicit FASTA → fall back to the validated case default target record (the
            # gap T10 found: design.py with no --fasta must reproduce run_alpha_design's target).
            from xenodesign.seed import read_target_sequence
            case = _case_for(cfg)
            record = _alpha_target_record(cfg)
            seq = read_target_sequence(case.fasta_path, name=record)
            name = record or "target"
        return ([{"type": "protein", "name": name, "sequence": seq,
                  "chirality": "L"}],
                (t.msa_dir or None) if t.msa else None, None)
    if t.target_type in ("rna", "dna"):
        recs = _read_fasta_records(t.fasta_path)
        name = t.chains[0] if t.chains else next(iter(recs))
        return [{"type": t.target_type, "name": name, "sequence": recs[name]}], None, None
    if t.target_type == "small_molecule":
        lig = {"type": "ligand", "name": "lig"}
        lig["smiles" if t.smiles else "ccd"] = t.smiles or t.ccd
        return [lig], None, None
    if t.target_type == "metal":
        # The dist-restraint patch is only needed when coordination restraints are actually emitted.
        # An UNGUIDED metal run (--no_restraints => restraints_on=False) applies no coordination
        # restraints, so the patch gate is irrelevant — skip it and let Chai hallucinate freely.
        if cfg.restraints_on and not _metal_patch_verified():
            raise RuntimeError(
                "metal target_type requires the token_dist_restraint patch "
                "(chai_patches._patch_dist_restraint_match) verified-applied on the metal probe; "
                "coordinator D/non-canonical residue match currently drops the restraint.")
        return [_metal_ligand_entity(t)], None, "metal_coordination"
    raise ValueError(f"unknown target_type {t.target_type!r}")
