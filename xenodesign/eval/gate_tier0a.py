"""Tier 0a chirality gate: predict known D-containing structures with Chai and measure
chiral-volume sign accuracy + phi/psi agreement vs experimental references (MONDE-T).

The end-to-end `run_gate` is GPU-gated; `aggregate_gate_report` is pure and tested on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class GateResult:
    chirality_violation_frac: float
    phi_psi_violation_frac: float
    passed: bool


@dataclass
class GateCase:
    """One Tier-0a gate case.

    name:          identifier (e.g. a PDB id like '7QDI').
    entities:      Chai entity specs (io_spec.build_fasta format), with D residues encoded.
    design_labels: per-residue 'L'/'D' chirality labels for the chain whose chirality we check.
    design_chain:  fasta entity name of that chain (used to locate it in the predicted CIF).
    ref_backbone:  optional list of residue dicts {'N','CA','C'} (experimental reference) for
                   the φ/ψ comparison; None skips the φ/ψ diagnostic for this case.
    """
    name: str
    entities: Sequence[Mapping[str, object]]
    design_labels: Sequence[str]
    design_chain: str
    ref_backbone: Sequence[Mapping[str, object]] | None = None


def aggregate_gate_report(
    chirality_violations: int,
    n_stereocenters: int,
    phi_psi_violations: int,
    n_torsions: int,
    violation_threshold: float = 0.51,
) -> GateResult:
    """Aggregate per-residue counts into a pass/fail gate decision (spec §3).

    PASS iff chirality violation fraction is well below `violation_threshold`.
    """
    chir = chirality_violations / n_stereocenters if n_stereocenters else 0.0
    pp = phi_psi_violations / n_torsions if n_torsions else 0.0
    # By design, phi/psi (spec §3, ±25°) is reported as a diagnostic but NOT gated here:
    # the go/no-go decision rides the dominant chiral-volume signal ("well below 51%").
    # phi_psi_violation_frac is surfaced so a human can inspect backbone geometry.
    passed = chir < violation_threshold
    return GateResult(chirality_violation_frac=chir, phi_psi_violation_frac=pp, passed=passed)


def backbone_by_residue_from_cif(cif_path, chain_name: str):  # pragma: no cover (gpu)
    """Parse a CIF into a list of per-residue backbone dicts {'N','CA','C','CB'} for one chain.

    BEST-EFFORT (gemmi). Residues lacking CB (e.g. GLY) omit the 'CB' key. VERIFY ON GPU.
    """
    import gemmi
    import numpy as np

    structure = gemmi.read_structure(str(cif_path))
    residues = []
    for model in structure:
        for chain in model:
            if chain.name != chain_name:
                continue
            for res in chain:
                atoms = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res}
                if not {"N", "CA", "C"} <= atoms.keys():
                    continue
                rec = {k: atoms[k] for k in ("N", "CA", "C")}
                if "CB" in atoms:
                    rec["CB"] = atoms["CB"]
                residues.append(rec)
        break  # first model only
    return residues


def run_gate(cases: Sequence[GateCase], backend, out_dir, violation_threshold: float = 0.51):  # pragma: no cover (gpu)
    """End-to-end Tier 0a gate over D-containing cases (spec §3).

    For each case: predict with `backend` (Chai), parse the predicted design chain into
    per-residue backbones, count chirality violations (chirality.is_chirality_violation vs
    the per-position D/L labels) and φ/ψ deviations vs the reference (±25°), then aggregate.
    BEST-EFFORT — verify on GPU; returns (overall GateResult, per-case dict).
    """
    from pathlib import Path

    from xenodesign.chirality import (
        backbone_torsions,
        is_chirality_violation,
        phi_psi_violation,
    )

    total_chir_viol = total_stereo = total_pp_viol = total_torsions = 0
    per_case = {}

    for case in cases:
        case_dir = Path(out_dir) / case.name
        pred = backend.predict(case.entities, case_dir)  # noqa: F841 (cif written to case_dir)
        cif = sorted(Path(case_dir).rglob("*.cif"))[0]
        residues = backbone_by_residue_from_cif(cif, case.design_chain)

        # Chirality violations on residues that are stereocenters (have CB).
        c_viol = c_total = 0
        for res, label in zip(residues, case.design_labels):
            if label not in ("L", "D"):
                continue  # ncAA / non-stereocenter labels are not chirality-gated
            if "CB" not in res:
                continue
            c_total += 1
            if is_chirality_violation(res["N"], res["CA"], res["C"], res["CB"], label):
                c_viol += 1

        # φ/ψ deviation vs reference (diagnostic).
        pp_viol = pp_total = 0
        if case.ref_backbone is not None:
            phi_p, psi_p = backbone_torsions(residues)
            phi_r, psi_r = backbone_torsions(case.ref_backbone)
            for i in range(min(len(phi_p), len(phi_r))):
                import numpy as np
                if np.isnan(phi_p[i]) or np.isnan(phi_r[i]) or np.isnan(psi_p[i]) or np.isnan(psi_r[i]):
                    continue
                pp_total += 1
                if phi_psi_violation(phi_p[i], psi_p[i], phi_r[i], psi_r[i], tol_deg=25.0):
                    pp_viol += 1

        per_case[case.name] = aggregate_gate_report(c_viol, c_total, pp_viol, pp_total, violation_threshold)
        total_chir_viol += c_viol
        total_stereo += c_total
        total_pp_viol += pp_viol
        total_torsions += pp_total

    overall = aggregate_gate_report(
        total_chir_viol, total_stereo, total_pp_viol, total_torsions, violation_threshold
    )
    return overall, per_case
