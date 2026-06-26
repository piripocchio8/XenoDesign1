"""Metal-coordination GEOMETRY gate via MetalHawk (Sgueglia/Vrettas, JCIM 2024).

A coordination-geometry analogue of the coiled-coil periodicity gate: instead of
hardcoding ideal donor-atom geometries (brittle), we let MetalHawk's trained ANN
*assign* a coordination class (LIN/TRI/TET/SPL/SQP/TBP/OCT) and read its confidence.

MetalHawk's actual output contract (see ``xenodesign/eval`` RUNBOOK / the upstream
``src/model_predictor.py``):

    MetalSitesPredictor(...)(sphere_pdb) -> (class_index, entropy)

where ``entropy`` is the natural-log Shannon entropy of the 7-class softmax
(``-sum p*ln p``; max for 7 equiprobable classes is ``ln 7 = 1.9459``). MetalHawk
has **no** "perplexity" field of its own — we derive it as ``perplexity = exp(entropy)``
(the effective number of competing geometry classes: 1.0 = one class is certain).
We gate on perplexity: **low perplexity == a confident, clean coordination geometry
== pass**.

Input contract: MetalHawk consumes a "sphere PDB" — a PDB whose first-shell atoms sit
around a single central metal (it auto-finds the metal nearest the centroid and the
closest non-H/non-C donors generally: N/O/S/Se/Cl/Br/I, not just His-N). We build that
sphere PDB straight from a predicted CIF here (no PyMOL needed, unlike upstream's
``extract_metal_sites.py``).

DESIGN NOTES
------------
* **Best-effort / guarded.** Every public entry point catches its own failures and
  returns a pass-through ``GateResult`` (``passed=True``, ``ok=False``, ``error=...``)
  so the gate can NEVER crash a design run. A gate that cannot run does not veto.
* **Off by default.** Wired as a user option via ``cfg.gates.metal_geometry`` with a
  ``metal_perplexity_thresh`` knob; nothing in the loop consults it automatically yet.
* **Isolated MetalHawk.** MetalHawk pins scikit-learn 1.0.2 model pickles, so we call
  it in its own env via subprocess (``metalhawk_env`` / ``metalhawk_dir``) rather than
  importing it into the repo's interpreter. The sphere-PDB build + parsing + threshold
  logic is pure CPU and unit-tested without MetalHawk installed.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

# 7 MetalHawk geometry classes (order matches upstream CLASS_TARGETS).
GEOMETRY_CLASSES = ("LIN", "TRI", "TET", "SPL", "SQP", "TBP", "OCT")

# Elements MetalHawk treats as candidate central metals (a superset of first-row TM +
# common heavier ones); we only need the membership test for metal auto-detection.
_METAL_ELEMENTS = {
    "MN", "FE", "CO", "NI", "CU", "ZN", "MO", "HO", "SC", "MG", "PT", "PD", "TA",
    "CD", "AU", "HG", "OS", "TI", "RH", "AS", "IR", "AG", "RU", "PB", "GD", "HF",
    "ZR", "EU", "GA", "PA", "RE", "SM", "CR", "NB", "TC", "LA", "Y", "U", "V", "W",
}

# Default perplexity gate threshold. perplexity = exp(entropy); 1.0 == one class is
# certain, ln(7)->~7 == maximally confused. A confident, clean geometry sits very close
# to 1.0; we allow modest hedging by default. Tunable via cfg.
DEFAULT_PERPLEXITY_THRESH = 1.5

# MetalHawk runs in its own env via ``micromamba run -n <env>``. The env name defaults to
# "metalhawk" but is overridable via the METALHAWK_ENV env var (so a non-micromamba / differently
# named install can point at its own env). The dir comes from METALHAWK_DIR (see metal_geometry_gate).
_DEFAULT_METALHAWK_ENV = os.environ.get("METALHAWK_ENV", "metalhawk")


@dataclass
class GateResult:
    """Outcome of the metal-geometry gate for one metal site.

    geometry:   MetalHawk's assigned class (e.g. 'TET'), or None if it could not run.
    entropy:    natural-log Shannon entropy of the class softmax, or None.
    perplexity: exp(entropy) -- effective number of competing classes, or None.
    threshold:  the perplexity threshold used for the pass decision.
    passed:     gate decision -- True if perplexity <= threshold (PASS = clean geometry).
                Best-effort: when the gate could not run (ok=False) this is True
                (pass-through) so a missing gate never vetoes a design.
    ok:         True iff MetalHawk actually produced a prediction.
    error:      short failure reason when ok is False, else None.
    """

    geometry: Optional[str] = None
    entropy: Optional[float] = None
    perplexity: Optional[float] = None
    threshold: float = DEFAULT_PERPLEXITY_THRESH
    passed: bool = True
    ok: bool = False
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Pure-CPU helpers (CIF parsing + sphere-PDB construction). No MetalHawk needed.
# --------------------------------------------------------------------------------------

def _parse_cif_atoms(cif_text: str) -> list[dict]:
    """Parse an mmCIF ``_atom_site`` loop into a list of column->value dicts.

    Tolerant whitespace-split parser (chai/ModelCIF style: one atom per line, no quoted
    coordinates). Reads the column order from the ``_atom_site.<col>`` header lines.
    """
    atoms: list[dict] = []
    order: list[str] = []
    in_loop = False
    for raw in cif_text.splitlines():
        s = raw.strip()
        if s.startswith("_atom_site."):
            order.append(s.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop:
            if s.startswith("ATOM") or s.startswith("HETATM"):
                parts = s.split()
                if len(parts) >= len(order):
                    atoms.append(dict(zip(order, parts)))
            elif s == "#" or s.startswith("loop_") or s.startswith("_"):
                # End of the atom_site loop.
                break
    return atoms


def _atom_xyz(a: dict) -> tuple[float, float, float]:
    return (float(a["Cartn_x"]), float(a["Cartn_y"]), float(a["Cartn_z"]))


def _select_metal(atoms: Sequence[dict], metal_element: Optional[str]) -> Optional[dict]:
    """Return the central metal atom record, or None.

    If ``metal_element`` is given (e.g. 'ZN'), pick the first atom of that element;
    otherwise pick the first atom whose element is in the known-metal set.
    """
    want = metal_element.upper() if metal_element else None
    for a in atoms:
        el = a.get("type_symbol", "").upper()
        if want is not None:
            if el == want:
                return a
        elif el in _METAL_ELEMENTS:
            return a
    return None


def _pdb_atom_line(serial: int, name: str, resn: str, element: str,
                   x: float, y: float, z: float) -> str:
    """Format a single strict-column PDB HETATM line (cols: name 13-16, xyz 31-54,
    element 77-78). Built by slice-assignment to avoid format-string drift."""
    buf = [" "] * 80

    def put(start1: int, s, width: int, right: bool = False) -> None:
        s = str(s)[:width]
        s = s.rjust(width) if right else s.ljust(width)
        buf[start1 - 1: start1 - 1 + width] = list(s)

    el = element.upper()
    put(1, "HETATM", 6)
    put(7, serial, 5, right=True)
    # Two-char elements (metals) start the name at col 13; 1-char elements at col 14
    # (the PDB convention that keeps the element right-justified within the name field).
    if len(name) >= 4 or len(el) >= 2:
        put(13, name, 4)
    else:
        put(14, name, 3)
    put(18, resn, 3, right=True)
    put(22, "A", 1)
    put(23, 1, 4, right=True)
    put(31, "%8.3f" % x, 8)
    put(39, "%8.3f" % y, 8)
    put(47, "%8.3f" % z, 8)
    put(55, "%6.2f" % 1.0, 6)
    put(61, "%6.2f" % 0.0, 6)
    put(77, el, 2, right=True)
    return "".join(buf).rstrip() + "\n"


def build_sphere_pdb(cif_text: str, metal_element: Optional[str] = None,
                     radius: float = 3.5) -> Optional[str]:
    """Build a MetalHawk sphere-PDB string from a predicted CIF.

    Selects the central metal (by ``metal_element`` or auto), keeps all atoms within
    ``radius`` Angstrom (the metal + its first shell; MetalHawk itself drops H/C and
    picks the closest donors), translates so the metal sits at the origin, and returns
    a PDB string. Returns None if no metal or no neighbours are found.
    """
    atoms = _parse_cif_atoms(cif_text)
    if not atoms:
        return None
    metal = _select_metal(atoms, metal_element)
    if metal is None:
        return None
    mx, my, mz = _atom_xyz(metal)

    def dist(a: dict) -> float:
        x, y, z = _atom_xyz(a)
        return math.dist((mx, my, mz), (x, y, z))

    near = sorted((a for a in atoms if dist(a) <= radius), key=dist)
    if len(near) < 2:  # need the metal + at least one donor
        return None

    lines: list[str] = []
    serial = 1
    for a in near:
        x, y, z = _atom_xyz(a)
        lines.append(_pdb_atom_line(
            serial,
            name=a.get("label_atom_id", a.get("type_symbol", "X")),
            resn=a.get("label_comp_id", "LIG")[:3],
            element=a.get("type_symbol", "X"),
            x=x - mx, y=y - my, z=z - mz,
        ))
        serial += 1
    lines.append("END\n")
    return "".join(lines)


# --------------------------------------------------------------------------------------
# Pure-CPU decision logic (testable without MetalHawk).
# --------------------------------------------------------------------------------------

def decision_from_prediction(class_index: int, entropy: float,
                             threshold: float = DEFAULT_PERPLEXITY_THRESH) -> GateResult:
    """Turn a raw MetalHawk ``(class_index, entropy)`` into a GateResult.

    perplexity = exp(entropy); PASS iff perplexity <= threshold.
    """
    geometry = (GEOMETRY_CLASSES[class_index]
                if 0 <= class_index < len(GEOMETRY_CLASSES) else None)
    perplexity = math.exp(entropy)
    return GateResult(
        geometry=geometry,
        entropy=float(entropy),
        perplexity=float(perplexity),
        threshold=float(threshold),
        passed=bool(perplexity <= threshold),
        ok=geometry is not None,
        error=None if geometry is not None else f"class_index out of range: {class_index}",
    )


def _parse_metalhawk_stdout(stdout: str) -> tuple[int, float]:
    """Parse the JSON line our runner script prints: ``{"class_index":..,"entropy":..}``."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            obj = json.loads(line)
            return int(obj["class_index"]), float(obj["entropy"])
    raise ValueError("no MetalHawk JSON result found in stdout")


# Runner executed inside the MetalHawk env. Kept tiny; reads a sphere PDB, prints JSON.
_RUNNER = r"""
import sys, json
from pathlib import Path
mh_dir, pdb_path, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, mh_dir)
from src.model_predictor import MetalSitesPredictor
p = MetalSitesPredictor(dir_model=Path(mh_dir) / "models" / model_name)
cls, ent = p(pdb_path)
print(json.dumps({"class_index": int(cls), "entropy": float(ent)}))
"""


def run_metalhawk(sphere_pdb: str,
                  metalhawk_dir: str,
                  metalhawk_env: Optional[str] = _DEFAULT_METALHAWK_ENV,
                  model_name: str = "HPO_CSD_CSD_CV.model",
                  timeout: float = 120.0) -> tuple[int, float]:
    """Run MetalHawk on a sphere-PDB string (subprocess into its env). Returns
    ``(class_index, entropy)``. Raises on any failure (callers guard)."""
    with tempfile.TemporaryDirectory() as td:
        pdb_path = Path(td) / "site.pdb"
        pdb_path.write_text(sphere_pdb)
        runner = Path(td) / "_mh_runner.py"
        runner.write_text(_RUNNER)
        if metalhawk_env:
            cmd = ["micromamba", "run", "-n", metalhawk_env, "python", str(runner)]
        else:
            cmd = ["python", str(runner)]
        cmd += [str(metalhawk_dir), str(pdb_path), model_name]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"MetalHawk exited {proc.returncode}: {proc.stderr.strip()[:400]}")
        return _parse_metalhawk_stdout(proc.stdout)


def metal_geometry_gate(cif_path: str | Path,
                        metal_element: Optional[str] = None,
                        threshold: float = DEFAULT_PERPLEXITY_THRESH,
                        radius: float = 3.5,
                        metalhawk_dir: Optional[str] = None,
                        metalhawk_env: Optional[str] = _DEFAULT_METALHAWK_ENV,
                        model_name: str = "HPO_CSD_CSD_CV.model") -> GateResult:
    """Best-effort metal-coordination geometry gate on a predicted CIF.

    Builds a sphere PDB around the (auto-detected or specified) metal, runs MetalHawk in
    its env, and returns a :class:`GateResult` (geometry + perplexity + pass/fail).

    NEVER raises: on any error returns a pass-through result (``passed=True, ok=False,
    error=...``) so it cannot break a design run. ``metalhawk_dir`` may be supplied or
    taken from the ``METALHAWK_DIR`` environment variable; if neither resolves, the gate
    is a no-op pass-through.
    """
    try:
        mh_dir = metalhawk_dir or os.environ.get("METALHAWK_DIR")
        if not mh_dir or not Path(mh_dir).is_dir():
            return GateResult(threshold=threshold, passed=True, ok=False,
                              error="MetalHawk dir not found (set METALHAWK_DIR or pass metalhawk_dir)")
        cif_text = Path(cif_path).read_text()
        sphere = build_sphere_pdb(cif_text, metal_element=metal_element, radius=radius)
        if sphere is None:
            return GateResult(threshold=threshold, passed=True, ok=False,
                              error="could not build sphere PDB (no metal / no donors)")
        class_index, entropy = run_metalhawk(
            sphere, metalhawk_dir=mh_dir, metalhawk_env=metalhawk_env,
            model_name=model_name)
        return decision_from_prediction(class_index, entropy, threshold=threshold)
    except Exception as exc:  # never crash a design run
        return GateResult(threshold=threshold, passed=True, ok=False,
                          error=f"{type(exc).__name__}: {exc}")
