# XenoDesign1 D-chirality benchmark layer

Data the GPU Tier-0a chirality gate and the Chai backend need but that could **not**
be fetched in the cloud build environment (RCSB / MONDE·T returned 403 there; only
GitHub was reachable). Assembled on the laptop on 2026-06-13 and committed so the
GPU machine (Ifrit) can run the gate reproducibly from `git pull` alone.

```
data/benchmark/
├── mondet/                      # the MONDE·T ncAA catalog (catalog.py input)
│   ├── mondet.csv                  normalized: component_id,parent,entity_count,smiles  ← feed to catalog.py
│   ├── mondet-noncanonical-classes.csv   raw per-code table (provenance)
│   ├── mondet-structures.tsv.gz   raw main browse table, gzipped (per-code → PDB entity lists)
│   └── PROVENANCE.md
├── d_codes/                     # derived from MONDE·T
│   ├── D_pdb_map.csv               each chai-supported D-code → PDB ids that contain it
│   └── D_CONTAINING_PDBS.txt       primary (curated) + extended (MONDE·T) benchmark PDB lists
├── curated/                     # hand-curated D-/heterochiral set
│   ├── pdb_list_annotated.txt      curated PDB ids with cofactor / cyclic / caveat notes
│   └── mirrored_fastas/            <id>.fasta + <id>_mirrored.fasta (D-CCD notation) per family
└── chai_output_fixture/         # a real Chai-1 output set, for CPU testing of the parser
    ├── pred.model_idx_{0..4}.cif
    └── scores.model_idx_{0..4}.npz
```

## How each piece is used

### 1. `mondet/mondet.csv` → `xenodesign.catalog`
`catalog.py` reads a CSV with a `component_id` column (comma-delimited, `csv.DictReader`).
The raw MONDE·T export is **tab-separated** with an `ID` column, so it is normalized here.

```python
from xenodesign.catalog import chai_supported_d_codes, ncaa_codes
codes = chai_supported_d_codes("data/benchmark/mondet/mondet.csv")
# -> the 19 chai-supported canonical D-codes present in MONDE·T:
#    DAL DAR DAS DCY DGL DGN DHI DIL DLE DLY DPN DPR DSG DSN DTH DTR DTY DVA MED
```

### 2. `d_codes/D_CONTAINING_PDBS.txt` → the chirality benchmark test
`tests/gpu/test_chirality_benchmark_gpu.py` ships only 4 placeholder ids
(`7QDI, 1JNO, 1MAG, 1GRM`). Replace/extend `D_CONTAINING_PDBS` from this file:

- **Primary set** (top of file): 28 hand-curated D-/heterochiral families
  (6UCX–6UGC, 7UBC–7UZL, 8CTO/8CUN/8CWA) — short D-peptides and heterochiral
  assemblies, several with metal cofactors. Start the gate here.
- **Extended pool**: 1,064 PDB entries that MONDE·T reports as containing any
  chai-supported D-code. Intersect with curation before spending GPU on it
  (`D_pdb_map.csv` gives the per-code breakdown).

The test fetches each CIF from RCSB **at runtime** — Ifrit has network, so no CIFs
are bundled.

### 3. `curated/` — ground-truth inputs and curation knowledge
- `pdb_list_annotated.txt` carries the curation that the bare ids cannot: which
  entries have Zn / Na / Fe4S4 cofactors, which are cyclic / N-methylated, and the
  critical caveat that **the deposited FASTA omits the D-residues** for most of these,
  so Chai may only have trained on the L part — read this before interpreting any gate
  failure.
- `mirrored_fastas/` gives, per family, the D-CCD sequence and its mirror partner in
  the exact notation `xenodesign.io_spec` / `mirror` consume.

### 4. `chai_output_fixture/` — CPU test fixture for the Chai parser
A real 5-model Chai-1 prediction (`pred.model_idx_*.cif` + `scores.*.npz`). Use it to
verify `ChaiBackend._to_prediction` and `scorer` parsing **without a GPU**: the npz keys
are the I/O contract (`aggregate_score`, `ptm`, `iptm`, `has_inter_chain_clashes`,
`chain_chain_clashes`, `plddt`, `pae`, `pde`).

## Sources / citation
MONDE·T (Waldherr, Freimann et al., 2026 export), https://mondet.tuebingen.mpg.de —
*MONDE·T: A Database and Interactive Webserver for Non-Canonical Amino Acids (ncAAs)
in the PDB*, bioRxiv 2025.12.21.695100. CCD codes from wwPDB.
Curated set sources noted inline in `curated/pdb_list_annotated.txt`
(refs: 10.1002/pro.3974; 10.1016/j.cell.2022.07.019; 10.1021/jacs.8b07553).
