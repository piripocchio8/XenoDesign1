# MONDE·T provenance

- **Source:** https://mondet.tuebingen.mpg.de (Max Planck Institute for Biology Tübingen)
- **Paper:** *MONDE·T: A Database and Interactive Webserver for Non-Canonical Amino
  Acids (ncAAs) in the PDB*, Waldherr, Freimann et al., bioRxiv 2025.12.21.695100.
- **Export version:** `2026-Apr-Waldherr-Freimann`
- **Fetched:** 2026-06-13, on the laptop (the cloud build env got 403 from this host).

## Files and how they were obtained

The webserver loads its tables client-side from these static endpoints:

| Committed file | Fetched from | Notes |
|---|---|---|
| `mondet-noncanonical-classes.csv` | `/static/data/mondet-2026-Apr-Waldherr-Freimann-noncanonical-classes.csv` | semicolon-delimited; one row per ncAA code (`ID;SMILES;entity_count;parent`); 2562 codes |
| `mondet-structures.tsv.gz` | `/static/data/mondet-2026-Apr-Waldherr-Freimann-tab.csv` (gzipped here) | **tab**-delimited main browse table; per-code with `entity_IDs` = `PDBID_entity` lists; 1913 rows |
| `mondet.csv` | derived from the classes CSV | normalized to comma + renamed `ID`→`component_id` so `xenodesign.catalog` reads it directly |

Not committed (size, not needed for the gate):
`mondet-2026-Apr-Waldherr-Freimann-ncaa-angles.csv` (~37 MB, per-residue φ/ψ for all
ncAA). The gate derives reference φ/ψ from each deposited CIF at runtime instead. Fetch
it from `/static/data/...-ncaa-angles.csv` if ever needed.

## To refresh
```bash
base=https://mondet.tuebingen.mpg.de/static/data
curl -sL "$base/mondet-2026-Apr-Waldherr-Freimann-noncanonical-classes.csv" -o classes.csv
curl -sL "$base/mondet-2026-Apr-Waldherr-Freimann-tab.csv"                  -o tab.csv
# then re-run the normalization (rename ID->component_id, semicolon->comma).
```
