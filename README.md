## mdf
Molecular Data Frame

Command-line utilities for working with molecular data stored in CSV/SMI files,
backed by `polars` and `RDKit`.

### Similarity

- **`diverse`**: sample rows in order, excluding molecules similar to already
  selected ones based on Tanimoto similarity of RDKit fingerprints.
- **`sims`**: add per-molecule Tanimoto similarity columns to an input dataframe
  for each molecule in a target dataframe.

Usage:

```bash
# Add Tanimoto similarity columns tanimoto_similarity_to_NAME* for each
# molecule in target.csv (or target.smi), using SMILES columns named "SMILES"
mdf sims input.csv --to target.csv > with_sims.csv
# or
mdf sims input.csv --to target.smi > with_sims.csv

# Explicitly set SMILES columns
mdf sims input.csv --to target.csv -s QUERY_SMILES --to-smiles-col REF_SMILES
```
