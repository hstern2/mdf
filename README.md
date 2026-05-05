## mdf
Molecular Data Frame

Command-line utilities for working with molecular data in CSV, SMI, and SDF files,
backed by [`polars`](https://pola.rs/) and [`RDKit`](https://www.rdkit.org/).

### Installation

```bash
pip install git+https://github.com/hstern/mdf.git
```

### Formats

Input and output formats are inferred from file extensions (`.csv`, `.smi`, `.sdf`).
When reading from stdin, use `-i/--stdin-fmt` to specify the format (default: `csv`).
When writing to stdout, use `-o/--stdout-fmt` (default: `csv`).

Most commands accept one or more input files, or read from stdin if no files are given:

```bash
mdf props molecules.csv > with_props.csv
cat molecules.smi | mdf props -i smi > with_props.csv
```

---

### Subcommands

#### Molecular Properties & Transformations

**`props`** — Add computed molecular property columns: `HAC`, `MW`, `CLOGP`, `HBD`, `HBA`, `TPSA`, `SA_SCORE`, `QED`, `N_ROT_BONDS`, `PASSES_RO5`.

```bash
mdf props molecules.csv > with_props.csv
mdf props -s MOL_SMILES molecules.csv   # use a non-default SMILES column
```

**`canon`** — Canonicalize SMILES strings in-place using RDKit.

```bash
mdf canon molecules.csv > canonical.csv
```

**`pX`** — Convert IC50-like values (in a column whose name encodes the unit: `M`, `mM`, `uM`/`µM`, `nM`, `pM`) to pX (e.g. pIC50).

```bash
mdf pX IC50_nM data.csv -n pIC50 > with_pIC50.csv
```

---

#### Similarity

**`sims`** — Add a `tanimoto_similarity_to_<NAME>` column for each molecule in a target file.

```bash
mdf sims input.csv --to targets.csv > with_sims.csv
mdf sims input.csv --to targets.smi -S > sorted_by_sim.csv   # -S sorts by first sim column
```

**`diverse`** — Greedily sample rows, skipping molecules with Tanimoto similarity ≥ threshold to any already-selected molecule.

```bash
mdf diverse molecules.csv -t 0.6 > diverse_subset.csv
```

---

#### Medicinal Chemistry Filters

**`sieve`** — Filter molecules using medicinal chemistry rules. Supports [Lilly MedChem Rules](https://github.com/IanAWatson/Lilly-Medchem-Rules) and [Pat Walters' rd_filters](https://github.com/PatWalters/rd_filters) alert sets.

```bash
mdf sieve -l molecules.csv               # Lilly rules (strict)
mdf sieve -r molecules.csv               # Lilly rules (relaxed)
mdf sieve -P PAINS -P Glaxo molecules.csv  # Pat Walters alert sets
```

Available `-P` alert sets: `Glaxo`, `Dundee`, `BMS`, `PAINS`, `SureChEMBL`, `MLSMR`, `Inpharmatica`, `LINT`, `all`.

---

#### Filtering & Selection

**`grep`** — Keep rows where a column (matched by regex) contains a value (matched by regex). Use `-v` to invert.

```bash
mdf grep series "scaffold_A" data.csv
mdf grep status "active" data.csv -v   # exclude rows where status matches "active"
```

**`minmax`** — Filter rows by numeric range on a column (matched by regex).

```bash
mdf minmax MW -m 200 -M 500 data.csv
```

**`cols`** — Select columns whose names match a regex.

```bash
mdf cols "^(NAME|SMILES|pIC50)" data.csv
```

**`uniq`** — Drop duplicate rows by canonical SMILES.

```bash
mdf uniq molecules.csv > deduped.csv
```

**`recent`** — When rows share a `NAME`, keep only the most recent one (by a date column).

```bash
mdf recent -d "Assay Date" data.csv > latest.csv
```

---

#### Sorting & Sampling

**`sort`** — Sort by a column (matched by regex). Use `-r` for descending.

```bash
mdf sort pIC50 -r data.csv > sorted.csv
```

**`shuffle`** — Randomly shuffle rows.

```bash
mdf shuffle data.csv > shuffled.csv
```

**`take`** — Sample N random rows.

```bash
mdf take 100 data.csv > sample.csv
```

---

#### Multi-objective Analysis

**`pareto`** — Assign each row a Pareto front index (`front` column, 0 = non-dominated) based on one or more objective columns to minimize.

```bash
mdf pareto -O MW -O "SA Score" data.csv > with_fronts.csv
mdf pareto -O MW -O "SA Score" -f data.csv > first_front_only.csv
```

---

#### I/O & Reshaping

**`cat`** — Concatenate multiple files.

```bash
mdf cat a.csv b.csv c.csv > combined.csv
mdf cat -H vertical a.csv b.csv   # vertical (stack rows), horizontal (align columns), or diagonal (default)
```

**`merge`** — Join multiple files on a key column (default: `NAME`).

```bash
mdf merge -k NAME -H inner a.csv b.csv > merged.csv
```

**`split`** — Split into N shards by round-robin row assignment.

```bash
mdf split 5 data.csv -t "shard_%.csv"   # writes shard_0.csv … shard_4.csv
```

**`ttsplit`** — Randomly shuffle and split into train/test CSV files.

```bash
mdf ttsplit data.csv -f 0.8 -p dataset   # writes dataset_train.csv and dataset_test.csv
```

**`smi`** — Output data as SMILES format (`SMILES NAME`).

```bash
mdf smi molecules.csv > output.smi
```

**`colnames`** — Print column names, one per line.

```bash
mdf colnames data.csv
```

**`rename`** — Rename the first column matching a regex.

```bash
mdf rename "Molecule Name" NAME data.csv
```

---

#### Visualization

**`viz`** — Open an HTML page with a table of rendered molecule structures and all data columns.

```bash
mdf viz molecules.csv
mdf viz molecules.csv --size 300,300 -t "My Compounds"
```

**`plot`** — Open an HTML scatter plot of one or more Y columns vs. an X column, with Pearson R displayed.

```bash
mdf plot data.csv -x MW -y pIC50
mdf plot data.csv -x MW -y pIC50 -y CLOGP -t "Property Correlations"
```
