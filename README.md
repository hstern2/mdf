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

Most commands accept one or more input files, or read from stdin if no files are given.

### Subcommands

#### Molecular Properties and Transformations

**`props`**: Calculate molecular properties from SMILES and add them as new columns: `HAC`, `MW`, `CLOGP`, `HBD`, `HBA`, `TPSA`, `SA_SCORE`, `QED`, `N_ROT_BONDS`, `PASSES_RO5`.

**`canon`**: Canonicalize SMILES in the specified column using RDKit.

**`pX`**: Calculate a new pX column as -log10(unit * x), with the unit (`M`, `mM`, `uM`, `nM`, `pM`) inferred from the matched column name.

#### Similarity

**`sims`**: Add Tanimoto similarity columns to the input dataframe for each molecule in a target dataframe.

**`diverse`**: Sample rows in order, excluding molecules similar to already selected ones based on Tanimoto similarity of RDKit fingerprints.

#### Medicinal Chemistry Filters

**`sieve`**: Filter molecules using the sieve library (Lilly MedChem Rules, Pat Walters alerts, ring filters). Use `-l` for Lilly rules, `-r` for relaxed Lilly rules, `-R` for the unprecedented ring filter, and `-P` for Pat Walters alert sets.

Available `-P` alert sets: `Glaxo`, `Dundee`, `BMS`, `PAINS`, `SureChEMBL`, `MLSMR`, `Inpharmatica`, `LINT`, `all`.

#### Filtering and Selection

**`grep`**: Print rows where any column matching the column regex has a value matching the value regex.

**`minmax`**: Filter rows by numeric range on the first column matching the regex pattern.

**`cols`**: Print only columns matching the regex pattern.

**`uniq`**: Drop duplicates based on canonical SMILES from the specified column.

**`recent`**: Keep only the most recent row for each NAME, based on the date column matching the regex pattern.

#### Sorting and Sampling

**`sort`**: Sort by the first column matching the regex pattern.

**`stats`**: Print count, mean, stddev, min, and max for each numeric column, excluding nulls and NaNs.

**`shuffle`**: Shuffle the rows randomly.

**`take`**: Sample n random rows from the input.

#### Multi-objective Analysis

**`pareto`**: Perform Pareto sorting on objective functions (columns to minimize). Returns all fronts with a `front` column (0=best, non-dominated front), or only the first front if `--only-first-front` is specified.

#### I/O and Reshaping

**`cat`**: Concatenate and print mdf files.

**`merge`**: Merge multiple mdf files on a column, removing duplicate columns except for the key.

**`split`**: Split the input into n shards (round-robin) and write each to a file using the template (with `%` replaced by the shard index).

**`ttsplit`**: Perform a random test-train split and write to separate files with the given prefix.

**`smi`**: Concatenate and print mdf files in SMILES format (equivalent to `cat -o smi`).

**`colnames`**: Print column names, one per line.

**`rename`**: Rename the first column matching the regex to the new name.

#### Visualization

**`viz`**: Write an HTML table of SVG molecule images alongside dataframe columns and open it in the browser.

**`plot`**: Scatter plot of columns matching `-x` (x axis) vs `-y` (y axis, repeatable), with optional error bars from `--xerr` and `--yerr`; shows Pearson R in legend. Use `--labels/--label-col` to annotate points, `--xlabel`/`--ylabel` to override axis labels, and `-o/--output plot.png` to write a high-quality PNG instead of opening the browser.
