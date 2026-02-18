from enum import Enum
import polars as ps
from polars.exceptions import SchemaError
from pathlib import Path
import re, io, sys, math
from typer import Typer, Option, Argument, Exit
from typing import Annotated, List, Iterable, Optional, Union
import typer, click, rich
from sys import stdin, stdout

__version__ = "0.1.0"


class MDFFormat(str, Enum):
    """fmt options for reading and writing mdf files"""

    csv = "csv"
    smi = "smi"
    viz = "viz"


class MergeHow(str, Enum):
    left = "left"
    right = "right"
    inner = "inner"
    full = "full"


class CatHow(str, Enum):
    vertical = "vertical"
    horizontal = "horizontal"
    diagonal = "diagonal"


FilesType = Optional[List[Path]]
FilesArg = Argument(
    None,
    help="Input files. If not provided, read from stdin.",
    exists=True,
    dir_okay=False,
    file_okay=True,
    allow_dash=True,
)

StdinFmtOpt = Annotated[
    MDFFormat, Option("-i", "--stdin-fmt", help="Format for reading from stdin")
]
StdoutFmtOpt = Annotated[
    MDFFormat, Option("-o", "--stdout-fmt", help="Format for writing to stdout")
]

# Constants for molecular properties
PROP_NAMES = [
    "HAC",
    "MW",
    "CLOGP",
    "HBD",
    "HBA",
    "TPSA",
    "SA_SCORE",
    "QED",
    "N_ROT_BONDS",
    "PASSES_RO5",
]
FLOAT_PROPS = {
    "MW",
    "CLOGP",
    "TPSA",
    "SA_SCORE",
    "QED",
}  # Properties that should be formatted


def format_number(value, digits: int):
    """Format a number to the specified number of significant digits"""
    if value is None or not isinstance(value, (int, float)):
        return value
    try:
        if value == 0:
            return 0.0
        # Calculate the order of magnitude
        magnitude = math.floor(math.log10(abs(value)))
        # Calculate the factor to scale the number
        scale = 10 ** (digits - 1 - magnitude)
        # Round and scale back
        return round(value * scale) / scale
    except (ValueError, OverflowError, ZeroDivisionError):
        return value


def _version_callback(value: bool):
    if value:
        rich.print(f"[bold blue]MDF - Molecular Data Frame[/bold blue]")
        rich.print(f"[dim]version: {__version__}[/dim]")
        raise Exit()


app = Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=f"Molecular data frame utilities for handling molecular data in various formats\n\nVersion: {__version__}",
)


@app.callback()
def main(
    version: bool = Option(
        False,
        "-v",
        "--version",
        help="Show version information and exit",
        callback=_version_callback,
    ),
):
    """
    MDF - Molecular Data Frame

    Utilities for handling molecular data in various formats including
    filtering, transformation, and analysis operations.
    """
    pass


class MDF:
    """molecular data frame class for handling molecular data in various formats"""

    def __init__(self, df: ps.DataFrame):
        """initialize mdf from a polars dataframe"""
        self._df = df

    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying DataFrame, wrapping DataFrame results"""
        attr = getattr(self._df, name)
        if callable(attr):

            def wrapped_method(*args, **kwargs):
                result = attr(*args, **kwargs)
                # If result is a DataFrame, wrap it as MDF
                if isinstance(result, ps.DataFrame):
                    return MDF(result)
                return result

            return wrapped_method
        return attr

    # Forward essential properties
    @property
    def columns(self):
        return self._df.columns

    @property
    def height(self):
        return self._df.height

    @property
    def width(self):
        return self._df.width

    @property
    def shape(self):
        return self._df.shape

    def __len__(self):
        """Return the number of rows"""
        return self._df.height

    def __str__(self):
        """String representation"""
        return str(self._df)

    def __repr__(self):
        """Repr representation"""
        return f"MDF({repr(self._df)})"

    def __getitem__(self, key):
        """Get item (column or slice)"""
        result = self._df.__getitem__(key)
        if isinstance(result, ps.DataFrame):
            return MDF(result)
        return result

    def write_file(self, file, fmt: MDFFormat = MDFFormat.csv):
        """write mdf to file in specified fmt; opens file if given a path or string"""
        if fmt == MDFFormat.viz:
            self.viz()
            return

        should_close = False
        if isinstance(file, (str, Path)):
            file = open(file, "w")
            should_close = True
        try:
            if fmt == MDFFormat.csv:
                self._df.write_csv(file)
            elif fmt == MDFFormat.smi:
                for row in self._df.iter_rows(named=True):
                    print(f"{row['SMILES']} {row['NAME']}", file=file)
            else:
                raise ValueError(f"Unknown fmt: {fmt}")
        except (BrokenPipeError, OSError) as e:
            # Handle the case where the output pipe is closed (e.g., head, grep)
            # This includes both Python's BrokenPipeError and OS-level broken pipe errors
            if "Broken pipe" in str(e) or e.errno == 32:  # EPIPE = 32
                pass  # Normal behavior when pipe is closed early
            else:
                raise  # Re-raise other OSErrors
        finally:
            if should_close:
                file.close()

    def viz(
        self,
        smiles_col: str = "SMILES",
        mols_per_row: int = 4,
        size: tuple = (250, 250),
        title: str = "molecules",
    ):
        """visualize molecules in a grid using RDKit, with NAME as labels"""
        # Lazy imports
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Draw
            import matplotlib.pyplot as plt
        except ImportError as e:
            print(
                f"RDKit and matplotlib are required for visualization: {e}",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)

        # Check required columns
        if smiles_col not in self.columns:
            print(f"Column '{smiles_col}' not found in dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

        if "NAME" not in self.columns:
            print("Column 'NAME' not found in dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

        # Extract SMILES and NAME
        smiles_list = []
        names_list = []
        for row in self._df.iter_rows(named=True):
            smiles = row[smiles_col]
            name = row.get("NAME", "")
            if smiles:
                smiles_list.append(smiles)
                names_list.append(str(name) if name else "")

        if not smiles_list:
            print("No SMILES found to visualize", file=sys.stderr)
            raise typer.Exit(code=1)

        # Create molecules from SMILES
        mols = []
        valid_names = []
        for s, n in zip(smiles_list, names_list):
            m = Chem.MolFromSmiles(s)
            if m is None:
                print(f"Warning: Invalid SMILES '{s}', skipping", file=sys.stderr)
                continue
            AllChem.Compute2DCoords(m)
            mols.append(m)
            valid_names.append(n)

        if not mols:
            print("No valid molecules to visualize", file=sys.stderr)
            raise typer.Exit(code=1)

        # Create grid image with NAME as legends
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=size,
            legends=valid_names,
        )

        # Display using matplotlib
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def matching_cols(self, pattern: str) -> List[str]:
        """get list of column names matching the regex pattern"""
        matching_cols = [col for col in self.columns if re.search(pattern, col)]
        if not matching_cols:
            print(f"No columns matching pattern '{pattern}'", file=sys.stderr)
            raise typer.Exit(code=1)
        return matching_cols

    def cols(self, pattern: str) -> "MDF":
        """return a new mdf with only columns matching the regex pattern"""
        return MDF(self._df.select(self.matching_cols(pattern)))

    def rename(self, from_pattern: str, to: str) -> "MDF":
        """rename the first column matching from_pattern to to"""
        cols = list(self.columns)
        for i, col in enumerate(cols):
            if re.search(from_pattern, col):
                old_name = col
                cols[i] = to
                break
        else:
            print(f"No column matching pattern '{from_pattern}'", file=sys.stderr)
            raise typer.Exit(code=1)
        return MDF(self._df.rename({old_name: to}))

    @classmethod
    def from_csv(cls, f):
        """create mdf from csv file"""
        return cls(ps.read_csv(f, infer_schema_length=None).rename({'Molecule Name': 'NAME'}, strict=False))

    @classmethod
    def from_smi(cls, fn):
        """create mdf from smiles file"""
        if isinstance(fn, (str, Path)):
            with open(fn) as f:
                lines = f.readlines()
        else:
            lines = fn.readlines()

        # Parse each line: first non-whitespace sequence is SMILES, rest is NAME
        parsed_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split on first whitespace: SMILES is everything before first whitespace,
            # NAME is everything after (including any subsequent whitespace)
            parts = line.split(None, 1)  # split on any whitespace, max 1 split
            if len(parts) == 1:
                # Only SMILES, no NAME
                smiles = parts[0]
                name = ""
            else:
                # SMILES and NAME
                smiles = parts[0]
                name = parts[1]  # This preserves any whitespace within the name

            parsed_lines.append(f"{smiles}\t{name}")

        # Create DataFrame from parsed lines using tab separator
        smi_buf = io.StringIO("\n".join(parsed_lines))
        df = ps.read_csv(
            smi_buf, separator="\t", has_header=False, new_columns=["SMILES", "NAME"]
        )
        return cls(df)

    @classmethod
    def cat(cls, mdfs: Iterable["MDF"], how: CatHow = CatHow.diagonal):
        """concatenate multiple mdfs into one"""
        # Extract the underlying DataFrames from MDF objects
        dataframes = [mdf._df for mdf in mdfs]
        try:
            return cls(ps.concat(dataframes, how=how.value))
        except SchemaError:
            # Cast Int64 columns to Float64 for compatibility
            for i, df in enumerate(dataframes):
                for col in df.columns:
                    if df.schema[col] == ps.Int64:
                        dataframes[i] = df.with_columns(ps.col(col).cast(ps.Float64))
            return cls(ps.concat(dataframes, how=how.value))

    @classmethod
    def from_file(cls, f, fmt: MDFFormat = None):
        """create mdf from file, automatically detecting fmt"""
        if fmt is None:
            if hasattr(f, "name") and str(f.name).endswith(".smi"):
                fmt = MDFFormat.smi
            elif isinstance(f, (str, Path)) and str(f).endswith(".smi"):
                fmt = MDFFormat.smi
            else:
                fmt = MDFFormat.csv
        if fmt == MDFFormat.smi:
            return cls.from_smi(f)
        elif fmt == MDFFormat.csv:
            return cls.from_csv(f)
        else:
            raise ValueError(f"Unknown fmt: {fmt}")

    @classmethod
    def from_files(cls, files: Union[None, Iterable[Path]]) -> "MDF":
        """create mdf by concatenating files"""
        if files is None:
            files = []
        mdfs = [cls.from_file(f) for f in files]
        if not mdfs:
            return cls(ps.DataFrame())  # Return empty MDF if no files
        return cls.cat(mdfs)

    @classmethod
    def from_stdin_and_files(
        cls, files: List[Path], stdin_fmt: MDFFormat, how: CatHow = CatHow.diagonal
    ) -> "MDF":
        """create mdf by concatenating stdin (if not isatty) and files"""
        mdfs = []
        if not stdin.isatty():
            # Read all stdin into a seekable buffer
            stdin_data = stdin.read()
            # Only process stdin if it's not empty
            if stdin_data.strip():
                stdin_buf = io.StringIO(stdin_data)
                mdfs.append(cls.from_file(stdin_buf, stdin_fmt))
        if files:
            mdfs.extend([cls.from_file(f) for f in files])
        if not mdfs:
            return cls(ps.DataFrame())  # Return empty MDF if no input
        return cls.cat(mdfs, how=how)

    @classmethod
    def merge(
        cls, dfs: Iterable["MDF"], on: str = "NAME", how: MergeHow = MergeHow.inner
    ) -> "MDF":
        """merge multiple mdfs on a column, coalescing duplicate columns (not just the key)"""

        def join_two(left, right):
            result = left._df.join(right._df, on=on, how=how.value)
            # Coalesce all columns with _right suffix
            for col in result.columns:
                if col.endswith("_right"):
                    base_col = col[:-6]
                    if base_col in result.columns:
                        result = result.with_columns(
                            result[base_col].fill_null(result[col]).alias(base_col)
                        ).drop(col)
            return cls(result)

        from functools import reduce

        merged = reduce(join_two, dfs)
        return merged

    def grep(
        self, column_pattern: str, value_pattern: str, invert_match: bool = False
    ) -> "MDF":
        """return a new mdf with rows where any column matching column_pattern has a value matching value_pattern (regex)"""
        cols = self.matching_cols(column_pattern)
        mask = (
            self._df.select(cols)
            .map_rows(lambda row: any(re.search(value_pattern, str(v)) for v in row))
            .to_series()
        )
        # Invert the mask if invert_match is True
        if invert_match:
            mask = ~mask
        return MDF(self._df.filter(mask))

    def sort(self, pattern: str, reverse: bool = False) -> "MDF":
        """return a new mdf sorted by the first column matching the regex pattern"""
        col = self.matching_cols(pattern)[0]
        return MDF(self._df.sort(col, descending=reverse))

    def range(
        self,
        pattern: str,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> "MDF":
        """return a new mdf with rows filtered by numeric range on the first column matching the regex pattern"""
        col = self.matching_cols(pattern)[0]

        # Build filter conditions
        conditions = []
        if minimum is not None:
            conditions.append(ps.col(col) >= minimum)
        if maximum is not None:
            conditions.append(ps.col(col) <= maximum)

        if not conditions:
            # No filtering conditions provided, return copy
            return MDF(self._df.clone())

        # Combine conditions with AND
        filter_expr = conditions[0]
        for condition in conditions[1:]:
            filter_expr = filter_expr & condition

        return MDF(self._df.filter(filter_expr))

    def uniq(self, smiles_col: str = "SMILES") -> "MDF":
        """return a new mdf with duplicates dropped based on canonical SMILES from the specified column"""
        try:
            from rdkit import Chem
        except ImportError:
            print(
                "RDKit is required for SMILES canonicalization. Please install rdkit.",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)

        if smiles_col not in self.columns:
            print(f"Column '{smiles_col}' not found in dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

        # Add canonical SMILES column using RDKit's CanonSmiles directly
        def safe_canon_smiles(x):
            if not x:
                return None
            try:
                return Chem.CanonSmiles(str(x))
            except:
                return None

        df_with_canonical = self._df.with_columns(
            ps.col(smiles_col)
            .map_elements(safe_canon_smiles, return_dtype=ps.Utf8)
            .alias("__canonical_smiles__")
        )

        # Remove rows where canonicalization failed (None values)
        df_with_canonical = df_with_canonical.filter(
            ps.col("__canonical_smiles__").is_not_null()
        )

        # Drop duplicates based on canonical SMILES
        result = df_with_canonical.unique(subset=["__canonical_smiles__"]).drop(
            "__canonical_smiles__"
        )

        return MDF(result)

    def split(self, n: int, template: str = "t%/a.csv"):
        """split the mdf into n shards (round-robin) and write each to a file using the template (with % replaced by the shard index); creates directories if needed; output format is determined by the file suffix (.smi or .csv)"""
        width = len(str(n - 1))
        # add a row count column for filtering
        df = self._df.with_row_index("__row__")
        fmt = MDFFormat.smi if Path(template).suffix == ".smi" else MDFFormat.csv
        for i in range(n):
            shard = df.filter(ps.col("__row__") % n == i).drop("__row__")
            idx_str = str(i).zfill(width)
            out_path = template.replace("%", idx_str)
            out_path = Path(out_path)
            if out_path.parent != Path("."):
                out_path.parent.mkdir(parents=True, exist_ok=True)
            MDF(shard).write_file(out_path, fmt)

    def take(self, n: int) -> "MDF":
        """return a new mdf with n random rows"""
        return MDF(self._df.sample(n=n))

    def shuffle(self) -> "MDF":
        """return a new mdf with rows shuffled randomly"""
        return MDF(self._df.sample(n=self.height, shuffle=True))

    def pareto(self, objectives: str, only_non_dominated_front: bool = False) -> "MDF":
        """return a new mdf with Pareto front rows, given a regex pattern to match objective function column names (columns to minimize)

        Args:
            objectives: Regex pattern to match objective function column names (columns to minimize)
            only_non_dominated_front: If True, return only the first (non-dominated) front.
                                    If False, return all fronts with a 'front' column indicating front number.
        """
        # Lazy import
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        # Check that objective columns exist
        matching_cols = self.matching_cols(objectives)
        if not matching_cols:
            print(
                f"No columns matching objective pattern '{objectives}'", file=sys.stderr
            )
            raise typer.Exit(code=1)

        # Convert to numpy array for NonDominatedSorting
        obj_values = self._df.select(matching_cols).to_numpy()

        # Get indices of solutions by front
        front_indices = NonDominatedSorting().do(
            obj_values, only_non_dominated_front=only_non_dominated_front
        )

        if only_non_dominated_front:
            # front_indices is a numpy array of indices for the first front only
            mask = [False] * self._df.height
            for idx in front_indices:
                mask[idx] = True
            return MDF(self._df.filter(ps.Series(mask)))
        else:
            # front_indices is a list of numpy arrays, one for each front
            # Create a list to store dataframes with front information
            front_dfs = []
            for front_num, indices in enumerate(front_indices):
                mask = [False] * self._df.height
                for idx in indices:
                    mask[idx] = True
                front_df = self._df.filter(ps.Series(mask)).with_columns(
                    ps.lit(front_num).alias("front")
                )
                front_dfs.append(front_df)

            # Concatenate all fronts
            combined_df = ps.concat(front_dfs)
            return MDF(combined_df)

    def props(self, smiles_col: str = "SMILES", digits: int = 3) -> "MDF":
        """return a new mdf with molecular properties calculated from SMILES in the specified column"""
        # Lazy imports
        try:
            from rdkit import Chem
            from rdkit.Chem.QED import qed
            from rdkit.Chem.Descriptors import TPSA, MolWt
            from rdkit.Chem.Crippen import MolLogP
            from rdkit.Chem.Lipinski import (
                NumHDonors,
                NumHAcceptors,
                HeavyAtomCount,
                NumRotatableBonds,
            )
            import os

            # Import sascorer with error handling
            try:
                sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, "SA_Score"))
                import sascorer
            except (ImportError, AttributeError):
                print(
                    "Warning: SA_Score not available, using dummy values",
                    file=sys.stderr,
                )
                sascorer = None
        except ImportError as e:
            print(
                f"RDKit is required for molecular property calculations: {e}",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)

        if smiles_col not in self.columns:
            print(f"Column '{smiles_col}' not found in dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

        def calculate_properties(smiles_str):
            """Calculate all molecular properties for a SMILES string"""
            if not smiles_str:
                return (None,) * len(PROP_NAMES)

            try:
                mol = Chem.MolFromSmiles(str(smiles_str))
                if mol is None:
                    return (None,) * len(PROP_NAMES)

                # Calculate raw properties
                raw_props = {
                    "HAC": HeavyAtomCount(mol),
                    "MW": MolWt(mol),
                    "CLOGP": MolLogP(mol),
                    "HBD": NumHDonors(mol),
                    "HBA": NumHAcceptors(mol),
                    "TPSA": TPSA(mol),
                    "SA_SCORE": sascorer.calculateScore(mol) if sascorer else None,
                    "QED": qed(mol),
                    "N_ROT_BONDS": NumRotatableBonds(mol),
                }

                # Format float properties
                for prop in FLOAT_PROPS:
                    if raw_props[prop] is not None:
                        raw_props[prop] = format_number(raw_props[prop], digits)

                # Calculate Lipinski Rule of 5 compliance
                mw, clogp, hbd, hba = (
                    raw_props["MW"],
                    raw_props["CLOGP"],
                    raw_props["HBD"],
                    raw_props["HBA"],
                )
                raw_props["PASSES_RO5"] = (
                    mw is not None
                    and clogp is not None
                    and mw <= 500
                    and clogp <= 5
                    and hbd <= 5
                    and hba <= 10
                )

                return tuple(raw_props[prop] for prop in PROP_NAMES)

            except (ValueError, TypeError, AttributeError) as e:
                # More specific exception handling
                return (None,) * len(PROP_NAMES)

        # Apply the function to calculate properties
        properties_df = self._df.select(smiles_col).map_rows(
            lambda row: calculate_properties(row[0])
        )

        # Rename columns more concisely
        column_mapping = {f"column_{i}": prop for i, prop in enumerate(PROP_NAMES)}
        properties_df = properties_df.rename(column_mapping)

        # Combine with original dataframe
        return MDF(self._df.hstack(properties_df))

    def diverse(self, smiles_col: str = "SMILES", threshold: float = 0.7) -> "MDF":
        """return a new mdf with diverse molecules, sampling rows in order but excluding molecules similar to already seen ones"""
        # Lazy imports
        try:
            from rdkit.DataStructs import TanimotoSimilarity
            from rdkit.Chem import AllChem, MolFromSmiles
            from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
        except ImportError as e:
            print(
                f"RDKit is required for molecular similarity calculations: {e}",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)

        if smiles_col not in self.columns:
            print(f"Column '{smiles_col}' not found in dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

        # Initialize fingerprint generator and tautomer enumerator
        fpgen = AllChem.GetRDKitFPGenerator()
        te = TautomerEnumerator()

        def fingerprint(smi: str):
            """Generate fingerprint for a SMILES string"""
            try:
                if smi is None or not str(smi).strip():
                    return None
                mol = MolFromSmiles(str(smi))
                if mol is None:
                    return None
                return fpgen.GetFingerprint(te.Canonicalize(mol))
            except (ValueError, TypeError, AttributeError):
                return None

        def contains_similar(fp1, seen_fps):
            """Check if fp1 is similar to any fingerprint in seen_fps"""
            if fp1 is None:
                return False
            for fp2 in seen_fps:
                if fp2 is not None and TanimotoSimilarity(fp1, fp2) > threshold:
                    return True
            return False

        # Process rows efficiently using boolean mask to avoid index columns
        mask = [False] * self._df.height
        seen_fingerprints = []

        # Get SMILES column as series for efficient access
        smiles_series = self._df.get_column(smiles_col)

        for i in range(self._df.height):
            smiles = smiles_series[i]
            fp = fingerprint(smiles)

            # Skip if fingerprint couldn't be generated or if similar to already seen
            if fp is None or contains_similar(fp, seen_fingerprints):
                continue

            # Mark row as selected
            mask[i] = True
            seen_fingerprints.append(fp)

        # Return filtered dataframe using boolean mask
        return MDF(self._df.filter(ps.Series(mask)))

    def pX(self, column: str, newcol: str = "pIC50", unit: float = 1e-6) -> "MDF":
        """return a new mdf with a pX column calculated as -log10(unit * x) where x is the value in the first column matching the regex pattern"""
        import math

        # Find the first matching column
        matching_cols = self.matching_cols(column)
        col_name = matching_cols[0]

        def calculate_pX(value):
            """Calculate pX value: -log10(unit * x)"""
            if value is None or not isinstance(value, (int, float)):
                return None
            try:
                if value <= 0:
                    return None
                return -math.log10(unit * value)
            except (ValueError, OverflowError, ZeroDivisionError):
                return None

        # Add the new column with pX calculations
        new_df = self._df.with_columns(
            ps.col(col_name)
            .map_elements(calculate_pX, return_dtype=ps.Float64)
            .alias(newcol)
        )

        return MDF(new_df)

    def recent(self, date_pattern: str = "Run Date") -> "MDF":
        """return a new mdf with only the most recent row for each NAME, based on the date column matching the regex pattern"""
        # Find the first matching date column
        date_cols = self.matching_cols(date_pattern)
        date_col = date_cols[0]

        if "NAME" not in self.columns:
            print("Column 'NAME' not found in dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

        # Convert date column to datetime for proper sorting
        # First, try to infer the date format and convert
        try:
            df_with_datetime = self._df.with_columns(
                ps.col(date_col)
                .str.to_datetime(strict=False)
                .alias("__temp_datetime__")
            )
        except:
            # If datetime conversion fails, try to sort as string (fallback)
            print(
                f"Warning: Could not parse dates in column '{date_col}', sorting as strings",
                file=sys.stderr,
            )
            df_with_datetime = self._df.with_columns(
                ps.col(date_col).alias("__temp_datetime__")
            )

        # Group by NAME and get the row with the maximum date for each group
        result = (
            df_with_datetime.sort(
                "__temp_datetime__", descending=True
            )  # Sort by date descending (most recent first)
            .group_by("NAME", maintain_order=True)
            .first()  # Take the first (most recent) row for each NAME
            .drop("__temp_datetime__")  # Remove the temporary datetime column
        )

        return MDF(result)

    def canon(self, smiles_col: str = "SMILES") -> "MDF":
        """return a new mdf with canonical SMILES in the specified column"""
        from rdkit import Chem

        # Replace the SMILES column with canonical SMILES
        new_df = self._df.with_columns(
            ps.col(smiles_col)
            .map_elements(
                lambda x: Chem.CanonSmiles(x) if x else x, return_dtype=ps.Utf8
            )
            .alias(smiles_col)
        )

        return MDF(new_df)

    def ttsplit(self, fraction: float = 0.9, prefix: str = "a"):
        """perform a random test-train split and write to separate files with the given prefix

        Args:
            fraction: Fraction of data to use for training (default: 0.9 for 90% train, 10% test)
            prefix: Output file prefix (default: "a" creates "a_train.csv" and "a_test.csv")
        """
        # Shuffle the dataframe to randomize the split
        shuffled_df = self._df.sample(n=self.height, shuffle=True)

        # Calculate split point
        train_size = int(self.height * fraction)

        # Split into train and test
        train_df = shuffled_df.slice(0, train_size)
        test_df = shuffled_df.slice(train_size, self.height - train_size)

        # Create output filenames
        train_filename = f"{prefix}_train.csv"
        test_filename = f"{prefix}_test.csv"

        # Write to files
        MDF(train_df).write_file(train_filename, MDFFormat.csv)
        MDF(test_df).write_file(test_filename, MDFFormat.csv)

        print(f"Train set: {train_size} rows -> {train_filename}")
        print(f"Test set: {self.height - train_size} rows -> {test_filename}")

    def colnames(self):
        """return the list of column names"""
        return list(self.columns)

    def sieve(self, lilly_medchem_rules=False, relaxed=False,
              unprecedented_rings=False, ring_db=None, PW_alerts=[]):
        """filter molecules using the sieve library"""
        from sieve import Sieve, RING_DB
        if ring_db is None:
            ring_db = RING_DB
        pd_df = self._df.to_pandas()
        filtered_pd = Sieve(relaxed, unprecedented_rings, ring_db,
                            lilly_medchem_rules, PW_alerts)(pd_df)
        return MDF(ps.from_pandas(filtered_pd))


def show_help_and_exit_if_nothing(files: List[Path]):
    """show help and exit if no input files are provided"""
    if sys.stdin.isatty() and not files:
        typer.echo(click.get_current_context().get_help(), err=True)
        raise typer.Exit(code=1)


@app.command()
def canon(
    files: FilesType = FilesArg,
    smiles_col: str = Option(
        "SMILES",
        "-s",
        "--smiles-col",
        help="Column name containing SMILES to canonicalize",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """canonicalize SMILES in the specified column using RDKit"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).canon(smiles_col).write_file(stdout, stdout_fmt)


@app.command()
def cat(
    files: FilesType = FilesArg,
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
    how: CatHow = Option(
        CatHow.diagonal,
        "--how",
        "-H",
        help="type of concatenation: vertical, horizontal, diagonal",
    ),
):
    """concatenate and print mdf files"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt, how=how).write_file(stdout, stdout_fmt)


@app.command()
def colnames(
    files: FilesType = FilesArg,
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
):
    """print column names, one per line"""
    show_help_and_exit_if_nothing(files)
    mdf = MDF.from_stdin_and_files(files, stdin_fmt)
    for col in mdf.colnames():
        print(col)


@app.command()
def cols(
    pattern: str = Argument(help="Regex pattern to match column names"),
    files: FilesType = FilesArg,
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """print only columns matching the regex pattern"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).cols(pattern).write_file(stdout, stdout_fmt)


@app.command()
def diverse(
    files: FilesType = FilesArg,
    smiles_col: str = Option(
        "SMILES",
        "-s",
        "--smiles-col",
        help="Column name containing SMILES to check for similarity",
    ),
    threshold: float = Option(
        0.7,
        "-t",
        "--threshold",
        help="Tanimoto similarity threshold (molecules above this threshold are considered similar)",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """sample rows in order, excluding molecules similar to already selected ones based on Tanimoto similarity of RDKit fingerprints"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).diverse(
        smiles_col, threshold
    ).write_file(stdout, stdout_fmt)


@app.command()
def grep(
    column_pattern: str = Argument(..., help="Regex pattern to match column names"),
    value_pattern: str = Argument(
        ..., help="Regex pattern to match values in the column(s)"
    ),
    files: FilesType = FilesArg,
    invert_match: bool = Option(
        False,
        "-v",
        "--invert-match",
        help="Invert the match, return rows that do NOT match",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """print rows where any column matching the column regex has a value matching the value regex"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).grep(
        column_pattern, value_pattern, invert_match
    ).write_file(stdout, stdout_fmt)


@app.command()
def merge(
    files: FilesType = FilesArg,
    on: str = Option("NAME", "--on", "-k", help="column to merge on"),
    how: MergeHow = Option(
        MergeHow.inner, "--how", "-H", help="type of join: left, right, inner, full"
    ),
    stdin_format: StdinFmtOpt = MDFFormat.csv,
    stdout_format: StdoutFmtOpt = MDFFormat.csv,
):
    """merge multiple mdf files on a column, removing duplicate columns except for the key"""
    show_help_and_exit_if_nothing(files)
    # Get individual MDFs from stdin and files
    mdfs = []
    if not stdin.isatty():
        stdin_data = stdin.read()
        stdin_buf = io.StringIO(stdin_data)
        mdfs.append(MDF.from_file(stdin_buf, stdin_format))
    if files:
        mdfs.extend([MDF.from_file(f) for f in files])
    MDF.merge(mdfs, on=on, how=how).write_file(stdout, stdout_format)


@app.command()
def minmax(
    pattern: str = Argument(
        ..., help="Regex pattern to match column name to filter by"
    ),
    files: FilesType = FilesArg,
    minimum: Optional[float] = Option(
        None, "-m", "--minimum", help="Minimum value (inclusive)"
    ),
    maximum: Optional[float] = Option(
        None, "-M", "--maximum", help="Maximum value (inclusive)"
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """filter rows by numeric range on the first column matching the regex pattern"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).range(
        pattern, minimum, maximum
    ).write_file(stdout, stdout_fmt)


@app.command()
def pareto(
    files: FilesType = FilesArg,
    objectives: str = Option(
        ...,
        "--objective",
        "-O",
        help="Regex pattern to match objective function column names (columns to minimize).",
    ),
    only_non_dominated_front: bool = Option(
        False,
        "--only-first-front",
        "-f",
        help="Return only the first (non-dominated) front. If False, return all fronts with a 'front' column.",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """perform Pareto sorting on objective functions (columns to minimize). Returns all fronts with a 'front' column (0=best, non-dominated front), or only the first front if --only-first-front is specified"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).pareto(
        objectives, only_non_dominated_front
    ).write_file(stdout, stdout_fmt)


@app.command()
def props(
    files: FilesType = FilesArg,
    smiles_col: str = Option(
        "SMILES",
        "-s",
        "--smiles-col",
        help="Column name containing SMILES to calculate properties for",
    ),
    digits: int = Option(
        3,
        "-d",
        "--digits",
        help="Number of significant digits for numerical properties",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """calculate molecular properties from SMILES and add them as new columns (HAC, MW, CLOGP, HBD, HBA, TPSA, SA_SCORE, QED, N_ROT_BONDS, PASSES_RO5)"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).props(smiles_col, digits).write_file(stdout, stdout_fmt)


@app.command(name="pX")
def pX(
    column: str = Argument(
        ..., help="Regex pattern to match column name containing values to convert"
    ),
    files: FilesType = FilesArg,
    newcol: str = Option(
        "pIC50",
        "--newcol",
        "-n",
        help="Name of the new pX column to create",
    ),
    unit: float = Option(
        1e-6,
        "--unit",
        "-u",
        help="Unit multiplier for the calculation (default: 1e-6 for micromolar)",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """calculate a new pX column as -log10(unit * x) where x is the value in the first column matching the regex pattern"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).pX(column, newcol, unit).write_file(stdout, stdout_fmt)


@app.command()
def recent(
    files: FilesType = FilesArg,
    date: str = Option(
        "Run Date",
        "--date",
        "-d",
        help="Regex pattern to match date column name for determining most recent entries",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """keep only the most recent row for each NAME, based on the date column matching the regex pattern"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).recent(date).write_file(stdout, stdout_fmt)


@app.command()
def rename(
    from_pattern: str = Argument(
        ..., help="Regex pattern to match column name to rename"
    ),
    to: str = Argument(..., help="New column name"),
    files: FilesType = FilesArg,
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """rename the first column matching the regex to the new name"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).rename(from_pattern, to).write_file(stdout, stdout_fmt)


@app.command()
def shuffle(
    files: FilesType = FilesArg,
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """shuffle the rows randomly"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).shuffle().write_file(stdout, stdout_fmt)


@app.command()
def sieve(
    files: FilesType = FilesArg,
    lilly: bool = Option(
        False, "-l", "--lilly", help="filter using Lilly Medchem Rules"
    ),
    relaxed: bool = Option(
        False,
        "-r",
        "--relaxed",
        help="Use the -relaxed option for Lilly Medchem Rules",
    ),
    unprecedented_rings: bool = Option(
        False,
        "-R",
        "--unprecedented-rings",
        help="filter out unprecedented rings using LillyMol smi2rings_bdb",
    ),
    ring_db: Optional[str] = Option(
        None, "-d", "--ring-db", help="path to the ring database"
    ),
    PW_alerts: Optional[List[str]] = Option(
        None,
        "-P",
        "--PW-alerts",
        help="filter using one or more Pat Walters REOS alerts (multiple may be given)",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """filter molecules using the sieve library (Lilly Medchem Rules, Pat Walters alerts, ring filters)"""
    show_help_and_exit_if_nothing(files)
    ring_db_path = Path(ring_db) if ring_db is not None else None
    MDF.from_stdin_and_files(files, stdin_fmt).sieve(
        lilly_medchem_rules=lilly,
        relaxed=relaxed,
        unprecedented_rings=unprecedented_rings,
        ring_db=ring_db_path,
        PW_alerts=PW_alerts or [],
    ).write_file(stdout, stdout_fmt)


@app.command()
def smi(
    files: FilesType = FilesArg,
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    how: CatHow = Option(
        CatHow.diagonal,
        "--how",
        "-H",
        help="type of concatenation: vertical, horizontal, diagonal",
    ),
):
    """concatenate and print mdf files in SMILES format (equivalent to cat -o smi)"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt, how=how).write_file(stdout, MDFFormat.smi)


@app.command()
def sort(
    pattern: str = Argument(..., help="Regex pattern to match column name to sort by"),
    files: FilesType = FilesArg,
    reverse: bool = Option(False, "-r", "--reverse", help="Sort in descending order"),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """sort by the first column matching the regex pattern"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).sort(pattern, reverse).write_file(stdout, stdout_fmt)


@app.command()
def split(
    n: int = Argument(..., help="Number of shards to split into"),
    files: FilesType = FilesArg,
    template: str = Option(
        "t%/a.csv",
        "--template",
        "-t",
        help="Output filename template, use % for shard index (default: 't%/a.csv')",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
):
    """split the input into n shards (round-robin) and write each to a file using the template (with % replaced by the shard index); output format is determined by the file suffix (.smi or .csv)"""
    show_help_and_exit_if_nothing(files)
    mdf = MDF.from_stdin_and_files(files, stdin_fmt)
    mdf.split(n, template)


@app.command()
def take(
    n: int = Argument(..., help="Number of random rows to sample"),
    files: FilesType = FilesArg,
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """sample n random rows from the input"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).take(n).write_file(stdout, stdout_fmt)


@app.command()
def ttsplit(
    files: FilesType = FilesArg,
    fraction: float = Option(
        0.9,
        "--fraction",
        "-f",
        help="Fraction of data to use for training (default: 0.9 for 90% train, 10% test)",
    ),
    prefix: str = Option(
        "a",
        "--prefix",
        "-p",
        help="Output file prefix (creates PREFIX_train.csv and PREFIX_test.csv)",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
):
    """perform a random test-train split and write to separate files with the given prefix"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).ttsplit(fraction, prefix)


@app.command()
def uniq(
    files: FilesType = FilesArg,
    smiles_col: str = Option(
        "SMILES",
        "-s",
        "--smiles-col",
        help="Column name containing SMILES to deduplicate by canonical form",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """drop duplicates based on canonical SMILES from the specified column"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).uniq(smiles_col).write_file(stdout, stdout_fmt)


@app.command()
def viz(
    files: FilesType = FilesArg,
    smiles_col: str = Option(
        "SMILES",
        "-s",
        "--smiles-col",
        help="Column name containing SMILES to visualize",
    ),
    mols_per_row: int = Option(
        4,
        "-r",
        "--mols-per-row",
        help="Number of molecules per row in the grid",
    ),
    size: str = Option(
        "250,250",
        "--size",
        help="Size of each molecule image as 'width,height' (default: '250,250')",
    ),
    title: str = Option(
        "molecules",
        "-t",
        "--title",
        help="Title for the visualization (default: 'molecules')",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
):
    """visualize molecules in a grid using RDKit, with NAME as labels"""
    show_help_and_exit_if_nothing(files)
    mdf = MDF.from_stdin_and_files(files, stdin_fmt)

    # Parse size
    try:
        width, height = map(int, size.split(","))
        size_tuple = (width, height)
    except ValueError:
        print(
            f"Invalid size format '{size}'. Expected format: 'width,height' (e.g., '250,250')",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    mdf.viz(smiles_col=smiles_col, mols_per_row=mols_per_row, size=size_tuple, title=title)


if __name__ == "__main__":
    import sys

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        app(["--help"])
    else:
        app()
