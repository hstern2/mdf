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
    sdf = "sdf"
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



def _nice_ticks(lo, hi, target=6):
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    span = hi - lo
    raw_step = span / target
    mag = 10 ** math.floor(math.log10(raw_step))
    step = min([1, 2, 2.5, 5, 10], key=lambda s: abs(s * mag - raw_step)) * mag
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 0.01:
        ticks.append(round(t, 10))
        t += step
    if ticks[-1] < hi:
        ticks.append(round(t, 10))
    return ticks[0], ticks[-1], ticks


def _fmt_tick(v):
    if not math.isfinite(v):
        return str(v)
    if v == int(v):
        return str(int(v))
    return f"{v:.3g}"


def _open_html(doc: str):
    import tempfile, webbrowser
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(doc)
        path = f.name
    webbrowser.open(f"file://{path}")


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
        except BrokenPipeError:
            pass  # downstream pipe closed early (e.g. head, grep)
        finally:
            if should_close:
                file.close()

    def viz(
        self,
        smiles_col: str = "SMILES",
        size: tuple = (250, 250),
        title: str = "",
        no_smiles: bool = False,
        columns: int = 1,
    ):
        """write an HTML table of SVG molecule images alongside dataframe columns and open it in the browser"""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.Chem.Draw import rdMolDraw2D
        except ImportError as e:
            print(f"RDKit is required for visualization: {e}", file=sys.stderr)
            raise typer.Exit(code=1)

        from html import escape

        if smiles_col not in self.columns:
            print(f"Column '{smiles_col}' not found in dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

        def render_svg(smi):
            if not smi:
                return None
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            AllChem.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            return re.sub(r"<\?xml[^?]*\?>", "", svg, count=1).lstrip()

        cols = [c for c in self.columns if not (no_smiles and c == smiles_col)]
        single_header = "<th>structure</th>" + "".join(f"<th>{escape(c)}</th>" for c in cols)
        header = single_header * columns

        mol_cells = []
        for row in self._df.iter_rows(named=True):
            svg = render_svg(row.get(smiles_col)) or ""
            cells = f'<td class="mol">{svg}</td>'
            for c in cols:
                v = row.get(c)
                cells += f"<td>{escape('' if v is None else str(v))}</td>"
            mol_cells.append(cells)

        if not mol_cells:
            print("No rows to visualize", file=sys.stderr)
            raise typer.Exit(code=1)

        empty_cells = '<td class="mol"></td>' + "<td></td>" * len(cols)
        body_rows = []
        for i in range(0, len(mol_cells), columns):
            group = mol_cells[i:i + columns]
            group += [empty_cells] * (columns - len(group))
            body_rows.append("<tr>" + "".join(group) + "</tr>")

        doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body {{ font-family: sans-serif; margin: 1em; }}
h1 {{ font-size: 1.2em; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ddd; padding: 4px; vertical-align: middle; }}
th {{ background: #f4f4f4; text-align: left; position: sticky; top: 0; }}
td.mol {{ text-align: center; }}
td.mol svg {{ display: block; }}
</style>
</head>
<body>
{f'<h1>{escape(title)}</h1>' if title else ''}
<table>
<thead><tr>{header}</tr></thead>
<tbody>
{''.join(body_rows)}
</tbody>
</table>
</body>
</html>
"""

        _open_html(doc)

    def plot(self, x_col: str, y_cols: List[str], title: str = ""):
        """generate a scatter plot and open it in the browser"""
        from html import escape

        COLORS = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        series = []
        for y_col in y_cols:
            sub = self._df.select([x_col, y_col]).cast(ps.Float64, strict=False).drop_nulls()
            if sub.is_empty():
                print(f"No numeric data for column '{y_col}'", file=sys.stderr)
                continue
            xs_s, ys_s = sub[x_col], sub[y_col]
            r = sub.select(ps.corr(x_col, y_col)).item()
            series.append((y_col, xs_s.to_list(), ys_s.to_list(), r))

        if not series:
            print("No data to plot", file=sys.stderr)
            raise typer.Exit(code=1)

        all_x = [x for _, xs, _, _ in series for x in xs]
        all_y = [y for _, _, ys, _ in series for y in ys]
        x_min, x_max, x_ticks = _nice_ticks(min(all_x), max(all_x))
        y_min, y_max, y_ticks = _nice_ticks(min(all_y), max(all_y))

        W, H = 740, 500
        ml, mr, mt, mb = 70, 210 if len(series) > 1 else 30, 40, 55
        pw, ph = W - ml - mr, H - mt - mb

        def to_svg(x, y):
            px = ml + (x - x_min) / (x_max - x_min) * pw
            py = mt + ph - (y - y_min) / (y_max - y_min) * ph
            return px, py

        els = []
        els.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#ccc"/>')

        for t in x_ticks:
            px, py = to_svg(t, y_min)
            els.append(f'<line x1="{px:.1f}" y1="{mt}" x2="{px:.1f}" y2="{mt+ph}" stroke="#eee"/>')
            els.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{py+5:.1f}" stroke="#555"/>')
            els.append(f'<text x="{px:.1f}" y="{py+18:.1f}" text-anchor="middle" font-size="11" fill="#444">{_fmt_tick(t)}</text>')
        for t in y_ticks:
            px, py = to_svg(x_min, t)
            els.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml+pw}" y2="{py:.1f}" stroke="#eee"/>')
            els.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px-5:.1f}" y2="{py:.1f}" stroke="#555"/>')
            els.append(f'<text x="{px-8:.1f}" y="{py+4:.1f}" text-anchor="end" font-size="11" fill="#444">{_fmt_tick(t)}</text>')

        els.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#555" stroke-width="1.5"/>')
        els.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#555" stroke-width="1.5"/>')

        cx = ml + pw / 2
        els.append(f'<text x="{cx:.1f}" y="{H-8}" text-anchor="middle" font-size="13" fill="#333">{escape(x_col)}</text>')
        cy = mt + ph / 2
        if len(series) == 1:
            y_col_name, _, _, r = series[0]
            r_str = f" (r={r:.3f})" if not math.isnan(r) else ""
            y_label = escape(y_col_name) + r_str
        else:
            y_label = ""
        if y_label:
            els.append(f'<text transform="rotate(-90,16,{cy:.1f})" x="16" y="{cy:.1f}" text-anchor="middle" font-size="13" fill="#333">{y_label}</text>')
        if title:
            els.append(f'<text x="{W/2:.1f}" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#222">{escape(title)}</text>')

        for i, (y_col, xs_data, ys_data, _) in enumerate(series):
            color = COLORS[i % len(COLORS)]
            for x, y in zip(xs_data, ys_data):
                px, py = to_svg(x, y)
                els.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" fill-opacity="0.65" stroke="{color}" stroke-width="0.5"/>')

        if len(series) > 1:
            lx, ly = ml + pw + 15, mt + 10
            for i, (y_col, _, _, r) in enumerate(series):
                color = COLORS[i % len(COLORS)]
                r_str = f"r={r:.3f}" if not math.isnan(r) else "r=N/A"
                els.append(f'<rect x="{lx}" y="{ly + i*22}" width="12" height="12" fill="{color}" fill-opacity="0.8"/>')
                els.append(f'<text x="{lx+16}" y="{ly + i*22 + 10}" font-size="11" fill="#333">{escape(y_col)} ({r_str})</text>')

        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">\n{"".join(els)}\n</svg>'
        doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>body {{ font-family: sans-serif; margin: 2em; background: #fff; }}</style>
</head><body>{svg}</body></html>"""

        _open_html(doc)

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
        """create mdf from smiles file (one SMILES [NAME] per line)"""
        if isinstance(fn, (str, Path)):
            with open(fn) as f:
                lines = f.readlines()
        else:
            lines = fn.readlines()

        smiles, names = [], []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            smiles.append(parts[0])
            names.append(parts[1] if len(parts) > 1 else "")

        return cls(ps.DataFrame(
            {"SMILES": smiles, "NAME": names},
            schema={"SMILES": ps.Utf8, "NAME": ps.Utf8},
        ))

    @classmethod
    def from_sdf(cls, f):
        """create mdf from SDF file; produces NAME, SMILES, FILE columns plus any SDF tags"""
        try:
            from rdkit.Chem import SDMolSupplier, ForwardSDMolSupplier, MolToSmiles
        except ImportError as e:
            print(f"RDKit is required for SDF input: {e}", file=sys.stderr)
            raise typer.Exit(code=1)

        if isinstance(f, (str, Path)):
            filename = str(f)
            supplier = SDMolSupplier(filename)
        else:
            filename = getattr(f, "name", "<stdin>")
            data = f.read()
            if isinstance(data, str):
                data = data.encode()
            supplier = ForwardSDMolSupplier(io.BytesIO(data))

        rows = []
        for m in supplier:
            if m is None:
                continue
            d = {
                "NAME": m.GetProp("_Name") if m.HasProp("_Name") else "",
                "SMILES": MolToSmiles(m),
                "FILE": filename,
            }
            d.update(m.GetPropsAsDict())
            rows.append(d)
        return cls(ps.DataFrame(rows))

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
        """create mdf from file, automatically detecting fmt from suffix"""
        if fmt is None:
            if hasattr(f, "name"):
                suffix = Path(f.name).suffix
            elif isinstance(f, (str, Path)):
                suffix = Path(f).suffix
            else:
                suffix = ""
            fmt = {".smi": MDFFormat.smi, ".sdf": MDFFormat.sdf}.get(
                suffix, MDFFormat.csv
            )
        if fmt == MDFFormat.csv:
            return cls.from_csv(f)
        if fmt == MDFFormat.smi:
            return cls.from_smi(f)
        if fmt == MDFFormat.sdf:
            return cls.from_sdf(f)
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
    def mdfs_from_stdin_and_files(
        cls, files: Optional[List[Path]], stdin_fmt: MDFFormat
    ) -> List["MDF"]:
        """return a list of MDFs read from stdin (if not isatty) and files"""
        mdfs = []
        if not stdin.isatty():
            stdin_data = stdin.read()
            if stdin_data.strip():
                mdfs.append(cls.from_file(io.StringIO(stdin_data), stdin_fmt))
        if files:
            mdfs.extend(cls.from_file(f) for f in files)
        return mdfs

    @classmethod
    def from_stdin_and_files(
        cls, files: List[Path], stdin_fmt: MDFFormat, how: CatHow = CatHow.diagonal
    ) -> "MDF":
        """create mdf by concatenating stdin (if not isatty) and files"""
        mdfs = cls.mdfs_from_stdin_and_files(files, stdin_fmt)
        if not mdfs:
            return cls(ps.DataFrame())
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
        except ImportError as e:
            print(
                f"RDKit is required for molecular property calculations: {e}",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)

        try:
            sys.path.append(f"{Chem.RDConfig.RDContribDir}/SA_Score")
            import sascorer
        except (ImportError, AttributeError):
            print("Warning: SA_Score not available, using dummy values", file=sys.stderr)
            sascorer = None

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
        TanimotoSimilarity, fingerprint = self._similarity_tools()

        if smiles_col not in self.columns:
            print(f"Column '{smiles_col}' not found in dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

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

    def sims(
        self,
        to: "MDF",
        smiles_col: str = "SMILES",
        to_smiles_col: str = "SMILES",
        sort: bool = False,
    ) -> "MDF":
        """return a new mdf with additional columns tanimoto_similarity_to_NAME for molecules in another mdf"""
        TanimotoSimilarity, fingerprint = self._similarity_tools()

        if smiles_col not in self.columns:
            print(f"Column '{smiles_col}' not found in dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

        if to_smiles_col not in to.columns:
            print(
                f"Column '{to_smiles_col}' not found in target dataframe",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)

        if "NAME" not in to.columns:
            print("Column 'NAME' not found in target dataframe", file=sys.stderr)
            raise typer.Exit(code=1)

        # Prepare fingerprints for the target dataframe
        to_smiles_series = to._df.get_column(to_smiles_col)
        to_name_series = to._df.get_column("NAME")

        target_fps = []
        target_colnames = []

        # Track counts to ensure unique and safe column names, even with duplicate or
        # problematic NAME values.
        name_counts = {}

        for smi, raw_name in zip(to_smiles_series, to_name_series):
            fp = fingerprint(smi)
            if fp is None:
                continue
            target_fps.append(fp)
            base_name = str(raw_name) if raw_name is not None else ""
            # Sanitize NAME to make a safe column suffix
            safe_name = re.sub(r"\W+", "_", base_name).strip("_") or "target"
            count = name_counts.get(safe_name, 0)
            name_counts[safe_name] = count + 1
            if count > 0:
                safe_name = f"{safe_name}_{count}"
            target_colnames.append(f"tanimoto_similarity_to_{safe_name}")

        if not target_fps:
            print(
                "No valid fingerprints could be generated from target dataframe",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)

        # Prepare storage for similarity columns
        n_rows = self._df.height
        similarity_data = {colname: [None] * n_rows for colname in target_colnames}

        smiles_series = self._df.get_column(smiles_col)

        for i in range(n_rows):
            smi = smiles_series[i]
            fp = fingerprint(smi)
            if fp is None:
                continue
            for colname, target_fp in zip(target_colnames, target_fps):
                try:
                    similarity_data[colname][i] = float(
                        TanimotoSimilarity(fp, target_fp)
                    )
                except Exception:
                    similarity_data[colname][i] = None

        similarity_df = ps.DataFrame(similarity_data)
        result_df = self._df.hstack(similarity_df)
        if sort:
            result_df = result_df.sort(target_colnames[0], descending=True, nulls_last=True)
        return MDF(result_df)

    _UNIT_MULTIPLIERS = {
        "M": 1,
        "mM": 1e-3,
        "uM": 1e-6,
        "µM": 1e-6,
        "nM": 1e-9,
        "pM": 1e-12,
    }

    @staticmethod
    def _parse_unit_from_column(col_name: str) -> float:
        """Parse a concentration unit from a column name. Looks for known unit strings (e.g. uM, nM) in the column name."""
        for unit_str, multiplier in MDF._UNIT_MULTIPLIERS.items():
            if re.search(r'(?<![a-zA-Z])' + re.escape(unit_str) + r'(?![a-zA-Z])', col_name):
                return multiplier
        raise ValueError(
            f"Could not determine unit from column name '{col_name}'. "
            f"Expected one of: {', '.join(MDF._UNIT_MULTIPLIERS.keys())}"
        )

    def pX(self, column: str, newcol: str = "pIC50") -> "MDF":
        """return a new mdf with a pX column calculated as -log10(unit * x), with the unit inferred from the matched column name"""
        # Find the first matching column
        matching_cols = self.matching_cols(column)
        col_name = matching_cols[0]
        unit = self._parse_unit_from_column(col_name)

        def calculate_pX(value):
            """Calculate pX value: -log10(unit * x)"""
            if value is None:
                return None
            try:
                value = float(value)
                if value <= 0:
                    return None
                return -math.log10(unit * value)
            except (ValueError, TypeError, OverflowError, ZeroDivisionError):
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

        # Parse string dates to datetime so sorting is chronological; other types sort as-is.
        col = ps.col(date_col)
        sort_key = col.str.to_datetime(strict=False) if self._df.schema[date_col] == ps.Utf8 else col

        return MDF(
            self._df.with_columns(sort_key.alias("__sort__"))
            .sort("__sort__", descending=True)
            .group_by("NAME", maintain_order=True)
            .first()
            .drop("__sort__")
        )

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

    @staticmethod
    def _similarity_tools():
        """Return RDKit TanimotoSimilarity function and a fingerprint generator"""
        try:
            from rdkit.DataStructs import TanimotoSimilarity
            from rdkit.Chem import AllChem, MolFromSmiles
            from rdkit.Chem.MolStandardize.rdMolStandardize import (
                TautomerEnumerator,
            )
        except ImportError as e:
            print(
                f"RDKit is required for molecular similarity calculations: {e}",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)

        fpgen = AllChem.GetRDKitFPGenerator()
        te = TautomerEnumerator()

        def fingerprint(smi: str):
            try:
                if smi is None or not str(smi).strip():
                    return None
                mol = MolFromSmiles(str(smi))
                if mol is None:
                    return None
                return fpgen.GetFingerprint(te.Canonicalize(mol))
            except (ValueError, TypeError, AttributeError):
                return None

        return TanimotoSimilarity, fingerprint

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
    for col in mdf.columns:
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
    mdfs = MDF.mdfs_from_stdin_and_files(files, stdin_format)
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
def plot(
    files: FilesType = FilesArg,
    x_col: Annotated[str, Option("-x", "--x-col", help="Regex for x-axis column")] = ...,
    y_cols: Annotated[Optional[List[str]], Option("-y", "--y-col", help="Regex for y-axis columns (repeatable)")] = None,
    title: Annotated[str, Option("-t", "--title", help="Plot title (omit for no title)")] = "",
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
):
    """scatter plot of columns matching -x (x axis) vs -y (y axis, repeatable); shows Pearson R in legend"""
    show_help_and_exit_if_nothing(files)
    mdf = MDF.from_stdin_and_files(files, stdin_fmt)
    x_matches = mdf.matching_cols(x_col)
    if len(x_matches) > 1:
        print(f"Warning: multiple columns match '{x_col}', using '{x_matches[0]}'", file=sys.stderr)
    x = x_matches[0]
    y_patterns = y_cols or []
    if not y_patterns:
        print("At least one -y pattern is required", file=sys.stderr)
        raise typer.Exit(code=1)
    ys = []
    for pat in y_patterns:
        ys.extend(mdf.matching_cols(pat))
    mdf.plot(x_col=x, y_cols=ys, title=title)


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
        ..., help="Regex pattern to match column name containing values to convert (unit is inferred from the column name, e.g. uM, nM, mM)"
    ),
    files: FilesType = FilesArg,
    newcol: str = Option(
        "pIC50",
        "--newcol",
        "-n",
        help="Name of the new pX column to create",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """calculate a new pX column as -log10(unit * x), with the unit (M, mM, uM, nM, pM) inferred from the matched column name"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).pX(column, newcol).write_file(stdout, stdout_fmt)


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
    PW_alerts: List[str] = Option(
        [],
        "-P",
        "--PW-alerts",
        help="filter using one or more Pat Walters REOS alerts (multiple may be given): all, Glaxo, Dundee, BMS, PAINS, SureChEMBL, MLSMR, Inpharmatica, LINT",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """filter molecules using the sieve library (Lilly Medchem Rules, Pat Walters alerts, ring filters)"""
    show_help_and_exit_if_nothing(files)
    from sieve import PW_Alert_Name
    valid = {e.value for e in PW_Alert_Name}
    parsed_alerts = []
    for a in PW_alerts:
        if a not in valid:
            print(f"Invalid PW alert '{a}'. Valid values: {', '.join(sorted(valid))}", file=sys.stderr)
            raise typer.Exit(code=1)
        parsed_alerts.append(PW_Alert_Name(a))
    ring_db_path = Path(ring_db) if ring_db is not None else None
    MDF.from_stdin_and_files(files, stdin_fmt).sieve(
        lilly_medchem_rules=lilly,
        relaxed=relaxed,
        unprecedented_rings=unprecedented_rings,
        ring_db=ring_db_path,
        PW_alerts=parsed_alerts,
    ).write_file(stdout, stdout_fmt)


@app.command()
def sims(
    files: FilesType = FilesArg,
    to: Path = Option(
        ...,
        "-t",
        "--to",
        help="Target dataframe file (.csv or .smi) to compute similarities to",
    ),
    smiles_col: str = Option(
        "SMILES",
        "-s",
        "--smiles-col",
        help="Column name containing SMILES in the primary dataframe",
    ),
    to_smiles_col: str = Option(
        "SMILES",
        "--to-smiles-col",
        help="Column name containing SMILES in the target dataframe",
    ),
    sort: bool = Option(
        False,
        "-S",
        "--sort",
        help="Sort by the first new Tanimoto similarity column (descending, most similar at the top)",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """add Tanimoto similarity columns to the input dataframe for each molecule in a target dataframe"""
    show_help_and_exit_if_nothing(files)
    mdf = MDF.from_stdin_and_files(files, stdin_fmt)
    to_mdf = MDF.from_file(to)
    mdf.sims(
        to_mdf, smiles_col=smiles_col, to_smiles_col=to_smiles_col, sort=sort
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
    size: str = Option(
        "250,250",
        "--size",
        help="Size of each molecule image as 'width,height' (default: '250,250')",
    ),
    title: str = Option(
        "",
        "-t",
        "--title",
        help="Title for the visualization (default: no title)",
    ),
    no_smiles: bool = Option(
        False,
        "-n",
        "--no-smiles",
        help="Omit the SMILES column from the table",
    ),
    columns: int = Option(
        1,
        "-c",
        "--columns",
        help="Number of molecule columns per row (default: 1)",
    ),
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
):
    """write an HTML table of SVG molecule images alongside dataframe columns and open it in the browser"""
    show_help_and_exit_if_nothing(files)
    mdf = MDF.from_stdin_and_files(files, stdin_fmt)

    try:
        width, height = map(int, size.split(","))
        size_tuple = (width, height)
    except ValueError:
        print(
            f"Invalid size format '{size}'. Expected format: 'width,height' (e.g., '250,250')",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    mdf.viz(smiles_col=smiles_col, size=size_tuple, title=title, no_smiles=no_smiles, columns=columns)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        app(["--help"])
    else:
        app()
