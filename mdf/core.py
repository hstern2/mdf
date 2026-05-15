import io
import math
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Union

import polars as ps
import typer
from polars.exceptions import SchemaError

from .constants import FLOAT_PROPS, PROP_NAMES, TITLE_COLUMN_PRIORITY
from .formats import CatHow, MDFFormat, MergeHow
from .plotting import _fmt_tick, _nice_ticks, _open_html, _plot_label_text, _write_plot_png


_SDF_MOLBLOCK_COLUMN = "__MOLBLOCK"


def _has_3d_coords(mol) -> bool:
    """Return True when a molecule has at least one non-zero Z coordinate."""
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    return any(
        abs(conf.GetAtomPosition(i).z) > 1e-8
        for i in range(mol.GetNumAtoms())
    )


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

    def _infer_title_column(self) -> Optional[str]:
        """Return the best column to use as a molecule title, if one exists."""
        columns = list(self.columns)
        column_set = set(columns)
        for col in TITLE_COLUMN_PRIORITY:
            if col in column_set:
                return col
        for col in columns:
            if col in {"SMILES", "FILE"}:
                continue
            if re.search(r"name|title|id", col, re.IGNORECASE):
                return col
        return None

    @staticmethod
    def _title_value(value) -> str:
        """Render a title-like field value without treating nulls as names."""
        if value is None:
            return ""
        try:
            if math.isnan(value):
                return ""
        except TypeError:
            pass
        return str(value).strip()

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
                if _SDF_MOLBLOCK_COLUMN in self.columns:
                    self._df.drop(_SDF_MOLBLOCK_COLUMN).write_csv(file)
                else:
                    self._df.write_csv(file)
            elif fmt == MDFFormat.smi:
                if "SMILES" not in self.columns:
                    print("Column 'SMILES' not found in dataframe", file=sys.stderr)
                    raise typer.Exit(code=1)
                title_col = self._infer_title_column()
                for row in self._df.iter_rows(named=True):
                    smiles = row.get("SMILES")
                    smiles = "" if smiles is None else str(smiles)
                    title = self._title_value(row.get(title_col)) if title_col else ""
                    print(f"{smiles} {title}" if title else smiles, file=file)
            elif fmt == MDFFormat.sdf:
                try:
                    from rdkit import Chem
                    from rdkit.Chem import AllChem
                except ImportError as e:
                    print(f"RDKit is required for SDF output: {e}", file=sys.stderr)
                    raise typer.Exit(code=1)
                if "SMILES" not in self.columns:
                    print("Column 'SMILES' not found in dataframe", file=sys.stderr)
                    raise typer.Exit(code=1)
                title_col = self._infer_title_column()
                skip = {"SMILES", "FILE", _SDF_MOLBLOCK_COLUMN}
                for row in self._df.iter_rows(named=True):
                    mol = None
                    molblock = row.get(_SDF_MOLBLOCK_COLUMN)
                    if molblock:
                        mol = Chem.MolFromMolBlock(
                            str(molblock), sanitize=True, removeHs=False
                        )
                    if mol is None:
                        smiles = row.get("SMILES", "")
                        mol = Chem.MolFromSmiles(smiles) if smiles else None
                        if mol is not None:
                            AllChem.Compute2DCoords(mol)
                    if mol is None:
                        continue
                    title = self._title_value(row.get(title_col)) if title_col else ""
                    mol.SetProp("_Name", title)
                    file.write(Chem.MolToMolBlock(mol))
                    for col, val in row.items():
                        if col not in skip and val is not None:
                            file.write(f">  <{col}>\n{val}\n\n")
                    file.write("$$$$\n")
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

    def plot(
        self,
        x_col: str,
        y_cols: List[str],
        title: str = "",
        x_err_col: Optional[str] = None,
        y_err_cols: Optional[List[str]] = None,
        output: Optional[Path] = None,
        label_col: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
    ):
        """generate a scatter plot and open it in the browser or write it to PNG"""
        from html import escape

        COLORS = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        y_err_cols = y_err_cols or []
        if y_err_cols and len(y_err_cols) not in (1, len(y_cols)):
            print(
                "Number of y-error columns must be 1 or match the number of y columns",
                file=sys.stderr,
            )
            raise typer.Exit(code=1)
        y_err_for_y = (
            [y_err_cols[0]] * len(y_cols)
            if len(y_err_cols) == 1
            else y_err_cols or [None] * len(y_cols)
        )

        series = []
        for y_col, y_err_col in zip(y_cols, y_err_for_y):
            numeric_cols = [x_col, y_col]
            if x_err_col:
                numeric_cols.append(x_err_col)
            if y_err_col:
                numeric_cols.append(y_err_col)
            numeric_cols = list(dict.fromkeys(numeric_cols))
            cols = numeric_cols + ([label_col] if label_col else [])
            sub = self._df.select(list(dict.fromkeys(cols)))
            sub = sub.with_columns(
                [ps.col(col).cast(ps.Float64, strict=False) for col in numeric_cols]
            ).drop_nulls(numeric_cols)
            if sub.is_empty():
                print(f"No numeric data for column '{y_col}'", file=sys.stderr)
                continue
            xs_s, ys_s = sub[x_col], sub[y_col]
            r = sub.select(ps.corr(x_col, y_col)).item()
            x_errs = [abs(v) for v in sub[x_err_col].to_list()] if x_err_col else None
            y_errs = [abs(v) for v in sub[y_err_col].to_list()] if y_err_col else None
            point_labels = sub[label_col].to_list() if label_col else None
            series.append((y_col, xs_s.to_list(), ys_s.to_list(), x_errs, y_errs, point_labels, r))

        if not series:
            print("No data to plot", file=sys.stderr)
            raise typer.Exit(code=1)

        all_x = []
        all_y = []
        for _, xs, ys, x_errs, y_errs, _, _ in series:
            if x_errs is None:
                all_x.extend(xs)
            else:
                for x, err in zip(xs, x_errs):
                    all_x.extend([x - err, x + err])
            if y_errs is None:
                all_y.extend(ys)
            else:
                for y, err in zip(ys, y_errs):
                    all_y.extend([y - err, y + err])
        x_min, x_max, x_ticks = _nice_ticks(min(all_x), max(all_x))
        y_min, y_max, y_ticks = _nice_ticks(min(all_y), max(all_y))

        W, H = 740, 500
        ml, mr, mt, mb = 70, 210 if len(series) > 1 else 30, 40, 55
        pw, ph = W - ml - mr, H - mt - mb

        if output:
            _write_plot_png(
                output,
                x_col,
                title,
                series,
                x_min,
                x_max,
                y_min,
                y_max,
                x_ticks,
                y_ticks,
                COLORS,
                W,
                H,
                x_label=x_label,
                y_label=y_label,
            )
            return

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
        axis_x_label = x_label if x_label is not None else x_col
        if axis_x_label:
            els.append(f'<text x="{cx:.1f}" y="{H-8}" text-anchor="middle" font-size="13" fill="#333">{escape(axis_x_label)}</text>')
        cy = mt + ph / 2
        if y_label is not None:
            axis_y_label = escape(y_label)
        elif len(series) == 1:
            y_col_name, _, _, _, _, _, r = series[0]
            r_str = f" (r={r:.3f})" if not math.isnan(r) else ""
            axis_y_label = escape(y_col_name) + r_str
        else:
            axis_y_label = ""
        if axis_y_label:
            els.append(f'<text transform="rotate(-90,16,{cy:.1f})" x="16" y="{cy:.1f}" text-anchor="middle" font-size="13" fill="#333">{axis_y_label}</text>')
        if title:
            els.append(f'<text x="{W/2:.1f}" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#222">{escape(title)}</text>')

        cap = 4
        for i, (y_col, xs_data, ys_data, x_errs, y_errs, point_labels, _) in enumerate(series):
            color = COLORS[i % len(COLORS)]
            for j, (x, y) in enumerate(zip(xs_data, ys_data)):
                px, py = to_svg(x, y)
                if x_errs is not None and x_errs[j] > 0:
                    x1, _ = to_svg(x - x_errs[j], y)
                    x2, _ = to_svg(x + x_errs[j], y)
                    els.append(f'<line x1="{x1:.1f}" y1="{py:.1f}" x2="{x2:.1f}" y2="{py:.1f}" stroke="{color}" stroke-opacity="0.55" stroke-width="1.2"/>')
                    els.append(f'<line x1="{x1:.1f}" y1="{py-cap:.1f}" x2="{x1:.1f}" y2="{py+cap:.1f}" stroke="{color}" stroke-opacity="0.55" stroke-width="1.2"/>')
                    els.append(f'<line x1="{x2:.1f}" y1="{py-cap:.1f}" x2="{x2:.1f}" y2="{py+cap:.1f}" stroke="{color}" stroke-opacity="0.55" stroke-width="1.2"/>')
                if y_errs is not None and y_errs[j] > 0:
                    _, y1 = to_svg(x, y - y_errs[j])
                    _, y2 = to_svg(x, y + y_errs[j])
                    y_top, y_bottom = sorted((y1, y2))
                    els.append(f'<line x1="{px:.1f}" y1="{y_top:.1f}" x2="{px:.1f}" y2="{y_bottom:.1f}" stroke="{color}" stroke-opacity="0.55" stroke-width="1.2"/>')
                    els.append(f'<line x1="{px-cap:.1f}" y1="{y_top:.1f}" x2="{px+cap:.1f}" y2="{y_top:.1f}" stroke="{color}" stroke-opacity="0.55" stroke-width="1.2"/>')
                    els.append(f'<line x1="{px-cap:.1f}" y1="{y_bottom:.1f}" x2="{px+cap:.1f}" y2="{y_bottom:.1f}" stroke="{color}" stroke-opacity="0.55" stroke-width="1.2"/>')
                els.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" fill-opacity="0.65" stroke="{color}" stroke-width="0.5"/>')
                if point_labels:
                    point_label = _plot_label_text(point_labels[j])
                    if point_label:
                        els.append(f'<text x="{px+6:.1f}" y="{py-6:.1f}" font-size="10" fill="#333" stroke="#fff" stroke-width="3" paint-order="stroke">{escape(point_label)}</text>')

        if len(series) > 1:
            lx, ly = ml + pw + 15, mt + 10
            for i, (y_col, _, _, _, _, _, r) in enumerate(series):
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

    def drop(self, pattern: str) -> "MDF":
        """return a new mdf without columns matching the regex pattern"""
        return MDF(self._df.drop(self.matching_cols(pattern)))

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
        return cls(ps.read_csv(f, infer_schema_length=None))

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
        """create mdf from SDF file; produces NAME, SMILES columns plus any SDF tags"""
        try:
            from rdkit import Chem
        except ImportError as e:
            print(f"RDKit is required for SDF input: {e}", file=sys.stderr)
            raise typer.Exit(code=1)

        if isinstance(f, (str, Path)):
            supplier = Chem.SDMolSupplier(str(f), removeHs=False)
        else:
            data = f.read()
            if isinstance(data, str):
                data = data.encode()
            supplier = Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=False)

        rows = []
        for m in supplier:
            if m is None:
                continue
            try:
                smiles_mol = Chem.RemoveHs(m)
            except Exception:
                smiles_mol = m
            d = {
                "NAME": m.GetProp("_Name") if m.HasProp("_Name") else "",
                "SMILES": Chem.MolToSmiles(smiles_mol),
            }
            props = m.GetPropsAsDict()
            props.pop(_SDF_MOLBLOCK_COLUMN, None)
            d.update(props)
            if _has_3d_coords(m):
                d[_SDF_MOLBLOCK_COLUMN] = Chem.MolToMolBlock(m)
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
        if not sys.stdin.isatty():
            stdin_data = sys.stdin.read()
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

    def stats(self) -> "MDF":
        """return summary statistics for each numeric column, excluding nulls and NaNs"""
        rows = []
        for col, dtype in zip(self.columns, self._df.dtypes):
            if not dtype.is_numeric():
                continue
            values = self._df.select(
                ps.col(col).cast(ps.Float64, strict=False).drop_nans().drop_nulls()
            ).to_series()
            rows.append(
                {
                    "column": col,
                    "count": len(values),
                    "mean": values.mean(),
                    "stddev": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
        return MDF(
            ps.DataFrame(
                rows,
                schema={
                    "column": ps.Utf8,
                    "count": ps.Int64,
                    "mean": ps.Float64,
                    "stddev": ps.Float64,
                    "min": ps.Float64,
                    "max": ps.Float64,
                },
            )
        )

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
