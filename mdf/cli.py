import sys
from pathlib import Path
from typing import Annotated, List, Optional

import click
import rich
import typer
from typer import Argument, Exit, Option, Typer

from .core import MDF
from .formats import CatHow, MDFFormat, MergeHow
from .version import __version__


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


def _version_callback(value: bool):
    if value:
        rich.print(f"[bold blue]MDF - Molecular Data Frame[/bold blue]")
        rich.print(f"[dim]version: {__version__}[/dim]")
        raise Exit()


app = Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=f"Molecular data frame utilities for handling molecular data in various formats\n\nVersion: {__version__}",
    no_args_is_help=True,
)


def run():
    """Console script entrypoint."""
    if len(sys.argv) == 1:
        app(["--help"])
    else:
        app()


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
    MDF.from_stdin_and_files(files, stdin_fmt).canon(smiles_col).write_file(sys.stdout, stdout_fmt)


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
    MDF.from_stdin_and_files(files, stdin_fmt, how=how).write_file(sys.stdout, stdout_fmt)


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
    MDF.from_stdin_and_files(files, stdin_fmt).cols(pattern).write_file(sys.stdout, stdout_fmt)


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
    ).write_file(sys.stdout, stdout_fmt)


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
    ).write_file(sys.stdout, stdout_fmt)


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
    MDF.merge(mdfs, on=on, how=how).write_file(sys.stdout, stdout_format)


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
    ).write_file(sys.stdout, stdout_fmt)


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
    ).write_file(sys.stdout, stdout_fmt)


@app.command()
def plot(
    files: FilesType = FilesArg,
    x_col: Annotated[str, Option("-x", "--x-col", help="Regex for x-axis column")] = ...,
    y_cols: Annotated[Optional[List[str]], Option("-y", "--y-col", help="Regex for y-axis columns (repeatable)")] = None,
    x_err_col: Annotated[Optional[str], Option("--xerr", help="Regex for x-error column")] = None,
    y_err_cols: Annotated[Optional[List[str]], Option("--yerr", help="Regex for y-error columns (repeatable; one shared column or one per y column)")] = None,
    title: Annotated[str, Option("-t", "--title", help="Plot title (omit for no title)")] = "",
    output: Annotated[Optional[Path], Option("-o", "--output", help="Write a high-quality PNG plot to this file instead of opening the browser")] = None,
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
):
    """scatter plot of columns matching -x vs -y, with optional x/y error bars; shows Pearson R in legend"""
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
    xerr = None
    if x_err_col:
        xerr_matches = mdf.matching_cols(x_err_col)
        if len(xerr_matches) > 1:
            print(f"Warning: multiple columns match '{x_err_col}', using '{xerr_matches[0]}'", file=sys.stderr)
        xerr = xerr_matches[0]
    yerrs = []
    for pat in y_err_cols or []:
        yerrs.extend(mdf.matching_cols(pat))
    if yerrs and len(yerrs) not in (1, len(ys)):
        print(
            "Number of --yerr columns must be 1 or match the number of resolved -y columns",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)
    mdf.plot(x_col=x, y_cols=ys, title=title, x_err_col=xerr, y_err_cols=yerrs, output=output)


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
    MDF.from_stdin_and_files(files, stdin_fmt).props(smiles_col, digits).write_file(sys.stdout, stdout_fmt)


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
    MDF.from_stdin_and_files(files, stdin_fmt).pX(column, newcol).write_file(sys.stdout, stdout_fmt)


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
    MDF.from_stdin_and_files(files, stdin_fmt).recent(date).write_file(sys.stdout, stdout_fmt)


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
    MDF.from_stdin_and_files(files, stdin_fmt).rename(from_pattern, to).write_file(sys.stdout, stdout_fmt)


@app.command()
def shuffle(
    files: FilesType = FilesArg,
    stdin_fmt: StdinFmtOpt = MDFFormat.csv,
    stdout_fmt: StdoutFmtOpt = MDFFormat.csv,
):
    """shuffle the rows randomly"""
    show_help_and_exit_if_nothing(files)
    MDF.from_stdin_and_files(files, stdin_fmt).shuffle().write_file(sys.stdout, stdout_fmt)


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
    ).write_file(sys.stdout, stdout_fmt)


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
    ).write_file(sys.stdout, stdout_fmt)


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
    MDF.from_stdin_and_files(files, stdin_fmt, how=how).write_file(sys.stdout, MDFFormat.smi)


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
    MDF.from_stdin_and_files(files, stdin_fmt).sort(pattern, reverse).write_file(sys.stdout, stdout_fmt)


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
    MDF.from_stdin_and_files(files, stdin_fmt).take(n).write_file(sys.stdout, stdout_fmt)


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
    MDF.from_stdin_and_files(files, stdin_fmt).uniq(smiles_col).write_file(sys.stdout, stdout_fmt)


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
