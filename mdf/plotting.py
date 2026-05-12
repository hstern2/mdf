import math
import sys
from pathlib import Path

import typer


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


def _write_plot_png(
    output: Path,
    x_col: str,
    title: str,
    series,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks,
    y_ticks,
    colors,
    width_px: int,
    height_px: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def r_text(r):
        return f"r={r:.3f}" if isinstance(r, (int, float)) and math.isfinite(r) else "r=N/A"

    output = output.expanduser()
    if output.suffix.lower() != ".png":
        print("Output filename must end in .png", file=sys.stderr)
        raise typer.Exit(code=1)

    dpi = 300
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(x_col)

    for i, (y_col, xs_data, ys_data, x_errs, y_errs, r) in enumerate(series):
        color = colors[i % len(colors)]
        label = f"{y_col} ({r_text(r)})"
        ax.errorbar(
            xs_data,
            ys_data,
            xerr=x_errs,
            yerr=y_errs,
            fmt="o",
            markersize=4,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=4,
            alpha=0.75,
            markeredgewidth=0.5,
            label=label,
        )

    if len(series) == 1:
        y_col, _, _, _, _, r = series[0]
        ax.set_ylabel(f"{y_col} ({r_text(r)})")
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    try:
        fig.savefig(output, format="png", dpi=dpi, bbox_inches="tight")
        print(f"Wrote plot to '{output}'", file=sys.stderr)
    except OSError as e:
        print(f"Could not write plot to '{output}': {e}", file=sys.stderr)
        raise typer.Exit(code=1) from e
    finally:
        plt.close(fig)
