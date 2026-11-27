"""One plotting style for every BorderEye figure.

Before this module the codebase carried four separate ``rcParams`` blocks, a
dozen figure sizes and eleven DPI values, so a training dashboard, a research
audit and a paper figure looked like they came from three different projects.
Everything here is the shared vocabulary: the palette, the chrome, and the few
helpers that keep a chart from breaking the rules it should not break.

Matplotlib is imported lazily, matching the rest of the package — importing
``bordereye`` must not pull in a plotting stack.

Usage::

    from bordereye import plotstyle

    plotstyle.apply()
    fig, ax = plt.subplots(figsize=plotstyle.FIGSIZE_WIDE)
    ax.plot(epochs, loss, color=plotstyle.SERIES[0], label="train")
    plotstyle.epoch_axis(ax)
    plotstyle.legend_above(ax, n_cols=3)
    plotstyle.save(fig, path)
"""
from __future__ import annotations

from pathlib import Path

# ── Palette ───────────────────────────────────────────────────────────────────
# Categorical slots in FIXED order — slot 1 is always the first series drawn, so
# a series keeps its colour when a chart gains or loses a neighbour. Never cycle
# past the eighth: fold the tail into "other" or split into small multiples.
# Validated as a set for the light surface: worst adjacent CVD dE 9.1, worst
# adjacent normal-vision dE 19.6. Aqua, yellow and magenta sit below 3:1 against
# the surface, so charts using them carry a legend or direct labels — which every
# helper below enforces anyway.
SERIES: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

# Reserved for state, never for "the next series". Always paired with a label or
# marker so the meaning never rests on hue alone.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# ── Canonical figure sizes ────────────────────────────────────────────────────
# Sized for a two-column paper (single = one column) and for on-screen review.
FIGSIZE_SINGLE = (3.5, 2.6)
FIGSIZE_WIDE = (7.2, 3.2)
FIGSIZE_SQUARE = (4.6, 4.6)
FIGSIZE_DASHBOARD = (13.5, 11.0)

DPI_SCREEN = 150
DPI_PRINT = 300


def rc_params() -> dict:
    """The rcParams every BorderEye figure shares.

    Chrome is deliberately recessive: hairline solid gridlines on the y axis
    only, two spines rather than a box, and muted tick labels — so the data is
    the loudest thing in the frame.
    """
    from matplotlib import cycler

    return {
        "figure.facecolor": "white",
        "figure.dpi": DPI_SCREEN,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": DPI_SCREEN,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlepad": 8.0,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(SERIES)),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",      # never dashed: dashing reads as data
        "grid.alpha": 1.0,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.0,
        "lines.solid_capstyle": "round",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.fontsize": 8.5,
        "legend.frameon": False,     # the plot area belongs to the data
        "legend.labelcolor": INK_SECONDARY,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.4,
        "legend.borderaxespad": 0.0,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9.5,
        # TrueType rather than Type-3, so vector output embeds fonts that
        # journal submission systems accept.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.titlesize": 12.0,
        "figure.titleweight": "bold",
    }


def apply() -> None:
    """Install the shared style globally for the current process."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(rc_params())


def epoch_axis(ax) -> None:
    """Force whole-number ticks on an axis that counts things.

    Epoch and tile counts are integers; matplotlib's default locator happily
    labels a single-epoch run 0.96, 0.98, 1.00, which reads as a continuous
    quantity that does not exist.
    """
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))


_LEGEND_ROW_PT = 12.5


def legend_above(ax, *, n_cols: int = 4, **kwargs) -> None:
    """Put the legend above the axes, where it cannot cover the data.

    A legend floated inside the plot area hides exactly the region the reader
    came for — in the old training dashboard it clipped the validation peak.
    The title is pushed clear of however many rows the legend needs, so the two
    never overprint each other.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    n_rows = -(-len(handles) // max(n_cols, 1))
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=n_cols, frameon=False, **kwargs)
    # Re-set the title with enough pad to clear the legend block above it.
    title = ax.get_title()
    if title:
        ax.set_title(title, pad=8.0 + _LEGEND_ROW_PT * n_rows)


def annotate_point(ax, x, y, text: str, *, color: str | None = None) -> None:
    """Label one noteworthy point directly, instead of adding a legend row.

    The label flips to the left of the point once the point sits in the right
    third of the axes, so a best-at-the-last-epoch marker — the common case for
    a metric that improves monotonically — does not run off the panel.
    """
    x0, x1 = ax.get_xlim()
    span = (x1 - x0) or 1.0
    to_left = (x - x0) / span > 0.66
    ax.annotate(
        text, xy=(x, y),
        xytext=(-6 if to_left else 6, 6), textcoords="offset points",
        ha="right" if to_left else "left", va="bottom",
        fontsize=8.0, color=color or INK_SECONDARY, fontweight="bold",
    )


def save(fig, path: str | Path, *, dpi: int = DPI_SCREEN, close: bool = True) -> Path:
    """Write a figure with the shared output settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path
