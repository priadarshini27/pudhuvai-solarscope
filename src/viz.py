"""Shared matplotlib style for notebooks and reports.

Solar palette: #773701 umber, #F49005 sun orange, #FBB946 gold, #C1ADBC haze,
on warm cream neutrals. Region colours are fixed per entity (never by rank).
Gold and haze are < 3:1 against the surface, so charts always carry a legend
and/or direct labels, and notebooks print the underlying table.
"""
from __future__ import annotations

import matplotlib as mpl

REGION_ORDER = ["puducherry", "karaikal", "mahe", "yanam"]
REGION_LABEL = {"puducherry": "Puducherry", "karaikal": "Karaikal", "mahe": "Mahe", "yanam": "Yanam"}
UMBER, SUN, GOLD, HAZE = "#773701", "#F49005", "#FBB946", "#C1ADBC"
REGION_COLOR = {"puducherry": UMBER, "karaikal": SUN, "mahe": HAZE, "yanam": GOLD}
NEUTRAL = "#8C7B6B"
INK, INK_2, GRID, AXIS, SURFACE = "#2B1A0B", "#5C4632", "#F0E4D4", "#D9C7B2", "#FFFAF3"


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "lines.markersize": 4, "font.size": 10, "text.color": INK,
    })


def label_line_ends(ax, series: dict, x_last, min_gap_frac: float = 0.045):
    """Direct-label each line at its right end (<= 4 series), nudging labels
    apart vertically so close end values do not collide."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(series.items(), key=lambda kv: kv[1])
    placed = []
    for key, y in items:
        y_txt = y if not placed else max(y, placed[-1] + gap)
        placed.append(y_txt)
        ax.annotate(REGION_LABEL.get(key, key), (x_last, y), xytext=(x_last + 0.25, y_txt),
                    textcoords="data", va="center", fontsize=9, color=INK_2)
