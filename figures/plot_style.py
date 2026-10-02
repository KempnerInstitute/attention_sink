"""Shared plot style for the paper figures.

Clean, minimal, ink-on-white; Product Sans body text + Computer Modern math, no grid, detached spines.

Product Sans is not redistributed with this repository. To render the figures exactly as in the paper, put the
Product Sans .ttf files in `figures/fonts/product_sans/` (or point ATTN_SINK_FONT_DIR at a folder containing them);
otherwise the figures fall back to DejaVu Sans (matplotlib's default) with identical layout and colours.

    import plot_style as ps
    ps.apply_style()                         # fonts + rcParams (call once)
    fig, ax = plt.subplots()
    ax.plot(x, y, color=ps.PRIMARY)
    ps.finish(ax)                            # spines / grid / ink
    ps.save(fig, "figures/my_plot.png")      # dpi 350, opaque white
"""

import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib import font_manager
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.environ.get("ATTN_SINK_FONT_DIR", os.path.join(HERE, "fonts", "product_sans"))

# ----------------------------------------------------------------------------
# palette
# ----------------------------------------------------------------------------
PRIMARY = "#2563eb"   # blue-600   -- main condition / variable of interest
THIRD   = "#8b5cf6"   # violet-500 -- second series
ROSE    = "#f43f5e"   # rose-500   -- third series
AMBER   = "#f59e0b"   # amber-500  -- fourth series

META      = "#e2e8f0"   # slate-200  -- neutral fills, bands
META_LINE = "#94a3b8"   # slate-400  -- thin reference/baseline lines
INK       = "#020617"   # slate-950  -- axis, ticks, all body text

QUARTET = [PRIMARY, THIRD, ROSE, AMBER]

# n >= 5 categorical fallback: tab10 hue slots in Tailwind 600-family shades.
TAB10_TW = ["#2563EB",  # blue-600
            "#EA580C",  # orange-600
            "#059669",  # emerald-600
            "#DC2626",  # red-600
            "#7C3AED",  # violet-600
            "#92400E",  # amber-800  (brown)
            "#DB2777",  # pink-600
            "#64748B",  # slate-500  (gray)
            "#65A30D",  # lime-600   (olive)
            "#0891B2"]  # cyan-600
TAB10 = TAB10_TW

# slate ramp -- monochrome sequential greys (META = SLATE[0], META_LINE = SLATE[2], INK = SLATE[-1]).
SLATE = ["#e2e8f0", "#cbd5e1", "#94a3b8", "#64748b", "#475569",
         "#334155", "#1e293b", "#0f172a", "#020617"]

# ordered layer-depth colours: managua with the pale ends trimmed so the first/last layer stays visible on white.
DEPTH = mcolors.LinearSegmentedColormap.from_list(
    "depth_trim", matplotlib.colormaps["managua"](np.linspace(0.12, 0.88, 256)))


def apply_style():
    """Reset rcParams, register Product Sans if available (else DejaVu Sans), set Computer Modern math. Call once."""
    matplotlib.rcdefaults()
    ttfs = glob.glob(os.path.join(FONT_DIR, "*.ttf"))
    for ttf in ttfs:
        font_manager.fontManager.addfont(ttf)
    plt.rcParams["font.family"] = "Product Sans" if ttfs else "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False    # ascii hyphen (Product Sans has it)
    plt.rcParams["mathtext.fontset"] = "cm"        # Computer Modern for $...$ math


def cat_colors(n):
    """n categorical colors: 1-4 series -> QUARTET; 5-10 -> Tailwind tab10; more -> tab20."""
    if n <= 1:
        return [PRIMARY]
    if n <= 4:
        return QUARTET[:n]
    if n <= len(TAB10):
        return list(TAB10[:n])
    return list(matplotlib.colormaps["tab20"].colors)[:n]


# ----------------------------------------------------------------------------
# bars with slightly rounded ends
# ----------------------------------------------------------------------------
R_PT = 2.5   # bar corner radius, in POINTS


def _px_per_unit(ax):
    """(x, y) pixels per data unit. Requires the axes limits to be set."""
    o = ax.transData.transform((0, 0))
    return (abs(ax.transData.transform((1, 0))[0] - o[0]),
            abs(ax.transData.transform((0, 1))[1] - o[1]))


def _rounded(ax, verts, color, alpha, zorder):
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(patches.PathPatch(Path(verts, codes), facecolor=color,
                                   edgecolor="none", alpha=alpha, zorder=zorder))


def bar(ax, x0, w, y1, y0=0.0, color=None, r_pt=R_PT, alpha=1.0, zorder=3):
    """Vertical bar with slightly rounded TOP corners.

    The radius is ABSOLUTE (in points), not a fraction of bar width -- a width
    fraction gives fat bars a fat corner and thin grouped bars a hairline. It is
    converted separately per axis so the corner stays circular on screen.

    SET THE AXES LIMITS FIRST: the conversion reads the data transform.

        ax.set_xlim(...); ax.set_ylim(...)
        for i, (v, c) in enumerate(zip(vals, ps.cat_colors(len(vals)))):
            ps.bar(ax, i - 0.4, 0.8, v, color=c)
    """
    color = PRIMARY if color is None else color
    if r_pt <= 0:
        ax.add_patch(patches.Rectangle((x0, y0), w, y1 - y0, facecolor=color,
                                       edgecolor="none", alpha=alpha, zorder=zorder))
        return
    sx, sy = _px_per_unit(ax)
    px = r_pt * ax.figure.dpi / 72.0
    rx = min(px / sx, w / 2)
    ry = min(px / sy, abs(y1 - y0) * 0.9)
    _rounded(ax, [(x0, y0), (x0, y1 - ry), (x0, y1), (x0 + rx, y1),
                  (x0 + w - rx, y1), (x0 + w, y1), (x0 + w, y1 - ry),
                  (x0 + w, y0), (x0, y0)], color, alpha, zorder)


def finish(ax, offset=10, trim=False):
    """Spine/grid/ink cleanup: no grid, detached 'dent' spines, ink axes.

    trim=False (default) lets spines run the full range (overhang past the last
    tick is fine). trim=True stops each spine at its outermost visible tick
    (seaborn-style), including minor ticks so log axes still reach the data.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", offset))
    ax.spines["bottom"].set_position(("outward", offset))
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK)
    ax.grid(False)
    if trim:
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        xt = [t for t in (*ax.get_xticks(), *ax.get_xticks(minor=True)) if x0 <= t <= x1]
        yt = [t for t in (*ax.get_yticks(), *ax.get_yticks(minor=True)) if y0 <= t <= y1]
        if xt:
            ax.spines["bottom"].set_bounds(min(xt), max(xt))
        if yt:
            ax.spines["left"].set_bounds(min(yt), max(yt))
