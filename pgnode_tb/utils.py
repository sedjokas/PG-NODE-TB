"""
utils.py
========
Shared plotting utilities and file-handling helpers.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# ------------------------------------------------------------------ #
#  Global matplotlib style
# ------------------------------------------------------------------ #
def set_style() -> None:
    """Apply the paper's uniform matplotlib style."""
    matplotlib.use('Agg')
    plt.rcParams.update({
        'font.size':          11,
        'axes.titlesize':     12,
        'axes.labelsize':     11,
        'legend.fontsize':     9,
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'figure.dpi':        150,
    })


# ------------------------------------------------------------------ #
#  Figure saving
# ------------------------------------------------------------------ #
def save_figure(fig: plt.Figure, outdir: str, name: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure to both PDF and PNG.

    Parameters
    ----------
    fig    : matplotlib Figure
    outdir : output directory path
    name   : base filename (without extension)
    dpi    : PNG resolution
    """
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches='tight', dpi=(dpi if ext == 'png' else None))
    print(f"  Saved: {name}.pdf / {name}.png  ->  {outdir}")


# ------------------------------------------------------------------ #
#  Metrics
# ------------------------------------------------------------------ #
def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root Mean Squared Error between two arrays."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def cumulative_cases_averted(I_baseline: np.ndarray,
                              I_scenario: np.ndarray,
                              dt: float) -> np.ndarray:
    """
    Compute cumulative TB cases averted relative to a baseline trajectory.

    Parameters
    ----------
    I_baseline : baseline infectious prevalence array
    I_scenario : scenario infectious prevalence array
    dt         : time step (yr)

    Returns
    -------
    np.ndarray : cumulative averted cases (same length as inputs)
    """
    return np.cumsum(I_baseline - I_scenario) * dt


# ------------------------------------------------------------------ #
#  Architecture diagram helpers
# ------------------------------------------------------------------ #
def rbox(ax, cx, cy, w, h, txt,
         fc='#d4e6f1', ec='#2c3e50', fs=9.5, fw='normal') -> None:
    """Draw a rounded rectangle with centred text on ax."""
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.12',
        facecolor=fc, edgecolor=ec, linewidth=1.8
    )
    ax.add_patch(patch)
    ax.text(cx, cy, txt,
            ha='center', va='center',
            fontsize=fs, fontweight=fw,
            multialignment='center')


def arrow(ax, x1, y1, x2, y2,
          lbl='', lbl_dy=0.22, color='#2c3e50') -> None:
    """Draw an annotated arrow on ax."""
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.8)
    )
    if lbl:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + lbl_dy,
                lbl, ha='center', fontsize=8.0,
                color='#555', style='italic')
