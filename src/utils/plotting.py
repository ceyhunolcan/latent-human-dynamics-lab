"""Minimal matplotlib helpers.

We use matplotlib only — no seaborn, no plotly — to keep figure
generation reproducible and dependency-light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


def setup_matplotlib() -> None:
    """Apply a sober rcParams profile suitable for research figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: str | Path, *, close: bool = True) -> Path:
    """Save a figure and optionally close it. Returns the resolved Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    if close:
        plt.close(fig)
    return p


def annotate_research_only(fig: plt.Figure, msg: Optional[str] = None) -> None:
    """Attach a small disclaimer line to a figure footer."""
    text = msg or "Research prototype only. Not medical advice or diagnosis."
    fig.text(0.5, -0.03, text, ha="center", va="top", fontsize=7, color="#666666")
