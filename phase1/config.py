"""
=============================================================================
 Phase 1: Shared Configuration
=============================================================================
 Paths, matplotlib settings, and the chart-saving helper used by all steps.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Matplotlib ----
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
sns.set_palette("Set2")

def clean_output_dir() -> None:
    """Remove all old PNGs before re-running."""
    for png in OUTPUT_DIR.glob("*.png"):
        png.unlink()
    for txt in OUTPUT_DIR.glob("*.txt"):
        txt.unlink()


def savefig(name: str) -> str:
    """Save current figure, overwriting any existing file with the same name."""
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path, bbox_inches="tight")
    print(f"   [chart] Saved: {path}")
    plt.close()
    return str(path)
