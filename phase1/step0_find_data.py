"""
=============================================================================
 Phase 1, Step 0: Locate data file
=============================================================================
"""

import sys
from pathlib import Path

from .config import DATA_DIR


def run() -> Path:
    """Find the CSV data file. Exits if not found."""
    if not DATA_DIR.exists():
        print("[ERROR] data/ directory not found. Please download dataset first:")
        print("        python data/download_dataset.py")
        sys.exit(1)

    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("[ERROR] No CSV files in data/. Please download dataset first:")
        print("        python data/download_dataset.py")
        sys.exit(1)

    preferred = DATA_DIR / "csgo_round_snapshots.csv"
    if preferred.exists():
        return preferred

    print(f"[WARN] csgo_round_snapshots.csv not found, using {csv_files[0].name}")
    return csv_files[0]
