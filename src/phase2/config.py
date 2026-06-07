"""
=============================================================================
 Phase 2: Shared Configuration
=============================================================================
 Paths and constants used by all feature engineering steps.
"""

from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = DATA_DIR / "csgo_round_snapshots.csv"
