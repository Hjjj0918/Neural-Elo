"""
=============================================================================
 Phase 3: Model Training — Master Orchestrator
=============================================================================
 Loads Phase 2 data, trains LightGBM + MLP, compares, saves everything.

 Usage:
     python -m src.phase3.run_all
     python phase3_eda.py              # thin wrapper at project root
=============================================================================
"""

import json
import warnings

import numpy as np

from .config import MODELS_DIR
from .step1_load import run as step1_load
from .step2_lgbm import run as step2_lgbm
from .step3_mlp import run as step3_mlp
from .step4_evaluate import run as step4_evaluate

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    print("\n" + "#" * 70)
    print("#   Neural-Elo Phase 3: Model Training")
    print("#   CS:GO Round Winner Prediction")
    print("#" * 70)

    # ---- Step 1: Load data ----
    data = step1_load()

    # ---- Step 2: LightGBM ----
    lgbm_result = step2_lgbm(data)

    # ---- Step 3: MLP ----
    mlp_result = step3_mlp(data)

    # ---- Step 4: Evaluate ----
    step4_evaluate(lgbm_result, mlp_result, data["y_test"])

    print("\n" + "#" * 70)
    print("#   Phase 3 Complete! Ready for Phase 4 (Player Embeddings).")
    print("#" * 70)


if __name__ == "__main__":
    main()
