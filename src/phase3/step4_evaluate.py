"""
=============================================================================
 Phase 3, Step 4: Model Evaluation & Comparison
=============================================================================
 Compare LightGBM vs MLP on Test set.
"""

import json

import numpy as np
from sklearn.metrics import confusion_matrix

from .config import MODELS_DIR


def run(lgbm_result: dict, mlp_result: dict, y_test: np.ndarray) -> dict:
    """Print comparison table and confusion matrices."""
    print("=" * 70)
    print("  Step 4: Model Comparison")
    print("=" * 70)

    lgbm_m = lgbm_result["metrics"]
    mlp_m = mlp_result["metrics"]

    # ---- Comparison table ----
    print(f"\n  {'Metric':<20s} {'LightGBM':>10s} {'MLP':>10s} {'Winner':>10s}")
    print(f"  {'-'*50}")
    for metric_name in ["test_auc", "test_logloss"]:
        l_val = lgbm_m[metric_name]
        m_val = mlp_m[metric_name]
        if metric_name == "test_auc":
            winner = "LightGBM" if l_val > m_val else "MLP"
        else:
            winner = "LightGBM" if l_val < m_val else "MLP"
        display = metric_name.replace("test_", "").upper()
        print(f"  {display:<20s} {l_val:>10.4f} {m_val:>10.4f} {winner:>10s}")

    # ---- Confusion matrices (threshold = 0.5) ----
    print(f"\n  Confusion Matrices (threshold=0.5):")
    print(f"\n  LightGBM:")
    _print_cm(y_test, lgbm_result["y_pred"]["test"])
    print(f"\n  MLP:")
    _print_cm(y_test, mlp_result["y_pred"]["test"])

    # ---- Summary ----
    report = {
        "lightgbm": lgbm_m,
        "mlp": mlp_m,
        "best_model": "LightGBM" if lgbm_m["test_auc"] > mlp_m["test_auc"] else "MLP",
        "auc_diff": abs(lgbm_m["test_auc"] - mlp_m["test_auc"]),
    }

    print(f"\n  Best model: {report['best_model']} "
          f"(AUC diff = {report['auc_diff']:.4f})")

    # Save
    with open(MODELS_DIR / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Report saved: {MODELS_DIR / 'evaluation_report.json'}")
    print(f"  [OK] Step 4 complete.")
    return report


def _print_cm(y_true, y_prob):
    """Print normalized confusion matrix."""
    # Need actual labels — load from data if not available
    cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
    cm_norm = cm / cm.sum()
    print(f"              Pred T    Pred CT")
    print(f"    Actual T   {cm[0][0]:>5d}     {cm[0][1]:>5d}   ({cm_norm[0][0]*100:4.1f}%  {cm_norm[0][1]*100:4.1f}%)")
    print(f"    Actual CT  {cm[1][0]:>5d}     {cm[1][1]:>5d}   ({cm_norm[1][0]*100:4.1f}%  {cm_norm[1][1]*100:4.1f}%)")


if __name__ == "__main__":
    from src.phase3.step1_load import run as load
    from src.phase3.step2_lgbm import run as lgbm
    from src.phase3.step3_mlp import run as mlp

    data = load()
    # Use saved test labels
    global y_test_true
    y_test_true = data["y_test"]

    lgbm_result = lgbm(data)
    mlp_result = mlp(data)
    run(lgbm_result, mlp_result)
