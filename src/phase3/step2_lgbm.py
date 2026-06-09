"""
=============================================================================
 Phase 3, Step 2: LightGBM Baseline
=============================================================================
 Fast, strong baseline. Trees naturally handle feature interactions.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from .config import MODELS_DIR


def run(data: dict) -> dict:
    """
    Train a LightGBM classifier. Returns dict with model, predictions, metrics.
    """
    print("=" * 70)
    print("  Step 2: LightGBM Baseline")
    print("=" * 70)

    import lightgbm as lgb

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # ---- Model config ----
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    print(f"\n  Training LightGBM...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    # ---- Predictions ----
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # ---- Metrics ----
    metrics = {
        "train_auc": roc_auc_score(y_train, y_train_prob),
        "val_auc": roc_auc_score(y_val, y_val_prob),
        "test_auc": roc_auc_score(y_test, y_test_prob),
        "train_logloss": log_loss(y_train, y_train_prob),
        "val_logloss": log_loss(y_val, y_val_prob),
        "test_logloss": log_loss(y_test, y_test_prob),
    }

    print(f"\n  LightGBM Results:")
    print(f"    Train AUC: {metrics['train_auc']:.4f}  |  LogLoss: {metrics['train_logloss']:.4f}")
    print(f"    Val   AUC: {metrics['val_auc']:.4f}  |  LogLoss: {metrics['val_logloss']:.4f}")
    print(f"    Test  AUC: {metrics['test_auc']:.4f}  |  LogLoss: {metrics['test_logloss']:.4f}")

    # Feature importance
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print(f"\n  Top 15 feature importances:")
    for rank, idx in enumerate(top_idx):
        name = data["feature_names"][idx]
        print(f"    {rank+1:2d}. {name:30s}  {importances[idx]:.4f}")

    # Save
    import joblib
    path = MODELS_DIR / "lgbm_baseline.pkl"
    joblib.dump(model, path)
    print(f"\n  Model saved: {path}")

    print(f"  [OK] Step 2 complete.")
    return {"model": model, "metrics": metrics, "y_pred": {"train": y_train_prob, "val": y_val_prob, "test": y_test_prob}}


if __name__ == "__main__":
    from src.phase3.step1_load import run as load
    data = load()
    run(data)
