"""
=============================================================================
 Phase 3, Step 3: MLP Neural Network
=============================================================================
 Simple 3-layer MLP with BatchNorm + Dropout. Establishes whether deep learning
 adds value over tree-based baselines for this feature set.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, log_loss

from .config import MODELS_DIR


class MLP(nn.Module):
    """3-layer perceptron with BatchNorm and Dropout."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run(data: dict) -> dict:
    """Train MLP, return model and metrics."""
    print("=" * 70)
    print("  Step 3: MLP Neural Network")
    print("=" * 70)

    X_train, y_train = data["X_train"].astype(np.float32), data["y_train"].astype(np.float32)
    X_val, y_val = data["X_val"].astype(np.float32), data["y_val"].astype(np.float32)
    X_test, y_test = data["X_test"].astype(np.float32), data["y_test"].astype(np.float32)

    input_dim = X_train.shape[1]

    # ---- Config ----
    config = {
        "hidden_dims": [128, 64, 32],
        "dropout": 0.3,
        "lr": 0.001,
        "batch_size": 512,
        "max_epochs": 100,
        "patience": 15,
    }
    print(f"\n  Architecture: {input_dim} -> {config['hidden_dims']} -> 1 (sigmoid)")
    print(f"  Config: lr={config['lr']}, batch={config['batch_size']}, "
          f"patience={config['patience']}")

    # ---- DataLoaders ----
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    # ---- Model, loss, optimizer ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim, config["hidden_dims"], config["dropout"]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    print(f"  Device: {device}")

    # ---- Training loop ----
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(config["max_epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        train_loss /= len(train_ds)
        history["train_loss"].append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * len(xb)
                val_probs.extend(pred.cpu().numpy().tolist())
                val_labels.extend(yb.cpu().numpy().tolist())

        val_loss /= len(val_ds)
        val_auc = roc_auc_score(val_labels, val_probs)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "mlp_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping at epoch {epoch+1} (best: {best_epoch})")
                break

    # ---- Load best model ----
    model.load_state_dict(torch.load(MODELS_DIR / "mlp_best.pt"))
    model.eval()

    # ---- Test predictions ----
    with torch.no_grad():
        y_train_prob = model(torch.tensor(X_train).to(device)).cpu().numpy()
        y_val_prob = model(torch.tensor(X_val).to(device)).cpu().numpy()
        y_test_prob = model(torch.tensor(X_test).to(device)).cpu().numpy()

    # ---- Metrics ----
    metrics = {
        "train_auc": roc_auc_score(y_train, y_train_prob),
        "val_auc": roc_auc_score(y_val, y_val_prob),
        "test_auc": roc_auc_score(y_test, y_test_prob),
        "train_logloss": log_loss(y_train, y_train_prob),
        "val_logloss": log_loss(y_val, y_val_prob),
        "test_logloss": log_loss(y_test, y_test_prob),
        "best_epoch": best_epoch,
    }

    print(f"\n  MLP Results:")
    print(f"    Train AUC: {metrics['train_auc']:.4f}  |  LogLoss: {metrics['train_logloss']:.4f}")
    print(f"    Val   AUC: {metrics['val_auc']:.4f}  |  LogLoss: {metrics['val_logloss']:.4f}")
    print(f"    Test  AUC: {metrics['test_auc']:.4f}  |  LogLoss: {metrics['test_logloss']:.4f}")
    print(f"    Best epoch: {best_epoch}")

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "input_dim": input_dim,
    }, MODELS_DIR / "mlp_baseline.pt")
    print(f"\n  Model saved: {MODELS_DIR / 'mlp_baseline.pt'}")

    # Save training history
    with open(MODELS_DIR / "mlp_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  [OK] Step 3 complete.")
    return {"model": model, "metrics": metrics, "y_pred": {"train": y_train_prob, "val": y_val_prob, "test": y_test_prob}}


if __name__ == "__main__":
    from src.phase3.step1_load import run as load
    data = load()
    run(data)
