"""
MLP - Wine Dataset (sklearn, RÉEL)
Projet Deep Learning - Ismail Haddaoui - 4IAD G3 - Encadrant: Zineb Hdila
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/z/my-project/download/deep_learning_project_ismail")
from utils.config import set_seed, DEVICE, MLP_CONFIG, MODELS_DIR, RESULTS_DIR, PLOTS_DIR
from utils.metrics_logger import MetricsLogger, compute_classification_metrics

set_seed()


class MLP(nn.Module):
    """Multilayer Perceptron avec BatchNorm + Dropout."""
    def __init__(self, input_dim=13, hidden_dims=(64, 32), n_classes=3, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_wine_data():
    """Charge Wine (178 samples, 13 features, 3 classes) — dataset RÉEL sklearn."""
    data = load_wine()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    # Split 60/20/20
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    sc = StandardScaler().fit(X_tr)
    return (sc.transform(X_tr), y_tr, sc.transform(X_val), y_val, sc.transform(X_te), y_te, data.target_names)


def run_mlp_experiment():
    print("\n" + "="*70)
    print("  MLP - Wine Dataset (sklearn)")
    print("="*70)

    X_tr, y_tr, X_val, y_val, X_te, y_te, class_names = load_wine_data()
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_te.shape}")
    print(f"  Classes: {list(class_names)}")

    cfg = MLP_CONFIG
    n_classes = len(class_names)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_te),  torch.from_numpy(y_te))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = MLP(input_dim=X_tr.shape[1], hidden_dims=cfg["hidden_dims"],
                n_classes=n_classes, dropout=cfg["dropout"]).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger("MLP", RESULTS_DIR)

    best_val_f1, best_state, patience_counter = 0, None, 0
    for epoch in range(1, cfg["epochs"]+1):
        # --- Train ---
        model.train()
        train_losses = []
        all_p, all_t = [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            all_p.extend(out.argmax(1).cpu().numpy())
            all_t.extend(yb.cpu().numpy())
        from sklearn.metrics import f1_score, accuracy_score
        train_acc = accuracy_score(all_t, all_p)
        train_f1 = f1_score(all_t, all_p, average="macro", zero_division=0)

        # --- Val ---
        model.eval()
        val_losses, vp, vt = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_losses.append(criterion(out, yb).item())
                vp.extend(out.argmax(1).cpu().numpy())
                vt.extend(yb.cpu().numpy())
        val_acc = accuracy_score(vt, vp)
        val_f1 = f1_score(vt, vp, average="macro", zero_division=0)

        logger.log_epoch(epoch, np.mean(train_losses), np.mean(val_losses),
                         train_acc, val_acc, train_f1, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping à l'epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "MLP_best.pt"))

    # --- Test ---
    model.eval()
    test_pred, test_true, test_proba = [], [], []
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            test_loss += criterion(out, yb).item() * len(yb)
            test_pred.extend(out.argmax(1).cpu().numpy())
            test_true.extend(yb.cpu().numpy())
            test_proba.extend(torch.softmax(out, dim=1).cpu().numpy())
    test_proba = np.array(test_proba)
    metrics = compute_classification_metrics(test_true, test_pred, test_proba, n_classes)
    metrics["test_loss"] = float(test_loss / len(test_te := y_te))
    metrics["total_params"] = total_params
    metrics["n_classes"] = n_classes
    metrics["class_names"] = list(class_names)

    print(f"\n  >>> Test Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    logger.save_history()
    logger.save_results(metrics)

    # --- Plots ---
    plot_mlp_results(logger.history, metrics, class_names)
    return metrics


def plot_mlp_results(history, metrics, class_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Font setup
    import matplotlib.font_manager as fm
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    except Exception:
        pass
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. Courbes d'apprentissage
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history["epochs"]
    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Val", marker="s", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("MLP - Wine - Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=3)
    axes[1].plot(epochs, history["train_f1"], label="Train F1", marker="^", markersize=3, linestyle="--")
    axes[1].plot(epochs, history["val_f1"], label="Val F1", marker="v", markersize=3, linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("MLP - Wine - Accuracy / F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "MLP_learning_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Matrice de confusion
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prédit"); plt.ylabel("Réel")
    plt.title("MLP - Matrice de confusion (Wine)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "MLP_confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print("  Plots MLP sauvegardés.")


if __name__ == "__main__":
    run_mlp_experiment()
