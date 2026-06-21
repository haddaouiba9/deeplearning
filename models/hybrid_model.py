"""
Architecture Hybride CNN-LSTM - Jena Climate (RÉEL)
Projet Deep Learning - Ismail Haddaoui - 4IAD G3 - Encadrant: Zineb Hdila

Le CNN extrait des features locales (patterns courts) à partir des séquences,
le LSTM capture la dynamique temporelle longue portée.
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "/home/z/my-project/download/deep_learning_project_ismail")
from utils.config import set_seed, DEVICE, HYBRID_CONFIG, MODELS_DIR, RESULTS_DIR, PLOTS_DIR
from utils.metrics_logger import MetricsLogger, compute_classification_metrics
from models.rnn_models import prepare_data, CLASS_NAMES

set_seed()


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybride pour classification de séries temporelles multivariées.
    - Conv1D sur la dimension temporelle (filtres appris sur les features)
    - LSTM sur les features extraites
    """
    def __init__(self, n_features=14, cnn_channels=(32, 64),
                 lstm_hidden=64, lstm_layers=1, n_classes=4, dropout=0.3):
        super().__init__()
        # CNN 1D : input (B, C_in, L) → on met n_features comme channels
        cnn_blocks = []
        prev = n_features
        for c in cnn_channels:
            cnn_blocks.append(nn.Sequential(
                nn.Conv1d(prev, c, kernel_size=3, padding=1),
                nn.BatchNorm1d(c),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ))
            prev = c
        self.cnn = nn.Sequential(*cnn_blocks)
        # LSTM sur la sortie CNN (transposée)
        self.lstm = nn.LSTM(input_size=prev, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        # x : (B, L, C) → (B, C, L) pour Conv1d
        x = x.transpose(1, 2)
        x = self.cnn(x)              # (B, C_out, L_reduced)
        x = x.transpose(1, 2)        # (B, L_reduced, C_out)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def run_hybrid_experiment():
    print("\n" + "="*70)
    print("  CNN-LSTM Hybride - Jena Climate Dataset")
    print("="*70)

    cfg = HYBRID_CONFIG
    total_samples = cfg["train_samples"] + cfg["test_samples"] + 2000
    X_tr, y_tr, X_val, y_val, X_te, y_te = prepare_data(n_samples=total_samples)
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_te.shape}")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_te),  torch.from_numpy(y_te))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = CNNLSTM(n_features=cfg["n_features"], cnn_channels=cfg["cnn_channels"],
                    lstm_hidden=cfg["lstm_hidden"], lstm_layers=cfg["lstm_layers"],
                    n_classes=cfg["n_classes"], dropout=cfg["dropout"]).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger("CNN_LSTM", RESULTS_DIR)

    best_val_f1, best_state, patience_counter = 0, None, 0
    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        train_losses, all_p, all_t = [], [], []
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
        train_acc = accuracy_score(all_t, all_p)
        train_f1 = f1_score(all_t, all_p, average="macro", zero_division=0)

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
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "CNN_LSTM_best.pt"))

    # Test
    model.eval()
    test_pred, test_true, test_proba, test_loss = [], [], [], 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            test_loss += criterion(out, yb).item() * len(yb)
            test_pred.extend(out.argmax(1).cpu().numpy())
            test_true.extend(yb.cpu().numpy())
            test_proba.extend(torch.softmax(out, dim=1).cpu().numpy())
    test_proba = np.array(test_proba)
    metrics = compute_classification_metrics(test_true, test_pred, test_proba, cfg["n_classes"])
    metrics["test_loss"] = float(test_loss / len(test_true))
    metrics["total_params"] = total_params
    metrics["n_classes"] = cfg["n_classes"]
    metrics["class_names"] = CLASS_NAMES

    print(f"\n  >>> Test Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    logger.save_history()
    logger.save_results(metrics)
    plot_hybrid_results(logger.history, metrics)
    return metrics


def plot_hybrid_results(history, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.font_manager as fm
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    except Exception:
        pass
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # Courbes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history["epochs"]
    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Val", marker="s", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("CNN-LSTM - Jena - Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=3)
    axes[1].plot(epochs, history["train_f1"], label="Train F1", marker="^", markersize=3, linestyle="--")
    axes[1].plot(epochs, history["val_f1"], label="Val F1", marker="v", markersize=3, linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("CNN-LSTM - Jena - Accuracy / F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "CNN_LSTM_learning_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Matrice de confusion
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Prédit"); plt.ylabel("Réel")
    plt.title("CNN-LSTM - Matrice de confusion (Jena Climate)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "CNN_LSTM_confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Plots CNN-LSTM sauvegardés.")


if __name__ == "__main__":
    run_hybrid_experiment()
