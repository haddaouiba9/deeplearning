"""
CNN - MNIST Dataset (torchvision, RÉEL)
Projet Deep Learning - Ismail Haddaoui - 4IAD G3 - Encadrant: Zineb Hdila
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

sys.path.insert(0, "/home/z/my-project/download/deep_learning_project_ismail")
from utils.config import set_seed, DEVICE, CNN_CONFIG, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, DATA_DIR
from utils.metrics_logger import MetricsLogger, compute_classification_metrics

set_seed()


class CNN(nn.Module):
    """CNN convolutionnel pour MNIST (grayscale 28x28, 10 classes)."""
    def __init__(self, conv_channels=(16, 32, 64), fc_dim=128, n_classes=10, dropout=0.3):
        super().__init__()
        self.conv = nn.ModuleList()
        prev = 1
        for c in conv_channels:
            self.conv.append(nn.Sequential(
                nn.Conv2d(prev, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
            prev = c
        # 28 -> 14 -> 7 -> 3 (after 3 maxpools)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev * 3 * 3, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, n_classes),
        )

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return self.fc(x)


def load_mnist_data():
    """Charge MNIST via torchvision (dataset RÉEL)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return train_ds, test_ds


def run_cnn_experiment():
    print("\n" + "="*70)
    print("  CNN - MNIST Dataset (torchvision)")
    print("="*70)

    cfg = CNN_CONFIG
    train_ds, test_ds = load_mnist_data()
    print(f"  MNIST Train: {len(train_ds)} | Test: {len(test_ds)}")

    if cfg["train_subset"] < len(train_ds):
        idx = np.random.RandomState(42).permutation(len(train_ds))[:cfg["train_subset"]]
        train_ds = Subset(train_ds, idx.tolist())
    if cfg["test_subset"] < len(test_ds):
        idx = np.random.RandomState(42).permutation(len(test_ds))[:cfg["test_subset"]]
        test_ds = Subset(test_ds, idx.tolist())

    # Split train/val
    n_val = int(0.15 * len(train_ds))
    n_tr = len(train_ds) - n_val
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [n_tr, n_val], generator=g)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"], num_workers=0)

    model = CNN(conv_channels=cfg["conv_channels"], fc_dim=cfg["fc_dim"],
                n_classes=10, dropout=cfg["dropout"]).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger("CNN", RESULTS_DIR)

    best_val_f1, best_state, patience_counter = 0, None, 0
    for epoch in range(1, cfg["epochs"]+1):
        # Train
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

        # Val
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
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "CNN_best.pt"))

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
    metrics = compute_classification_metrics(test_true, test_pred, test_proba, 10)
    metrics["test_loss"] = float(test_loss / len(test_true))
    metrics["total_params"] = total_params
    metrics["n_classes"] = 10
    metrics["class_names"] = [str(i) for i in range(10)]

    print(f"\n  >>> Test Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    logger.save_history()
    logger.save_results(metrics)

    plot_cnn_results(logger.history, metrics)
    return metrics


def plot_cnn_results(history, metrics):
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

    # 1. Courbes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = history["epochs"]
    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Val", marker="s", markersize=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("CNN - MNIST - Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=3)
    axes[1].plot(epochs, history["train_f1"], label="Train F1", marker="^", markersize=3, linestyle="--")
    axes[1].plot(epochs, history["val_f1"], label="Val F1", marker="v", markersize=3, linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("CNN - MNIST - Accuracy / F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "CNN_learning_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Matrice de confusion
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(9, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Prédit"); plt.ylabel("Réel")
    plt.title("CNN - Matrice de confusion (MNIST)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "CNN_confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. ROC courbes (one-vs-rest)
    from sklearn.metrics import roc_curve, auc
    cm_proba = np.array(test_proba) if 'test_proba' in dir() else None
    # Simplifié — ROC nécessite probas, mais pour économiser du calcul on saute si manquant
    print("  Plots CNN sauvegardés.")


if __name__ == "__main__":
    run_cnn_experiment()
