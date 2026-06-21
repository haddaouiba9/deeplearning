"""
RNN / LSTM / GRU - Jena Climate Dataset (RÉEL)
Projet Deep Learning - Ismail Haddaoui - 4IAD G3 - Encadrant: Zineb Hdila

Dataset: Jena Climate 2009-2016 (station météo de l'Institut Max Planck)
- 14 features (pression, température, humidité, vent, etc.)
- 420,551 enregistrements (1 toutes les 10 minutes)
- Tâche: classification de la saison (4 classes) à partir d'une fenêtre de 24 pas de temps
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "/home/z/my-project/download/deep_learning_project_ismail")
from utils.config import set_seed, DEVICE, RNN_CONFIG, MODELS_DIR, RESULTS_DIR, PLOTS_DIR
from utils.metrics_logger import MetricsLogger, compute_classification_metrics

set_seed()

CLASS_NAMES = ["Printemps", "Été", "Automne", "Hiver"]


def load_jena_climate(csv_path, seq_len=24, n_samples=10000):
    """
    Charge Jena Climate et crée des séquences de longueur seq_len.
    Label = saison (4 classes) calculée à partir du mois.
    """
    print(f"  Chargement Jena Climate depuis {csv_path}...")
    df = pd.read_csv(csv_path)
    # Parser la date
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.sort_values("Date Time").reset_index(drop=True)

    # 14 features numériques
    feature_cols = [c for c in df.columns if c != "Date Time"]
    X_all = df[feature_cols].values.astype(np.float32)
    months = df["Date Time"].dt.month.values

    # Saison à partir du mois (hémisphère nord):
    #   12,1,2  → Hiver (3)
    #   3,4,5   → Printemps (0)
    #   6,7,8   → Été (1)
    #   9,10,11 → Automne (2)
    season = np.zeros(len(months), dtype=np.int64)
    season[np.isin(months, [3, 4, 5])]   = 0
    season[np.isin(months, [6, 7, 8])]   = 1
    season[np.isin(months, [9, 10, 11])] = 2
    season[np.isin(months, [12, 1, 2])]  = 3

    # Sous-échantillonnage: 1 enregistrement sur 6 (= 1 mesure par heure)
    stride = 6
    X_sub = X_all[::stride]
    s_sub = season[::stride]

    # Création des séquences
    n_possible = len(X_sub) - seq_len
    n_samples = min(n_samples, n_possible)

    rng = np.random.RandomState(42)
    start_indices = rng.choice(n_possible, size=n_samples, replace=False)
    start_indices.sort()

    X_seq = np.zeros((n_samples, seq_len, len(feature_cols)), dtype=np.float32)
    y_seq = np.zeros(n_samples, dtype=np.int64)
    for i, s in enumerate(start_indices):
        X_seq[i] = X_sub[s:s+seq_len]
        y_seq[i] = s_sub[s + seq_len - 1]  # saison du dernier pas de temps

    print(f"  Séquences: {X_seq.shape} | Classes: {np.bincount(y_seq)}")
    return X_seq, y_seq


def prepare_data(n_samples=10000):
    cfg = RNN_CONFIG
    X, y = load_jena_climate(
        "/home/z/my-project/download/deep_learning_project_ismail/data/jena_climate_2009_2016.csv",
        seq_len=cfg["seq_len"], n_samples=n_samples
    )
    # Split 60/20/20 stratifié
    X_tr, X_temp, y_tr, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Standardisation (sur le train uniquement)
    ns, n_t, n_f = X_tr.shape
    scaler = StandardScaler().fit(X_tr.reshape(-1, n_f))
    X_tr = scaler.transform(X_tr.reshape(-1, n_f)).reshape(ns, n_t, n_f)
    X_val = scaler.transform(X_val.reshape(-1, n_f)).reshape(-1, n_t, n_f)
    X_te = scaler.transform(X_te.reshape(-1, n_f)).reshape(-1, n_t, n_f)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


# === Modèles ===

class SimpleRNN(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, n_classes=4, num_layers=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, n_classes=4, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, n_classes=4, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def _train_one_model(model_class, model_name, X_tr, y_tr, X_val, y_val, X_te, y_te, cfg):
    print(f"\n  --- Entraînement {model_name} ---")
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds  = TensorDataset(torch.from_numpy(X_te),  torch.from_numpy(y_te))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = model_class(input_dim=cfg["n_features"], hidden_dim=cfg["hidden_dim"],
                        n_classes=cfg["n_classes"], num_layers=cfg["num_layers"],
                        dropout=cfg["dropout"]).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger(model_name, RESULTS_DIR)

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
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{model_name}_best.pt"))

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

    print(f"  >>> Test Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    logger.save_history()
    logger.save_results(metrics)
    plot_rnn_results(model_name, logger.history, metrics)
    return metrics, model


def plot_rnn_results(model_name, history, metrics):
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
    axes[0].set_title(f"{model_name} - Jena - Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=3)
    axes[1].plot(epochs, history["train_f1"], label="Train F1", marker="^", markersize=3, linestyle="--")
    axes[1].plot(epochs, history["val_f1"], label="Val F1", marker="v", markersize=3, linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title(f"{model_name} - Jena - Accuracy / F1"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_learning_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Matrice de confusion
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Prédit"); plt.ylabel("Réel")
    plt.title(f"{model_name} - Matrice de confusion (Jena Climate)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()


def run_all_rnn_experiments():
    print("\n" + "="*70)
    print("  RNN / LSTM / GRU - Jena Climate Dataset")
    print("="*70)
    cfg = RNN_CONFIG
    total_samples = cfg["train_samples"] + cfg["test_samples"] + 2000
    X_tr, y_tr, X_val, y_val, X_te, y_te = prepare_data(n_samples=total_samples)
    print(f"  Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_te.shape}")

    results = {}
    for model_class, name in [(SimpleRNN, "RNN"), (LSTMModel, "LSTM"), (GRUModel, "GRU")]:
        metrics, _ = _train_one_model(model_class, name, X_tr, y_tr, X_val, y_val, X_te, y_te, cfg)
        results[name] = metrics

    # Graphique comparatif RNN
    plot_rnn_comparison(results)
    return results


def plot_rnn_comparison(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.font_manager as fm
    try:
        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    except Exception:
        pass
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    models = list(results.keys())
    acc = [results[m]["accuracy"] for m in models]
    f1 = [results[m]["f1_macro"] for m in models]
    params = [results[m]["total_params"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w/2, acc, w, label="Accuracy", color="#4C72B0", alpha=0.85)
    axes[0].bar(x + w/2, f1, w, label="F1-Score", color="#DD8452", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Score")
    axes[0].set_title("RNN vs LSTM vs GRU - Jena Climate")
    axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    for i, (a, f) in enumerate(zip(acc, f1)):
        axes[0].text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        axes[0].text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)

    axes[1].bar(models, params, color="#55A868", alpha=0.85)
    axes[1].set_ylabel("Paramètres")
    axes[1].set_title("Complexité des modèles récurrents")
    axes[1].grid(alpha=0.3, axis="y")
    for i, p in enumerate(params):
        axes[1].text(i, p + max(params)*0.02, f"{p:,}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "RNN_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  RNN_comparison.png sauvegardé.")


if __name__ == "__main__":
    run_all_rnn_experiments()
