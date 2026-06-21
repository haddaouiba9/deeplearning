
Datasets utilisés (TOUS RÉELS) :
  - MLP  → Wine (sklearn, 178 samples, 13 features, 3 classes)
  - CNN  → MNIST (torchvision, 70k images 28x28, 10 classes)
  - RNN / LSTM / GRU / CNN-LSTM → Jena Climate (420k lignes, 14 features, 4 classes météo)
"""

import os
import random
import numpy as np
import torch

# === Dossiers du projet ===
PROJECT_DIR = "./"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
REPORT_DIR = os.path.join(PROJECT_DIR, "report")
JENA_CSV = os.path.join(DATA_DIR, "jena_climate_2009_2016.csv")

for d in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, MODELS_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Reproductibilité ===
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === Hyperparamètres MLP (Wine) ===
MLP_CONFIG = {
    "hidden_dims": [64, 32],
    "dropout": 0.3,
    "epochs": 60,
    "batch_size": 16,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 12,
}

# === Hyperparamètres CNN (MNIST) ===
CNN_CONFIG = {
    "conv_channels": [16, 32, 64],
    "fc_dim": 128,
    "dropout": 0.3,
    "epochs": 5,           # court pour finir à temps
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 2,
    "train_subset": 5000,  # sous-ensemble pour entraînement rapide
    "test_subset": 1500,
}

# === Hyperparamètres RNN/LSTM/GRU (Jena Climate) ===
RNN_CONFIG = {
    "hidden_dim": 64,
    "num_layers": 1,
    "dropout": 0.2,
    "epochs": 8,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 3,
    "seq_len": 24,        # 24 pas de temps (4h de données)
    "n_features": 14,     # 14 features Jena Climate
    "n_classes": 4,       # 4 saisons: printemps/été/automne/hiver
    "train_samples": 4000,
    "test_samples": 1000,
}

# === Hyperparamètres CNN-LSTM hybride ===
HYBRID_CONFIG = {
    "cnn_channels": [32, 64],
    "lstm_hidden": 64,
    "lstm_layers": 1,
    "dropout": 0.3,
    "epochs": 8,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 3,
    "seq_len": 24,
    "n_features": 14,
    "n_classes": 4,
    "train_samples": 4000,
    "test_samples": 1000,
}
