## 🎯 Description

Guide de référence pour l'entraînement de 6 architectures de Deep Learning :
- **MLP** (Perceptron Multicouche)
- **CNN** (Réseau de Neurones Convolutif)
- **RNN** (Réseau Récurrent Simple)
- **LSTM** (Long Short-Term Memory)
- **GRU** (Gated Recurrent Unit)
- **CNN-LSTM** (Architecture Hybride)

## 📊 Datasets Réels Utilisés

| Modèle(s) | Dataset | Source | Taille | Classes |
|-----------|---------|--------|--------|---------|
| MLP | Wine | sklearn | 178 samples, 13 features | 3 cultivars |
| CNN | MNIST | torchvision | 70k images 28×28 (sous-ensemble 5k) | 10 chiffres |
| RNN/LSTM/GRU/CNN-LSTM | Jena Climate 2009-2016 | Max Planck Institute | 420k enregistrements, 14 features | 4 saisons |

## 📁 Structure du projet

```
deep_learning_project_ismail/
├── main.py                      # Script principal
├── README.md
├── utils/
│   ├── config.py                # Configuration globale
│   └── metrics_logger.py        # Logger de métriques
├── models/
│   ├── mlp_model.py             # MLP (Wine)
│   ├── cnn_model.py             # CNN (MNIST)
│   ├── rnn_models.py            # RNN/LSTM/GRU (Jena Climate)
│   ├── hybrid_model.py          # CNN-LSTM hybride
│   └── *_best.pt                # Poids entraînés
├── data/
│   ├── jena_climate_2009_2016.csv  # Dataset Jena Climate (réel)
│   └── MNIST/                      # Dataset MNIST (auto-téléchargé)
├── results/
│   ├── all_results_summary.json    # Résumé global
│   ├── {model}_results.json        # Métriques par modèle
│   ├── {model}_history.json        # Historique d'entraînement
│   └── plots/                      # Graphiques PNG
└── report/
    └── (rapport PDF externe)
```

## 🚀 Installation et lancement

### 1. Pré-requis

```bash
pip install torch torchvision scikit-learn matplotlib numpy pandas seaborn
```

### 2. Lancer l'entraînement (tous les modèles)

```bash
python main.py
```

Durée approximative sur CPU : ~5-10 minutes (avec les hyperparamètres réduits).

### 3. Lancer une phase spécifique

```bash
# Seulement le MLP
python -c "from models.mlp_model import run_mlp_experiment; run_mlp_experiment()"

# Seulement le CNN
python -c "from models.cnn_model import run_cnn_experiment; run_cnn_experiment()"

# RNN/LSTM/GRU
python -c "from models.rnn_models import run_all_rnn_experiments; run_all_rnn_experiments()"

# CNN-LSTM hybride
python -c "from models.hybrid_model import run_hybrid_experiment; run_hybrid_experiment()"
```

## 📈 Résultats

| Modèle | Dataset | Accuracy | F1-Score | Paramètres |
|--------|---------|----------|----------|------------|
| MLP | Wine | 97.22% | 97.52% | 3,267 |
| CNN | MNIST | 97.20% | 97.19% | 98,666 |
| RNN | Jena Climate | 66.71% | 65.57% | 5,380 |
| LSTM | Jena Climate | 69.71% | 68.31% | 20,740 |
| GRU | Jena Climate | 67.86% | 67.39% | 15,620 |
| **CNN-LSTM** | **Jena Climate** | **71.93%** | **71.55%** | **41,316** |

## 🛠️ Technologies

- **Python 3.10+**
- **PyTorch** (deep learning)
- **scikit-learn** (prétraitement, métriques)
- **torchvision** (MNIST)
- **matplotlib + seaborn** (visualisation)
- **pandas + numpy** (manipulation de données)


