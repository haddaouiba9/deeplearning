
Datasets RÉELS utilisés :
  1. Wine (sklearn)        → MLP
  2. MNIST (torchvision)   → CNN
  3. Jena Climate (Max Planck Institute, 2009-2016) → RNN, LSTM, GRU, CNN-LSTM
"""

import os
import sys
import json
import time

PROJECT_DIR = "/home/z/my-project/download/deep_learning_project_ismail"
sys.path.insert(0, PROJECT_DIR)

from utils.config import set_seed, DEVICE, RESULTS_DIR, PLOTS_DIR

def main():
    print("="*70)
    print(f"  Device : {DEVICE}")
    print("="*70)

    set_seed()
    all_results = {}
    start_time = time.time()

    # === 1. MLP - Wine ===
    print("\n\n" + "#"*30 + " PHASE 1 : MLP " + "#"*30)
    try:
        from models.mlp_model import run_mlp_experiment
        all_results['MLP'] = run_mlp_experiment()
    except Exception as e:
        print(f"Erreur MLP : {e}")
        import traceback; traceback.print_exc()

    # === 2. CNN - MNIST ===
    print("\n\n" + "#"*30 + " PHASE 2 : CNN " + "#"*30)
    try:
        from models.cnn_model import run_cnn_experiment
        all_results['CNN'] = run_cnn_experiment()
    except Exception as e:
        print(f"Erreur CNN : {e}")
        import traceback; traceback.print_exc()

    # === 3. RNN / LSTM / GRU - Jena Climate ===
    print("\n\n" + "#"*30 + " PHASE 3 : RNN/LSTM/GRU " + "#"*30)
    try:
        from models.rnn_models import run_all_rnn_experiments
        rnn_results = run_all_rnn_experiments()
        all_results.update(rnn_results)
    except Exception as e:
        print(f"Erreur RNN : {e}")
        import traceback; traceback.print_exc()

    # === 4. CNN-LSTM Hybride ===
    print("\n\n" + "#"*30 + " PHASE 4 : HYBRIDE " + "#"*30)
    try:
        from models.hybrid_model import run_hybrid_experiment
        all_results['CNN_LSTM'] = run_hybrid_experiment()
    except Exception as e:
        print(f"Erreur Hybride : {e}")
        import traceback; traceback.print_exc()

    # === Résumé global ===
    elapsed = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"  RÉSUMÉ GLOBAL — ISMAIL HADDAOUI")
    print(f"{'='*70}")
    print(f"  Temps total : {elapsed/60:.1f} minutes")
    print(f"\n  {'Modèle':<12} {'Dataset':<15} {'Accuracy':>10} {'F1-Score':>10} {'Params':>12}")
    print(f"  {'-'*60}")

    datasets_map = {"MLP": "Wine", "CNN": "MNIST",
                    "RNN": "Jena", "LSTM": "Jena", "GRU": "Jena", "CNN_LSTM": "Jena"}
    for name, res in all_results.items():
        acc = res.get('accuracy', 0)
        f1 = res.get('f1_macro', 0)
        params = res.get('total_params', 0)
        ds = datasets_map.get(name, "-")
        print(f"  {name:<12} {ds:<15} {acc:>10.4f} {f1:>10.4f} {params:>12,}")

    # Sauvegarde du résumé
    with open(os.path.join(RESULTS_DIR, 'all_results_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Graphique comparatif global
    plot_global_comparison(all_results)

    print(f"\n  Résultats : {RESULTS_DIR}")
    print(f"  Plots : {PLOTS_DIR}")
    return all_results


def plot_global_comparison(results):
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
    if not models:
        return

    datasets_map = {"MLP": "Wine", "CNN": "MNIST",
                    "RNN": "Jena", "LSTM": "Jena", "GRU": "Jena", "CNN_LSTM": "Jena"}
    labels = [f"{m}\n({datasets_map.get(m, '')})" for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(models))
    width = 0.35

    accuracies = [results[m].get('accuracy', 0) for m in models]
    f1_scores = [results[m].get('f1_macro', 0) for m in models]

    bars1 = axes[0].bar(x - width/2, accuracies, width, label='Accuracy', color='#4C72B0', alpha=0.85)
    bars2 = axes[0].bar(x + width/2, f1_scores, width, label='F1-Score', color='#DD8452', alpha=0.85)
    axes[0].set_xlabel('Modèle'); axes[0].set_ylabel('Score')
    axes[0].set_title('Comparaison globale — Ismail Haddaoui')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    axes[0].legend(); axes[0].set_ylim(0, 1.1); axes[0].grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    params = [results[m].get('total_params', 0) for m in models]
    bars = axes[1].bar(models, params, color='#55A868', alpha=0.85)
    axes[1].set_xlabel('Modèle'); axes[1].set_ylabel('Nombre de paramètres')
    axes[1].set_title('Complexité des modèles — Ismail Haddaoui')
    axes[1].tick_params(axis='x', rotation=45); axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, params):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.02,
                    f'{val:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'global_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  global_comparison.png sauvegardé.")


if __name__ == "__main__":
    main()
