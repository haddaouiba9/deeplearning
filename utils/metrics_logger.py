

import os
import json
import time
import numpy as np
from datetime import datetime


class MetricsLogger:
    """Logger léger pour métriques d'entraînement (style W&B)."""

    def __init__(self, model_name, results_dir):
        self.model_name = model_name
        self.results_dir = results_dir
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
            "epochs": [],
        }
        self.start_time = time.time()
        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1):
        self.history["epochs"].append(epoch)
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["train_acc"].append(float(train_acc))
        self.history["val_acc"].append(float(val_acc))
        self.history["train_f1"].append(float(train_f1))
        self.history["val_f1"].append(float(val_f1))

        if val_f1 > self.best_val_f1:
            self.best_val_f1 = float(val_f1)
            self.best_epoch = int(epoch)

        elapsed = time.time() - self.start_time
        print(f"  Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
              f"Elapsed: {elapsed:.1f}s")

    def save_history(self):
        path = os.path.join(self.results_dir, f"{self.model_name}_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return path

    def save_results(self, results: dict):
        results["model"] = self.model_name
        results["best_val_f1"] = self.best_val_f1
        results["best_epoch"] = self.best_epoch
        path = os.path.join(self.results_dir, f"{self.model_name}_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return path


def compute_classification_metrics(y_true, y_pred, y_proba=None, n_classes=None):
    """Calcule accuracy, F1 (macro), precision, recall et AUC multiclasse."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix
    )
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None and n_classes is not None:
        try:
            if n_classes == 2:
                metrics["auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass
    return metrics
