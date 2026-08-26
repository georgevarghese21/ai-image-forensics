"""Seeding, device selection and metrics."""
import os
import random

import numpy as np
import torch
import yaml
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def set_seed(seed=42):
    """Seed every RNG so runs are reproducible (spec section 24)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """y_true: 1 = AI-generated. y_prob: model's P(AI-generated)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Per-generator slices can contain only fakes, where AUC is undefined.
    # Return nan rather than crashing the whole evaluation.
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
