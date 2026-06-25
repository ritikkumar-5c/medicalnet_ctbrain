"""Shared helpers: config loading, manifest reading, metrics, seeding."""
import csv
import os
import random

import numpy as np
import torch

CLASS_NAMES = ["normal", "near_normal", "abnormal"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path, overrides=None):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            # dotted keys like "train.lr"
            d = cfg
            parts = k.split(".")
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
    return cfg


def read_manifest(path, split=None):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            r["label"] = int(r["label"])
            r["n_slices"] = int(r.get("n_slices", 0) or 0)
            rows.append(r)
    return rows


def class_counts(rows, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for r in rows:
        counts[r["label"]] += 1
    return counts


def balanced_class_weights(rows, num_classes):
    """Inverse-frequency CE weights, normalised to mean 1 over *present* classes.

    Classes with zero samples (e.g. abnormal while data is still downloading)
    get weight 0 and are excluded from the normalisation — otherwise their
    (clipped) huge weight would dominate the mean and crush the real classes'
    weights toward zero, silently shrinking gradients.
    """
    counts = class_counts(rows, num_classes).astype(np.float64)
    present = counts > 0
    w = np.zeros(num_classes, dtype=np.float64)
    if present.any():
        n_present = int(present.sum())
        w[present] = counts[present].sum() / (n_present * counts[present])
        w[present] /= w[present].mean()  # mean 1 over present classes
    return w.astype(np.float32)


def sample_weights(rows, num_classes):
    """Per-sample weights for WeightedRandomSampler (inverse class freq)."""
    counts = class_counts(rows, num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)
    per_class = 1.0 / counts
    return np.array([per_class[r["label"]] for r in rows], dtype=np.float64)


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    """Return a dict of classification metrics robust to absent classes."""
    from sklearn.metrics import (
        balanced_accuracy_score, confusion_matrix, f1_score,
        precision_recall_fscore_support, roc_auc_score,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   labels=list(range(num_classes)), zero_division=0)),
    }
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    out["confusion_matrix"] = cm.tolist()

    # Per-class sensitivity (= recall) and specificity, one-vs-rest from the
    # confusion matrix. For class i: TP=cm[i,i]; FN=row i minus TP;
    # FP=col i minus TP; TN=everything else.
    total = cm.sum()
    sens_list, spec_list = [], []
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        out[f"f1_{name}"] = float(f[i])
        out[f"precision_{name}"] = float(p[i])
        out[f"recall_{name}"] = float(r[i])
        out[f"sensitivity_{name}"] = float(sensitivity)  # == recall
        out[f"specificity_{name}"] = float(specificity)
        # only average over classes that have ground-truth samples present
        if (tp + fn) > 0:
            sens_list.append(sensitivity)
            spec_list.append(specificity)
    out["macro_sensitivity"] = float(np.mean(sens_list)) if sens_list else 0.0
    out["macro_specificity"] = float(np.mean(spec_list)) if spec_list else 0.0

    # Macro AUC (one-vs-rest) — only over classes present in y_true.
    try:
        present = sorted(set(int(t) for t in y_true))
        if y_prob is not None and len(present) >= 2:
            yp = np.asarray(y_prob)
            out["macro_auc"] = float(roc_auc_score(
                y_true, yp, multi_class="ovr", average="macro",
                labels=list(range(num_classes)),
            ))
    except Exception:
        pass
    return out
