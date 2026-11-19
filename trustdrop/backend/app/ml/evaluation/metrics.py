"""Metrics + stratified cross-validation, identical for every model (no cherry-picking).

Multiclass strategy:
  * precision / recall / F1 are MACRO averages (each class counts equally, so the small
    'Enrolled' class is not hidden by the large ones). Weighted F1 is reported too.
  * ROC-AUC is One-vs-Rest, macro-averaged, computed from predicted probabilities.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from app.ml.preprocessing.schema import CLASSES

METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
               "f1_macro", "f1_weighted", "roc_auc_ovr_macro", "log_loss"]
LABELS = list(range(len(CLASSES)))


def compute_metrics(y_true, proba) -> dict[str, float]:
    """All metrics from true labels and an (n, 3) probability matrix."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    y_pred = proba.argmax(axis=1)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(y_true, proba, multi_class="ovr",
                                           average="macro", labels=LABELS),
        "log_loss": log_loss(y_true, proba, labels=LABELS),
    }
    return {k: float(v) for k, v in out.items()}


def cross_validate_model(pipeline, X, y, n_splits: int, seed: int) -> dict:
    """Stratified K-fold on TRAINING data only. Every fold re-fits the whole pipeline
    (preprocessing included) on that fold's training part, then scores the held-out part."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, va in skf.split(X, y):
        model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        folds.append(compute_metrics(y.iloc[va], model.predict_proba(X.iloc[va])))
    arr = {k: np.array([f[k] for f in folds]) for k in METRIC_KEYS}
    return {
        "n_splits": n_splits,
        "mean": {k: float(v.mean()) for k, v in arr.items()},
        "std": {k: float(v.std(ddof=1)) for k, v in arr.items()},
        "folds": [{k: float(v) for k, v in f.items()} for f in folds],
    }


def confusion_report(y_true, y_pred) -> dict:
    """Counts and row-normalised (recall per true class) confusion matrix, ready for the UI."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return {"labels": CLASSES, "matrix": cm.tolist(), "normalized_by_true": norm.round(4).tolist()}
