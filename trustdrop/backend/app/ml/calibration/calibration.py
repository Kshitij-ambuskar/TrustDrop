"""Probability calibration: are predicted probabilities honest?
(A model saying '80%' should be right about 80% of the time.)

Leakage rule: calibrators are fitted ONLY on training data. CalibratedClassifierCV does its own
internal cross-fitting (the calibrator for each fold learns from predictions on rows the base model
did not train on). The test set is used only to REPORT before/after calibration."""
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.ml.preprocessing.schema import CLASSES

N_BINS = 10


def multiclass_brier(y_true, proba) -> float:
    """Mean over samples of sum_k (p_k - 1[y=k])^2. 0 = perfect; uniform guess on 3 classes = 0.667."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    onehot = np.eye(proba.shape[1])[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def reliability_curve(y_true, proba, n_bins: int = N_BINS) -> list[dict]:
    """Top-label reliability: bin predictions by confidence (max prob); in each bin compare the
    mean confidence with the actual accuracy. A calibrated model lies on the diagonal."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    conf, correct = proba.max(axis=1), (proba.argmax(axis=1) == y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        bins.append({"bin_lower": float(lo), "bin_upper": float(hi), "count": n,
                     "mean_confidence": float(conf[mask].mean()) if n else None,
                     "accuracy": float(correct[mask].mean()) if n else None})
    return bins


def expected_calibration_error(y_true, proba, n_bins: int = N_BINS) -> float:
    """ECE = sum over bins of (bin_size / N) * |accuracy - mean confidence|."""
    curve = reliability_curve(y_true, proba, n_bins)
    total = sum(b["count"] for b in curve)
    return float(sum(b["count"] / total * abs(b["accuracy"] - b["mean_confidence"])
                     for b in curve if b["count"]))


def evaluate_calibration(y_true, proba) -> dict:
    return {"n_samples": int(len(y_true)), "brier_score": multiclass_brier(y_true, proba),
            "ece": expected_calibration_error(y_true, proba),
            "reliability_curve": reliability_curve(y_true, proba)}


def fit_calibrated_model(pipeline, X_train, y_train, seed: int, cv_inner: int = 3,
                         methods=("sigmoid", "isotonic")):
    """Decide whether (and how) to calibrate, using TRAINING data only.

    1. Get out-of-fold probabilities on the training set for: uncalibrated, Platt (sigmoid),
       isotonic. 2. Compare out-of-fold Brier scores. 3. Calibrate only if a method strictly
       beats the uncalibrated model; refit the winner on the full training set.
    Returns (final_model, base_pipeline_fitted, selection_report)."""
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    def oof(est):
        return cross_val_predict(est, X_train, y_train, cv=outer, method="predict_proba")

    scores = {"none": multiclass_brier(y_train, oof(clone(pipeline)))}
    for m in methods:
        scores[m] = multiclass_brier(
            y_train, oof(CalibratedClassifierCV(clone(pipeline), method=m, cv=cv_inner)))
    best = min(scores, key=scores.get)
    applied = best != "none"
    base = clone(pipeline).fit(X_train, y_train)  # kept uncalibrated for SHAP + 'before' numbers
    final = (CalibratedClassifierCV(clone(pipeline), method=best, cv=cv_inner).fit(X_train, y_train)
             if applied else base)
    report = {"selection_basis": "out-of-fold Brier score on TRAINING data (lower is better)",
              "oof_brier_by_option": scores, "chosen": best, "calibration_applied": applied,
              "cv_inner": cv_inner, "classes": CLASSES}
    return final, base, report
