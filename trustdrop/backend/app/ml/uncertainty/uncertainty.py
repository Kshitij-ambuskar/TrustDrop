"""Predictive uncertainty from class probabilities.

For probabilities p over K classes:
  confidence      = max_k p_k
  margin          = p_(1) - p_(2)            (gap between top two classes; small => model torn)
  entropy         = -sum_k p_k ln p_k        (Shannon; 0 = certain, ln K = uniform guess)
  norm. entropy   = entropy / ln K  in [0,1]
These describe how DECIDED the model is. They are NOT a probability that the prediction is correct
(that is only meaningful to the extent probabilities are calibrated - see the calibration module).
Level thresholds below are project-defined heuristics, not universal constants."""
import numpy as np

HIGH_ENTROPY, HIGH_MARGIN = 0.85, 0.15   # HIGH  if norm_entropy > 0.85 OR margin < 0.15
LOW_ENTROPY, LOW_MARGIN = 0.65, 0.40     # LOW   if norm_entropy <= 0.65 AND margin >= 0.40
THRESHOLDS = {"high": f"normalized entropy > {HIGH_ENTROPY} or margin < {HIGH_MARGIN}",
              "low": f"normalized entropy <= {LOW_ENTROPY} and margin >= {LOW_MARGIN}", "medium": "otherwise"}


def normalized_entropy(proba) -> np.ndarray:
    p = np.clip(np.atleast_2d(proba), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


def level_from(norm_entropy: float, margin: float) -> str:
    if norm_entropy > HIGH_ENTROPY or margin < HIGH_MARGIN:
        return "HIGH"
    return "LOW" if (norm_entropy <= LOW_ENTROPY and margin >= LOW_MARGIN) else "MEDIUM"


def analyze(proba, classes: list[str], ensemble_std: float | None = None) -> dict:
    p = np.asarray(proba, dtype=float)
    order = np.argsort(p)[::-1]
    margin = float(p[order[0]] - p[order[1]])
    h = float(normalized_entropy(p)[0])
    level = level_from(h, margin)
    return {"confidence": float(p[order[0]]), "margin": margin,
            "entropy": float(h * np.log(len(p))), "normalized_entropy": h,
            "top_two": [classes[order[0]], classes[order[1]]], "level": level,
            "ensemble_disagreement": ensemble_std, "flag_for_review": level == "HIGH",
            "thresholds": THRESHOLDS,
            "note": "Uncertainty describes how decided the model is; it is not the probability that the prediction is correct."}
