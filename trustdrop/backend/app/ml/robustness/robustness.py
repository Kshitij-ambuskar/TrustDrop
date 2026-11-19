"""Robustness: does the prediction survive SMALL, VALID changes to the input?

Perturbations (each stays inside valid ranges and keeps unit counts consistent):
  * grades (previous qualification, admission, 1st/2nd sem unit grades): Gaussian noise, sd = 0.10 x the
    feature's training standard deviation
  * curricular unit counts (evaluations, approved, per semester): randomly -1, 0 or +1
  * then clipped to valid ranges and approved <= enrolled + credited is enforced
Metrics (for the originally predicted class c): stability = share of perturbed copies with the same
predicted class; delta = |p_c(perturbed) - p_c(original)|; score = stability * (1 - min(1, mean_delta/0.20)).
Status: stable if stability >= 0.90 and max delta <= 0.15; unstable if stability < 0.70 or max delta > 0.30;
otherwise sensitive. Thresholds are project-defined heuristics."""
import numpy as np
import pandas as pd

from app.ml.preprocessing.feature_meta import LABELS
from app.ml.preprocessing.schema import CLASSES, FEATURES, VALID_RANGES

GRADES = ["previous_qualification_grade", "admission_grade", "cu1_grade", "cu2_grade"]
COUNTS = ["cu1_evaluations", "cu1_approved", "cu2_evaluations", "cu2_approved"]
REL_NOISE = 0.10
STABLE = dict(stability=0.90, max_delta=0.15)
UNSTABLE = dict(stability=0.70, max_delta=0.30)
METHOD = ("Grades: Gaussian noise (0.10 x training std). Unit counts: -1/0/+1. Clipped to valid ranges; "
          "approved <= enrolled + credited enforced.")


def perturb_frame(row: dict, n: int, rng: np.random.Generator, reference: dict) -> pd.DataFrame:
    df = pd.DataFrame([row] * n)[FEATURES].astype(float)
    for f in GRADES:
        sd = reference["numeric"][f]["std"]
        df[f] = (df[f] + rng.normal(0, REL_NOISE * sd, n)).round(2)
    for f in COUNTS:
        df[f] = df[f] + rng.choice([-1, 0, 1], n)
    for f in GRADES + COUNTS:
        r = VALID_RANGES[f]
        df[f] = df[f].clip(r.lo, r.hi)
    for s in (1, 2):
        df[f"cu{s}_approved"] = np.minimum(df[f"cu{s}_approved"], df[f"cu{s}_enrolled"] + df[f"cu{s}_credited"])
    return df


def _metrics(p0: np.ndarray, P: np.ndarray) -> dict:
    c = int(p0.argmax())
    stability = float((P.argmax(axis=1) == c).mean())
    delta = np.abs(P[:, c] - p0[c])
    mean_d, max_d = float(delta.mean()), float(delta.max())
    score = stability * (1 - min(1.0, mean_d / 0.20))
    status = ("stable" if stability >= STABLE["stability"] and max_d <= STABLE["max_delta"] else
              "unstable" if stability < UNSTABLE["stability"] or max_d > UNSTABLE["max_delta"] else "sensitive")
    return {"stability_rate": stability, "mean_abs_prob_change": mean_d, "max_abs_prob_change": max_d,
            "score": float(score), "status": status}


def run_robustness(row: dict, predict_proba, reference: dict, n: int = 30, seed: int = 42) -> dict:
    """predict_proba: callable(DataFrame)->(n,3) probabilities of the FINAL model."""
    n = int(min(max(n, 5), 200))
    rng = np.random.default_rng(seed)
    pert = perturb_frame(row, n, rng, reference)
    p0 = predict_proba(pd.DataFrame([row])[FEATURES])[0]
    P = predict_proba(pert)
    m = _metrics(p0, P)
    c = int(p0.argmax())
    items = []
    for i in range(n):
        ch = [{"feature": f, "label": LABELS[f], "from": float(row[f]), "to": float(pert.iloc[i][f])}
              for f in GRADES + COUNTS if abs(float(row[f]) - float(pert.iloc[i][f])) > 1e-9]
        items.append({"changes": ch, "predicted_class": CLASSES[int(P[i].argmax())],
                      "probabilities": {k: float(v) for k, v in zip(CLASSES, P[i])},
                      "changed_prediction": bool(P[i].argmax() != c)})
    warn = None if m["status"] == "stable" else (
        "WARNING: the prediction changed or shifted noticeably under tiny valid input changes - the model may be "
        "sensitive for this student. Treat with extra caution.")
    return {"n_perturbations": n, "seed": seed, "method": METHOD, "original": {
        "predicted_class": CLASSES[c], "probabilities": {k: float(v) for k, v in zip(CLASSES, p0)}},
        "metrics": m, "warning": warn, "perturbations": items}


def global_robustness(X: pd.DataFrame, predict_proba, reference: dict, n_students: int = 100,
                      n_per: int = 15, seed: int = 42) -> dict:
    """System-level robustness on a sample of rows (evaluation only)."""
    rng = np.random.default_rng(seed)
    sample = X.sample(min(n_students, len(X)), random_state=seed)
    res = []
    for _, r in sample.iterrows():
        row = r.to_dict()
        P = predict_proba(perturb_frame(row, n_per, rng, reference))
        res.append(_metrics(predict_proba(pd.DataFrame([row])[FEATURES])[0], P))
    return {"n_students": len(res), "n_perturbations_each": n_per, "method": METHOD,
            "mean_score": float(np.mean([r["score"] for r in res])),
            "mean_stability_rate": float(np.mean([r["stability_rate"] for r in res])),
            "share_stable": float(np.mean([r["status"] == "stable" for r in res])),
            "share_unstable": float(np.mean([r["status"] == "unstable" for r in res]))}
