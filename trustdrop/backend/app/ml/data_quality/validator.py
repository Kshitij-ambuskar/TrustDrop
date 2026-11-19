"""Input validation + data-quality / out-of-distribution (OOD) detection.

Three outcomes, deliberately different from MODEL uncertainty:
  INVALID  - cannot be a real record (missing, wrong type, out of range, impossible combination).
             The model is NOT run.
  UNUSUAL  - valid, but atypical compared with TRAINING data (outside training range, extreme robust
             z-score, unseen category code, or high Mahalanobis distance). Prediction allowed, flagged.
  NORMAL   - valid and typical.
Reference statistics are fitted on the TRAINING split only."""
import math

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from app.ml.preprocessing.feature_meta import LABELS
from app.ml.preprocessing.schema import BINARY, FEATURES, NOMINAL, NUMERIC, VALID_RANGES

OOD_PERCENTILE = 99.5          # Mahalanobis threshold = this percentile of TRAINING distances
Z_LIMIT = 6.0                  # robust z-score (median/MAD) beyond which a value is 'extreme'
RULE_RELAX_RATE = 0.005        # if >0.5% of training rows break a combo rule, it is only a warning
INTEGER_FEATURES = set(NOMINAL + BINARY + ["application_order", "age_at_enrollment"]
                       + [f for f in NUMERIC if f.startswith("cu") and not f.endswith("grade")])


def _rule_violations(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {f"cu{s}_approved_exceeds_enrolled":
            df[f"cu{s}_approved"] > df[f"cu{s}_enrolled"] + df[f"cu{s}_credited"] for s in (1, 2)}


def build_reference(X_train: pd.DataFrame) -> dict:
    """Learn 'what normal looks like' from TRAINING rows only."""
    num = X_train[NUMERIC].astype(float)
    med = num.median()
    stats = {}
    for f in NUMERIC:
        mad = float((num[f] - med[f]).abs().median())
        stats[f] = {"min": float(num[f].min()), "max": float(num[f].max()), "median": float(med[f]),
                    "mad": mad, "std": float(num[f].std(ddof=0))}
    filled = num.fillna(med)
    lw = LedoitWolf().fit(filled)                      # shrinkage covariance: stable with correlated features
    d2 = lw.mahalanobis(filled)                        # squared Mahalanobis distance of each training row
    return {"numeric": stats,
            "categories": {f: sorted(int(v) for v in X_train[f].dropna().unique()) for f in NOMINAL},
            "ood": {"model": lw, "median": med, "threshold": float(np.percentile(d2, OOD_PERCENTILE)),
                    "train_d2_sorted": np.sort(d2), "percentile": OOD_PERCENTILE},
            "rule_violation_rate": {k: float(v.mean()) for k, v in _rule_violations(X_train).items()},
            "n_train": int(len(X_train))}


def _to_float(v):
    """Returns (float|None, error_code|None)."""
    if v is None or isinstance(v, bool):
        return None, "missing_value" if v is None else "not_numeric"
    if isinstance(v, str):
        v = v.strip()
        if v == "":
            return None, "missing_value"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None, "not_numeric"
    return (None, "not_numeric") if (math.isnan(x) or math.isinf(x)) else (x, None)


def _issue(field, level, code, msg):
    return {"field": field, "level": level, "code": code, "message": msg}


def validate_student(raw: dict, reference: dict) -> dict:
    """Validate ONE raw student record. Returns status, issues, cleaned values (if valid) and OOD info."""
    issues, vals = [], {}
    if not isinstance(raw, dict):
        return {"status": "invalid", "issues": [_issue(None, "error", "bad_payload", "Student data must be an object.")],
                "cleaned": None, "ood": None, "summary": "INVALID INPUT"}
    for k in raw:
        if k not in FEATURES:
            issues.append(_issue(k, "warning", "unknown_field", f"Unknown field '{k}' was ignored."))
    for f in FEATURES:
        label = LABELS[f]
        x, err = _to_float(raw.get(f))
        if err == "missing_value":
            issues.append(_issue(f, "error", err, f"{label} is required."))
        elif err:
            issues.append(_issue(f, "error", err, f"{label} must be a number."))
        else:
            vals[f] = x
    for f, x in list(vals.items()):
        label = LABELS[f]
        if f in INTEGER_FEATURES and x != int(x):
            issues.append(_issue(f, "error", "not_integer", f"{label} must be a whole number.")); vals.pop(f); continue
        if f in VALID_RANGES and not (VALID_RANGES[f].lo <= x <= VALID_RANGES[f].hi):
            r = VALID_RANGES[f]
            issues.append(_issue(f, "error", "out_of_range",
                                 f"{label} must be within the valid range {r.lo:g}-{r.hi:g} (received {x:g})."))
        elif f in BINARY and x not in (0, 1):
            issues.append(_issue(f, "error", "invalid_category", f"{label} must be 0 or 1 (received {x:g})."))
        elif f in NOMINAL and x < 0:
            issues.append(_issue(f, "error", "invalid_category", f"{label} must be a non-negative code."))
    # impossible combinations (only when both parts are individually usable)
    for s in (1, 2):
        a, e, c = (vals.get(f"cu{s}_{k}") for k in ("approved", "enrolled", "credited"))
        if None not in (a, e, c) and a > e + c:
            rate = reference["rule_violation_rate"].get(f"cu{s}_approved_exceeds_enrolled", 0.0)
            level = "error" if rate <= RULE_RELAX_RATE else "warning"
            issues.append(_issue(f"cu{s}_approved", level, "impossible_combination",
                                 f"Semester {s}: approved units ({a:g}) exceed enrolled + credited units ({e + c:g})."))
    errors = [i for i in issues if i["level"] == "error"]
    if errors:
        return {"status": "invalid", "issues": issues, "cleaned": None, "ood": None, "summary": "INVALID INPUT"}

    cleaned = {f: (int(vals[f]) if f in INTEGER_FEATURES else float(vals[f])) for f in FEATURES}
    for f in NOMINAL:  # valid but never seen in training
        if cleaned[f] not in reference["categories"][f]:
            issues.append(_issue(f, "warning", "unseen_category",
                                 f"{LABELS[f]} code {cleaned[f]} never appeared in the training data."))
    for f in NUMERIC:
        s, x = reference["numeric"][f], cleaned[f]
        if x < s["min"] or x > s["max"]:
            issues.append(_issue(f, "warning", "outside_training_range",
                                 f"{LABELS[f]} ({x:g}) is outside the range seen in training ({s['min']:g}-{s['max']:g})."))
        elif s["mad"] > 0 and abs(x - s["median"]) / (1.4826 * s["mad"]) > Z_LIMIT:
            issues.append(_issue(f, "warning", "extreme_value", f"{LABELS[f]} ({x:g}) is extreme relative to typical values."))
    ood = ood_score(cleaned, reference)
    if ood["is_ood"]:
        issues.append(_issue(None, "warning", "ood", "Overall profile is unusual compared with the training data (multivariate outlier)."))
    status = "unusual" if issues and any(i["level"] == "warning" and i["code"] != "unknown_field" for i in issues) else "normal"
    return {"status": status, "issues": issues, "cleaned": cleaned, "ood": ood,
            "summary": {"normal": "Normal input", "unusual": "Valid but unusual input"}[status]}


def ood_score(cleaned: dict, reference: dict) -> dict:
    """Mahalanobis distance of the numeric profile from the training distribution.
    A distance-based heuristic: it flags atypical combinations, it does not prove the model is wrong."""
    o = reference["ood"]
    x = pd.DataFrame([{f: cleaned[f] for f in NUMERIC}])
    d2 = float(o["model"].mahalanobis(x.fillna(o["median"]))[0])
    pct = float(np.searchsorted(o["train_d2_sorted"], d2) / len(o["train_d2_sorted"]) * 100)
    return {"mahalanobis_sq": d2, "threshold": o["threshold"], "train_percentile": min(pct, 100.0),
            "is_ood": bool(d2 > o["threshold"]),
            "method": f"Mahalanobis distance (Ledoit-Wolf covariance), threshold = {o['percentile']}th percentile of training rows"}
