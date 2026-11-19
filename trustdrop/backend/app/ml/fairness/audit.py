"""Fairness audit on HELD-OUT predictions (evaluation only - nothing here is used to train or tune).

Decision audited: the 'at-risk flag' = model predicts Dropout (this is what triggers counselor attention).
Per group we report accuracy (3-class), flag rate (selection rate), precision, recall (TPR),
FPR, FNR using Fairlearn's MetricFrame. Between-group gaps:
  demographic parity diff = max-min flag rate          (outcomes differ - may reflect different base rates!)
  equal opportunity diff  = max-min TPR                (are truly-at-risk students flagged equally?)
  equalized odds diff     = max(TPR gap, FPR gap)
Status thresholds (project-defined, NOT legal standards): worst gap < 0.10 fair, < 0.20 potential
disparity, else requires review. Differences alone do not prove bias. Groups with n < MIN_N are
excluded from gaps (too noisy)."""
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate

MIN_N = 30
FAIR_BELOW, REVIEW_FROM = 0.10, 0.20
THRESHOLDS = {"fair": f"all gaps < {FAIR_BELOW}", "potential_disparity": f"worst gap in [{FAIR_BELOW}, {REVIEW_FROM})",
              "requires_review": f"worst gap >= {REVIEW_FROM}", "min_group_size": MIN_N}
ATTRIBUTES = {"gender": "Gender", "age_group": "Age group at enrollment", "international": "International student",
              "special_needs": "Educational special needs", "displaced": "Displaced"}
ORDER = ["fair", "potential_disparity", "requires_review"]


def group_labels(X: pd.DataFrame) -> pd.DataFrame:
    g = pd.DataFrame(index=X.index)
    g["gender"] = X["gender"].map({0: "Female", 1: "Male"})
    g["age_group"] = pd.cut(X["age_at_enrollment"], [-1, 20, 25, 200], labels=["<=20", "21-25", ">=26"]).astype(str)
    for f in ("international", "special_needs", "displaced"):
        g[f] = X[f].map({0: "No", 1: "Yes"})
    return g


def _tpr(y, p):
    pos = (y == 1).sum()
    return np.nan if pos == 0 else float(((y == 1) & (p == 1)).sum() / pos)


def _fpr(y, p):
    neg = (y == 0).sum()
    return np.nan if neg == 0 else float(((y == 0) & (p == 1)).sum() / neg)


def _prec(y, p):
    flagged = (p == 1).sum()
    return np.nan if flagged == 0 else float(((y == 1) & (p == 1)).sum() / flagged)


def _gap(series: pd.Series) -> float | None:
    s = series.dropna()
    return float(s.max() - s.min()) if len(s) >= 2 else None


def audit(X: pd.DataFrame, y_true, proba, provenance: str) -> dict:
    y_true, pred = np.asarray(y_true), np.asarray(proba).argmax(axis=1)
    y_bin, p_bin = (y_true == 0).astype(int), (pred == 0).astype(int)   # class 0 = Dropout
    groups = group_labels(X)
    attrs = {}
    for key, label in ATTRIBUTES.items():
        mf = MetricFrame(metrics={"n": lambda y, p: len(y), "actual_dropout_rate": lambda y, p: float(np.mean(y)),
                                  "flag_rate": selection_rate, "recall": _tpr, "fpr": _fpr, "precision": _prec},
                         y_true=y_bin, y_pred=p_bin, sensitive_features=groups[key].values)
        t = mf.by_group.copy()
        t["accuracy"] = pd.Series(pred == y_true).groupby(groups[key].values).mean()
        t["fnr"] = 1 - t["recall"]
        ok = t[t["n"] >= MIN_N]
        excluded = [str(i) for i in t.index[t["n"] < MIN_N]]
        warnings = [f"Group '{g}' has fewer than {MIN_N} students and was excluded from disparity gaps." for g in excluded]
        if len(ok) < 2:
            attrs[key] = {"label": label, "status": "not_evaluated", "groups": _rows(t), "excluded_groups": excluded,
                          "gaps": {}, "warnings": warnings + ["Fewer than two sufficiently large groups."]}
            continue
        dp, eo = _gap(ok["flag_rate"]), _gap(ok["recall"])
        fp = _gap(ok["fpr"])
        eodds = max(v for v in (eo, fp) if v is not None) if (eo is not None or fp is not None) else None
        worst = max(v for v in (dp, eo, eodds) if v is not None)
        status = ORDER[0] if worst < FAIR_BELOW else ORDER[1] if worst < REVIEW_FROM else ORDER[2]
        if status != "fair":
            warnings.append("Gaps can reflect real base-rate differences between groups; treat as a prompt for review, not proof of bias.")
        attrs[key] = {"label": label, "status": status, "groups": _rows(t), "excluded_groups": excluded,
                      "gaps": {"demographic_parity_diff": dp, "equal_opportunity_diff": eo, "fpr_diff": fp,
                               "equalized_odds_diff": eodds}, "warnings": warnings}
    evald = [a for a in attrs.values() if a["status"] != "not_evaluated"]
    overall = "not_evaluated" if not evald else max((a["status"] for a in evald), key=ORDER.index)
    worst_eodds = max((a["gaps"].get("equalized_odds_diff") or 0.0 for a in evald), default=None)
    return {"provenance": provenance, "evaluated_on": "held-out test split", "audited_decision": "Model flags student as Dropout",
            "thresholds": THRESHOLDS, "attributes": attrs, "overall_status": overall,
            "worst_equalized_odds_diff": worst_eodds,
            "caveat": "Gender, age and other audited attributes are also model inputs in this dataset. Group gaps do not by themselves prove discrimination."}


def _rows(t: pd.DataFrame) -> list[dict]:
    out = []
    for g, r in t.iterrows():
        out.append({"group": str(g), **{k: (None if pd.isna(v) else float(v)) for k, v in r.items()}, "n": int(r["n"])})
    return out
