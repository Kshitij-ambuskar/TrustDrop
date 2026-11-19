"""SHAP explanations computed from the REAL fitted (uncalibrated) base pipeline.

SHAP assigns each feature a signed contribution to the model's output for one prediction; the
contributions add up (with a base value) to that output. The pipeline one-hot encodes nominal columns,
so SHAP values are first computed per encoded column and then SUMMED back to the 36 original features
(valid because SHAP values are additive). Explainer choice by model type:
  XGBoost / Random Forest -> TreeExplainer (exact, fast)    Logistic Regression -> LinearExplainer
  ANN (MLP)               -> KernelExplainer (model-agnostic, approximate, slower)
Output units depend on the model (log-odds for XGBoost/LR, probability for RF/ANN) and are reported."""
import logging

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from app.ml.preprocessing.feature_meta import LABELS
from app.ml.training.models import BalancedWeightClassifier
from app.ml.preprocessing.schema import CLASSES, FEATURES, NOMINAL

log = logging.getLogger(__name__)


def _nfc(sv) -> np.ndarray:
    """Normalise SHAP output (list-per-class or array) to shape (n_rows, n_features, n_classes)."""
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    return sv[..., None] if sv.ndim == 2 else sv


class ShapExplainer:
    def __init__(self, base_pipeline, background: pd.DataFrame, classes: list[str] = CLASSES):
        self.classes = classes
        self.pre = base_pipeline.named_steps["pre"]
        clf = base_pipeline.named_steps["clf"]
        self.inner = clf.estimator_ if isinstance(clf, BalancedWeightClassifier) else clf  # unwrap ONLY our wrapper
        bg = self.pre.transform(background[FEATURES])
        if isinstance(self.inner, XGBClassifier):
            self.kind, self.units = "TreeExplainer", "log-odds (raw model margin)"
            self.expl = shap.TreeExplainer(self.inner)
        elif isinstance(self.inner, RandomForestClassifier):
            self.kind, self.units = "TreeExplainer", "probability"
            self.expl = shap.TreeExplainer(self.inner)
        elif isinstance(self.inner, LogisticRegression):
            self.kind, self.units = "LinearExplainer", "log-odds"
            self.expl = shap.LinearExplainer(self.inner, bg)
        elif isinstance(self.inner, MLPClassifier):
            self.kind, self.units = "KernelExplainer", "probability"
            self.expl = shap.KernelExplainer(self.inner.predict_proba, bg[:25])
        else:
            raise TypeError(f"No SHAP strategy for {type(self.inner).__name__}")
        names = list(self.pre.get_feature_names_out())
        nominal_by_len = sorted(NOMINAL, key=len, reverse=True)
        origin = [n if n in FEATURES else next((f for f in nominal_by_len if n.startswith(f + "_")), n) for n in names]
        self.M = np.zeros((len(names), len(FEATURES)))     # encoded column -> original feature
        for i, o in enumerate(origin):
            self.M[i, FEATURES.index(o)] = 1.0

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """(n_rows, 36 original features, n_classes)."""
        Z = np.asarray(self.pre.transform(X[FEATURES]), dtype=float)
        kw = {"nsamples": 100, "silent": True} if self.kind == "KernelExplainer" else {}
        sv = _nfc(self.expl.shap_values(Z, **kw)) if self.kind != "TreeExplainer" else _nfc(self.expl.shap_values(Z))
        return np.einsum("nfc,fo->noc", sv, self.M)

    def base_values(self) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.expl.expected_value, dtype=float))

    def global_importance(self, X: pd.DataFrame) -> dict:
        """Mean |SHAP| per feature (averaged over classes) on a sample of rows."""
        sv = np.abs(self.shap_values(X))
        overall, per_class = sv.mean(axis=(0, 2)), sv.mean(axis=0)
        total = float(overall.sum()) or 1.0
        rows = [{"feature": f, "label": LABELS[f], "mean_abs_shap": float(overall[i]), "share": float(overall[i] / total),
                 "per_class": {c: float(per_class[i, k]) for k, c in enumerate(self.classes)}}
                for i, f in enumerate(FEATURES)]
        return {"method": self.kind, "units": self.units, "n_rows": int(len(X)),
                "features": sorted(rows, key=lambda r: -r["mean_abs_shap"])}

    def explain_one(self, row: dict, predicted_idx: int, top_k: int = 8) -> dict:
        X = pd.DataFrame([row])[FEATURES]
        sv, base = self.shap_values(X)[0], self.base_values()
        by_class = {}
        for k, c in enumerate(self.classes):
            order = np.argsort(-np.abs(sv[:, k]))[:top_k]
            by_class[c] = {"base_value": float(base[k]) if k < len(base) else None,
                           "contributions": [{"feature": FEATURES[i], "label": LABELS[FEATURES[i]],
                                              "value": row[FEATURES[i]], "shap": float(sv[i, k]),
                                              "effect": "increases" if sv[i, k] > 0 else "decreases"} for i in order]}
        pc = self.classes[predicted_idx]
        top = by_class[pc]["contributions"][:3]
        txt = ", ".join(f"{t['label']} ({'+' if t['shap'] > 0 else '-'})" for t in top)
        return {"available": True, "method": self.kind, "units": self.units, "predicted_class": pc,
                "by_class": by_class,
                "summary": (f"These features contributed most to the model's score for '{pc}': {txt}. "
                            "Contributions describe how the model behaves, not causes of dropout."),
                "note": "Explanations are computed on the underlying uncalibrated model; the displayed probabilities are calibrated."}
