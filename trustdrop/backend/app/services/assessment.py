"""TrustEngine: the single place that connects validation -> prediction -> uncertainty -> explanation
-> robustness -> trust indicator for ONE student. API routes only call this; no ML logic lives in routes."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.data_quality.validator import validate_student
from app.ml.explainability.shap_explainer import ShapExplainer
from app.ml.prediction.predictor import load_bundle, predict_student
from app.ml.prediction.risk import risk_level, safe_statement
from app.ml.preprocessing.schema import CLASSES, FEATURES
from app.ml.robustness.robustness import run_robustness
from app.ml.trust import indicator
from app.ml.uncertainty.uncertainty import analyze
from app.services.reports import ReportStore

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class TrustEngine:
    def __init__(self, bundle: dict, reports_dir: Path):
        self.bundle, self.reports = bundle, ReportStore(reports_dir)
        self.trust = bundle.get("trust")
        self.is_synthetic = bundle["provenance"]["is_synthetic"]
        self.label = bundle["provenance"]["label"]
        self._explainer, self._explainer_error = None, None
        if self.trust:
            try:
                self._explainer = ShapExplainer(bundle["base_pipeline"], self.trust["background"])
            except Exception as e:  # noqa: BLE001 - degrade gracefully, never crash startup
                log.exception("SHAP explainer unavailable")
                self._explainer_error = str(e)

    @classmethod
    def load(cls, models_dir: Path, reports_dir: Path) -> "TrustEngine":
        return cls(load_bundle(Path(models_dir) / "best_model.joblib"), reports_dir)

    # ---------- helpers ----------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.bundle["model"].predict_proba(df[FEATURES])

    def _ensemble_std(self, row: dict) -> float | None:
        """Disagreement between the cross-fitted ensemble members (epistemic-uncertainty proxy)."""
        members = getattr(self.bundle["model"], "calibrated_classifiers_", None)
        if not members:
            return None
        X = pd.DataFrame([row])[FEATURES]
        return float(np.std([m.predict_proba(X)[0] for m in members], axis=0).mean())

    def model_info(self) -> dict:
        b = self.bundle
        return {"model_name": b["model_name"], "classes": b["classes"], "n_features": len(b["features"]),
                "calibration": b["calibration"], "provenance": self.label, "is_synthetic": self.is_synthetic,
                "versions": b.get("versions", {}), "has_trust_artifacts": self.trust is not None,
                "disclaimer": "Research/educational decision-support prototype. Not for automated decisions about students."}

    def validate(self, raw: dict) -> dict:
        if not self.trust:
            raise ModelUnavailableError("Model bundle has no trust reference data; retrain.")
        return validate_student(raw, self.trust["reference"])

    # ---------- main entry ----------
    def assess(self, raw: dict, n_perturbations: int = 30, with_explanation: bool = True) -> dict:
        val = self.validate(raw)
        if val["status"] == "invalid":
            return {"status": "invalid", "validation": _public(val), "message": "INVALID INPUT"}
        row = val["cleaned"]
        pred = predict_student(row, self.bundle)
        p = np.array([pred.probabilities[c] for c in CLASSES])
        p_drop = float(pred.probabilities["Dropout"])
        risk = risk_level(p_drop)
        unc = analyze(p, CLASSES, self._ensemble_std(row))
        rob = run_robustness(row, self.predict_proba, self.trust["reference"], n=n_perturbations)
        expl = self._explain(row, int(p.argmax())) if with_explanation else {"available": False, "error": "not requested"}

        sysi = self.trust["indicator"]["components"]
        comps = {"explainability": 1.0 if expl.get("available") else 0.0,
                 "uncertainty": float(1 - unc["normalized_entropy"]),
                 "calibration": sysi["calibration"]["score"], "fairness": sysi["fairness"]["score"],
                 "robustness": rob["metrics"]["score"],
                 "data_quality": 1.0 if val["status"] == "normal" else 0.5}
        reasons = []
        if unc["level"] == "HIGH":
            reasons.append("High predictive uncertainty (the model is torn between classes).")
        if risk == "HIGH":
            reasons.append("High estimated dropout probability - counselor review is recommended before any action.")
        if val["status"] == "unusual":
            reasons.append("Input is valid but unusual compared with the training data.")
        if rob["metrics"]["status"] != "stable":
            reasons.append("Prediction is sensitive to small valid input changes.")
        return {
            "status": "ok", "provenance": self.label, "is_synthetic": self.is_synthetic,
            "validation": _public(val),
            "prediction": {**pred.to_dict(), "p_dropout": p_drop, "risk_level": risk,
                           "statement": safe_statement(p_dropout=p_drop, level=risk)},
            "uncertainty": unc, "explanation": expl, "robustness": rob,
            "calibration": self.calibration_info(),
            "fairness": self.fairness_summary(),
            "trust": {"prediction_level": indicator.compute(comps), "system_level": self.trust["indicator"]},
            "review": {"recommended": bool(reasons), "reasons": reasons,
                       "message": ("Human counselor review recommended." if reasons else
                                   "No automatic review trigger fired; a counselor still makes the final decision."),
                       "final_decision_by": "human counselor"},
        }

    def _explain(self, row: dict, idx: int) -> dict:
        if self._explainer is None:
            return {"available": False, "error": self._explainer_error or "SHAP explainer not initialised"}
        try:
            return self._explainer.explain_one(row, idx)
        except Exception as e:  # noqa: BLE001
            log.exception("SHAP explanation failed")
            return {"available": False, "error": str(e)}

    def calibration_info(self) -> dict:
        rep = self.reports.read("calibration/calibration_report.json")
        if not rep:
            return {"available": False}
        return {"available": True, "applied": rep["selection"]["calibration_applied"], "method": rep["selection"]["chosen"],
                "before": {k: rep["before_calibration"][k] for k in ("brier_score", "ece")},
                "after": {k: rep["after_calibration"][k] for k in ("brier_score", "ece")},
                "reliability_after": rep["after_calibration"]["reliability_curve"],
                "reliability_before": rep["before_calibration"]["reliability_curve"], "provenance": rep["provenance"]}

    def fairness_summary(self) -> dict:
        f = self.trust.get("fairness")
        if not f:
            return {"available": False}
        return {"available": True, "overall_status": f["overall_status"],
                "attribute_status": {k: v["status"] for k, v in f["attributes"].items()},
                "note": "Fairness is a property of the model across groups (held-out audit), not of this single prediction."}

    def global_explanation(self) -> dict | None:
        return self.trust.get("global_importance") if self.trust else None


def _public(val: dict) -> dict:
    """Strip internals (model objects) from a validation result before sending to clients."""
    return {k: v for k, v in val.items() if k != "cleaned"} | ({"cleaned": val["cleaned"]} if val.get("cleaned") else {})
