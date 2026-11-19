"""Builds all SYSTEM-LEVEL trust artifacts at training time and stores them in the model bundle + reports/trust/.

Leakage policy: reference statistics (data-quality/OOD) and the SHAP background come from the TRAINING split.
Fairness, calibration, robustness, uncertainty and data-quality summaries are EVALUATIONS on the held-out
test split - they are reported, never used to fit or tune anything."""
import logging
from pathlib import Path

import numpy as np

from app.ml.data_quality.validator import build_reference, validate_student
from app.ml.evaluation.reporting import write_json
from app.ml.explainability.shap_explainer import ShapExplainer
from app.ml.fairness.audit import audit
from app.ml.robustness.robustness import global_robustness
from app.ml.trust import indicator
from app.ml.uncertainty.uncertainty import level_from, normalized_entropy
from app.ml.preprocessing.schema import FEATURES

log = logging.getLogger(__name__)


def build_trust_artifacts(final, base, X_train, X_test, y_test, label: str, reports_dir: Path,
                          seed: int, calibration_after: dict) -> dict:
    out_dir = Path(reports_dir) / "trust"
    reference = build_reference(X_train)
    background = X_train.sample(min(100, len(X_train)), random_state=seed)[FEATURES]
    proba_test = final.predict_proba(X_test)
    predict = lambda df: final.predict_proba(df[FEATURES])  # noqa: E731
    art = {"provenance": label, "errors": {}}
    comps: dict[str, float | None] = {}

    # --- explainability (global SHAP on a TRAINING sample) ---
    try:
        ex = ShapExplainer(base, background)
        n = 40 if ex.kind == "KernelExplainer" else 200
        art["global_importance"] = {"provenance": label, **ex.global_importance(X_train.sample(min(n, len(X_train)), random_state=seed))}
        comps["explainability"] = 1.0
    except Exception as e:  # noqa: BLE001 - SHAP must never break training
        log.exception("Global SHAP failed")
        art["global_importance"], art["errors"]["explainability"] = None, str(e)
        comps["explainability"] = 0.0

    # --- uncertainty (system summary on test set) ---
    h = normalized_entropy(proba_test)
    top2 = np.sort(proba_test, axis=1)[:, -2:]
    levels = [level_from(a, b - c) for a, (c, b) in zip(h, top2)]
    art["uncertainty"] = {"provenance": label, "mean_normalized_entropy": float(h.mean()),
                          "share_high": float(np.mean([lv == "HIGH" for lv in levels])),
                          "share_medium": float(np.mean([lv == "MEDIUM" for lv in levels])),
                          "share_low": float(np.mean([lv == "LOW" for lv in levels]))}
    comps["uncertainty"] = float(1 - h.mean())

    # --- calibration ---
    comps["calibration"] = indicator.calibration_score(calibration_after["ece"])
    art["calibration"] = {"provenance": label, "ece": calibration_after["ece"], "brier_score": calibration_after["brier_score"]}

    # --- fairness ---
    try:
        art["fairness"] = audit(X_test, y_test, proba_test, label)
        comps["fairness"] = indicator.fairness_score(art["fairness"]["worst_equalized_odds_diff"])
    except Exception as e:  # noqa: BLE001
        log.exception("Fairness audit failed")
        art["fairness"], art["errors"]["fairness"] = None, str(e)
        comps["fairness"] = None

    # --- robustness (global) ---
    try:
        art["robustness"] = {"provenance": label, **global_robustness(X_test, predict, reference, seed=seed)}
        comps["robustness"] = art["robustness"]["mean_score"]
    except Exception as e:  # noqa: BLE001
        log.exception("Robustness failed")
        art["robustness"], art["errors"]["robustness"] = None, str(e)
        comps["robustness"] = None

    # --- data quality (share of held-out rows that validate as 'normal') ---
    sample = X_test.head(500)
    statuses = [validate_student(r.to_dict(), reference)["status"] for _, r in sample.iterrows()]
    art["data_quality"] = {"provenance": label, "n_checked": len(statuses),
                           **{s: int(statuses.count(s)) for s in ("normal", "unusual", "invalid")}}
    comps["data_quality"] = statuses.count("normal") / len(statuses)

    art["indicator"] = {"provenance": label, **indicator.compute(comps)}
    for key in ("global_importance", "fairness", "robustness", "uncertainty", "calibration", "data_quality", "indicator"):
        if art.get(key) is not None:
            write_json(art[key], out_dir / f"{key}.json")
    art["reference"], art["background"] = reference, background
    return art
