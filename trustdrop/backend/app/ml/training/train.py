"""End-to-end training pipeline (Stage 2). Orchestrates the modules; contains no ML logic itself.

load -> inspect -> split -> CV every model (train only) -> select best (CV only) ->
fit on train -> evaluate on test -> calibrate (train only) -> save model bundle + reports.
The test set is touched ONLY for final reporting."""
import logging
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from app.config import settings
from app.ml.calibration.calibration import evaluate_calibration, fit_calibrated_model
from app.ml.calibration.plots import save_reliability_plot
from app.ml.evaluation.metrics import compute_metrics, confusion_report, cross_validate_model
from app.ml.evaluation.reporting import comparison_table, summary_markdown, write_json
from app.ml.prediction.predictor import BUNDLE_NAME
from app.ml.preprocessing.data_loader import load_dataset
from app.ml.preprocessing.pipeline import split_data
from app.ml.preprocessing.schema import CLASSES, FEATURES, TARGET
from app.ml.training.models import MODEL_NAMES, get_model_specs
from app.ml.trust.builder import build_trust_artifacts

log = logging.getLogger(__name__)

SELECTION_CRITERION = ("highest mean cross-validated macro-F1 on the TRAINING split "
                       "(ties broken by mean one-vs-rest macro ROC-AUC). The test set is never used for selection.")
SYNTH_LABEL = "DEVELOPMENT / SYNTHETIC DATA - pipeline validation only, NOT real project results"
REAL_LABEL = "Real UCI dataset (Predict Students' Dropout and Academic Success)"


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Class balance + missingness; justifies the class-imbalance handling."""
    counts = df[TARGET].value_counts()
    return {"n_rows": int(len(df)), "n_features": len(FEATURES),
            "class_counts": {c: int(counts.get(c, 0)) for c in CLASSES},
            "class_share": {c: round(float(counts.get(c, 0) / len(df)), 4) for c in CLASSES},
            "imbalance_ratio_max_over_min": round(float(counts.max() / counts.min()), 3),
            "missing_values_total": int(df[FEATURES].isna().sum().sum()),
            "imbalance_handling": "balanced class weights / sample weights computed from training labels inside each fit"}


def select_best(results: dict[str, dict]) -> str:
    return max(results, key=lambda n: (results[n]["cv"]["mean"]["f1_macro"],
                                       results[n]["cv"]["mean"]["roc_auc_ovr_macro"]))


def run_training(df: pd.DataFrame, is_synthetic: bool, models_dir: Path | None = None,
                 reports_dir: Path | None = None, n_splits: int = 5, model_names: list[str] | None = None,
                 seed: int | None = None, build_trust: bool = True) -> dict:
    models_dir, reports_dir = Path(models_dir or settings.models_dir), Path(reports_dir or settings.reports_dir)
    seed = settings.random_seed if seed is None else seed
    label = SYNTH_LABEL if is_synthetic else REAL_LABEL
    X_train, X_test, y_train, y_test = split_data(df)
    specs = {n: p for n, p in get_model_specs(seed).items() if n in (model_names or MODEL_NAMES)}
    write_json({"provenance": label, **inspect_dataset(df)}, reports_dir / "dataset_summary.json")

    results, fitted = {}, {}
    for name, pipe in specs.items():
        log.info("Cross-validating %s ...", name)
        cv = cross_validate_model(pipe, X_train, y_train, n_splits, seed)
        model = clone(pipe).fit(X_train, y_train)          # final fit on the whole training split
        proba = model.predict_proba(X_test)                # test set: reporting only
        results[name] = {"cv": cv, "test": compute_metrics(y_test, proba)}
        fitted[name] = model
        write_json({"provenance": label, "model": name, "evaluated_on": "held-out test split",
                    **confusion_report(y_test, proba.argmax(axis=1))},
                   reports_dir / "confusion_matrices" / f"{name.lower().replace(' ', '_')}.json")

    best = select_best(results)
    table = comparison_table(results)
    write_json({"provenance": label, "n_train": len(X_train), "n_test": len(X_test), "cv_folds": n_splits,
                "multiclass_strategy": "macro precision/recall/F1; one-vs-rest macro ROC-AUC",
                "selection_criterion": SELECTION_CRITERION, "selected_model": best, "results": results},
               reports_dir / "model_comparison.json")
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "model_comparison_summary.md").write_text(
        summary_markdown(table, best, SELECTION_CRITERION, label, n_splits, len(X_train), len(X_test)))

    # ---- calibration of the selected model (fitted on TRAIN only; evaluated on TEST) ----
    final, base, sel = fit_calibrated_model(specs[best], X_train, y_train, seed)
    before = evaluate_calibration(y_test, base.predict_proba(X_test))
    after = evaluate_calibration(y_test, final.predict_proba(X_test))
    cal_dir = reports_dir / "calibration"
    write_json({"provenance": label, "model": best, "selection": sel,
                "before_calibration": before, "after_calibration": after,
                "note": "If calibration_applied is false, 'after' equals 'before' (no calibrator was warranted)."},
               cal_dir / "calibration_report.json")
    save_reliability_plot(before, after, cal_dir / "reliability_curve.png",
                          " - SYNTHETIC" if is_synthetic else "")

    bundle = {"model": final, "base_pipeline": base, "model_name": best, "classes": CLASSES,
              "features": FEATURES, "calibration": {"applied": sel["calibration_applied"], "method": sel["chosen"]},
              "provenance": {"label": label, "is_synthetic": is_synthetic}, "seed": seed,
              "versions": {"sklearn": sklearn.__version__, "python": platform.python_version()}}
    if build_trust:  # Stage 3: system-level trust artifacts (reference stats, SHAP, fairness, robustness, indicator)
        bundle["trust"] = build_trust_artifacts(final, base, X_train, X_test, y_test, label, reports_dir, seed, after)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, models_dir / BUNDLE_NAME)
    log.info("Selected %s; saved %s", best, models_dir / BUNDLE_NAME)
    return {"best": best, "results": results, "calibration": {"before": before, "after": after, "selection": sel},
            "provenance": label}


def main(allow_synthetic: bool = False, n_splits: int = 5) -> dict:
    data = load_dataset(allow_synthetic=allow_synthetic)
    return run_training(data.df, data.is_synthetic, n_splits=n_splits)
