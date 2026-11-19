"""Stage-2 tests. Uses a SMALL synthetic dataset purely as a test fixture."""
import json

import joblib
import numpy as np
import pytest

from app.ml.calibration.calibration import (evaluate_calibration, expected_calibration_error,
                                            fit_calibrated_model, multiclass_brier, reliability_curve)
from app.ml.evaluation.metrics import METRIC_KEYS, compute_metrics, cross_validate_model
from app.ml.prediction.predictor import BUNDLE_NAME, load_bundle, predict_student, predict_students
from app.ml.preprocessing.pipeline import encode_target, split_data
from app.ml.preprocessing.schema import CLASSES, FEATURES, TARGET
from app.ml.training.models import MODEL_NAMES, get_model_specs
from app.ml.training.train import run_training, select_best
from tests.scripts_import import make_synth


@pytest.fixture(scope="module")
def df():
    return make_synth(700)


@pytest.fixture(scope="module")
def trained(df, tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    out = run_training(df, True, root / "models", root / "reports", n_splits=3)
    return out, root


def test_all_four_models_train_and_predict_valid_probabilities(df):
    Xtr, Xte, ytr, _ = split_data(df)
    specs = get_model_specs()
    assert list(specs) == MODEL_NAMES
    for name, pipe in specs.items():
        pipe.fit(Xtr, ytr)
        assert list(pipe.classes_) == [0, 1, 2], name
        p = pipe.predict_proba(Xte)
        assert p.shape == (len(Xte), 3) and np.allclose(p.sum(axis=1), 1, atol=1e-6), name


def test_cross_validation_runs_and_reports_all_metrics(df):
    Xtr, _, ytr, _ = split_data(df)
    cv = cross_validate_model(get_model_specs()["Logistic Regression"], Xtr, ytr, 3, 42)
    assert cv["n_splits"] == 3 and len(cv["folds"]) == 3
    assert set(cv["mean"]) == set(METRIC_KEYS)
    assert all(0 <= cv["mean"][k] <= 1 for k in ("accuracy", "f1_macro", "roc_auc_ovr_macro"))


def test_compute_metrics_perfect_and_uniform():
    y = np.array([0, 1, 2, 0, 1, 2])
    perfect = compute_metrics(y, np.eye(3)[y] * 0.94 + 0.02)
    assert perfect["accuracy"] == 1 and perfect["f1_macro"] == 1 and perfect["roc_auc_ovr_macro"] == 1


def test_training_creates_all_artifacts(trained):
    out, root = trained
    for f in ["model_comparison.json", "model_comparison.csv", "model_comparison_summary.md",
              "dataset_summary.json", "calibration/calibration_report.json",
              "calibration/reliability_curve.png"] + [f"confusion_matrices/{n.lower().replace(' ', '_')}.json" for n in MODEL_NAMES]:
        assert (root / "reports" / f).exists(), f
    assert (root / "models" / BUNDLE_NAME).exists()
    comp = json.loads((root / "reports" / "model_comparison.json").read_text())
    assert "SYNTHETIC" in comp["provenance"] and set(comp["results"]) == set(MODEL_NAMES)
    cm = json.loads((root / "reports/confusion_matrices/xgboost.json").read_text())
    assert np.array(cm["matrix"]).shape == (3, 3)


def test_best_model_is_chosen_by_cv_f1_not_by_test(trained):
    out, _ = trained
    r = out["results"]
    assert out["best"] == select_best(r)
    assert r[out["best"]]["cv"]["mean"]["f1_macro"] == max(v["cv"]["mean"]["f1_macro"] for v in r.values())


def test_serialization_and_inference_from_raw_values(trained, df):
    _, root = trained
    bundle = load_bundle(root / "models" / BUNDLE_NAME)
    raw = df.iloc[0][FEATURES].to_dict()  # RAW values, no manual transformation
    res = predict_student(raw, bundle)
    assert res.predicted_class in CLASSES and res.classes == CLASSES
    assert sum(res.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert res.confidence == max(res.probabilities.values())
    assert "SYNTHETIC" in res.provenance
    # reloading from disk gives identical probabilities
    again = joblib.load(root / "models" / BUNDLE_NAME)
    assert predict_student(raw, again).probabilities == res.probabilities
    assert len(predict_students([raw, raw], bundle)) == 2


def test_predictor_rejects_missing_features(trained, df):
    _, root = trained
    raw = df.iloc[0][FEATURES].to_dict(); raw.pop("admission_grade")
    with pytest.raises(ValueError, match="admission_grade"):
        predict_student(raw, load_bundle(root / "models" / BUNDLE_NAME))


# ---------------- calibration ----------------
def test_ece_and_brier_known_values():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([[1, 0, 0]] * 2 + [[0, 1, 0]] * 2, dtype=float)
    assert multiclass_brier(y, perfect) == 0 and expected_calibration_error(y, perfect) == 0
    # always 100% confident but only half right -> ECE = |0.5 - 1.0|
    wrong = np.array([[1, 0, 0]] * 4, dtype=float)
    assert expected_calibration_error(y, wrong) == pytest.approx(0.5)
    uniform = np.full((4, 3), 1 / 3)
    assert multiclass_brier(y, uniform) == pytest.approx(2 / 3)


def test_reliability_curve_counts_cover_all_samples():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), 200); y = rng.integers(0, 3, 200)
    assert sum(b["count"] for b in reliability_curve(y, p)) == 200
    ev = evaluate_calibration(y, p)
    assert 0 <= ev["ece"] <= 1 and ev["n_samples"] == 200


def test_calibration_selection_runs_on_training_data_only(df):
    Xtr, Xte, ytr, yte = split_data(df)
    final, base, rep = fit_calibrated_model(get_model_specs()["Logistic Regression"], Xtr, ytr, 42)
    assert rep["chosen"] in {"none", "sigmoid", "isotonic"}
    assert set(rep["oof_brier_by_option"]) == {"none", "sigmoid", "isotonic"}
    assert np.allclose(final.predict_proba(Xte).sum(axis=1), 1, atol=1e-6)


# ---------------- leakage ----------------
def test_preprocessing_is_fit_on_train_split_only(trained, df):
    _, root = trained
    bundle = load_bundle(root / "models" / BUNDLE_NAME)
    Xtr, _, _, _ = split_data(df)
    pre = bundle["base_pipeline"].named_steps["pre"]
    num = pre.named_transformers_["num"]
    idx = 2  # admission_grade position inside NUMERIC
    imp = num.named_steps["impute"]
    assert imp.statistics_[idx] == pytest.approx(Xtr["admission_grade"].median())


def test_corrupting_test_labels_does_not_change_trained_model(df, tmp_path, monkeypatch):
    """End-to-end leakage check. Both runs receive IDENTICAL train data (split is pinned, because
    stratified splitting itself looks at labels); only the TEST labels are scrambled in run 2.
    If preprocessing, model selection or calibration peeked at test labels, results would differ."""
    import app.ml.training.train as train_mod
    Xtr, Xte, ytr, yte = split_data(df)
    scrambled = yte.copy()
    scrambled[:] = np.random.default_rng(1).integers(0, 3, len(yte))
    kw = dict(n_splits=3, model_names=["Logistic Regression", "Random Forest"])
    monkeypatch.setattr(train_mod, "split_data", lambda _df: (Xtr, Xte, ytr, yte))
    a = run_training(df, True, tmp_path / "m1", tmp_path / "r1", **kw)
    monkeypatch.setattr(train_mod, "split_data", lambda _df: (Xtr, Xte, ytr, scrambled))
    b = run_training(df, True, tmp_path / "m2", tmp_path / "r2", **kw)
    assert a["best"] == b["best"]
    assert a["calibration"]["selection"] == b["calibration"]["selection"]
    for n in a["results"]:
        assert a["results"][n]["cv"] == b["results"][n]["cv"]                 # CV never sees test
        assert a["results"][n]["test"] != b["results"][n]["test"]            # sanity: test metrics DO react
    ma, mb = (joblib.load(tmp_path / d / BUNDLE_NAME) for d in ("m1", "m2"))
    np.testing.assert_allclose(ma["model"].predict_proba(Xte), mb["model"].predict_proba(Xte))
