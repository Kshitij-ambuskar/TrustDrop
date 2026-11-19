"""Tests for the Trustworthiness Engine modules (known-answer tests, not just 'it runs')."""
import numpy as np
import pandas as pd
import pytest

from app.ml.data_quality.validator import build_reference, validate_student
from app.ml.explainability.shap_explainer import ShapExplainer
from app.ml.fairness.audit import MIN_N, audit
from app.ml.prediction.risk import risk_level, safe_statement
from app.ml.preprocessing.pipeline import split_data
from app.ml.preprocessing.schema import CLASSES, FEATURES, VALID_RANGES
from app.ml.robustness.robustness import perturb_frame, run_robustness
from app.ml.training.models import get_model_specs
from app.ml.trust import indicator
from app.ml.uncertainty.uncertainty import analyze, normalized_entropy
from tests.scripts_import import make_synth


@pytest.fixture(scope="module")
def data():
    df = make_synth(900)
    Xtr, Xte, ytr, yte = split_data(df)
    return Xtr, Xte, ytr, yte


@pytest.fixture(scope="module")
def ref(data):
    return build_reference(data[0])


@pytest.fixture
def row(data):
    return data[1].iloc[0][FEATURES].to_dict()


# ---------------- data quality ----------------
def test_normal_input_is_normal(ref, data):
    statuses = [validate_student(r.to_dict(), ref)["status"] for _, r in data[1].head(100).iterrows()]
    assert statuses.count("invalid") == 0 and statuses.count("normal") >= 90


def test_out_of_range_is_invalid_with_clear_message(ref, row):
    row["admission_grade"] = 250
    v = validate_student(row, ref)
    assert v["status"] == "invalid" and v["cleaned"] is None
    assert any(i["field"] == "admission_grade" and i["code"] == "out_of_range" and "0-200" in i["message"] for i in v["issues"])


@pytest.mark.parametrize("field,value,code", [
    ("admission_grade", None, "missing_value"), ("admission_grade", "abc", "not_numeric"),
    ("admission_grade", float("nan"), "not_numeric"), ("gender", 2, "invalid_category"),
    ("age_at_enrollment", 12, "out_of_range"), ("age_at_enrollment", 20.5, "not_integer"),
    ("course", -3, "invalid_category"), ("cu1_grade", 25, "out_of_range"), ("debtor", True, "not_numeric")])
def test_invalid_cases(ref, row, field, value, code):
    row[field] = value
    v = validate_student(row, ref)
    assert v["status"] == "invalid" and any(i["field"] == field and i["code"] == code for i in v["issues"])


def test_missing_field_and_bad_payload(ref, row):
    row.pop("gdp")
    assert any(i["field"] == "gdp" for i in validate_student(row, ref)["issues"])
    assert validate_student("nope", ref)["status"] == "invalid"


def test_impossible_combination_is_invalid(ref, row):
    row.update(cu1_enrolled=2, cu1_credited=0, cu1_approved=9)
    v = validate_student(row, ref)
    assert v["status"] == "invalid" and any(i["code"] == "impossible_combination" for i in v["issues"])


def test_valid_but_unusual_is_not_invalid(ref, row):
    row["course"] = 123456          # code never seen in training
    v = validate_student(row, ref)
    assert v["status"] == "unusual" and v["cleaned"] is not None
    assert any(i["code"] == "unseen_category" for i in v["issues"])


def test_extreme_multivariate_profile_flagged_ood(ref, row):
    row.update(unemployment_rate=60, inflation_rate=40, admission_grade=199, cu1_enrolled=50, cu2_enrolled=50)
    v = validate_student(row, ref)
    assert v["status"] == "unusual" and v["ood"]["is_ood"]


def test_ood_threshold_flags_about_half_percent_of_training(ref, data):
    flagged = np.mean([validate_student(r.to_dict(), ref)["ood"]["is_ood"] for _, r in data[0].head(400).iterrows()])
    assert flagged <= 0.03          # threshold = 99.5th percentile of TRAIN distances


# ---------------- uncertainty ----------------
def test_uncertainty_examples_from_spec():
    low = analyze([0.82, 0.12, 0.06], CLASSES)
    high = analyze([0.38, 0.35, 0.27], CLASSES)
    assert low["level"] == "LOW" and low["confidence"] == pytest.approx(0.82) and low["margin"] == pytest.approx(0.70)
    assert high["level"] == "HIGH" and high["flag_for_review"] and high["margin"] == pytest.approx(0.03)
    assert analyze([0.6, 0.3, 0.1], CLASSES)["level"] == "MEDIUM"      # H_norm 0.82, margin 0.30
    assert analyze([0.5, 0.3, 0.2], CLASSES)["level"] == "HIGH"        # H_norm 0.94 > 0.85: top class only 50% of 3


def test_entropy_bounds_and_not_just_one_minus_max():
    assert normalized_entropy([[1 / 3] * 3])[0] == pytest.approx(1.0)
    assert normalized_entropy([[1.0, 0, 0]])[0] == pytest.approx(0.0, abs=1e-9)
    a, b = analyze([0.6, 0.4, 0.0], CLASSES), analyze([0.6, 0.2, 0.2], CLASSES)
    assert a["confidence"] == b["confidence"] and a["normalized_entropy"] != b["normalized_entropy"]


def test_risk_and_safe_wording():
    assert (risk_level(0.8), risk_level(0.4), risk_level(0.1)) == ("HIGH", "MEDIUM", "LOW")
    s = safe_statement(0.82, "HIGH").lower()
    assert "will drop out" not in s and "estimates" in s and "counselor" in s


# ---------------- explainability ----------------
@pytest.mark.parametrize("name", ["Logistic Regression", "XGBoost", "Random Forest"])
def test_shap_is_additive_on_real_model_output(data, name):
    """Local accuracy: base value + sum of feature contributions == the model's own raw output."""
    Xtr, Xte, ytr, _ = data
    pipe = get_model_specs()[name].fit(Xtr, ytr)
    ex = ShapExplainer(pipe, Xtr.sample(60, random_state=0))
    X = Xte.head(5)
    sv = ex.shap_values(X)                                    # (5, 36, 3) aggregated back to original features
    assert sv.shape == (5, 36, 3)
    total = sv.sum(axis=1) + ex.base_values()
    Z = pipe.named_steps["pre"].transform(X)
    from app.ml.training.models import BalancedWeightClassifier
    inner = pipe.named_steps["clf"]; inner = inner.estimator_ if isinstance(inner, BalancedWeightClassifier) else inner
    if name == "Logistic Regression":
        expected = inner.decision_function(Z)
    elif name == "XGBoost":
        expected = inner.predict(Z, output_margin=True)
    else:
        expected = inner.predict_proba(Z)
    np.testing.assert_allclose(total, expected, atol=1e-3)


def test_explain_one_structure_and_top_features(data):
    Xtr, Xte, ytr, _ = data
    pipe = get_model_specs()["Logistic Regression"].fit(Xtr, ytr)
    ex = ShapExplainer(pipe, Xtr.sample(60, random_state=0))
    out = ex.explain_one(Xte.iloc[0][FEATURES].to_dict(), 0)
    assert out["available"] and set(out["by_class"]) == set(CLASSES)
    c = out["by_class"]["Dropout"]["contributions"]
    assert len(c) == 8 and all(abs(c[i]["shap"]) >= abs(c[i + 1]["shap"]) for i in range(7))
    g = ex.global_importance(Xtr.head(100))
    assert sum(f["share"] for f in g["features"]) == pytest.approx(1.0)
    top = [f["feature"] for f in g["features"][:6]]
    assert {"cu1_approved", "cu2_approved", "debtor", "tuition_up_to_date"} & set(top)   # planted signal is recovered


def test_shap_ann_kernel_runs(data):
    Xtr, Xte, ytr, _ = data
    pipe = get_model_specs()["ANN"].fit(Xtr, ytr)
    ex = ShapExplainer(pipe, Xtr.sample(30, random_state=0))
    assert ex.shap_values(Xte.head(2)).shape == (2, 36, 3)


# ---------------- fairness ----------------
def _fair_frame(n_f, n_m):
    n = n_f + n_m
    X = pd.DataFrame({"gender": [0] * n_f + [1] * n_m, "age_at_enrollment": 19, "international": 0,
                      "special_needs": 0, "displaced": 0})
    return X


def test_fairness_known_gaps():
    X = _fair_frame(100, 100)
    y = np.array([0] * 50 + [2] * 50 + [0] * 50 + [2] * 50)       # same true dropout rate (50%) in both groups
    pred = y.copy()
    pred[100:150] = 2                                             # male: every true dropout missed
    proba = np.eye(3)[pred]
    rep = audit(X, y, proba, "TEST")
    g = rep["attributes"]["gender"]
    assert g["gaps"]["equal_opportunity_diff"] == pytest.approx(1.0)
    assert g["gaps"]["demographic_parity_diff"] == pytest.approx(0.5)
    assert g["status"] == "requires_review" and rep["overall_status"] == "requires_review"
    rows = {r["group"]: r for r in g["groups"]}
    assert rows["Female"]["recall"] == 1.0 and rows["Male"]["recall"] == 0.0 and rows["Male"]["fnr"] == 1.0


def test_fairness_perfect_parity_is_fair_and_small_groups_excluded():
    X = _fair_frame(100, 100)
    y = np.tile([0, 2], 100)
    rep = audit(X, y, np.eye(3)[y], "TEST")
    assert rep["attributes"]["gender"]["status"] == "fair"
    X2 = _fair_frame(100, MIN_N - 1)
    y2 = np.tile([0, 2], 65)[:len(X2)]
    r2 = audit(X2, y2, np.eye(3)[y2], "TEST")["attributes"]["gender"]
    assert r2["status"] == "not_evaluated" and "Male" in r2["excluded_groups"]


# ---------------- robustness ----------------
def test_perturbations_stay_valid_and_are_reproducible(ref, row):
    p1 = perturb_frame(row, 50, np.random.default_rng(1), ref)
    p2 = perturb_frame(row, 50, np.random.default_rng(1), ref)
    pd.testing.assert_frame_equal(p1, p2)
    for f, r in VALID_RANGES.items():
        assert p1[f].between(r.lo, r.hi).all(), f
    for s in (1, 2):
        assert (p1[f"cu{s}_approved"] <= p1[f"cu{s}_enrolled"] + p1[f"cu{s}_credited"]).all()
    assert (p1["gender"] == row["gender"]).all() and (p1["course"] == row["course"]).all()  # demographics untouched
    assert all(validate_student(r.to_dict(), ref)["status"] != "invalid" for _, r in p1.iterrows())


def test_robustness_stable_vs_sensitive_models(ref, row):
    const = lambda df: np.tile([0.7, 0.2, 0.1], (len(df), 1))                      # noqa: E731
    stable = run_robustness(row, const, ref, n=20)
    assert stable["metrics"]["stability_rate"] == 1 and stable["metrics"]["score"] == 1 and stable["warning"] is None

    def jumpy(df):  # prediction flips with tiny grade change -> deliberately fragile
        up = (df["admission_grade"].values * 100 % 2 < 1)[:, None]
        return np.where(up, [[0.9, 0.05, 0.05]], [[0.05, 0.9, 0.05]])
    bad = run_robustness(row, jumpy, ref, n=60)
    assert bad["metrics"]["status"] in ("unstable", "sensitive") and bad["warning"] and bad["metrics"]["score"] < 1


# ---------------- trust indicator ----------------
def test_indicator_weights_and_partial_evaluation():
    assert sum(indicator.WEIGHTS.values()) == pytest.approx(1.0)
    full = indicator.compute({k: 0.8 for k in indicator.WEIGHTS})
    assert full["score"] == pytest.approx(0.8) and full["level"] == "High" and not full["partial"]
    part = indicator.compute({**{k: 0.8 for k in indicator.WEIGHTS}, "fairness": None})
    assert part["partial"] and part["evaluated"] == 5 and part["components"]["fairness"]["status"] == "NOT_EVALUATED"
    assert part["score"] == pytest.approx(0.8)             # missing component is NOT counted as 0 or invented
    assert indicator.compute({})["score"] is None and indicator.compute({})["level"] == "Not evaluated"


def test_indicator_caps_level_when_one_component_is_weak():
    r = indicator.compute({"explainability": 1, "uncertainty": 1, "calibration": 1, "fairness": 0.2, "robustness": 1, "data_quality": 1})
    assert r["score"] > 0.75 and r["capped"] and r["level"] == "Moderate" and r["weakest_component"] == "fairness"


def test_component_normalisation():
    assert indicator.calibration_score(0) == 1 and indicator.calibration_score(0.3) == 0
    assert indicator.fairness_score(0.15) == pytest.approx(0.5) and indicator.fairness_score(None) is None
