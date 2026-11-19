"""API, database, integration and end-to-end workflow tests. A small model is trained on a synthetic TEST
fixture into a temp dir; the real models/ and reports/ folders are never touched."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.database.models import Student
from app.ml.prediction.predictor import BUNDLE_NAME, load_bundle, predict_student
from app.ml.preprocessing.pipeline import split_data
from app.ml.preprocessing.schema import CLASSES, FEATURES
from app.ml.training.train import run_training
from app.main import create_app
from tests.scripts_import import make_synth


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    df = make_synth(800)
    run_training(df, True, root / "models", root / "reports", n_splits=3,
                 model_names=["Logistic Regression", "Random Forest"])
    _, Xte, _, _ = split_data(df)
    app = create_app(root / "models", root / "reports", "sqlite:///:memory:", seed=False)
    with TestClient(app) as c:
        with app.state.session_factory() as db:           # demo students (synthetic names) from held-out rows
            for i, (_, r) in enumerate(Xte.head(15).iterrows(), 1):
                db.add(Student(display_name=f"Demo Student {i}", source="test",
                               features={f: (int(r[f]) if float(r[f]).is_integer() else float(r[f])) for f in FEATURES}))
            db.commit()
        yield c, root


@pytest.fixture
def client(env):
    return env[0]


def _features(client, sid=1):
    return client.get(f"/students/{sid}").json()["features"]


def test_health_and_model_info_are_labelled_synthetic(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["database_ok"] and h["is_synthetic"] is True
    info = client.get("/model/info").json()
    assert "SYNTHETIC" in info["provenance"] and info["classes"] == CLASSES and info["has_trust_artifacts"]
    assert "decision-support" in info["disclaimer"]


def test_schema_endpoint_describes_all_36_features(client):
    feats = client.get("/schema/features").json()["features"]
    assert len(feats) == 36 and {f["name"] for f in feats} == set(FEATURES)
    assert all("label" in f and "group" in f for f in feats)


def test_validate_endpoint_statuses(client):
    ok = client.post("/validate", json={"features": _features(client)}).json()
    assert ok["status"] in ("normal", "unusual")
    bad = _features(client); bad["admission_grade"] = 250
    r = client.post("/validate", json={"features": bad}).json()
    assert r["status"] == "invalid" and r["summary"] == "INVALID INPUT"


def test_invalid_input_never_produces_a_prediction(client, env):
    bad = _features(client); bad["admission_grade"] = 250
    before = len(client.get("/predictions").json()["predictions"])
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422
    d = r.json()["detail"]
    assert d["message"] == "INVALID INPUT" and "must be within the valid range" in d["issues"][0]["message"]
    assert len(client.get("/predictions").json()["predictions"]) == before   # nothing stored


def test_request_schema_errors(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"student_id": 1, "features": {}}).status_code == 422
    assert client.post("/predict", json={"student_id": 9999}).status_code == 404
    assert client.get("/students/9999").status_code == 404
    assert client.post("/trust/robustness", json={"student_id": 1, "n_perturbations": 1}).status_code == 422


def test_predict_returns_full_trust_report_and_persists(client):
    r = client.post("/predict", json={"student_id": 1})
    assert r.status_code == 200
    j = r.json()
    for k in ("validation", "prediction", "uncertainty", "explanation", "robustness", "calibration",
              "fairness", "trust", "review", "provenance"):
        assert k in j, k
    p = j["prediction"]
    assert sum(p["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert p["predicted_class"] == max(p["probabilities"], key=p["probabilities"].get)
    assert "SYNTHETIC" in j["provenance"] and "will drop out" not in p["statement"].lower()
    assert j["review"]["final_decision_by"] == "human counselor"
    # persisted and retrievable
    got = client.get(f"/predictions/{j['prediction_id']}").json()
    assert got["prediction"]["probabilities"] == p["probabilities"]
    assert any(x["id"] == j["prediction_id"] for x in client.get("/predictions").json()["predictions"])


def test_trust_modules_are_actually_connected_to_the_prediction(client):
    j = client.post("/predict", json={"student_id": 2}).json()
    comps = j["trust"]["prediction_level"]["components"]
    assert comps["uncertainty"]["score"] == pytest.approx(1 - j["uncertainty"]["normalized_entropy"])
    assert comps["robustness"]["score"] == pytest.approx(j["robustness"]["metrics"]["score"])
    assert comps["explainability"]["score"] == (1.0 if j["explanation"]["available"] else 0.0)
    assert comps["data_quality"]["score"] == (1.0 if j["validation"]["status"] == "normal" else 0.5)
    assert j["explanation"]["available"] and j["explanation"]["by_class"]["Dropout"]["contributions"]
    assert j["calibration"]["available"] and j["fairness"]["available"]
    assert j["trust"]["system_level"]["label"] == "Project-defined Trustworthiness Indicator"


def test_prediction_consistency_with_direct_model_and_repeat_calls(client, env):
    feats = _features(client, 3)
    a = client.post("/predict", json={"features": feats}).json()
    b = client.post("/predict", json={"features": feats}).json()
    assert a["prediction"]["probabilities"] == b["prediction"]["probabilities"]
    assert a["robustness"]["metrics"] == b["robustness"]["metrics"]           # seeded => reproducible
    direct = predict_student(feats, load_bundle(env[1] / "models" / BUNDLE_NAME))
    for k in CLASSES:
        assert a["prediction"]["probabilities"][k] == pytest.approx(direct.probabilities[k])


def test_unusual_input_is_flagged_not_rejected(client):
    f = _features(client); f["course"] = 424242
    j = client.post("/predict", json={"features": f}).json()
    assert j["validation"]["status"] == "unusual" and j["review"]["recommended"]
    assert any("unusual" in r.lower() for r in j["review"]["reasons"])
    assert j["trust"]["prediction_level"]["components"]["data_quality"]["score"] == 0.5


def test_explain_and_robustness_endpoints(client):
    e = client.post("/explain", json={"student_id": 1}).json()
    assert e["explanation"]["available"] and "summary" in e["explanation"]
    r = client.post("/trust/robustness", json={"student_id": 1, "n_perturbations": 12}).json()
    assert r["n_perturbations"] == 12 and len(r["perturbations"]) == 12 and "SYNTHETIC" in r["provenance"]
    assert 0 <= r["metrics"]["stability_rate"] <= 1


def test_trust_system_endpoints_and_labels(client):
    s = client.get("/trust/summary").json()
    assert "SYNTHETIC" in s["provenance"] and s["indicator"]["label"] == "Project-defined Trustworthiness Indicator"
    f = client.get("/trust/fairness").json()
    assert "SYNTHETIC" in f["provenance"] and f["overall_status"] in ("fair", "potential_disparity", "requires_review", "not_evaluated")
    assert client.get("/trust/calibration").json()["after"]["ece"] >= 0
    g = client.get("/trust/explainability").json()
    assert len(g["features"]) == 36 and "SYNTHETIC" in g["provenance"]
    perf = client.get("/model/performance").json()
    assert "SYNTHETIC" in perf["provenance"] and set(perf["confusion_matrices"]) == {"Logistic Regression", "Random Forest"}


def test_every_indicator_component_is_evaluated_or_marked_not_evaluated(client):
    comps = client.get("/trust/summary").json()["indicator"]["components"]
    for c in comps.values():
        assert (c["score"] is None) == (c["status"] == "NOT_EVALUATED")


# ---------------- counseling / database ----------------
def test_review_accept_and_override_workflow(client):
    pid = client.post("/predict", json={"student_id": 4}).json()["prediction_id"]
    pred = client.get(f"/predictions/{pid}").json()["prediction"]["predicted_class"]
    acc = client.post("/counseling/review", json={"prediction_id": pid, "counselor_name": "Dr. Demo", "action": "accept"})
    assert acc.status_code == 201 and acc.json()["final_decision"] == pred and acc.json()["model_prediction"] == pred
    other = next(c for c in CLASSES if c != pred)
    ov = client.post("/counseling/review", json={"prediction_id": pid, "counselor_name": "Dr. Demo", "action": "override",
                                                 "final_decision": other, "reason": "Recent family situation resolved", "notes": "Follow up in May"})
    assert ov.status_code == 201 and ov.json()["final_decision"] == other and ov.json()["model_prediction"] == pred
    hist = client.get("/counseling/reviews").json()
    assert {h["id"] for h in hist} >= {acc.json()["id"], ov.json()["id"]}
    assert next(x for x in client.get("/predictions").json()["predictions"] if x["id"] == pid)["reviewed"] is True


def test_override_requires_decision_and_reason(client):
    pid = client.post("/predict", json={"student_id": 5}).json()["prediction_id"]
    base = {"prediction_id": pid, "counselor_name": "Dr. Demo", "action": "override"}
    assert client.post("/counseling/review", json=base).status_code == 422
    assert client.post("/counseling/review", json={**base, "final_decision": "Graduate"}).status_code == 422
    assert client.post("/counseling/review", json={**base, "final_decision": "Nonsense", "reason": "x"}).status_code == 422
    assert client.post("/counseling/review", json={**base, "prediction_id": 99999, "action": "accept"}).status_code == 404


def test_schedule_counseling(client):
    r = client.post("/counseling/schedule", json={"student_id": 1, "scheduled_for": "2030-01-15T10:00:00Z", "notes": "Intro session"})
    assert r.status_code == 201 and r.json()["student_id"] == 1
    assert client.post("/counseling/schedule", json={"scheduled_for": "2030-01-15T10:00:00Z"}).status_code == 422
    assert client.post("/counseling/schedule", json={"student_id": 9999, "scheduled_for": "2030-01-15T10:00:00Z"}).status_code == 404
    assert any(s["id"] == r.json()["id"] for s in client.get("/counseling/schedules").json())


def test_end_to_end_workflow_and_dashboard(client):
    """Enter data -> validated prediction -> trust report -> counselor review -> stored -> visible on dashboard."""
    j = client.post("/predict", json={"features": _features(client, 6), "display_name": "Walk-in Demo"}).json()
    assert j["display_name"] == "Walk-in Demo" and j["student_id"] is None
    rv = client.post("/counseling/review", json={"prediction_id": j["prediction_id"], "counselor_name": "Dr. Demo",
                                                 "action": "accept", "notes": "Reviewed model explanation"})
    assert rv.status_code == 201
    o = client.get("/dashboard/overview").json()
    assert o["counts"]["predictions"] >= 1 and o["counts"]["reviews"] >= 1 and o["counts"]["students"] == 15
    assert sum(o["risk_distribution"].values()) == 15 and "SYNTHETIC" in o["provenance"]
    assert o["model"]["test"]["accuracy"] is not None and o["trust"]["indicator"]["label"]
    students = client.get("/students").json()["students"]
    assert len(students) == 15 and all("quick_prediction" in s for s in students)


def test_service_degrades_gracefully_without_a_model(tmp_path):
    app = create_app(tmp_path / "none", tmp_path / "none", "sqlite:///:memory:", seed=False)
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["model_loaded"] is False
        assert c.post("/predict", json={"features": {}}).status_code == 503


def test_logs_do_not_contain_student_feature_values(client, caplog):
    caplog.set_level("INFO")
    client.post("/predict", json={"student_id": 7})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "admission_grade" not in text and "features" not in text


def test_cors_headers_for_frontend_origin(client):
    r = client.options("/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
