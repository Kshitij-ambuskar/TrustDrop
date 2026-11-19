from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_engine
from app.ml.preprocessing.feature_meta import feature_meta
from app.schemas.api import HealthOut
from app.services import repository as repo
from app.services.assessment import TrustEngine

router = APIRouter()


@router.get("/health", response_model=HealthOut, tags=["system"])
def health(request: Request, db: Session = Depends(get_db)):
    eng = request.app.state.engine
    try:
        db.execute(text("SELECT 1")); db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    return HealthOut(status="ok" if eng and db_ok else "degraded", model_loaded=eng is not None,
                     is_synthetic=eng.is_synthetic if eng else None, provenance=eng.label if eng else None, database_ok=db_ok)


@router.get("/model/info", tags=["model"])
def model_info(eng: TrustEngine = Depends(get_engine)):
    return eng.model_info()


@router.get("/model/performance", tags=["model"])
def performance(eng: TrustEngine = Depends(get_engine)):
    perf = eng.reports.model_performance()
    if perf is None:
        raise HTTPException(404, detail="No model comparison report found. Run training.")
    return perf


@router.get("/schema/features", tags=["model"])
def schema(eng: TrustEngine = Depends(get_engine)):
    return {"features": feature_meta(eng.trust["reference"] if eng.trust else None)}


@router.get("/trust/summary", tags=["trust"])
def trust_summary(eng: TrustEngine = Depends(get_engine)):
    t = eng.trust
    if not t:
        raise HTTPException(404, detail="No trust artifacts. Retrain.")
    return {"provenance": eng.label, "indicator": t["indicator"], "uncertainty": t["uncertainty"],
            "robustness": t["robustness"], "data_quality": t["data_quality"], "calibration": eng.calibration_info(),
            "fairness": eng.fairness_summary(), "errors": t["errors"]}


@router.get("/trust/fairness", tags=["trust"])
def trust_fairness(eng: TrustEngine = Depends(get_engine)):
    if not eng.trust or not eng.trust.get("fairness"):
        raise HTTPException(404, detail="Fairness audit not available.")
    return eng.trust["fairness"]


@router.get("/trust/calibration", tags=["trust"])
def trust_calibration(eng: TrustEngine = Depends(get_engine)):
    c = eng.calibration_info()
    if not c["available"]:
        raise HTTPException(404, detail="Calibration report not available.")
    return c


@router.get("/trust/explainability", tags=["trust"])
def trust_explainability(eng: TrustEngine = Depends(get_engine)):
    g = eng.global_explanation()
    if g is None:
        raise HTTPException(404, detail="Global explanation not available (SHAP failed or not built).")
    return g


@router.get("/dashboard/overview", tags=["dashboard"])
def overview(eng: TrustEngine = Depends(get_engine), db: Session = Depends(get_db)):
    import pandas as pd
    from app.ml.prediction.risk import risk_level
    from app.ml.preprocessing.schema import FEATURES
    students = repo.list_students(db)
    dist = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    if students:
        P = eng.predict_proba(pd.DataFrame([s.features for s in students])[FEATURES])
        for p in P:
            dist[risk_level(float(p[0]))] += 1          # column 0 = Dropout
    perf = eng.reports.model_performance()
    best = perf["results"][perf["selected_model"]] if perf else None
    reviewed = repo.reviewed_prediction_ids(db)
    recent = repo.list_predictions(db, 8)
    return {"provenance": eng.label, "is_synthetic": eng.is_synthetic, "counts": repo.counts(db), "risk_distribution": dist,
            "class_distribution": (eng.reports.read("dataset_summary.json") or {}).get("class_counts"),
            "model": {"name": eng.bundle["model_name"], "test": best["test"] if best else None,
                      "cv": best["cv"]["mean"] if best else None},
            "trust": {"indicator": eng.trust["indicator"] if eng.trust else None, "fairness": eng.fairness_summary(),
                      "calibration": eng.calibration_info()},
            "recent_predictions": [repo.prediction_summary(r, r.id in reviewed) for r in recent],
            "pending_reviews": sum(1 for r in recent if r.needs_review and r.id not in reviewed)}
