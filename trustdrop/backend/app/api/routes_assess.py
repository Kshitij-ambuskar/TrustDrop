import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_engine
from app.schemas.api import RobustnessRequest, StudentInput, ValidateRequest
from app.services import repository as repo
from app.services.assessment import TrustEngine

router = APIRouter(tags=["assessment"])
log = logging.getLogger(__name__)


def _resolve(body: StudentInput, db: Session) -> tuple[dict, int | None, str]:
    if body.student_id is not None:
        s = repo.get_student(db, body.student_id)
        if s is None:
            raise HTTPException(404, detail=f"Student {body.student_id} not found.")
        return s.features, s.id, s.display_name
    return body.features, None, body.display_name or "Ad-hoc student"


def _invalid(report: dict):
    return HTTPException(422, detail={"status": "invalid", "message": "INVALID INPUT", "issues": report["validation"]["issues"]})


@router.post("/validate")
def validate(body: ValidateRequest, engine: TrustEngine = Depends(get_engine)):
    """Data-quality check only (INVALID / UNUSUAL / NORMAL). The model is not run."""
    v = engine.validate(body.features)
    return {k: v[k] for k in ("status", "summary", "issues", "ood")}


@router.post("/predict")
def predict(body: StudentInput, engine: TrustEngine = Depends(get_engine), db: Session = Depends(get_db)):
    """Full pipeline: validate -> predict -> uncertainty -> SHAP -> robustness -> trust indicator. Persists the result."""
    feats, sid, name = _resolve(body, db)
    report = engine.assess(feats, n_perturbations=engine_default_n())
    if report["status"] == "invalid":
        raise _invalid(report)
    rec = repo.save_prediction(db, report, sid, name)
    log.info("prediction stored id=%s class=%s", rec.id, rec.predicted_class)  # no student data in logs
    return {"prediction_id": rec.id, "student_id": sid, "display_name": name, **report}


@router.post("/explain")
def explain(body: StudentInput, engine: TrustEngine = Depends(get_engine), db: Session = Depends(get_db)):
    feats, _, _ = _resolve(body, db)
    r = engine.assess(feats, n_perturbations=5)
    if r["status"] == "invalid":
        raise _invalid(r)
    return {"prediction": r["prediction"], "explanation": r["explanation"], "provenance": r["provenance"]}


@router.post("/trust/robustness")
def robustness(body: RobustnessRequest, engine: TrustEngine = Depends(get_engine), db: Session = Depends(get_db)):
    from app.ml.robustness.robustness import run_robustness
    feats, _, _ = _resolve(body, db)
    val = engine.validate(feats)
    if val["status"] == "invalid":
        raise _invalid({"validation": val})
    return {"provenance": engine.label, **run_robustness(val["cleaned"], engine.predict_proba,
                                                         engine.trust["reference"], n=body.n_perturbations)}


def engine_default_n() -> int:
    from app.config import settings
    return settings.n_perturbations
