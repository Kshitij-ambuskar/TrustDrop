import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_engine
from app.ml.prediction.risk import risk_level
from app.ml.preprocessing.schema import CLASSES, FEATURES
from app.services import repository as repo
from app.services.assessment import TrustEngine

router = APIRouter(tags=["students"])


@router.get("/students")
def students(eng: TrustEngine = Depends(get_engine), db: Session = Depends(get_db)):
    rows = repo.list_students(db)
    quick = {}
    if rows:
        P = eng.predict_proba(pd.DataFrame([s.features for s in rows])[FEATURES])
        quick = {s.id: P[i] for i, s in enumerate(rows)}
    return {"provenance": eng.label, "students": [
        {"id": s.id, "display_name": s.display_name, "source": s.source,
         "quick_prediction": {"predicted_class": CLASSES[int(quick[s.id].argmax())],
                              "p_dropout": float(quick[s.id][0]), "risk_level": risk_level(float(quick[s.id][0]))}}
        for s in rows]}


@router.get("/students/{sid}")
def student(sid: int, db: Session = Depends(get_db)):
    s = repo.get_student(db, sid)
    if s is None:
        raise HTTPException(404, detail=f"Student {sid} not found.")
    return {"id": s.id, "display_name": s.display_name, "source": s.source, "features": s.features}


@router.get("/predictions")
def predictions(db: Session = Depends(get_db)):
    reviewed = repo.reviewed_prediction_ids(db)
    return {"predictions": [repo.prediction_summary(r, r.id in reviewed) for r in repo.list_predictions(db)]}


@router.get("/predictions/{pid}")
def prediction(pid: int, db: Session = Depends(get_db)):
    r = repo.get_prediction(db, pid)
    if r is None:
        raise HTTPException(404, detail=f"Prediction {pid} not found.")
    return {"prediction_id": r.id, "student_id": r.student_id, "display_name": r.display_name, **r.report}
