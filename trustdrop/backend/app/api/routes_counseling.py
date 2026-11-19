import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.api import ReviewCreate, ReviewOut, ScheduleCreate, ScheduleOut
from app.services import repository as repo

router = APIRouter(prefix="/counseling", tags=["counseling"])
log = logging.getLogger(__name__)


@router.post("/review", response_model=ReviewOut, status_code=201)
def review(body: ReviewCreate, db: Session = Depends(get_db)):
    """Human-in-the-loop: the counselor accepts or overrides the model. The final decision is theirs."""
    pred = repo.get_prediction(db, body.prediction_id)
    if pred is None:
        raise HTTPException(404, detail=f"Prediction {body.prediction_id} not found.")
    rev = repo.create_review(db, body, pred)
    log.info("review stored id=%s action=%s", rev.id, rev.action)
    return rev


@router.get("/reviews", response_model=list[ReviewOut])
def reviews(db: Session = Depends(get_db)):
    return repo.list_reviews(db)


@router.post("/schedule", response_model=ScheduleOut, status_code=201)
def schedule(body: ScheduleCreate, db: Session = Depends(get_db)):
    if body.student_id is not None and repo.get_student(db, body.student_id) is None:
        raise HTTPException(404, detail=f"Student {body.student_id} not found.")
    if body.prediction_id is not None and repo.get_prediction(db, body.prediction_id) is None:
        raise HTTPException(404, detail=f"Prediction {body.prediction_id} not found.")
    return repo.create_schedule(db, body)


@router.get("/schedules", response_model=list[ScheduleOut])
def schedules(db: Session = Depends(get_db)):
    return repo.list_schedules(db)
