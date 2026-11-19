"""Database access functions (no ML here)."""
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.models import PredictionRecord, Review, Schedule, Student
from app.schemas.api import ReviewCreate, ScheduleCreate


def list_students(db: Session) -> list[Student]:
    return list(db.scalars(select(Student).order_by(Student.id)))


def get_student(db: Session, sid: int) -> Student | None:
    return db.get(Student, sid)


def save_prediction(db: Session, report: dict, student_id: int | None, name: str) -> PredictionRecord:
    t, p = report["trust"]["prediction_level"], report["prediction"]
    rec = PredictionRecord(
        student_id=student_id, display_name=name, predicted_class=p["predicted_class"],
        probabilities=p["probabilities"], risk_level=p["risk_level"], uncertainty_level=report["uncertainty"]["level"],
        trust_score=t["score"], trust_level=t["level"], data_quality_status=report["validation"]["status"],
        needs_review=report["review"]["recommended"], model_name=p["model_name"], provenance=report["provenance"],
        report=report)
    db.add(rec); db.commit(); db.refresh(rec)
    return rec


def get_prediction(db: Session, pid: int) -> PredictionRecord | None:
    return db.get(PredictionRecord, pid)


def list_predictions(db: Session, limit: int = 50) -> list[PredictionRecord]:
    return list(db.scalars(select(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit)))


def prediction_summary(r: PredictionRecord, reviewed: bool = False) -> dict:
    return {"id": r.id, "student_id": r.student_id, "display_name": r.display_name, "created_at": r.created_at,
            "predicted_class": r.predicted_class, "probabilities": r.probabilities, "risk_level": r.risk_level,
            "uncertainty_level": r.uncertainty_level, "trust_score": r.trust_score, "trust_level": r.trust_level,
            "data_quality_status": r.data_quality_status, "needs_review": r.needs_review, "provenance": r.provenance,
            "reviewed": reviewed}


def reviewed_prediction_ids(db: Session) -> set[int]:
    return set(db.scalars(select(Review.prediction_id).distinct()))


def create_review(db: Session, body: ReviewCreate, pred: PredictionRecord) -> Review:
    final = pred.predicted_class if body.action == "accept" else body.final_decision
    rev = Review(prediction_id=pred.id, counselor_name=body.counselor_name.strip(), action=body.action,
                 model_prediction=pred.predicted_class, final_decision=final, reason=body.reason, notes=body.notes)
    db.add(rev); db.commit(); db.refresh(rev)
    return rev


def list_reviews(db: Session, limit: int = 100) -> list[Review]:
    return list(db.scalars(select(Review).order_by(Review.id.desc()).limit(limit)))


def create_schedule(db: Session, body: ScheduleCreate) -> Schedule:
    s = Schedule(student_id=body.student_id, prediction_id=body.prediction_id,
                 scheduled_for=body.scheduled_for, notes=body.notes)
    db.add(s); db.commit(); db.refresh(s)
    return s


def list_schedules(db: Session) -> list[Schedule]:
    return list(db.scalars(select(Schedule).order_by(Schedule.scheduled_for)))


def counts(db: Session) -> dict:
    n = lambda m: db.scalar(select(func.count()).select_from(m))  # noqa: E731
    return {"students": n(Student), "predictions": n(PredictionRecord), "reviews": n(Review), "schedules": n(Schedule)}
