"""SQLite tables (SQLAlchemy 2.0). Student names are SYNTHETIC demo names only."""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Student(Base):
    __tablename__ = "students"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_name: Mapped[str] = mapped_column(String(80))
    features: Mapped[dict] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(120), default="manual")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PredictionRecord(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int | None] = mapped_column(ForeignKey("students.id"), nullable=True)
    display_name: Mapped[str] = mapped_column(String(80), default="Ad-hoc student")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    predicted_class: Mapped[str] = mapped_column(String(20))
    probabilities: Mapped[dict] = mapped_column(JSON)
    risk_level: Mapped[str] = mapped_column(String(10))
    uncertainty_level: Mapped[str] = mapped_column(String(10))
    trust_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    trust_level: Mapped[str] = mapped_column(String(20))
    data_quality_status: Mapped[str] = mapped_column(String(10))
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    model_name: Mapped[str] = mapped_column(String(40))
    provenance: Mapped[str] = mapped_column(String(200))
    report: Mapped[dict] = mapped_column(JSON)          # full trust report returned to the UI


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))
    counselor_name: Mapped[str] = mapped_column(String(80))
    action: Mapped[str] = mapped_column(String(10))      # accept | override
    model_prediction: Mapped[str] = mapped_column(String(20))
    final_decision: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Schedule(Base):
    __tablename__ = "schedules"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int | None] = mapped_column(ForeignKey("students.id"), nullable=True)
    prediction_id: Mapped[int | None] = mapped_column(ForeignKey("predictions.id"), nullable=True)
    scheduled_for: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
