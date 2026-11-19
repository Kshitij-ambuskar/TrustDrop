"""Pydantic request/response schemas. Features arrive as a free dict on purpose: the data-quality
module reports ALL problems at once (missing, range, type, combos) instead of failing on the first."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ClassName = Literal["Dropout", "Enrolled", "Graduate"]


class StudentInput(BaseModel):
    features: dict[str, Any] | None = None
    student_id: int | None = None
    display_name: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def _one_source(self):
        if (self.features is None) == (self.student_id is None):
            raise ValueError("Provide exactly one of 'features' or 'student_id'.")
        return self


class RobustnessRequest(StudentInput):
    n_perturbations: int = Field(default=30, ge=5, le=200)


class ValidateRequest(BaseModel):
    features: dict[str, Any]


class ReviewCreate(BaseModel):
    prediction_id: int
    counselor_name: str = Field(min_length=1, max_length=80)
    action: Literal["accept", "override"]
    final_decision: ClassName | None = None
    reason: str | None = Field(default=None, max_length=1000)
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _override_needs_details(self):
        if self.action == "override" and (not self.final_decision or not (self.reason or "").strip()):
            raise ValueError("An override requires both 'final_decision' and a non-empty 'reason'.")
        return self


class ScheduleCreate(BaseModel):
    student_id: int | None = None
    prediction_id: int | None = None
    scheduled_for: datetime
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _target(self):
        if self.student_id is None and self.prediction_id is None:
            raise ValueError("Provide 'student_id' or 'prediction_id'.")
        return self


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    prediction_id: int
    counselor_name: str
    action: str
    model_prediction: str
    final_decision: str
    reason: str | None
    notes: str | None
    created_at: datetime


class ScheduleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    student_id: int | None
    prediction_id: int | None
    scheduled_for: datetime
    notes: str | None
    created_at: datetime


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    is_synthetic: bool | None
    provenance: str | None
    database_ok: bool
