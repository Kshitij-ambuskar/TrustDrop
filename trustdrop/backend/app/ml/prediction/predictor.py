"""Inference on RAW student values using the saved bundle (preprocessing is inside the model)."""
import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from app.config import settings
from app.ml.preprocessing.schema import CLASSES, FEATURES

log = logging.getLogger(__name__)
BUNDLE_NAME = "best_model.joblib"


@dataclass
class PredictionResult:
    predicted_class: str
    probabilities: dict[str, float]   # class label -> probability (sums to 1)
    confidence: float                 # = probability of the predicted class (NOT a certainty guarantee)
    classes: list[str]
    model_name: str
    calibrated: bool
    provenance: str

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=2)
def _load(path: str) -> dict:
    return joblib.load(path)


def load_bundle(path: Path | None = None) -> dict:
    path = path or settings.models_dir / BUNDLE_NAME
    if not Path(path).exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python scripts/train.py")
    return _load(str(path))


def to_frame(students: dict | list[dict]) -> pd.DataFrame:
    """Raw dict(s) -> DataFrame with the exact training columns. Missing/unknown fields are
    rejected here; richer validation (ranges, OOD) belongs to the Stage-3 data-quality module."""
    rows = [students] if isinstance(students, dict) else students
    df = pd.DataFrame(rows)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    return df[FEATURES]


def predict_students(students: list[dict], bundle: dict | None = None) -> list[PredictionResult]:
    bundle = bundle or load_bundle()
    proba = bundle["model"].predict_proba(to_frame(students))
    out = []
    for p in proba:
        probs = {c: float(v) for c, v in zip(bundle["classes"], p)}
        top = max(probs, key=probs.get)
        out.append(PredictionResult(top, probs, probs[top], list(bundle["classes"]),
                                    bundle["model_name"], bundle["calibration"]["applied"],
                                    bundle["provenance"]["label"]))
    return out


def predict_student(student_data: dict, bundle: dict | None = None) -> PredictionResult:
    """Predict for ONE student given raw feature values (dict keyed by schema names)."""
    return predict_students([student_data], bundle)[0]
