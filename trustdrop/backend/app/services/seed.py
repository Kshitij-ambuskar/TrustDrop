"""Seed demo students (SYNTHETIC NAMES) from held-out test rows of the dataset the model was trained on.
Never silently falls back: synthetic rows are used only if the loaded model itself is synthetic."""
import logging

from sqlalchemy.orm import Session

from app.database.models import Student
from app.ml.preprocessing.data_loader import DatasetNotFoundError, load_dataset
from app.ml.preprocessing.pipeline import split_data
from app.ml.preprocessing.schema import FEATURES

log = logging.getLogger(__name__)
FIRST = ["Alex", "Sam", "Jordan", "Riley", "Casey", "Taylor", "Morgan", "Jamie", "Robin", "Avery"]
LAST = ["Rivera", "Okafor", "Silva", "Novak", "Tanaka", "Haddad", "Larsen", "Mehta", "Costa", "Ibarra"]


def seed_demo_students(db: Session, is_synthetic_model: bool, n: int = 40) -> int:
    if db.query(Student).count() > 0:
        return 0
    try:
        df = load_dataset(allow_synthetic=is_synthetic_model).df
    except DatasetNotFoundError:
        log.warning("No dataset available to seed demo students.")
        return 0
    _, X_test, _, _ = split_data(df)
    for i, (_, row) in enumerate(X_test.head(n).iterrows(), start=1):
        feats = {f: (int(row[f]) if float(row[f]).is_integer() else float(row[f])) for f in FEATURES}
        db.add(Student(display_name=f"{FIRST[i % 10]} {LAST[(i * 7) % 10]} (demo #{i:03d})", features=feats,
                       source="demo: held-out test row"))
    db.commit()
    return min(n, len(X_test))
