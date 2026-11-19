"""Generate a SYNTHETIC stand-in dataset (same columns as UCI) ONLY so the code can be run
and tested when the real data.csv is unavailable. It is random data with a weak planted
signal. NEVER report model results trained on it as real findings."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.config import settings  # noqa: E402
from app.ml.preprocessing.schema import CLASSES, FEATURES, TARGET  # noqa: E402


def make(n: int = 4424, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    d = {
        "marital_status": r.choice([1, 2, 3, 4], n, p=[.88, .08, .02, .02]),
        "application_mode": r.choice([1, 2, 5, 15, 17, 39, 42], n),
        "course": r.choice([33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991], n),
        "previous_qualification": r.choice([1, 2, 3, 9, 12, 19, 39], n, p=[.84, .02, .02, .02, .02, .06, .02]),
        "nationality": r.choice([1, 2, 6, 11, 41], n, p=[.97, .01, .01, .005, .005]),
        "mothers_qualification": r.choice([1, 2, 3, 19, 34, 37], n),
        "fathers_qualification": r.choice([1, 2, 3, 19, 34, 37], n),
        "mothers_occupation": r.choice([0, 4, 5, 7, 9, 10], n),
        "fathers_occupation": r.choice([0, 4, 5, 7, 9, 10], n),
        "daytime_evening": r.choice([0, 1], n, p=[.1, .9]),
        "displaced": r.choice([0, 1], n), "special_needs": r.choice([0, 1], n, p=[.99, .01]),
        "debtor": r.choice([0, 1], n, p=[.88, .12]), "tuition_up_to_date": r.choice([0, 1], n, p=[.12, .88]),
        "gender": r.choice([0, 1], n, p=[.65, .35]), "scholarship_holder": r.choice([0, 1], n, p=[.75, .25]),
        "international": r.choice([0, 1], n, p=[.98, .02]),
        "application_order": r.integers(0, 7, n),
        "previous_qualification_grade": np.clip(r.normal(132, 13, n), 95, 190).round(1),
        "admission_grade": np.clip(r.normal(127, 14, n), 95, 190).round(1),
        "age_at_enrollment": np.clip(r.gamma(4, 5, n) + 17, 17, 70).astype(int),
        "unemployment_rate": r.choice([7.6, 9.4, 10.8, 12.4, 13.9, 15.5, 16.2], n),
        "inflation_rate": r.choice([-0.8, 0.3, 0.6, 1.4, 2.6, 3.7], n),
        "gdp": r.choice([-4.06, -3.12, -1.7, 0.32, 0.79, 1.74, 2.02], n),
    }
    for s in (1, 2):
        enr = r.integers(0, 9, n) + 3
        d[f"cu{s}_credited"] = r.choice([0, 0, 0, 0, 3, 6], n)
        d[f"cu{s}_enrolled"] = enr
        d[f"cu{s}_evaluations"] = enr + r.integers(0, 5, n)
        d[f"cu{s}_approved"] = np.minimum(enr, r.binomial(enr, 0.65))
        d[f"cu{s}_grade"] = np.where(d[f"cu{s}_approved"] > 0, np.clip(r.normal(12, 2, n), 0, 20), 0).round(2)
        d[f"cu{s}_without_eval"] = r.choice([0, 0, 0, 1, 2], n)
    df = pd.DataFrame(d)
    # Weak planted signal: fewer approved units / unpaid fees -> more dropout.
    score = (-(df.cu1_approved + df.cu2_approved) * 0.35 + df.debtor * 0.9
             - df.tuition_up_to_date * 0.9 + r.normal(0, 1.2, n))
    q = np.quantile(score, [0.32, 0.50])  # ~32% dropout, ~18% enrolled, ~50% graduate
    # high score = high risk
    hi, mid = np.quantile(score, 0.68), np.quantile(score, 0.50)
    df[TARGET] = np.where(score >= hi, CLASSES[0], np.where(score >= mid, CLASSES[1], CLASSES[2]))
    return df[FEATURES + [TARGET]]


if __name__ == "__main__":
    out = settings.data_dir / settings.synthetic_filename
    out.parent.mkdir(exist_ok=True)
    make().to_csv(out, index=False)
    print(f"SYNTHETIC dev data written to {out} (NOT real data)")
