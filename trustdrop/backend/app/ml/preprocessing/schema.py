"""Feature schema of the UCI 'Predict Students' Dropout and Academic Success' dataset.

The dataset has 36 predictor columns + 1 target column (37 columns total).
Single source of truth: preprocessing, validation and the UI all read from here.
"""
from dataclasses import dataclass

TARGET = "target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]  # fixed order => class index 0/1/2

# Raw UCI column name -> clean snake_case name used everywhere in our code.
RAW_TO_CLEAN = {
    "Marital status": "marital_status", "Application mode": "application_mode",
    "Application order": "application_order", "Course": "course",
    "Daytime/evening attendance\t": "daytime_evening", "Daytime/evening attendance": "daytime_evening",
    "Previous qualification": "previous_qualification",
    "Previous qualification (grade)": "previous_qualification_grade",
    "Nacionality": "nationality", "Mother's qualification": "mothers_qualification",
    "Father's qualification": "fathers_qualification", "Mother's occupation": "mothers_occupation",
    "Father's occupation": "fathers_occupation", "Admission grade": "admission_grade",
    "Displaced": "displaced", "Educational special needs": "special_needs", "Debtor": "debtor",
    "Tuition fees up to date": "tuition_up_to_date", "Gender": "gender",
    "Scholarship holder": "scholarship_holder", "Age at enrollment": "age_at_enrollment",
    "International": "international",
    "Curricular units 1st sem (credited)": "cu1_credited",
    "Curricular units 1st sem (enrolled)": "cu1_enrolled",
    "Curricular units 1st sem (evaluations)": "cu1_evaluations",
    "Curricular units 1st sem (approved)": "cu1_approved",
    "Curricular units 1st sem (grade)": "cu1_grade",
    "Curricular units 1st sem (without evaluations)": "cu1_without_eval",
    "Curricular units 2nd sem (credited)": "cu2_credited",
    "Curricular units 2nd sem (enrolled)": "cu2_enrolled",
    "Curricular units 2nd sem (evaluations)": "cu2_evaluations",
    "Curricular units 2nd sem (approved)": "cu2_approved",
    "Curricular units 2nd sem (grade)": "cu2_grade",
    "Curricular units 2nd sem (without evaluations)": "cu2_without_eval",
    "Unemployment rate": "unemployment_rate", "Inflation rate": "inflation_rate", "GDP": "gdp",
    "Target": TARGET,
}

# Nominal codes (the integers are labels, NOT quantities) -> one-hot encoded.
NOMINAL = [
    "marital_status", "application_mode", "course", "previous_qualification", "nationality",
    "mothers_qualification", "fathers_qualification", "mothers_occupation", "fathers_occupation",
]
# 0/1 flags. daytime_evening: 1 = daytime, 0 = evening. gender: 1 = male, 0 = female.
BINARY = ["daytime_evening", "displaced", "special_needs", "debtor", "tuition_up_to_date",
          "gender", "scholarship_holder", "international"]
NUMERIC = [
    "application_order", "previous_qualification_grade", "admission_grade", "age_at_enrollment",
    "cu1_credited", "cu1_enrolled", "cu1_evaluations", "cu1_approved", "cu1_grade", "cu1_without_eval",
    "cu2_credited", "cu2_enrolled", "cu2_evaluations", "cu2_approved", "cu2_grade", "cu2_without_eval",
    "unemployment_rate", "inflation_rate", "gdp",
]
FEATURES = NOMINAL + BINARY + NUMERIC  # 9 + 8 + 19 = 36


@dataclass(frozen=True)
class Range:
    """Hard validity range (outside => INVALID input). Used by the data-quality module."""
    lo: float
    hi: float


# Documented UCI ranges: grades are 0-200 (admission/prev. qualification) or 0-20 (unit grades).
VALID_RANGES: dict[str, Range] = {
    "application_order": Range(0, 9), "previous_qualification_grade": Range(0, 200),
    "admission_grade": Range(0, 200), "age_at_enrollment": Range(16, 100),
    "cu1_grade": Range(0, 20), "cu2_grade": Range(0, 20),
    "unemployment_rate": Range(0, 100), "inflation_rate": Range(-10, 50),
    **{f"cu{s}_{k}": Range(0, 60) for s in (1, 2)
       for k in ("credited", "enrolled", "evaluations", "approved", "without_eval")},
}
assert len(FEATURES) == 36 and len(set(FEATURES)) == 36
