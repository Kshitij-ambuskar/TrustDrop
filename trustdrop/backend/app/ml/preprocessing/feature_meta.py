"""Human-readable metadata for the 36 features (labels, UI groups, kinds). Reuses schema.py."""
from app.ml.preprocessing.schema import BINARY, NOMINAL, NUMERIC, RAW_TO_CLEAN, VALID_RANGES

LABELS: dict[str, str] = {}
for _raw, _clean in RAW_TO_CLEAN.items():
    LABELS.setdefault(_clean, _raw.strip())

GROUPS = {
    "Demographics": ["marital_status", "nationality", "gender", "age_at_enrollment", "international",
                     "displaced", "special_needs"],
    "Application & background": ["application_mode", "application_order", "course", "daytime_evening",
                                 "previous_qualification", "previous_qualification_grade", "admission_grade",
                                 "mothers_qualification", "fathers_qualification", "mothers_occupation",
                                 "fathers_occupation"],
    "Financial": ["debtor", "tuition_up_to_date", "scholarship_holder"],
    "1st semester": [f for f in NUMERIC if f.startswith("cu1_")],
    "2nd semester": [f for f in NUMERIC if f.startswith("cu2_")],
    "Economic context": ["unemployment_rate", "inflation_rate", "gdp"],
}
BINARY_OPTIONS = {"gender": {0: "Female", 1: "Male"}, "daytime_evening": {0: "Evening", 1: "Daytime"}}


def feature_meta(reference: dict | None = None) -> list[dict]:
    """Schema for the UI form: one dict per feature, in group order."""
    out = []
    for group, feats in GROUPS.items():
        for f in feats:
            kind = "nominal" if f in NOMINAL else "binary" if f in BINARY else "numeric"
            m = {"name": f, "label": LABELS[f], "group": group, "kind": kind}
            if kind == "binary":
                m["options"] = [{"value": k, "label": v} for k, v in
                                BINARY_OPTIONS.get(f, {0: "No", 1: "Yes"}).items()]
            if kind == "nominal" and reference:
                m["options"] = [{"value": int(c), "label": f"Code {int(c)}"} for c in reference["categories"][f]]
            if f in VALID_RANGES:
                m["min"], m["max"] = VALID_RANGES[f].lo, VALID_RANGES[f].hi
            out.append(m)
    return out
