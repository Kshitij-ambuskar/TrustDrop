"""PROJECT-DEFINED Trustworthiness Indicator (NOT an established scientific or industry metric).

Six components, each normalised to [0,1] (1 = best):
  explainability  1 if a SHAP explanation was produced, else 0 (measures availability, not explanation quality)
  uncertainty     1 - normalized predictive entropy
  calibration     1 - min(1, ECE / 0.15)
  fairness        1 - min(1, worst equalized-odds gap / 0.30)
  robustness      stability-based score (see robustness module)
  data_quality    normal = 1.0, valid-but-unusual = 0.5   (invalid input -> no indicator)
Indicator = weighted mean of the EVALUATED components (weights renormalised over those present);
a component that was not evaluated is shown as 'Not evaluated' and never invented.
Level: >= 0.75 High, >= 0.50 Moderate, else Low - and if any evaluated component < 0.40 the level is
capped at Moderate (so one weak dimension cannot hide behind averages).
Limitations: weights and thresholds are subjective; components measure different things on different
scales; the number is a communication aid, not a guarantee of correctness or fairness."""
WEIGHTS = {"explainability": 0.10, "uncertainty": 0.20, "calibration": 0.15,
           "fairness": 0.20, "robustness": 0.20, "data_quality": 0.15}
LEVEL_AT = {"High": 0.75, "Moderate": 0.50}
CAP_BELOW = 0.40
LABEL = "Project-defined Trustworthiness Indicator"
FORMULA = ("score = sum(w_i * c_i) / sum(w_i) over evaluated components; weights = "
           + ", ".join(f"{k} {v:.2f}" for k, v in WEIGHTS.items()))


def calibration_score(ece: float | None):
    return None if ece is None else 1 - min(1.0, ece / 0.15)


def fairness_score(worst_eodds: float | None):
    return None if worst_eodds is None else 1 - min(1.0, worst_eodds / 0.30)


def status_of(score: float | None) -> str:
    return "NOT_EVALUATED" if score is None else "GOOD" if score >= 0.75 else "FAIR" if score >= 0.50 else "POOR"


def compute(components: dict[str, float | None]) -> dict:
    comp = {k: {"score": components.get(k), "weight": w, "status": status_of(components.get(k))} for k, w in WEIGHTS.items()}
    ev = {k: v for k, v in comp.items() if v["score"] is not None}
    if not ev:
        return {"label": LABEL, "score": None, "level": "Not evaluated", "components": comp, "evaluated": 0,
                "total": len(WEIGHTS), "formula": FORMULA, "weakest_component": None, "capped": False}
    tw = sum(v["weight"] for v in ev.values())
    score = sum(v["weight"] * v["score"] for v in ev.values()) / tw
    level = "High" if score >= LEVEL_AT["High"] else "Moderate" if score >= LEVEL_AT["Moderate"] else "Low"
    weakest = min(ev, key=lambda k: ev[k]["score"])
    capped = level == "High" and ev[weakest]["score"] < CAP_BELOW
    return {"label": LABEL, "score": float(score), "level": "Moderate" if capped else level, "components": comp,
            "evaluated": len(ev), "total": len(WEIGHTS), "partial": len(ev) < len(WEIGHTS), "formula": FORMULA,
            "weakest_component": weakest, "capped": capped,
            "disclaimer": "Project-defined indicator for demonstration; not a universal measure of trustworthiness."}
