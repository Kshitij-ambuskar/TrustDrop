"""Project-defined risk bands from the (calibrated) probability of Dropout, plus safe wording."""
HIGH_AT, MEDIUM_AT = 0.50, 0.30


def risk_level(p_dropout: float) -> str:
    return "HIGH" if p_dropout >= HIGH_AT else "MEDIUM" if p_dropout >= MEDIUM_AT else "LOW"


def safe_statement(p_dropout: float, level: str) -> str:
    """Never says a student 'will' drop out - always a probabilistic estimate for human review."""
    word = {"HIGH": "a high", "MEDIUM": "a moderate", "LOW": "a low"}[level]
    return (f"The model estimates {word} probability of dropout ({p_dropout:.0%}) based on the available data. "
            "This is a statistical estimate to support a counselor, not a determination about this student.")
