"""Writes report artifacts (JSON for the frontend, CSV + Markdown for humans)."""
import json
from pathlib import Path

import pandas as pd

from app.ml.evaluation.metrics import METRIC_KEYS

HEADLINE = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model: CV mean +/- std (used for selection) and held-out TEST metrics."""
    rows = []
    for name, r in results.items():
        row = {"model": name}
        for k in METRIC_KEYS:
            row[f"cv_{k}_mean"] = r["cv"]["mean"][k]
            row[f"cv_{k}_std"] = r["cv"]["std"][k]
        for k in METRIC_KEYS:
            row[f"test_{k}"] = r["test"][k]
        rows.append(row)
    return pd.DataFrame(rows)


def summary_markdown(df: pd.DataFrame, best: str, criterion: str, provenance: str,
                     n_splits: int, n_train: int, n_test: int) -> str:
    lines = [f"# Model comparison", "", f"> **{provenance}**", "",
             f"Train rows: {n_train} | Held-out test rows: {n_test} | CV: stratified {n_splits}-fold on train only.",
             "Precision/recall/F1 are macro-averaged; ROC-AUC is one-vs-rest macro.", "",
             "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |  (cross-validation mean ± std)",
             "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        cells = [f"{r[f'cv_{k}_mean']:.3f} ± {r[f'cv_{k}_std']:.3f}" for k in HEADLINE]
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    lines += ["", "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |  (held-out test set)",
              "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['model']} | " + " | ".join(f"{r[f'test_{k}']:.3f}" for k in HEADLINE) + " |")
    lines += ["", f"**Selected model: {best}.** Criterion: {criterion}", ""]
    return "\n".join(lines)
