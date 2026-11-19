"""Dataset loading. Prefers the REAL UCI file; falls back to a flagged synthetic dev file."""
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.config import settings
from app.ml.preprocessing.schema import CLASSES, FEATURES, RAW_TO_CLEAN, TARGET

log = logging.getLogger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when no dataset file exists; message explains how to fix it."""


@dataclass
class LoadedData:
    df: pd.DataFrame
    source: Path
    is_synthetic: bool  # True => results are NOT real and must never be reported as such


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw UCI columns, validate the schema and the target labels."""
    df = df.rename(columns=lambda c: RAW_TO_CLEAN.get(c, RAW_TO_CLEAN.get(str(c).strip(), c)))
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    bad = set(df[TARGET].dropna().unique()) - set(CLASSES)
    if bad:
        raise ValueError(f"Unexpected target labels: {bad}; expected {CLASSES}")
    return df[FEATURES + [TARGET]].copy()


def _read_csv(path: Path) -> pd.DataFrame:
    """UCI ships data.csv semicolon-separated. We try ';' then ',' explicitly: delimiter
    *sniffing* is unsafe because one real header contains a stray TAB character."""
    for sep in (";", ","):
        df = pd.read_csv(path, sep=sep)
        if df.shape[1] > 1:
            return df
    raise ValueError(f"Could not parse {path} as ';' or ',' separated CSV")


def load_dataset(allow_synthetic: bool = False) -> LoadedData:
    real = settings.data_dir / settings.dataset_filename
    synth = settings.data_dir / settings.synthetic_filename
    if real.exists():
        df = _read_csv(real)
        log.info("Loaded real dataset: %s rows", len(df))
        return LoadedData(clean_columns(df), real, False)
    if allow_synthetic and synth.exists():
        log.warning("REAL DATASET NOT FOUND - using SYNTHETIC dev data. Results are not real.")
        return LoadedData(clean_columns(pd.read_csv(synth)), synth, True)
    raise DatasetNotFoundError(
        f"Dataset not found at {real}. Download 'data.csv' from "
        "https://archive.ics.uci.edu/dataset/697 and place it in the data/ directory."
    )
