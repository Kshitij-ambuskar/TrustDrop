import numpy as np
import pandas as pd
import pytest

from app.ml.preprocessing.data_loader import DatasetNotFoundError, clean_columns, load_dataset
from app.ml.preprocessing.pipeline import build_preprocessor, encode_target, split_data
from app.ml.preprocessing.schema import CLASSES, FEATURES, TARGET
from tests.scripts_import import make_synth


def test_schema_has_36_unique_features():
    assert len(FEATURES) == 36 and len(set(FEATURES)) == 36


def test_target_encoding_order():
    y = encode_target(pd.Series(["Graduate", "Dropout", "Enrolled"]))
    assert y.tolist() == [2, 0, 1]


def test_clean_columns_renames_raw_uci_names_and_rejects_bad_labels():
    df = make_synth(50)
    raw = df.rename(columns={"marital_status": "Marital status", "gdp": "GDP", TARGET: "Target"})
    assert "marital_status" in clean_columns(raw).columns
    bad = df.copy(); bad[TARGET] = "Unknown"
    with pytest.raises(ValueError):
        clean_columns(bad)


def test_split_is_stratified_and_reproducible():
    df = make_synth(600)
    Xtr, Xte, ytr, yte = split_data(df)
    assert len(Xte) == pytest.approx(0.2 * len(df), abs=2)
    assert abs(ytr.mean() - yte.mean()) < 0.05
    Xtr2, *_ = split_data(df)
    assert Xtr.index.equals(Xtr2.index)  # same seed => same split


def test_preprocessor_is_fit_on_train_only_and_handles_missing_and_unseen():
    df = make_synth(600)
    Xtr, Xte, _, _ = split_data(df)
    pre = build_preprocessor().fit(Xtr)
    Xte = Xte.copy()
    Xte.iloc[0, Xte.columns.get_loc("admission_grade")] = np.nan   # missing numeric
    Xte.iloc[1, Xte.columns.get_loc("course")] = 99999            # unseen category
    out = pre.transform(Xte)
    assert not np.isnan(out).any()
    # train statistics, not test statistics, were used for scaling:
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    assert scaler.mean_[2] == pytest.approx(Xtr["admission_grade"].mean(), rel=1e-6)


def test_loader_raises_helpful_error_without_data(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    with pytest.raises(DatasetNotFoundError, match="archive.ics.uci.edu"):
        load_dataset()


def test_real_uci_raw_format_loads_and_trains_without_code_changes(tmp_path, monkeypatch):
    """Simulates data/data.csv exactly as UCI ships it: ';' separator, original column names
    (including the trailing TAB in 'Daytime/evening attendance'), 'Target' header."""
    from app.config import settings
    from app.ml.preprocessing.schema import RAW_TO_CLEAN
    from app.ml.training.train import run_training
    clean = make_synth(400)
    inverse = {}
    for raw, c in RAW_TO_CLEAN.items():
        inverse.setdefault(c, raw)
    inverse["daytime_evening"] = "Daytime/evening attendance\t"
    clean.rename(columns=inverse).to_csv(tmp_path / "data.csv", sep=";", index=False)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    data = load_dataset()                      # NOTE: allow_synthetic NOT set
    assert not data.is_synthetic and len(data.df) == 400
    out = run_training(data.df, data.is_synthetic, tmp_path / "m", tmp_path / "r",
                       n_splits=3, model_names=["Logistic Regression"])
    assert "Real UCI" in out["provenance"]
