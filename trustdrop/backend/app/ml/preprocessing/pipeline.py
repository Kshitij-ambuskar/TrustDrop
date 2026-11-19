"""Leak-free preprocessing: every step that LEARNS from data lives inside an sklearn
ColumnTransformer, so it is fitted on training folds only (never on test data)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import settings
from app.ml.preprocessing.schema import BINARY, CLASSES, FEATURES, NOMINAL, NUMERIC, TARGET


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Numeric: median-impute (+scale). Nominal: mode-impute + one-hot. Binary: mode-impute.
    Tree models don't need scaling; LR / ANN do (scale_numeric=True)."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC),
            ("nom", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              # unseen category at predict time -> all zeros, no crash
                              ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                                       sparse_output=False))]), NOMINAL),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
        ],
        verbose_feature_names_out=False,
    )


def encode_target(y: pd.Series) -> pd.Series:
    """Dropout/Enrolled/Graduate -> 0/1/2 using the fixed CLASSES order."""
    return y.map({c: i for i, c in enumerate(CLASSES)}).astype(int)


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split with a fixed seed. Stratify keeps class ratios equal in both."""
    X, y = df[FEATURES], encode_target(df[TARGET])
    return train_test_split(X, y, test_size=settings.test_size,
                            stratify=y, random_state=settings.random_seed)
