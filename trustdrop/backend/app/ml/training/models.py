"""The four candidate models. Each is a full sklearn Pipeline: raw student columns -> prediction.
Keeping preprocessing INSIDE the pipeline means it is re-fitted on training folds only (no leakage)
and inference accepts raw feature values."""
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from app.ml.preprocessing.pipeline import build_preprocessor

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "ANN"]


class BalancedWeightClassifier(ClassifierMixin, BaseEstimator):
    """Class-imbalance handling for models without a `class_weight` option (XGBoost, MLP).

    On every fit() it computes 'balanced' sample weights from the labels it is given
    (rare class => larger weight) and passes them to the wrapped model. Because this happens
    inside fit(), weights are computed from training folds only. Works with CalibratedClassifierCV.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def get_model_specs(seed: int = 42) -> dict[str, Pipeline]:
    """Unfitted pipelines, one per model. Linear/NN models need scaled inputs; trees do not."""
    return {
        "Logistic Regression": Pipeline([
            ("pre", build_preprocessor(scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "Random Forest": Pipeline([
            ("pre", build_preprocessor(scale_numeric=False)),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=seed))]),
        "XGBoost": Pipeline([
            ("pre", build_preprocessor(scale_numeric=False)),
            ("clf", BalancedWeightClassifier(XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                colsample_bytree=0.8, objective="multi:softprob", tree_method="hist",
                n_jobs=-1, random_state=seed)))]),
        "ANN": Pipeline([
            ("pre", build_preprocessor(scale_numeric=True)),
            ("clf", BalancedWeightClassifier(MLPClassifier(
                hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=seed)))]),
    }
