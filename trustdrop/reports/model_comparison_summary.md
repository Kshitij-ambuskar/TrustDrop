# Model comparison

> **DEVELOPMENT / SYNTHETIC DATA - pipeline validation only, NOT real project results**

Train rows: 3539 | Held-out test rows: 885 | CV: stratified 5-fold on train only.
Precision/recall/F1 are macro-averaged; ROC-AUC is one-vs-rest macro.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |  (cross-validation mean ± std)
|---|---|---|---|---|---|
| Logistic Regression | 0.580 ± 0.024 | 0.530 ± 0.022 | 0.525 ± 0.026 | 0.524 ± 0.024 | 0.730 ± 0.019 |
| Random Forest | 0.632 ± 0.014 | 0.417 ± 0.008 | 0.494 ± 0.013 | 0.450 ± 0.011 | 0.726 ± 0.010 |
| XGBoost | 0.605 ± 0.019 | 0.525 ± 0.015 | 0.530 ± 0.016 | 0.526 ± 0.014 | 0.730 ± 0.013 |
| ANN | 0.557 ± 0.020 | 0.506 ± 0.015 | 0.501 ± 0.016 | 0.500 ± 0.015 | 0.707 ± 0.015 |

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |  (held-out test set)
|---|---|---|---|---|---|
| Logistic Regression | 0.577 | 0.538 | 0.533 | 0.530 | 0.732 |
| Random Forest | 0.623 | 0.414 | 0.485 | 0.443 | 0.718 |
| XGBoost | 0.590 | 0.514 | 0.518 | 0.515 | 0.726 |
| ANN | 0.554 | 0.514 | 0.503 | 0.504 | 0.712 |

**Selected model: XGBoost.** Criterion: highest mean cross-validated macro-F1 on the TRAINING split (ties broken by mean one-vs-rest macro ROC-AUC). The test set is never used for selection.
