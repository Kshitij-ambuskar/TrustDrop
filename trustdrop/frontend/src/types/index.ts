export type Cls = 'Dropout' | 'Enrolled' | 'Graduate'
export const CLASSES: Cls[] = ['Dropout', 'Enrolled', 'Graduate']
export interface Issue { field: string | null; level: 'error' | 'warning'; code: string; message: string }
export interface Ood { mahalanobis_sq: number; threshold: number; train_percentile: number; is_ood: boolean; method: string }
export interface Validation { status: 'normal' | 'unusual' | 'invalid'; summary: string; issues: Issue[]; ood?: Ood | null }
export interface Contribution { feature: string; label: string; value: number; shap: number; effect: string }
export interface Explanation {
  available: boolean; error?: string; method?: string; units?: string; summary?: string; note?: string
  by_class?: Record<string, { base_value: number | null; contributions: Contribution[] }>
}
export interface Component { score: number | null; weight: number; status: string }
export interface Indicator {
  label: string; score: number | null; level: string; components: Record<string, Component>; evaluated: number
  total: number; partial?: boolean; formula: string; weakest_component: string | null; capped: boolean
  disclaimer?: string; provenance?: string
}
export interface Uncertainty {
  confidence: number; margin: number; entropy: number; normalized_entropy: number; top_two: string[]
  level: 'LOW' | 'MEDIUM' | 'HIGH'; ensemble_disagreement: number | null; flag_for_review: boolean
  thresholds: Record<string, string>; note: string
}
export interface RobMetrics { stability_rate: number; mean_abs_prob_change: number; max_abs_prob_change: number; score: number; status: string }
export interface Perturbation {
  changes: { feature: string; label: string; from: number; to: number }[]; predicted_class: Cls
  probabilities: Record<string, number>; changed_prediction: boolean
}
export interface Robustness {
  n_perturbations: number; method: string; warning: string | null; metrics: RobMetrics; perturbations: Perturbation[]
  original: { predicted_class: Cls; probabilities: Record<string, number> }; provenance?: string
}
export interface CalPoint { bin_lower: number; bin_upper: number; count: number; mean_confidence: number | null; accuracy: number | null }
export interface Calibration {
  available: boolean; applied: boolean; method: string; provenance: string
  before: { brier_score: number; ece: number }; after: { brier_score: number; ece: number }
  reliability_before: CalPoint[]; reliability_after: CalPoint[]
}
export interface FairnessSummary { available: boolean; overall_status: string; attribute_status: Record<string, string>; note: string }
export interface Report {
  prediction_id: number; student_id: number | null; display_name: string; status: 'ok'; provenance: string; is_synthetic: boolean
  validation: Validation
  prediction: { predicted_class: Cls; probabilities: Record<string, number>; confidence: number; p_dropout: number
    risk_level: 'HIGH' | 'MEDIUM' | 'LOW'; statement: string; calibrated: boolean; model_name: string }
  uncertainty: Uncertainty; explanation: Explanation; robustness: Robustness; calibration: Calibration
  fairness: FairnessSummary; trust: { prediction_level: Indicator; system_level: Indicator }
  review: { recommended: boolean; reasons: string[]; message: string; final_decision_by: string }
}
export interface PredictionSummary {
  id: number; student_id: number | null; display_name: string; created_at: string; predicted_class: Cls
  probabilities: Record<string, number>; risk_level: string; uncertainty_level: string; trust_score: number | null
  trust_level: string; data_quality_status: string; needs_review: boolean; reviewed: boolean; provenance: string
}
export interface StudentListItem { id: number; display_name: string; quick_prediction: { predicted_class: Cls; p_dropout: number; risk_level: string } }
export interface FeatureMeta {
  name: string; label: string; group: string; kind: 'nominal' | 'binary' | 'numeric'
  options?: { value: number; label: string }[]; min?: number; max?: number
}
export interface Overview {
  provenance: string; is_synthetic: boolean; counts: Record<string, number>; risk_distribution: Record<string, number>
  class_distribution: Record<string, number> | null; pending_reviews: number
  model: { name: string; test: Record<string, number> | null; cv: Record<string, number> | null }
  trust: { indicator: Indicator | null; fairness: FairnessSummary; calibration: Calibration }
  recent_predictions: PredictionSummary[]
}
export interface GroupRow { group: string; n: number; actual_dropout_rate: number | null; flag_rate: number | null; accuracy: number | null
  precision: number | null; recall: number | null; fpr: number | null; fnr: number | null }
export interface FairnessAttr { label: string; status: string; groups: GroupRow[]; excluded_groups: string[]; warnings: string[]; gaps: Record<string, number | null> }
export interface Fairness {
  provenance: string; overall_status: string; audited_decision: string; caveat: string; worst_equalized_odds_diff: number | null
  thresholds: Record<string, string | number>; attributes: Record<string, FairnessAttr>
}
export interface GlobalImportance {
  provenance: string; method: string; units: string; n_rows: number
  features: { feature: string; label: string; mean_abs_shap: number; share: number }[]
}
export interface Perf {
  provenance: string; selection_criterion: string; selected_model: string; n_train: number; n_test: number; cv_folds: number
  multiclass_strategy: string
  results: Record<string, { cv: { mean: Record<string, number>; std: Record<string, number> }; test: Record<string, number> }>
  confusion_matrices: Record<string, { labels: string[]; matrix: number[][]; normalized_by_true: number[][] }>
}
export interface ReviewRow {
  id: number; prediction_id: number; counselor_name: string; action: string; model_prediction: string; final_decision: string
  reason: string | null; notes: string | null; created_at: string
}
