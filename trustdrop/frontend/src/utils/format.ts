export const pct = (v: number | null | undefined, d = 0) => (v == null ? 'n/a' : `${(v * 100).toFixed(d)}%`)
export const num = (v: number | null | undefined, d = 3) => (v == null ? 'n/a' : v.toFixed(d))
export const signed = (v: number, d = 2) => `${v >= 0 ? '+' : '-'}${Math.abs(v).toFixed(d)}`
export const label = (s: string) => s.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase())
export const FAIR_TEXT: Record<string, string> = { fair: 'Fair', potential_disparity: 'Potential disparity', requires_review: 'Requires review', not_evaluated: 'Not evaluated' }
export const COMP_NAMES: Record<string, string> = {
  explainability: 'Explainability', uncertainty: 'Uncertainty', calibration: 'Calibration',
  fairness: 'Fairness', robustness: 'Robustness', data_quality: 'Data quality',
}
