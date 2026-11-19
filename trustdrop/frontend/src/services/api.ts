const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

export class ApiError extends Error {
  constructor(message: string, public status: number, public issues?: Issue[]) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, { headers: { 'Content-Type': 'application/json' }, ...init })
  } catch {
    throw new ApiError(`Cannot reach the API at ${BASE}. Is the backend running (uvicorn app.main:app)?`, 0)
  }
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const d = body?.detail
    if (d && typeof d === 'object' && !Array.isArray(d) && d.issues) throw new ApiError(d.message ?? 'Invalid input', res.status, d.issues)
    if (Array.isArray(d)) throw new ApiError(d.map((e: { msg: string }) => e.msg).join('; '), res.status)
    throw new ApiError(typeof d === 'string' ? d : `Request failed (${res.status})`, res.status)
  }
  return body as T
}
const post = <T,>(p: string, data: unknown) => request<T>(p, { method: 'POST', body: JSON.stringify(data) })

import type { Issue, Fairness, FeatureMeta, GlobalImportance, Overview, Perf, PredictionSummary, Report, ReviewRow, Robustness, StudentListItem, Validation } from '../types'

export const api = {
  overview: () => request<Overview>('/dashboard/overview'),
  schema: () => request<{ features: FeatureMeta[] }>('/schema/features'),
  students: () => request<{ students: StudentListItem[] }>('/students'),
  student: (id: number) => request<{ id: number; display_name: string; features: Record<string, number> }>(`/students/${id}`),
  validate: (features: Record<string, unknown>) => post<Validation>('/validate', { features }),
  predict: (body: { features?: Record<string, unknown>; student_id?: number; display_name?: string }) => post<Report>('/predict', body),
  prediction: (id: number) => request<Report>(`/predictions/${id}`),
  predictions: () => request<{ predictions: PredictionSummary[] }>('/predictions'),
  robustness: (student_id: number, n: number) => post<Robustness>('/trust/robustness', { student_id, n_perturbations: n }),
  fairness: () => request<Fairness>('/trust/fairness'),
  globalImportance: () => request<GlobalImportance>('/trust/explainability'),
  performance: () => request<Perf>('/model/performance'),
  reviews: () => request<ReviewRow[]>('/counseling/reviews'),
  review: (b: Record<string, unknown>) => post<ReviewRow>('/counseling/review', b),
  schedule: (b: Record<string, unknown>) => post<{ id: number }>('/counseling/schedule', b),
}
