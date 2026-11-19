import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { useAssessment } from '../context/AssessmentContext'
import { useApi } from '../hooks/useApi'
import { Empty } from './ui'

/** Lets pages reuse the latest assessment, or load any stored one from SQLite. */
export function ReportPicker() {
  const { report, setReport } = useAssessment()
  const { data } = useApi(() => api.predictions(), [])
  useEffect(() => {}, [data])
  const rows = data?.predictions ?? []
  return (
    <div className="mb-4 flex flex-wrap items-center gap-3 text-sm">
      <label htmlFor="rp" className="font-medium">Assessment</label>
      <select id="rp" className="rounded border border-line bg-white px-2 py-1" value={report?.prediction_id ?? ''}
        onChange={e => e.target.value && api.prediction(Number(e.target.value)).then(setReport)}>
        <option value="">{rows.length ? 'Choose a stored assessment…' : 'No stored assessments yet'}</option>
        {rows.map(r => <option key={r.id} value={r.id}>#{r.id} · {r.display_name} · {r.predicted_class}</option>)}
      </select>
      <Link className="text-accent underline" to="/predict">Run a new assessment</Link>
    </div>
  )
}
export const NoReport = () => <Empty>No assessment selected. <Link className="text-accent underline" to="/predict">Run a student assessment</Link> or pick a stored one above.</Empty>
