import { FormEvent, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, ApiError } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useAssessment } from '../context/AssessmentContext'
import { Badge, Card, Diverging, ErrorBox, IndicatorHeader, Loading, PageTitle, ProbBars, RiskBadge, SyntheticBanner, TrustBreakdown, dqTone, uncTone } from '../components/ui'
import { pct } from '../utils/format'
import type { FeatureMeta, Issue } from '../types'

export default function Predict() {
  const nav = useNavigate()
  const { report, setReport } = useAssessment()
  const schema = useApi(() => api.schema(), [])
  const students = useApi(() => api.students(), [])
  const [vals, setVals] = useState<Record<string, string>>({})
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [issues, setIssues] = useState<Issue[]>([])

  if (schema.loading) return <Loading what="form" />
  if (schema.error || !schema.data) return <ErrorBox message={schema.error ?? 'No schema'} onRetry={schema.reload} />
  const groups = [...new Set(schema.data.features.map(f => f.group))]

  const loadDemo = async (id: string) => {
    if (!id) return
    const s = await api.student(Number(id))
    setVals(Object.fromEntries(Object.entries(s.features).map(([k, v]) => [k, String(v)]))); setName(s.display_name)
    setIssues([]); setErr(null)
  }
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setBusy(true); setErr(null); setIssues([])
    // blank stays blank so the backend reports it as a missing value (no silent defaults)
    const features = Object.fromEntries(Object.entries(vals).map(([k, v]) => [k, v === '' ? null : Number(v)]))
    try { setReport(await api.predict({ features, display_name: name || undefined })) }
    catch (ex) { const a = ex as ApiError; setErr(a.message); setIssues(a.issues ?? []) }
    finally { setBusy(false) }
  }
  const field = (f: FeatureMeta) => {
    const v = vals[f.name] ?? ''
    const bad = issues.find(i => i.field === f.name && i.level === 'error')
    const common = { id: f.name, value: v, onChange: (e: { target: { value: string } }) => setVals({ ...vals, [f.name]: e.target.value }), 'aria-invalid': !!bad, className: `w-full rounded border px-2 py-1 text-sm ${bad ? 'border-rose-500' : 'border-line'}` }
    return (
      <div key={f.name}><label htmlFor={f.name} className="mb-0.5 block text-xs font-medium text-slate-700">{f.label}{f.min != null ? ` (${f.min}–${f.max})` : ''}</label>
        {f.kind === 'numeric' ? <input type="number" step="any" {...common} /> :
          <select {...common}><option value="">Select…</option>{f.options?.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}</select>}
        {bad && <p className="mt-0.5 text-xs text-rose-700">{bad.message}</p>}</div>)
  }
  const r = report

  return (
    <>
      <PageTitle title="Student prediction" sub="Enter a student record or load a demo student. The input is validated first; the model only runs on valid data." />
      <SyntheticBanner provenance={r?.provenance} />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <form onSubmit={submit} className="space-y-4">
          <Card title="Student information" right={
            <select aria-label="Load demo student" className="rounded border border-line px-2 py-1 text-sm" defaultValue="" onChange={e => loadDemo(e.target.value)}>
              <option value="">Load demo student…</option>{students.data?.students.map(s => <option key={s.id} value={s.id}>{s.display_name}</option>)}</select>}>
            <label htmlFor="nm" className="mb-0.5 block text-xs font-medium">Display name (optional, use a synthetic name)</label>
            <input id="nm" maxLength={80} value={name} onChange={e => setName(e.target.value)} className="mb-3 w-full rounded border border-line px-2 py-1 text-sm" />
            {groups.map(g => <fieldset key={g} className="mb-4"><legend className="mb-2 text-sm font-semibold text-ink">{g}</legend>
              <div className="grid gap-3 sm:grid-cols-2">{schema.data!.features.filter(f => f.group === g).map(field)}</div></fieldset>)}
            <p className="text-xs text-slate-500">Nominal fields use the dataset's numeric codes (course, qualification, occupation, …).</p>
          </Card>
          {err && <div role="alert" className="rounded-lg border border-rose-400 bg-rose-50 p-4 text-rose-900"><p className="font-semibold">{issues.length ? 'INVALID INPUT' : 'Request failed'}</p><p className="text-sm">{err}</p>
            <ul className="mt-2 list-disc pl-5 text-sm">{issues.filter(i => i.level === 'error').map((i, k) => <li key={k}>{i.message}</li>)}</ul>{issues.length > 0 && <p className="mt-2 text-sm">No prediction was made.</p>}</div>}
          <button disabled={busy} className="rounded bg-accent px-4 py-2 font-medium text-white disabled:opacity-60">{busy ? 'Assessing…' : 'Run assessment'}</button>
        </form>

        <div className="space-y-4">
          {!r ? <Card title="Result"><p className="text-slate-600">Run an assessment to see the prediction, uncertainty, explanation and trust analysis here.</p></Card> : <>
            <Card title={`Assessment #${r.prediction_id} · ${r.display_name}`} right={<RiskBadge r={r.prediction.risk_level} />}>
              <p className="mb-1 font-display text-xl font-semibold text-ink">Most likely class: {r.prediction.predicted_class}</p>
              <p className="mb-3 text-sm text-slate-700">{r.prediction.statement}</p>
              <ProbBars probs={r.prediction.probabilities} />
              <div className="mt-3 flex flex-wrap gap-2 text-sm"><Badge tone={uncTone(r.uncertainty.level)}>Uncertainty {r.uncertainty.level}</Badge><span>Confidence {pct(r.uncertainty.confidence)}</span><span>Margin {pct(r.uncertainty.margin)}</span><Badge tone={dqTone(r.validation.status)}>Input {r.validation.status}</Badge></div>
              <p className="mt-2 text-xs text-slate-500">{r.uncertainty.note}</p>
            </Card>
            <Card title="Why? Main factors for Dropout">
              {r.explanation.available && r.explanation.by_class ? <><Diverging rows={r.explanation.by_class.Dropout.contributions.slice(0, 6).map(c => ({ label: c.label, value: c.shap, sub: String(c.value) }))} />
                <p className="mt-2 text-xs text-slate-600">▲ pushes toward Dropout, ▼ pushes away (SHAP, {r.explanation.units}). {r.explanation.summary}</p></> : <p className="text-sm text-amber-800">Explanation unavailable: {r.explanation.error}</p>}
            </Card>
            <Card title="Trust analysis"><IndicatorHeader ind={r.trust.prediction_level} /><TrustBreakdown ind={r.trust.prediction_level} /></Card>
            <Card title={r.review.recommended ? 'Human counselor review recommended' : 'Review'}>
              {r.review.reasons.length > 0 ? <ul className="list-disc pl-5 text-sm">{r.review.reasons.map(x => <li key={x}>{x}</li>)}</ul> : <p className="text-sm">{r.review.message}</p>}
              <div className="mt-3 flex flex-wrap gap-2"><Link to="/trust" className="rounded border border-line px-3 py-1.5 text-sm">View full trust report</Link>
                <button onClick={() => nav('/review')} className="rounded bg-ink px-3 py-1.5 text-sm text-white">Send to counselor review</button></div>
            </Card></>}
        </div>
      </div>
    </>
  )
}
