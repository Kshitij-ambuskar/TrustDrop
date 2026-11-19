import { FormEvent, useEffect, useState } from 'react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useAssessment } from '../context/AssessmentContext'
import { Badge, Card, Diverging, Empty, ErrorBox, IndicatorHeader, Loading, PageTitle, ProbBars, RiskBadge, SyntheticBanner, TrustBreakdown, uncTone } from '../components/ui'
import { CLASSES, Report } from '../types'
import { pct } from '../utils/format'

export default function Counselor() {
  const { report: ctxReport, setReport } = useAssessment()
  const list = useApi(() => api.predictions(), [])
  const reviews = useApi(() => api.reviews(), [])
  const [r, setR] = useState<Report | null>(ctxReport)
  const [name, setName] = useState('Dr. Demo Counselor'); const [action, setAction] = useState<'accept' | 'override'>('accept')
  const [decision, setDecision] = useState('Graduate'); const [reason, setReason] = useState(''); const [notes, setNotes] = useState('')
  const [when, setWhen] = useState(''); const [msg, setMsg] = useState<string | null>(null); const [err, setErr] = useState<string | null>(null)
  useEffect(() => setR(ctxReport), [ctxReport])

  const pick = async (id: string) => { if (!id) return; const x = await api.prediction(Number(id)); setReport(x); setMsg(null); setErr(null) }
  const save = async (e: FormEvent) => {
    e.preventDefault(); if (!r) return; setErr(null); setMsg(null)
    try {
      await api.review({ prediction_id: r.prediction_id, counselor_name: name, action, final_decision: action === 'override' ? decision : undefined, reason: reason || undefined, notes: notes || undefined })
      if (when) await api.schedule({ prediction_id: r.prediction_id, student_id: r.student_id ?? undefined, scheduled_for: new Date(when).toISOString(), notes: notes || undefined })
      setMsg(`Review saved${when ? ' and counseling scheduled' : ''}.`); setReason(''); setNotes(''); setWhen(''); list.reload(); reviews.reload()
    } catch (ex) { setErr((ex as Error).message) }
  }
  return (
    <>
      <PageTitle title="Counselor review" sub="The model offers an estimate; the counselor decides. Accept it, or override it with a reason. Every review is stored." />
      <SyntheticBanner provenance={r?.provenance} />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card title="Assessments">
            {list.loading ? <Loading /> : list.error ? <ErrorBox message={list.error} onRetry={list.reload} /> : list.data!.predictions.length === 0 ? <Empty>No assessments yet. Run one on the Student prediction page.</Empty> :
              <div className="max-h-56 overflow-auto"><table className="w-full text-left text-sm"><thead className="text-slate-500"><tr><th>#</th><th>Student</th><th>Prediction</th><th>Risk</th><th>Status</th><th /></tr></thead><tbody>
                {list.data!.predictions.map(p => <tr key={p.id} className="border-t border-line"><td className="py-1.5">{p.id}</td><td>{p.display_name}</td><td>{p.predicted_class}</td><td><RiskBadge r={p.risk_level} /></td><td>{p.reviewed ? 'Reviewed' : p.needs_review ? 'Review recommended' : 'Optional'}</td><td><button onClick={() => pick(String(p.id))} className="text-accent underline">Open</button></td></tr>)}</tbody></table></div>}
          </Card>
          {!r ? <Empty>Open an assessment to review it.</Empty> : <>
            <Card title={`${r.display_name} · assessment #${r.prediction_id}`} right={<RiskBadge r={r.prediction.risk_level} />}>
              <p className="mb-2 text-sm">{r.prediction.statement}</p>
              <div className="grid gap-4 md:grid-cols-2"><div><ProbBars probs={r.prediction.probabilities} />
                <div className="mt-2 flex flex-wrap gap-2 text-sm"><span>Confidence {pct(r.uncertainty.confidence)}</span><Badge tone={uncTone(r.uncertainty.level)}>Uncertainty {r.uncertainty.level}</Badge></div></div>
                <div><p className="mb-1 text-sm font-medium">Main factors (Dropout)</p>{r.explanation.available && r.explanation.by_class ? <Diverging rows={r.explanation.by_class.Dropout.contributions.slice(0, 5).map(c => ({ label: c.label, value: c.shap }))} /> : <p className="text-sm text-amber-800">Explanation unavailable.</p>}</div></div>
              {r.review.reasons.length > 0 && <div className="mt-3 rounded border border-amber-300 bg-amber-50 p-2 text-sm"><strong>Human counselor review recommended:</strong><ul className="list-disc pl-5">{r.review.reasons.map(x => <li key={x}>{x}</li>)}</ul></div>}
            </Card>
            <Card title="Trust information"><IndicatorHeader ind={r.trust.prediction_level} /><TrustBreakdown ind={r.trust.prediction_level} /></Card>
            <Card title="Your decision">
              <form onSubmit={save} className="space-y-3 text-sm">
                <div><label htmlFor="cn" className="block font-medium">Counselor name</label><input id="cn" required maxLength={80} value={name} onChange={e => setName(e.target.value)} className="w-full rounded border border-line px-2 py-1" /></div>
                <fieldset><legend className="font-medium">Decision</legend>
                  {(['accept', 'override'] as const).map(a => <label key={a} className="mr-4"><input type="radio" name="act" checked={action === a} onChange={() => setAction(a)} /> {a === 'accept' ? `Accept recommendation (${r.prediction.predicted_class})` : 'Override'}</label>)}</fieldset>
                {action === 'override' && <><div><label htmlFor="fd" className="block font-medium">Your decision</label><select id="fd" value={decision} onChange={e => setDecision(e.target.value)} className="rounded border border-line px-2 py-1">{CLASSES.map(c => <option key={c}>{c}</option>)}</select></div>
                  <div><label htmlFor="rs" className="block font-medium">Reason (required)</label><textarea id="rs" required maxLength={1000} value={reason} onChange={e => setReason(e.target.value)} className="w-full rounded border border-line px-2 py-1" rows={2} /></div></>}
                <div><label htmlFor="nt" className="block font-medium">Notes (optional)</label><textarea id="nt" maxLength={2000} value={notes} onChange={e => setNotes(e.target.value)} className="w-full rounded border border-line px-2 py-1" rows={2} /></div>
                <div><label htmlFor="wh" className="block font-medium">Schedule counseling (optional)</label><input id="wh" type="datetime-local" value={when} onChange={e => setWhen(e.target.value)} className="rounded border border-line px-2 py-1" /></div>
                {err && <p role="alert" className="text-rose-800">{err}</p>}{msg && <p role="status" className="text-emerald-800">✓ {msg}</p>}
                <button className="rounded bg-accent px-4 py-2 font-medium text-white">Save review</button>
                <p className="text-xs text-slate-500">Do not enter personal details in notes. Prototype only: not for automated decisions or for penalising students.</p>
              </form></Card></>}
        </div>
        <Card title="Review history">
          {reviews.loading ? <Loading /> : reviews.error ? <ErrorBox message={reviews.error} onRetry={reviews.reload} /> : reviews.data!.length === 0 ? <Empty>No reviews saved yet.</Empty> :
            <ul className="space-y-3 text-sm">{reviews.data!.map(v => <li key={v.id} className="border-b border-line pb-2"><div className="font-medium">#{v.prediction_id} · {v.counselor_name}</div>
              <div>{v.action === 'accept' ? 'Accepted' : 'Overrode'}: model said {v.model_prediction}, final {v.final_decision}</div>{v.reason && <div className="text-slate-600">Reason: {v.reason}</div>}{v.notes && <div className="text-slate-600">Notes: {v.notes}</div>}
              <div className="text-xs text-slate-500">{new Date(v.created_at).toLocaleString()}</div></li>)}</ul>}
        </Card>
      </div>
    </>
  )
}
