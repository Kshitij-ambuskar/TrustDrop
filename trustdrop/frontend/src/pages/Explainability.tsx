import { useState } from 'react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { useAssessment } from '../context/AssessmentContext'
import { NoReport, ReportPicker } from '../components/ReportPicker'
import { Bars, Card, Diverging, ErrorBox, Loading, Method, PageTitle, SyntheticBanner } from '../components/ui'
import { CLASSES } from '../types'

export default function Explainability() {
  const g = useApi(() => api.globalImportance(), [])
  const { report: r } = useAssessment()
  const [cls, setCls] = useState<string>('Dropout')
  const ex = r?.explanation
  return (
    <>
      <PageTitle title="Explainability" sub="SHAP shows which features moved the model's score. It describes the model's behaviour; it does not prove what caused a student's outcome." />
      <SyntheticBanner provenance={g.data?.provenance ?? r?.provenance} />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Global feature importance">
          {g.loading ? <Loading /> : g.error || !g.data ? <ErrorBox message={g.error ?? 'Unavailable'} onRetry={g.reload} /> : <>
            <Bars max={g.data.features[0].share} rows={g.data.features.slice(0, 12).map(f => ({ label: f.label, value: f.share, text: `${(f.share * 100).toFixed(1)}%` }))} />
            <Method><p>Mean |SHAP| per feature on {g.data.n_rows} training rows, averaged over the three classes, shown as a share of the total. Explainer: {g.data.method}; units: {g.data.units}.</p></Method></>}
        </Card>
        <Card title="This student">
          <ReportPicker />
          {!r || !ex ? <NoReport /> : !ex.available || !ex.by_class ? <ErrorBox message={`Explanation unavailable: ${ex.error}`} /> : <>
            <div role="tablist" className="mb-3 flex gap-1">{CLASSES.map(c => <button key={c} role="tab" aria-selected={cls === c} onClick={() => setCls(c)} className={`rounded border px-3 py-1 text-sm ${cls === c ? 'border-ink bg-ink text-white' : 'border-line'}`}>{c}</button>)}</div>
            <p className="mb-2 text-sm">Features moving the score for <strong>{cls}</strong> (▲ raises it, ▼ lowers it):</p>
            <Diverging rows={ex.by_class[cls].contributions.map(c => ({ label: c.label, value: c.shap, sub: String(c.value) }))} />
            <p className="mt-3 text-sm text-slate-700">{ex.summary}</p><p className="mt-1 text-xs text-slate-500">{ex.note}</p>
            <Method><p>Each bar is a SHAP contribution in {ex.units}. Base value + sum of all contributions equals the model's raw output. One-hot columns were summed back to their original feature.</p></Method></>}
        </Card>
      </div>
    </>
  )
}
