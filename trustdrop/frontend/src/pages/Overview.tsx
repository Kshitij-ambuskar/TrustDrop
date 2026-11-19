import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Bars, Card, Empty, ErrorBox, FairBadge, IndicatorHeader, Loading, PageTitle, RiskBadge, Stat, SyntheticBanner, TrustBreakdown, Badge, dqTone } from '../components/ui'
import { num, pct } from '../utils/format'

export default function Overview() {
  const { data: o, error, loading, reload } = useApi(() => api.overview(), [])
  if (loading) return <Loading what="overview" />
  if (error || !o) return <ErrorBox message={error ?? 'No data'} onRetry={reload} />
  const rd = o.risk_distribution, test = o.model.test
  const total = Math.max(1, o.counts.students)
  return (
    <>
      <PageTitle title="Overview" sub="Dropout risk estimates come with explanations, uncertainty, fairness, robustness and data-quality checks. A human counselor makes every final decision." />
      <SyntheticBanner provenance={o.provenance} />
      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Students (demo records)" className="lg:col-span-1">
          <div className="mb-4 grid grid-cols-4 gap-2"><Stat name="Total" value={o.counts.students} /><Stat name="High" value={rd.HIGH} /><Stat name="Medium" value={rd.MEDIUM} /><Stat name="Low" value={rd.LOW} /></div>
          <Bars max={total} rows={[{ label: 'High risk', value: rd.HIGH, text: String(rd.HIGH) }, { label: 'Medium risk', value: rd.MEDIUM, text: String(rd.MEDIUM) }, { label: 'Low risk', value: rd.LOW, text: String(rd.LOW) }]} />
          <p className="mt-3 text-xs text-slate-500">Risk bands come from the model's estimated probability of dropout (≥50% high, ≥30% medium). Names are synthetic.</p>
        </Card>
        <Card title="Selected model" className="lg:col-span-1">
          <p className="mb-3 font-medium">{o.model.name}{o.trust.calibration.applied ? ` · ${o.trust.calibration.method} calibration` : ''}</p>
          {test ? <div className="grid grid-cols-2 gap-3"><Stat name="Accuracy" value={pct(test.accuracy, 1)} /><Stat name="Macro F1" value={pct(test.f1_macro, 1)} /><Stat name="Macro precision" value={pct(test.precision_macro, 1)} /><Stat name="Macro recall" value={pct(test.recall_macro, 1)} /><Stat name="ROC-AUC (OvR)" value={num(test.roc_auc_ovr_macro)} /></div> : <Empty>No evaluation report found.</Empty>}
          <p className="mt-3 text-xs text-slate-500">Held-out test split. <Link className="text-accent underline" to="/models">All models</Link></p>
        </Card>
        <Card title="Class distribution (dataset)" className="lg:col-span-1">
          {o.class_distribution ? <Bars max={Math.max(...Object.values(o.class_distribution))} rows={Object.entries(o.class_distribution).map(([k, v]) => ({ label: k, value: v, text: String(v) }))} /> : <Empty>Not available.</Empty>}
        </Card>
        <Card title="System trust" className="lg:col-span-2" right={<Link className="text-sm text-accent underline" to="/trust">Details</Link>}>
          {o.trust.indicator ? <><IndicatorHeader ind={o.trust.indicator} /><TrustBreakdown ind={o.trust.indicator} /></> : <Empty>No trust artifacts. Retrain the model.</Empty>}
          <div className="mt-4 flex flex-wrap items-center gap-3 text-sm"><span>Fairness audit:</span><FairBadge s={o.trust.fairness.overall_status} /></div>
          <p className="mt-2 text-xs text-slate-500">Project-defined indicator for demonstration only. It is not a universal measure of how trustworthy an AI is.</p>
        </Card>
        <Card title="Needs human attention">
          <Stat name="Recent assessments awaiting review" value={o.pending_reviews} />
          <Link to="/review" className="mt-3 inline-block rounded bg-accent px-3 py-1.5 text-sm text-white">Open counselor review</Link>
        </Card>
        <Card title="Recent assessments" className="lg:col-span-3">
          {o.recent_predictions.length === 0 ? <Empty>No assessments yet. <Link className="text-accent underline" to="/predict">Assess a student</Link>.</Empty> :
            <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="text-slate-500"><tr><th className="py-1">Student</th><th>Prediction</th><th>Risk</th><th>Uncertainty</th><th>Data</th><th>Trust</th><th>Review</th></tr></thead><tbody>
              {o.recent_predictions.map(r => <tr key={r.id} className="border-t border-line"><td className="py-2">{r.display_name}</td><td>{r.predicted_class}</td><td><RiskBadge r={r.risk_level} /></td><td>{r.uncertainty_level}</td>
                <td><Badge tone={dqTone(r.data_quality_status)}>{r.data_quality_status}</Badge></td><td>{r.trust_score == null ? 'n/a' : `${pct(r.trust_score)} ${r.trust_level}`}</td><td>{r.reviewed ? 'Reviewed' : r.needs_review ? 'Recommended' : 'Optional'}</td></tr>)}</tbody></table></div>}
        </Card>
      </div>
    </>
  )
}
