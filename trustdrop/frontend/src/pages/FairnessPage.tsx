import { useState } from 'react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Bars, Card, Empty, ErrorBox, FairBadge, Loading, Method, PageTitle, SyntheticBanner } from '../components/ui'
import { num, pct } from '../utils/format'

const GAPS: [string, string, string][] = [
  ['demographic_parity_diff', 'Demographic parity difference', 'Gap in how often groups are flagged as Dropout. Can reflect real differences in dropout rates, so it is a prompt to look closer, not proof of bias.'],
  ['equal_opportunity_diff', 'Equal opportunity difference', 'Gap in recall: among students who really dropped out, are all groups flagged equally often?'],
  ['fpr_diff', 'False-positive-rate difference', 'Gap in how often students who did not drop out are wrongly flagged.'],
  ['equalized_odds_diff', 'Equalized odds difference', 'The larger of the recall gap and the false-positive gap.'],
]

export default function FairnessPage() {
  const { data: f, error, loading, reload } = useApi(() => api.fairness(), [])
  const [attr, setAttr] = useState('gender')
  if (loading) return <Loading what="fairness audit" />
  if (error || !f) return <ErrorBox message={error ?? 'Unavailable'} onRetry={reload} />
  const a = f.attributes[attr] ?? Object.values(f.attributes)[0]
  const ok = a.groups.filter(g => !a.excluded_groups.includes(g.group))
  return (
    <>
      <PageTitle title="Fairness audit" sub={`Audited decision: ${f.audited_decision}. Evaluated on the held-out test split; nothing here is used for training.`} />
      <SyntheticBanner provenance={f.provenance} />
      <div className="mb-4 flex flex-wrap items-center gap-3"><label htmlFor="attr" className="text-sm font-medium">Attribute</label>
        <select id="attr" value={attr} onChange={e => setAttr(e.target.value)} className="rounded border border-line bg-white px-2 py-1 text-sm">{Object.entries(f.attributes).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}</select>
        <span className="text-sm">Overall:</span><FairBadge s={f.overall_status} /><span className="text-sm">This attribute:</span><FairBadge s={a.status} /></div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Group metrics">
          <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="text-slate-500"><tr><th>Group</th><th>n</th><th>Actual dropout</th><th>Flagged</th><th>Precision</th><th>Recall</th><th>FPR</th><th>FNR</th><th>Acc.</th></tr></thead><tbody>
            {a.groups.map(g => <tr key={g.group} className="border-t border-line"><td className="py-1.5 font-medium">{g.group}{a.excluded_groups.includes(g.group) ? ' *' : ''}</td><td>{g.n}</td><td>{pct(g.actual_dropout_rate)}</td><td>{pct(g.flag_rate)}</td><td>{pct(g.precision)}</td><td>{pct(g.recall)}</td><td>{pct(g.fpr)}</td><td>{pct(g.fnr)}</td><td>{pct(g.accuracy)}</td></tr>)}</tbody></table></div>
          {a.excluded_groups.length > 0 && <p className="mt-2 text-xs text-slate-600">* fewer than {String(f.thresholds.min_group_size)} students: excluded from gaps (too noisy).</p>}
        </Card>
        <Card title="Gaps between groups">
          {a.status === 'not_evaluated' ? <Empty>Not evaluated: fewer than two groups are large enough.</Empty> : <ul className="space-y-2 text-sm">
            {GAPS.map(([k, name]) => <li key={k} className="flex items-center justify-between gap-3"><span>{name}</span><span className="tabular-nums font-medium">{a.gaps[k] == null ? 'n/a' : num(a.gaps[k], 3)}</span></li>)}</ul>}
          {a.warnings.map(w => <p key={w} className="mt-2 text-xs text-amber-900">! {w}</p>)}
        </Card>
        <Card title="Flag rate by group"><Bars rows={ok.map(g => ({ label: `${g.group} (n=${g.n})`, value: g.flag_rate ?? 0 }))} /></Card>
        <Card title="Recall by group"><Bars rows={ok.map(g => ({ label: `${g.group} (n=${g.n})`, value: g.recall ?? 0 }))} /></Card>
        <Card title="What do these metrics mean?" className="lg:col-span-2">
          <dl className="grid gap-3 text-sm md:grid-cols-2">{GAPS.map(([k, name, desc]) => <div key={k}><dt className="font-medium">{name}</dt><dd className="text-slate-600">{desc}</dd></div>)}</dl>
          <Method><p>Status uses the worst gap among the three: &lt; {String(f.thresholds.fair).replace('all gaps < ', '')} Fair; below 0.20 Potential disparity; otherwise Requires review. Thresholds are project-defined, not legal standards.</p><p>{f.caveat}</p></Method>
        </Card>
      </div>
    </>
  )
}
