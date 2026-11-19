import { useState } from 'react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Badge, Card, Empty, ErrorBox, Loading, Method, PageTitle, ProbBars, Stat, SyntheticBanner } from '../components/ui'
import { pct } from '../utils/format'
import type { Robustness } from '../types'

export default function RobustnessPage() {
  const students = useApi(() => api.students(), [])
  const [sid, setSid] = useState(''); const [n, setN] = useState(30)
  const [res, setRes] = useState<Robustness | null>(null); const [busy, setBusy] = useState(false); const [err, setErr] = useState<string | null>(null)
  const run = async () => {
    if (!sid) return
    setBusy(true); setErr(null)
    try { setRes(await api.robustness(Number(sid), n)) } catch (e) { setErr((e as Error).message) } finally { setBusy(false) }
  }
  if (students.loading) return <Loading what="students" />
  if (students.error) return <ErrorBox message={students.error} onRetry={students.reload} />
  const m = res?.metrics
  return (
    <>
      <PageTitle title="Robustness test" sub="Make small, valid changes to a student's grades and unit counts and check whether the prediction stays the same." />
      <SyntheticBanner provenance={res?.provenance} />
      <Card className="mb-4"><div className="flex flex-wrap items-end gap-4">
        <div><label htmlFor="s" className="block text-sm font-medium">Student</label><select id="s" value={sid} onChange={e => setSid(e.target.value)} className="rounded border border-line bg-white px-2 py-1.5 text-sm"><option value="">Select a student…</option>{students.data?.students.map(s => <option key={s.id} value={s.id}>{s.display_name}</option>)}</select></div>
        <div><label htmlFor="n" className="block text-sm font-medium">Perturbations: {n}</label><input id="n" type="range" min={5} max={100} value={n} onChange={e => setN(Number(e.target.value))} /></div>
        <button onClick={run} disabled={!sid || busy} className="rounded bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50">{busy ? 'Running…' : 'Run perturbation tests'}</button></div></Card>
      {err && <ErrorBox message={err} />}
      {!res && !err && <Empty>Choose a student and run the test to see how stable the prediction is.</Empty>}
      {res && m && <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Result" right={<Badge tone={m.status === 'stable' ? 'good' : m.status === 'sensitive' ? 'warn' : 'bad'}>{m.status}</Badge>}>
          <div className="mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4"><Stat name="Prediction stability" value={pct(m.stability_rate)} /><Stat name="Mean prob. shift" value={pct(m.mean_abs_prob_change, 1)} /><Stat name="Max prob. shift" value={pct(m.max_abs_prob_change, 1)} /><Stat name="Robustness score" value={pct(m.score)} /></div>
          {res.warning && <p role="alert" className="rounded border border-amber-400 bg-amber-50 p-2 text-sm text-amber-900">{res.warning}</p>}
          <p className="mt-3 text-sm font-medium">Original: {res.original.predicted_class}</p><ProbBars probs={res.original.probabilities} />
          <Method><p>{res.method}</p><p>Stable: ≥90% same class and max shift ≤15 points. Unstable: &lt;70% same class or max shift &gt;30 points. Otherwise sensitive.</p></Method>
        </Card>
        <Card title="Perturbed copies"><div className="max-h-[28rem] overflow-auto"><table className="w-full text-left text-sm"><thead className="sticky top-0 bg-white text-slate-500"><tr><th>#</th><th>Changes</th><th>Prediction</th></tr></thead><tbody>
          {res.perturbations.map((p, i) => <tr key={i} className="border-t border-line align-top"><td className="py-1.5">{i + 1}</td><td>{p.changes.length ? p.changes.map(c => `${c.label}: ${c.from} → ${c.to}`).join('; ') : 'no change'}</td>
            <td>{p.predicted_class} {pct(p.probabilities[p.predicted_class])} {p.changed_prediction && <Badge tone="warn">changed</Badge>}</td></tr>)}</tbody></table></div></Card></div>}
    </>
  )
}
