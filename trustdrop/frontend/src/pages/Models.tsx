import { useState } from 'react'
import { api } from '../services/api'
import { useApi } from '../hooks/useApi'
import { Bars, Card, ErrorBox, Loading, Method, PageTitle, SyntheticBanner, Badge } from '../components/ui'
import { num, pct } from '../utils/format'

const COLS: [string, string][] = [['accuracy', 'Accuracy'], ['precision_macro', 'Precision'], ['recall_macro', 'Recall'], ['f1_macro', 'F1'], ['roc_auc_ovr_macro', 'ROC-AUC']]

export default function Models() {
  const { data: p, error, loading, reload } = useApi(() => api.performance(), [])
  const [sel, setSel] = useState<string | null>(null)
  if (loading) return <Loading what="model comparison" />
  if (error || !p) return <ErrorBox message={error ?? 'Unavailable'} onRetry={reload} />
  const cur = sel ?? p.selected_model, cm = p.confusion_matrices[cur]
  return (
    <>
      <PageTitle title="Model performance" sub="All four models are trained and scored with the same split, the same cross-validation and the same metrics." />
      <SyntheticBanner provenance={p.provenance} />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={`Cross-validation (${p.cv_folds}-fold, ${p.n_train} training rows)`} className="lg:col-span-2">
          <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="text-slate-500"><tr><th>Model</th>{COLS.map(([, n]) => <th key={n}>{n}</th>)}</tr></thead><tbody>
            {Object.entries(p.results).map(([m, r]) => <tr key={m} className="border-t border-line"><td className="py-1.5 font-medium">{m} {m === p.selected_model && <Badge tone="good">selected</Badge>}</td>
              {COLS.map(([k]) => <td key={k} className="tabular-nums">{num(r.cv.mean[k])} ± {num(r.cv.std[k])}</td>)}</tr>)}</tbody></table></div>
          <Method><p>Selection criterion: {p.selection_criterion}</p><p>Multiclass strategy: {p.multiclass_strategy}.</p></Method>
        </Card>
        <Card title={`Held-out test split (${p.n_test} rows)`}>
          <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead className="text-slate-500"><tr><th>Model</th>{COLS.map(([, n]) => <th key={n}>{n}</th>)}</tr></thead><tbody>
            {Object.entries(p.results).map(([m, r]) => <tr key={m} className="border-t border-line"><td className="py-1.5">{m}</td>{COLS.map(([k]) => <td key={k} className="tabular-nums">{num(r.test[k])}</td>)}</tr>)}</tbody></table></div>
          <p className="mt-3 text-sm font-medium">Macro F1 comparison (cross-validated)</p>
          <Bars rows={Object.entries(p.results).map(([m, r]) => ({ label: m, value: r.cv.mean.f1_macro }))} />
        </Card>
        <Card title="Confusion matrix" right={<select aria-label="Model" value={cur} onChange={e => setSel(e.target.value)} className="rounded border border-line px-2 py-1 text-sm">{Object.keys(p.results).map(m => <option key={m}>{m}</option>)}</select>}>
          <table className="w-full text-center text-sm"><thead><tr><th className="text-left text-xs font-normal text-slate-500">true ↓ / predicted →</th>{cm.labels.map(l => <th key={l}>{l}</th>)}</tr></thead><tbody>
            {cm.matrix.map((row, i) => <tr key={i}><th className="text-left">{cm.labels[i]}</th>{row.map((v, j) => <td key={j} className="border border-line p-2" style={{ background: `rgba(15,118,110,${0.08 + 0.6 * cm.normalized_by_true[i][j]})` }}><div className="font-semibold">{v}</div><div className="text-xs">{pct(cm.normalized_by_true[i][j])}</div></td>)}</tr>)}</tbody></table>
          <p className="mt-2 text-xs text-slate-500">Cell shade and percentage = share of that true class (row).</p>
        </Card>
      </div>
    </>
  )
}
