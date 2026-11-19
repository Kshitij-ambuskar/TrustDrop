import { ReactNode } from 'react'
import type { CalPoint, Indicator } from '../types'
import { COMP_NAMES, FAIR_TEXT, pct, signed } from '../utils/format'

type Tone = 'good' | 'warn' | 'bad' | 'neutral'
const TONE: Record<Tone, string> = {
  good: 'bg-emerald-50 text-emerald-900 border-emerald-300', warn: 'bg-amber-50 text-amber-900 border-amber-300',
  bad: 'bg-rose-50 text-rose-900 border-rose-300', neutral: 'bg-slate-100 text-slate-700 border-slate-300',
}
const SYMBOL: Record<Tone, string> = { good: '✓', warn: '!', bad: '✕', neutral: '–' }

/** Status chip: colour AND symbol AND text, so meaning never depends on colour alone. */
export function Badge({ tone, children }: { tone: Tone; children: ReactNode }) {
  return <span className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs font-medium ${TONE[tone]}`}><span aria-hidden>{SYMBOL[tone]}</span>{children}</span>
}
export const riskTone = (r: string): Tone => (r === 'HIGH' ? 'bad' : r === 'MEDIUM' ? 'warn' : 'good')
export const uncTone = (r: string): Tone => (r === 'HIGH' ? 'bad' : r === 'MEDIUM' ? 'warn' : 'good')
export const compTone = (s: string): Tone => (s === 'GOOD' ? 'good' : s === 'FAIR' ? 'warn' : s === 'POOR' ? 'bad' : 'neutral')
export const fairTone = (s: string): Tone => (s === 'fair' ? 'good' : s === 'potential_disparity' ? 'warn' : s === 'requires_review' ? 'bad' : 'neutral')
export const dqTone = (s: string): Tone => (s === 'normal' ? 'good' : s === 'unusual' ? 'warn' : 'bad')
export const RiskBadge = ({ r }: { r: string }) => <Badge tone={riskTone(r)}>{r} risk</Badge>
export const FairBadge = ({ s }: { s: string }) => <Badge tone={fairTone(s)}>{FAIR_TEXT[s] ?? s}</Badge>

export function Card({ title, right, children, className = '' }: { title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-line bg-white ${className}`}>
      {title && <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-3"><h2 className="font-semibold text-ink">{title}</h2>{right}</header>}
      <div className="p-4">{children}</div>
    </section>
  )
}
export function Stat({ name, value, hint }: { name: string; value: ReactNode; hint?: string }) {
  return <div><div className="text-sm text-slate-600">{name}</div><div className="font-display text-2xl font-semibold text-ink">{value}</div>{hint && <div className="text-xs text-slate-500">{hint}</div>}</div>
}
export const Loading = ({ what = 'data' }: { what?: string }) => <div role="status" className="p-6 text-slate-600">Loading {what}…</div>
export function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div role="alert" className="rounded-lg border border-rose-300 bg-rose-50 p-4 text-rose-900"><p className="font-medium">Something went wrong</p><p className="text-sm">{message}</p>{onRetry && <button onClick={onRetry} className="mt-2 rounded border border-rose-400 px-3 py-1 text-sm hover:bg-rose-100">Try again</button>}</div>
}
export const Empty = ({ children }: { children: ReactNode }) => <div className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-slate-600">{children}</div>
export function Method({ children }: { children: ReactNode }) {
  return <details className="mt-3 text-sm text-slate-600"><summary className="cursor-pointer font-medium text-accent">How is this calculated?</summary><div className="mt-2 space-y-1">{children}</div></details>
}
export const PageTitle = ({ title, sub }: { title: string; sub?: string }) => <div className="mb-5"><h1 className="font-display text-2xl font-semibold text-ink">{title}</h1>{sub && <p className="mt-1 max-w-3xl text-slate-600">{sub}</p>}</div>

export function SyntheticBanner({ provenance }: { provenance?: string }) {
  if (!provenance || !provenance.includes('SYNTHETIC')) return null
  return <div role="note" className="mb-4 rounded border-2 border-amber-500 bg-amber-50 px-4 py-2 text-sm text-amber-900"><strong>Development / synthetic data.</strong> All numbers shown come from random synthetic data used only to verify the pipeline. They are not real project results.</div>
}

/** Horizontal bar list. */
export function Bars({ rows, max = 1 }: { rows: { label: string; value: number; text?: string }[]; max?: number }) {
  return <ul className="space-y-2">{rows.map(r => (
    <li key={r.label} className="grid grid-cols-[minmax(0,10rem)_1fr_auto] items-center gap-3 text-sm">
      <span className="truncate" title={r.label}>{r.label}</span>
      <div className="h-3 rounded bg-slate-100"><div className="h-3 rounded bg-accent" style={{ width: `${Math.max(2, Math.min(100, (r.value / max) * 100))}%` }} /></div>
      <span className="w-14 text-right tabular-nums">{r.text ?? pct(r.value)}</span>
    </li>))}</ul>
}
export function ProbBars({ probs }: { probs: Record<string, number> }) {
  return <Bars rows={Object.entries(probs).map(([k, v]) => ({ label: k, value: v }))} />
}
/** Signed contribution bars (SHAP): sign shown in text as well as direction. */
export function Diverging({ rows }: { rows: { label: string; value: number; sub?: string }[] }) {
  const m = Math.max(...rows.map(r => Math.abs(r.value)), 1e-9)
  return <ul className="space-y-2">{rows.map(r => (
    <li key={r.label} className="grid grid-cols-[minmax(0,12rem)_1fr_auto] items-center gap-3 text-sm">
      <span className="truncate" title={r.sub ? `${r.label} (value ${r.sub})` : r.label}>{r.label}</span>
      <div className="relative h-3 rounded bg-slate-100"><div className="absolute inset-y-0 left-1/2 w-px bg-slate-400" />
        <div className={`absolute inset-y-0 rounded ${r.value >= 0 ? 'bg-rose-600' : 'bg-teal-700'}`}
          style={r.value >= 0 ? { left: '50%', width: `${(r.value / m) * 50}%` } : { right: '50%', width: `${(-r.value / m) * 50}%` }} /></div>
      <span className="w-20 text-right tabular-nums">{r.value >= 0 ? '▲ ' : '▼ '}{signed(r.value)}</span>
    </li>))}</ul>
}

/** The six components of the project-defined indicator, each with score, status text and weight. */
export function TrustBreakdown({ ind }: { ind: Indicator }) {
  return <ul className="space-y-2">{Object.entries(ind.components).map(([k, c]) => (
    <li key={k} className="grid grid-cols-[8rem_1fr_auto] items-center gap-3 text-sm">
      <span>{COMP_NAMES[k] ?? k}</span>
      <div className="h-3 rounded bg-slate-100">{c.score != null && <div className="h-3 rounded bg-accent" style={{ width: `${c.score * 100}%` }} />}</div>
      <span className="flex w-40 items-center justify-end gap-2"><span className="tabular-nums">{c.score == null ? '' : pct(c.score)}</span><Badge tone={compTone(c.status)}>{c.status === 'NOT_EVALUATED' ? 'Not evaluated' : c.status.charAt(0) + c.status.slice(1).toLowerCase()}</Badge></span>
    </li>))}</ul>
}
export function IndicatorHeader({ ind }: { ind: Indicator }) {
  return <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
    <div><div className="text-sm text-slate-600">{ind.label}</div>
      <div className="font-display text-4xl font-semibold text-ink">{ind.score == null ? 'Not evaluated' : pct(ind.score)}</div></div>
    <div className="text-right text-sm"><Badge tone={ind.level === 'High' ? 'good' : ind.level === 'Moderate' ? 'warn' : ind.level === 'Low' ? 'bad' : 'neutral'}>{ind.level} (project-defined)</Badge>
      <div className="mt-1 text-slate-600">{ind.evaluated} of {ind.total} components evaluated{ind.capped ? ' · capped: one weak component' : ''}</div></div></div>
}

export function Reliability({ before, after }: { before: CalPoint[]; after: CalPoint[] }) {
  const S = 220, P = 28, xy = (v: number) => P + v * (S - 2 * P)
  const line = (pts: CalPoint[]) => pts.filter(p => p.count > 0 && p.mean_confidence != null && p.accuracy != null)
  const path = (pts: CalPoint[]) => line(pts).map(p => `${xy(p.mean_confidence!)},${S - xy(p.accuracy!) + P - P}`).join(' ')
  return <svg viewBox={`0 0 ${S} ${S}`} className="w-full max-w-xs" role="img" aria-label="Reliability diagram: observed accuracy versus predicted confidence">
    <rect x={P} y={P} width={S - 2 * P} height={S - 2 * P} fill="none" stroke="#cbd5e1" />
    <line x1={P} y1={S - P} x2={S - P} y2={P} stroke="#64748b" strokeDasharray="4 3" />
    <polyline fill="none" stroke="#94a3b8" strokeWidth="2" strokeDasharray="2 2" points={path(before)} />
    <polyline fill="none" stroke="#0f766e" strokeWidth="2" points={path(after)} />
    {line(after).map((p, i) => <circle key={i} cx={xy(p.mean_confidence!)} cy={S - xy(p.accuracy!) + P - P} r="3" fill="#0f766e" />)}
    <text x={S / 2} y={S - 6} fontSize="9" textAnchor="middle" fill="#475569">Predicted confidence</text>
    <text x="8" y={S / 2} fontSize="9" fill="#475569" transform={`rotate(-90 8 ${S / 2})`} textAnchor="middle">Observed accuracy</text>
  </svg>
}
