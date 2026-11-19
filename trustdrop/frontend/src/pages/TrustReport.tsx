import { Link } from 'react-router-dom'
import { useAssessment } from '../context/AssessmentContext'
import { NoReport, ReportPicker } from '../components/ReportPicker'
import { Badge, Card, FairBadge, IndicatorHeader, Method, PageTitle, Reliability, Stat, SyntheticBanner, TrustBreakdown, dqTone, uncTone } from '../components/ui'
import { FAIR_TEXT, num, pct } from '../utils/format'

export default function TrustReport() {
  const { report: r } = useAssessment()
  return (
    <>
      <PageTitle title="Trust report" sub="Every dimension we evaluate for one assessment, with the method behind it. Missing evaluations are shown as 'Not evaluated', never invented." />
      <ReportPicker />
      {!r ? <NoReport /> : <>
        <SyntheticBanner provenance={r.provenance} />
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title={`Project-defined indicator · ${r.display_name}`} className="lg:col-span-2">
            <IndicatorHeader ind={r.trust.prediction_level} /><TrustBreakdown ind={r.trust.prediction_level} />
            <Method><p>{r.trust.prediction_level.formula}</p><p>Level: ≥75% High, ≥50% Moderate, otherwise Low; if any component is below 40% the level is capped at Moderate. Components not evaluated are left out and weights are renormalised.</p>
              <p>{r.trust.prediction_level.disclaimer} Limitations: weights and thresholds are subjective; components have different meanings; the score does not guarantee correctness or fairness.</p></Method>
            <p className="mt-2 text-sm text-slate-600">System-level indicator (whole model, held-out evaluation): {r.trust.system_level.score == null ? 'Not evaluated' : pct(r.trust.system_level.score)} ({r.trust.system_level.level})</p>
          </Card>

          <Card title="Data quality / out-of-distribution" right={<Badge tone={dqTone(r.validation.status)}>{r.validation.summary}</Badge>}>
            {r.validation.issues.length === 0 ? <p className="text-sm">No data-quality issues found.</p> : <ul className="list-disc space-y-1 pl-5 text-sm">{r.validation.issues.map((i, k) => <li key={k}>{i.message}</li>)}</ul>}
            {r.validation.ood && <p className="mt-2 text-sm">Typicality: this profile is more unusual than {pct(r.validation.ood.train_percentile / 100)} of training students{r.validation.ood.is_ood ? ' (flagged)' : ''}.</p>}
            <Method><p>Invalid = cannot be a real record (missing, wrong type, out of range, impossible combination); the model is not run. Unusual = valid but atypical versus training data (unseen code, outside training range, extreme value, or high Mahalanobis distance). This is a data signal and is separate from model uncertainty.</p>{r.validation.ood && <p>{r.validation.ood.method}.</p>}</Method>
          </Card>

          <Card title="Uncertainty" right={<Badge tone={uncTone(r.uncertainty.level)}>{r.uncertainty.level}</Badge>}>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4"><Stat name="Confidence" value={pct(r.uncertainty.confidence)} /><Stat name="Top-2 margin" value={pct(r.uncertainty.margin)} /><Stat name="Entropy (norm.)" value={num(r.uncertainty.normalized_entropy, 2)} />
              <Stat name="Member disagreement" value={r.uncertainty.ensemble_disagreement == null ? 'n/a' : num(r.uncertainty.ensemble_disagreement, 3)} /></div>
            <p className="mt-2 text-sm">Closest classes: {r.uncertainty.top_two.join(' vs ')}.{r.uncertainty.flag_for_review ? ' High uncertainty: counselor review recommended.' : ''}</p>
            <Method><p>Entropy H = −Σ p·ln p, divided by ln 3 so 0 is certain and 1 is a uniform guess. Margin = top probability − second probability. HIGH if entropy &gt; 0.85 or margin &lt; 0.15; LOW if entropy ≤ 0.65 and margin ≥ 0.40.</p><p>{r.uncertainty.note}</p></Method>
          </Card>

          <Card title="Calibration (whole model)" right={<Badge tone={r.calibration.available ? 'good' : 'neutral'}>{r.calibration.available ? (r.calibration.applied ? `${r.calibration.method} applied` : 'Not needed') : 'Not evaluated'}</Badge>}>
            {r.calibration.available ? <div className="flex flex-wrap items-center gap-4"><Reliability before={r.calibration.reliability_before} after={r.calibration.reliability_after} />
              <div className="space-y-2 text-sm"><Stat name="ECE before → after" value={`${num(r.calibration.before.ece)} → ${num(r.calibration.after.ece)}`} /><Stat name="Brier before → after" value={`${num(r.calibration.before.brier_score)} → ${num(r.calibration.after.brier_score)}`} />
                <p className="text-xs text-slate-500">Dashed grey: before. Teal: after. Diagonal: perfect.</p></div></div> : <p className="text-sm">No calibration report.</p>}
            <Method><p>ECE averages |accuracy − confidence| over 10 confidence bins. Brier = mean squared error of the probability vector. The calibrator is chosen using training data only; the test set only reports the result.</p></Method>
          </Card>

          <Card title="Robustness (this student)" right={<Badge tone={r.robustness.metrics.status === 'stable' ? 'good' : r.robustness.metrics.status === 'sensitive' ? 'warn' : 'bad'}>{r.robustness.metrics.status}</Badge>}>
            <div className="grid grid-cols-3 gap-3"><Stat name="Same class" value={pct(r.robustness.metrics.stability_rate)} /><Stat name="Mean prob. shift" value={pct(r.robustness.metrics.mean_abs_prob_change, 1)} /><Stat name="Score" value={pct(r.robustness.metrics.score)} /></div>
            {r.robustness.warning && <p role="alert" className="mt-2 text-sm text-amber-900">{r.robustness.warning}</p>}
            <p className="mt-2 text-sm"><Link className="text-accent underline" to="/robustness">Run more perturbation tests</Link></p>
            <Method><p>{r.robustness.method}</p><p>{r.robustness.n_perturbations} perturbed copies; score = stability × (1 − min(1, mean shift / 0.20)).</p></Method>
          </Card>

          <Card title="Fairness (whole model)" right={<FairBadge s={r.fairness.overall_status} />}>
            <ul className="space-y-1 text-sm">{Object.entries(r.fairness.attribute_status).map(([k, v]) => <li key={k} className="flex items-center justify-between"><span>{k.replace('_', ' ')}</span><FairBadge s={v} /></li>)}</ul>
            <p className="mt-2 text-xs text-slate-600">{r.fairness.note} <Link className="text-accent underline" to="/fairness">Full audit</Link></p>
            <Method><p>Statuses ({Object.values(FAIR_TEXT).join(', ')}) use project-defined gap thresholds (0.10 / 0.20). Different outcomes between groups can reflect different base rates and do not by themselves prove bias.</p></Method>
          </Card>

          <Card title="Explainability" right={<Badge tone={r.explanation.available ? 'good' : 'bad'}>{r.explanation.available ? 'Available' : 'Unavailable'}</Badge>}>
            <p className="text-sm">{r.explanation.available ? r.explanation.summary : r.explanation.error}</p>
            <p className="mt-2 text-sm"><Link className="text-accent underline" to="/explain">See contribution charts</Link></p>
            <Method><p>SHAP values from the fitted model ({r.explanation.method}); contributions are summed back to the 36 original features. They describe model behaviour, not causes. Indicator component = availability only.</p></Method>
          </Card>
        </div></>}
    </>
  )
}
