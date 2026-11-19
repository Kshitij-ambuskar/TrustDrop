import { NavLink, Route, Routes } from 'react-router-dom'
import Overview from './pages/Overview'
import Predict from './pages/Predict'
import TrustReport from './pages/TrustReport'
import Explainability from './pages/Explainability'
import FairnessPage from './pages/FairnessPage'
import Models from './pages/Models'
import RobustnessPage from './pages/RobustnessPage'
import Counselor from './pages/Counselor'

const NAV = [['/', 'Overview'], ['/predict', 'Student prediction'], ['/trust', 'Trust report'], ['/explain', 'Explainability'],
  ['/fairness', 'Fairness'], ['/models', 'Model performance'], ['/robustness', 'Robustness'], ['/review', 'Counselor review']]

export default function App() {
  return (
    <div className="min-h-screen md:grid md:grid-cols-[15rem_1fr]">
      <aside className="bg-ink px-4 py-5 text-slate-200 md:min-h-screen">
        <div className="font-display text-xl font-semibold text-white">TrustDrop</div>
        <p className="mb-4 mt-1 text-xs text-slate-400">Trustworthy dropout decision support. Research prototype; a counselor decides.</p>
        <nav aria-label="Main" className="flex flex-wrap gap-1 md:flex-col">
          {NAV.map(([to, name]) => (
            <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) =>
              `rounded px-3 py-1.5 text-sm ${isActive ? 'bg-white/15 font-medium text-white' : 'hover:bg-white/10'}`}>{name}</NavLink>))}
        </nav>
      </aside>
      <main className="min-w-0 p-4 md:p-8">
        <Routes>
          <Route path="/" element={<Overview />} /><Route path="/predict" element={<Predict />} />
          <Route path="/trust" element={<TrustReport />} /><Route path="/explain" element={<Explainability />} />
          <Route path="/fairness" element={<FairnessPage />} /><Route path="/models" element={<Models />} />
          <Route path="/robustness" element={<RobustnessPage />} /><Route path="/review" element={<Counselor />} />
        </Routes>
      </main>
    </div>
  )
}
