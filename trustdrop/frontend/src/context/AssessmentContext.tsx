import { createContext, ReactNode, useContext, useState } from 'react'
import type { Report } from '../types'

interface Ctx { report: Report | null; setReport: (r: Report | null) => void }
const C = createContext<Ctx>({ report: null, setReport: () => {} })
export const useAssessment = () => useContext(C)
export function AssessmentProvider({ children }: { children: ReactNode }) {
  const [report, setReport] = useState<Report | null>(null)
  return <C.Provider value={{ report, setReport }}>{children}</C.Provider>
}
