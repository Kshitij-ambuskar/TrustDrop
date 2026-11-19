import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import App from './App'
import { AssessmentProvider } from './context/AssessmentContext'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><HashRouter><AssessmentProvider><App /></AssessmentProvider></HashRouter></React.StrictMode>,
)
