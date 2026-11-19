# TrustDrop – Trustworthy AI-Based Prediction System

TrustDrop is a full-stack AI application designed to make machine learning predictions more transparent, explainable, fair, and reliable.

Instead of providing predictions alone, TrustDrop incorporates explainability, fairness analysis, robustness checks, and a Trustworthiness Indicator to help users understand and evaluate AI-generated decisions.

## Key Features

- **AI Predictions:** Uses a trained Logistic Regression model to generate predictions.
- **Explainable AI:** Uses SHAP to help explain the factors influencing model predictions.
- **Fairness Analysis:** Integrates Fairlearn to examine potential fairness issues in model decisions.
- **Robustness Analysis:** Provides functionality to validate model reliability.
- **Trustworthiness Indicator:** Combines trust-related evaluation information to give users a better understanding of prediction reliability.
- **Counselor Override:** Supports human intervention when reviewing AI-generated decisions.
- **Student Dashboard:** Displays student information and prediction-related insights.

## Tech Stack

### Frontend
- React.js
- Vite
- JavaScript
- HTML and CSS

### Backend
- Python
- FastAPI
- Uvicorn

### Machine Learning and Explainability
- Scikit-learn
- Logistic Regression
- SHAP
- Fairlearn
- XGBoost

### Data and Visualization
- Pandas
- NumPy
- SQLAlchemy
- SQLite
- Matplotlib

## Project Architecture

```text
TrustDrop/
│
├── backend/
│   ├── app/
│   │   └── main.py
│   ├── models/
│   │   └── best_model.joblib
│   └── data/
│       └── trustdrop.db
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── ...
│
└── README.md
```

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `/predict` | Generate model predictions |
| `/validate` | Validate prediction-related information |
| `/students` | Retrieve student information |
| `/trust/summary` | Retrieve trustworthiness summary |

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Kshitij-ambuskar/TrustDrop.git
cd TrustDrop
```

### 2. Set Up the Backend

Create and activate a Python virtual environment.

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
pip install fastapi uvicorn scikit-learn xgboost shap fairlearn pandas numpy sqlalchemy matplotlib joblib
```

Start the backend server from the project root:

```powershell
python -m uvicorn app.main:app --reload --port 8000 --app-dir backend
```

Backend URL:

`http://localhost:8000`

API documentation:

`http://localhost:8000/docs`

### 3. Set Up the Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite in your browser.

## Current Development Status

- Full-stack frontend and backend integration.
- Trained Logistic Regression model.
- SHAP-based explainability functionality
