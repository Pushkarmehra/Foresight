<p align="center">
  <img src="banner.png" alt="Predictive Maintenance System banner" width="100%">
</p>

<h1 align="center">Predictive Maintenance System</h1>
<p align="center"><b>Will this machine fail in the next 7 days? — and if so, when, why, and how confident are we?</b></p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/Model-XGBoost%20%2F%20LightGBM-00A98F">
  <img alt="Survival Analysis" src="https://img.shields.io/badge/Survival-lifelines-orange">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white">
  <img alt="MLflow" src="https://img.shields.io/badge/Tracking-MLflow-0194E2">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
  <img alt="Status" src="https://img.shields.io/badge/Status-Phase%200%20🚀-blue">
</p>

---

## Quick Navigation

- [The Problem](#the-problem) — Why this matters
- [Quick Start Decision Tree](#-quick-start-decision-tree) — Choose your path
- [What This Is](#what-this-is) — System architecture & approach
- [Tech Stack](#tech-stack-professional-resume-defensible) — Professional tools & why
- [Key Differentiators](#key-differentiators) — What makes this production-grade
- [Project Roadmap](#project-roadmap) — 12 phases with time estimates
- [Getting Started](#getting-started) — Run it locally in 2 minutes
- [Common Pitfalls](#-common-pitfalls) — What to avoid
- [Project Structure](#project-structure) — Folder organization

---

## ⚡ Quick Start Decision Tree

**What's your goal?**

```
Are you a...

📚 STUDENT learning ML end-to-end?
   → Do Phases 0-6 (core: 60-80 hrs)
   → Then pick ONE phase 7-12 to deepen
   → Recommendation: Phase 5 (Anomaly Detection) or Phase 7 (Survival Analysis)

💼 CANDIDATE building a portfolio?
   → Do Phases 0-8 (core + explainability: 90-120 hrs)
   → Push Phase 10 (React frontend) for "I can ship"
   → Recommendation: Complete phases 0-8, then pick Phase 10 OR Phase 11

🏭 ENGINEER solving a real problem?
   → Phases 0-11 (production-ready: 120-160 hrs)
   → Phase 12 (documentation) last
   → Recommendation: All core phases, at least one push-your-limits per phase

⚡ FAST TRACK (weekend project)?
   → Phases 0, 1 (data), 2 (labels), 6 (XGBoost), 10 (API only)
   → Skip: survival analysis, autoencoder, React, drift monitoring
   → Time: 20-30 hrs, skip push-your-limits
```

---

## The Problem

Industrial machines fail without warning—or do they? Equipment degrades gradually across multiple sensor channels: vibration, temperature, pressure, electrical signals. The challenge:

- **Binary predictions are incomplete**: Saying "yes, failure" doesn't help if you don't know *when*.
- **Black boxes aren't trusted**: Operations teams need to see *why* the model said failure is imminent.
- **Real data has traps**: Random train/test splits, leaky timestamps, uncalibrated confidence scores—the difference between an academic exercise and a system that works.

This project builds the **real thing**: a production-shaped ML system that answers the three questions maintenance teams actually ask:
1. **Will it fail?** (probability)
2. **When?** (failure window with confidence interval)
3. **Why?** (top signals, direction of drift, physical interpretation)

---

## What This Is

An end-to-end, production-shaped predictive maintenance system for industrial machines. Sensors (temperature, pressure, vibration, RPM, voltage, current, humidity) stream in; the system continuously estimates:

- **Health Score** — single-number readiness (0–100)
- **Failure Probability** — next 7 days, calibrated confidence
- **Estimated Failure Window** — "3–6 days" not "72 hours"
- **Top Signals** — *why* the prediction, in plain language

This is deliberately built to go past the "load CSV → RandomForest → 94% accuracy" version. See [`docs/why-this-is-hard.md`](docs/why-this-is-hard.md) for the reasoning behind every architectural decision.

<p align="center">
  <img src="architecture.png" alt="System architecture diagram" width="100%">
</p>

---

## Tech Stack (Professional, Resume-Defensible)

| Layer | Tools | Why this and not the "easy" option |
|---|---|---|
| Language | Python 3.11 | Standard for ML tooling |
| Data / feature pipeline | Pandas, Polars (for speed on large sensor logs), NumPy | Polars forces you to learn a modern, faster alternative to Pandas — a real differentiator |
| Anomaly detection | scikit-learn `IsolationForest`, PyTorch autoencoder | Two approaches so you can compare classical vs. deep methods |
| Classification | XGBoost, LightGBM | Industry standard for tabular time-series-derived features |
| Time-to-event modeling | `lifelines` (Cox PH, Weibull AFT) | Produces a real calibrated failure window instead of a guessed range |
| Explainability | SHAP | Turns "72% risk" into "72% risk because vibration and current are trending up" |
| Experiment tracking | MLflow | Shows you track experiments like a real ML team, not just notebooks |
| Validation | scikit-learn `TimeSeriesSplit`, `GroupKFold` | Enforces no data leakage — see the leakage checklist below |
| Serving | FastAPI | Async, typed, production-grade API — not just a notebook function |
| Frontend | Streamlit (fast MVP) → React + Recharts (stretch) | Start functional, upgrade to a real frontend to prove range |
| Containerization | Docker + docker-compose | Reproducible, deployable, the bar for "professional-level app" |
| CI | GitHub Actions | Automated tests + linting on every push |
| Testing | pytest | Unit tests on feature engineering and leakage guards specifically |
| Monitoring (stretch) | Evidently AI | Data/model drift detection — this is what separates a project from a *system* |

---

## Key Differentiators

What makes this *resume-worthy* instead of just another ML project:

### 1. **Survival Analysis for Confidence Intervals**
Most projects predict a single number. This predicts a *distribution*: "3–6 days (80% confidence)." This is graduate-level statistics that 99% of portfolio projects skip.

### 2. **Uncompromising Data Leakage Guards**
- Time-series-aware train/test split (no random shuffle)
- Hold-out machines the model has never seen
- Unit tests that fail if leakage is reintroduced
- Embargo gaps around label boundaries
- Scaler/feature fitting only on train data

This is what separates a toy model from a production system.

### 3. **Explainability by Design**
SHAP values + domain-specific interpretation. The model doesn't just say "72% risk"—it says "72% risk because vibration is trending up 15% and current draw has spiked, consistent with bearing degradation."

### 4. **Layered Architecture**
- Layer 1: Anomaly detection (Isolation Forest + autoencoder)
- Layer 2: Failure classification (XGBoost/LightGBM)
- Layer 3: Failure window (survival model)

Not a monolith. Each layer is independently interpretable and swappable.

### 5. **Instrumented for Production**
- Experiment tracking in MLflow (not just Jupyter notebooks)
- FastAPI backend (async, typed, deployable)
- Docker containerization
- Structured logging
- CI/CD pipeline

You can actually *ship* this, not just present results.

---

## Project Roadmap

**Current Status:** Phase 0 — Setting up the foundation. Check off phases as you complete them.

### Phase 0 — Setup (Foundation)
**⏱️ Time: 2–4 hours** | **Difficulty: Beginner**

- [ ] Initialize repo with `src/`, `data/`, `notebooks/`, `tests/`, `app/`, `docs/`, `assets/`
- [ ] Set up a virtual environment + `requirements.txt` / `pyproject.toml`
- [ ] Set up `pre-commit` with `black`, `ruff`, `isort`
- [ ] Set up GitHub Actions CI (lint + test on push)
- **🌟 Push past your limits:** Use `uv` or `poetry` instead of raw `pip` — learn modern Python packaging.

**📚 Resources:** [Virtual Environments](https://docs.python.org/3/tutorial/venv.html) | [Pre-commit](https://pre-commit.com/) | [GitHub Actions](https://docs.github.com/en/actions/quickstart)

### Phase 1 — Data (Ingestion & Exploration)
- [ ] Choose data source: NASA CMAPSS, AI4I 2020 UCI dataset, or self-written simulator
- [ ] Write a data simulator if going synthetic — inject realistic degradation trajectories
- [ ] Exploratory analysis: plot sensor trends leading up to failures
- [ ] Document the failure modes present in the data
- **🌟 Push past your limits:** Model correlated multi-sensor drift (e.g., vibration + current rising together) — real bearing faults look like this, not random spikes.

### Phase 2 — Labeling (Failure Horizons & RUL)
- [ ] Implement failure-horizon labeling (label = 1 for rows within 7 days before failure)
- [ ] Compute continuous Remaining Useful Life (RUL) target
- [ ] Add embargo gaps around failure boundaries to avoid boundary leakage
- **🌟 Push past your limits:** Support multiple failure modes per machine and label each separately.

### Phase 3 — Leakage-Safe Splitting (Critical!)
- [ ] Implement time-based + machine-grouped train/test split
- [ ] Hold out entire machines the model has never seen
- [ ] Write unit tests: assert no test timestamp precedes latest train timestamp per machine
- **🌟 Push past your limits:** Build a `tests/test_no_leakage.py` suite that would fail if someone reintroduces leakage. Senior-level habit.

### Phase 4 — Feature Engineering (Signal Processing)
- [ ] Rolling statistics (mean, std, min, max, range) at multiple windows
- [ ] Lag features (t-1, t-6, t-24)
- [ ] Rate-of-change / derivative features
- [ ] Cross-sensor ratio features (vibration/RPM, current/voltage)
- [ ] Rolling skewness for asymmetric drift detection
- **🌟 Push past your limits:** Add FFT-based features for vibration — rolling dominant frequency and spectral energy. Real industrial technique, almost nobody attempts it.

### Phase 5 — Layer 1: Anomaly Detection (Classical + Deep)
- [ ] Train Isolation Forest on healthy-only data, per machine or globally
- [ ] Train PyTorch autoencoder as second approach
- [ ] Compare reconstruction error vs. Isolation Forest anomaly score
- [ ] Use anomaly score as both standalone signal and feature into Layer 2
- **🌟 Push past your limits:** Implement autoencoder from scratch (not a library black box) — this is the key question interviewers ask after "I built a neural net."

### Phase 6 — Layer 2: Failure Classification (XGBoost/LightGBM)
- [ ] Train XGBoost and LightGBM classifiers
- [ ] Handle class imbalance with `scale_pos_weight` (not blind oversampling)
- [ ] Tune hyperparameters with time-series-aware cross-validation
- [ ] Track every experiment in MLflow
- **🌟 Push past your limits:** Implement naive baseline (static thresholds) and formally compare ML model vs. baseline in a table. Very senior move.

### Phase 7 — Layer 3: Failure Window Estimation (Survival Analysis)
- [ ] Fit Weibull AFT or Cox Proportional Hazards with `lifelines`
- [ ] Convert point RUL prediction into confidence interval (e.g., "3–6 days")
- [ ] Validate calibration: do 80% intervals contain true failure ~80% of the time?
- **🌟 Push past your limits:** This entire phase *is* the push — graduate-level statistics almost nobody includes. Mastering hazard functions is a genuine skill upgrade.

### Phase 8 — Explainability (SHAP Values)
- [ ] Compute global SHAP feature importance
- [ ] Compute per-prediction SHAP values, map top-3 to sensor name + direction
- [ ] Generate SHAP summary plots for README/report
- **🌟 Push past your limits:** Add SHAP waterfall plot to the app itself for the selected machine.

### Phase 9 — Health Score & Evaluation (Metrics That Matter)
- [ ] Compose health score from failure probability + anomaly score + RUL fraction
- [ ] Evaluate with PR-AUC, recall-at-fixed-precision (not accuracy)
- [ ] Build lead-time histogram — how many days early does the model warn?
- [ ] Per-machine confusion matrices to confirm generalization
- **🌟 Push past your limits:** Justify decision threshold with explicit cost model (cost of missed failure vs. false alarm) instead of defaulting to 0.5.

### Phase 10 — API + App (Backend + Frontend)
- [ ] Build FastAPI backend serving predictions (health score, risk, top signals, failure window)
- [ ] Add Pydantic request validation
- [ ] Build Streamlit dashboard (fast MVP) or React frontend
- [ ] Add rolling sensor trend chart with anomaly onset highlighted
- **🌟 Push past your limits:** Build React + Recharts version instead of stopping at Streamlit — this is "I can script a model" vs. "I can ship a product."

### Phase 11 — Productionization (Docker + CI/CD)
- [ ] Dockerize API and app (`docker-compose.yml` for both)
- [ ] Add GitHub Actions: lint, test, build Docker image on push
- [ ] Add structured logging
- [ ] Write model card documenting assumptions, limitations, intended use
- **🌟 Push past your limits:** Add Evidently AI (or hand-rolled PSI/KL-divergence check) for data drift monitoring. This is what makes it a *system*, not a one-off script.

### Phase 12 — Documentation & Polish (Launch!)
- [ ] Finish README with real metrics once trained
- [ ] Write `docs/why-this-is-hard.md` explaining leakage checklist and design decisions
- [ ] Add demo GIF of running app
- [ ] Write model card / limitations section
- **🌟 Push past your limits:** Record a 2–3 minute Loom/YouTube walkthrough. Recruiters will watch 2 minutes before cloning your repo.

---

---

## Full task list — every phase, in order

**Total project time estimate:**
- **Core (Phases 0–6):** 60–80 hours
- **Core + Advanced (Phases 0–8):** 90–110 hours  
- **Full project (Phases 0–12):** 120–160 hours
- **With all push-your-limits items:** 150–200 hours

Check items off as you go. Each phase has a **core task list** (required) and a **push-your-limits** list (optional, but what turns this from "a project" into "a portfolio centerpiece"). Do at least one push-your-limits item per phase.

### Phase 0 — Setup
- [ ] Initialize repo with `src/`, `data/`, `notebooks/`, `tests/`, `app/`, `docs/`, `assets/`
- [ ] Set up a virtual environment + `requirements.txt` / `pyproject.toml`
- [ ] Set up `pre-commit` with `black`, `ruff`, `isort`
- [ ] Set up GitHub Actions CI (lint + test on push)
- **Push past your limits:** Use `uv` or `poetry` instead of raw `pip` for dependency management — learn modern Python packaging.

### Phase 1 — Data
- [ ] Choose data source: NASA CMAPSS, AI4I 2020 UCI dataset, or a self-written simulator for the 7 sensors
- [ ] Write a data simulator if going synthetic — inject realistic degradation trajectories, not random noise
- [ ] Exploratory analysis: plot sensor trends leading up to each historical failure
- [ ] Document the failure modes present in the data
- **Push past your limits:** Model at least one failure signature as a *correlated multi-sensor drift* (e.g., vibration and current rising together), not an independent spike — this is what real bearing/motor faults look like, and shows you understand the physics, not just the data.

### Phase 2 — Labeling
- [ ] Implement failure-horizon labeling (label = 1 for all rows within 7 days before a failure event)
- [ ] Compute a continuous Remaining Useful Life (RUL) target, capped at a max value
- [ ] Add an "embargo" gap around failure boundaries to avoid boundary leakage
- **Push past your limits:** Support *multiple failure modes* per machine (not just binary failure) and label each separately — closer to how real industrial systems classify failure type, not just failure presence.

### Phase 3 — Leakage-safe splitting
- [ ] Implement time-based + machine-grouped train/test split (never random shuffle)
- [ ] Hold out entire machines the model has never seen
- [ ] Write unit tests that assert no timestamp in test data precedes the latest train timestamp per machine
- **Push past your limits:** Write a small `tests/test_no_leakage.py` suite that would fail loudly if someone (future-you) refactors the pipeline and reintroduces leakage. This is a genuinely rare, senior-level habit.

### Phase 4 — Feature engineering
- [ ] Rolling statistics (mean, std, min, max, range) at multiple window sizes
- [ ] Lag features (t-1, t-6, t-24)
- [ ] Rate-of-change / derivative features
- [ ] Cross-sensor ratio features (vibration/RPM, current/voltage)
- [ ] Rolling skewness for asymmetric drift detection
- **Push past your limits:** Add an FFT-based feature for vibration — rolling dominant frequency and spectral energy. This is a real vibration-analysis technique used in actual industrial condition monitoring, and almost no portfolio project attempts it.

### Phase 5 — Layer 1: Anomaly detection
- [ ] Train an Isolation Forest on healthy-only operating data, per machine or globally
- [ ] Train a simple autoencoder (PyTorch) as a second approach; compare reconstruction error vs. Isolation Forest anomaly score
- [ ] Use the anomaly score as both a standalone signal and a feature into Layer 2
- **Push past your limits:** Learn and implement a basic autoencoder from scratch (not a library black box) — this is the single most common "I built a neural net" resume claim that interviewers actually probe on, so make sure you can explain every layer.

### Phase 6 — Layer 2: Failure classification
- [ ] Train XGBoost and LightGBM classifiers, compare
- [ ] Handle class imbalance with `scale_pos_weight` (not blind oversampling)
- [ ] Tune hyperparameters with time-series-aware cross-validation
- [ ] Track every experiment in MLflow (params, metrics, artifacts)
- **Push past your limits:** Implement a naive baseline (static sensor thresholds) and formally compare it against your ML model in a table — quantifying the value of ML over rules is a very senior move.

### Phase 7 — Layer 3: Failure window estimation
- [ ] Fit a Weibull AFT or Cox Proportional Hazards model with `lifelines`
- [ ] Convert point RUL prediction into a confidence interval (e.g., "3–6 days")
- [ ] Validate calibration: do 80% confidence intervals actually contain the true failure time ~80% of the time?
- **Push past your limits:** This entire phase *is* the push — survival analysis is graduate-level statistics that almost nobody puts in a portfolio project. Understanding hazard functions well enough to explain them in an interview is a genuine skill upgrade.

### Phase 8 — Explainability
- [ ] Compute global SHAP feature importance
- [ ] Compute per-prediction SHAP values, map top-3 to sensor name + direction (↑/↓)
- [ ] Generate SHAP summary plots for the README/report
- **Push past your limits:** Add a SHAP waterfall plot to the app itself for the currently selected machine, not just a static report image.

### Phase 9 — Health score + evaluation
- [ ] Compose the health score from failure probability + anomaly score + RUL fraction
- [ ] Evaluate with PR-AUC, recall-at-fixed-precision (not accuracy)
- [ ] Build the "lead time before failure" histogram — how many days early does the model warn, on true positives?
- [ ] Per-machine confusion matrices to confirm generalization
- **Push past your limits:** Justify your chosen decision threshold with an explicit cost model (cost of a missed failure vs. cost of a false alarm) rather than defaulting to 0.5.

### Phase 10 — API + App
- [ ] Build a FastAPI backend serving predictions (health score, risk, top signals, failure window) per machine
- [ ] Add request validation with Pydantic models
- [ ] Build a Streamlit dashboard (or React frontend) matching the target mockup: gauge, risk bar, top signals, failure window
- [ ] Add a rolling sensor trend chart with anomaly onset highlighted
- **Push past your limits:** Build the React + Recharts version instead of stopping at Streamlit — this is the difference between "I can script a model" and "I can ship a product."

### Phase 11 — Productionization
- [ ] Dockerize the API and the app (`docker-compose.yml` for both)
- [ ] Add GitHub Actions: lint, test, build Docker image on push
- [ ] Add structured logging
- [ ] Write a model card documenting assumptions, limitations, and intended use
- **Push past your limits:** Add Evidently AI (or a hand-rolled PSI/KL-divergence check) for data drift monitoring — detect when incoming sensor distributions start diverging from training data. This is what makes it a *system*, not a one-off script.

### Phase 12 — Documentation & polish
- [ ] Finish this README with real metrics once trained (replace placeholders below)
- [ ] Write `docs/why-this-is-hard.md` explaining the leakage checklist and design decisions
- [ ] Add a demo GIF of the running app to the README
- [ ] Write a short model card / limitations section
- **Push past your limits:** Record a 2–3 minute Loom/YouTube walkthrough and link it at the top of the README — recruiters and interviewers are far more likely to watch 2 minutes than clone and run your repo.

---

---

## ⚠️ Common Pitfalls (What NOT to Do)

| Pitfall | Impact | Fix |
|---|---|---|
| **Random train/test split** | Severe leakage; model sees future data | Always split by **time** and **machine group** |
| **Fitting scaler on full dataset** | Moderate leakage; test metrics too optimistic | Fit on train only, transform test with it |
| **Centered rolling windows** | Subtle leakage; features see future | Use only **backward-looking** (trailing) windows |
| **Skipping class imbalance handling** | High false negatives; misses failures | Use `scale_pos_weight`, not blind oversampling |
| **Defaulting to accuracy as metric** | Wrong decision threshold; poor precision | Use PR-AUC, recall@precision for imbalanced data |
| **No baseline comparison** | Can't justify ML complexity | Always compare against naive baseline |
| **Treating all features equally** | Poor explainability; hard to debug | Use SHAP to understand what drives predictions |
| **Predictions without confidence** | Operations can't trust model | Always return confidence intervals or uncertainty |
| **No validation on held-out machines** | Good metrics on familiar machines, fails in production | Evaluate on machines the model has never seen |
| **Skipping documentation** | Impossible to reproduce; knowledge lost | Document every design decision, trade-off, limitation |

---

## Data leakage checklist (keep this, it's a genuine differentiator)

| Leakage source | Fix |
|---|---|
| Random train/test split shuffles time | Split by time **and** by machine (some machines fully held out) |
| Rolling/lag features computed before splitting | Compute features after splitting, or ensure windows never cross the split boundary |
| Scaler fit on the full dataset | Fit only on train, transform test with it |
| Label window overlapping the split boundary | Add an embargo gap between train and test |
| Centered rolling windows | Only use backward-looking (trailing) windows |
| Resampling applied before splitting | Resample only the training fold |

---

## Results

**Note:** Currently in Phase 0. Metrics will be populated after model training in later phases.

| Metric | Status | Target |
|---|---|---|
| PR-AUC | — | ≥ 0.85 |
| ROC-AUC | — | ≥ 0.90 |
| Recall @ 90% precision | — | ≥ 0.75 |
| Avg. lead time before failure (true positives) | — | 2–5 days |
| Baseline (static threshold) PR-AUC | — | < Model PR-AUC |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- Docker & Docker Compose (optional, for containerization)
- ~5 GB disk space for data

### Quick Setup (2 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/predictive-maintenance-system.git
cd predictive-maintenance-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python -c "import pandas, xgboost, lifelines; print('✓ Setup complete')"
```

### Running the Pipeline (Coming in Phase 1)

Once data simulation is ready (Phase 1):

```bash
# Generate or place data
python src/data/simulate.py

# Run the full pipeline
python src/pipeline.py

# Launch the dashboard (Phase 10)
streamlit run app/dashboard/app.py
```

### Running with Docker (Coming in Phase 11)

```bash
docker-compose up
# Open http://localhost:8000 for API docs
# Open http://localhost:8501 for dashboard
```

### Project Structure

### Project Structure

```
predictive-maintenance-system/
│
├── 📄 README.md                        # This file
├── 📄 LICENSE                          # MIT license
├── 📄 requirements.txt                 # Python dependencies
├── 🐳 docker-compose.yml              # Multi-container orchestration
│
├── 📁 src/                             # Production pipeline code
│   ├── data/                           # Data simulation & ingestion
│   │   ├── simulate.py                 # Synthetic sensor data generator
│   │   └── loader.py                   # Data loading utilities
│   │
│   ├── features/                       # Feature engineering
│   │   ├── rolling_stats.py            # Rolling statistics (mean, std, etc.)
│   │   ├── lags.py                     # Lag feature generation
│   │   ├── rates.py                    # Rate-of-change features
│   │   └── fft.py                      # FFT-based vibration features
│   │
│   ├── labeling/                       # Target creation
│   │   ├── failure_horizon.py          # 7-day failure horizon labels
│   │   └── rul.py                      # Remaining Useful Life targets
│   │
│   ├── models/
│   │   ├── anomaly/                    # Layer 1: Anomaly detection
│   │   │   ├── isolation_forest.py     # Classical approach
│   │   │   └── autoencoder.py          # Deep learning approach
│   │   │
│   │   ├── classifier/                 # Layer 2: Failure classification
│   │   │   ├── xgboost_model.py        
│   │   │   └── lightgbm_model.py       
│   │   │
│   │   └── survival/                   # Layer 3: Failure window estimation
│   │       ├── weibull_aft.py          
│   │       └── cox_ph.py               
│   │
│   ├── explainability/                 # SHAP & interpretation
│   │   ├── shap_analysis.py            
│   │   └── explain_prediction.py       
│   │
│   ├── validation/                     # Leakage guards & cross-validation
│   │   ├── time_series_split.py        # Time-aware splitting
│   │   ├── group_kfold.py              # Machine-grouped splits
│   │   └── leakage_checks.py           # Unit test utilities
│   │
│   ├── health_score.py                 # Composite health score logic
│   └── pipeline.py                     # Orchestrate full pipeline
│
├── 📁 app/                             # Production serving & UI
│   ├── api/                            # FastAPI backend
│   │   ├── main.py                     # API entry point
│   │   ├── models.py                   # Pydantic request/response schemas
│   │   └── routes.py                   # Prediction & health endpoints
│   │
│   └── dashboard/                      # Frontend (Streamlit → React)
│       ├── app.py                      # Streamlit dashboard
│       └── components/                 # Reusable UI components
│
├── 📁 notebooks/                       # Exploration only, no production code
│   └── 01_eda.ipynb                    # Exploratory analysis
│
├── 📁 tests/                           # Unit & integration tests
│   ├── test_no_leakage.py              # Leakage-guard tests
│   ├── test_features.py                # Feature engineering tests
│   └── test_api.py                     # API endpoint tests
│
├── 📁 data/                            # Data (not committed to Git)
│   ├── raw/                            # Raw sensor data
│   ├── processed/                      # Features + labels
│   └── models/                         # Trained model artifacts
│
├── 📁 docs/                            # Project documentation
│   ├── why-this-is-hard.md             # Design decisions & trade-offs
│   ├── architecture.md                 # System design overview
│   └── model-card.md                   # Model assumptions & limitations
│
└── 📁 assets/                          # Images for README
    ├── banner.png                      
    └── architecture.png                
```

---

## Why This Project?

### For Students
- Learn the *complete* ML lifecycle, not just model training
- Understand production constraints (leakage, calibration, explainability)
- Portfolio project that actually demonstrates range

### For Candidates
- Shows you understand industrial applications, not just Kaggle
- Survival analysis + layered architecture = graduate-level technical depth
- Production-ready code (Docker, CI, logging, tests) = hire-able skills

### For Industry
- Real-world problem with real constraints
- Interpretable predictions (SHAP) instead of black boxes
- Deployable system, not a notebook

---

## Code Snippet Examples

### Example 1: Time-Series-Safe Train/Test Split
```python
from sklearn.model_selection import GroupKFold

def train_test_split_no_leakage(df, test_size=0.2):
    """Split by time AND machine — ensures no leakage."""
    df = df.sort_values(['machine_id', 'hour']).reset_index(drop=True)
    
    splitter = GroupKFold(n_splits=int(1/test_size))
    train_idx, test_idx = next(splitter.split(df, groups=df['machine_id']))
    
    # Verify no time leakage
    assert df.loc[test_idx, 'hour'].min() >= df.loc[train_idx, 'hour'].max(), \
        "Time leakage detected!"
    
    return df.iloc[train_idx], df.iloc[test_idx]
```

### Example 2: Failure-Horizon Labeling
```python
def create_failure_labels(df, horizon_days=7, embargo_days=2):
    """Label rows within horizon_days before failure."""
    df = df.copy()
    df['label'] = 0
    
    for machine_id in df['machine_id'].unique():
        machine_data = df[df['machine_id'] == machine_id]
        failure_hour = machine_data['hour'].max()
        
        # Label rows in the failure horizon
        mask = (machine_data['hour'] >= failure_hour - horizon_days * 24) & \
               (machine_data['hour'] <= failure_hour - embargo_days * 24)
        df.loc[mask, 'label'] = 1
    
    return df
```

### Example 3: Rolling Feature Engineering
```python
def create_rolling_features(df, columns=['vibration'], windows=[24, 168, 720]):
    """Create rolling statistics without leakage."""
    for col in columns:
        for window in windows:
            df[f'{col}_mean_{window}h'] = df.groupby('machine_id')[col].rolling(window).mean().reset_index(drop=True)
            df[f'{col}_std_{window}h'] = df.groupby('machine_id')[col].rolling(window).std().reset_index(drop=True)
            df[f'{col}_delta_{window}h'] = df.groupby('machine_id')[col].diff(window)
    
    # IMPORTANT: Drop NaN rows created by rolling windows
    df = df.dropna()
    return df
```

---

## Frequently Asked Questions

**Q: Is this too complicated?**  
A: No, it's deliberately complex in *the right ways*—the parts that matter for real systems. You'll skip the toy stuff and learn what actually separates junior from senior ML engineers. If you're new to ML, start with Phase 0–3 to build foundations, then pick ONE hard phase (Phase 5 or Phase 7) to go deep.

**Q: Can I use real data instead of simulated data?**  
A: Absolutely. NASA CMAPSS and AI4I 2020 are standard benchmarks. Phase 1 supports both. Real data will expose leakage issues more quickly—highly recommended if you have access.

**Q: Do I need to do all 12 phases?**  
A: Not all. **Minimum viable project:** Phases 0–6 (working ML pipeline, 60–80 hrs). **Impressive portfolio:** Phases 0–8 + Phase 10 (add explainability + API, 100–130 hrs). **Production-grade:** Phases 0–12 (full system, 140–180 hrs).

**Q: How long will this take?**  
A: See the time estimates in the roadmap per phase. **Core (0–6):** 60–80 hours. **Core + Advanced (0–8):** 90–110 hours. **Full project (0–12):** 120–160 hours. Add 50% if doing all push-your-limits items.

**Q: What if I'm stuck on a phase?**  
A: Each phase has a core task list (required) and push-your-limits (optional). Complete the core list and move on. You can loop back to push-your-limits later. Also check the "Common Pitfalls" section—90% of issues are in that table.

**Q: Should I do push-your-limits for every phase?**  
A: No. Do at least ONE per phase. Prioritize: Phase 3 (leakage tests), Phase 5 (autoencoder), Phase 7 (survival analysis), Phase 10 (React), Phase 11 (drift monitoring). Doing even 3-4 of these makes your project stand out.

**Q: How do I know if my model is actually good?**  
A: Use these checks: (1) Test on **held-out machines** the model has never seen, (2) Evaluate with **PR-AUC**, not accuracy, (3) Compare against a **naive baseline** (static thresholds), (4) Plot **lead time histogram**—average how many days early does it warn?, (5) Validate **calibration**—do predicted probabilities match actual frequencies?

**Q: What tools should I use for experiment tracking?**  
A: MLflow is built into the tech stack. Alternatives: Weights & Biases (W&B), Neptune, Comet. MLflow is free and self-hosted—start there, upgrade later if needed.

**Q: Can I skip phases and come back later?**  
A: No. Each phase builds on the previous one. You can't do Phase 6 (classification) without Phase 4 (features) and Phase 2 (labels). Do them in order, but you can skip optional items and return to them.

---

## Getting Help

**Stuck on a phase?** Try these in order:
1. Re-read the phase description and core tasks
2. Check the "Common Pitfalls" table above
3. Search the learning resources provided (links per phase)
4. Open an issue with: phase #, what you tried, what went wrong
5. Check issues from other contributors — your problem has likely been solved

**Debugging leakage issues?** This is the #1 source of confusion.
- Always verify: `test_min_timestamp >= train_max_timestamp` per machine
- Use the `test_no_leakage.py` unit test examples in Phase 3
- If in doubt, re-split the data — it's quick

**Model not learning?** Check in this order:
1. Do you have enough training data? (check `df.shape`)
2. Is your label distribution reasonable? (check `df['label'].value_counts()`)
3. Did you drop NaN after feature engineering? (check `df.isna().sum()`)
4. Did you fit scalers/encoders only on train? (check your code)

---

## Resources & Inspiration

- **Survival Analysis:** [Lifelines Documentation](https://lifelines.readthedocs.io/)
- **Data Leakage:** [Kaggle: A Leakage Primer](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/57175)
- **SHAP:** [SHAP GitHub](https://github.com/slundberg/shap)
- **Datasets:** [NASA CMAPSS](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/), [AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance)
- **Time Series CV:** [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

---

## Getting Started

```bash
git clone https://github.com/<your-username>/predictive-maintenance-system.git
cd predictive-maintenance-system
pip install -r requirements.txt

# generate or place data
python src/data/simulate.py

# run the full pipeline
python src/pipeline.py

# launch the app
docker-compose up
```

---

## License

MIT — see [LICENSE](LICENSE).

#   F o r e s i g h t 
 
 
