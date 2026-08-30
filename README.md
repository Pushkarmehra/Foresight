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
  <img alt="Status" src="https://img.shields.io/badge/Status-In%20Progress-yellow">
</p>

---

## What this is

An end-to-end, production-shaped predictive maintenance system for industrial machines. Sensors (temperature, pressure, vibration, RPM, voltage, current, humidity) stream in; the system continuously estimates a **health score**, a **failure probability within the next 7 days**, the **specific signals driving that risk**, and an **estimated failure window** — not just a single black-box number.

This is deliberately built to go past the "load CSV → RandomForest → 94% accuracy" version of this project. See [`docs/why-this-is-hard.md`](docs/why-this-is-hard.md) for the reasoning behind every architectural decision below.

<p align="center">
  <img src="architecture.png" alt="System architecture diagram" width="100%">
</p>

---

## Tech stack (professional, resume-defensible)

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

## Full task list — every phase, in order

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

> Fill in once trained — this section is what recruiters read first.

| Metric | Value |
|---|---|
| PR-AUC | — |
| ROC-AUC | — |
| Recall @ 90% precision | — |
| Avg. lead time before failure (true positives) | — days |
| Baseline (static threshold) PR-AUC | — |

---

## Project structure

```
predictive-maintenance-system/
├── assets/                 banner + diagram images for this README
├── data/                   raw / simulated sensor data (not committed)
├── src/
│   ├── data/                simulation & ingestion
│   ├── features/            rolling stats, lags, FFT, cross-sensor features
│   ├── labeling/             failure-horizon + RUL labeling
│   ├── models/
│   │   ├── anomaly/           Isolation Forest, autoencoder
│   │   ├── classifier/        XGBoost / LightGBM
│   │   └── survival/          Cox PH / Weibull AFT
│   ├── explainability/       SHAP utilities
│   └── health_score.py       score composition logic
├── app/
│   ├── api/                  FastAPI backend
│   └── dashboard/             Streamlit / React frontend
├── notebooks/                exploration only, no pipeline logic here
├── tests/                    including leakage guard tests
├── docs/
│   └── why-this-is-hard.md
├── docker-compose.yml
└── README.md
```

---

## Getting started

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
