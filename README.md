<div align="center">

  <img src="banner.png" alt="Foresight - Industrial AI & Cloud Predictive Maintenance System Banner" width="100%" />

  # ⚙️ Foresight — Industrial AI & Cloud Predictive Maintenance System

  ### *Will this aircraft turbofan engine fail in the next 30 flight cycles? — and if so, when, why, and how confident are we?*

  <p align="center">
    <a href="https://github.com/Pushkarmehra/Foresight/stargazers"><img src="https://img.shields.io/github/stars/Pushkarmehra/Foresight?style=for-the-badge&color=ffd700&logo=star" alt="GitHub Stars"></a>
    <a href="https://github.com/Pushkarmehra/Foresight/network/members"><img src="https://img.shields.io/github/forks/Pushkarmehra/Foresight?style=for-the-badge&color=00c7b7" alt="GitHub Forks"></a>
    <a href="https://github.com/Pushkarmehra/Foresight/issues"><img src="https://img.shields.io/github/issues/Pushkarmehra/Foresight?style=for-the-badge&color=ff6b6b" alt="Open Issues"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-4ecdc4?style=for-the-badge" alt="License"></a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Storage-AWS%20S3-FF9900?style=flat-square&logo=amazons3&logoColor=white" alt="AWS S3">
    <img src="https://img.shields.io/badge/Classification-XGBoost%20%7C%20LightGBM-EB5424?style=flat-square&logo=xgboost&logoColor=white" alt="XGBoost">
    <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Survival%20Analysis-lifelines-FFA500?style=flat-square" alt="Lifelines">
    <img src="https://img.shields.io/badge/Explainability-SHAP-008080?style=flat-square" alt="SHAP">
    <img src="https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/MLOps-MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white" alt="MLflow">
    <img src="https://img.shields.io/badge/Container-Docker-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
    <img src="https://img.shields.io/badge/Status-Phase%200%20(Foundation)-blue?style=flat-square" alt="Status">
  </p>

  <p align="center">
    <a href="#-quick-navigation">Quick Navigation</a> •
    <a href="#-the-problem">The Problem</a> •
    <a href="#-system-architecture">Architecture</a> •
    <a href="#-key-differentiators">Differentiators</a> •
    <a href="#-tech-stack">Tech Stack</a> •
    <a href="#-project-roadmap">Roadmap</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-project-structure">Project Structure</a> •
    <a href="#-faq">FAQ</a>
  </p>

</div>

---

## 🧭 Quick Navigation

- [⚡ Quick Start Decision Tree](#-quick-start-decision-tree) — Choose your learning track
- [🔍 The Problem](#-the-problem) — Why industrial predictive maintenance is hard
- [💡 What This Is](#-what-this-is) — Production-shaped ML system overview
- [🏗️ System Architecture](#️-system-architecture) — Multi-layer intelligence pipeline
- [⭐ Key Differentiators](#-key-differentiators) — What makes this production-grade
- [🛠️ Tech Stack](#️-tech-stack) — Professional, resume-defensible tooling
- [🛡️ Data Leakage Integrity Checklist](#️-data-leakage-integrity-checklist) — Non-negotiable safety guards
- [📈 Benchmark & Evaluation](#-benchmark--evaluation) — Target metrics vs. baselines
- [🗺️ Project Roadmap](#️-project-roadmap) — Comprehensive 12-phase roadmap
- [🚀 Getting Started](#-getting-started) — Run locally in 2 minutes
- [📂 Project Structure](#-project-structure) — Enterprise folder architecture
- [💻 Code Highlights](#-code-highlights) — Critical implementation patterns
- [⚠️ Common Pitfalls](#️-common-pitfalls) — Top 10 traps to avoid
- [❓ FAQ](#-faq) — Frequently asked questions
- [📄 License](#-license) — MIT License

---

## ⚡ Quick Start Decision Tree

Select your learning trajectory based on your background and objective:

```text
Are you a...

📚 STUDENT learning ML end-to-end?
   ├── Follow: Phases 0 → 6 (Core Foundation: 60–80 hrs)
   ├── Deepen: Pick ONE from Phase 5 (Autoencoders) or Phase 7 (Survival Analysis)
   └── Outcome: Strong full-lifecycle ML competency

💼 CANDIDATE building a standout portfolio?
   ├── Follow: Phases 0 → 8 (Core + Explainability: 90–120 hrs)
   ├── Ship: Push Phase 10 (Full-Stack / React Dashboard)
   └── Outcome: Resume centerpiece demonstrating production-level breadth & depth

🏭 ENGINEER deploying a real-world system?
   ├── Follow: Phases 0 → 11 (Production-Ready: 120–160 hrs)
   ├── Fortify: Implement data drift monitoring & complete Phase 12 documentation
   └── Outcome: Battle-tested, zero-leakage, containerized microservice

⚡ FAST TRACK (Weekend Accelerator)?
   ├── Follow: Phases 0 → 1 (Data) → 2 (Labels) → 6 (XGBoost) → 10 (FastAPI Serving)
   └── Time: 20–30 hrs (Skip deep survival modeling & drift monitoring)
```

---

## 🔍 The Problem

Industrial machines do not fail out of nowhere—equipment degrades gradually across multi-modal sensor telemetry: vibration signatures, thermal dynamics, hydraulic pressure, RPM fluctuations, and electrical draw.

Standard machine learning approaches fail in production for three critical reasons:

1. **Binary predictions are actionable-deficient:** Telling operations *"failure = True"* without estimating **when** leads to premature shutdowns or catastrophic downtime.
2. **Black-box models are rejected by operators:** Field engineers will not take a multimillion-dollar assembly offline without knowing **why** the model triggered an alarm.
3. **Data leakage creates deceptive validation metrics:** Random train/test splits, lookahead rolling windows, and uncalibrated risk scores result in models that boast 98% accuracy in notebooks but fail catastrophically in the field.

### The Three Questions We Answer

```mermaid
flowchart LR
    A[Telemetry Stream] --> B(Foresight ML Engine)
    B --> C["1. Will it fail?<br/><b>Calibrated Failure Probability</b>"]
    B --> D["2. When will it fail?<br/><b>Confidence-Bounded Time Horizon</b>"]
    B --> E["3. Why is it failing?<br/><b>SHAP-Based Physical Root Cause</b>"]
```

---

## 💡 What This Is

**Foresight** is an end-to-end, production-grade predictive maintenance platform integrated with **AWS S3** and validated on high-fidelity aerospace turbofan engine simulations (**NASA C-MAPSS**). As high-frequency telemetry streams across critical aircraft engine channels (*High-Pressure Compressor exit temperature $T_{30}$, static pressure $Ps_{30}$, core speed $N_c$, bypass ratio $BPR$, bleed enthalpy, vibration*), the system continuously estimates:

- **System Vitality Gauge (0–100):** Real-time composite equipment vitality index reflecting degradation across thermal, mechanical, and aerodynamic subsystems.
- **Failure Horizon Probability:** Calibrated probability of failure within the next 30 flight cycles (~1–2 operational weeks).
- **Estimated Failure Window:** Statistical confidence bounds (*e.g., "34–48 cycles at 80% CI"* rather than an uncalibrated naive point estimate).
- **Physical Root Cause Attribution:** Transparent, SHAP-derived explanations translated into plain aeronautical engineering diagnostics (*e.g., "High Risk (86%): Driven by HPC exit temperature (T30) rising +14°R and static pressure drop, indicating compressor blade erosion"*).
- **Cloud-Native Data Lake:** Seamless integration with **Amazon S3** (`s3://foresight-predictive-maintenance/`) for raw archives, Parquet feature stores, and automated model registry tracking.

> [!NOTE]
> This repository is engineered specifically to surpass typical "toy" tutorials (*CSV → Random Forest → 95% Accuracy*). It enforces industrial aerospace constraints: zero data leakage, Weibull survival analysis, and production microservice packaging.

---

## 🏗️ System Architecture

<div align="center">
  <img src="architecture.png" alt="Foresight System Architecture Diagram" width="100%" />
</div>

The system is constructed as a decoupled, multi-layer intelligence hierarchy:

| Layer | Component | Methodology | Primary Purpose |
|---|---|---|---|
| **Cloud Storage** | **AWS S3 Data Lake** | Amazon S3 + `boto3` Integration | Centralized repository for raw NASA C-MAPSS archives, Parquet feature stores, and serialized models. |
| **Layer 1** | **Anomaly Detection** | Isolation Forest & PyTorch Deep Autoencoder | Detect subtle deviations from nominal healthy baselines; outputs unsupervised anomaly scores. |
| **Layer 2** | **Failure Classification** | XGBoost & LightGBM with Imbalance Tuning | Classifies imminent failure risks within rolling 7-day / 30-cycle lookahead windows. |
| **Layer 3** | **Time-to-Event Survival** | Cox Proportional Hazards & Weibull AFT | Computes Remaining Useful Life (RUL) distributions with statistically rigorous 80% confidence intervals. |
| **Layer 4** | **Explainability & API** | TreeSHAP & FastAPI Microservice | Translates complex model features into actionable sensor telemetry insights served over async REST endpoints. |

---

## ⭐ Key Differentiators

What makes this project **production-grade** and **resume-defensible**:

### 1. 📊 Survival Analysis for Confidence-Bounded Windows
Most projects estimate a naive point estimate (*"failure in 72 hours"*). Foresight uses **Weibull Accelerated Failure Time (AFT)** and **Cox Proportional Hazards** from `lifelines` to output calibrated uncertainty intervals (*"3–6 days at 80% confidence"*).

### 2. 🛡️ Uncompromising Data Leakage Guards
- **Strict Time-Series & Group Splitting:** Machines in the test set are completely held-out or split temporally.
- **Embargo Gaps:** Failure boundaries enforce blackout periods to prevent label-leakage contamination.
- **Trailing-Only Windows:** Rolling transformations look strictly backward in time.
- **Leakage Test Suite:** Dedicated `tests/test_no_leakage.py` runs in CI to catch regression risks.

### 3. 🔬 Explainability by Design (SHAP)
Predictions are paired with **TreeSHAP** feature attribution. Operators don't just see *"High Risk (84%)"*; they receive structured physical insights:
> *"Risk elevated due to +18% vibration drift and concurrent current spikes, indicating bearing race degradation."*

### 4. 🧩 Layered & Swappable Modularity
Each layer operates independently. Anomaly scores feed downstream classifiers as engineered features, yet remain accessible as standalone early-warning signals.

### 5. 🚢 Industrial MLOps & Production Instrumentation
- **MLflow Tracking:** Logs parameters, metrics, confusion matrices, and model artifacts.
- **FastAPI Backend:** Fully typed with Pydantic schemas and OpenAPI documentation.
- **Docker Compose:** One-command orchestration for backend APIs and dashboards.
- **Automated CI/CD:** GitHub Actions test pipeline verifying zero-leakage constraints on every push.

---

## 🛠️ Tech Stack

| Layer | Tools | Architectural Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Modern typing, performance optimizations, and broad ML ecosystem support. |
| **Cloud Storage** | AWS S3 (`boto3`, `s3fs`) | Centralized data lake for raw C-MAPSS archives, Parquet feature tables, and model registry. |
| **Data Engine** | Pandas, Polars, NumPy | Polars enables high-throughput streaming and out-of-core sensor window aggregations. |
| **Anomaly Detection** | scikit-learn `IsolationForest`, PyTorch | Dual-paradigm approach comparing classical tree ensembles against deep reconstruction error. |
| **Classification** | XGBoost, LightGBM | Gradient boosted decision trees optimized for tabular sensor time-series features. |
| **Survival Analysis** | `lifelines` (Weibull AFT, Cox PH) | Yields true probability distributions and hazard curves rather than arbitrary heuristics. |
| **Explainability** | SHAP (`shap`) | Provides mathematically grounded Shapley attribution for local and global interpretability. |
| **Experiment Tracking** | MLflow | Structured experiment logging, artifact tracking, and model registry management. |
| **Validation & Safety** | `TimeSeriesSplit`, `GroupKFold` | Strict temporal and group validation preventing cross-machine data leakage. |
| **Serving** | FastAPI, Pydantic | Asynchronous, typed, high-performance REST API with automated OpenAPI docs. |
| **Frontend** | Streamlit *(MVP)* → React + Recharts | Live operational dashboard visualizing equipment telemetry, risk gauges, and SHAP waterfalls. |
| **Containerization** | Docker, Docker Compose | Fully reproducible, hermetic development and deployment environments. |
| **Testing & CI** | pytest, GitHub Actions | Automated leakage testing, feature validation, and linting on every commit. |
| **Monitoring (Stretch)** | Evidently AI | Production drift detection monitoring sensor distribution shifts and model degradation. |

---

## 🛡️ Data Leakage Integrity Checklist

> [!IMPORTANT]
> Data leakage is the #1 reason predictive maintenance systems fail when transitioning from notebooks to physical production plants. Foresight strictly enforces the following rules:

| Vulnerability | Production Risk | Foresight Engineering Solution |
|---|---|---|
| **Random Train/Test Shuffle** | Future machine states leak into training sets. | Split strictly by **time** and **machine grouping** (`GroupKFold` / `TimeSeriesSplit`). |
| **Global Preprocessing Scalers** | Test distribution statistics leak into scalers. | Fit scalers and transformers **only on train data**; transform test data downstream. |
| **Centered Rolling Windows** | Feature calculations peek into future timesteps. | Enforce strictly **trailing (backward-looking)** windows (`closed='left'`). |
| **Boundary Label Contamination** | Sensor features overlap with physical failure points. | Introduce **embargo blackout gaps** between training windows and failure events. |
| **Premature Resampling** | Synthetic samples duplicate into validation splits. | Resample / rebalance **only within training folds**. |

---

## 📈 Benchmark & Evaluation

> [!NOTE]
> Currently in **Phase 0 (Foundation)**. Live evaluation metrics will be updated upon completion of training phases.

| Evaluation Metric | Baseline (Static Threshold) | Foresight Target | Production Significance |
|---|---|---|---|
| **PR-AUC** | ~0.42 | **≥ 0.85** | Primary metric for severely imbalanced failure events. |
| **ROC-AUC** | ~0.65 | **≥ 0.90** | Discriminative power across all operational thresholds. |
| **Recall @ 90% Precision** | < 0.30 | **≥ 0.75** | Ensures alerts are actionable with fewer than 10% false alarms. |
| **Mean Early Warning Lead Time** | < 12 Hours | **2–5 Days** | Gives maintenance crews sufficient runway to procure parts and schedule downtime. |
| **Interval Calibration (80% CI)** | N/A | **75%–85%** | Verifies that survival confidence bounds reflect actual empirical failure intervals. |

---

## 🗺️ Project Roadmap

```
Phase 0: Setup & Tooling       [████░░░░░░░░░░░░] 20%
Phase 1-4: Data & Features     [░░░░░░░░░░░░░░░░]  0%
Phase 5-8: ML & Survival       [░░░░░░░░░░░░░░░░]  0%
Phase 9-12: App & Production   [░░░░░░░░░░░░░░░░]  0%
```

### Phase 0 — Environment & Engineering Foundation
**⏱️ Estimate: 2–4 hours** | **Difficulty: Beginner**
- [ ] Initialize repository structure (`src/`, `data/`, `notebooks/`, `tests/`, `app/`, `docs/`, `assets/`)
- [ ] Configure virtual environment and dependencies via `requirements.txt` / `pyproject.toml`
- [ ] Set up pre-commit formatting hooks (`black`, `ruff`, `isort`)
- [ ] Establish GitHub Actions CI pipeline (automated linting and pytest runners)
- **🌟 Push Past Your Limits:** Adopt `uv` or `poetry` for ultra-fast, deterministic dependency resolution.

### Phase 1 — Data Ingestion & Physical Degradation Simulation
**⏱️ Estimate: 6–10 hours** | **Difficulty: Intermediate**
- [ ] Ingest standard benchmarks (NASA CMAPSS, AI4I 2020) or develop synthetic telemetry generator
- [ ] Engineer multi-sensor degradation profiles with realistic noise and operational regime shifts
- [ ] Conduct Exploratory Data Analysis (EDA) plotting sensor trends leading to physical breakdown
- [ ] Document specific failure mode physics (e.g., bearing spalling, thermal runaway, motor winding burnout)
- **🌟 Push Past Your Limits:** Implement correlated multi-sensor drift signatures (*e.g., vibration harmonics rising concurrently with amperage draw*).

### Phase 2 — Failure Horizons & Remaining Useful Life (RUL)
**⏱️ Estimate: 4–6 hours** | **Difficulty: Intermediate**
- [ ] Implement lookahead failure-horizon labeling (`label = 1` for rows within 7 days of failure)
- [ ] Compute piecewise-linear Remaining Useful Life (RUL) targets with max-life clipping
- [ ] Apply embargo blackout buffers around failure events to eliminate boundary leakage
- **🌟 Push Past Your Limits:** Support multi-class failure modes, labeling failure types independently.

### Phase 3 — Zero-Leakage Validation Framework
**⏱️ Estimate: 4–6 hours** | **Difficulty: Advanced**
- [ ] Implement temporal and machine-grouped cross-validation splits
- [ ] Reserve hold-out validation machines never observed during training
- [ ] Write unit tests verifying that no test timestamp precedes training timestamps
- **🌟 Push Past Your Limits:** Build `tests/test_no_leakage.py` with invariant assertions to guard against regression during pipeline refactoring.

### Phase 4 — Feature Engineering & Signal Processing
**⏱️ Estimate: 8–12 hours** | **Difficulty: Advanced**
- [ ] Extract multi-window rolling statistics (mean, std, skewness, min/max, energy)
- [ ] Compute lag features ($t-1, t-6, t-24$) and discrete rates of change
- [ ] Formulate cross-sensor ratio indicators (e.g., Vibration / RPM, Current / Voltage)
- **🌟 Push Past Your Limits:** Implement FFT spectral analysis to extract rolling dominant frequencies and spectral energy distributions from vibration telemetry.

### Phase 5 — Layer 1: Anomaly Detection (Classical & Deep)
**⏱️ Estimate: 8–12 hours** | **Difficulty: Intermediate**
- [ ] Train Isolation Forest baseline exclusively on healthy operational data
- [ ] Construct a PyTorch deep autoencoder for reconstruction error analysis
- [ ] Benchmark Isolation Forest anomaly scores against autoencoder reconstruction loss
- [ ] Propagate anomaly scores as engineered features into Layer 2 classifiers
- **🌟 Push Past Your Limits:** Implement the autoencoder from scratch with custom PyTorch training loops, loss functions, and latent space visualizations.

### Phase 6 — Layer 2: Imbalance-Aware Classification
**⏱️ Estimate: 8–12 hours** | **Difficulty: Intermediate**
- [ ] Train and optimize XGBoost and LightGBM binary classification models
- [ ] Mitigate extreme class imbalance using `scale_pos_weight` and focal loss
- [ ] Hyperparameter tune models using time-series cross-validation
- [ ] Log parameters, metrics, ROC curves, and artifacts in MLflow
- **🌟 Push Past Your Limits:** Benchmark model performance against a rule-based threshold baseline and compute the exact financial value-add table.

### Phase 7 — Layer 3: Survival Analysis & Time-to-Event
**⏱️ Estimate: 10–14 hours** | **Difficulty: Graduate Level**
- [ ] Fit Weibull Accelerated Failure Time (AFT) and Cox Proportional Hazards models with `lifelines`
- [ ] Transform point RUL estimates into statistical confidence intervals (*e.g., "3–6 days (80% CI)"*)
- [ ] Validate interval calibration across test machine degradation lifecycles
- **🌟 Push Past Your Limits:** Derive hazard functions and survival curves dynamically conditioned on real-time sensor state vectors.

### Phase 8 — Model Interpretability (TreeSHAP)
**⏱️ Estimate: 6–8 hours** | **Difficulty: Intermediate**
- [ ] Compute global TreeSHAP feature importance summaries
- [ ] Compute per-prediction local SHAP attributions and map top-3 features to plain English
- [ ] Generate beeswarm and dependency plots for documentation
- **🌟 Push Past Your Limits:** Integrate interactive SHAP waterfall plots into the application interface for live machine diagnostics.

### Phase 9 — Composite Health Scoring & Economic Evaluation
**⏱️ Estimate: 6–8 hours** | **Difficulty: Advanced**
- [ ] Formulate a composite Health Score ($0-100$) combining failure probability, anomaly score, and RUL fraction
- [ ] Evaluate with PR-AUC and Recall@Precision metrics
- [ ] Plot early-warning lead-time distributions on true positive failure events
- **🌟 Push Past Your Limits:** Optimize the classification decision threshold against an explicit industrial cost matrix (*cost of unpredicted failure vs. cost of false alarm maintenance inspection*).

### Phase 10 — Serving & Interactive Dashboard
**⏱️ Estimate: 10–16 hours** | **Difficulty: Advanced**
- [ ] Build FastAPI microservice delivering real-time inferences with Pydantic validation
- [ ] Implement Streamlit operational dashboard featuring gauges, risk meters, and telemetry plots
- **🌟 Push Past Your Limits:** Build a modern React + Recharts frontend dashboard communicating with the FastAPI backend over WebSockets/REST.

### Phase 11 — Productionization, Docker & Monitoring
**⏱️ Estimate: 8–12 hours** | **Difficulty: Advanced**
- [ ] Containerize services with multi-stage Dockerfiles and `docker-compose.yml`
- [ ] Implement structured JSON logging and health check endpoints
- [ ] Draft a comprehensive Model Card documenting assumptions, performance bounds, and operating envelopes
- **🌟 Push Past Your Limits:** Deploy Evidently AI to detect real-time feature drift and covariate shift in incoming telemetry streams.

### Phase 12 — Documentation, Walkthrough & Final Release
**⏱️ Estimate: 4–6 hours** | **Difficulty: Beginner**
- [ ] Finalize README with empirical benchmarks and model evaluation figures
- [ ] Complete `docs/why-this-is-hard.md` detailing architectural trade-offs
- [ ] Embed demo GIFs and video walkthroughs
- **🌟 Push Past Your Limits:** Record a 3-minute technical walkthrough video highlighting architecture, zero-leakage engineering, and live dashboard inference.

---

## 🚀 Getting Started

### Prerequisites
- **Python:** Version `3.11` or newer
- **Git**
- **Docker & Docker Compose** *(optional, for containerized execution)*

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/Pushkarmehra/Foresight.git
cd Foresight

# Create and activate virtual environment
python -m venv venv
# On macOS / Linux:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure AWS S3 Credentials & Verify Installation

Create or populate your local `.env` file at the root of the project:

```ini
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=foresight-predictive-maintenance
```

Verify core libraries and cloud connectivity:

```bash
python -c "import boto3, pandas, xgboost, lifelines, fastapi, streamlit; print('✓ All core dependencies & AWS client ready!')"
```

### 3. Ingest Data & Sync to AWS S3

```bash
# 1. Download official NASA C-MAPSS turbofan benchmark dataset
python src/data/download_cmapss.py

# 2. Upload raw telemetry archives to Amazon S3 Data Lake
python src/data/s3_utils.py

# 3. Execute end-to-end feature extraction and model training
python src/pipeline.py --dataset FD001 --use-s3 --train-all

# 4. Launch FastAPI inference server
uvicorn app.api.main:app --reload --port 8000

# 5. Launch interactive turbofan fleet operations dashboard
streamlit run app/dashboard/app.py
```

### 4. Run with Docker Compose *(Available in Phase 11)*

```bash
docker-compose up --build
# REST API Documentation: http://localhost:8000/docs
# Interactive Dashboard:   http://localhost:8501
```

---

## 📂 Project Structure

```text
Foresight/
├── 📄 README.md                        # Project documentation & master roadmap
├── 📄 LICENSE                          # MIT open-source license
├── 📄 requirements.txt                 # Pinned project dependencies
├── 🐳 docker-compose.yml              # Multi-container orchestration (API + Dashboard)
│
├── 📁 src/                             # Core production source code
│   ├── data/                           # Ingestion, AWS S3 sync & benchmarks
│   │   ├── download_cmapss.py          # NASA C-MAPSS dataset downloader & extractor
│   │   ├── s3_utils.py                 # AWS S3 upload/download & sync client (boto3)
│   │   ├── loader.py                   # NASA C-MAPSS parser & stream loader
│   │   └── simulate.py                 # Telemetry simulator with physical failure modes
│   │
│   ├── features/                       # Signal processing & feature engineering
│   │   ├── rolling_stats.py            # Trailing rolling statistics (mean, std, skew)
│   │   ├── lags.py                     # Temporal lag extraction (t-1, t-6, t-24)
│   │   ├── rates.py                    # Rate-of-change and acceleration derivatives
│   │   └── fft.py                      # Fast Fourier Transform spectral energy features
│   │
│   ├── labeling/                       # Target generation & embargo logic
│   │   ├── failure_horizon.py          # 7-day lookahead horizon labeling
│   │   └── rul.py                      # Continuous Remaining Useful Life calculation
│   │
│   ├── models/                         # Multi-layer modeling engine
│   │   ├── anomaly/                    # Layer 1: Anomaly detection
│   │   │   ├── isolation_forest.py     # Classical isolation forest estimator
│   │   │   └── autoencoder.py          # PyTorch deep autoencoder
│   │   │
│   │   ├── classifier/                 # Layer 2: Imbalance-aware classification
│   │   │   ├── xgboost_model.py        # XGBoost classifier with custom weighting
│   │   │   └── lightgbm_model.py       # LightGBM classifier
│   │   │
│   │   └── survival/                   # Layer 3: Time-to-event survival modeling
│   │       ├── weibull_aft.py          # Weibull Accelerated Failure Time model
│   │       └── cox_ph.py               # Cox Proportional Hazards model
│   │
│   ├── explainability/                 # SHAP interpretation & root-cause mapping
│   │   ├── shap_analysis.py            # Global & local TreeSHAP attribution
│   │   └── explain_prediction.py       # Physical explanation generator
│   │
│   ├── validation/                     # Leakage prevention & validation splits
│   │   ├── time_series_split.py        # Temporal cross-validation
│   │   ├── group_kfold.py              # Machine-grouped fold splitting
│   │   └── leakage_checks.py           # Invariant validation assertions
│   │
│   ├── health_score.py                 # Composite equipment health scoring engine
│   └── pipeline.py                     # Master end-to-end execution orchestrator
│
├── 📁 app/                             # Serving layer & user interfaces
│   ├── api/                            # FastAPI microservice
│   │   ├── main.py                     # Application entry point
│   │   ├── models.py                   # Pydantic request/response data contracts
│   │   └── routes.py                   # Health, prediction, and explanation routes
│   │
│   └── dashboard/                      # UI Dashboard
│       ├── app.py                      # Streamlit dashboard application
│       └── components/                 # Reusable telemetry charts & gauge widgets
│
├── 📁 notebooks/                       # Research & exploratory notebooks (non-prod)
│   └── 01_eda.ipynb                    # Sensor trend exploration & degradation profiles
│
├── 📁 tests/                           # Automated test suite
│   ├── test_no_leakage.py              # Strict data leakage regression tests
│   ├── test_features.py                # Feature calculation & trailing window tests
│   └── test_api.py                     # API contract & validation tests
│
├── 📁 data/                            # Local data store (git-ignored)
│   ├── raw/                            # Raw telemetry archives
│   ├── processed/                      # Extracted feature tables & labels
│   └── models/                         # Serialized model binaries (.json, .pt, .pkl)
│
├── 📁 docs/                            # Deep technical architecture documentation
│   ├── why-this-is-hard.md             # Design decisions & production trade-offs
│   ├── architecture.md                 # System diagrams and dataflow specifications
│   └── model-card.md                   # Model limitations and operational envelopes
│
└── 📁 assets/                          # Static image assets & architecture diagrams
    ├── banner.png                      # Hero banner graphic
    └── architecture.png                # System topology visual
```

---

## 💻 Code Highlights

### 1. Leakage-Free Temporal & Machine-Grouped Splitting

```python
import pandas as pd
from sklearn.model_selection import GroupKFold

def train_test_split_no_leakage(df: pd.DataFrame, test_size: float = 0.2):
    """
    Partitions telemetry by machine groupings and strictly enforces temporal ordering
    to guarantee zero lookahead leakage.
    """
    df_sorted = df.sort_values(by=['machine_id', 'timestamp']).reset_index(drop=True)
    
    n_splits = int(1.0 / test_size)
    splitter = GroupKFold(n_splits=n_splits)
    
    train_idx, test_idx = next(splitter.split(df_sorted, groups=df_sorted['machine_id']))
    train_df, test_df = df_sorted.iloc[train_idx], df_sorted.iloc[test_idx]
    
    # Assert zero cross-set timestamp contamination for individual machines
    for m_id in set(train_df['machine_id']).intersection(set(test_df['machine_id'])):
        max_train_time = train_df[train_df['machine_id'] == m_id]['timestamp'].max()
        min_test_time = test_df[test_df['machine_id'] == m_id]['timestamp'].min()
        assert min_test_time > max_train_time, f"Temporal leakage detected on machine {m_id}!"
        
    return train_df, test_df
```

### 2. Failure-Horizon Labeling with Embargo Protection

```python
import pandas as pd

def apply_failure_horizon_labels(
    df: pd.DataFrame, 
    horizon_hours: int = 168,  # 7 days
    embargo_hours: int = 48    # 2 days blackout buffer
) -> pd.DataFrame:
    """
    Generates binary classification targets for failure horizons while applying
    an embargo buffer immediately preceding catastrophic breakdown.
    """
    labeled_df = df.copy()
    labeled_df['target_failure'] = 0
    
    for machine_id, group in labeled_df.groupby('machine_id'):
        failure_timestamp = group[group['is_failed'] == 1]['timestamp'].min()
        if pd.isna(failure_timestamp):
            continue
            
        horizon_start = failure_timestamp - pd.Timedelta(hours=horizon_hours)
        embargo_start = failure_timestamp - pd.Timedelta(hours=embargo_hours)
        
        # Label active risk window
        risk_mask = (labeled_df['machine_id'] == machine_id) & \
                    (labeled_df['timestamp'] >= horizon_start) & \
                    (labeled_df['timestamp'] < embargo_start)
        labeled_df.loc[risk_mask, 'target_failure'] = 1
        
        # Blackout embargo rows to avoid boundary confusion
        embargo_mask = (labeled_df['machine_id'] == machine_id) & \
                       (labeled_df['timestamp'] >= embargo_start) & \
                       (labeled_df['timestamp'] <= failure_timestamp)
        labeled_df.loc[embargo_mask, 'target_failure'] = -1  # Filtered prior to training
        
    return labeled_df[labeled_df['target_failure'] != -1].reset_index(drop=True)
```

---

## ⚠️ Common Pitfalls

| # | Anti-Pattern | Operational Consequence | Engineering Remedy |
|---|---|---|---|
| 1 | **Random Shuffle Splits** | 99% test accuracy in notebook; complete failure in production due to temporal lookahead leakage. | Always split strictly by **time** and **machine identifier**. |
| 2 | **Global Scaling on Full Dataset** | Test set distribution parameters leak into feature normalization statistics. | Fit scalers **exclusively on train fold**; apply `.transform()` on test fold. |
| 3 | **Centered Rolling Windows** | Future sensor values inadvertently bleed into feature calculations. | Enforce strictly **trailing / backward-looking** rolling calculations. |
| 4 | **Blind SMOTE Oversampling** | Creates synthetic tabular noise that breaks physical multi-sensor correlations. | Tune `scale_pos_weight` in GBDTs and use focal loss functions. |
| 5 | **Optimizing for Standard Accuracy** | Predicts "healthy" 100% of time on 99:1 imbalanced datasets, missing all failures. | Evaluate with **PR-AUC**, **F-beta ($\beta=2$)**, and **Recall@Precision**. |
| 6 | **Point-Estimate RUL Without Uncertainty** | Operators lack confidence context on whether failure is in 2 hours or 2 days. | Fit **Weibull AFT / Cox survival models** to output calibrated prediction intervals. |
| 7 | **Absence of Baseline Benchmarks** | Inability to prove whether complex deep learning outperforms simple static threshold rules. | Formally benchmark against a **naive statistical baseline rule** in a comparison matrix. |
| 8 | **Uninterpretable Black-Box Inferences** | Maintenance personnel ignore alerts because root cause cannot be verified. | Provide **SHAP value feature attributions** translated into physical engineering units. |
| 9 | **Testing Only on Seen Machines** | Model memorizes specific machine signatures rather than learning true degradation physics. | Evaluate generalization on **completely held-out machine IDs**. |
| 10 | **Notebook-Only Delivery** | System cannot be integrated into existing SCADA / industrial IoT networks. | Deliver as a **containerized FastAPI REST service** with OpenAPI contracts. |

---

## ❓ FAQ

<details>
<summary><b>Q: Is this system too complex for an initial ML portfolio project?</b></summary>
<br>
<b>A:</b> It is designed to be challenging in the <i>exact areas that real employers look for</i>. Instead of building another generic classification script, Foresight teaches zero-leakage temporal validation, survival uncertainty intervals, and production containerization. If you are starting out, complete <b>Phases 0–6</b> first, then choose one advanced phase (such as <b>Phase 7: Survival Analysis</b>) to dive deep.
</details>

<details>
<summary><b>Q: Can I train this on real industrial sensor datasets?</b></summary>
<br>
<b>A:</b> Yes. Foresight is architected to accept standard real-world benchmarks including <b>NASA CMAPSS Turbofan</b> and the <b>UCI AI4I 2020 Predictive Maintenance Dataset</b>, as well as synthetic multi-sensor streams generated in Phase 1.
</details>

<details>
<summary><b>Q: Why is Survival Analysis preferred over standard regression for RUL?</b></summary>
<br>
<b>A:</b> Standard regression models output a single deterministic number without confidence boundaries and struggle with right-censored data (machines that haven't failed yet). Survival analysis explicitly handles censoring and outputs full probability distributions, allowing operations teams to plan maintenance around calibrated risk confidence bands (e.g. <i>"80% probability machine will fail between day 3 and day 6"</i>).
</details>

<details>
<summary><b>Q: How are predictions served in real-time?</b></summary>
<br>
<b>A:</b> Telemetry payloads are submitted via HTTP POST requests to the <b>FastAPI microservice</b> (`/predict/health`). The backend executes the multi-layer pipeline: evaluating anomaly scores, computing failure horizon probabilities, generating SHAP explanations, and estimating RUL intervals before returning a typed JSON response.
</details>

<details>
<summary><b>Q: Do I have to complete all 12 phases?</b></summary>
<br>
<b>A:</b> Not necessarily. You can select your goal:
<ul>
  <li><b>Core ML Pipeline:</b> Phases 0–6 (60–80 hours)</li>
  <li><b>Portfolio Showcase:</b> Phases 0–8 + Phase 10 (90–120 hours)</li>
  <li><b>Full Production Platform:</b> Phases 0–12 (120–160 hours)</li>
</ul>
</details>

---

## 👥 Contributing & Community

Contributions, issues, and feature requests are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for complete terms.

<div align="center">
  <sub>Built with precision for reliable industrial intelligence. Developed by <a href="https://github.com/Pushkarmehra">Pushkar Mehra</a>.</sub>
</div>
