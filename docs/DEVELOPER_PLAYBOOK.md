# 🛠️ Foresight: Developer Execution Playbook
### *Your Step-by-Step Hands-On Guide to Building the System with AWS & NASA C-MAPSS*

Welcome to your personal developer playbook! This document is designed for **you** to execute step-by-step.
It contains the structural blueprints, class and function contracts, input/output specifications, and formulas so you can **write the code yourself and master every concept**.

---

## 🚦 Progress Tracker

- [x] **Step 1: AWS Setup & S3 Bucket Configuration** (`s3://foresight-predictive-maintenance`)
- [x] **Step 2: Dataset Download** (`src/data/download_cmapss.py`)
- [x] **Step 3: S3 Sync Client** (`src/data/s3_utils.py` — 15 raw files uploaded)
- [ ] **Step 4: Data Ingestion & NASA Schema Loader** (`src/data/loader.py`)
- [ ] **Step 5: Target Labeling & Embargo Windows** (`src/labeling/rul.py`, `src/labeling/failure_horizon.py`)
- [ ] **Step 6: Zero-Leakage Validation Framework** (`src/validation/`, `tests/test_no_leakage.py`)
- [ ] **Step 7: Signal Processing & Feature Engineering** (`src/features/`)
- [ ] **Step 8: Layer 1 Anomaly Detection** (`src/models/anomaly/`)
- [ ] **Step 9: Layer 2 Imbalance-Aware Classifier** (`src/models/classifier/`)
- [ ] **Step 10: Layer 3 Weibull Survival Analysis** (`src/models/survival/`)
- [ ] **Step 11: SHAP Explainability & Health Scoring** (`src/explainability/`, `src/health_score.py`)
- [ ] **Step 12: Master Pipeline Orchestrator** (`src/pipeline.py`)
- [ ] **Step 13: FastAPI Microservice & Streamlit Fleet UI** (`app/api/`, `app/dashboard/`)

---

## 📑 Detailed Architecture Blueprints for Each File

---

### 📂 Phase 1: Data Ingestion & NASA Schema (`src/data/`)

#### 📄 File: `src/data/loader.py`
**Your Goal**: Write functions to load and parse C-MAPSS text files into pandas DataFrames, assigning correct column names and handling S3 or local paths.

**Blueprint & Contracts**:
```python
# Constants you should define:
COLUMNS = [
    'unit_nr', 'time_cycles',
    'setting_1', 'setting_2', 'setting_3',
    'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30',
    'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc',
    'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd',
    'W31', 'W32'
]

# Sensors that have 0.0 variance in FD001:
INVARIANT_SENSORS_FD001 = ['T2', 'P2', 'P15', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd', 'setting_3']

# Active sensors with degrading signals (14 sensors):
ACTIVE_SENSORS_FD001 = ['T24', 'T30', 'T50', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']
```

**Functions to Implement**:
1. `load_raw_train(dataset: str = "FD001", from_s3: bool = False) -> pd.DataFrame`
   - Path: `data/raw/train_{dataset}.txt` (or from S3 key `raw/cmapss/train_{dataset}.txt`).
   - Read using: `pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMNS)`.
2. `load_raw_test(dataset: str = "FD001", from_s3: bool = False) -> pd.DataFrame`
   - Read `test_{dataset}.txt` using the same schema.
3. `load_test_rul(dataset: str = "FD001", from_s3: bool = False) -> pd.DataFrame`
   - Read `RUL_{dataset}.txt`. Columns: `['true_rul']`, with index starting at engine 1 to 100 (`unit_nr`).
4. `get_active_sensors(dataset: str = "FD001") -> list[str]`
   - Returns the list of active sensors with variance $> 0$.

---

### 📂 Phase 2: Target Labeling & Embargo (`src/labeling/`)

#### 📄 File: `src/labeling/rul.py`
**Your Goal**: Calculate the Remaining Useful Life target for every engine cycle.

**Mathematical Logic**:
1. **Training RUL**:
   For each `unit_nr`, find $T_{\max} = \max(\text{time\_cycles})$.
   $$\text{RUL}_{i, t} = T_{\max, i} - t$$
2. **Piecewise-Linear RUL Clipping**:
   During initial healthy flight cycles, degradation hasn't begun. Clip RUL at standard threshold:
   $$\text{RUL}_{\text{clipped}} = \min(\text{RUL}, 125)$$
3. **Test RUL**:
   For test engines, the trajectory stops at $T_{\text{last}}$. Ground truth RUL at $T_{\text{last}}$ is in `RUL_FD001.txt`. Back-project to earlier test cycles:
   $$\text{RUL}_{\text{test}}(t) = \text{True\_RUL}_i + (T_{\text{last}, i} - t)$$

**Functions to Implement**:
- `add_rul_labels(df: pd.DataFrame, max_rul_clip: int = 125) -> pd.DataFrame`
- `add_test_rul_labels(test_df: pd.DataFrame, rul_df: pd.DataFrame, max_rul_clip: int = 125) -> pd.DataFrame`

---

#### 📄 File: `src/labeling/failure_horizon.py`
**Your Goal**: Generate the binary early-warning failure target for classification.

**Mathematical Logic**:
$$\text{failure\_in\_horizon} = \begin{cases} 1 & \text{if } \text{RUL} \le w \\ 0 & \text{if } \text{RUL} > w \end{cases}$$
Where $w = 30\text{ flight cycles}$ (approximately 1–2 weeks of operations).

**Functions to Implement**:
- `add_failure_horizon_label(df: pd.DataFrame, horizon: int = 30, embargo_buffer: int = 0) -> pd.DataFrame`
  - `embargo_buffer`: If $> 0$, drops the final $N$ cycles immediately preceding breakdown to eliminate catastrophic boundary shocks.

---

### 📂 Phase 3: Zero-Leakage Validation Framework (`src/validation/`)

> [!CAUTION]
> In industrial fleet telemetry, you must split by **machine identity (`unit_nr`)**, NEVER by random shuffling.

#### 📄 File: `src/validation/group_kfold.py`
**Your Goal**: Implement an engine-grouped cross-validator.

**Functions to Implement**:
- `get_unit_group_kfold(df: pd.DataFrame, n_splits: int = 5, seed: int = 42)`
  - Uses `sklearn.model_selection.GroupKFold(n_splits=n_splits)`.
  - Groups by `df['unit_nr']`.
  - Yields `(train_indices, val_indices)`.

#### 📄 File: `src/validation/leakage_checks.py`
**Functions to Implement**:
- `assert_no_unit_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame)`:
  - Asserts $\text{set}(\text{train}['unit\_nr']) \cap \text{set}(\text{val}['unit\_nr']) = \emptyset$.
- `assert_temporal_monotonicity(df: pd.DataFrame)`:
  - Asserts `time_cycles` strictly increases per engine.

#### 📄 File: `tests/test_no_leakage.py`
- Write unit tests with `pytest` asserting zero overlap across all 5 folds.
- Run: `pytest tests/test_no_leakage.py -v`.

---

### 📂 Phase 4: Signal Processing & Feature Engineering (`src/features/`)

#### 📄 File: `src/features/rolling_stats.py`
**Your Goal**: Extract trailing rolling statistics across active sensors.

**Rules**:
- Windows: $W \in [5, 10, 20]\text{ cycles}$.
- Metrics: `mean`, `std`, `min`, `max`, `skew`.
- **Zero-Lookahead Rule**: Must group by `unit_nr` and compute rolling stats using trailing windows (`closed='left'` or `.shift(1)`).

#### 📄 File: `src/features/lags.py`
**Your Goal**: Extract historical lag vectors for active sensors.
- Lags: $t-1, t-3, t-5$.
- Computed per `unit_nr`.

#### 📄 File: `src/features/rates.py`
**Your Goal**: Extract rate of degradation / sensor velocity:
$$\text{rate}_k(s) = \frac{s_t - s_{t-k}}{k} \quad (k \in [3, 5])$$

#### Feature Matrix S3 Sync:
Once you extract the complete feature table:
- Save locally to `data/processed/train_features.parquet`.
- Upload using `src/data/s3_utils.py` to `s3://foresight-predictive-maintenance/processed/train_features.parquet`.

---

### 📂 Phase 5: Multi-Layer Modeling Engine (`src/models/`)

#### Layer 1: Anomaly Detection (`src/models/anomaly/`)
1. **`isolation_forest.py`**:
   - Fit `sklearn.ensemble.IsolationForest(contamination=0.05, random_state=42)` **exclusively on healthy cycles** ($\text{RUL} > 100$).
   - Predict normalized anomaly score $[0, 1]$.
2. **`autoencoder.py` (PyTorch)**:
   - Architecture: Linear(14, 32) $\rightarrow$ ReLU $\rightarrow$ Linear(32, 16) $\rightarrow$ Linear(16, 4) $\rightarrow$ Linear(4, 16) $\rightarrow$ Linear(16, 32) $\rightarrow$ Linear(32, 14).
   - Loss: `nn.MSELoss()`.
   - Save checkpoint `autoencoder.pt` and upload to S3 `models/autoencoder.pt`.

#### Layer 2: Imbalance-Aware Classifier (`src/models/classifier/`)
1. **`xgboost_model.py`**:
   - Classifier: `xgboost.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05)`.
   - Set `scale_pos_weight = count(negative) / count(positive)`.
   - Evaluate with `average_precision_score` (PR-AUC) and `roc_auc_score`.
   - Target metrics: **PR-AUC $\ge 0.85$**, **ROC-AUC $\ge 0.90$**.
   - Save model to `data/models/xgboost_model.json` and sync to S3.

#### Layer 3: Survival Analysis & Confidence Intervals (`src/models/survival/`)
1. **`weibull_aft.py`**:
   - Use `lifelines.WeibullAFTFitter(penalizer=0.01)`.
   - Dataset columns: `duration` (cycles), `event` (1 for failure, 0 for censored test units), and top sensor covariates (e.g. $T_{30}, Ps_{30}, BPR$).
   - Predict survival curve $S(t | X)$ and extract **80% Confidence Interval**:
     - Lower bound = $10\text{th}$ percentile survival time.
     - Upper bound = $90\text{th}$ percentile survival time.
     - Output: *"Engine 24 will fail in 34–48 cycles (80% CI)"*.

---

### 📂 Phase 6: Explainability & Health Scoring

#### 📄 File: `src/explainability/shap_analysis.py`
- Initialize `shap.TreeExplainer(xgb_model)`.
- Compute Shapley values for test instances.
- Generate global beeswarm plot `assets/shap_summary.png`.

#### 📄 File: `src/explainability/explain_prediction.py`
- Function: `explain_engine_risk(unit_nr, cycle) -> dict`:
  - Returns top-3 physical drivers:
    *e.g., "High Risk (86%): Driven by HPC exit temp (T30) rising +14°R and static pressure (Ps30) drop, indicating High-Pressure Compressor degradation."*

#### 📄 File: `src/health_score.py`
- Function: `calculate_health_score(failure_prob, anomaly_score, rul_clipped) -> float`:
  $$\text{HealthScore} = 100 \times \left(1 - 0.5 \cdot P(\text{fail}) - 0.3 \cdot \text{AnomalyScore} - 0.2 \cdot \left(1 - \frac{\text{RUL}_{\text{clipped}}}{125}\right)\right)$$
  Clamped between $0$ and $100$.

---

### 📂 Phase 7: Master Pipeline Orchestrator (`src/pipeline.py`)
Build a unified CLI script using `argparse`:
```powershell
python src/pipeline.py --dataset FD001 --use-s3 --train-all
```
Sequence:
1. Load dataset (local or S3).
2. Compute labels and piecewise RUL.
3. Extract trailing features.
4. Execute GroupKFold cross-validation.
5. Train Layer 1, Layer 2, Layer 3.
6. Evaluate against NASA test ground truth.
7. Save models locally and sync to `s3://foresight-predictive-maintenance/models/`.

---

### 📂 Phase 8: Serving Layer & Fleet Dashboard (`app/`)

#### 📄 FastAPI Microservice (`app/api/`)
- `app/api/models.py`: Pydantic input schemas (`SensorSnapshot`) and response schemas (`PredictionResponse`, `HealthResponse`).
- `app/api/routes.py`:
  - `GET /health`
  - `POST /predict`
  - `POST /explain`
- Launch with:
  ```powershell
  uvicorn app.api.main:app --reload --port 8000
  ```

#### 📄 Streamlit Dashboard (`app/dashboard/app.py`)
- Fleet grid showing all 100 engines with color-coded risk (Red / Amber / Green).
- Individual engine selector with degradation line charts ($T_{30}, Ps_{30}, BPR$).
- Survival risk card with 80% CI interval bar and SHAP waterfall chart.
- Launch with:
  ```powershell
  streamlit run app/dashboard/app.py
  ```

---

## 🎯 Your Next Concrete Coding Step

Open [`src/data/loader.py`](file:///c:/Users/pushk/OneDrive/Desktop/AIML/Project_1_ai/src/data/loader.py) and implement:
1. The `COLUMNS` list and active sensor constants.
2. `load_raw_train()`, `load_raw_test()`, and `load_test_rul()`.
3. Test your loader by running:
   ```powershell
   python -c "from src.data.loader import load_raw_train; df = load_raw_train(); print(df.shape, df.head())"
   ```
