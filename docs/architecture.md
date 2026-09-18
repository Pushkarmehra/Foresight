# Foresight System Architecture & Technical Specifications

Foresight is engineered as a multi-tier modular pipeline that decouples data ingestion, feature transformation, layered statistical modeling, and asynchronous inference serving.

```mermaid
flowchart TD
    subgraph Data["1. Telemetry Ingestion"]
        S[7 Sensor Channels: Vib, Temp, Press, RPM, Volt, Curr, Hum] --> B[Trailing Feature Extraction Engine]
    end

    subgraph Features["2. Feature & Signal Processing"]
        B --> F1[Trailing Rolling Stats: Mean, Std, Skew, Energy]
        B --> F2[Lag Features: t-1, t-6, t-24]
        B --> F3[Cross-Sensor Ratios & FFT Harmonics]
    end

    subgraph Modeling["3. Multi-Layer Intelligence Engine"]
        F1 & F2 & F3 --> L1[Layer 1: Anomaly Detection<br/>Isolation Forest + Deep Autoencoder]
        L1 --> L2[Layer 2: Failure Classification<br/>XGBoost / LightGBM Imbalance-Tuned]
        F1 & F2 & F3 --> L3[Layer 3: Survival Analysis<br/>Weibull AFT & Cox PH with 80% CI]
    end

    subgraph Serving["4. Serving & Interpretability"]
        L2 --> SHAP[TreeSHAP Attribution Engine]
        L1 & L2 & L3 & SHAP --> API[FastAPI Microservice]
        API --> UI[Streamlit / React Telemetry Dashboard]
    end
```

---

## Component Details

### 1. Ingestion & Preprocessing (`src/data/`)
- Ingests high-frequency sensor readings across 7 channels.
- Enforces strict trailing-window temporal sequencing without lookahead leakage.

### 2. Feature Engineering (`src/features/`)
- **`rolling_stats.py`**: Trailing rolling mean, standard deviation, skewness, and peak-to-peak amplitude.
- **`lags.py`**: Multi-scale lag vectors capturing short and long-term state velocity.
- **`rates.py`**: First and second-order temporal derivatives (acceleration of degradation).
- **`fft.py`**: Fast Fourier Transform spectral energy and dominant frequency decomposition for rotational vibration telemetry.

### 3. Layered Modeling (`src/models/`)
- **Anomaly Detection (`anomaly/`)**: Baseline deviation scoring trained strictly on nominal operating states.
- **Failure Classification (`classifier/`)**: Gradient boosted trees tuned with `scale_pos_weight` for 7-day failure horizon prediction.
- **Survival Analysis (`survival/`)**: Parametric and semi-parametric time-to-event curves providing confidence intervals for Remaining Useful Life.

### 4. Explainability & Health Scoring (`src/explainability/`, `src/health_score.py`)
- Calculates local Shapley values to pinpoint dominant physical failure mechanisms.
- Synthesizes a calibrated **Health Score (0–100)** for real-time asset monitoring.
