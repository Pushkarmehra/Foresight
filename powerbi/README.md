# 📊 Foresight: Power BI Fleet Analytics & Executive Dashboard

This folder contains assets, schemas, DAX formulas, and reports for the **Foresight Turbofan Fleet Monitoring** Power BI dashboard.

---

## 📁 Directory Structure
```
powerbi/
├── README.md               # Power BI setup guide & dashboard layout blueprints
├── dax_measures.md         # Ready-to-use DAX measures for fleet KPIs & health metrics
├── reports/                # Place exported .pbix files and templates here
└── data_exports/           # Staging folder for Power BI-optimized CSV/Parquet extracts
```

---

## 🔌 Data Source Connections

### 1. Local Processed Telemetry
* Connect Power BI via **Get Data -> Text/CSV** or **Folder**:
  * Point to: `../data/processed/train_FD001_processed.csv`
  * Or use all 4 subsets: `train_FD001_processed.csv` through `train_FD004_processed.csv`.

### 2. Live Cloud Connection (AWS S3)
* Direct connection to processed S3 bucket:
  * `s3://foresight-predictive-maintenance/processed/`

---

## 🎨 Recommended Dashboard Layout

### 1. Executive Fleet Summary (Page 1)
* **Top KPI Cards**:
  * Total Monitored Turbofans
  * Active Failure Alerts ($RUL \le 30$)
  * Fleet Average Remaining Useful Life (RUL)
  * Average Engine Health Score ($0 - 100$)
* **Fleet Matrix Grid**:
  * Engine ID (`unit_nr`), Current Flight Cycle, Health Status (🔴 Critical, 🟡 Maintenance Scheduled, 🟢 Normal), Predicted RUL.
* **Risk Distribution Donut Chart**:
  * Engine distribution by risk category.

### 2. Engine Deep-Dive & Physics Degradation (Page 2)
* **Engine Selector Slicer** (`unit_nr`).
* **Degradation Line Charts**:
  * High-Pressure Compressor Outlet Temp ($T_{30}$) vs. Flight Cycle.
  * Static Pressure ($Ps_{30}$) vs. Flight Cycle.
  * Bypass Ratio ($BPR$) vs. Flight Cycle.
* **Countdown Gauge**:
  * Remaining Useful Life countdown toward failure threshold.
