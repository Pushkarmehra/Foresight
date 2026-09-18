# Foresight Model Card & Operational Envelopes

## 1. Model Overview
- **Name:** Foresight Multi-Tier Predictive Maintenance Engine
- **Version:** `0.1.0-alpha`
- **Architecture:** Layered Ensemble (Isolation Forest + Autoencoder $\rightarrow$ XGBoost/LightGBM $\rightarrow$ Weibull Survival AFT)
- **Primary Task:** Continuous asset health scoring, 7-day failure horizon prediction, and failure window estimation with 80% confidence intervals.

---

## 2. Intended Use & Target Operational Scope
- **Intended Assets:** Industrial rotating equipment (pumps, compressors, turbofan turbines, high-load electric motors).
- **Required Telemetry Inputs:** 7-channel multi-sensor stream (Sampling frequency $\ge$ 1 reading/hour):
  1. Vibration ($g$ / mm/s RMS)
  2. Temperature ($^\circ\text{C}$)
  3. Pressure ($\text{bar}$ / $\text{PSI}$)
  4. Rotational Velocity ($\text{RPM}$)
  5. Voltage ($V$)
  6. Current ($A$)
  7. Ambient Humidity ($\%$)

---

## 3. Performance Metrics & Validation Target Envelopes
- **PR-AUC:** $\ge 0.85$ (Primary operational target on held-out machinery)
- **ROC-AUC:** $\ge 0.90$
- **Recall @ 90% Precision:** $\ge 0.75$
- **Early Warning Lead Time:** $2\text{--}5\text{ days}$ average advance warning prior to catastrophic failure.

---

## 4. Out-of-Scope & Known Operational Boundaries
- **Sudden Impact / Instant Foreign Object Damage (FOD):** Model detects progressive degradation (bearing wear, fatigue, thermal runaway) and is not intended for instant catastrophic impact events.
- **Uncalibrated Sensor Drift:** Sensor failures resulting in disconnected telemetry must be handled by upstream data validation guards.
