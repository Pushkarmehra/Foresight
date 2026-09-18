# Why Predictive Maintenance is Hard: Engineering & Statistical Realities

Predictive Maintenance (PdM) is frequently oversimplified in academic tutorials as a toy tabular classification problem. Real industrial systems, however, operate under severe physical, statistical, and operational constraints that make standard ML workflows fail in the field.

---

## 1. The Data Leakage Catastrophe

In standard machine learning, random $K$-fold cross-validation or `train_test_split(shuffle=True)` is routine. In industrial time-series telemetry, random splitting is fatal:

- **Temporal Contamination:** Shuffling mixes future telemetry into the training fold. The model learns to interpolate between time steps rather than forecasting forward.
- **Machine State Memorization:** If identical machines appear in both train and test splits across overlapping dates, the model memorizes machine-specific baseline quirks rather than learning generalized degradation dynamics.
- **Rolling Window Lookahead:** Calculating centered rolling averages ($t \pm k$) allows the feature matrix to look into the future.
- **Embargo Windows:** Sensor values immediately preceding catastrophic failure often exhibit chaotic failure dynamics that contaminate boundary decision surfaces. Foresight introduces an *embargo blackout buffer* prior to breakdown.

---

## 2. Binary Classification vs. Time-to-Event (Survival Analysis)

Telling plant operators *"Machine 4 will fail"* is insufficient and often ignored:
- **No Operational Horizon:** Without knowing *when*, operators cannot order replacement bearings or schedule non-disruptive maintenance shifts.
- **Right-Censored Data:** In reality, most machines are decommissioned or serviced *before* complete catastrophic breakdown. Standard regression models cannot handle censored data properly.
- **Uncalibrated Point Estimates:** Point predictions (*"failure in 48 hours"*) fail to communicate epistemic uncertainty. Foresight utilizes **Weibull Accelerated Failure Time (AFT)** models to output calibrated risk distributions with 80% confidence bounds (*"3–6 days"*).

---

## 3. The Class Imbalance Paradox

In a well-maintained facility, failures represent less than 0.1% to 1% of total sensor observations.
- Standard accuracy metric is meaningless (predicting "healthy" yields 99.9% accuracy).
- Blind SMOTE / oversampling generates synthetic vectors that violate physical multi-sensor correlations (e.g. creating high vibration without corresponding RPM harmonic signatures).
- Foresight utilizes cost-sensitive decision thresholds, focal loss, and optimizes for **PR-AUC** and **Recall at High Precision**.

---

## 4. Black-Box Rejection & Physical Interpretability

Operations teams will not shut down a critical continuous production line based on an unexplainable neural network score. Foresight integrates **TreeSHAP** local attributions directly into the serving layer, converting complex feature combinations into physical root-cause diagnoses (*"Vibration +18% drift and current harmonic spike consistent with bearing cage failure"*).
