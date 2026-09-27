# Kathmandu Air Quality Intelligence

**Statistical analysis, time-series modeling, extreme-event analysis, and 24-hour-ahead PM2.5 forecasting for Kathmandu.**

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://air-quality-intelligence-kathmandu.streamlit.app/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ggNzvyVNOlOV7N-dsJRenPkeBff1CPgy#scrollTo=d76586c8)

---

## Overview

This project analyzes hourly air-pollution and meteorological data for Kathmandu from **September 2022 to April 2025**.

It combines:

- exploratory data analysis
- statistical inference
- time-series diagnostics
- extreme-pollution event analysis
- machine-learning forecasting
- forecast explainability
- interactive visualization

into a reproducible end-to-end workflow.

The integrated dataset contains:

- **23,352 hourly observations**
- **973 daily observations**
- **0 duplicate timestamps**
- **0 missing timestamps**

### Main research questions

1. What temporal patterns exist in Kathmandu air pollution?
2. Which meteorological conditions are associated with pollution?
3. How persistent is PM2.5 over hourly, daily, and weekly timescales?
4. What conditions characterize extreme pollution episodes?
5. Can PM2.5 be forecast 24 hours ahead better than a strong persistence baseline?

---

## Key findings

- **Strong seasonality:** winter had the highest daily PM2.5 concentrations, while monsoon had the lowest. The seasonal effect was large (`epsilon² ≈ 0.43`).
- **Strong temporal persistence:** hourly PM2.5 correlation was approximately **0.965 at 1 hour** and **0.877 at 24 hours**.
- **Weak weekday effect:** weekday/weekend and day-of-week differences were negligible compared with seasonal and diurnal variation.
- **Extreme episodes clustered in winter:** the top-10% PM2.5 threshold was **66.1**, producing **303 episodes**, of which **254** lasted at least two hours.
- The longest top-10% PM2.5 episode lasted **65 hours**.
- **Rainy conditions were associated with fewer extreme observations**, although several raw meteorological relationships weakened after temporal and multivariable adjustment.
- The **24-hour persistence forecast** was a strong forecasting benchmark with test MAE **12.09**.
- A regularized **Ridge regression** model achieved test MAE **11.31**, RMSE **15.03**, and R² **0.777**.
- Ridge improved test MAE by approximately **6.4%** relative to persistence.
- Ridge added the most value when pollution changed substantially over 24 hours, but it still tended to **underpredict the highest PM2.5 concentrations**.

---

## Project workflow

| Phase | Notebook | Purpose |
|:---:|---|---|
| 01 | [`01_data_loading.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/01_data_loading.ipynb) | Load, align, merge, and validate raw hourly datasets |
| 02 | [`02_data_quality_cleaning.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/02_data_quality_cleaning.ipynb) | Data-quality checks, physical validation, and outlier diagnostics |
| 03 | [`03_feature_engineering.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/03_feature_engineering.ipynb) | Calendar/cyclic features, daily aggregation, and particulate AQI |
| 04 | [`04_exploratory_data_analysis.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/04_exploratory_data_analysis.ipynb) | Pollution distributions, temporal patterns, meteorology, and EDA |
| 05 | [`05_time_series_analysis.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/05_time_series_analysis.ipynb) | Lag correlations, ACF/PACF, persistence, and stationarity |
| 06 | [`06_statistical_meteorological_analysis.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/06_statistical_meteorological_analysis.ipynb) | Hypothesis tests, effect sizes, and HAC regression |
| 07 | [`07_extreme_pollution_event_analysis.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/07_extreme_pollution_event_analysis.ipynb) | Extreme episodes, duration, severity, and AQI episodes |
| 08 | [`08_ml_feature_engineering.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/08_ml_feature_engineering.ipynb) | Leakage-safe 24-hour forecasting dataset and chronological splits |
| 09 | [`09_ml_model_training.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/09_ml_model_training.ipynb) | Ridge, Random Forest, HGB, persistence baselines, and final test |
| 10 | [`10_explainability_error_analysis.ipynb`](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/notebooks/10_explainability_error_analysis.ipynb) | Feature importance, subgroup errors, calibration, and residual analysis |

### Notebook hub

Start here:

[**Open `00_START_HERE.ipynb` in Google Colab**](https://colab.research.google.com/drive/1ggNzvyVNOlOV7N-dsJRenPkeBff1CPgy#scrollTo=d76586c8)

The hub provides direct Colab links to all 10 project phases.

---

## Data

### Pollution variables

- PM2.5
- PM10
- NO2
- SO2
- CO

### Meteorological variables

- Temperature
- Relative humidity
- Rainfall
- Wind speed
- Wind direction
- Surface pressure
- Dew point

### Study period

| Property | Value |
|---|---|
| Start | 2022-09-01 00:00 |
| End | 2025-04-30 23:00 |
| Frequency | Hourly |
| Hourly observations | 23,352 |
| Daily observations | 973 |
| Duplicate timestamps | 0 |
| Missing timestamps | 0 |

> **Important:** pollutant unit symbols were mangled in the original exported CSV headers. The project therefore avoids asserting exact pollutant units until the original source metadata is verified.

---

## Methods

The analysis uses:

- descriptive statistics and robust distribution summaries
- Pearson and Spearman correlations
- Kruskal-Wallis tests
- Mann-Whitney tests
- Holm multiple-testing correction
- Benjamini-Hochberg false-discovery-rate correction
- rank-biserial correlation
- epsilon-squared effect size
- Cramér's V
- ACF and PACF analysis
- Augmented Dickey-Fuller test
- KPSS stationarity test
- Ljung-Box diagnostics
- STL/MSTL decomposition
- HAC/Newey-West regression standard errors
- consecutive extreme-event detection
- chronological train/validation/test splitting
- 24-hour purge gaps
- expanding-window `TimeSeriesSplit`
- Ridge regression
- Random Forest regression
- Histogram Gradient Boosting regression
- permutation importance
- post-hoc forecast-error analysis

---

## AQI methodology

The project calculates a **particulate AQI** using daily PM2.5 and PM10.

Current U.S. EPA PM2.5 AQI breakpoints were applied consistently across the study period.

The result is intentionally described as **particulate AQI**, rather than a complete official Kathmandu AQI, because gas-pollutant unit metadata and required averaging periods were not sufficiently documented in the source export.

---

## Time-series findings

PM2.5 showed strong temporal dependence.

Selected hourly lag correlations:

| Lag | Correlation |
|---:|---:|
| 1 hour | 0.965 |
| 3 hours | 0.848 |
| 6 hours | 0.666 |
| 12 hours | 0.461 |
| 24 hours | 0.877 |
| 48 hours | 0.801 |
| 72 hours | 0.753 |
| 168 hours | 0.687 |

The strong rebound at the 24-hour lag demonstrates an important daily persistence pattern.

The raw daily PM2.5 series showed evidence of non-stationarity, while first differencing produced a series consistent with stationarity.

---

## Extreme pollution analysis

The study used percentile-based PM2.5 thresholds:

| Threshold | PM2.5 |
|---|---:|
| 90th percentile | 66.1 |
| 95th percentile | 83.6 |

For the top-10% threshold:

- **2,354 extreme hours**
- **303 episodes**
- **254 sustained episodes lasting at least 2 hours**
- **median duration: 6 hours**
- **mean duration: 7.77 hours**
- **maximum duration: 65 hours**

Extreme episode starts were strongly concentrated in winter.

---

## Forecasting design

The forecasting task is:

> Predict PM2.5 at **t + 24 hours** using only information available at or before forecast origin **t**.

### Forecast features

Features include:

- current pollutant observations
- current meteorology
- PM2.5 lags:
  - 1 hour
  - 3 hours
  - 6 hours
  - 12 hours
  - 24 hours
  - 48 hours
  - 72 hours
  - 168 hours
- rolling PM2.5 means
- rolling PM2.5 standard deviations
- PM2.5 change and momentum features
- co-pollutant history
- recent meteorological summaries
- calendar variables known in advance

Actual future observed weather is **not used**.

The data are split chronologically with **24-hour purge gaps** between training, validation, and test periods.

---

## Model comparison

### Cross-validation

| Model | CV MAE | CV RMSE | CV R² |
|---|---:|---:|---:|
| Persistence | **7.339** | 9.796 | 0.451 |
| Random Forest | 7.575 | **9.624** | **0.471** |
| HistGradientBoosting | 8.274 | 10.479 | 0.373 |
| Ridge | 8.587 | 10.682 | 0.333 |

Persistence remained extremely competitive during model development.

### Validation and test results

| Model / Split | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence — Validation | **7.212** | **10.912** | **0.801** |
| Ridge — Validation | 7.224 | 11.120 | 0.793 |
| Persistence — Test | 12.085 | 16.284 | 0.738 |
| **Ridge — Test** | **11.314** | **15.028** | **0.777** |

The held-out test period was substantially more polluted than the development period.

Although Ridge did not outperform persistence during validation, the frozen Ridge model generalized better during the shifted high-pollution test regime.

### Forecast skill on test

- **MAE improvement:** approximately **6.4%**
- **RMSE improvement:** approximately **7.7%**
- **R²:** increased from **0.738 to 0.777**

---

## Explainability and forecast errors

Post-hoc analysis showed that recent pollution dynamics dominated predictive performance.

Important predictive inputs included:

- current PM2.5
- PM2.5 lag 1 hour
- PM2.5 lag 24 hours
- PM2.5 change over 6 hours
- PM10 lag 24 hours
- recent rolling PM2.5 behavior
- selected co-pollutant features

### When Ridge helped most

Ridge's advantage increased when PM2.5 changed substantially between the forecast origin and the next 24 hours.

| Absolute 24h PM2.5 change | Ridge MAE skill vs persistence |
|---|---:|
| 0–5 | -116.0% |
| 5–10 | +6.7% |
| 10–20 | +14.2% |
| 20–40 | +16.3% |
| 40+ | +15.5% |

This suggests that persistence is highly effective under stable conditions, while Ridge provides more value when pollution changes meaningfully.

### Main forecasting limitation

Ridge showed regression toward the mean:

- lower pollution levels tended to be overpredicted
- very high pollution levels tended to be underpredicted

Therefore, improved overall MAE did **not** completely solve the problem of forecasting pollution peaks.

---

## Interactive dashboard

Explore the full project visually:

### [Launch the live dashboard](https://air-quality-intelligence-kathmandu.streamlit.app/)

Dashboard sections include:

- Overview
- Temporal Patterns
- Meteorology & Statistical Analysis
- Extreme Pollution Events
- 24-Hour Forecasting
- Explainability & Error Analysis
- Methods & Data Quality

---

## Repository structure

```text
Kathmandu-Air-Quality-Intelligence/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── ml/
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_quality_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_exploratory_data_analysis.ipynb
│   ├── 05_time_series_analysis.ipynb
│   ├── 06_statistical_meteorological_analysis.ipynb
│   ├── 07_extreme_pollution_event_analysis.ipynb
│   ├── 08_ml_feature_engineering.ipynb
│   ├── 09_ml_model_training.ipynb
│   └── 10_explainability_error_analysis.ipynb
│
├── dashboard/
│   └── app.py
│
├── models/
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── docs/
│   └── Kathmandu_Air_Quality_Intelligence_Detailed_Report.pdf
│
├── requirements.txt
└── README.md
```

---

## Run locally

Clone the repository:

```bash
git clone https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence.git
cd Kathmandu-Air-Quality-Intelligence
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in numerical order from the `notebooks/` directory.

Run the dashboard with:

```bash
streamlit run dashboard/app.py
```

---

## Google Colab

The notebooks can be opened directly from GitHub in Google Colab.

Start from the project hub:

### [Open the complete notebook guide in Google Colab](https://colab.research.google.com/drive/1ggNzvyVNOlOV7N-dsJRenPkeBff1CPgy#scrollTo=d76586c8)

The hub links to all 10 analysis phases.

For full reproduction, follow:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10
```

---

## Reproducibility notes

- Do not use random train/test splitting for the forecasting task.
- Do not fit scalers or other learned preprocessing on the complete dataset.
- Keep the final held-out test period locked during model selection.
- High-pollution observations are retained; statistical outliers are not automatically invalid data.
- Meteorological relationships are observational associations and should not be described as causal.
- Processed timestamps are timezone-naive; local-hour interpretations depend on verifying the original timestamp timezone.
- Phase 10 is post-hoc error analysis and should not be used to retune the frozen model.

---

## Limitations

Important limitations include:

1. **Pollutant unit metadata**  
   Unit symbols were mangled in the original exported pollutant headers.

2. **Timestamp timezone**  
   Processed timestamps are timezone-naive, so literal local-hour interpretation requires verification of the original source timezone.

3. **AQI scope**  
   The project calculates particulate AQI from PM2.5 and PM10 rather than a complete official multi-pollutant AQI.

4. **Observational meteorology**  
   Meteorological associations do not establish causality.

5. **Extreme-value forecasting**  
   Ridge improves overall test performance but still tends to underpredict the highest PM2.5 concentrations.

6. **Residual temporal dependence**  
   Forecast residuals remain autocorrelated, indicating additional temporal information may still be exploitable.

---

## Detailed report

A detailed project report covering methodology, statistical results, modeling decisions, limitations, and reproducibility is available in the repository:

[**Kathmandu Air Quality Intelligence — Detailed Report**](https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence/blob/main/docs/Kathmandu_Air_Quality_Intelligence_Detailed_Report.pdf)

---

## Future work

Potential extensions include:

- verification and restoration of original pollutant-unit metadata
- verification of timestamp timezone
- integration of forecasted meteorological variables
- regime-specific forecasting models
- residual autoregressive correction
- probabilistic or quantile forecasting
- sequence-based forecasting methods
- improved high-pollution event prediction
- external validation on newer Kathmandu observations

---

## Project links

- **Repository:** https://github.com/rubeshnpl13/Kathmandu-Air-Quality-Intelligence
- **Live Dashboard:** https://air-quality-intelligence-kathmandu.streamlit.app/
- **Google Colab Hub:** https://colab.research.google.com/drive/1ggNzvyVNOlOV7N-dsJRenPkeBff1CPgy#scrollTo=d76586c8

---

## License

This project is licensed under the MIT License. You can use, modify, and reuse the code.

### Data

The data used in this project may be subject to the terms and conditions of its original data providers. Please refer to the respective source for information about data licensing and redistribution.
