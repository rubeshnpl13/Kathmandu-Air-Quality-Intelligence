# Kathmandu Air Quality Intelligence — Streamlit Dashboard

This dashboard visualizes the outputs from the project's data-quality, EDA, time-series,
statistical, extreme-event, and 24-hour PM2.5 forecasting phases.

## Expected repository structure

Place `app.py` here:

```text
kathmandu-air-quality-intelligence/
├── dashboard/
│   └── app.py
├── data/
│   └── processed/
│       ├── air_quality_features.csv
│       ├── air_quality_daily.csv
│       └── ml/
│           ├── train.csv
│           ├── validation.csv
│           └── test.csv
├── models/
│   └── phase9/
│       └── ridge_24h.joblib
└── reports/
    └── tables/
        ├── phase5_time_series/
        ├── phase6_statistics/
        ├── phase7_extreme_events/
        ├── phase8_ml_preparation/
        ├── phase9_models/
        └── phase10_explainability/
```

The dashboard treats most Phase 5–10 result tables as optional. Core processed hourly/daily
CSVs are required. The Forecasting page can reconstruct test predictions from
`models/phase9/ridge_24h.joblib` if the Phase 9 prediction CSV has not been saved.

## Install

From the repository root:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r dashboard/requirements.txt
```

## Run

```bash
streamlit run dashboard/app.py
```

## Dashboard sections

- **Overview** — study KPIs, daily PM2.5, seasonality, particulate AQI
- **Temporal Patterns** — diurnal/monthly patterns, season-hour heatmap, persistence, stationarity
- **Meteorology & Statistics** — interactive pollutant/weather relationships, correlations, rain/wind, HAC results
- **Extreme Events** — top-10%/top-5% thresholds, episode durations, season rates, wind/rain, AQI episodes
- **Forecasting** — frozen Ridge vs persistence on the held-out test period, regime/change/season performance
- **Explainability & Errors** — Ridge coefficients, permutation importance, calibration, residuals, worst misses
- **Methods & Data Quality** — reproducibility notes, quality summary, limitations

## Important interpretation notes

- The original pollutant unit symbols were mangled in the raw CSV headers, so the dashboard
  avoids asserting exact pollutant concentration units until source metadata is verified.
- Processed timestamps are timezone-naive; clock-hour interpretation uses timestamps as supplied.
- AQI shown in the project is a calculated **particulate AQI** based on PM2.5 and PM10, not a
  claimed full official Kathmandu multi-pollutant AQI.
- Meteorological relationships are observational associations, not causal effects.
- Phase 10 explainability is post-hoc; it does not modify or retune the frozen forecasting model.
