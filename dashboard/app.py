from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Kathmandu Air Quality Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "tables"
MODELS_DIR = PROJECT_ROOT / "models" / "phase9"

PHASE5_DIR = REPORTS_DIR / "phase5_time_series"
PHASE6_DIR = REPORTS_DIR / "phase6_statistics"
PHASE7_DIR = REPORTS_DIR / "phase7_extreme_events"
PHASE8_DIR = REPORTS_DIR / "phase8_ml_preparation"
PHASE9_DIR = REPORTS_DIR / "phase9_models"
PHASE10_DIR = REPORTS_DIR / "phase10_explainability"

SEASON_ORDER = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
AQI_ORDER = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]
WIND_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    [data-testid="stMetricValue"] {font-size: 1.75rem;}
    .aqi-note {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 0.8rem;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0 1rem 0;
        background: rgba(128,128,128,.06);
    }
    .small-note {font-size: 0.88rem; opacity: .78;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def load_core_data():
    hourly_path = PROCESSED_DIR / "air_quality_features.csv"
    daily_path = PROCESSED_DIR / "air_quality_daily.csv"

    if not hourly_path.exists() or not daily_path.exists():
        return None, None

    hourly = pd.read_csv(hourly_path, parse_dates=["time"], index_col="time")
    daily = pd.read_csv(daily_path, parse_dates=["time"], index_col="time")

    return hourly.sort_index(), daily.sort_index()


@st.cache_data(show_spinner=False)
def load_optional_tables():
    paths = {
        # Phase 5
        "hourly_lags": PHASE5_DIR / "hourly_pm25_lag_correlations.csv",
        "daily_lags": PHASE5_DIR / "daily_pm25_lag_correlations.csv",
        "seasonal_persistence": PHASE5_DIR / "seasonal_pm25_persistence.csv",
        "stationarity": PHASE5_DIR / "stationarity_tests.csv",
        "ljung_raw": PHASE5_DIR / "ljung_box_daily_raw.csv",
        "ljung_diff": PHASE5_DIR / "ljung_box_daily_differenced.csv",

        # Phase 6
        "season_summary": PHASE6_DIR / "seasonal_pm25_summary.csv",
        "season_pairwise": PHASE6_DIR / "season_pairwise_tests.csv",
        "met_spearman": PHASE6_DIR / "meteorological_spearman_tests.csv",
        "hourly_hac": PHASE6_DIR / "hourly_hac_regression.csv",
        "daily_hac": PHASE6_DIR / "daily_hac_regression.csv",
        "vif": PHASE6_DIR / "meteorological_vif.csv",

        # Phase 7
        "episode_summary": PHASE7_DIR / "episode_summary.csv",
        "top10_episodes": PHASE7_DIR / "top10_pm25_episodes.csv",
        "top5_episodes": PHASE7_DIR / "top5_pm25_episodes.csv",
        "episode_season": PHASE7_DIR / "episode_rate_by_season.csv",
        "extreme_wind": PHASE7_DIR / "extreme_rate_by_wind_sector.csv",
        "extreme_rain": PHASE7_DIR / "extreme_rate_by_rain_status.csv",
        "extreme_weather": PHASE7_DIR / "extreme_weather_tests.csv",
        "extreme_odds": PHASE7_DIR / "extreme_logistic_odds_ratios.csv",
        "aqi100": PHASE7_DIR / "aqi_over_100_episodes.csv",
        "aqi150": PHASE7_DIR / "aqi_over_150_episodes.csv",

        # Phase 9
        "validation_predictions": PHASE9_DIR / "validation_predictions.csv",
        "test_predictions": PHASE9_DIR / "test_predictions.csv",
        "cv_comparison": PHASE9_DIR / "cv_model_comparison.csv",
        "selection": PHASE9_DIR / "model_selection.csv",
        "test_results": PHASE9_DIR / "final_test_results.csv",
        "generalization": PHASE9_DIR / "generalization_summary.csv",

        # Phase 10
        "ridge_coefficients": PHASE10_DIR / "ridge_standardized_coefficients.csv",
        "permutation": PHASE10_DIR / "permutation_importance.csv",
        "bias": PHASE10_DIR / "forecast_bias_summary.csv",
        "error_season": PHASE10_DIR / "forecast_error_by_season.csv",
        "error_month": PHASE10_DIR / "forecast_error_by_month.csv",
        "error_hour": PHASE10_DIR / "forecast_error_by_hour.csv",
        "error_regime": PHASE10_DIR / "forecast_error_by_pollution_regime.csv",
        "error_change": PHASE10_DIR / "forecast_error_by_24h_change.csv",
        "high_pollution_detection": PHASE10_DIR / "extreme_detection_metrics.csv",
        "calibration": PHASE10_DIR / "ridge_calibration.csv",
        "ridge_ljung": PHASE10_DIR / "ridge_error_ljung_box.csv",
        "test_error": PHASE10_DIR / "test_error_analysis.csv",
    }

    tables = {}
    for key, path in paths.items():
        try:
            tables[key] = pd.read_csv(path) if path.exists() else None
        except Exception:
            tables[key] = None
    return tables


@st.cache_data(show_spinner=False)
def load_ml_splits():
    ml_dir = PROCESSED_DIR / "ml"
    result = {}
    for name in ["train", "validation", "test"]:
        path = ml_dir / f"{name}.csv"
        if path.exists():
            result[name] = pd.read_csv(
                path,
                parse_dates=["time", "target_time"],
                index_col="time",
            ).sort_index()
        else:
            result[name] = None
    return result


@st.cache_resource(show_spinner=False)
def load_ridge_model():
    path = MODELS_DIR / "ridge_24h.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Pre-Monsoon"
    if month in (6, 7, 8, 9):
        return "Monsoon"
    return "Post-Monsoon"


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
    }


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["time", "target_time", "start_time", "end_time", "peak_time", "start_date", "end_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def display_missing_table_message(label: str):
    st.info(
        f"{label} is not available yet. Re-run the corresponding notebook's "
        "save-output cells and refresh the dashboard."
    )


hourly, daily = load_core_data()
tables = load_optional_tables()
splits = load_ml_splits()

if hourly is None or daily is None:
    st.error(
        "Core processed files were not found. Expected:\n"
        "- data/processed/air_quality_features.csv\n"
        "- data/processed/air_quality_daily.csv"
    )
    st.stop()

# Normalize useful categories safely.
if "season" in hourly.columns:
    hourly["season"] = pd.Categorical(hourly["season"], SEASON_ORDER, ordered=True)
if "season" in daily.columns:
    daily["season"] = pd.Categorical(daily["season"], SEASON_ORDER, ordered=True)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("Kathmandu AQI Intelligence")
page = st.sidebar.radio(
    "Explore",
    [
        "Overview",
        "Temporal Patterns",
        "Meteorology & Statistics",
        "Extreme Events",
        "Forecasting",
        "Explainability & Errors",
        "Methods & Data Quality",
    ],
)

st.sidebar.divider()
st.sidebar.caption(
    "Study period: "
    f"{hourly.index.min():%d %b %Y} – {hourly.index.max():%d %b %Y}"
)
st.sidebar.caption(f"{len(hourly):,} hourly observations • {len(daily):,} daily observations")


# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------

if page == "Overview":
    st.title("Kathmandu Air Quality Intelligence")
    st.caption(
        "Statistical analysis, time-series structure, extreme-event analysis, "
        "and 24-hour-ahead PM2.5 forecasting."
    )

    pm25_p90 = hourly["pm25"].quantile(0.90)
    aqi_over_100 = int(daily["aqi_over_100"].sum()) if "aqi_over_100" in daily else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hourly observations", f"{len(hourly):,}")
    c2.metric("Daily observations", f"{len(daily):,}")
    c3.metric("Mean PM2.5", f"{hourly['pm25'].mean():.1f}")
    c4.metric("PM2.5 90th percentile", f"{pm25_p90:.1f}")
    c5.metric("Days particulate AQI > 100", f"{aqi_over_100:,}" if pd.notna(aqi_over_100) else "—")

    st.markdown(
        """
        <div class="aqi-note">
        <b>Dashboard scope.</b> The AQI displayed here is the project's calculated
        <b>particulate AQI</b>, based on daily PM2.5 and PM10 using the documented
        U.S. EPA particulate breakpoints. It is not presented as a complete official
        Kathmandu AQI because gas-pollutant units/averaging periods were not sufficiently
        documented in the source export.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Daily PM2.5 through time")
    overview_daily = daily.copy()
    overview_daily["PM2.5 (daily mean)"] = overview_daily["pm25_mean"]
    overview_daily["30-day mean"] = overview_daily["pm25_mean"].rolling(30, min_periods=10).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=overview_daily.index,
            y=overview_daily["PM2.5 (daily mean)"],
            name="Daily mean",
            mode="lines",
            opacity=0.42,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=overview_daily.index,
            y=overview_daily["30-day mean"],
            name="30-day mean",
            mode="lines",
            line=dict(width=3),
        )
    )
    fig.update_layout(
        hovermode="x unified",
        yaxis_title="PM2.5 concentration",
        xaxis_title=None,
        legend_title=None,
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Seasonal PM2.5")
        season_plot = (
            daily.groupby("season", observed=False)["pm25_mean"]
            .agg(["mean", "median"])
            .reindex(SEASON_ORDER)
            .reset_index()
        )
        fig = go.Figure()
        fig.add_bar(x=season_plot["season"], y=season_plot["mean"], name="Mean")
        fig.add_bar(x=season_plot["season"], y=season_plot["median"], name="Median")
        fig.update_layout(
            barmode="group",
            yaxis_title="Daily PM2.5",
            xaxis_title=None,
            legend_title=None,
            height=370,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Particulate AQI categories")
        if "particle_aqi_category" in daily:
            counts = (
                daily["particle_aqi_category"]
                .value_counts()
                .reindex(AQI_ORDER, fill_value=0)
                .reset_index()
            )
            counts.columns = ["category", "days"]
            fig = px.bar(
                counts,
                x="days",
                y="category",
                orientation="h",
                labels={"days": "Days", "category": ""},
            )
            fig.update_layout(height=370, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Research findings at a glance")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            "**Strong seasonality**  \n"
            "Winter daily PM2.5 is far higher than monsoon conditions, while "
            "weekday/weekend differences are negligible."
        )
    with k2:
        st.markdown(
            "**Strong persistence**  \n"
            "Hourly PM2.5 is highly autocorrelated, with a pronounced 24-hour cycle "
            "and persistent day-to-day structure."
        )
    with k3:
        st.markdown(
            "**Forecasting result**  \n"
            "Persistence is extremely competitive, but the frozen Ridge model "
            "generalizes better in the high-pollution test regime."
        )


# -----------------------------------------------------------------------------
# Temporal Patterns
# -----------------------------------------------------------------------------

elif page == "Temporal Patterns":
    st.title("Temporal Patterns & Time-Series Structure")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Diurnal", "Calendar", "Persistence", "Stationarity"]
    )

    with tab1:
        st.subheader("Hourly PM2.5 profile")
        hourly_profile = (
            hourly.groupby("hour")["pm25"]
            .agg(["mean", "median"])
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_profile["hour"], y=hourly_profile["mean"], mode="lines+markers", name="Mean"))
        fig.add_trace(go.Scatter(x=hourly_profile["hour"], y=hourly_profile["median"], mode="lines+markers", name="Median"))
        fig.update_layout(
            xaxis=dict(dtick=1),
            xaxis_title="Hour",
            yaxis_title="PM2.5",
            hovermode="x unified",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Season × hour heatmap")
        season_hour = (
            hourly.groupby(["season", "hour"], observed=False)["pm25"]
            .mean()
            .unstack("hour")
            .reindex(SEASON_ORDER)
        )
        fig = px.imshow(
            season_hour,
            aspect="auto",
            labels=dict(x="Hour", y="Season", color="Mean PM2.5"),
        )
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Hour-of-day interpretation assumes timestamps are already expressed in the intended local time. "
            "The source timezone should be documented before treating clock-hour patterns as definitive Kathmandu local time."
        )

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Monthly PM2.5")
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            month_profile = hourly.groupby("month")["pm25"].agg(["mean", "median"]).reindex(range(1, 13)).reset_index()
            month_profile["month_name"] = month_labels
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=month_profile["month_name"], y=month_profile["mean"], mode="lines+markers", name="Mean"))
            fig.add_trace(go.Scatter(x=month_profile["month_name"], y=month_profile["median"], mode="lines+markers", name="Median"))
            fig.update_layout(yaxis_title="PM2.5", xaxis_title=None, height=390)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Day-of-week profile")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if "day_name" in hourly:
                dow = (
                    hourly.groupby("day_name")["pm25"]
                    .mean()
                    .reindex(dow_order)
                    .reset_index()
                )
                fig = px.bar(dow, x="day_name", y="pm25", labels={"day_name": "", "pm25": "Mean PM2.5"})
                fig.update_layout(height=390, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Seasonal distribution of daily PM2.5")
        plot_df = daily.reset_index()
        fig = px.box(
            plot_df,
            x="season",
            y="pm25_mean",
            category_orders={"season": SEASON_ORDER},
            points=False,
            labels={"season": "", "pm25_mean": "Daily PM2.5"},
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Lag correlation")
        hourly_lags = tables["hourly_lags"]
        if hourly_lags is None:
            lags = [1, 3, 6, 12, 24, 48, 72, 168]
            hourly_lags = pd.DataFrame(
                {
                    "lag_hours": lags,
                    "correlation": [hourly["pm25"].corr(hourly["pm25"].shift(l)) for l in lags],
                }
            )
        else:
            # Handle CSVs saved with lag as index.
            if "lag_hours" not in hourly_lags.columns:
                first = hourly_lags.columns[0]
                if first.startswith("Unnamed"):
                    hourly_lags = hourly_lags.rename(columns={first: "lag_hours"})
            if "lag_hours" not in hourly_lags.columns:
                hourly_lags["lag_hours"] = [1, 3, 6, 12, 24, 48, 72, 168][: len(hourly_lags)]

        fig = px.line(
            hourly_lags,
            x="lag_hours",
            y="correlation",
            markers=True,
            labels={"lag_hours": "Lag (hours)", "correlation": "Correlation"},
        )
        fig.update_layout(height=410)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "The drop through intermediate lags followed by a strong rebound at 24 hours "
            "is consistent with the pronounced daily cycle identified in the hourly series."
        )

        if tables["seasonal_persistence"] is not None:
            st.subheader("Persistence by season")
            sp = tables["seasonal_persistence"].copy()
            # Saved pivot may have season in an unnamed first column.
            if "season" not in sp.columns:
                first = sp.columns[0]
                sp = sp.rename(columns={first: "season"})
            st.dataframe(sp, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Stationarity tests")
        if tables["stationarity"] is not None:
            st.dataframe(tables["stationarity"], use_container_width=True, hide_index=True)
            st.caption(
                "ADF null: unit root/non-stationarity. KPSS null: stationarity. "
                "The raw daily series was non-stationary, while first differencing produced evidence consistent with stationarity."
            )
        else:
            display_missing_table_message("Phase 5 stationarity table")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Ljung–Box: raw daily PM2.5")
            if tables["ljung_raw"] is not None:
                st.dataframe(tables["ljung_raw"], use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Ljung–Box: differenced daily PM2.5")
            if tables["ljung_diff"] is not None:
                st.dataframe(tables["ljung_diff"], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Meteorology & Statistics
# -----------------------------------------------------------------------------

elif page == "Meteorology & Statistics":
    st.title("Meteorology & Statistical Associations")
    st.caption("Associations are observational and should not be interpreted as causal effects.")

    pollutant_map = {
        "PM2.5": "pm25",
        "PM10": "pm10",
        "NO₂": "no2",
        "SO₂": "so2",
        "CO": "co",
    }
    weather_map = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Rain": "rain",
        "Wind speed": "wind_speed",
        "Pressure": "pressure",
        "Dew point": "dew_point",
    }

    c1, c2 = st.columns(2)
    pollutant_label = c1.selectbox("Pollutant", list(pollutant_map))
    weather_label = c2.selectbox("Meteorological variable", list(weather_map))

    pcol = pollutant_map[pollutant_label]
    wcol = weather_map[weather_label]

    sample_n = min(6000, len(hourly))
    scatter_df = hourly[[pcol, wcol, "season"]].dropna()
    if len(scatter_df) > sample_n:
        scatter_df = scatter_df.sample(sample_n, random_state=42)

    fig = px.scatter(
        scatter_df.reset_index(),
        x=wcol,
        y=pcol,
        color="season",
        opacity=0.35,
        category_orders={"season": SEASON_ORDER},
        labels={wcol: weather_label, pcol: pollutant_label, "season": "Season"},
    )
    fig.update_layout(height=470)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Spearman correlation matrix")
    daily_pollutants = [c for c in ["pm25_mean", "pm10_mean", "no2_mean", "so2_mean", "co_mean"] if c in daily.columns]
    daily_weather = [c for c in ["temperature_mean", "humidity_mean", "rain_total", "wind_speed_mean", "pressure_mean", "dew_point_mean"] if c in daily.columns]

    corr = daily[daily_pollutants + daily_weather].corr(method="spearman").loc[daily_pollutants, daily_weather]
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        labels=dict(x="Meteorology", y="Pollutant", color="Spearman ρ"),
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(height=430)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rainfall and PM2.5")
        rain_plot = hourly.copy()
        rain_plot["Condition"] = np.where(rain_plot["rain"] > 0, "Rain", "Dry")
        fig = px.box(
            rain_plot.reset_index(),
            x="Condition",
            y="pm25",
            points=False,
            labels={"pm25": "PM2.5"},
        )
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("PM2.5 by wind sector")
        if "wind_sector" in hourly:
            wind = (
                hourly.groupby("wind_sector", observed=False)["pm25"]
                .agg(["median", "mean", "count"])
                .reindex(WIND_ORDER)
                .reset_index()
            )
            fig = px.bar(
                wind,
                x="wind_sector",
                y="median",
                labels={"wind_sector": "Wind sector", "median": "Median PM2.5"},
            )
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Adjusted meteorological associations")
    if tables["hourly_hac"] is not None:
        hac = tables["hourly_hac"].copy()
        # accommodate saved index
        if "term" not in hac.columns and "variable" not in hac.columns:
            first = hac.columns[0]
            if first.startswith("Unnamed"):
                hac = hac.rename(columns={first: "term"})
        label_col = "term" if "term" in hac.columns else ("variable" if "variable" in hac.columns else hac.columns[0])
        if "coefficient" in hac.columns:
            hac_chart = hac.sort_values("coefficient")
            fig = px.bar(
                hac_chart,
                x="coefficient",
                y=label_col,
                orientation="h",
                labels={"coefficient": "HAC-adjusted coefficient", label_col: ""},
            )
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hac, use_container_width=True, hide_index=True)
    else:
        display_missing_table_message("Phase 6 HAC regression table")

    with st.expander("Seasonal statistical comparisons"):
        if tables["season_pairwise"] is not None:
            st.dataframe(tables["season_pairwise"], use_container_width=True, hide_index=True)
        else:
            display_missing_table_message("Phase 6 seasonal pairwise tests")


# -----------------------------------------------------------------------------
# Extreme Events
# -----------------------------------------------------------------------------

elif page == "Extreme Events":
    st.title("Extreme Pollution Events")

    p90 = hourly["pm25"].quantile(0.90)
    p95 = hourly["pm25"].quantile(0.95)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top-10% threshold", f"{p90:.1f}")
    c2.metric("Top-5% threshold", f"{p95:.1f}")
    c3.metric("Top-10% hours", f"{int((hourly['pm25'] >= p90).sum()):,}")
    c4.metric("Top-5% hours", f"{int((hourly['pm25'] >= p95).sum()):,}")

    episodes = tables["top10_episodes"]
    if episodes is not None:
        episodes = normalize_time_column(episodes)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Top-10% episodes", f"{len(episodes):,}")
        if "duration_hours" in episodes:
            k2.metric("Median duration", f"{episodes['duration_hours'].median():.0f} h")
            k3.metric("Longest duration", f"{episodes['duration_hours'].max():.0f} h")
        if "peak_pm25" in episodes:
            k4.metric("Highest episode peak", f"{episodes['peak_pm25'].max():.1f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Episode duration distribution")
            fig = px.histogram(
                episodes,
                x="duration_hours",
                nbins=35,
                labels={"duration_hours": "Duration (hours)"},
            )
            fig.update_layout(height=390, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Episode rate by season")
            season_table = tables["episode_season"]
            if season_table is not None:
                stbl = season_table.copy()
                if "season" not in stbl.columns:
                    first = stbl.columns[0]
                    stbl = stbl.rename(columns={first: "season"})
                if "episodes_per_1000_hours" in stbl.columns:
                    fig = px.bar(
                        stbl,
                        x="season",
                        y="episodes_per_1000_hours",
                        category_orders={"season": SEASON_ORDER},
                        labels={"season": "", "episodes_per_1000_hours": "Episodes per 1,000 hours"},
                    )
                    fig.update_layout(height=390)
                    st.plotly_chart(fig, use_container_width=True)

        st.subheader("Longest top-10% episodes")
        cols = [
            c for c in
            ["start_time", "end_time", "duration_hours", "mean_pm25", "peak_pm25", "excess_burden", "start_season"]
            if c in episodes.columns
        ]
        st.dataframe(
            episodes.sort_values("duration_hours", ascending=False)[cols].head(15),
            use_container_width=True,
            hide_index=True,
        )

    else:
        display_missing_table_message("Phase 7 episode table")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Extreme-hour probability by wind sector")
        wind = tables["extreme_wind"]
        if wind is not None:
            wind = wind.copy()
            if "wind_sector" not in wind.columns:
                wind = wind.rename(columns={wind.columns[0]: "wind_sector"})
            ycol = "extreme_percent" if "extreme_percent" in wind else "mean"
            fig = px.bar(
                wind,
                x="wind_sector",
                y=ycol,
                category_orders={"wind_sector": WIND_ORDER},
                labels={"wind_sector": "Wind sector", ycol: "Extreme hours (%)" if ycol == "extreme_percent" else "Extreme fraction"},
            )
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Extreme-hour probability in rain vs dry conditions")
        rain = tables["extreme_rain"]
        if rain is not None:
            rain = rain.copy()
            if "rain_status" not in rain.columns:
                rain = rain.rename(columns={rain.columns[0]: "rain_status"})
            ycol = "extreme_percent" if "extreme_percent" in rain else "mean"
            fig = px.bar(
                rain,
                x="rain_status",
                y=ycol,
                labels={"rain_status": "", ycol: "Extreme hours (%)" if ycol == "extreme_percent" else "Extreme fraction"},
            )
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Consecutive particulate-AQI episodes")
    aqi100 = tables["aqi100"]
    aqi150 = tables["aqi150"]

    if aqi100 is not None and aqi150 is not None:
        aqi100 = normalize_time_column(aqi100)
        aqi150 = normalize_time_column(aqi150)
        a, b = st.columns(2)
        with a:
            st.markdown("**AQI > 100**")
            st.metric("Episodes", len(aqi100))
            if "duration_days" in aqi100:
                st.metric("Longest episode", f"{aqi100['duration_days'].max():.0f} days")
        with b:
            st.markdown("**AQI > 150**")
            st.metric("Episodes", len(aqi150))
            if "duration_days" in aqi150:
                st.metric("Longest episode", f"{aqi150['duration_days'].max():.0f} days")

        st.caption(
            "These are consecutive-day episodes based on the project's calculated particulate AQI, "
            "not a complete official multi-pollutant AQI."
        )


# -----------------------------------------------------------------------------
# Forecasting
# -----------------------------------------------------------------------------

elif page == "Forecasting":
    st.title("24-Hour-Ahead PM2.5 Forecasting")

    test_df = splits["test"]
    if test_df is None:
        st.error("data/processed/ml/test.csv is required for the forecasting page.")
        st.stop()

    pred_table = tables["test_predictions"]
    if pred_table is not None:
        pred_table = normalize_time_column(pred_table)
        if "time" in pred_table.columns:
            pred_table = pred_table.set_index("time")
        # Identify actual/persistence/ridge columns robustly.
        if "actual" in pred_table.columns:
            forecast = pred_table.copy()
        else:
            forecast = None
    else:
        forecast = None

    if forecast is None:
        model = load_ridge_model()
        feature_path = PHASE8_DIR / "forecast_feature_list.csv"

        if model is None or not feature_path.exists():
            st.error(
                "Forecast predictions could not be reconstructed. Expected either "
                "reports/tables/phase9_models/test_predictions.csv or "
                "models/phase9/ridge_24h.joblib + the Phase 8 feature list."
            )
            st.stop()

        feature_columns = pd.read_csv(feature_path)["feature"].tolist()
        ridge_pred = model.predict(test_df[feature_columns])

        forecast = pd.DataFrame(index=test_df.index)
        forecast["actual"] = test_df["target_pm25_24h"]
        forecast["Persistence_24h"] = test_df["baseline_persistence"]
        forecast["Ridge"] = ridge_pred

    ridge_col = "Ridge" if "Ridge" in forecast.columns else next(
        (c for c in forecast.columns if c.lower() == "ridge"), None
    )
    persistence_col = "Persistence_24h" if "Persistence_24h" in forecast.columns else next(
        (c for c in forecast.columns if "persistence" in c.lower()), None
    )

    if ridge_col is None or persistence_col is None:
        st.error("Could not identify Ridge and persistence prediction columns.")
        st.stop()

    ridge_m = metrics(forecast["actual"], forecast[ridge_col])
    pers_m = metrics(forecast["actual"], forecast[persistence_col])

    mae_skill = 1 - ridge_m["MAE"] / pers_m["MAE"]
    rmse_skill = 1 - ridge_m["RMSE"] / pers_m["RMSE"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ridge MAE", f"{ridge_m['MAE']:.2f}")
    c2.metric("Persistence MAE", f"{pers_m['MAE']:.2f}")
    c3.metric("MAE skill", f"{mae_skill * 100:.1f}%")
    c4.metric("RMSE skill", f"{rmse_skill * 100:.1f}%")
    c5.metric("Ridge R²", f"{ridge_m['R²']:.3f}")

    st.caption(
        "The Ridge model was frozen before the held-out test period was examined. "
        "Positive skill indicates lower error than 24-hour persistence."
    )

    st.subheader("Test-period forecasts")
    min_date = forecast.index.min().date()
    max_date = forecast.index.max().date()
    date_range = st.date_input(
        "Display range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    display_forecast = forecast
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        display_forecast = forecast.loc[start:end]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=display_forecast.index, y=display_forecast["actual"], name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=display_forecast.index, y=display_forecast[persistence_col], name="Persistence", mode="lines", opacity=0.65))
    fig.add_trace(go.Scatter(x=display_forecast.index, y=display_forecast[ridge_col], name="Ridge", mode="lines", opacity=0.85))
    fig.update_layout(
        height=470,
        hovermode="x unified",
        yaxis_title="PM2.5",
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Pollution regime", "24h change", "Season/month"])

    with tab1:
        regime = tables["error_regime"]
        if regime is not None:
            if "pollution_regime" not in regime.columns:
                regime = regime.rename(columns={regime.columns[0]: "pollution_regime"})
            st.dataframe(regime, use_container_width=True, hide_index=True)
            if "MAE_Skill" in regime:
                fig = px.bar(
                    regime,
                    x="pollution_regime",
                    y="MAE_Skill",
                    labels={"pollution_regime": "", "MAE_Skill": "Ridge MAE skill"},
                )
                fig.add_hline(y=0, line_dash="dash")
                fig.update_layout(height=390)
                st.plotly_chart(fig, use_container_width=True)
        else:
            display_missing_table_message("Phase 10 pollution-regime error table")

    with tab2:
        change = tables["error_change"]
        if change is not None:
            if "change_regime" not in change.columns:
                change = change.rename(columns={change.columns[0]: "change_regime"})
            st.dataframe(change, use_container_width=True, hide_index=True)
            if "MAE_Skill" in change:
                fig = px.bar(
                    change,
                    x="change_regime",
                    y="MAE_Skill",
                    labels={"change_regime": "|Actual 24h PM2.5 change|", "MAE_Skill": "Ridge MAE skill"},
                )
                fig.add_hline(y=0, line_dash="dash")
                fig.update_layout(height=390)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        month = tables["error_month"]
        if month is not None:
            if "target_month" not in month.columns:
                month = month.rename(columns={month.columns[0]: "target_month"})
            if {"Ridge_MAE", "Persistence_MAE"}.issubset(month.columns):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=month["target_month"], y=month["Ridge_MAE"], mode="lines+markers", name="Ridge"))
                fig.add_trace(go.Scatter(x=month["target_month"], y=month["Persistence_MAE"], mode="lines+markers", name="Persistence"))
                fig.update_layout(xaxis=dict(dtick=1), xaxis_title="Month", yaxis_title="MAE", height=390)
                st.plotly_chart(fig, use_container_width=True)
        season = tables["error_season"]
        if season is not None:
            st.dataframe(season, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------

elif page == "Explainability & Errors":
    st.title("Explainability & Forecast Error Analysis")
    st.caption(
        "Post-hoc diagnostics only: nothing on this page is used to retune the frozen forecasting model."
    )

    coef = tables["ridge_coefficients"]
    perm = tables["permutation"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Largest standardized Ridge coefficients")
        if coef is not None:
            coef = coef.copy()
            coef["abs_coefficient"] = coef["coefficient"].abs()
            top = coef.nlargest(15, "abs_coefficient").sort_values("coefficient")
            fig = px.bar(
                top,
                x="coefficient",
                y="feature",
                orientation="h",
                labels={"coefficient": "Standardized coefficient", "feature": ""},
            )
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            display_missing_table_message("Ridge coefficient table")

    with col2:
        st.subheader("Permutation importance")
        if perm is not None:
            top = perm.nlargest(15, "importance_mean").sort_values("importance_mean")
            fig = px.bar(
                top,
                x="importance_mean",
                y="feature",
                orientation="h",
                error_x="importance_std" if "importance_std" in top else None,
                labels={"importance_mean": "Increase in MAE after permutation", "feature": ""},
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            display_missing_table_message("Permutation-importance table")

    st.info(
        "Correlated lagged and rolling predictors share information. Individual Ridge signs and "
        "permutation scores should therefore be interpreted as predictive diagnostics, not causal effects."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Forecast calibration")
        cal = tables["calibration"]
        if cal is not None and {"predicted_mean", "actual_mean"}.issubset(cal.columns):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=cal["predicted_mean"],
                    y=cal["actual_mean"],
                    mode="lines+markers",
                    name="Calibration",
                )
            )
            low = min(cal["predicted_mean"].min(), cal["actual_mean"].min())
            high = max(cal["predicted_mean"].max(), cal["actual_mean"].max())
            fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect calibration", line=dict(dash="dash")))
            fig.update_layout(
                xaxis_title="Mean predicted PM2.5",
                yaxis_title="Mean actual PM2.5",
                height=410,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("High-pollution detection")
        detection = tables["high_pollution_detection"]
        if detection is not None:
            st.dataframe(detection, use_container_width=True)

    error = tables["test_error"]
    if error is not None:
        error = normalize_time_column(error)
        if "time" in error.columns:
            error = error.set_index("time")

        st.subheader("Forecast error distribution")
        if "ridge_error" in error:
            fig = px.histogram(
                error,
                x="ridge_error",
                nbins=60,
                labels={"ridge_error": "Ridge error (prediction − actual)"},
            )
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(height=390, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Residual autocorrelation")
            lags = list(range(1, 73))
            ac = [error["ridge_error"].autocorr(lag=l) for l in lags]
            ac_df = pd.DataFrame({"lag": lags, "autocorrelation": ac})
            fig = px.line(
                ac_df,
                x="lag",
                y="autocorrelation",
                markers=False,
                labels={"lag": "Lag (hours)", "autocorrelation": "Autocorrelation"},
            )
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True)

        if {"target_time", "actual", "ridge", "persistence", "ridge_error"}.issubset(error.columns):
            st.subheader("Largest Ridge underpredictions")
            misses = error.sort_values("ridge_error").head(15)[
                ["target_time", "actual", "ridge", "persistence", "ridge_error"]
            ]
            st.dataframe(misses, use_container_width=True)

    st.subheader("Key interpretation")
    st.markdown(
        """
        - **Persistence wins when conditions are nearly unchanged.**
        - **Ridge adds value as 24-hour pollution changes become larger.**
        - The model performs better than persistence in elevated/high pollution regimes, but it
          still tends to **underpredict the highest actual concentrations**.
        - Forecast residuals retain temporal structure, showing that the current linear model
          does not exhaust all predictable time-series information.
        """
    )


# -----------------------------------------------------------------------------
# Methods & Data Quality
# -----------------------------------------------------------------------------

elif page == "Methods & Data Quality":
    st.title("Methods, Data Quality & Reproducibility")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hourly rows", f"{len(hourly):,}")
    c2.metric("Duplicate timestamps", f"{hourly.index.duplicated().sum():,}")
    c3.metric("Missing values", f"{hourly.isna().sum().sum():,}")
    inferred = pd.infer_freq(hourly.index)
    c4.metric("Inferred frequency", inferred or "Irregular")

    st.subheader("Study structure")
    st.markdown(
        """
        1. **Data integration & validation** — aligned pollution and meteorology hourly.
        2. **Data quality** — physical validity, outliers, skewness, continuity.
        3. **Feature engineering** — calendar/cyclic features, wind encoding, rain flags,
           particulate AQI, daily aggregation.
        4. **EDA** — pollutant distributions and temporal/meteorological patterns.
        5. **Time-series analysis** — lag correlations, ACF/PACF, stationarity, persistence.
        6. **Statistical analysis** — nonparametric tests, effect sizes, HAC regression.
        7. **Extreme events** — percentile episodes, duration/severity, AQI episodes.
        8. **Forecast preparation** — leakage-safe 24-hour target and chronological splits.
        9. **Model comparison** — Ridge, Random Forest, HistGradientBoosting vs persistence.
        10. **Explainability & error analysis** — coefficients, permutation importance,
            subgroup errors, high-pollution detection and residual structure.
        """
    )

    st.subheader("Important methodological notes")
    st.markdown(
        """
        **Units.** Pollutant unit symbols were mangled in the original export headers. The dashboard
        therefore avoids asserting exact pollutant concentration units unless verified from the
        original source metadata.

        **Timezone.** The timestamp series is timezone-naive in the processed data. Hour-of-day
        findings use the timestamps as supplied; the original source timezone should be verified
        before publication-quality interpretation of local clock time.

        **AQI.** The displayed AQI is a reproducible **particulate AQI** calculated from daily
        PM2.5 and PM10. It is not claimed to be a full official multi-pollutant AQI.

        **Outliers.** High pollution observations were retained. Statistical outliers are not
        automatically invalid measurements and are central to the extreme-event research question.

        **Inference.** Meteorological results are associations in observational data. Even adjusted
        regression coefficients should not be described as causal effects.

        **Forecasting.** The 24-hour forecast uses only information available at or before the
        forecast origin plus calendar information known in advance. Actual future observed weather
        is excluded. The train/validation/test split is chronological and uses purge gaps.
        """
    )

    st.subheader("Core variable summary")
    summary_cols = [
        c for c in
        ["pm25", "pm10", "no2", "so2", "co", "temperature", "humidity", "rain", "wind_speed", "pressure", "dew_point"]
        if c in hourly.columns
    ]
    summary = hourly[summary_cols].describe().T[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    st.dataframe(summary.round(2), use_container_width=True)

    csv = summary.to_csv().encode("utf-8")
    st.download_button(
        "Download variable summary CSV",
        data=csv,
        file_name="air_quality_variable_summary.csv",
        mime="text/csv",
    )

st.sidebar.divider()
st.sidebar.caption(
    "Kathmandu Air Quality Intelligence • Research dashboard\n\nAuthor: Nishanta Nepal"
)
