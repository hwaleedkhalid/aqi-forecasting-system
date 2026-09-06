"""Pearls AQI Predictor - Dashboard UI Components.

Reusable rendering components and Plotly chart builders for the Streamlit dashboard,
strictly adhering to domain threshold semantics, transparent data freshness notices,
and empirical error interval visualizations.
"""

from __future__ import annotations

from typing import Any
import plotly.graph_objects as go
import streamlit as st


def build_forecast_figure(forecasts: list[dict[str, Any]]) -> go.Figure:
    """Build interactive Plotly figure displaying 72-hour forecast with empirical error intervals.

    Args:
        forecasts: List of 72 forecast horizon dictionaries matching contract.

    Returns:
        Configured Plotly Figure instance.
    """
    times = [f["forecast_time"] for f in forecasts]
    aqis = [f["aqi"] for f in forecasts]
    lowers = [f["error_lower"] for f in forecasts]
    uppers = [f["error_upper"] for f in forecasts]
    horizons = [f["horizon"] for f in forecasts]
    categories = [f["category"] for f in forecasts]

    custom_data = list(zip(horizons, categories, lowers, uppers))

    fig = go.Figure()

    # 1. Shaded Empirical Prediction Error Interval Band
    # Upper bound (invisible line)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=uppers,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Lower bound with fill to upper bound
    fig.add_trace(
        go.Scatter(
            x=times,
            y=lowers,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(33, 150, 243, 0.18)",
            name="Empirical Error Range (Walk-forward 10th–90th out-of-fold residuals)",
            hoverinfo="skip",
        )
    )

    # 2. Predicted AQI Main Curve
    fig.add_trace(
        go.Scatter(
            x=times,
            y=aqis,
            mode="lines+markers",
            name="Predicted AQI (EXP-019)",
            line=dict(color="#1976D2", width=3),
            marker=dict(size=4, color="#0D47A1"),
            customdata=custom_data,
            hovertemplate=(
                "<b>Horizon:</b> +%{customdata[0]}h<br>"
                "<b>Forecast Time:</b> %{x}<br>"
                "<b>Predicted AQI:</b> %{y:.1f}<br>"
                "<b>Category:</b> %{customdata[1]}<br>"
                "<b>Empirical Error Interval:</b> [%{customdata[2]:.1f}, %{customdata[3]:.1f}]<extra></extra>"
            ),
        )
    )

    # 3. Horizontal Reference Lines for Domain Thresholds
    fig.add_hline(
        y=200,
        line_dash="dash",
        line_color="#8F3F97",
        line_width=1.5,
        annotation_text="High Severity (>200)",
        annotation_position="top left",
        annotation_font=dict(color="#8F3F97", size=10),
    )
    fig.add_hline(
        y=300,
        line_dash="dash",
        line_color="#7E0023",
        line_width=1.5,
        annotation_text="Hazardous (>300)",
        annotation_position="top left",
        annotation_font=dict(color="#7E0023", size=10),
    )

    # 4. Layout formatting
    max_val = max(max(uppers), max(aqis), 350.0)
    fig.update_layout(
        title=dict(
            text="<b>72-Hour AQI Multi-Horizon Trajectory</b><br><sup>Empirical prediction error intervals derived from out-of-fold residuals (not parametric model confidence)</sup>",
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title="Forecast Target Time (UTC)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis=dict(
            title="Air Quality Index (AQI)",
            rangemode="nonnegative",
            range=[0, max_val + 20],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        margin=dict(l=40, r=30, t=80, b=40),
        hovermode="x unified",
        height=480,
    )

    return fig


def render_freshness_banner(is_stale: bool, observed_at: str, age_hours: float) -> None:
    """Render explicit warning banner when input telemetry is historical."""
    if is_stale:
        st.warning(
            f"⚠️ **Telemetry Notice**: Latest available observation is historical "
            f"(Observed: `{observed_at}`, Age: **{age_hours:.1f} hours**). "
            f"Forecasts are anchored to this historical observation origin, not current browser time."
        )


def render_metadata_header(
    forecast_origin: str,
    generated_at: str,
    source_mode: str,
    latency_ms: float,
) -> None:
    """Render provenance headers cleanly separating forecast origin and generation time."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Forecast Origin (Observation Time)")
        st.write(f"`{forecast_origin}`")
    with col2:
        st.caption("Served / Generated At")
        st.write(f"`{generated_at}`")
    with col3:
        st.caption("Data Source Mode")
        badge_color = "green" if "REST API" in source_mode else "orange"
        st.markdown(f":{badge_color}[**{source_mode}**]")
    with col4:
        st.caption("Inference Latency")
        st.write(f"**{latency_ms:.1f} ms**")


def render_current_observation_card(obs: dict[str, Any]) -> None:
    """Render metric card for current air quality observation."""
    aqi_val = obs.get("current_aqi", 0.0)
    category = obs.get("category", "Unknown")
    color = obs.get("color", "#808080")
    dominant = obs.get("dominant_pollutant", "pm2_5")
    advisory = obs.get("health_advisory", "")

    st.markdown(
        f"""
        <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 16px; border-radius: 6px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 2.4rem; font-weight: bold; color: #111;">{aqi_val:.0f} <span style="font-size: 1.1rem; font-weight: normal; color: #555;">AQI</span></span>
                <span style="background-color: {color}; color: white; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 0.95rem;">{category}</span>
            </div>
            <div style="margin-top: 6px; font-size: 0.9rem; color: #444;">
                <b>Dominant Pollutant:</b> <code>{dominant.upper()}</code>
            </div>
            <div style="margin-top: 8px; font-size: 0.92rem; line-height: 1.4; color: #222;">
                <b>Public Health Advisory:</b> {advisory}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_telemetry_breakdown(obs: dict[str, Any]) -> None:
    """Render pollutant concentrations and surface weather conditions."""
    pollutants = obs.get("pollutants", {})
    weather = obs.get("weather", {})

    st.subheader("Telemetry Observations")
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
    with p_col1:
        st.metric("PM2.5", f"{pollutants.get('pm2_5', 'N/A')} µg/m³")
    with p_col2:
        st.metric("PM10", f"{pollutants.get('pm10', 'N/A')} µg/m³")
    with p_col3:
        st.metric("NO₂", f"{pollutants.get('no2', 'N/A')} µg/m³")
    with p_col4:
        st.metric("SO₂", f"{pollutants.get('so2', 'N/A')} µg/m³")
    with p_col5:
        st.metric("CO", f"{pollutants.get('co', 'N/A')} µg/m³")
    with p_col6:
        st.metric("O₃", f"{pollutants.get('o3', 'N/A')} µg/m³")

    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        st.metric("Temperature", f"{weather.get('temperature_2m', 'N/A')} °C")
    with w_col2:
        st.metric("Humidity", f"{weather.get('relative_humidity_2m', 'N/A')} %")
    with w_col3:
        st.metric("Wind Speed", f"{weather.get('wind_speed_10m', 'N/A')} m/s")
    with w_col4:
        st.metric("Pressure", f"{weather.get('surface_pressure', 'N/A')} hPa")


def render_sidebar(model_info: dict[str, Any], summary: dict[str, Any], source_mode: str) -> None:
    """Render sidebar with model provenance, architecture, and forecast summary."""
    st.sidebar.title("System & Model Provenance")

    st.sidebar.markdown(f"**Data Pipeline:** `{source_mode}`")

    st.sidebar.subheader("Champion Model (EXP-019)")
    st.sidebar.markdown(f"- **Architecture:** `{model_info.get('architecture', 'Hybrid Specialist')}`")
    st.sidebar.markdown(f"- **Input Dimensionality:** `{model_info.get('feature_count', 114)} canonical features`")
    st.sidebar.markdown(f"- **Schema Version:** `{model_info.get('feature_schema_version', 'v2_weather_enriched')}`")
    st.sidebar.markdown(f"- **Status:** `{model_info.get('status', 'validated_champion')}`")

    benchmarks = model_info.get("test_benchmark_metrics", {})
    if benchmarks:
        st.sidebar.subheader("Held-Out Test Benchmark (Phase 10.5E)")
        st.sidebar.markdown(f"- **Overall RMSE:** `{benchmarks.get('overall_rmse')} AQI`")
        st.sidebar.markdown(f"- **Overall MAE:** `{benchmarks.get('overall_mae')} AQI`")
        st.sidebar.markdown(f"- **Overall R²:** `{benchmarks.get('overall_r2')}`")
        st.sidebar.markdown(f"- **h+1 RMSE:** `{benchmarks.get('h1_rmse')} AQI`")
        st.sidebar.markdown(f"- **h+72 RMSE:** `{benchmarks.get('h72_rmse')} AQI`")

    st.sidebar.subheader("72-Hour Forecast Summary")
    if summary:
        st.sidebar.markdown(f"- **Peak AQI:** `{summary.get('peak_aqi', 0.0):.1f}` (+{summary.get('peak_horizon', 0)}h)")
        st.sidebar.markdown(f"- **Peak Category:** `{summary.get('peak_category', 'N/A')}`")
        if summary.get("has_hazardous"):
            st.sidebar.error("⚠️ Severe Alert: Hazardous AQI (>300) predicted within 72h!")
        elif summary.get("has_high_severity"):
            st.sidebar.warning("⚠️ Alert: High Severity AQI (>200) predicted within 72h.")
        else:
            st.sidebar.success("No extreme AQI (>200) predicted in 72h window.")
