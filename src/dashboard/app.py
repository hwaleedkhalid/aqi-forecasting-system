"""Pearls AQI Predictor - Streamlit Dashboard Application.

Interactive 72-hour AQI forecasting dashboard for Lahore, Pakistan.
Visualizes multi-horizon predictions with empirical error intervals,
telemetry observations, domain threshold alerts, and provenance metadata.
Seamlessly interfaces with the Flask REST API with automatic local inference fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.dashboard.components import (
    build_forecast_figure,
    render_current_observation_card,
    render_explainability_section,
    render_freshness_banner,
    render_metadata_header,
    render_sidebar,
    render_telemetry_breakdown,
)
from src.dashboard.data_client import DashboardDataClient
from src.logger import logger


def init_page_config() -> None:
    """Initialize Streamlit page configuration and styling."""
    st.set_page_config(
        page_title="Pearls AQI Predictor — 72-Hour AQI Forecast Dashboard",
        page_icon="🌫️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_horizon_milestones(forecasts: list[dict]) -> None:
    """Render summary metric cards for key milestone horizons (+1h, +12h, +24h, +48h, +72h)."""
    if not forecasts:
        return

    milestone_horizons = [1, 12, 24, 48, 72]
    lookup = {f["horizon"]: f for f in forecasts}
    available_milestones = [lookup[h] for h in milestone_horizons if h in lookup]

    if not available_milestones:
        return

    st.subheader("Key Forecast Milestones")
    cols = st.columns(len(available_milestones))

    for col, item in zip(cols, available_milestones):
        h = item["horizon"]
        aqi_val = item["aqi"]
        category = item["category"]
        lower = item["error_lower"]
        upper = item["error_upper"]
        color = item.get("color", "#1976D2")

        with col:
            st.markdown(
                f"""
                <div style="border: 1px solid #ddd; border-top: 4px solid {color}; border-radius: 6px; padding: 12px; background: #fafafa; text-align: center;">
                    <div style="font-size: 0.85rem; color: #666; font-weight: bold; text-transform: uppercase;">Horizon +{h}h</div>
                    <div style="font-size: 1.8rem; font-weight: bold; color: #111; margin: 4px 0;">{aqi_val:.1f}</div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: {color}; margin-bottom: 6px;">{category}</div>
                    <div style="font-size: 0.75rem; color: #777;">Residual Range:<br><b>[{lower:.1f}, {upper:.1f}]</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    """Main Streamlit execution entry point."""
    init_page_config()

    # Title and Subtitle - explicitly adhering to scientific naming (never falsely claimed 'live')
    st.title("Pearls AQI Predictor — 72-Hour AQI Forecast Dashboard")
    st.caption("Lahore, Pakistan (31.52° N, 74.35° E) • Multi-Horizon Persistence-Aware ML System")

    # Header controls
    top_col1, top_col2 = st.columns([5, 1])
    with top_col2:
        force_refresh = st.button("🔄 Refresh Forecast", help="Bypass cache and force recomputation")

    client = DashboardDataClient()

    # Data fetching with graceful error handling
    try:
        with st.spinner("Retrieving latest telemetry and multi-horizon forecast..."):
            obs_data, current_source = client.fetch_current()
            forecast_data, forecast_source = client.fetch_forecast(force_refresh=force_refresh)
            model_info, _ = client.fetch_model_info()
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}", exc_info=True)
        st.error(f"Failed to load air quality data: {e}")
        return

    # 1. Stale Data Notice Banner (prominent when telemetry is historical)
    is_stale = forecast_data.get("is_stale", obs_data.get("is_stale", False))
    observed_at = forecast_data.get("input_observed_at", obs_data.get("input_observed_at", "Unknown"))
    age_hours = forecast_data.get("input_age_hours", obs_data.get("input_age_hours", 0.0))
    render_freshness_banner(is_stale, observed_at, age_hours)

    # 2. Metadata & Provenance Row
    forecast_origin = forecast_data.get("forecast_origin", observed_at)
    generated_at = forecast_data.get("generated_at", "Unknown")
    latency_ms = forecast_data.get("inference_latency_ms", 0.0)
    render_metadata_header(forecast_origin, generated_at, forecast_source, latency_ms)

    st.markdown("---")

    # 3. Current Observation Card
    st.subheader("Current Telemetry & Baseline AQI")
    render_current_observation_card(obs_data)

    # 4. Multi-Horizon Forecast Chart
    st.subheader("72-Hour Ahead AQI Trajectory")
    forecasts = forecast_data.get("forecasts", [])
    if forecasts:
        fig = build_forecast_figure(forecasts)
        st.plotly_chart(fig, use_container_width=True)

        # 5. Horizon Milestones (+1h, +12h, +24h, +48h, +72h)
        render_horizon_milestones(forecasts)
    else:
        st.info("No forecast horizon data available.")

    st.markdown("---")

    # 6. Model Explainability & Feature Attribution Section
    selected_horizon = st.slider(
        "Select Forecast Horizon for Feature Attribution:",
        min_value=1,
        max_value=72,
        value=24,
        step=1,
        help="Inspect SHAP feature attribution and persistence decomposition for any hourly horizon.",
    )
    with st.spinner(f"Computing SHAP attribution for horizon +{selected_horizon}h..."):
        explain_data, _ = client.fetch_explain(horizon=selected_horizon, top_k=10)
    render_explainability_section(explain_data)

    st.markdown("---")

    # 7. Detailed Telemetry Observations Breakdown
    render_telemetry_breakdown(obs_data)

    # 8. Sidebar Provenance & Summary
    summary = forecast_data.get("summary", {})
    render_sidebar(model_info, summary, forecast_source)


if __name__ == "__main__":
    main()
