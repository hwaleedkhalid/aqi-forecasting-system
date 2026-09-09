"""Pearls AQI Predictor - Streamlit Dashboard Application.

Modern, public-facing 72-hour AQI forecast dashboard for Lahore, Pakistan.
Information architecture (top to bottom):

  1. App header (logo + location + relative time + Refresh)
  2. Hero: Current AQI (left) | 72-Hour Outlook (right)
  3. Compact alert / status area
  4. 72-Hour AQI Forecast chart
  5. Milestone cards (+1h +12h +24h +48h +72h)
  6. Air Pollutants | Weather (compact grids)
  7. Health Guidance
  8. Advanced Insights (tabs: Forecast Drivers | Global Importance)
  9. Model & System Details (collapsed expander)

Backend, ML model, alerting logic, Feature Store, and production artifacts
are completely unchanged. This module is frontend-only.
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
    build_aqi_gauge_figure,
    build_forecast_figure,
    render_alert_banners,
    render_aqi_legend,
    render_cold_start_error,
    render_current_observation_card,
    render_explainability_section,
    render_forecast_narrative_card,
    render_health_guidance,
    render_metadata_header,
    render_model_system_details,
    render_sidebar,
    render_telemetry_breakdown,
    render_72h_outlook_card,
)
from src.dashboard.data_client import DashboardDataClient
from src.dashboard.styles import DASHBOARD_CSS
from src.dashboard.ui_helpers import (
    category_bg_color,
    category_border_color,
    category_color,
    category_text_color,
    format_relative_age,
    safe_val,
)
from src.logger import logger


def init_page_config() -> None:
    """Initialize Streamlit page configuration and inject design system CSS."""
    st.set_page_config(
        page_title="Pearls AQI Predictor — Lahore",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def render_app_header(
    age_hours: float | None = None,
    force_refresh: bool = False,
    key: str = "refresh_btn",
) -> bool:
    """Render the application header with logo, subtitle, location, and refresh button.

    Args:
        age_hours: Age of the current observation in hours (for relative time display).
        force_refresh: Current state of the refresh button (unused; for signature parity).
        key: Unique Streamlit widget key for the refresh button.

    Returns:
        True if the Refresh button was clicked this run, False otherwise.
    """
    col_left, col_right = st.columns([7, 1], gap="small")

    with col_left:
        freshness = f"Updated {format_relative_age(age_hours)}" if age_hours is not None else ""
        st.markdown(
            f"""
<div class="prl-app-header">
  <div class="prl-app-header-left">
    <div class="prl-app-logo">🌿 Pearls Air</div>
    <div class="prl-app-subtitle">Lahore Air Quality Forecast &nbsp;·&nbsp; Current conditions and 72-hour outlook</div>
  </div>
  <div class="prl-app-header-right">
    <span class="prl-location-tag">📍 Lahore, Pakistan</span><br>
    {freshness}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_right:
        clicked = st.button(
            "Refresh",
            key=key,
            help="Bypass cache and force recomputation of the 72-hour forecast",
            use_container_width=True,
        )
    return bool(clicked)


def render_horizon_milestones(forecasts: list[dict]) -> None:
    """Render redesigned milestone cards for key forecast horizons.

    Shows +1h, +12h, +24h, +48h, +72h with category-tinted background,
    category accent strip, large AQI value, category label, and empirical range.

    Args:
        forecasts: List of forecast horizon dictionaries matching API contract.
    """
    if not forecasts:
        return

    milestone_horizons = [1, 12, 24, 48, 72]
    lookup = {f["horizon"]: f for f in forecasts}
    available = [lookup[h] for h in milestone_horizons if h in lookup]

    if not available:
        return

    cols = st.columns(len(available), gap="small")
    for col, item in zip(cols, available):
        h = item["horizon"]
        aqi_val = item["aqi"]
        category = item["category"]
        lower = item["error_lower"]
        upper = item["error_upper"]
        accent = category_color(category)
        bg = category_bg_color(category)
        border = category_border_color(category)
        text_col = category_text_color(category)

        aqi_str = f"{aqi_val:.0f}" if aqi_val is not None else "—"
        range_str = f"[{safe_val(lower, decimals=0)}, {safe_val(upper, decimals=0)}]"

        with col:
            st.markdown(
                f"""
<div class="prl-milestone-card" style="background:{bg};border:1px solid {border};">
  <div class="prl-milestone-accent" style="background:{accent};"></div>
  <div class="prl-milestone-horizon">+{h}h</div>
  <div class="prl-milestone-aqi">{aqi_str}</div>
  <div class="prl-milestone-cat" style="color:{text_col};">{category}</div>
  <div class="prl-milestone-range">Range: {range_str}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def main() -> None:
    """Main Streamlit execution entry point."""
    init_page_config()

    force_refresh = bool(st.session_state.get("refresh_btn", False))

    client = DashboardDataClient()

    # ── Data fetching with cold-start / error handling ─────────────────────
    try:
        with st.spinner("Loading air quality data…"):
            obs_data, current_source = client.fetch_current()
            forecast_data, forecast_source = client.fetch_forecast(force_refresh=force_refresh)
            model_info, _ = client.fetch_model_info()
            # Fetch baseline explanation for human-readable card (+24h)
            try:
                explain_data_24, _ = client.fetch_explain(horizon=24, top_k=6)
            except Exception:
                explain_data_24 = {}
    except ConnectionError as e:
        logger.warning(f"Dashboard API connection error: {e}")
        render_app_header(age_hours=None)
        render_cold_start_error(e)
        return
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}", exc_info=True)
        render_app_header(age_hours=None)
        st.error("**Failed to load air quality data.** Please try again shortly.")
        with st.expander("Technical details"):
            st.code(str(e))
        return

    # ── Header ────────────────────────────────────────────────────────────
    age_hours = obs_data.get("input_age_hours")
    render_app_header(age_hours=age_hours)

    # ── 1. Hero: two-column Current AQI + 72h Outlook ─────────────────────
    hero_left, hero_right = st.columns(2, gap="medium")
    with hero_left:
        render_current_observation_card(obs_data)
    with hero_right:
        render_72h_outlook_card(forecast_data, current_category=obs_data.get("category", ""))

    # ── 2. Compact alert / status area ────────────────────────────────────
    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
    render_alert_banners(obs_data, forecast_data)

    # ── Compact metadata status line ──────────────────────────────────────
    observed_at = forecast_data.get(
        "input_observed_at",
        obs_data.get("input_observed_at", "Unknown"),
    )
    forecast_origin = forecast_data.get("forecast_origin", observed_at)
    generated_at = forecast_data.get("generated_at", "Unknown")
    latency_ms = forecast_data.get("inference_latency_ms", 0.0)
    render_metadata_header(forecast_origin, generated_at, forecast_source, latency_ms)

    st.divider()

    # ── 3. 72-Hour AQI Forecast chart & Legend ────────────────────────────
    st.markdown('<div class="prl-section-heading">72-Hour AQI Forecast</div>', unsafe_allow_html=True)
    st.caption("Hourly air quality outlook for Lahore · Empirical prediction intervals from walk-forward out-of-fold residuals")
    render_aqi_legend()

    forecasts = forecast_data.get("forecasts", [])
    if forecasts:
        fig = build_forecast_figure(forecasts)
        st.plotly_chart(fig, use_container_width=True)

        # ── 4. Milestone cards ─────────────────────────────────────────────
        render_horizon_milestones(forecasts)
    else:
        st.info("No forecast horizon data available.")

    # ── "Why this forecast?" Human-readable explanation card ──────────────
    if explain_data_24:
        render_forecast_narrative_card(explain_data_24, horizon=24)

    st.divider()

    # ── 5. Air Pollutants + Weather ───────────────────────────────────────
    st.markdown('<div class="prl-section-heading">Air Pollutants &amp; Weather</div>', unsafe_allow_html=True)
    render_telemetry_breakdown(obs_data)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── 6. Health Guidance ────────────────────────────────────────────────
    st.markdown('<div class="prl-section-heading">Health Guidance</div>', unsafe_allow_html=True)
    render_health_guidance(obs_data, forecast_data)

    st.divider()

    # ── 7. Advanced Insights (SHAP) ───────────────────────────────────────
    st.markdown('<div class="prl-section-heading">Advanced Insights</div>', unsafe_allow_html=True)
    st.caption("Model feature attribution and global importance analysis for EXP-019 hybrid architecture.")

    tab_drivers, tab_global = st.tabs(["🔍 Forecast Drivers", "📊 Global Importance"])

    with tab_drivers:
        selected_horizon = st.slider(
            "Forecast Horizon for Feature Attribution:",
            min_value=1,
            max_value=72,
            value=24,
            step=1,
            help="Inspect SHAP feature attribution and persistence decomposition for any hourly horizon.",
        )

        with st.spinner(f"Computing SHAP attribution for horizon +{selected_horizon}h…"):
            explain_data, _ = client.fetch_explain(horizon=selected_horizon, top_k=10)
        render_explainability_section(explain_data)

    with tab_global:
        global_feats = explain_data.get("global_top_features", []) if explain_data else []
        p_mean = explain_data.get("global_persistence_mean_contribution", 0.0) if explain_data else 0.0
        if global_feats:
            import pandas as pd
            st.markdown(
                f"**Mean Absolute Persistence Contribution across 72h:** `{p_mean:.2f} AQI points`"
            )
            st.caption("500-sample stratified cohort · Aggregated SHAP magnitudes across all 72 specialist sub-models.")
            df = pd.DataFrame(global_feats)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Global feature importance data not yet available.")

    st.divider()

    # ── 8. Model & System Details (collapsed) ─────────────────────────────
    render_model_system_details(model_info, forecast_data, obs_data, forecast_source)

    # ── Sidebar: no-op (intentionally empty) ──────────────────────────────
    # render_sidebar stub kept for import/test compatibility; sidebar is collapsed.
    render_sidebar(model_info, forecast_data.get("summary", {}), forecast_source)


if __name__ == "__main__":
    main()
