"""Pearls AQI Predictor - Dashboard UI Components.

Reusable rendering components and Plotly chart builders for the Streamlit dashboard,
strictly adhering to domain threshold semantics, transparent data freshness notices,
and empirical error interval visualizations.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
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

    # 3. Horizontal Reference Lines for EPA Category Boundaries
    fig.add_hline(
        y=151,
        line_dash="dash",
        line_color="#FF0000",
        line_width=1.5,
        annotation_text="Unhealthy (151)",
        annotation_position="top left",
        annotation_font=dict(color="#FF0000", size=10),
    )
    fig.add_hline(
        y=201,
        line_dash="dash",
        line_color="#8F3F97",
        line_width=1.5,
        annotation_text="Very Unhealthy (201)",
        annotation_position="top left",
        annotation_font=dict(color="#8F3F97", size=10),
    )
    fig.add_hline(
        y=301,
        line_dash="dash",
        line_color="#7E0023",
        line_width=1.5,
        annotation_text="Hazardous (301)",
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


def render_alert_banners(obs_data: dict[str, Any], forecast_data: dict[str, Any]) -> None:
    """Render prioritized, uncertainty-aware alert banners for observations, forecasts, and freshness.

    Priority hierarchy:
    1. Current Severe / Hazardous observation
    2. Forecast Severe / Hazardous trajectory (within 72 hours)
    3. Stale telemetry notice (if input observation > 3h old)
    4. Current or Forecast Unhealthy warning / Advisory (if no higher tier active)
    5. Upper empirical residual error interval crossing Hazardous threshold (if forecast < 201)

    Args:
        obs_data: Latest observation dictionary matching API contract.
        forecast_data: Forecast result dictionary matching API contract.
    """
    # 1. Extract observation alert metadata
    obs_alert = obs_data.get("alert")
    if not obs_alert:
        curr_aqi = obs_data.get("current_aqi", 0.0)
        from src.inference.alerting import evaluate_current_alert
        obs_alert = evaluate_current_alert(
            aqi=curr_aqi,
            data_is_stale=obs_data.get("is_stale", False),
            feature_source=obs_data.get("feature_source", "unknown"),
            fallback_active=obs_data.get("fallback_active", False),
        ).to_dict()

    obs_level = obs_alert.get("level", "none")
    obs_aqi = obs_alert.get("aqi", obs_data.get("current_aqi", 0.0))
    obs_cat = obs_alert.get("category", obs_data.get("category", "Unknown"))
    obs_msg = obs_alert.get("message", "")

    # 2. Extract forecast alert metadata
    f_alert = forecast_data.get("forecast_alert") or forecast_data.get("summary", {}).get("forecast_alert") or {}
    f_level = f_alert.get("highest_level", "none")
    f_msg = f_alert.get("message", "")

    # Priority 1: Current Severe / Hazardous observation
    if obs_level == "hazardous":
        st.error(
            f"🚨 **HAZARDOUS AIR QUALITY EMERGENCY**: Current observed air quality in Lahore is "
            f"**{obs_aqi:.0f} AQI** ({obs_cat}). {obs_msg}"
        )
    elif obs_level == "severe":
        st.error(
            f"⚠️ **VERY UNHEALTHY AIR QUALITY ALERT**: Current observed air quality in Lahore is "
            f"**{obs_aqi:.0f} AQI** ({obs_cat}). {obs_msg}"
        )

    # Priority 2: Forecast Severe / Hazardous trajectory
    if f_level == "hazardous":
        st.error(f"🚨 **HAZARDOUS AQI FORECAST**: {f_msg}")
    elif f_level == "severe":
        st.error(f"⚠️ **VERY UNHEALTHY AQI FORECAST**: {f_msg}")

    # Priority 3: Stale Data Notice (displayed alongside severe alerts if data is historical)
    is_stale = forecast_data.get("is_stale", obs_data.get("is_stale", False))
    observed_at = forecast_data.get("input_observed_at", obs_data.get("input_observed_at", "Unknown"))
    age_hours = forecast_data.get("input_age_hours", obs_data.get("input_age_hours", 0.0))
    if is_stale:
        render_freshness_banner(is_stale=True, observed_at=observed_at, age_hours=age_hours)

    # Priority 4: Warnings & Advisories (only if no severe/hazardous banners were shown for that tier)
    if obs_level not in ("hazardous", "severe") and f_level not in ("hazardous", "severe"):
        if obs_level == "warning":
            st.warning(
                f"⚠️ **UNHEALTHY AIR QUALITY WARNING**: Current observed air quality is "
                f"**{obs_aqi:.0f} AQI** ({obs_cat}). {obs_msg}"
            )
        elif f_level == "warning":
            st.warning(f"⚠️ **UNHEALTHY AQI FORECAST**: {f_msg}")
        elif obs_level == "advisory":
            st.info(
                f"ℹ️ **AIR QUALITY ADVISORY**: Current observed air quality is "
                f"**{obs_aqi:.0f} AQI** ({obs_cat}). {obs_msg}"
            )
        elif f_level == "advisory":
            st.info(f"ℹ️ **AIR QUALITY ADVISORY FORECAST**: {f_msg}")

    # Priority 5: Upper residual error interval crosses hazardous while point forecast does not
    if f_alert.get("upper_interval_crosses_hazardous") and f_level not in ("hazardous", "severe"):
        st.info(
            "ℹ️ **Uncertainty Notice**: The 90th percentile empirical error interval crosses the Hazardous "
            "threshold (>300 AQI) at one or more horizons, indicating extreme air pollution tail risk. "
            "Monitor ongoing hourly telemetry updates."
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

        forecast_alert = summary.get("forecast_alert") or {}
        highest_level = forecast_alert.get("highest_level")
        if not highest_level:
            if summary.get("has_hazardous"):
                highest_level = "hazardous"
            elif summary.get("has_high_severity"):
                highest_level = "severe"
            else:
                highest_level = "none"

        if highest_level == "hazardous":
            first_h = forecast_alert.get("first_hazardous_horizon")
            h_str = f" (first at +{first_h}h)" if first_h else ""
            st.sidebar.error(f"🚨 **Hazardous AQI Emergency**: Forecast enters Hazardous (>=301){h_str}!")
        elif highest_level == "severe":
            first_h = forecast_alert.get("first_very_unhealthy_horizon")
            h_str = f" (first at +{first_h}h)" if first_h else ""
            st.sidebar.error(f"⚠️ **Very Unhealthy Alert**: Forecast enters Very Unhealthy (201–300){h_str}.")
        elif highest_level == "warning":
            first_h = forecast_alert.get("first_unhealthy_horizon")
            h_str = f" (first at +{first_h}h)" if first_h else ""
            st.sidebar.warning(f"⚠️ **Unhealthy Warning**: Forecast enters Unhealthy (151–200){h_str}.")
        elif highest_level == "advisory":
            first_h = forecast_alert.get("first_advisory_horizon")
            h_str = f" (first at +{first_h}h)" if first_h else ""
            st.sidebar.info(f"ℹ️ **Advisory**: Forecast enters Sensitive Groups (101–150){h_str}.")
        else:
            st.sidebar.success("Good or Moderate air quality across all 72 forecast hours.")


def build_feature_attribution_figure(top_features: list[dict[str, Any]], horizon: int) -> go.Figure:
    """Build horizontal bar chart for top SHAP feature attributions at a specific horizon.

    Args:
        top_features: List of feature attribution dictionaries.
        horizon: Forecast horizon number.

    Returns:
        Configured Plotly Figure.
    """
    if not top_features:
        return go.Figure()

    # Invert so largest magnitude appears at top
    features_rev = [f["feature"] for f in reversed(top_features)]
    shaps_rev = [f["shap_value"] for f in reversed(top_features)]
    raws_rev = [f["raw_value"] for f in reversed(top_features)]
    scaleds_rev = [f["scaled_value"] for f in reversed(top_features)]
    colors = ["#D32F2F" if v > 0 else "#388E3C" for v in shaps_rev]

    custom_data = list(zip(raws_rev, scaleds_rev))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=features_rev,
            x=shaps_rev,
            orientation="h",
            marker=dict(color=colors),
            customdata=custom_data,
            hovertemplate=(
                "<b>Feature:</b> %{y}<br>"
                "<b>SHAP Attribution:</b> %{x:+.2f} AQI points<br>"
                "<b>Raw Input Value:</b> %{customdata[0]:.2f}<br>"
                "<b>Scaled Feature Value:</b> %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Feature Attribution for Horizon +{horizon}h</b><br><sup>Top features moving model prediction relative to baseline reference</sup>",
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title="SHAP Attribution (AQI contribution relative to reference)",
            zeroline=True,
            zerolinecolor="#333",
            zerolinewidth=1.5,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
        ),
        yaxis=dict(
            title="",
            automargin=True,
        ),
        margin=dict(l=10, r=20, t=60, b=40),
        height=380,
    )
    return fig


def render_explainability_section(explanation: dict[str, Any]) -> None:
    """Render interactive model explainability and persistence decomposition section."""
    st.subheader("Model Explainability & Feature Attribution")
    st.caption("SHAP attributions and persistence blending decomposition for EXP-019 hybrid architecture.")

    h = explanation.get("horizon", 24)
    spec_type = explanation.get("specialist_type", "Specialist")
    w = explanation.get("blend_weight", 1.0)
    raw_spec = explanation.get("raw_specialist_output", 0.0)
    m_comp = explanation.get("model_component", 0.0)
    p_comp = explanation.get("persistence_component", 0.0)
    preclip = explanation.get("explained_output_preclip", 0.0)
    pred_aqi = explanation.get("predicted_aqi", 0.0)
    base_val = explanation.get("base_value", 0.0)
    err = explanation.get("additivity_error", 0.0)

    # 1. Decomposition Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Specialist Sub-Model", spec_type)
    with c2:
        st.metric("Blend Weight (w)", f"{w:.2f}")
    with c3:
        st.metric("Model Component", f"{m_comp:.1f}")
    with c4:
        st.metric("Persistence Term", f"{p_comp:.1f}")
    with c5:
        st.metric("Pre-Clip Output", f"{preclip:.1f}", delta=f"Final AQI: {pred_aqi:.1f}")

    # 2. Plotly Waterfall / Attribution Chart
    top_features = explanation.get("top_features", [])
    if top_features:
        fig = build_feature_attribution_figure(top_features, h)
        st.plotly_chart(fig, use_container_width=True)

    # 3. Global Top Features & Persistence Table
    with st.expander("📊 Global Multi-Horizon Feature Importance (500-Sample Stratified Cohort)"):
        p_mean = explanation.get("global_persistence_mean_contribution", 0.0)
        st.markdown(f"**Mean Absolute Persistence Contribution across 72h:** `{p_mean:.2f} AQI points`")

        global_feats = explanation.get("global_top_features", [])
        if global_feats:
            df_global = pd.DataFrame(global_feats)
            st.dataframe(df_global, use_container_width=True, hide_index=True)

