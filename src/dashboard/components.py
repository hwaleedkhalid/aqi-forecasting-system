"""Pearls AQI Predictor - Dashboard UI Components.

Streamlit-native rendering components and Plotly chart builders for the modern
dark-themed environmental intelligence dashboard.

Components use native Streamlit primitives:
  st.container(), st.columns(), st.metric(), st.caption(), st.info(),
  st.warning(), st.error(), st.tabs(), st.expander(), st.plotly_chart().
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure project root and dashboard directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from src.dashboard.ui_helpers import (
        category_bg_color,
        category_border_color,
        category_card_class,
        category_color,
        category_icon,
        category_solid_text_color,
        category_text_color,
        format_horizon_label,
        format_relative_age,
        format_shap_narrative,
        format_stale_age,
        format_timestamp,
        humanize_feature_name,
        safe_val,
        source_badge_class,
        source_display_name,
    )
except ImportError:
    from ui_helpers import (  # type: ignore[no-redef]
        category_bg_color,
        category_border_color,
        category_card_class,
        category_color,
        category_icon,
        category_solid_text_color,
        category_text_color,
        format_horizon_label,
        format_relative_age,
        format_shap_narrative,
        format_stale_age,
        format_timestamp,
        humanize_feature_name,
        safe_val,
        source_badge_class,
        source_display_name,
    )


# ── Speedometer / AQI Gauge (Dark Theme Aligned) ─────────────────────────────

def build_aqi_gauge_figure(current_aqi: float | None, category: str = "") -> go.Figure:
    """Build semi-circular Plotly speedometer/gauge for current observed AQI.

    Shows 6 EPA ranges: Good (0-50), Moderate (51-100), Sensitive (101-150),
    Unhealthy (151-200), Very Unhealthy (201-300), Hazardous (301-500).

    Args:
        current_aqi: Current observed AQI numeric value.
        category: AQI category string for needle/bar accent color.

    Returns:
        Plotly Figure instance with dark theme styling.
    """
    val = float(current_aqi) if current_aqi is not None else 0.0
    accent = category_color(category)

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=val,
            gauge=dict(
                shape="angular",
                axis=dict(
                    range=[0, 500],
                    tickmode="array",
                    tickvals=[0, 50, 100, 150, 200, 300, 500],
                    ticktext=["0", "50", "100", "150", "200", "300", "500"],
                    tickfont=dict(size=9, color="#94A3B8"),
                ),
                bar=dict(color=accent, thickness=0.32),
                bgcolor="rgba(255, 255, 255, 0.05)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 50], color="rgba(34, 197, 94, 0.45)"),
                    dict(range=[50, 100], color="rgba(234, 179, 8, 0.45)"),
                    dict(range=[100, 150], color="rgba(249, 115, 22, 0.45)"),
                    dict(range=[150, 200], color="rgba(239, 68, 68, 0.45)"),
                    dict(range=[200, 300], color="rgba(139, 92, 246, 0.45)"),
                    dict(range=[300, 500], color="rgba(127, 29, 29, 0.65)"),
                ],
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=10, b=0),
        height=140,
    )
    return fig


# ── Forecast chart (Dark Theme Aligned) ──────────────────────────────────────

def build_forecast_figure(forecasts: list[dict[str, Any]]) -> go.Figure:
    """Build interactive Plotly figure for the 72-hour AQI forecast.

    Preserves all scientific content: point values, empirical error band,
    151 / 201 / 301 EPA threshold reference lines, UTC timestamps.
    Dark navy background aligned with environmental command center theme.

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

    # 1. Shaded empirical prediction error interval band
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
    fig.add_trace(
        go.Scatter(
            x=times,
            y=lowers,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.15)",
            name="Empirical Error Range (Walk-forward 10th–90th out-of-fold residuals)",
            hoverinfo="skip",
        )
    )

    # 2. Predicted AQI main curve (High-contrast cyan line)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=aqis,
            mode="lines+markers",
            name="Predicted AQI (EXP-019)",
            line=dict(color="#38BDF8", width=2.5),
            marker=dict(size=4, color="#38BDF8"),
            customdata=custom_data,
            hovertemplate=(
                "<b>Horizon:</b> +%{customdata[0]}h<br>"
                "<b>Forecast Time:</b> %{x}<br>"
                "<b>Predicted AQI:</b> %{y:.1f}<br>"
                "<b>Category:</b> %{customdata[1]}<br>"
                "<b>Empirical Error Range:</b> [%{customdata[2]:.1f}, %{customdata[3]:.1f}]"
                "<extra></extra>"
            ),
        )
    )

    # 3. EPA category boundary reference lines (151 / 201 / 301 — required)
    fig.add_hline(
        y=151,
        line_dash="dash",
        line_color="#EF4444",
        line_width=1.2,
        annotation_text="Unhealthy (151)",
        annotation_position="top left",
        annotation_font=dict(color="#F87171", size=9),
    )
    fig.add_hline(
        y=201,
        line_dash="dash",
        line_color="#8B5CF6",
        line_width=1.2,
        annotation_text="Very Unhealthy (201)",
        annotation_position="top left",
        annotation_font=dict(color="#A78BFA", size=9),
    )
    fig.add_hline(
        y=301,
        line_dash="dash",
        line_color="#7F1D1D",
        line_width=1.2,
        annotation_text="Hazardous (301)",
        annotation_position="top left",
        annotation_font=dict(color="#FCA5A5", size=9),
    )

    # 4. Background colored category bands (subtle dark opacity)
    max_val = max(max(uppers), max(aqis), 350.0)
    zones = [
        (0, 50, "rgba(34, 197, 94, 0.03)"),
        (50, 100, "rgba(234, 179, 8, 0.03)"),
        (100, 150, "rgba(249, 115, 22, 0.03)"),
        (150, 200, "rgba(239, 68, 68, 0.03)"),
        (200, 300, "rgba(139, 92, 246, 0.03)"),
        (300, max(500.0, max_val + 20.0), "rgba(127, 29, 29, 0.06)"),
    ]
    for y0, y1, fill in zones:
        fig.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=fill,
            line_width=0,
            layer="below",
        )

    # 5. Layout (Dark theme with crisp contrast)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A1628",
        title=dict(text="", font=dict(size=1)),
        xaxis=dict(
            title="Forecast Target Time (UTC)",
            showgrid=True,
            gridcolor="rgba(34, 50, 74, 0.6)",
            tickfont=dict(size=10, color="#94A3B8"),
            title_font=dict(size=11, color="#94A3B8"),
        ),
        yaxis=dict(
            title="Air Quality Index (AQI)",
            rangemode="nonnegative",
            range=[0, max_val + 20],
            showgrid=True,
            gridcolor="rgba(34, 50, 74, 0.6)",
            tickfont=dict(size=10, color="#94A3B8"),
            title_font=dict(size=11, color="#94A3B8"),
        ),
        legend=dict(
            title=dict(text=""),
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            font=dict(size=10, color="#94A3B8"),
        ),
        margin=dict(l=50, r=30, t=30, b=50),
        hovermode="x unified",
        height=420,
    )

    return fig


# ── Freshness / Alert banners ─────────────────────────────────────────────────

def render_freshness_banner(is_stale: bool, observed_at: str, age_hours: float) -> None:
    """Render explicit warning banner when input telemetry is historical."""
    if is_stale:
        st.warning(
            f"⚠️ **Telemetry Notice**: Latest available observation is historical "
            f"(Observed: `{observed_at}`, Age: **{age_hours:.1f} hours**). "
            f"Forecasts are anchored to this historical observation origin, not current browser time."
        )


def render_alert_banners(obs_data: dict[str, Any], forecast_data: dict[str, Any]) -> None:
    """Render a compact, unified alert area using native Streamlit alert components.

    Priority hierarchy (semantics preserved from alerting.py):
    1. Current Severe / Hazardous → st.error
    2. Forecast Severe / Hazardous → st.error
    3. Current or Forecast Unhealthy warning → st.warning (only if no severe above)
    4. Current or Forecast Advisory → st.info (only if no warning above)
    5. Stale notice → compact st.caption line
    6. Uncertainty tail-risk → st.expander (collapsed by default)

    Args:
        obs_data: Latest observation dictionary matching API contract.
        forecast_data: Forecast result dictionary matching API contract.
    """
    # --- Extract observation alert ---
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

    # --- Extract forecast alert ---
    f_alert = (
        forecast_data.get("forecast_alert")
        or forecast_data.get("summary", {}).get("forecast_alert")
        or {}
    )
    f_level = f_alert.get("highest_level", "none")
    f_peak = f_alert.get("peak_aqi")
    f_peak_h = f_alert.get("peak_horizon")
    f_msg = f_alert.get("message", "")

    # --- Determine highest overall severity ---
    _rank = {"none": 0, "advisory": 1, "warning": 2, "severe": 3, "hazardous": 4}
    obs_rank = _rank.get(obs_level, 0)
    f_rank = _rank.get(f_level, 0)
    highest_rank = max(obs_rank, f_rank)

    # --- Render single primary banner ---
    if highest_rank >= 4:  # hazardous
        if obs_rank >= 4:
            st.error(
                f"🚨 **Hazardous Air Quality Emergency** — Current AQI **{obs_aqi:.0f}** "
                f"({obs_cat}). {obs_msg}"
            )
        else:
            peak_str = f" · Peak AQI {f_peak:.0f} at {format_horizon_label(f_peak_h)}" if f_peak else ""
            st.error(f"🚨 **Hazardous AQI Forecast**{peak_str} — {f_msg}")

    elif highest_rank == 3:  # severe (very unhealthy)
        if obs_rank >= 3:
            st.error(
                f"⚠️ **Very Unhealthy Air Quality** — Current AQI **{obs_aqi:.0f}** "
                f"({obs_cat}). {obs_msg}"
            )
        else:
            peak_str = f" · Peak AQI {f_peak:.0f} at {format_horizon_label(f_peak_h)}" if f_peak else ""
            st.error(f"⚠️ **Very Unhealthy AQI Forecast**{peak_str} — {f_msg}")

    elif highest_rank == 2:  # warning (unhealthy)
        if obs_rank >= 2:
            st.warning(
                f"⚠️ **Unhealthy Air Quality** — Current AQI **{obs_aqi:.0f}** "
                f"({obs_cat}). {obs_msg}"
            )
        else:
            peak_str = f" · Peak {f_peak:.0f} AQI at {format_horizon_label(f_peak_h)}" if f_peak else ""
            st.warning(f"⚠️ **Unhealthy AQI Forecast**{peak_str} — {f_msg}")

    elif highest_rank == 1:  # advisory
        if obs_rank >= 1:
            st.info(
                f"ℹ️ **Air Quality Advisory** — Current AQI **{obs_aqi:.0f}** "
                f"({obs_cat}). {obs_msg}"
            )
        else:
            st.info(f"ℹ️ **Air Quality Advisory Forecast** — {f_msg}")

    # --- Stale compact notice (always below primary banner if present) ---
    is_stale = forecast_data.get("is_stale", obs_data.get("is_stale", False))
    age_hours = forecast_data.get("input_age_hours", obs_data.get("input_age_hours", 0.0))
    if is_stale:
        observed_at = forecast_data.get(
            "input_observed_at", obs_data.get("input_observed_at", "Unknown")
        )
        st.caption(
            f"🕒 Historical observation · {age_hours:.1f}h old "
            f"(Observed: {format_timestamp(observed_at)}). "
            "Forecasts anchored to this observation, not current time."
        )

    # --- Uncertainty tail-risk (collapsed expander) ---
    if f_alert.get("upper_interval_crosses_hazardous") and highest_rank < 4:
        with st.expander("⚠️ Forecast uncertainty notice", expanded=False):
            st.info(
                "The 90th-percentile empirical error interval exceeds the Hazardous "
                "threshold (>300 AQI) at one or more forecast horizons, indicating elevated "
                "extreme pollution tail risk. Monitor ongoing hourly updates."
            )


# ── Current observation card (LEFT hero column) ───────────────────────────────

def render_current_observation_card(obs: dict[str, Any]) -> None:
    """Render the left hero column with native Streamlit container & metrics.

    Args:
        obs: Latest observation dictionary matching API contract.
    """
    aqi_val = obs.get("current_aqi")
    category = obs.get("category", "Unknown")
    dominant = obs.get("dominant_pollutant", "pm2_5")
    feature_source = obs.get("feature_source", "")
    fallback_active = obs.get("fallback_active", False)
    is_stale = obs.get("is_stale", False)
    age_hours = obs.get("input_age_hours")

    source_name = source_display_name(feature_source, fallback_active, is_stale)
    if is_stale:
        freshness_line = format_stale_age(age_hours)
    else:
        freshness_line = f"Updated {format_relative_age(age_hours)}"

    aqi_display = f"{aqi_val:.0f}" if aqi_val is not None else "—"

    # Extract dominant pollutant concentration and weather
    pollutants = obs.get("pollutants", {})
    weather = obs.get("weather", {})
    dom_key = dominant.lower().replace(".", "_")
    dom_val = pollutants.get(dom_key, pollutants.get("pm2_5"))
    dom_val_str = safe_val(dom_val, "µg/m³", 1)
    temp_str = safe_val(weather.get("temperature_2m"), "°C", 1)
    wind_str = safe_val(weather.get("wind_speed_10m"), "m/s", 1)

    icon = category_icon(category)
    gauge_fig = build_aqi_gauge_figure(float(aqi_val) if aqi_val is not None else 0.0, category)

    st.markdown(
        f"### Current Air Quality · Lahore\n\n"
        f"**Observed AQI:** `{aqi_display}` ({icon} {category})  \n"
        f"**What is affecting air quality now?**  \n"
        f"• **Primary factor:** {dominant.upper().replace('_', '.')} ({dom_val_str})  \n"
        f"• **Weather telemetry:** Temp {temp_str} · Wind {wind_str}  \n"
        f"*{freshness_line} · Source: {source_name}*"
    )
    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})


# ── 72h Outlook card (RIGHT hero column) ─────────────────────────────────────

def render_72h_outlook_card(
    forecast_data: dict[str, Any],
    current_category: str = "",
) -> None:
    """Render the right hero column with native Streamlit container and metrics.

    Args:
        forecast_data: Forecast result dictionary matching API contract.
        current_category: Category of current observation for progression.
    """
    f_alert = (
        forecast_data.get("forecast_alert")
        or forecast_data.get("summary", {}).get("forecast_alert")
        or {}
    )
    summary = forecast_data.get("summary", {})

    peak_aqi = f_alert.get("peak_aqi") or summary.get("peak_aqi")
    peak_cat = f_alert.get("peak_category") or summary.get("peak_category", "—")
    peak_h = f_alert.get("peak_horizon") or summary.get("peak_horizon")
    first_unhealthy = f_alert.get("first_unhealthy_horizon")
    first_very = f_alert.get("first_very_unhealthy_horizon")
    first_haz = f_alert.get("first_hazardous_horizon")

    peak_aqi_str = f"{peak_aqi:.0f}" if peak_aqi is not None else "—"
    curr_cat = current_category or "Moderate"
    peak_h_tag = f"+{peak_h}h" if peak_h is not None else "72h"

    rows = [
        f"• **Highest Category:** `{peak_cat}`"
    ]
    if first_unhealthy:
        rows.append(f"• **First Unhealthy Hour:** `{format_horizon_label(first_unhealthy)}`")
    if first_very:
        rows.append(f"• **First Very Unhealthy Hour:** `{format_horizon_label(first_very)}`")
    if first_haz:
        rows.append(f"• **First Hazardous Hour:** `{format_horizon_label(first_haz)}`")
    if not first_unhealthy and not first_very and not first_haz:
        rows.append("• **72h Status:** `✅ Good / Moderate throughout`")

    if peak_h:
        rows.append(f"• **Peak Horizon:** `{format_horizon_label(peak_h)}`")

    rows_str = "  \n".join(rows)

    st.markdown(
        f"### 72-Hour Outlook\n\n"
        f"**Peak Predicted AQI:** `{peak_aqi_str}` ({peak_cat})  \n"
        f"**Progression:** `Now ({curr_cat})` ──→ `Peak ({peak_cat} at {peak_h_tag})`  \n\n"
        f"{rows_str}"
    )


# ── Compact status / metadata line ────────────────────────────────────────────

def render_metadata_header(
    forecast_origin: str,
    generated_at: str,
    source_mode: str,
    latency_ms: float,
) -> None:
    """Render compact user-friendly status line.

    Args:
        forecast_origin: ISO timestamp of observation used as forecast anchor.
        generated_at: ISO timestamp when forecast was generated.
        source_mode: Data source mode string.
        latency_ms: Inference latency in milliseconds.
    """
    from datetime import datetime, timezone

    age_str = "—"
    try:
        dt = datetime.fromisoformat(forecast_origin)
        now = datetime.now(timezone.utc)
        age_h = (now - dt.astimezone(timezone.utc)).total_seconds() / 3600.0
        if age_h < 0:
            age_h = 0.0
        age_str = format_relative_age(age_h)
    except Exception:
        age_str = format_timestamp(forecast_origin)

    if "REST API" in (source_mode or "") or source_mode == "REST API":
        src_label = "Hopsworks"
    elif "Direct Local" in (source_mode or ""):
        src_label = "Local inference"
    else:
        src_label = source_mode or "Unknown"

    st.markdown(f"🟢 **Status:** Updated {age_str} · {src_label} · Latency: {latency_ms:.1f} ms")


# ── Telemetry breakdown (Pollutants + Weather) ────────────────────────────────

def render_telemetry_breakdown(obs: dict[str, Any]) -> None:
    """Render compact pollutant and weather metrics using native Streamlit layout.

    Args:
        obs: Latest observation dictionary matching API contract.
    """
    pollutants = obs.get("pollutants", {})
    weather = obs.get("weather", {})

    col_p, col_w = st.columns(2, gap="medium")

    with col_p:
        pm25 = safe_val(pollutants.get("pm2_5"), "µg/m³", 1)
        pm10 = safe_val(pollutants.get("pm10"), "µg/m³", 1)
        no2 = safe_val(pollutants.get("no2"), "µg/m³", 1)
        so2 = safe_val(pollutants.get("so2"), "µg/m³", 1)
        co = safe_val(pollutants.get("co"), "µg/m³", 1)
        o3 = safe_val(pollutants.get("o3"), "µg/m³", 1)
        st.markdown(
            f"**Air Pollutants**  \n"
            f"• **PM2.5:** `{pm25}` · **PM10:** `{pm10}`  \n"
            f"• **NO₂:** `{no2}` · **SO₂:** `{so2}`  \n"
            f"• **O₃:** `{o3}` · **CO:** `{co}`"
        )

    with col_w:
        temp = safe_val(weather.get("temperature_2m"), "°C", 1)
        hum = safe_val(weather.get("relative_humidity_2m"), "%", 0)
        wind = safe_val(weather.get("wind_speed_10m"), "m/s", 1)
        pres = safe_val(weather.get("surface_pressure"), "hPa", 1)
        precip = safe_val(weather.get("precipitation"), "mm", 1) if weather.get("precipitation") is not None else None
        precip_str = f" · **Precip.:** `{precip}`" if precip is not None else ""
        st.markdown(
            f"**Weather Conditions**  \n"
            f"• **Temperature:** `{temp}` · **Humidity:** `{hum}`  \n"
            f"• **Wind:** `{wind}` · **Pressure:** `{pres}`{precip_str}"
        )


# ── Health Guidance section ───────────────────────────────────────────────────

def render_health_guidance(obs_data: dict[str, Any], forecast_data: dict[str, Any]) -> None:
    """Render Health Guidance card using category-level advisory text.

    Args:
        obs_data: Latest observation dictionary matching API contract.
        forecast_data: Forecast result dictionary matching API contract.
    """
    category = obs_data.get("category", "Unknown")
    advisory = obs_data.get("health_advisory", "")
    icon = category_icon(category)

    # Check if forecast is worse than current
    _rank = {"none": 0, "advisory": 1, "warning": 2, "severe": 3, "hazardous": 4}
    obs_alert = obs_data.get("alert") or {}
    obs_rank = _rank.get(obs_alert.get("level", "none"), 0)

    f_alert = (
        forecast_data.get("forecast_alert")
        or forecast_data.get("summary", {}).get("forecast_alert")
        or {}
    )
    f_level = f_alert.get("highest_level", "none")
    f_rank = _rank.get(f_level, 0)
    f_peak_cat = f_alert.get("peak_category", "")
    f_peak_h = f_alert.get("peak_horizon")

    upgrade_text = ""
    if f_rank > obs_rank and f_rank >= 2:
        peak_str = f" (peak at {format_horizon_label(f_peak_h)})" if f_peak_h else ""
        upgrade_text = (
            f"\n\n📈 **Forecast outlook:** Air quality is expected to reach **{f_peak_cat}** "
            f"conditions within 72 hours{peak_str}. Monitor updates and take precautions."
        )

    st.markdown(
        f"#### {icon} Health Guidance · {category}\n\n"
        f"{advisory or 'Air quality information is currently unavailable.'}"
        f"{upgrade_text}"
    )


# ── Model & System Details expander ──────────────────────────────────────────

def render_model_system_details(
    model_info: dict[str, Any],
    forecast_data: dict[str, Any],
    obs_data: dict[str, Any],
    source_mode: str,
) -> None:
    """Render collapsed Model & System Details expander with all technical provenance.

    Args:
        model_info: Model metadata dictionary matching API contract.
        forecast_data: Forecast result dictionary matching API contract.
        obs_data: Latest observation dictionary.
        source_mode: Data source mode string.
    """
    with st.expander("🔬 Model & System Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Champion Model**")
            st.markdown(f"- **ID:** `{model_info.get('model_id', 'EXP-019')}`")
            st.markdown(f"- **Architecture:** `{model_info.get('architecture', 'Hybrid Specialist')}`")
            st.markdown(f"- **Input Dimensionality:** `{model_info.get('feature_count', 114)} canonical features`")
            st.markdown(f"- **Schema Version:** `{model_info.get('feature_schema_version', 'v2_weather_enriched')}`")
            st.markdown(f"- **Status:** `{model_info.get('status', 'validated_champion')}`")
            st.markdown(f"- **Data Pipeline:** `{source_mode}`")
            st.markdown(f"- **Feature Source:** `{obs_data.get('feature_source', '—')}`")

            benchmarks = model_info.get("test_benchmark_metrics", {})
            if benchmarks:
                st.markdown("**Held-Out Test Benchmarks (Phase 10.5E)**")
                if benchmarks.get("partition"):
                    st.markdown(f"- **Partition:** `{benchmarks.get('partition')}`")
                st.markdown(f"- **Overall RMSE:** `{benchmarks.get('overall_rmse')} AQI`")
                st.markdown(f"- **Overall MAE:** `{benchmarks.get('overall_mae')} AQI`")
                st.markdown(f"- **Overall R²:** `{benchmarks.get('overall_r2')}`")
                st.markdown(f"- **h+1 RMSE:** `{benchmarks.get('h1_rmse')} AQI`")
                st.markdown(f"- **h+72 RMSE:** `{benchmarks.get('h72_rmse')} AQI`")
                if benchmarks.get("hazardous_gt300_rmse"):
                    st.markdown(f"- **Hazardous (>300 AQI) RMSE:** `{benchmarks.get('hazardous_gt300_rmse')} AQI`")

        with col2:
            st.markdown("**Forecast Provenance**")
            forecast_origin = forecast_data.get("forecast_origin", "—")
            generated_at = forecast_data.get("generated_at", "—")
            latency_ms = forecast_data.get("inference_latency_ms", 0.0)
            input_observed = forecast_data.get(
                "input_observed_at", obs_data.get("input_observed_at", "—")
            )
            age_hours = forecast_data.get(
                "input_age_hours", obs_data.get("input_age_hours", None)
            )
            st.markdown(f"- **Forecast Origin:** `{format_timestamp(forecast_origin)}`")
            st.markdown(f"- **Input Observed At:** `{format_timestamp(input_observed)}`")
            st.markdown(f"- **Input Age:** `{safe_val(age_hours, 'h', 1)}`")
            st.markdown(f"- **Generated At:** `{format_timestamp(generated_at)}`")
            st.markdown(f"- **Inference Latency:** `{latency_ms:.1f} ms`")

            summary = forecast_data.get("summary", {})
            if summary:
                st.markdown("**72-Hour Forecast Summary**")
                st.markdown(f"- **Peak AQI:** `{safe_val(summary.get('peak_aqi'), decimals=1)}`")
                st.markdown(f"- **Peak Horizon:** `{format_horizon_label(summary.get('peak_horizon'))}`")
                st.markdown(f"- **Peak Category:** `{summary.get('peak_category', '—')}`")


# ── Cold-start / connection error state ──────────────────────────────────────

def render_cold_start_error(error: Exception) -> None:
    """Render user-friendly service-waking-up screen."""
    st.markdown(
        "### 🌫️ Forecast service is waking up\n\n"
        "The forecasting service is starting from standby. "
        "This usually takes less than a minute. Click **Retry** to check again."
    )
    if st.button("🔄 Retry", help="Check if the service is back online"):
        st.rerun()
    with st.expander("Technical details", expanded=False):
        st.code(str(error))


# ── Sidebar (no-op stub — content moved to Model & System Details) ────────────

def render_sidebar(model_info: dict[str, Any], summary: dict[str, Any], source_mode: str) -> None:
    """No-op stub. Preserved for backward compatibility with existing test imports."""
    pass


# ── SHAP / Feature attribution (Dark Theme Aligned) ──────────────────────────

def build_feature_attribution_figure(top_features: list[dict[str, Any]], horizon: int) -> go.Figure:
    """Build horizontal bar chart for top SHAP feature attributions at a specific horizon.

    Args:
        top_features: List of feature attribution dictionaries.
        horizon: Forecast horizon number.

    Returns:
        Configured Plotly Figure with dark theme styling.
    """
    if not top_features:
        return go.Figure()

    # Invert so largest magnitude appears at top
    features_rev = [f["feature"] for f in reversed(top_features)]
    shaps_rev = [f["shap_value"] for f in reversed(top_features)]
    raws_rev = [f["raw_value"] for f in reversed(top_features)]
    scaleds_rev = [f["scaled_value"] for f in reversed(top_features)]
    colors = ["#EF4444" if v > 0 else "#22C55E" for v in shaps_rev]

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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0A1628",
        title=None,
        xaxis=dict(
            title="SHAP Attribution (AQI contribution relative to reference)",
            zeroline=True,
            zerolinecolor="#475569",
            zerolinewidth=1.2,
            showgrid=True,
            gridcolor="rgba(34, 50, 74, 0.6)",
            tickfont=dict(size=10, color="#94A3B8"),
            title_font=dict(size=10, color="#94A3B8"),
        ),
        yaxis=dict(
            title="",
            automargin=True,
            tickfont=dict(size=10, color="#CBD5E1"),
        ),
        margin=dict(l=10, r=20, t=20, b=40),
        height=360,
    )
    return fig


def render_explainability_section(explanation: dict[str, Any]) -> None:
    """Render model explainability and persistence decomposition.

    Called inside the 'Forecast Drivers' tab of Advanced Insights.

    Args:
        explanation: Explanation dictionary from /api/explain endpoint.
    """
    h = explanation.get("horizon", 24)
    spec_type = explanation.get("specialist_type", "Specialist")
    w = explanation.get("blend_weight", 1.0)
    m_comp = explanation.get("model_component", 0.0)
    p_comp = explanation.get("persistence_component", 0.0)
    preclip = explanation.get("explained_output_preclip", 0.0)
    pred_aqi = explanation.get("predicted_aqi", 0.0)
    base_val = explanation.get("base_value", 0.0)

    # Decomposition compact row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Sub-Model", spec_type)
    with c2:
        st.metric("Blend Weight (w)", f"{w:.2f}")
    with c3:
        st.metric("Model Component", safe_val(m_comp, decimals=1))
    with c4:
        st.metric("Persistence Term", safe_val(p_comp, decimals=1))
    with c5:
        st.metric("Pre-Clip Output", safe_val(preclip, decimals=1), delta=f"Final: {pred_aqi:.1f}")

    # SHAP chart
    top_features = explanation.get("top_features", [])
    if top_features:
        st.caption(
            f"Top features driving the horizon +{h}h prediction relative to the model's "
            f"baseline reference (SHAP, base value: {base_val:.1f})."
        )
        fig = build_feature_attribution_figure(top_features, h)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature attribution data not available for this horizon.")

    # Global importance
    with st.expander("📊 Global Multi-Horizon Feature Importance (500-Sample Stratified Cohort)"):
        p_mean = explanation.get("global_persistence_mean_contribution", 0.0)
        st.markdown(f"**Mean Absolute Persistence Contribution across 72h:** `{p_mean:.2f} AQI points`")
        global_feats = explanation.get("global_top_features", [])
        if global_feats:
            df_global = pd.DataFrame(global_feats)
            st.dataframe(df_global, use_container_width=True, hide_index=True)


# ── Compact AQI Legend ────────────────────────────────────────────────────────

def render_aqi_legend() -> None:
    """Render compact horizontal 6-category AQI scale legend."""
    st.markdown(
        "🟢 **0–50** Good &nbsp;·&nbsp; "
        "🟡 **51–100** Moderate &nbsp;·&nbsp; "
        "🟠 **101–150** Sensitive &nbsp;·&nbsp; "
        "🔴 **151–200** Unhealthy &nbsp;·&nbsp; "
        "🟣 **201–300** Very Unhealthy &nbsp;·&nbsp; "
        "🟤 **301+** Hazardous"
    )


# ── "Why this forecast?" Human-readable SHAP summary card ────────────────────

def render_forecast_narrative_card(
    explain_data: dict[str, Any],
    horizon: int = 24,
) -> None:
    """Render the human-readable 'Why this forecast?' SHAP summary card.

    Translates top model attributions into plain English directional statements
    without claiming direct physical causality.

    Args:
        explain_data: Explanation dictionary from /api/explain or predictor.
        horizon: Horizon number being explained.
    """
    if not explain_data:
        return

    top_features = explain_data.get("top_features", [])
    if not top_features:
        return

    narratives = format_shap_narrative(top_features[:4], default_horizon=horizon)
    if not narratives:
        return

    statements = []
    for n in narratives:
        arrow = n["arrow"]
        name = n["human_name"]
        direction = n["direction"]
        shap = n["shap_value"]
        sign = "+" if n["is_upward"] else ""
        strength = n["strength"]
        statements.append(
            f"• **{arrow} {name}** contributed **{direction} pressure** "
            f"(`{sign}{shap:.1f} AQI`) · *{strength} impact*"
        )

    statements_str = "  \n".join(statements)

    st.markdown(
        f"#### 🔍 Why this forecast? (+{horizon}h Key Drivers)\n\n"
        f"{statements_str}\n\n"
        f"**Attribution relative to model reference**  \n"
        f"*Attributions indicate statistical feature influence within the EXP-019 hybrid model, not direct physical causality.*"
    )
