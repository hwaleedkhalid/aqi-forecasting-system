"""Pearls AQI Predictor - Dashboard Minimal Styles.

Minimal CSS adjustments for Streamlit native components.
All primary application theming is driven by .streamlit/config.toml.
"""

DASHBOARD_CSS: str = """
<style>
/* Hide sidebar controls for single-page environmental command center */
[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
</style>
"""
