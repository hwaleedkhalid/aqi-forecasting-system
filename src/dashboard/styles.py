"""Pearls AQI Predictor - Dashboard Design Tokens and CSS.

Single-source CSS stylesheet injected once at startup via st.markdown.
All design decisions live here; components reference CSS class names only.
"""

DASHBOARD_CSS: str = """
<style>
/* ── Global reset / page background ─────────────────────────── */
[data-testid="stAppViewContainer"] > .main {
    background: #F7F8FA;
}
[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Base typography ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: #1F2937;
}

/* ── App header ───────────────────────────────────────────────── */
.prl-app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 12px 0;
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 20px;
}
.prl-app-header-left {}
.prl-app-logo {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1E3A5F;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.prl-app-subtitle {
    font-size: 0.88rem;
    color: #6B7280;
    margin-top: 2px;
}
.prl-app-header-right {
    text-align: right;
    font-size: 0.82rem;
    color: #6B7280;
    line-height: 1.5;
}
.prl-location-tag {
    font-weight: 600;
    color: #374151;
}

/* ── Cards (generic and category-tinted surfaces) ────────────── */
.prl-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
    height: 100%;
    transition: all 0.2s ease-in-out;
}
.prl-card-good {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
}
.prl-card-moderate {
    background: #FEFCE8;
    border: 1px solid #FEF08A;
}
.prl-card-usg {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
}
.prl-card-unhealthy {
    background: #FEF2F2;
    border: 1px solid #FECACA;
}
.prl-card-very-unhealthy {
    background: #FAF5FF;
    border: 1px solid #E9D5FF;
}
.prl-card-hazardous {
    background: #FFF1F2;
    border: 1px solid #FFE4E6;
}
.prl-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #4B5563;
    margin-bottom: 8px;
}

/* ── Hero: Current AQI ────────────────────────────────────────── */
.prl-hero-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
.prl-hero-aqi {
    font-size: 3.6rem;
    font-weight: 800;
    line-height: 1;
    color: #1F2937;
    letter-spacing: -2px;
}
.prl-hero-aqi-unit {
    font-size: 1.1rem;
    font-weight: 500;
    color: #6B7280;
    margin-left: 4px;
    letter-spacing: 0;
}
.prl-category-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.84rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.02em;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.prl-hero-meta {
    font-size: 0.82rem;
    color: #4B5563;
    margin-top: 10px;
    line-height: 1.6;
}
.prl-source-badge {
    display: inline-block;
    background: #ECFDF5;
    color: #065F46;
    border: 1px solid #A7F3D0;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
}
.prl-source-badge-stale {
    background: #FFFBEB;
    color: #92400E;
    border-color: #FCD34D;
}
.prl-source-badge-fallback {
    background: #FFF7ED;
    color: #9A3412;
    border-color: #FDBA74;
}

/* ── What is affecting air quality now panel ─────────────────── */
.prl-observed-factors {
    margin-top: 12px;
    padding: 10px 14px;
    background: rgba(255, 255, 255, 0.85);
    border-radius: 12px;
    border: 1px solid rgba(0, 0, 0, 0.08);
    font-size: 0.82rem;
    color: #374151;
    line-height: 1.5;
}
.prl-observed-factors-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #4B5563;
    margin-bottom: 4px;
}

/* ── Hero: 72h Outlook ───────────────────────────────────────── */
.prl-outlook-peak {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1F2937;
    line-height: 1.1;
    letter-spacing: -1px;
}
.prl-outlook-label {
    font-size: 0.78rem;
    color: #6B7280;
    font-weight: 500;
    margin-bottom: 2px;
}
.prl-outlook-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1F2937;
}
.prl-horizon-tag {
    display: inline-block;
    background: #F3F4F6;
    border-radius: 8px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #374151;
    margin-left: 4px;
}
.prl-no-outlook {
    font-size: 0.9rem;
    color: #059669;
    font-weight: 600;
}

/* ── Future-risk progression indicator ───────────────────────── */
.prl-progression-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin: 12px 0 8px 0;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.85);
    border-radius: 10px;
    border: 1px solid rgba(0, 0, 0, 0.08);
}
.prl-progression-step {
    display: flex;
    flex-direction: column;
}
.prl-progression-tag {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 2px;
}
.prl-progression-arrow {
    font-size: 1.1rem;
    color: #9CA3AF;
    font-weight: 700;
    padding: 0 4px;
}

/* ── Compact status / freshness line ─────────────────────────── */
.prl-status-line {
    font-size: 0.82rem;
    color: #6B7280;
    padding: 6px 0 0 0;
}
.prl-status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #10B981;
    margin-right: 5px;
    vertical-align: middle;
}
.prl-status-dot-stale {
    background: #F59E0B;
}

/* ── Compact AQI Legend ──────────────────────────────────────── */
.prl-aqi-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: center;
    align-items: center;
    margin: 10px 0 16px 0;
    padding: 8px 14px;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
}
.prl-legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.74rem;
    color: #4B5563;
    font-weight: 500;
    padding: 2px 6px;
    border-radius: 6px;
}
.prl-legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

/* ── Milestone cards ──────────────────────────────────────────── */
.prl-milestone-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 14px 12px 12px 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
    transition: transform 0.15s ease-in-out;
}
.prl-milestone-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 14px 14px 0 0;
}
.prl-milestone-horizon {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6B7280;
    margin-top: 6px;
}
.prl-milestone-aqi {
    font-size: 1.95rem;
    font-weight: 800;
    color: #1F2937;
    letter-spacing: -1px;
    margin: 4px 0 2px 0;
    line-height: 1;
}
.prl-milestone-cat {
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.prl-milestone-range {
    font-size: 0.68rem;
    color: #6B7280;
}

/* ── "Why this forecast?" Human-readable SHAP Narrative Card ── */
.prl-narrative-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 18px 22px;
    margin-top: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.prl-narrative-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
}
.prl-narrative-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1F2937;
}
.prl-shap-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    margin-bottom: 6px;
    background: #F9FAFB;
    border-radius: 8px;
    border: 1px solid #F3F4F6;
    font-size: 0.83rem;
    gap: 8px;
}
.prl-shap-statement {
    color: #374151;
    font-weight: 500;
}
.prl-shap-badge {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    white-space: nowrap;
}
.prl-shap-badge-strong {
    background: #FEE2E2;
    color: #991B1B;
}
.prl-shap-badge-mod {
    background: #FEF3C7;
    color: #92400E;
}
.prl-shap-badge-minor {
    background: #E5E7EB;
    color: #4B5563;
}
.prl-narrative-disclaimer {
    font-size: 0.74rem;
    color: #9CA3AF;
    margin-top: 10px;
    font-style: italic;
}

/* ── Pollutant / Weather compact grid ────────────────────────── */
.prl-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(92px, 1fr));
    gap: 8px;
    margin-top: 6px;
}
.prl-metric-pill {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 10px 6px;
    text-align: center;
    min-width: 0;
}
.prl-metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    margin-bottom: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prl-metric-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1F2937;
}
.prl-metric-unit {
    font-size: 0.65rem;
    color: #9CA3AF;
    font-weight: 500;
}

/* ── Health Guidance card ─────────────────────────────────────── */
.prl-guidance-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 20px 24px;
    margin-top: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.prl-guidance-icon {
    font-size: 1.4rem;
    margin-bottom: 6px;
    display: block;
}
.prl-guidance-headline {
    font-size: 1rem;
    font-weight: 700;
    color: #1F2937;
    margin-bottom: 4px;
}
.prl-guidance-body {
    font-size: 0.88rem;
    color: #374151;
    line-height: 1.55;
}
.prl-guidance-upgrade {
    margin-top: 12px;
    padding: 10px 14px;
    background: #FFF7ED;
    border-left: 3px solid #F59E0B;
    border-radius: 6px;
    font-size: 0.84rem;
    color: #92400E;
    line-height: 1.45;
}

/* ── Section headings ─────────────────────────────────────────── */
.prl-section-heading {
    font-size: 1.08rem;
    font-weight: 700;
    color: #1F2937;
    margin: 24px 0 6px 0;
}
.prl-section-sub {
    font-size: 0.82rem;
    color: #6B7280;
    margin-bottom: 12px;
}

/* ── Cold-start / service unavailable ────────────────────────── */
.prl-cold-start {
    text-align: center;
    padding: 60px 40px;
}
.prl-cold-start-icon {
    font-size: 3rem;
    margin-bottom: 12px;
    display: block;
}
.prl-cold-start-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1F2937;
    margin-bottom: 8px;
}
.prl-cold-start-body {
    font-size: 0.9rem;
    color: #6B7280;
    max-width: 400px;
    margin: 0 auto 20px auto;
    line-height: 1.5;
}

/* ── Compact alert area ───────────────────────────────────────── */
.prl-alert-compact {
    font-size: 0.85rem;
}
</style>
"""
