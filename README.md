# Pearls AQI Predictor

Serverless 72-hour Air Quality Index forecasting system for Lahore, Pakistan (31.55°N, 74.34°E). The project predicts US EPA AQI (0-500 scale) for 72 continuous hourly horizons using data sourced from OpenWeather Air Pollution History API (Nov 2020 to present) and the Open-Meteo Historical Weather Archive.

The production champion model is EXP-019 (Persistence-Aware Hybrid Model).

## Live Demo

* **Frontend Dashboard**: [https://aqi-forecasting.streamlit.app](https://aqi-forecasting.streamlit.app) (Hosted on Streamlit Community Cloud)
* **Backend REST API**: [https://aqi-forecasting-xyyb.onrender.com/api](https://aqi-forecasting-xyyb.onrender.com/api) (Hosted on Render)

*Note: These represent the verified deployment endpoints at final project deployment and testing time. Third-party hosting availability is not permanently guaranteed.*

## Project Status

The forecasting system is fully implemented, tested, and publicly deployed. The application is accessible through a Streamlit web dashboard backed by a public Flask REST API.

When fresh automated pipeline data is not active in the cloud, the deployed system serves a committed canonical bootstrap observation timestamped `2026-08-31T07:00:00+00:00`. The application transparently marks telemetry as stale once the observation age exceeds the configured 3-hour threshold. Forecast horizons (h1 to h72) are anchored strictly to the physical observation timestamp rather than the request time.

## Architecture

The system topology separates frontend presentation from backend machine learning inference:

```text
GitHub Repository
       │
       ├── Streamlit Community Cloud
       │       └── Streamlit Dashboard (src/dashboard/app.py)
       │                │
       │                │ HTTPS REST
       │                ▼
       └── Render
               └── Flask REST API (src/api/app.py)
                    ├── RuntimeAssetResolver
                    ├── EXP-019 production model
                    ├── empirical prediction error intervals
                    ├── SHAP explainability engine
                    └── bootstrap feature vector
```

* **Frontend**: API-first Streamlit application. Hosted local inference fallback is disabled (`ENABLE_LOCAL_FALLBACK=false`), communicating with the backend over HTTPS via `FLASK_API_URL`.
* **Backend**: Flask WSGI application serving predictions, model provenance, and SHAP attributions from self-contained runtime assets. Heavy machine learning dependencies remain isolated on the backend.

The EXP-019 hybrid architecture operates across the 72-hour horizon as follows:
* **h=1..6**: LightGBM Direct Multi-Output (6 independent gradient-boosted regressors)
* **h=7..37**: Ridge Regression (alpha=1.0)
* **h=38..72**: Smoothly blended Ridge + Persistence (blend weight decays linearly from 1.0 at h38 to 0.7 at h72 over 35 horizons)

The model takes a 114 weather-enriched input feature vector scaled via StandardScaler (fitted on training data only).

## Deployment Configuration

### Backend (Render)
* **Platform**: Render Free Web Service (Python 3.10.14 via `.python-version`)
* **Server**: Gunicorn WSGI (`gunicorn src.api.app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`)
* **Environment Variables**: `RUNTIME_ASSET_MODE=runtime`, `CORS_ORIGINS=*`
* **Dependencies**: `requirements.txt` (production runtime only: scikit-learn, LightGBM, SHAP, Flask, Gunicorn, pandas, numpy, requests, python-dotenv)

*Concurrency Note: While `--threads 4` was used during deployment, memory profiling under continuous SHAP attribution indicates that a single worker (`--workers 1`) is the recommended configuration on 512 MB memory instances.*

### Frontend (Streamlit Community Cloud)
* **Platform**: Streamlit Community Cloud (Python 3.10)
* **Entrypoint**: `src/dashboard/app.py`
* **Dependencies**: `src/dashboard/requirements.txt` (lightweight frontend subset: streamlit, plotly, pandas, requests, python-dotenv)
* **Environment Configuration**: `FLASK_API_URL=https://aqi-forecasting-xyyb.onrender.com/api`, `ENABLE_LOCAL_FALLBACK=false`

## Runtime Assets

Clean cloud deployments rely on a versioned, self-contained runtime asset package in `data/runtime/`:

```text
data/runtime/
├── production/
│   ├── production_hybrid_model.joblib
│   ├── feature_scaler_v2_weather.joblib
│   ├── feature_schema_v2_weather.json
│   ├── empirical_error_intervals.json
│   └── manifest.json
├── explainability/
│   ├── shap_background.npy
│   ├── global_shap_importance.json
│   └── explainer_manifest.json
└── bootstrap/
    └── latest_feature_vector.json
```

These versioned assets enable deterministic clean-clone deployment without requiring historical training datasets (`data/raw/`, `data/processed/`) or local `.env` files. `RuntimeAssetResolver` enforces SHA256 integrity verification, schema validation (114 features), finite numeric checks, and timestamp validation on startup.

## Hopsworks Integration

The live Hopsworks Feature Store, Model Registry, and Scheduled Hourly Feature Pipeline have been implemented and verified:
* **Feature Group**: `aqi_weather_features_v2` (version 1, ID `52526`) on project `aqi_predictor_by_Waleed` (`https://eu-west.cloud.hopsworks.ai`).
* **Key Architecture**: `primary_key=["location_id"]`, `event_time="dt"`, `online_enabled=True`, `time_travel_format="HUDI"`.
* **Offline Storage**: Contains 48,718 historical hourly observations spanning November 28, 2020 through September 7, 2026 for Lahore across 115 columns (`location_id` + 114 canonical features). Idempotency is verified: re-upserting historical records maintains row count without inflation.
* **Online Storage & Unified Feature Resolver**:
  * Live online features served via `FeatureObservationResolver` (`src/inference/observation_resolver.py`), providing an authoritative single observation contract across `/api/current`, `/api/forecast`, and `/api/explain`.
  * *Hierarchy Modes*: Controlled via `FEATURE_SOURCE_MODE` (`auto` [default], `hopsworks` [strict fail-closed], `bootstrap` [committed vector fallback]).
  * *Targeted Entity Retrieval*: Queries `fg.select_all().filter(fg.location_id == "lahore").read(online=True)` with exact 114 canonical schema validation and non-finite numeric rejection.
  * *Dynamic Freshness & Ephemerality*: Freshness (`input_age_hours`, `is_stale`) and dominant pollutant are dynamically calculated per response rather than frozen in cache. Short-TTL (60s) thread-safe in-memory caching protects the online store against request floods while supporting `force_refresh=true`.
  * *Graceful Fallback*: In `auto` mode, missing cloud credentials or network disruptions seamlessly activate the committed canonical bootstrap vector without service degradation or secret leakage.
* **Hourly Live Feature Pipeline**: The feature pipeline is configured to run automatically every hour via GitHub Actions (`.github/workflows/feature_pipeline.yml`, cron schedule `17 * * * *`), and the exact live workflow path has been successfully verified through `workflow_dispatch` in production workflow run [34152458548](https://github.com/hwaleedkhalid/aqi-forecasting-system/actions/runs/34152458548), fetching live 72h OpenWeather pollution telemetry and Open-Meteo meteorology, verifying temporal grid continuity, computing canonical 114 features, and mutating Hopsworks Feature Store with zero latency regressions.
  * *Telemetry Providers*: OpenWeather Air Pollution History API (72h criteria pollutants) + Open-Meteo Weather API (`past_days=3, forecast_days=1` surface meteorology).
  * *Temporal Continuity*: Enforces strict hourly grid continuity over $[T-24\text{h}, T]$ with established production limit=3 linear/time interpolation for short sensor dropouts, failing closed on unbridgeable gaps $> 3$ hours.
  * *Production Timestamp*: Derived as $T = \min(T_{\text{AQ}}, T_{\text{weather}})$ floored to the hour boundary, strictly $\le \text{now\_utc}$ (future forecast hours discarded).
  * *Dual-Store Upsert*: Synchronously updates online RonDB via Kafka (`upsert_if_newer=True`) and triggers offline Hudi materialization asynchronously (`wait_for_job=False`), completing cloud runner execution in ~43 seconds without timeouts.
  * *CLI Commands*:
    * Dry run: `python -m src.feature_pipeline.run_hourly_ingestion --dry-run`
    * Live ingestion: `python -m src.feature_pipeline.run_hourly_ingestion --live`
  * *Isolated Dependencies*: Scheduled workflow uses `requirements-feature-pipeline.txt` (`hopsworks[python]==5.0.6`), supplying `confluent-kafka` and `pyarrow` for streaming feature group ingestion while isolating cloud dependencies from the deployed Render/Streamlit environments.
* **Daily Candidate Training & Evaluation Pipeline**: The daily candidate training pipeline is configured to run daily and its exact production path has been verified through `workflow_dispatch` via GitHub Actions (`.github/workflows/training_pipeline.yml`, cron schedule `45 2 * * *`).
  * *Authoritative Feature Store Retrieval*: Directly queries the offline Hopsworks Feature Store (`aqi_weather_features_v2` v1, `location_id="lahore"`, 48,718 rows retrieved) via `FeatureStoreTrainingLoader`. Auditable deduplication records any identical duplicates, and schema audit validates all 114 canonical columns, integral timestamps, and numeric finiteness.
  * *Protected Holdout Preservation*: Filters training candidates strictly to observations prior to `2025-06-07T00:00:00Z` (38,507 development samples), keeping the post-cutoff/quarantined region (10,211 rows, containing the formal 9,311-sample 2025-06-07 to 2026-08-28 final test set plus later accumulated observations) 100% untouched.
  * *Physical Timestamp Targets*: Enforces exact physical timestamp semantics ($T + h \times 3600$) for horizons $h=1 \dots 72$. Missing target timestamps cleanly drop the sample without synthetic interpolation (37,163 usable samples retained).
  * *Anti-Leakage Embargo Split*: Implements strict chronological 80/20 train/validation partitioning with a 73-hour embargo gap enforcing $\min(T_{\text{val\_input}}) > \max(T_{\text{train\_input}}) + 72\text{h}$ and asserting $\max(T_{\text{train\_target}}) < \min(T_{\text{val\_input}})$.
  * *Leakage-Safe Preprocessing*: `StandardScaler` is fitted strictly on the training partition; validation data is scaled using train-fitted parameters.
  * *Candidate Evaluation & Baselines*: Evaluates candidate model families (candidate `ridge` with `alpha=10.0`, `hybrid`, `lightgbm`, `random_forest`) across overall RMSE, MAE, R², milestone horizons (h1, h6, h24, h48, h72), and extreme AQI regimes (>200, >300), benchmarked against Naive Persistence. Recommendation gate strictly enforces same-protocol criteria (beats persistence overall and across h1/h24/h72, $\ge 20.0\%$ relative improvement, and zero material regressions on extreme subsets AQI >200/>300; EXP-019 75.91 holdout RMSE is excluded from gate logic).
  * *Candidate Isolation & Champion Protection*: Frozen EXP-019 champion remains the production model. All candidate models and evaluation reports are written to isolated directories (`data/models/candidates/<run_id>/`) with SHA256 manifests and uploaded as workflow artifacts. Under NO circumstances does the pipeline automatically promote, overwrite, or deploy candidate models.
* **Inference Contract & Parity**: Strict 114-column projection preserves canonical schema order with 0.00000000 maximum numerical difference between forecast points and SHAP attributions, with transparent staleness detection and exact `observation_dt` tracking.
* **Model Registry**: Frozen production champion registered under `pearls_aqi_production_champion` (version 1, ID `pearls_aqi_production_champion_1`). Packages the complete EXP-019 runtime bundle (hybrid model, scaler, canonical 114-feature schema, empirical residual intervals, and explainability assets) with SHA256 integrity manifest.
* **Clean-Download & Parity Verification**: Independent clean-download from Hopsworks into an isolated directory verified exact SHA256 matches across all files, standalone loadability, and 0.00000000 maximum absolute prediction error across all 72 horizons (including boundary horizons h1, h6, h7, h24, h37, h38, h39, h72) and explainability parity on h1, h24, h72.
* **Production Boundary**: The deployed Render API and Streamlit UI remain self-contained using the frozen Git deployment bundle for production stability; Model Registry registration fulfills the central artifact tracking requirement and is independently reproducible via `python -m src.inference.hopsworks_registry --all`.

## Key Results

Based on the final out-of-time test across 9,311 samples (2025-06-07 to 2026-08-28 UTC):
* **EXP-019 Performance**: Overall RMSE = 75.91, MAE = 53.55, R² = 0.4858
  * h+1 RMSE = 50.43
  * h+72 RMSE = 77.43
* **Benchmark Comparison**: Beats Naive Persistence (RMSE 85.35) by 11.06% relative RMSE reduction.
* **Walk-forward Validation**: 4/4 temporal folds won against Naive Persistence, with a mean relative gain of 24.46% (mean RMSE 83.44 vs 110.46).
* **Test Suite**: 458 automated tests passing across 35 test modules (74.00% code coverage).


## REST API Endpoints

The Flask backend is served under the `/api` route:
* `GET /api/health` — Service liveness and runtime artifact integrity check (HTTP 200/503)
* `GET /api/current` — Latest observed telemetry and metadata without model inference
* `GET /api/forecast` — 72-hour AQI trajectory with empirical prediction error intervals (supports `force_refresh=true`)
* `GET /api/model/info` — Production champion architecture provenance and benchmark metrics
* `GET /api/explain` — Multi-horizon SHAP feature attribution and persistence decomposition (`horizon=1..72`, `top_k=1..50`)

## Explainability and Uncertainty

* **Multi-Horizon SHAP**: TreeExplainer for h1–6, LinearExplainer for h7–37, and weighted Ridge attribution with explicitly separated persistence contribution for h38–72. SHAP reflects model feature attribution, not causal environmental mechanisms.
* **Empirical Prediction Error Intervals**: Residual quantiles (10th and 90th percentiles) derived from 8,636 out-of-fold walk-forward validation residuals. These represent empirical error ranges, not parametric confidence intervals.

## High-Severity & Hazardous AQI Alert System

The system implements a centralized, authoritative AQI alert classification and severity evaluation layer (`src/inference/alerting.py`):
* **Single Source of Truth**: Evaluates alerts strictly through the path `numeric AQI -> get_aqi_category() -> category -> classify_category_alert()`. This guarantees 100% boundary consistency between EPA categories and normalized alert levels:
  * Good (0–50): `none` / rank 0
  * Moderate (51–100): `none` / rank 1
  * Unhealthy for Sensitive Groups (101–150): `advisory` / rank 2
  * Unhealthy (151–200): `warning` / rank 3
  * Very Unhealthy (201–300): `severe` / rank 4
  * Hazardous (301–500+): `hazardous` / rank 5 (extreme values >500 preserved unclipped)
* **Authoritative Multi-Horizon Scanning**: Evaluates the full 72-hour forecast trajectory to detect:
  * Peak AQI, horizon, and peak EPA category
  * First threshold crossings: `first_advisory_horizon`, `first_unhealthy_horizon`, `first_very_unhealthy_horizon`, `first_hazardous_horizon` (with corresponding timestamps)
  * Category-specific counts: `advisory_horizon_count`, `unhealthy_horizon_count`, `very_unhealthy_horizon_count`, `hazardous_horizon_count`, and `severe_or_higher_horizon_count`
  * Tail risk auditing: `upper_interval_crosses_hazardous` triggers an uncertainty advisory if the 90th percentile empirical error bound reaches $\ge 301$ even when the expected forecast remains below the severe threshold.
* **Prioritized Dashboard Banners**: Structured top-level hierarchy prevents contradictory notices:
  1. Current Severe/Hazardous Observation Emergency Banner
  2. Forecast Severe/Hazardous 72h Trajectory Banner
  3. Stale Data Provenance Notice (rendered alongside severe alerts if data is historical)
  4. Unhealthy Warning / Advisory Banners (if no higher tier is active)
* **Domain Chart Boundaries**: Interactive 72-hour forecast visualization renders dashed category boundary lines at exact EPA levels: 151 (Unhealthy, red), 201 (Very Unhealthy, purple), and 301 (Hazardous, maroon).
* **Backward Compatibility**: Preserves all legacy fields (`high_severity`, `hazardous`, `has_high_severity`, `has_hazardous`, `highest_alert_level`) alongside the enriched `alert` and `forecast_alert` contracts.

## Project Structure

```text
Pearls-AQI-Predictor/
├── data/
│   ├── runtime/              # Whitelisted versioned production deployment package
│   │   ├── production/       # Model, scaler, schema, intervals, manifest
│   │   ├── explainability/   # SHAP background, global importance, manifest
│   │   └── bootstrap/        # Canonical 114-feature bootstrap vector
│   ├── raw/                  # Excluded: raw API ingestion cache
│   ├── processed/            # Excluded: local training datasets
│   └── models/               # Excluded: local training checkpoints & registry
├── src/
│   ├── config.py             # Central configuration
│   ├── exceptions.py         # Domain exceptions
│   ├── logger.py             # Structured logging
│   ├── data_ingestion/       # OpenWeather & Open-Meteo API providers
│   ├── feature_pipeline/     # Feature engineering & Hopsworks client
│   ├── training_pipeline/    # Dataset builder, model trainer, walk-forward validator
│   ├── models/               # Model implementations (Naive, Ridge, RF, TF, Hybrid)
│   ├── inference/            # Predictor, explainer, runtime resolver, post-processor
│   ├── api/                  # Flask REST API blueprint & app factory
│   └── dashboard/            # Streamlit frontend application & components
├── tests/                    # 28 test modules + conftest (355 tests)
├── .github/workflows/        # CI, hourly ingestion, weekly evaluation
├── requirements.txt          # Production runtime dependencies
├── requirements-dev.txt      # Offline training, research & testing dependencies
├── .python-version           # Pinned Python 3.10.14
├── .gitattributes            # Line ending normalization
├── .env.example
└── .gitignore
```

## Quick Start (Local Development)

1. Clone the repository and create a virtual environment (Python 3.10):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Configure environment variables:
   Copy `.env.example` to `.env` and configure keys as needed.
4. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
5. Start the REST API:
   ```bash
   python -m flask --app src.api.app run
   ```
6. Start the dashboard (in a separate terminal):
   ```bash
   streamlit run src/dashboard/app.py
   ```

## Additional Documentation

Please refer to `report.md` for the complete project report covering the development journey, experimental evaluations, MLOps design decisions, and production deployment hardening.

**License**: Educational and research purposes.

