# Pearls AQI Predictor

Serverless 72-hour Air Quality Index forecasting system for Lahore, Pakistan (31.55°N, 74.34°E). The project predicts US EPA AQI (0-500 scale) for 72 continuous hourly horizons using data sourced from OpenWeather Air Pollution History API (Nov 2020 to present) and the Open-Meteo Historical Weather Archive. 

The production champion model is EXP-019 (Persistence-Aware Hybrid Model).

## Architecture

The system utilizes a 114 weather-enriched input feature set (pollutant concentrations, meteorology, temporal/lag/rolling aggregates) scaled via StandardScaler (fitted on training data only).

The EXP-019 architecture operates across the 72-hour horizon as follows:
* **h=1..6**: LightGBM Direct Multi-Output (6 independent gradient-boosted regressors)
* **h=7..37**: Ridge Regression (alpha=1.0)
* **h=38..72**: Smoothly blended Ridge + Persistence (blend weight decays from 1.0 to min_blend_weight over 35 horizons)

## Key Results

Based on the final out-of-time test across 9,311 samples:
* **EXP-019 Performance**: Overall RMSE = 75.91, MAE = 53.55, R² = 0.4858
  * h+1 RMSE = 50.43
  * h+72 RMSE = 77.43
* **Benchmark**: Beats Naive Persistence (RMSE 85.35) by 11.06%.
* **Walk-forward Validation**: 4/4 folds won vs Naive, with a mean relative gain of 24.5%.
* **Test Coverage**: 346 tests passing, 75% code coverage.

## Tech Stack

* **Machine Learning**: scikit-learn (Ridge), LightGBM, TensorFlow (explored), SHAP 0.49.1
* **Backend**: Flask REST API (5 endpoints)
* **Frontend**: Streamlit + Plotly (interactive 72h forecast chart with empirical error bands)
* **Orchestration**: GitHub Actions (CI, hourly feature ingestion, weekly candidate evaluation)
* **Feature Store**: Hopsworks (Phase 16 integration)
* **Language**: Python 3.10

## REST API Endpoints

The backend is built with Flask, served under the `/api` route:
* `GET /api/health` — Service liveness and model readiness
* `GET /api/current` — Latest observed telemetry without inference
* `GET /api/forecast` — 72-hour AQI forecast with empirical error intervals (supports `force_refresh` parameter)
* `GET /api/model/info` — Champion model specification and benchmark metrics
* `GET /api/explain` — SHAP feature attributions and persistence decomposition (parameters: `horizon=1..72`, `top_k=1..50`)

## Dashboard Features

The Streamlit interactive dashboard includes:
* Live/stale telemetry banner
* 72-hour interactive forecast curve with shaded empirical error bands (10th-90th percentile walk-forward residuals)
* EPA category color coding and severity threshold reference lines
* Milestone cards (+1h, +12h, +24h, +48h, +72h)
* Interactive SHAP attribution explorer (slider for horizon 1-72)
* Sidebar with model specs and test benchmarks
* Automatic API-to-local-inference fallback

## Project Structure

```text
Pearls-AQI-Predictor/
├── data/
│   ├── raw/                  # Raw API responses
│   ├── processed/            # Engineered features & schema
│   ├── models/               # Trained model artifacts & experiment registry
│   │   ├── experiments/      # EXP-001 to EXP-019 records
│   │   ├── walk_forward/     # Walk-forward reports & empirical error intervals
│   │   │   └── ablation/     # Winter/smog ablation results
│   │   └── explainability/   # SHAP background & global importance
│   ├── predictions/          # Cached predictions
│   └── logs/                 # Application logs
├── notebooks/                # 15 Jupyter notebooks (EDA through ablation)
├── src/
│   ├── config.py             # Centralized configuration
│   ├── exceptions.py         # Domain exception hierarchy
│   ├── logger.py             # Structured logging
│   ├── data_ingestion/       # OpenWeather & Open-Meteo API clients
│   ├── feature_pipeline/     # Feature engineering, AQI calculation, Hopsworks
│   ├── training_pipeline/    # Dataset builder, evaluator, walk-forward validator
│   ├── models/               # 11 model classes (Naive through Hybrid)
│   ├── inference/            # Predictor, explainer, post-processing, caching
│   ├── api/                  # Flask REST API
│   └── dashboard/            # Streamlit frontend
├── tests/                    # 27 test modules + conftest (346 tests)
├── .github/workflows/        # CI, hourly ingestion, weekly evaluation
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick Start

1. Clone the repository and create a virtual environment (Python 3.10):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables:
   Copy `.env.example` to `.env` and add your `OPENWEATHER_API_KEY`.
4. Run the test suite:
   ```bash
   pytest tests/ -v --cov=src
   ```
5. Start the REST API:
   ```bash
   python -m flask --app src.api.app run
   ```
6. Start the dashboard (in a new terminal):
   ```bash
   streamlit run src/dashboard/app.py
   ```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENWEATHER_API_KEY` | Yes | OpenWeather API key |
| `HOPSWORKS_API_KEY` | For cloud integration | Hopsworks Feature Store API key |
| `HOPSWORKS_PROJECT_NAME` | For cloud integration | Hopsworks project name |
| `CORS_ORIGINS` | No | Comma-separated origins for Flask CORS (default: localhost:8501) |
| `TARGET_CITY_NAME` | No | City name (default: Lahore) |
| `TARGET_LAT` | No | Latitude (default: 31.5497) |
| `TARGET_LON` | No | Longitude (default: 74.3436) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

## GitHub Actions Workflows

* `ci.yml`: Runs full test suite on push/PR to main (Python 3.10, ubuntu-latest, pytest with 70% coverage gate).
* `feature_pipeline.yml`: Runs hourly at `:17` — executes data ingestion and uploads telemetry snapshot artifact (7-day retention).
* `training_pipeline.yml`: Runs weekly on Sundays at 02:23 UTC — evaluates candidate models and enforces production model immutability via SHA256 hash comparison.

## Additional Documentation

Please refer to `report.md` for a detailed academic report that covers the development journey, extensive experiment results, and technical design decisions.

**License**: This project is for educational and research purposes.
