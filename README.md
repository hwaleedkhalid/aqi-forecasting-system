# 🌬️ Pearls AQI Predictor

**3-Day Air Quality Index Forecasting System**

An end-to-end serverless AQI forecasting system that predicts the Air Quality Index for the next 72 hours using machine learning.

## 🏗️ Architecture

```
OpenWeather API → Feature Pipeline → Model Training → Inference → Flask API → Streamlit Dashboard
```

**Key decisions:**
- **Prediction target**: EPA AQI (0-500 scale), 72 hourly values per forecast
- **Models**: Ridge Regression → Random Forest → TensorFlow (progressive complexity)
- **Storage**: Local CSV (Phases 1-15) → Hopsworks Feature Store (Phase 16+)
- **Automation**: GitHub Actions (hourly feature pipeline, daily retraining)

## 📂 Project Structure

```
Pearls-AQI-Predictor/
├── data/
│   ├── raw/                  # Raw API responses (JSON)
│   │   ├── air_quality/      # Historical air pollution data
│   │   └── weather/          # Weather data
│   ├── processed/            # Engineered features (CSV)
│   ├── models/               # Trained model artifacts
│   ├── predictions/          # Cached predictions
│   └── logs/                 # Application logs
├── notebooks/                # Jupyter notebooks (EDA, training, evaluation)
├── src/
│   ├── config.py             # Centralized configuration
│   ├── logger.py             # Logging setup
│   ├── exceptions.py         # Custom exception classes
│   ├── data_ingestion/       # API clients (OpenWeather, AQICN)
│   ├── feature_pipeline/     # Feature engineering & storage
│   ├── training_pipeline/    # Dataset creation & model training
│   ├── inference/            # Prediction & post-processing
│   ├── models/               # ML model wrappers
│   ├── api/                  # Flask REST API
│   └── dashboard/            # Streamlit frontend
├── tests/                    # pytest test suite
├── .github/workflows/        # GitHub Actions CI/CD
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── .gitignore
```

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd Pearls-AQI-Predictor
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run tests

```bash
pytest tests/ -v --cov=src
```

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENWEATHER_API_KEY` | Yes | OpenWeather API key ([get one free](https://openweathermap.org/api)) |
| `HOPSWORKS_API_KEY` | Phase 16+ | Hopsworks API key |
| `TARGET_CITY_NAME` | No | City name (default: Lahore) |
| `TARGET_LAT` | No | Latitude (default: 31.5497) |
| `TARGET_LON` | No | Longitude (default: 74.3436) |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

## 📊 Technology Stack

| Component | Technology |
|-----------|-----------|
| Data Source | OpenWeather API (Phase 1), AQICN (Phase 18) |
| Feature Store | Local CSV → Hopsworks |
| ML Models | Ridge, Random Forest, TensorFlow |
| Backend API | Flask |
| Dashboard | Streamlit + Plotly |
| Orchestration | GitHub Actions |
| Explainability | SHAP (Phase 17) |

## 📋 Implementation Phases

See [Phases.md](../Phases.md) for the full 18-phase implementation plan.

Current status: **Phase 1 — Project Setup & Configuration** ✅

## 📝 License

This project is for educational and research purposes.
