# Pearls AQI Predictor - Project Memory & State

## 1. Project Overview
- **Project Name**: Pearls AQI Predictor
- **Objective**: Serverless 3-day (72-hour) Air Quality Index forecasting system.
- **Repository Location**: `d:\10Perls\Pearls-AQI-Predictor`
- **Design Standard**: US EPA AQI (0-500 scale) calculated via piecewise linear interpolation from raw pollutant concentrations ($PM_{2.5}, PM_{10}, O_3, NO_2, SO_2, CO, NH_3$).

---

## 2. Key Architectural Decisions (Locked)
1. **Target**: 72 continuous hourly EPA AQI predictions (`[t+1, ..., t+72]`).
2. **Features (Phases 1-15)**: Chemical atmospheric pollutants + derived temporal/lag features only. Historical weather excluded due to API paywalls.
3. **Data Source**: OpenWeather Air Pollution History API (Free tier: Nov 27, 2020 to present, ~50,000 hourly observations globally).
4. **Storage Progression**: Local CSV (`data/raw/` and `data/processed/`) for Phases 1-15 -> Hopsworks Feature Store & Model Registry in Phase 16.
5. **Model Strategy**: Naive Persistence Baseline -> Ridge Regression -> Random Forest -> TensorFlow Feed-Forward Network (`Dense(72)`).
6. **Validation**: Time-based 80/20 train/test split + Walk-Forward Cross-Validation across seasons.
7. **Explainability**: SHAP deferred to Phase 17 (after complete pipeline works end-to-end).
8. **Orchestration**: GitHub Actions (hourly feature extraction, daily retraining).

---

## 3. Phase Progress Tracker

| Phase | Description | Status | Completion Date |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Project Setup & Configuration | **Completed** ✅ | 2026-08-30 |
| **Phase 2** | Investigate Historical OpenWeather Data | **Completed** ✅ | 2026-08-30 |
| **Phase 3** | Build OpenWeather Client | **Completed** ✅ | 2026-08-30 |
| **Phase 4** | Backfill Historical Data (~50k records) | *Ready to Start* ⏳ | - |
| **Phase 5** | Explore Dataset & AQI Conversion | Pending | - |
| **Phase 6** | Feature Engineering (Pollutants only) | Pending | - |
| **Phase 7** | Multi-Output Training Dataset Prep | Pending | - |
| **Phase 8** | Train Ridge Regression & Naive Baseline | Pending | - |
| **Phase 9** | Train Random Forest Regressor | Pending | - |
| **Phase 10**| Train TensorFlow Neural Network | Pending | - |
| **Phase 11**| Model Evaluation & Walk-Forward Validation | Pending | - |
| **Phase 12**| Build Multi-Horizon Inference Pipeline | Pending | - |
| **Phase 13**| Build Flask REST API | Pending | - |
| **Phase 14**| Build Streamlit UI Dashboard | Pending | - |
| **Phase 15**| Automate with GitHub Actions CI/CD | Pending | - |
| **Phase 16**| Hopsworks Cloud Feature Store Integration | Pending | - |
| **Phase 17**| SHAP Model Explainability | Pending | - |
| **Phase 18**| AQICN Multi-source Data Integration | Future | - |

---

## 4. Current State & Deliverables
### Phase 1: Setup & Configuration
- Directory structure, `.env.example`, `.env`, `requirements.txt`, `.gitignore`, `README.md`.
- `src/config.py`, `src/logger.py`, `src/exceptions.py`.
- `tests/test_phase1_setup.py` (34 passing tests, 100% coverage).

### Phase 2: OpenWeather API Investigation
- Verified live connectivity to OpenWeather Air Pollution endpoints (Current, History, Forecast).
- Confirmed global historical availability from Nov 27, 2020 (`1606482000` UTC) with hourly frequency.
- Implemented and demonstrated piecewise linear interpolation for EPA PM2.5 AQI calculation vs OpenWeather's 1-5 CAQI index.
- Created executable notebook [`notebooks/01_api_investigation.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/01_api_investigation.ipynb).

### Phase 3: OpenWeather Ingestion Client
- Built abstract interface [`src/data_ingestion/base_provider.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/data_ingestion/base_provider.py).
- Built concrete client [`src/data_ingestion/openweather_provider.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/data_ingestion/openweather_provider.py) supporting:
  - Current, forecast, and historical air pollution endpoints.
  - Exponential backoff retry logic (on 5xx, 429, timeouts).
  - Strict schema, coordinate, pollutant concentrations, and timestamp validation.
  - Safe credential logging redaction.
  - Chronological timestamp sorting and boundary deduplication.
- Built test suite [`tests/test_data_ingestion.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_data_ingestion.py) (62 total project tests passing, 98% overall coverage).
