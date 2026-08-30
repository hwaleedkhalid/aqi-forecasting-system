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
| **Phase 2** | Investigate Historical OpenWeather Data | *Ready to Start* ⏳ | - |
| **Phase 3** | Build OpenWeather Client | Pending | - |
| **Phase 4** | Backfill Historical Data (~50k records) | Pending | - |
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

## 4. Current State & Deliverables (Phase 1)
- **Directory Structure**: Initialized with all `data/`, `notebooks/`, `src/`, `tests/`, and `.github/` directories.
- **Environment**: Virtual environment created (`.venv`), dependencies installed.
- **Config & Core Utilities**:
  - `src/config.py`: Centralized configuration constants and hyperparameters.
  - `src/logger.py`: Standard logging to console and `data/logs/pearls_aqi.log`.
  - `src/exceptions.py`: Custom domain exceptions (`DataIngestionError`, `FeatureStoreError`, `ModelTrainingError`, `PredictionError`, `ValidationError`).
  - `.env.example` & `.env`: Environment variable management.
  - `.gitignore`: Configured to exclude runtime data, secrets, and cache files.
- **Tests**: `tests/test_phase1_setup.py` (34 test cases passing, 100% test coverage on `src`).
