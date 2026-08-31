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
6. **Evaluation Metric Hierarchy**:
   - **Primary Decision Metric**: **Overall RMSE** (Root Mean Squared Error) across all 72 prediction horizons.
   - **Secondary Diagnostic Metrics**: **Overall MAE**, **Overall $R^2$**, and **Per-Horizon RMSE/MAE**.
7. **Validation**: Time-based 80/20 train/test split with 72h anti-leakage embargo gap + Walk-Forward Cross-Validation across seasons.
8. **Explainability**: SHAP deferred to Phase 17 (after complete pipeline works end-to-end).
9. **Orchestration**: GitHub Actions (hourly feature extraction, daily retraining).

---

## 3. Phase Progress Tracker

| Phase | Description | Status | Completion Date |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Project Setup & Configuration | **Completed** ✅ | 2026-08-30 |
| **Phase 2** | Investigate Historical OpenWeather Data | **Completed** ✅ | 2026-08-30 |
| **Phase 3** | Build OpenWeather Client | **Completed** ✅ | 2026-08-30 |
| **Phase 4** | Backfill Historical Data (~50k records) | **Completed** ✅ | 2026-08-31 |
| **Phase 5** | Explore Dataset & AQI Conversion | **Completed** ✅ | 2026-08-31 |
| **Phase 6** | Feature Engineering (Pollutants only) | **Completed** ✅ | 2026-08-31 |
| **Phase 7** | Multi-Output Training Dataset Prep | **Completed** ✅ | 2026-08-31 |
| **Phase 8** | Train Ridge Regression & Naive Baseline | **Completed** ✅ | 2026-08-31 |
| **Phase 9** | Train Random Forest Regressor | **Completed** ✅ | 2026-08-31 |
| **Phase 10**| Train TensorFlow Neural Network | *Ready to Start* ⏳ | - |
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
- Verified live connectivity to OpenWeather Air Pollution endpoints.
- Confirmed global historical availability from Nov 27, 2020 (`1606482000` UTC) with hourly frequency.
- Created executable notebook [`notebooks/01_api_investigation.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/01_api_investigation.ipynb).

### Phase 3: OpenWeather Ingestion Client
- Built abstract interface [`src/data_ingestion/base_provider.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/data_ingestion/base_provider.py).
- Built concrete client [`src/data_ingestion/openweather_provider.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/data_ingestion/openweather_provider.py) (100% test coverage).
- Built test suite [`tests/test_data_ingestion.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_data_ingestion.py).

### Phase 4: Historical Data Backfill
- Built backfill pipeline [`src/feature_pipeline/backfill.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/feature_pipeline/backfill.py).
- Ingested **70 raw monthly JSON partitions** (49,483 hourly observations, 98.05% completeness).
- Built test suite [`tests/test_backfill.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_backfill.py).
- Created exploration notebook [`notebooks/02_raw_data_exploration.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/02_raw_data_exploration.ipynb).

### Phase 5: EDA & AQI Conversion
- Built [`src/feature_pipeline/aqi_calculator.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/feature_pipeline/aqi_calculator.py) (100% coverage).
- Built unit test suite [`tests/test_aqi_calculator.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_aqi_calculator.py).
- Generated clean labeled historical dataset [`data/processed/historical_aqi_clean.csv`](file:///d:/10Perls/Pearls-AQI-Predictor/data/processed/historical_aqi_clean.csv) (49,483 rows).
- Created analysis notebook [`notebooks/03_eda_and_aqi_conversion.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/03_eda_and_aqi_conversion.ipynb).

### Phase 6: Feature Engineering
- Built [`src/feature_pipeline/feature_engineering.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/feature_pipeline/feature_engineering.py) (100% test coverage): 64 engineered backward-looking features.
- Built unit test suite [`tests/test_feature_pipeline.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_feature_pipeline.py).
- Persisted feature dataset [`data/processed/features.csv`](file:///d:/10Perls/Pearls-AQI-Predictor/data/processed/features.csv) and schema [`data/processed/feature_schema.json`](file:///d:/10Perls/Pearls-AQI-Predictor/data/processed/feature_schema.json).
- Created analysis notebook [`notebooks/04_feature_engineering.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/04_feature_engineering.ipynb).

### Phase 7: Multi-Output Training Dataset Prep
- Built [`src/training_pipeline/dataset_builder.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/dataset_builder.py) (100% test coverage).
- Enforced chronological 80/20 train/test split with 72h anti-leakage embargo gap.
- Serialized dataset arrays: `X_train.npy` (38,917 $\times$ 64), `y_train.npy` (38,917 $\times$ 72), `X_test.npy` (9,748 $\times$ 64), `y_test.npy` (9,748 $\times$ 72), `current_aqi_train.npy`, `current_aqi_test.npy`.
- Built unit test suite [`tests/test_dataset_builder.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_dataset_builder.py).
- Created analysis notebook [`notebooks/05_dataset_preparation.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/05_dataset_preparation.ipynb).

### Phase 8: Ridge Regression & Naive Baseline
- Built base model and evaluator infrastructure.
- Ridge Regression beats Naive Persistence on Overall RMSE (82.97 vs 84.05) and for all horizons $h=1 \dots 37$.
- Saved model artifact `data/models/ridge_model.joblib`.
- Created analysis notebook [`notebooks/06_model_training_ridge.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/06_model_training_ridge.ipynb).

### Phase 9: Random Forest Regressor
- Built [`src/models/random_forest_model.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/models/random_forest_model.py) (100 trees, multi-output).
- Evaluated on test set: Overall RMSE = **89.18**, MAE = **65.06**, $R^2$ = **0.2767** (h+1 RMSE = **54.28**, h+24 RMSE = **79.13**).
- Extracted and saved feature importances [`data/models/rf_feature_importances.csv`](file:///d:/10Perls/Pearls-AQI-Predictor/data/models/rf_feature_importances.csv) (top features: `pm2_5_rolling_mean_12h` at 33.65%, `month_cos` + `month_sin` at 13.23%).
- Saved artifact `data/models/random_forest_model.joblib`.
- Created analysis notebook [`notebooks/07_model_training_rf.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/07_model_training_rf.ipynb).
