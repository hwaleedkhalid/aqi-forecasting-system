# Pearls AQI Predictor - Project Memory & State

## 1. Project Overview
- **Project Name**: Pearls AQI Predictor
- **Objective**: Serverless 3-day (72-hour) Air Quality Index forecasting system.
- **Repository Location**: `d:\10Perls\Pearls-AQI-Predictor`
- **Design Standard**: US EPA AQI (0-500 scale) calculated via piecewise linear interpolation from raw pollutant concentrations ($PM_{2.5}, PM_{10}, O_3, NO_2, SO_2, CO, NH_3$).

---

## 2. Key Architectural Decisions (Locked)
1. **Target**: 72 continuous hourly EPA AQI predictions (`[t+1, ..., t+72]`).
2. **Features**: 114 weather-enriched features (chemical atmospheric pollutants + meteorology from Open-Meteo + derived temporal/lag/rolling features).
3. **Data Sources**:
   - OpenWeather Air Pollution History API (Nov 27, 2020 to present).
   - Open-Meteo Historical Weather Archive (Nov 27, 2020 to present).
4. **Storage Progression**: Local CSV & NPY (`data/raw/` and `data/processed/`) for Phases 1-15 -> Hopsworks Feature Store & Model Registry in Phase 16.
5. **Model Strategy**: Naive Persistence Baseline -> Ridge Regression -> Random Forest -> TensorFlow Feed-Forward Network (`Dense(72)`) -> Hybrid Specialist Ensemble (LightGBM h1-6 + Ridge h7-72).
6. **Evaluation Metric Hierarchy**:
   - **Primary Decision Metric**: **Overall RMSE** (Root Mean Squared Error) across all 72 prediction horizons.
   - **Secondary Diagnostic Metrics**: **Overall MAE**, **Overall $R^2$**, and **Per-Horizon RMSE/MAE**.
7. **Validation Protocol**:
   - Strict test set freeze: `X_test_v2` / `y_test_v2` held out completely untouched until final benchmark.
   - Model selection evaluated exclusively via **3-Fold Expanding Chronological Cross-Validation** on training set.
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
| **Phase 10**| Train TensorFlow Neural Network | **Completed** ✅ | 2026-09-02 |
| **Phase 10.5A**| Diagnostics & Distribution Shift | **Completed** ✅ | 2026-09-02 |
| **Phase 10.5B**| Weather Ingestion & Enrichment | **Completed** ✅ | 2026-09-02 |
| **Phase 10.5C**| Systematic Tuning & Experiment Registry | **Completed** ✅ | 2026-09-02 |
| **Phase 10.5D**| Forecasting Architecture Experiments | **Completed** ✅ | 2026-09-02 |
| **Phase 10.5E**| Final Untouched Test Benchmark | *Ready to Start* ⏳ | - |
| **Phase 11**| Multi-Year Walk-Forward Cross-Validation | Pending | - |
| **Phase 12**| Build Multi-Horizon Inference Pipeline | Pending | - |
| **Phase 13**| Build Flask REST API | Pending | - |
| **Phase 14**| Build Streamlit UI Dashboard | Pending | - |
| **Phase 15**| Automate with GitHub Actions CI/CD | Pending | - |
| **Phase 16**| Hopsworks Cloud Feature Store Integration | Pending | - |
| **Phase 17**| SHAP Model Explainability | Pending | - |
| **Phase 18**| AQICN Multi-source Data Integration | Future | - |

---

## 4. Current State & Deliverables

### Phase 10.5D: Forecasting Architecture Experiments
- Built [`src/models/pollutant_to_aqi_model.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/models/pollutant_to_aqi_model.py) (79% coverage) with vectorized US EPA piecewise interpolation for continuous pollutant arrays.
- Built [`src/models/grouped_horizon_model.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/models/grouped_horizon_model.py) (67% coverage) with 3-tier partitioned horizon regressors ($h1-6$, $h7-24$, $h25-72$).
- Built [`src/models/hybrid_specialist_model.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/models/hybrid_specialist_model.py) (68% coverage) implementing:
  - `HybridAQISpecialistModel` (LightGBM $h1-6$ + Ridge $h7-72$).
  - `PersistenceAwareHybridModel` (LightGBM $h1-6$ + Ridge $h7-37$ + Blended Ridge/Persistence $h38-72$).
- Built [`src/training_pipeline/run_architecture_experiments.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/run_architecture_experiments.py) executing 5 architectural experiments across 3 expanding chronological folds.
- **Key Validation Findings**:
  - **EXP-017 (Hybrid AQI Specialist)** achieved the **#1 overall performance across all 19 project experiments**:
    - **Mean Val RMSE = 85.46** ($\pm 4.86$), **Mean Val MAE = 63.16**, **Mean Val $R^2 = 0.5154$**.
    - **$h+1$ RMSE = 46.18** (matches LightGBM's state-of-the-art short-horizon precision, beating standalone Ridge's $52.66$ by **12.3%**).
    - **$h+24$ RMSE = 81.99** & **$h+72$ RMSE = 92.39** (maintains Ridge's strong shrinkage stability across multi-day horizons).
    - Training time: **11.41 seconds**.
  - **EXP-019 (Persistence-Aware Hybrid)** delivered the lowest mean MAE (**62.95**).
  - **EXP-015 / EXP-018 (Multi-Pollutant $\to$ EPA AQI)** revealed that taking the $\max(\cdot)$ operator across 6 predicted pollutant sub-indices compounds individual over-prediction estimation noise (RMSE = **92.19**).
- Created analysis notebook [`notebooks/12_forecasting_architecture_experiments.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/12_forecasting_architecture_experiments.ipynb).
- Built test suite [`tests/test_forecasting_architectures.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_forecasting_architectures.py).
- **Total test suite**: **146/146 tests passing (80% total codebase coverage)**.
