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
5. **Model Strategy**: Naive Persistence Baseline -> Ridge Regression -> Random Forest -> TensorFlow Feed-Forward Network (`Dense(72)`) -> LightGBM Direct/Horizon strategies.
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
| **Phase 10.5D**| Forecasting Architecture Experiments | *Ready to Start* ⏳ | - |
| **Phase 10.5E**| Final Untouched Test Benchmark | Pending | - |
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

### Phase 10.5C: Systematic Model Tuning & Experiment Registry
- Built [`src/training_pipeline/experiment_registry.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/experiment_registry.py) (85% coverage) to persist structured run artifacts, hyperparameters, metrics, and JSON/CSV leaderboards in `data/models/experiments/`.
- Built [`src/training_pipeline/cv_evaluator.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/cv_evaluator.py) (98% coverage) executing 3-fold expanding chronological training-only cross-validation.
- Built [`src/models/lightgbm_models.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/models/lightgbm_models.py) (80% coverage) with `LightGBMDirectMultiOutput` and `LightGBMHorizonAsFeature`.
- Built [`src/training_pipeline/run_systematic_experiments.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/run_systematic_experiments.py) executing 14 systematic training-only experiments.
- **Key Validation Findings**:
  - **Ridge Regression ($\alpha=1.0$)**: Optimal multi-horizon balance with Val RMSE = **85.74** ($\pm 4.86$), Val MAE = **63.59**, Val $R^2 = 0.5123$, $h+1$ RMSE = **52.66**, $h+24$ RMSE = **81.99**, $h+72$ RMSE = **92.39**, Training time = **0.19s**.
  - **ElasticNet ($l_1=0.9$)**: Val RMSE = **85.76**, Val MAE = **63.55**, Val $R^2 = 0.5123$ (matches Ridge but with heavier coordinate descent overhead).
  - **LightGBM Direct (72 estimators)**: Val RMSE = **90.86**, Val MAE = **65.89**, Val $R^2 = 0.4568$. Achieves best-in-class short-horizon accuracy at $h+1$ (**46.18 RMSE**, beating Ridge by **12.3%**), but higher long-horizon variance ($h+72$ RMSE = **104.60**).
  - **LightGBM Horizon-as-Feature**: Val RMSE = **97.53**, Val MAE = **73.29**, Val $R^2 = 0.3762$.
- Created analysis notebook [`notebooks/11_systematic_model_tuning.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/11_systematic_model_tuning.ipynb).
- Built test suite [`tests/test_experiment_registry.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_experiment_registry.py).
- **Total test suite**: **140/140 tests passing (88% coverage)**.
