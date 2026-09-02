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
5. **Production Model Champion**: **Persistence-Aware Hybrid Model (EXP-019)** (`data/models/production_hybrid_model.joblib`):
   - Short horizons ($h=1\dots6$): LightGBM Direct Multi-Output.
   - Intermediate horizons ($h=7\dots37$): Ridge Regression ($\alpha=1.0$).
   - Multi-day horizons ($h=38\dots72$): Smoothly blended Ridge + Persistence ($w_h \hat{y}_{\text{Ridge}} + (1-w_h) y_t$).
6. **Evaluation Metric Hierarchy**:
   - **Primary Decision Metric**: **Overall RMSE** across all 72 prediction horizons.
   - **Secondary Diagnostic Metrics**: **Overall MAE**, **Overall $R^2$**, and **Per-Horizon RMSE/MAE**.
7. **Validation & Benchmark Outcomes**:
   - **Final Out-of-Time Test Partition**: 9,311 samples (`2025-06-07 to 2026-08-28 UTC`).
   - **Production Champion Performance**: **Overall RMSE = 75.91** (beating Naive by 11.06% and Ridge v1 by 8.51%), **$R^2 = 0.4858$**, **$h+1$ RMSE = 50.43**, **$h+72$ RMSE = 77.43**, **Hazardous (>300) RMSE = 142.65**.
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
| **Phase 10.5E**| Final Untouched Test Benchmark | **Completed** ✅ | 2026-09-02 |
| **Phase 11**| Multi-Year Walk-Forward Cross-Validation | *Ready to Start* ⏳ | - |
| **Phase 12**| Build Multi-Horizon Inference Pipeline | Pending | - |
| **Phase 13**| Build Flask REST API | Pending | - |
| **Phase 14**| Build Streamlit UI Dashboard | Pending | - |
| **Phase 15**| Automate with GitHub Actions CI/CD | Pending | - |
| **Phase 16**| Hopsworks Cloud Feature Store Integration | Pending | - |
| **Phase 17**| SHAP Model Explainability | Pending | - |
| **Phase 18**| AQICN Multi-source Data Integration | Future | - |

---

## 4. Current State & Deliverables

### Phase 10.5E: Final Out-of-Time Test Benchmark
- Built [`src/training_pipeline/benchmark_final_test.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/benchmark_final_test.py) and executed the single, frozen benchmark across 7 models on the 9,311-sample held-out test partition.
- **Definitive Test Leaderboard**:
  1. **Persistence-Aware Hybrid (EXP-019)** 🏆: Overall RMSE = **75.91**, MAE = **53.55**, $R^2 = \mathbf{0.4858}$, $h+1$ RMSE = **50.43**, $h+72$ RMSE = **77.43**, Severe (>200) RMSE = **120.55**.
  2. **Hybrid Specialist (EXP-017)**: Overall RMSE = **78.20**, MAE = **55.98**, $R^2 = 0.4543$, $h+1$ RMSE = **50.43**, $h+72$ RMSE = **84.53**.
  3. **Ridge v2 (EXP-005 Weather-Enriched)**: Overall RMSE = **78.38**, MAE = **56.17**, $R^2 = 0.4518$, $h+1$ RMSE = **54.41**, $h+72$ RMSE = **84.53**.
  4. **Ridge v1 (Pollutants-only)**: Overall RMSE = **82.97**, MAE = **63.14**, $R^2 = 0.3741$, $h+1$ RMSE = **53.18**, $h+72$ RMSE = **96.57**.
  5. **TensorFlow DNN v1**: Overall RMSE = **85.01**, MAE = **65.94**, $R^2 = 0.3429$, $h+1$ RMSE = **55.59**, $h+72$ RMSE = **102.26**.
  6. **Naive Persistence Baseline**: Overall RMSE = **85.35**, MAE = **46.64**, $R^2 = 0.3499$, $h+1$ RMSE = **68.95**, $h+72$ RMSE = **89.68**.
  7. **Random Forest v1**: Overall RMSE = **89.18**, MAE = **65.06**, $R^2 = 0.2768$, $h+1$ RMSE = **54.28**, $h+72$ RMSE = **99.64**.
- Saved production model artifact to [`data/models/production_hybrid_model.joblib`](file:///d:/10Perls/Pearls-AQI-Predictor/data/models/production_hybrid_model.joblib).
- Generated analysis notebook [`notebooks/13_final_test_benchmark.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/13_final_test_benchmark.ipynb).
- Generated benchmark JSON & CSV reports in `data/models/`.
- **Total test suite**: **147/147 tests passing (77% total codebase coverage)**.
