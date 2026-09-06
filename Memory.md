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
| **Phase 11**| Multi-Year Walk-Forward Cross-Validation | **Completed** ✅ | 2026-09-06 |
| **Phase 11.5**| Targeted Winter/Smog Feature Investigation | **Completed** ✅ | 2026-09-06 |
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

### Phase 11: Walk-Forward Stability & Diagnostic Validation
- Built [`src/training_pipeline/season_classifier.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/season_classifier.py): `LahoreSeasonClassifier` with documented month-to-season mapping as single authoritative source.
- Built [`src/training_pipeline/walk_forward_validator.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/walk_forward_validator.py): modular `WalkForwardFoldEngine`, `FoldTrainer`, `DiagnosticEvaluator` (5 dimensions), `WalkForwardReportGenerator`.
- Built [`src/training_pipeline/run_walk_forward.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/run_walk_forward.py): experiment runner executing 4 folds × 4 models = 16 evaluations.
- Embargo invariant explicitly verified: `val_start > train_end + 72h` for all 4 folds (gap: 4 days 1 hour each).
- All fold preprocessing (scaler + model) refit from scratch on each fold's training data only.
- **Walk-Forward Stability Results (EXP-019 vs Naive)**:
  - **4/4 folds**: EXP-019 outperforms Naive Persistence.
  - **Mean RMSE delta: +27.0** (EXP-019 consistently better).
  - **Worst fold advantage: +21.8 RMSE points** — EXP-019 never worse than Naive.
  - **Relative gain: 24.5%** vs Naive (stronger than the 11% observed on final test set).
  - **Fold std: 4.9** — consistent margin, not a regime-specific artifact.
- **Walk-Forward Per-Fold Overall RMSE**:
  | Fold | Season | EXP-019 | EXP-017 | Ridge v2 | Naive |
  | :--- | :--- | :---: | :---: | :---: | :---: |
  | F1 | Winter/Smog 2021 | 104.23 | 102.76 | 103.08 | 139.04 |
  | F2 | Transition+Summer 2022 | 72.43 | 72.67 | 72.95 | 94.28 |
  | F3 | Monsoon 2022 | 71.88 | 71.15 | 71.42 | 96.03 |
  | F4 | Winter/Smog 2023 | **85.21** | 85.47 | 85.79 | 112.49 |
- **Key diagnostic findings**:
  1. EXP-019's persistence blend is most impactful in Winter/Smog (highest AQI variance regime).
  2. Summer/Monsoon models generalize well (RMSE 71–72); Winter/Smog is the hardest regime.
  3. R² across folds = 0.14–0.38. Remaining error points to need for winter-specific features (crop burning calendar, inversion layer indicators, more training history).
  4. Distribution shift (Wasserstein) largest in Fold 1 (earliest winter with least training context).
- **Reports**: `data/models/walk_forward/walk_forward_report.json`, `walk_forward_summary.csv`.
- **Notebook**: [`notebooks/14_walk_forward_stability.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/14_walk_forward_stability.ipynb).

### Phase 11.5: Targeted Winter/Smog Feature Investigation
- Built [`src/feature_pipeline/winter_smog_features.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/feature_pipeline/winter_smog_features.py): 4 candidate feature families (Thermal Inversion Proxy, Fog/Mist Indicator, Stagnation Enhancement, Seasonal Emission Proxy), all backward-looking, no external API dependencies.
- Built [`src/training_pipeline/run_winter_ablation.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/run_winter_ablation.py): 6-way ablation matrix (`ABL-000` to `ABL-005`) evaluated across all 4 walk-forward folds using frozen EXP-019 architecture.
- Built [`tests/test_winter_smog_features.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_winter_smog_features.py): 43 comprehensive unit tests validating proxy naming, backward-looking constraints, crop burning logic, and 6-criterion adoption gate.
- **Ablation Results Summary (Overall RMSE by Configuration)**:
  | Ablation ID | Description | F1 (Winter 21) | F2 (Trans/Sum) | F3 (Monsoon) | F4 (Winter 23) | Mean RMSE | Winter Mean Δ |
  | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **ABL-000** | Baseline (113 features) | 104.23 | 72.43 | 71.88 | 85.21 | 83.44 | Baseline |
  | **ABL-001** | + Family 1 (Inversion Proxy) | 104.38 | 72.29 | 71.81 | 85.20 | 83.42 | -0.064 |
  | **ABL-002** | + Family 2 (Fog/Mist Indicator)| 104.30 | 72.40 | 71.84 | 84.94 | 83.37 | +0.103 |
  | **ABL-003** | + Family 3 (Stagnation Enh.) | 104.24 | 72.78 | 72.24 | 85.37 | 83.66 | -0.084 |
  | **ABL-004** | + Family 4 (Seasonal Emission) | 103.86 | 72.40 | 71.75 | 85.31 | 83.33 | +0.139 |
  | **ABL-005** | + All 4 Families Combined | 104.28 | 72.55 | 72.00 | 85.16 | 83.50 | +0.005 |
- **Key Scientific Findings**:
  1. **Surface Signal Saturation**: The entire spread across all ablations is **only 0.11 RMSE points** (83.33 to 83.44). Surface meteorological features are near their asymptotic information limit.
  2. **Stagnation Redundancy**: Family 3 produced zero/negative incremental gain, confirming that existing `stagnation_index` (`pm2_5 / (wind_speed + 0.5)`) and wind lags already capture surface dispersion dynamics.
  3. **Physical Limitation of Inversion Proxies**: Surface temperature changes cannot resolve the true vertical thermal structure (PBL height and lapse rates aloft) without upper-air sounding or vertical reanalysis data.
  4. **Decision: Outcome B (EXP-019 Retained as Champion)**: Marginal gains (<0.15 RMSE) do not warrant added schema complexity. EXP-019 remains the locked production model.
- **Reports**: `data/models/walk_forward/ablation/ablation_report.json`, `ablation_summary.csv`, `ablation_decision.json`.
- **Notebook**: [`notebooks/15_winter_smog_ablation.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/15_winter_smog_ablation.ipynb).
- **Total test suite**: **233/233 tests passing (75% total codebase coverage)**.


