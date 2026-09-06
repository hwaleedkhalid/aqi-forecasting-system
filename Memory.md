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
| **Phase 12**| Build Multi-Horizon Inference Pipeline | **Completed** ✅ | 2026-09-06 |
| **Phase 13**| Build Flask REST API | **Completed** ✅ | 2026-09-06 |
| **Phase 14**| Build Streamlit UI Dashboard | **Completed** ✅ | 2026-09-06 |
| **Phase 15**| Automate with GitHub Actions CI/CD | **Completed** ✅ | 2026-09-06 |
| **Phase 16**| Hopsworks Cloud Feature Store Integration | **Completed** ✅ | 2026-09-06 |
| **Phase 17**| SHAP Model Explainability | **Completed** ✅ | 2026-09-06 |
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
  | **ABL-000** | Baseline (114 canonical model-input columns / 113 predictors in walk-forward evaluation excluding dt) | 104.23 | 72.43 | 71.88 | 85.21 | 83.44 | Baseline |
  | **ABL-001** | + Family 1 (Inversion Proxy) | 104.38 | 72.29 | 71.81 | 85.20 | 83.42 | -0.064 |
  | **ABL-002** | + Family 2 (Fog/Mist Indicator)| 104.30 | 72.40 | 71.84 | 84.94 | 83.37 | +0.103 |
  | **ABL-003** | + Family 3 (Stagnation Enh.) | 104.24 | 72.78 | 72.24 | 85.37 | 83.66 | -0.084 |
  | **ABL-004** | + Family 4 (Seasonal Emission) | 103.86 | 72.40 | 71.75 | 85.31 | 83.33 | +0.139 |
  | **ABL-005** | + All 4 Families Combined | 104.28 | 72.55 | 72.00 | 85.16 | 83.50 | +0.005 |
- **Schema Note (114 canonical model-input columns / 113 predictors in walk-forward evaluation excluding dt)**: The production champion model EXP-019 is serialized with the canonical 114-feature schema (where feature index 4 is the numeric epoch timestamp `dt`, alongside 113 predictors including meteorological, pollutant, and temporal features). During dynamic fold evaluations in Phase 11/11.5, `dt` was excluded as metadata to focus validation strictly on the 113 predictors. Both refer to the identical feature space, and the production inference contract strictly operates on the canonical 114-feature vector.
- **Key Scientific Findings**:
  1. **Limited Incremental Value from Surface Proxies**: Across all 6 configurations, mean RMSE ranges from 83.33 to 83.66 (full spread of 0.33 RMSE points). The maximum improvement over baseline is 0.11 RMSE points (83.44 → 83.33 with ABL-004), indicating surface meteorological features are near their predictive limit for this task.
  2. **Stagnation Redundancy**: Family 3 produced slight regressions in mean RMSE (83.44 → 83.66), supporting the hypothesis that the new stagnation features are largely redundant with existing features (`stagnation_index` and wind lags).
  3. **Plausible Information Gaps Beyond Surface Telemetry**: Surface proxies cannot represent vertical atmospheric profiles (PBL height, lapse rates aloft) or real-time emission dynamics (active fire counts). These remain plausible sources of additional information that the current surface-only feature set cannot represent.
  4. **Decision: Outcome B (EXP-019 Retained as Champion)**: None of the tested feature families satisfied the predefined adoption criteria. The additional feature complexity is not justified by the observed walk-forward performance. EXP-019 with the canonical 114-feature schema remains the validated production champion.
- **Reports**: `data/models/walk_forward/ablation/ablation_report.json`, `ablation_summary.csv`, `ablation_decision.json`.
- **Notebook**: [`notebooks/15_winter_smog_ablation.ipynb`](file:///d:/10Perls/Pearls-AQI-Predictor/notebooks/15_winter_smog_ablation.ipynb).
- **Total test suite**: **233/233 tests passing (75% total codebase coverage)**.

### Phase 12: Multi-Horizon Inference Pipeline
- Built [`src/inference/model_loader.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/inference/model_loader.py): Thread-safe `ModelLoader` loading production model (`production_hybrid_model.joblib`), scaler (`feature_scaler_v2_weather.joblib`), and canonical schema (`feature_schema_v2_weather.json`, 114 features in exact order).
- Built [`src/inference/post_processing.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/inference/post_processing.py): `AQIPostProcessor` generating official EPA categories, colors, public health advisories, extreme-event flags (`high_severity > 200`, `hazardous > 300`), and empirical prediction error intervals derived from 8,636 walk-forward out-of-fold residuals (`empirical_error_intervals.json`). Preserves raw non-negative predictions above 500.
- Built [`src/inference/cache.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/inference/cache.py): `PredictionCache` file-based JSON cache (`data/cached_predictions.json`) with TTL expiration and corruption safety.
- Built [`src/inference/predictor.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/inference/predictor.py): `AQIPredictor` generating 72-hour forecast vectors with strict schema/column order validation, unscaled current AQI extraction, non-negative bounding ($\max(0, \hat{y})$), staleness and data freshness auditing (`input_observed_at`, `input_age_hours`, `is_stale`), and model provenance metadata.
- Generated [`data/models/walk_forward/empirical_error_intervals.json`](file:///d:/10Perls/Pearls-AQI-Predictor/data/models/walk_forward/empirical_error_intervals.json): Empirical 10th and 90th percentile residual quantiles ($e_h = y_h - \hat{y}_h$) per horizon.
- Created modular unit tests:
  - [`tests/test_model_loader.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_model_loader.py) (8 tests)
  - [`tests/test_post_processing.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_post_processing.py) (6 tests)
  - [`tests/test_prediction_cache.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_prediction_cache.py) (5 tests)
  - [`tests/test_predictor.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_predictor.py) (7 tests, including strict column order validation)
- **Total test suite**: **260/260 tests passing (74% total codebase coverage, 88–100% on inference modules)**.

### Phase 13: Flask REST API
- Built [`src/api/routes.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/api/routes.py): Flask Blueprint implementing endpoints:
  - `GET /api/health`: Service liveness and model readiness (returns 200 if healthy, 503 if degraded).
  - `GET /api/current`: Returns latest observed telemetry, pollutant breakdown, weather, and EPA category/color without model inference.
  - `GET /api/forecast`: Generates 72-hour forecast vectors with strict boolean query parsing (`force_refresh`), empirical error intervals, extreme alerts, and root-level freshness metadata (`data_status`, `input_observed_at`, `forecast_origin`, `generated_at`, `input_age_hours`, `is_stale`).
  - `GET /api/model/info`: Returns model provenance, architecture (EXP-019), canonical 114-feature schema, and Phase 10.5E benchmark metrics.
- Built [`src/api/app.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/api/app.py): Application factory `create_app()` with configurable CORS origins for Streamlit, standardized JSON error handlers (400, 404, 503, 500), and internal error protection (tracebacks and filesystem paths logged server-side only, never leaked in HTTP 500 responses).
- Built [`tests/test_api.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_api.py): 18 comprehensive integration and unit tests covering health status degradation, observation retrieval without inference, forecast contract validation, timestamp semantics, boolean query parsing, CORS headers, and error masking.
- **Total test suite**: **284/284 tests passing (75% total codebase coverage, 83–86% on API modules)**.

### Phase 14: Streamlit UI Dashboard
- Built [`src/dashboard/data_client.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/dashboard/data_client.py): Unified client supporting Flask REST API mode with seamless Direct Local Inference fallback.
- Built [`src/dashboard/components.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/dashboard/components.py): Reusable UI components for freshness banner, telemetry breakdown, forecast curve with empirical error bands (Plotly), and sidebar.
- Built [`src/dashboard/app.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/dashboard/app.py): Streamlit dashboard with interactive horizon inspection, EPA color coding, and live/stale telemetry alerts.
- Built [`tests/test_dashboard.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_dashboard.py): 16 comprehensive unit and contract parity tests.

### Phase 15: GitHub Actions Automation & CI
- Created [`.github/workflows/ci.yml`](file:///d:/10Perls/Pearls-AQI-Predictor/.github/workflows/ci.yml): Automated test suite execution on pull requests and pushes to main.
- Created [`.github/workflows/feature_pipeline.yml`](file:///d:/10Perls/Pearls-AQI-Predictor/.github/workflows/feature_pipeline.yml): Hourly ingestion with artifact retention.
- Created [`.github/workflows/training_pipeline.yml`](file:///d:/10Perls/Pearls-AQI-Predictor/.github/workflows/training_pipeline.yml): Weekly retraining and candidate evaluation with safety controls.
- Built [`src/feature_pipeline/run_hourly_ingestion.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/feature_pipeline/run_hourly_ingestion.py) & [`src/training_pipeline/run_candidate_evaluation.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/training_pipeline/run_candidate_evaluation.py).
- Built [`tests/test_workflows.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_workflows.py): 10 workflow validation tests.

### Phase 16: Hopsworks Cloud Feature Store Integration
- Built [`src/feature_pipeline/hopsworks_integration.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/feature_pipeline/hopsworks_integration.py): 115-column cloud storage schema (`location_id` primary key, `dt` event time) with synchronous job waiting (`wait=True`) and strict 114-column inference projection.
- Built [`src/inference/hopsworks_registry.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/inference/hopsworks_registry.py): Model registry bundle packaging with `manifest.json` SHA256 integrity verification.
- Built [`tests/test_hopsworks_integration.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_hopsworks_integration.py): 16 unit tests for cloud integration and offline fallbacks.

### Phase 17: SHAP Model Explainability
- Built [`src/inference/explainer.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/inference/explainer.py): Authoritative `ModelExplainer` engine implementing exact hybrid persistence decomposition across all 72 horizons:
  - $h=1\dots6$: Pure LightGBM TreeExplainer (interventional perturbation, check_additivity=False)
  - $h=7\dots37$: Pure Ridge LinearExplainer with background mean adjustment
  - $h=38\dots72$: Blended Ridge + Persistence ($w_h \hat{y}^{\text{Ridge}} + (1-w_h) y_t$) scaling both base value and feature attributions ($w_h b_h + \sum w_h \phi_{i,h} + (1-w_h) y_t = \hat{y}_h$)
  - Global importance aggregation: $I_i = \frac{1}{72} \sum_{h=1}^{72} \operatorname{mean}_n |\phi_{i,h,n}^{\text{final}}|$ evaluated across a 500-sample seasonal stratified cohort with a 100-sample background reference distribution.
  - Scaled-space attribution in model $z$-space with dual original-representation (`raw_value`) and normalized (`scaled_value`) API reporting.
  - Strict additivity validated against `explained_output_preclip` before post-processing non-negative bounds.
- Built [`src/inference/build_explainability_artifacts.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/inference/build_explainability_artifacts.py) and serialized `data/models/explainability/` artifacts (`shap_background.npy` [100 samples], `global_shap_importance.json` [500-sample cohort], `explainer_manifest.json`).
- Updated [`src/inference/predictor.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/inference/predictor.py): Added `explain_latest()` and `get_global_explainability()`.
- Updated [`src/api/routes.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/api/routes.py): Added `GET /api/explain` with integer validation (`horizon=1..72`, `top_k=1..114`) and root freshness metadata.
- Updated [`src/dashboard/components.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/dashboard/components.py) and [`src/dashboard/app.py`](file:///d:/10Perls/Pearls-AQI-Predictor/src/dashboard/app.py): Added interactive horizon attribution Plotly bar chart, persistence component breakdown cards, and global importance table.
- Built [`tests/test_explainer.py`](file:///d:/10Perls/Pearls-AQI-Predictor/tests/test_explainer.py): 20 comprehensive unit, mathematical additivity, routing boundary, and contract consistency tests.
- **Total test suite**: **346/346 tests passing (75% total codebase coverage)**.

