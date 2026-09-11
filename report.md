# Pearls-AQI-Predictor Project Report

## 1. Introduction and Problem Statement

### 1.1 Problem Context
Air quality in Lahore, Pakistan (31.55°N, 74.34°E) is consistently ranked among the worst globally, posing severe public health risks. The situation becomes particularly acute during the winter smog season, which typically spans from November to February. While monitoring infrastructure provides observations of current pollutant levels, there is a critical operational gap in actionable, multi-hour-ahead forecasting. Public health planning, municipal advisories, school closures, and personal mitigation strategies require reliable advance warning rather than solely reactive monitoring.

### 1.2 Project Goal
The primary objective of this project was to architect, validate, and deploy a cloud-native, fully managed 72-hour air quality forecasting and MLOps system. The system predicts the US Environmental Protection Agency (EPA) Air Quality Index (AQI) on its standardized 0 to 500 scale across 72 continuous hourly horizons ($h=1, \dots, 72$), providing a three-day operational window.

While the original educational specification described a "100% serverless" stack, the production implementation uses managed cloud platforms (Streamlit Community Cloud, Render Web Services and Cron Jobs, GitHub Actions, and Hopsworks Feature Store). This architecture fulfills the operational intent of zero self-managed server infrastructure while leveraging purpose-built managed services for WSGI serving, scheduled triggers, and online feature retrieval. A persistent Render WSGI web service does not satisfy a strict Function-as-a-Service (FaaS) definition, but represents a practical, cloud-native managed deployment.

### 1.3 Data Sources
The forecasting system integrates two primary external telemetry providers:
*   **OpenWeather Air Pollution History API**: Provides historical and near-real-time concentrations from November 27, 2020 to the present for eight pollutant components: Carbon Monoxide ($\text{CO}$), Nitrogen Monoxide ($\text{NO}$), Nitrogen Dioxide ($\text{NO}_2$), Ozone ($\text{O}_3$), Sulfur Dioxide ($\text{SO}_2$), Particulate Matter 2.5 ($\text{PM}_{2.5}$), Particulate Matter 10 ($\text{PM}_{10}$), and Ammonia ($\text{NH}_3$).
*   **Open-Meteo Historical Weather Archive & Forecast API**: Provides essential meteorological covariates, including 2-meter temperature, 2-meter relative humidity, surface pressure, 10-meter wind speed, 10-meter wind direction, and precipitation.

---

## 2. Preparation and Learning

Before writing application code, the first week was dedicated to foundational preparation, reviewing resources shared on the course Discord server. This study period covered Git and GitHub workflows, Python project structuring, CI/CD automation with GitHub Actions, and production machine learning pipelines.

While core machine learning, Python programming, and data manipulation concepts were familiar from university coursework, translating them into an automated, fault-tolerant production system required structured preparation. Version control best practices established strict Git hygiene from repository inception.

GitHub Actions and CI/CD pipelines were comparatively new topics. Engaging with the course materials provided an understanding of runner lifecycles, ephemeral virtual environments, encrypted secret handling, event triggers (`schedule`, `workflow_dispatch`, `repository_dispatch`), and workflow artifact persistence. This preparation proved foundational when designing the dual-scheduler feature pipeline, automated candidate model evaluation, and cross-platform CI test suites.

---

## 3. Data Collection and Exploration

### 3.1 Historical Data Ingestion
The foundational dataset was acquired from the OpenWeather Air Pollution History API, starting from the earliest available Unix timestamp (`1606482000`, corresponding to November 27, 2020 13:00:00 UTC). To handle network constraints and rate limits, a historical backfill module (`src/feature_pipeline/backfill.py`) was implemented. Downloads were executed in monthly chunks using `concurrent.futures.ThreadPoolExecutor` with worker throttling (`max_workers=5`).

Network resilience was built in through an exponential backoff retry strategy (3 retries with a 1.0-second base delay). To prevent corrupted partial files in the event of failure, data was written using atomic writes via a `.tmp` file rename pattern.

### 3.2 Dataset Validation & Completeness
Raw data underwent a completeness audit enforcing a `MIN_COMPLETENESS_RATIO` of 0.95 (95%). While overall historical completeness exceeded this threshold across the multi-year corpus, the raw time series was not perfectly continuous and contained occasional missing hourly timestamps due to upstream sensor dropouts. Preprocessing handled missing sensor sentinel values (recorded as `-9999`). Downstream training integrity was preserved by implementing exact physical timestamp target construction and boundary-gap filtering rather than assuming unbroken row sequences.

### 3.3 Meteorological Covariates
To supplement pollutant measurements, historical weather data was retrieved from the Open-Meteo archive. Variables were validated against atmospheric physical bounds:
*   Temperature: bounded within $[-50^\circ\text{C}, 65^\circ\text{C}]$
*   Relative humidity: bounded within $[0\%, 100\%]$
*   Surface pressure: bounded within $[800\text{ hPa}, 1100\text{ hPa}]$
*   Wind speed: non-negative values

The initial investigative work is documented in `01_api_investigation.ipynb` and `02_raw_data_exploration.ipynb`.

---

## 4. Feature Engineering

The feature space evolved from an initial 64 pollutant-only set to a comprehensive **114 canonical production model-input columns** (which correspond to **113 fitted predictors** when `dt` is excluded during walk-forward validation and candidate evaluation). `dt` is retained in the canonical schema for temporal ordering, fold construction, physical timestamp alignment, and provenance tracking.

```text
Canonical Feature Schema Breakdown (114 Total Columns):
├── Temporal & Cyclical Signals (7):
│   hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, is_weekend
├── Base Pollutants & Calculated AQI (9):
│   pm2_5, pm10, no2, so2, co, o3, nh3, no, epa_aqi
├── Base Meteorology (7):
│   temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, precipitation, wind_dir_sin, wind_dir_cos
├── Pollutant Lags (25):
│   1h, 3h, 6h, 12h, 24h lags for pm2_5, pm10, no2, o3, epa_aqi (5 variables × 5 lags)
├── Weather Lags (20):
│   1h, 3h, 6h, 12h, 24h lags for temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m (4 variables × 5 lags)
├── Rolling Statistics (34):
│   ├── Pollutant Rolling (16): 6h, 12h, 24h mean & std for pm2_5, epa_aqi (12) + 24h min & max for pm2_5, epa_aqi (4)
│   └── Weather Rolling (18): 6h, 12h, 24h mean & std for temperature_2m, relative_humidity_2m, wind_speed_10m (3 variables × 6 stats)
├── Differentials & Tendencies (6):
│   pm2_5_diff_1h, pm2_5_diff_24h, epa_aqi_diff_1h, epa_aqi_diff_24h, pressure_diff_1h, pressure_diff_24h
├── Chemical Ratios & Physical Indices (5):
│   pm_ratio, nitrogen_ozone_ratio, combustion_index, thermal_moisture_index, stagnation_index
└── Temporal Identifier (1):
    dt (integral Unix timestamp; present in canonical schema, excluded from fitted regressors)
```

### 4.1 Pollutant Features (Phase 6)
The initial feature set focused on intrinsic pollutant dynamics:
1.  **Temporal Cyclical Signals**: Sine/cosine trigonometric encodings for hour-of-day, day-of-year, and month, plus an `is_weekend` boolean flag.
2.  **Historical Lags**: Lags at $[1, 3, 6, 12, 24]$ hours for $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{O}_3$, and $\text{EPA AQI}$.
3.  **Rolling Aggregates**: Rolling mean and standard deviation over $[6, 12, 24]$ hours for $\text{PM}_{2.5}$ and $\text{EPA AQI}$, plus 24-hour min/max.
4.  **Chemical Ratios**: Interaction features capturing atmospheric chemistry:
    *   $\text{pm\_ratio} = \text{PM}_{2.5} / (\text{PM}_{10} + 1\text{e-}5)$
    *   $\text{nitrogen\_ozone\_ratio} = \text{NO}_2 / (\text{O}_3 + 1\text{e-}5)$
    *   $\text{combustion\_index} = \text{CO} / (\text{NO}_2 + 1\text{e-}5)$
5.  **Rate of Change**: 1-hour and 24-hour differentials for $\text{PM}_{2.5}$ and $\text{EPA AQI}$.

### 4.2 Weather Features (Phase 10.5B)
Atmospheric conditions dictate pollutant dispersion, accumulation, and chemical reaction rates:
1.  **Trigonometric Wind Decomposition**: `wind_dir_sin` and `wind_dir_cos` handling the circular topology of wind azimuth (derived from raw wind direction).
2.  **Barometric Pressure Tendency**: 1-hour and 24-hour pressure differentials (`pressure_diff_1h`, `pressure_diff_24h`).
3.  **Thermal Moisture Index**: $\text{temperature\_2m} \times (\text{relative\_humidity\_2m} / 100.0)$.
4.  **Weather Lags**: Lags at $[1, 3, 6, 12, 24]$ hours for temperature, humidity, wind speed, and surface pressure.
5.  **Weather Rolling Stats**: Rolling mean and standard deviation over $[6, 12, 24]$ hours for temperature, humidity, and wind speed.
6.  **Atmospheric Stagnation Index**: $\text{PM}_{2.5} / (\text{wind\_speed\_10m} + 0.5)$, capturing pollutant trapping during calm conditions.

### 4.3 EPA AQI Calculation
The target variable, US EPA AQI, is computed using official piecewise linear interpolation:
$$I_p = \frac{I_{\text{high}} - I_{\text{low}}}{C_{\text{high}} - C_{\text{low}}} (C_p - C_{\text{low}}) + I_{\text{low}}$$

This is computed across the six applicable criteria pollutants ($\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{O}_3$, $\text{NO}_2$, $\text{SO}_2$, $\text{CO}$). The overall AQI for an observation is the maximum sub-index across all six pollutants, bounded to $[0, 500]$ for historical labels while preserving unclipped floating-point outputs during model inference to track severe tail events.

### 4.4 Exploratory Winter/Smog Features (Phase 11.5)
To investigate extreme winter errors, four candidate feature families were tested:
1.  **Thermal / Inversion Proxies**: 24-hour diurnal temperature range, inversion risk proxy, 6-hour temperature drop.
2.  **Fog / Mist Indicators**: `fog_proxy` ($\text{RH} > 85\% \land \text{Temp} < 12^\circ\text{C} \land \text{Wind} < 2.0\text{ m/s}$) and 24-hour rolling fog hours.
3.  **Stagnation Enhancements**: 12h/24h rolling stagnation hours and 24h pressure stability.
4.  **Seasonal Emission Proxies**: Crop-residue burning seasonal flag (Oct 15–Nov 30) and winter emission intensity metric.

*(Note: As detailed in Section 10, these candidate features did not satisfy adoption criteria and were not included in the production feature set.)*

Documented in `03_eda_and_aqi_conversion.ipynb`, `04_feature_engineering.ipynb`, and `10_weather_enrichment.ipynb`.

---

## 5. Dataset Preparation

Formulating the dataset for multi-horizon sequence forecasting required strict temporal alignment:
*   **Multi-Output Target Construction**: The target matrix was constructed by forward-shifting the AQI series: $y_{t+h} = \text{AQI}_{t+h}$ for $h=1, \dots, 72$.
*   **Chronological Split**: An 80/20 chronological split was enforced. Random k-fold sampling was prohibited to eliminate future-to-past data leakage.
*   **Temporal Embargo**: A 72-hour embargo gap was enforced at the split boundary, purging training samples within 72 hours prior to the test partition.
*   **Feature Scaling**: `StandardScaler` was fitted strictly on the training partition ($X_{\text{train}}$) and applied without re-fitting to the test partition.

The held-out test partition consisted of **9,311 samples** spanning from June 7, 2025 to August 28, 2026 UTC. (Documented in `05_dataset_preparation.ipynb`).

---

## 6. Model Development Journey

Model development progressed from simple baselines to multi-stage hybrid architectures.

### 6.1 Phase 8: Ridge Regression vs Naive Persistence Baseline
Two foundational baselines were established:
*   **Naive Persistence**: Assumes AQI remains unchanged across all 72 horizons ($\hat{y}_{t+h} = y_t$). Zero trainable parameters.
*   **Ridge Regression (Ridge v1)**: `MultiOutputRegressor(Ridge(alpha=1.0))` wrapping 72 independent linear regressors on the initial 64 pollutant features.

Ridge v1 achieved an aggregate RMSE of 82.97 ($R^2 = 0.3741$), outperforming Naive Persistence (RMSE 85.35, $R^2 = 0.3499$). (Documented in `06_model_training_ridge.ipynb`).

### 6.2 Phase 9: Random Forest Regressor
A `RandomForestRegressor` (100 estimators) on 64 features achieved an aggregate RMSE of 89.18 ($R^2 = 0.2768$), performing substantially worse than both Ridge v1 and the parameter-free persistence baseline. The unconstrained tree ensemble overfit the training distribution and failed to generalize across multi-day forecast horizons. (Documented in `07_model_training_rf.ipynb`).

### 6.3 Phase 10: Deep Neural Network (TensorFlow)
A feed-forward deep neural network was evaluated:
$$\text{Input}(64) \to \text{Dense}(128, \text{ReLU}) + \text{Dropout}(0.3) \to \text{Dense}(64, \text{ReLU}) + \text{Dropout}(0.2) \to \text{Dense}(32, \text{ReLU}) \to \text{Dense}(72, \text{Linear})$$

Trained using Adam, MSE loss, `EarlyStopping` (patience=10), and `ReduceLROnPlateau` on a chronological validation tail. The DNN achieved an aggregate RMSE of 85.01 ($R^2 = 0.3429$). While marginally beating persistence, it did not match simple Ridge Regression. (Documented in `08_model_training_tf.ipynb`).

### 6.4 Phase 10.5A: Error & Distribution Diagnostics
Diagnostics analyzed train/test distribution shifts via Wasserstein distances and seasonal segmentation. Severe errors were heavily concentrated in the winter smog regime (November–February), where temperature inversions trap surface pollutants. (Documented in `09_error_and_distribution_diagnostics.ipynb`).

### 6.5 Phase 10.5B: Weather Enrichment
Adding meteorological covariates expanded the feature space to 114 canonical columns. Retraining Ridge on this weather-enriched dataset (Ridge v2 / EXP-005) reduced aggregate RMSE to **78.38** and raised $R^2$ to **0.4518**, demonstrating that meteorological physics provided critical predictive signal. (Documented in `10_weather_enrichment.ipynb`).

### 6.6 Phase 10.5C: Systematic Tuning & Experiment Registry
An experiment registry tracked systematic tuning iterations (EXP-001 to EXP-014) evaluated via 3-fold chronological cross-validation:
*   **EXP-001 to EXP-009**: Grid search across Ridge $\alpha \in [10^{-4}, 10^4]$.
*   **EXP-010 to EXP-012**: ElasticNet L1/L2 penalty sweeps.
*   **EXP-013 & EXP-014**: LightGBM direct multi-output and horizon-as-feature formulations.
(Documented in `11_systematic_model_tuning.ipynb`).

### 6.7 Phase 10.5D: Forecasting Architecture Experiments
Horizon-specific behavioral analysis led to hybrid architectures:
*   **EXP-015**: Multi-Pollutant Ridge (predicting 6 pollutants independently, applying EPA formula post-hoc).
*   **EXP-016**: Grouped Horizon Ridge (specialized feature subsets across $h1..6$, $h7..24$, $h25..72$).
*   **EXP-017**: Hybrid Specialist (LightGBM on $h1..6$, Ridge on $h7..72$).
*   **EXP-018**: Hybrid Multi-Pollutant Specialist.
*   **EXP-019**: Persistence-Aware Hybrid Model (combining LightGBM, Ridge, and long-horizon persistence blending). EXP-019 emerged as the champion.
(Documented in `12_forecasting_architecture_experiments.ipynb`).

---

## 7. Production Champion Model — EXP-019

The production champion is **EXP-019**, implemented in `src/models/hybrid_specialist_model.py` as `PersistenceAwareHybridModel`. It partitions the 72-hour horizon into three strategic zones:

```text
EXP-019 Horizon Partitioning:
┌───────────────────────────────────────┬───────────────────────────────────────┬───────────────────────────────────────┐
│       Zone 1: Short Horizon           │        Zone 2: Medium Horizon         │        Zone 3: Long Horizon           │
│              (h = 1..6)               │             (h = 7..37)               │             (h = 38..72)              │
├───────────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────────────┤
│ LightGBM Direct Multi-Output          │ Ridge Regression (alpha = 1.0)        │ Smooth Blended Ridge + Persistence    │
│ 6 independent GBDT regressors         │ 31 linear multi-output regressors     │ y_hat = w_h * y_Ridge + (1-w_h)*y_obs │
│ Captures non-linear local transitions │ Suppresses variance in mid-range      │ w_h linearly decays 1.0 -> 0.7        │
└───────────────────────────────────────┴───────────────────────────────────────┴───────────────────────────────────────┘
```

1.  **Short Horizon ($h=1..6$)**: LightGBM Direct Multi-Output (40 estimators, learning rate 0.1, max depth 4, 20 leaves). Tree-based gradient boosting captures non-linear diurnal shifts and rapid pollutant transitions.
2.  **Medium Horizon ($h=7..37$)**: Ridge Regression ($\alpha=1.0$). Regularized linear models suppress variance and maintain stability across day-ahead horizons.
3.  **Long Horizon ($h=38..72$)**: Blended Ridge and Persistence:
    $$\hat{y}_{t+h} = w_h \cdot \hat{y}_{\text{Ridge}, h} + (1 - w_h) \cdot y_{\text{current}}$$
    The blend weights $w_h$ decay linearly across the 35 long horizons:
    $$w_h = \text{linspace}(1.0, 0.7, 35) \quad \text{for } h = 38, \dots, 72$$

### Serialized Production Asset Configuration
While the Python class constructor default is `min_blend_weight=0.6`, the frozen EXP-019 production champion was trained, validated, and serialized with `min_blend_weight=0.7`. The serialized artifact stores its fitted parameters directly, and the class `load()` method dynamically reconstructs the exact weight array ($w_{38}=1.0 \to w_{72}=0.7$).

### Current AQI Contract Alignment
The production inference pipeline explicitly distinguishes:
*   `epa_aqi`: The observed AQI at current observation timestamp $T$.
*   `epa_aqi_lag_1h`: The historical observation at $T-1\text{h}$.

Both `AQIPredictor` and `SHAPExplainer` utilize the exact same observed current AQI (`epa_aqi`) for long-horizon persistence blending and explainability decompositions.

---

## 8. Final Test Benchmark (Phase 10.5E)

Seven finalized models were evaluated on the strict 9,311-sample held-out test partition (2025-06-07 to 2026-08-28 UTC):

| Rank | Model | Model ID | Features | RMSE | MAE | $R^2$ | h+1 RMSE | h+72 RMSE |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Persistence-Aware Hybrid** | **EXP-019** | **114** | **75.91** | **53.55** | **0.4858** | **50.43** | **77.43** |
| 2 | Hybrid Specialist | EXP-017 | 114 | 78.20 | 55.98 | 0.4543 | 50.43 | 84.53 |
| 3 | Ridge v2 (Weather-Enriched) | EXP-005 | 114 | 78.38 | 56.17 | 0.4518 | 54.41 | 84.53 |
| 4 | Ridge v1 (Pollutants-Only) | — | 64 | 82.97 | 63.14 | 0.3741 | 53.18 | 96.57 |
| 5 | TensorFlow DNN v1 | — | 64 | 85.01 | 65.94 | 0.3429 | 55.59 | 102.26 |
| 6 | Naive Persistence Baseline | — | 1 | 85.35 | 46.64 | 0.3499 | 68.95 | 89.68 |
| 7 | Random Forest Regressor | — | 64 | 89.18 | 65.06 | 0.2768 | 54.28 | 99.64 |

EXP-019 achieved the lowest aggregate RMSE (75.91), the highest $R^2$ (0.4858), and the strongest 72-hour horizon accuracy (77.43 vs 89.68 persistence). EXP-019 also demonstrated robust extreme-event tracking with an authoritative Hazardous (>300 AQI) subset RMSE of **142.65** on this partition. (Documented in `13_final_test_benchmark.ipynb`).

---

## 9. Walk-Forward Validation (Phase 11)

To evaluate temporal generalization, EXP-019 was tested across 4 temporally embargoed walk-forward folds. Each fold maintained a $>72\text{h}$ embargo gap between training and validation data. Preprocessing scalers and model weights were refitted from scratch on each fold:

| Fold | Climatological Regime | Validation Period | Validation Samples | EXP-019 RMSE | EXP-017 RMSE | Ridge v2 RMSE | Naive Persistence RMSE | EXP-019 $R^2$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1** | Winter / Smog 2021 | Nov 2021 – Feb 2022 | 2,159 | **104.23** | 102.76 | 103.08 | 139.04 | 0.1567 |
| **F2** | Transition + Summer 2022 | Mar 2022 – Jun 2022 | 2,183 | **72.43** | 72.67 | 72.95 | 94.28 | 0.1403 |
| **F3** | Monsoon 2022 | Jul 2022 – Sep 2022 | 2,135 | **71.88** | 71.15 | 71.42 | 96.03 | 0.1715 |
| **F4** | Winter / Smog 2023 | Nov 2023 – Feb 2024 | 2,159 | **85.21** | 85.47 | 85.79 | 112.49 | 0.3813 |

### Walk-Forward Findings & Scientific Calibration
*   **Consistent Advantage**: EXP-019 achieved a mean RMSE of **83.44** versus Naive Persistence's **110.46**, representing a **24.46% relative improvement** and winning 4 out of 4 folds.
*   **Regime-Dependent Error**: EXP-019 consistently outperformed persistence across all four walk-forward folds, although absolute error varied materially across temporal regimes, with winter/smog periods remaining substantially more difficult (RMSE 85–104) than summer and monsoon periods (RMSE 71–72).
*   **Error Dispersion**: Across the four folds ($[104.23, 72.43, 71.88, 85.21]$), the population standard deviation is approximately **13.14** (sample standard deviation 15.17), reflecting real seasonal variance across distinct atmospheric regimes.
(Documented in `14_walk_forward_stability.ipynb`).

---

## 10. Winter/Smog Feature Ablation (Phase 11.5)

To test whether engineered surface features could alleviate winter errors, four candidate feature families were evaluated across all 4 walk-forward folds with EXP-019 model architecture held fixed:

| Configuration | Description | Total Features | F1 RMSE | F2 RMSE | F3 RMSE | F4 RMSE | Mean RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ABL-000** | **Baseline (EXP-019)** | **114 (113 predictors)** | **104.23** | **72.43** | **71.88** | **85.21** | **83.44** |
| ABL-001 | + Thermal / Inversion Proxy | 117 | 104.38 | 72.29 | 71.81 | 85.20 | 83.42 |
| ABL-002 | + Fog / Mist Indicator | 116 | 104.30 | 72.40 | 71.84 | 84.94 | 83.37 |
| ABL-003 | + Stagnation Enhancement | 117 | 104.24 | 72.78 | 72.24 | 85.37 | 83.66 |
| ABL-004 | + Seasonal Emission Proxy | 116 | 103.86 | 72.40 | 71.75 | 85.31 | 83.33 |
| ABL-005 | + All 4 Feature Families | 124 | 104.28 | 72.55 | 72.00 | 85.16 | 83.50 |

### Scientific Conclusion
The spread of mean RMSE across all 6 ablation configurations was only **0.33 points** ($83.33$ to $83.66$), with a best improvement of just **0.11 RMSE points** over baseline. None of the candidate feature sets met the adoption threshold. The original 114-column EXP-019 feature set was retained.

This established that the tested surface-level proxy families produced negligible improvement under this experiment, suggesting that richer vertical atmospheric profiles (such as Planetary Boundary Layer height and vertical temperature lapse rates) and satellite-derived emission telemetry may be necessary for additional performance gains. (Documented in `15_winter_smog_ablation.ipynb`).

---

## 11. Inference Pipeline (Phase 12)

The inference pipeline (`src/inference/`) provides deterministic, leak-safe forecasting in production:

*   **`ModelLoader`**: Provides thread-safe instantiation of the model, feature scaler, and schema. Validates `n_features_in_ == 114`.
*   **`AQIPostProcessor`**: Maps float AQI predictions to EPA categories, colors, and severity ranks. Derives **empirical prediction error intervals** from 8,636 out-of-fold residuals from walk-forward validation:
    $$L_h = \max(0, \hat{y}_h + Q_{0.10}(e_h)), \quad U_h = \max(0, \hat{y}_h + Q_{0.90}(e_h))$$
    These are empirical residual quantile intervals, explicitly distinguished from parametric statistical confidence bounds. Extreme predictions and upper bounds above 500 are preserved numerically and categorized as Hazardous.
*   **`PredictionCache`**: File-based caching layer with TTL expiration (default: 1 hour).
*   **`AQIPredictor`**: Manages forecasting workflow and timestamp semantics:
    *   `input_observed_at`: Physical measurement timestamp $T$.
    *   `forecast_origin`: Anchored strictly to `input_observed_at`.
    *   `generated_at`: System time when prediction was computed.
    *   Staleness tracking: Flags inputs older than 3 hours (`is_stale=true`).

---

## 12. REST API (Phase 13)

The serving layer is a Flask WSGI application structured using the Blueprint pattern (`src/api/`):

### Endpoints
*   **`GET /api/health`**: Service health, loaded model ID (`EXP-019`), and feature source status (`hopsworks` / `bootstrap`).
*   **`GET /api/current`**: Latest observed telemetry, calculated AQI, dominant pollutant, and current alert status.
*   **`GET /api/forecast`**: 72-hour forecast sequence, empirical prediction error intervals, and multi-horizon alert summary.
*   **`GET /api/model/info`**: Model architecture, training metadata, and benchmark metrics.
*   **`GET /api/explain?horizon=H`**: Horizon-specific SHAP attribution vectors, feature contributions, and base values.

### Observation Resolver Architecture
`ObservationResolver` (`src/inference/observation_resolver.py`) implements a resilient observation hierarchy:
1.  **`auto` Mode (Default)**: Attempts live point query from Hopsworks RonDB online feature store (`aqi_weather_features_v2` v1). If online query fails or Hopsworks is unreachable, falls back gracefully to committed bootstrap vector with `is_stale=true`.
2.  **`hopsworks` Mode**: Queries only Hopsworks online store; fails closed if unavailable.
3.  **`bootstrap` Mode**: Uses committed local bootstrap vector (`data/runtime/bootstrap/latest_feature_vector.json`).

The resolver strictly validates the 114-column canonical schema order, integral Unix timestamp `dt`, finite numeric values, and calculates the current observed AQI.

---

## 13. Streamlit Dashboard (Phase 14)

The frontend is an interactive Streamlit application (`src/dashboard/`):

```text
Streamlit Dashboard Structure:
├── Header & Lahore Context: "Pearls Air — Lahore AQI Forecast"
├── Top-Level Alert Banners: Emergency alerts, forecast severe warnings, stale telemetry warnings
├── Hero Observation Card:
│   ├── Current AQI Value & EPA Category Badge
│   ├── Semicircular Gauge Component with EPA Color Mapping
│   ├── "What is affecting air quality now?" Context Card
│   └── Current Pollutant Breakdown (PM2.5, PM10, NO2, SO2, CO, O3) & Weather Metrics
├── 72-Hour Outlook Summary Card:
│   └── Current-to-Peak Progression, Peak AQI, Highest Category, First Unhealthy Hour, Peak Horizon
├── Interactive 72-Hour Trajectory Chart (Plotly):
│   ├── Point Prediction Curve anchored to physical observation timestamp
│   ├── Empirical Prediction Error Interval Bands (10th to 90th percentile OOF residuals)
│   ├── Reference Category Threshold Lines (151 Unhealthy, 201 Very Unhealthy, 301 Hazardous)
│   └── Multi-Horizon Milestone Cards (+1h, +12h, +24h, +48h, +72h)
├── Health Guidance & Actionable Recommendations: Category-tailored advice for sensitive groups & general public
├── SHAP-Powered Explainability ("Why this forecast?"):
│   ├── Horizon-Specific Attribution Decomposition (Interactive Horizon Slider h1..72)
│   ├── Top Driving Features with Non-Causal Directional Labels ("upward pressure", "downward pressure")
│   └── Global Feature Importance Visualization across 72 Horizons
└── Model Provenance & Telemetry Metadata Card:
    └── Architecture details, offline validation benchmarks, active data source, and physical observation timestamps
```

### Key Frontend Characteristics
*   **Decoupled Client**: `DashboardDataClient` queries the backend Flask API over HTTPS via `FLASK_API_URL`. Local inference fallback is disabled on cloud deployment (`ENABLE_LOCAL_FALLBACK=false`).
*   **Physical Time Anchoring**: Forecast hours are anchored to physical observation time $T$, preventing time distortion.
*   **Empirical Uncertainty Display**: Shaded error bands are clearly labeled as empirical residual quantiles from walk-forward testing.

---

## 14. Automation & MLOps Architecture (Phases 15 & 16)

The automation topology coordinates feature ingestion, feature store synchronization, candidate model evaluation, and inference serving:

```text
                                  MLOps Production Topology
                                  
   [OpenWeather API]               [Open-Meteo API]
          │                               │
          └───────────────┬───────────────┘
                          │
       ┌──────────────────┴──────────────────┐
       │ Primary Trigger:                    │ Backup Trigger:
       │ Render Cron Job                     │ GitHub Actions Cron
       │ (15 * * * * UTC)                    │ (17 * * * * UTC)
       │ aqi-hourly-ingest-trigger           │ feature_pipeline.yml
       └──────────────────┬──────────────────┘
                          │ repository_dispatch {"event_type":"hourly_ingest"}
                          ▼
            [GitHub Actions: Feature Pipeline]
            (run_hourly_ingestion --live --lookback-hours 72)
                          │
                          │ Validates 114/114 Schema & Streaming Write
                          ▼
             [Hopsworks Feature Store]
             (aqi_weather_features_v2, Version 1)
             ├── Online Store (RonDB) ──> Sub-second KV queries
             └── Offline Store (Hudi) ──> 48,718 historical records (Sep 7 Snapshot)
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼ (Daily 02:45 UTC)             ▼ (Hourly / On-Demand)
[Daily Candidate Training]        [Observation Resolver]
- Reads offline Hudi store        - Queries RonDB online store
- 73h Embargo Split               - Falls back to bootstrap vector
- Evaluates candidate models      - Validates 114 canonical features
- Asserts champion immutability           │
- Never auto-promotes                     ▼
                               [Render Web Service]
                               (Flask REST API: aqi-forecasting-api)
                               - Serves EXP-019 frozen runtime bundle
                               - Computes 72h forecast & SHAP attributions
                                          │
                                          │ HTTPS REST API
                                          ▼
                               [Streamlit Cloud Frontend]
                               (https://aqi-forecasting.streamlit.app)
```

### 14.1 Hourly Feature Ingestion Workflow
*   **Primary Scheduler**: Independent Render Cron Job `aqi-hourly-ingest-trigger` scheduled at `15 * * * *` UTC. Sends `POST /repos/hwaleedkhalid/aqi-forecasting-system/dispatches` with `{"event_type": "hourly_ingest"}`.
*   **Backup Scheduler**: Native GitHub Actions schedule at `17 * * * *` UTC in `.github/workflows/feature_pipeline.yml`.
*   **Single Ingestion Executor**: GitHub Actions runner executes canonical `run_hourly_ingestion.py`. No feature-engineering logic is duplicated on the external scheduler.
*   **Idempotency (`upsert_if_newer`)**: Dual triggers for the same provider timestamp $T$ execute idempotently without row duplication.
*   **Lookback Construction**: Fetches 72 hours of pollutant and weather telemetry to assemble unbroken 24-hour backward lag and rolling windows.
*   **Credential Security**: `GITHUB_TOKEN` is stored as a protected environment variable on Render with fine-grained permission `Contents: Read and write`.

---

## 15. Daily Candidate Training and Evaluation from Hopsworks Feature Store

### 15.1 Objectives and Operational Context
To complete the full MLOps automation lifecycle, a daily Feature-Store-driven candidate model training and evaluation pipeline was implemented. While the hourly ingestion pipeline continuously publishes fresh telemetry and meteorology into the Hopsworks Feature Store (`aqi_weather_features_v2` v1), the training pipeline operationalizes the consumption of this accumulated offline store.

The daily candidate training workflow operates under strict scientific and operational constraints:
1.  **Feature-Store-Driven Training**: Authoritative historical training data is retrieved directly from the Hopsworks Feature Store rather than re-downloading raw external APIs.
2.  **Strict Production Champion Immutability**: EXP-019 remains the immutable production champion. Under no circumstances does the daily pipeline automatically promote, overwrite, or deploy candidate models.
3.  **Temporal Validity & Anti-Leakage Invariants**: Strict chronological splitting, physical timestamp target construction, and a mandatory 73-hour embargo gap prevent any information leakage from future observations into earlier training partitions.
4.  **Protected Holdout Preservation**: Candidate development is strictly quarantined to observations prior to June 7, 2025. The entire 10,211-row post-cutoff/quarantined region (spanning June 7, 2025 through September 7, 2026), which contains the formal protected 9,311-sample final test partition (2025-06-07 to 2026-08-28) plus later accumulated observations, remains 100% untouched.

### 15.2 Authoritative Feature Store Data Retrieval
The training pipeline implements `FeatureStoreTrainingLoader` (`src/training_pipeline/dataset_builder.py`) to interface directly with the offline store:
*   **Feature Group & Entity Query**: Queries `aqi_weather_features_v2` (version 1) on project `aqi_predictor_by_Waleed` with an entity filter: `fg.select_all().filter(fg.location_id == "lahore").read()`.
*   **Bounded Arrow Flight Timeout**: Historical reads retrieve the full time series via Apache Arrow Flight with `read_options={"timeout": 300}` enforcing a bounded 5-minute timeout.
*   **Auditable Deduplication**: Event timestamps (`dt`) are validated for strict integral Unix seconds. Duplicate event timestamps are audited: identical feature vectors at duplicate timestamps are deduplicated and logged with full audit counts, while conflicting feature values at the same timestamp raise an immediate `ValidationError`.
*   **Schema & Finiteness Audits**: Asserts the exact presence of all 114 canonical model-input columns (including `dt`). All features are audited for numeric finiteness; any `NaN` or `Inf` values raise a `ValidationError`.

In live execution, the query retrieved **48,718 historical hourly records** (verified offline audit snapshot as of `2026-09-07 21:00:00 UTC`, `dt=1788814800`, spanning from `2020-11-28 13:00:00 UTC`).

### 15.3 Protected Holdout Preservation
To maintain scientific integrity and prevent data dredging across the production test set, candidate training enforces a hard temporal boundary:
$$\text{HOLDOUT\_START\_DT} = 1749254400 \quad (2025\text{-}06\text{-}07\text{T}00:00:00\text{Z})$$

*   **Quarantined Development Period**: Only observations strictly prior to `2025-06-07T00:00:00Z` ($dt < 1749254400$) are admitted into candidate development, yielding **38,507 development samples** (spanning November 28, 2020 to June 6, 2025).
*   **Post-Cutoff/Quarantined Region**: Exactly **10,211 samples** (June 7, 2025 through September 7, 2026) are quarantined and preserved completely untouched. This post-cutoff region contains the formal protected 9,311-sample out-of-time final test partition (2025-06-07 to 2026-08-28) plus later accumulated observations. EXP-019 holdout metrics (RMSE 75.91, MAE 53.55, $R^2$ 0.4858) serve strictly as frozen reference benchmarks.

### 15.4 Exact Physical Timestamp Target Construction
In contrast to naive row-shifting (which silently corrupts multi-horizon targets when sensor dropouts create timestamp gaps), `FeatureStoreTrainingLoader.construct_physical_targets()` enforces exact physical timestamp semantics:
$$y_{T, h} = \text{AQI}(T + h \times 3600) \quad \text{for } h = 1, \dots, 72$$

1.  Each observation at timestamp $T$ indexes the target lookup dictionary for the exact timestamp $T + h \times 3600$.
2.  If any of the required 72 physical future hourly observations is missing due to an unbridgeable historical gap, the training sample is cleanly dropped.
3.  No synthetic interpolation or forward-filling is applied to prediction targets.

Across the 38,507 development records, physical target alignment retained **37,163 valid training samples** (1,344 samples at historical gap boundaries were cleanly discarded).

### 15.5 Strict Anti-Leakage Embargo Split
To evaluate candidates without temporal leakage, the 37,163 usable samples are split chronologically (80% train / 20% validation) with a mandatory 73-hour embargo gap:
*   **Train Cutoff Calculation**: For split timestamp $T_{\text{split}}$, the last allowed training input timestamp is:
    $$T_{\text{train\_cutoff}} = T_{\text{split}} - (72 \times 3600) - 3600$$
*   **Leakage Invariant Verification**: The audit asserts two non-negotiable invariants:
    $$\min(T_{\text{val\_input}}) > \max(T_{\text{train\_input}}) + 72\text{h}$$
    $$\max(T_{\text{train\_target}}) < \min(T_{\text{val\_input}})$$
*   **Leakage Audit Results**:
    *   **Train Partition**: 29,658 samples (`2020-11-28 13:00:00 UTC` to `2024-06-20 06:00:00 UTC`)
    *   **Validation Partition**: 7,433 samples (`2024-06-23 07:00:00 UTC` to `2025-06-03 23:00:00 UTC`)
    *   **Embargo Separation**: Exact **73.0 hours** between final train input and first validation input (72 boundary samples purged).
    *   **Status**: Leakage audit **PASSED**.

### 15.6 Leakage-Safe Preprocessing & Feature Isolation
*   **Predictor Matrix**: The 114 canonical columns contain `dt`, which is retained exclusively for temporal ordering, fold construction, and provenance tracking. It is strictly excluded from the fitted feature matrix, leaving exactly **113 model predictors**.
*   **Train-Only Scaling**: `StandardScaler` is fitted strictly on the training partition ($X_{\text{train}}$). The validation partition ($X_{\text{val}}$) is transformed using these frozen mean and variance parameters without any re-fitting.

### 15.7 Candidate Model Architectures & Dynamic Recommendation Gate
The candidate training runner (`src/training_pipeline/run_daily_candidate.py`) supports four candidate model families:
1.  `ridge`: MultiOutputRegressor wrapping `Ridge(alpha=10.0, random_state=42)`. The candidate Ridge model employs L2 regularization parameter $\alpha=10.0$ tuned to prevent overfitting across the 113 collinear weather and pollutant features (in contrast to the internal Ridge specialist in EXP-019 which utilizes $\alpha=1.0$ within the hybrid pipeline).
2.  `hybrid`: Multi-stage hybrid architecture mirroring EXP-019 (LightGBM on h1–h6, Ridge on h7–h37, blended Ridge + Persistence on h38–h72 with `min_blend_weight=0.7`).
3.  `lightgbm`: Direct multi-output gradient boosting across all 72 horizons.
4.  `random_forest`: MultiOutputRegressor wrapping `RandomForestRegressor`.

#### Dynamic Recommendation Gate
Under no circumstances does the gate use EXP-019 out-of-time holdout RMSE (75.91) as a numeric threshold, because the candidate is evaluated on the pre-holdout development validation partition. The pipeline dynamically classifies the outcome into:
*   `retain_champion` (default): Candidate failed to strictly beat persistence across overall RMSE or milestone horizons (h1, h24, h72), failed to meet the $\ge 20.0\%$ quality improvement threshold (`meets_quality_threshold`), or exhibited material subset regressions on extreme AQI regimes (`extreme_gt200_ok`, `extreme_gt300_ok`).
*   `manual_review_recommended`: Candidate demonstrated statistically valid improvements over persistence on the same evaluation protocol ($\ge 20.0\%$ improvement, beating persistence at h1, h24, h72), passed all stability invariants without extreme subset regressions, and merits offline inspection by the engineering team (under no circumstances automatically promoted or deployed).

### 15.8 Controlled Live Training Run Results
A complete live training run was executed against the live Hopsworks Feature Store on GitHub Actions (Run [`34213609930`](https://github.com/hwaleedkhalid/aqi-forecasting-system/actions/runs/34213609930), Workflow Artifact `10050847367` `evaluation.json`, `candidate_family="ridge"`, $\alpha=10.0$):

| Metric / Dimension | Candidate Model (`ridge`, $\alpha=10.0$) | Naive Persistence Baseline | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Overall RMSE** | **86.93** | 121.07 | **+28.20% gain** |
| **Overall MAE** | **63.54** | 81.79 | **+22.31% gain** |
| **Overall $R^2$** | **0.5440** | 0.1155 | **+0.4285 delta** |
| **h+1 RMSE** | **55.05** | 68.03 | **+19.08% gain** |
| **h+6 RMSE** | **72.18** | 100.94 | **+28.50% gain** |
| **h+24 RMSE** | **84.14** | 105.76 | **+20.44% gain** |
| **h+48 RMSE** | **91.78** | 122.23 | **+24.91% gain** |
| **h+72 RMSE** | **94.06** | 130.77 | **+28.08% gain** |
| **Severe AQI (>200) RMSE** | **102.14** (n=262,312) | 140.83 | **+27.47% gain** |
| **Hazardous AQI (>300) RMSE** | **118.67** (n=152,831) | 157.59 | **+24.70% gain** |

*   **Execution Runtime**: 50.01 seconds.
*   **Recommendation**: `manual_review_recommended` (achieved 28.20% gain vs persistence exceeding $\ge 20.0\%$ threshold, beat persistence across h1, h24, h72 without extreme subset regressions on development validation).
*   **Reference Comparison Note**: Candidate validation metrics (evaluating 2024–2025 pre-holdout validation data) cannot be directly compared to EXP-019 holdout test metrics (evaluating 2025–2026 out-of-time holdout data: RMSE = 75.91, MAE = 53.55, $R^2$ = 0.4858). They reflect separate temporal evaluation regimes.

### 15.9 Candidate Artifact Isolation and SHA256 Integrity
Candidate artifacts are strictly quarantined to unique, timestamped directories under `data/models/candidates/<run_id>/`:
```text
data/models/candidates/candidate-20260908T094225Z-ridge/
├── candidate_model.joblib          # Trained candidate model
├── scaler.joblib                   # Train-fitted StandardScaler
├── evaluation.json                 # Comprehensive multi-horizon metrics
├── dataset_provenance.json         # Feature Store row counts, temporal bounds, and schema hash
├── candidate_comparison.json       # Comparison against persistence and EXP-019 reference
└── manifest.json                   # Cryptographic SHA256 hashes of all artifacts
```

The runner actively asserts that the candidate directory is outside `data/runtime/production/`.

### 15.10 Daily GitHub Actions Automation Workflow
The training workflow (`.github/workflows/training_pipeline.yml`) has been updated and verified:
*   **Cron Schedule**: `45 2 * * *` (Daily at 02:45 UTC, providing a 30-minute buffer after primary Render cron at minute 15 and a 28-minute buffer after backup GitHub cron at minute 17).
*   **Concurrency Control**: `group: model-training-pipeline`, `cancel-in-progress: false` ensures training runs execute sequentially without race conditions.
*   **Security & Secrets**: Requires only `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`, and `HOPSWORKS_HOST`. No OpenWeather API key is required, demonstrating complete independence from external ingestion APIs.
*   **Pre/Post Immutability Assertion**: Directly hashes all four authoritative assets in `data/runtime/production/` before and after execution (`sha256sum ... > /tmp/prod_authoritative_hashes_before.txt` and `diff`), failing the entire workflow if any byte modification occurs.
*   **Artifact Retention**: Automatically uploads `data/models/candidates/` as a workflow artifact with 14-day retention.
*   **Workflow Dispatch**: The daily candidate training pipeline is configured to run daily and its exact production path has been verified through `workflow_dispatch` with `candidate_family` (`ridge`, `hybrid`, `lightgbm`, `random_forest`) and `dry_run` boolean flags.

---

## 16. Hopsworks Feature Store & Model Registry Details

### 16.1 Production Feature Group
*   **Feature Group Name**: `aqi_weather_features_v2`
*   **Version**: **1** (Feature Group ID: `52526`)
*   **Project**: `aqi_predictor_by_Waleed` (`eu-west.cloud.hopsworks.ai`)
*   **Primary Key**: `["location_id"]`
*   **Event Time**: `dt`
*   **Online Storage**: RonDB enabled (sub-second point queries)
*   **Offline Storage**: Apache Hudi (verified offline audit snapshot of 48,718 historical hourly observations from 2020-11-28 13:00 UTC to 2026-09-07 21:00 UTC)
*   **Schema**: 115 columns (`location_id` + 114 canonical model-input features)

### 16.2 Model Registry
*   **Registered Model**: `pearls_aqi_production_champion` (Version 1, ID `pearls_aqi_production_champion_1`)
*   **Registered Assets**: Runtime bundle matching `data/runtime/production/` with verified checksums.
*   **Governance Policy**: Exactly 1 registered champion version; 0 candidate models in production registry; no automated promotion.
*   **Serving Architecture**: Render WSGI service serves the verified, frozen local runtime bundle (`data/runtime/production/`) to eliminate startup download latency and external registry availability risks.

---

## 17. SHAP Explainability Engine (Phase 17)

Explainability decomposes predictions according to the three zones of `PersistenceAwareHybridModel`:
1.  **$h=1..6$**: `shap.TreeExplainer` on LightGBM estimators via interventional perturbation.
2.  **$h=7..37$**: Interventional Linear SHAP on Ridge regressors:
    $$\phi_{i, h} = \text{coef}_{i, h} \cdot (x_{\text{scaled}, i} - \mu_{\text{background}, i})$$
3.  **$h=38..72$**: Mathematical blended decomposition:
    $$\text{base\_value}_h = w_h \cdot b_{\text{Ridge}, h}, \quad \phi_{\text{final}, i, h} = w_h \cdot \phi_{\text{unweighted}, i, h}, \quad \text{persistence\_comp}_h = (1 - w_h) \cdot y_{\text{current}}$$

### Validation and Additivity
Mathematical additivity is asserted:
$$\left| \left(\text{base\_value}_h + \sum_{i=1}^{114} \phi_{\text{final}, i, h} + \text{persistence\_comp}_h\right) - \hat{y}_{\text{preclip}, h} \right| < 10^{-6}$$

Global feature importance was evaluated on a 500-sample seasonally stratified evaluation cohort. The reference background distribution utilized a separate, independent 100-sample background dataset. Explanations use non-causal directional language ("contributed upward pressure", "contributed downward pressure").

---

## 18. Automated Testing & Code Coverage

The codebase is validated by **540 automated tests** spanning 35 test modules:

```text
=============================== Test Suite Summary ===============================
Collected Items: 540 passed, 0 failed, 36 warnings
Total Execution Time: ~1h 11m (including full multi-horizon SHAP tree evaluations)
Total Code Coverage: 75% (74.97% unrounded, enforcing --cov-fail-under=70)
----------------------------------------------------------------------------------
Critical Module Coverage Breakdown:
├── src/config.py                          100%
├── src/exceptions.py                      100%
├── src/feature_pipeline/aqi_calculator.py 100%
├── src/feature_pipeline/feature_eng.py    100%
├── src/inference/alerting.py               99%
├── src/feature_pipeline/winter_features.py 99%
├── src/training_pipeline/cv_evaluator.py   98%
├── src/feature_pipeline/weather_feat.py    97%
├── src/training_pipeline/diagnostics.py    97%
├── src/inference/observation_resolver.py   94%
├── src/dashboard/components.py             93%
├── src/models/tensorflow_model.py          93%
├── src/inference/model_loader.py           92%
├── src/logger.py                           92%
├── src/models/random_forest_model.py       91%
├── src/models/ridge_model.py               91%
├── src/dashboard/app.py                    90%
├── src/inference/post_processing.py        89%
├── src/inference/cache.py                  88%
├── src/inference/explainer.py              87%
├── src/inference/runtime_resolver.py       86%
├── src/api/routes.py                       83%
├── src/models/lightgbm_models.py           80%
├── src/inference/predictor.py              80%
└── src/dashboard/data_client.py            79%
==================================================================================
```

All CI workflow runs execute tests in isolated environments with external cloud dependencies 100% mocked.

---

## 19. Production Hardening & Clean-Clone Verification

To guarantee that the system executes deterministically across bare environments and cloud containers, comprehensive production-hardening protocols were implemented:

### 19.1 Clean-Clone Verification
A clean-clone test was executed in an isolated temporary directory completely devoid of untracked local developer files:
1.  Cloned repository cleanly; confirmed that `.env`, `data/models/`, `data/processed/`, `data/raw/`, and `data/logs/` were completely absent.
2.  Booted Flask WSGI using solely committed `data/runtime/` production assets.
3.  Executed live endpoint assertions:
    *   `GET /api/health` $	o$ HTTP 200 OK (`service_ready: true`, `model_id: "EXP-019"`).
    *   `GET /api/current` $	o$ HTTP 200 OK (returns valid observation schema, current AQI, stale telemetry indicator).
    *   `GET /api/forecast` $	o$ HTTP 200 OK (72 horizons anchored to physical observation origin $T$, empirical prediction error intervals).
    *   `GET /api/model/info` $	o$ HTTP 200 OK (`EXP-019`, 114 features, $R^2$ 0.4858).
    *   `GET /api/explain?horizon=24` $	o$ HTTP 200 OK (mathematical additivity error $< 10^{-6}$).

### 19.2 Runtime Asset Resolver & Fail-Closed Integrity
`RuntimeAssetResolver` (`src/inference/runtime_resolver.py`) enforces strict cryptographic and schema validation on startup:
*   **Cryptographic SHA-256 Checksums**: Validates all runtime files against `manifest.json`. Any byte modification or file tampering raises an immediate `ValidationError` and fails closed with HTTP 503.
*   **Model-Explainer Binding**: `explainer_manifest.json` binds the exact SHA-256 hashes of the model, schema, and background matrix, preventing explainability drift.
*   **Bootstrap Schema Binding**: `latest_feature_vector.json` encodes `schema_sha256`, which must match the active production schema.
*   **Finite Numeric Enforcement**: Verifies that all 114 features are present in canonical order and rejects non-finite numeric entries (`NaN`, `Infinity`, `-Infinity`).
*   **Timestamp Parsing**: Strictly validates ISO 8601 UTC timestamps.

### 19.3 Cross-Platform Line-Ending Normalization
On Windows, Git checked out JSON files with CRLF (`\r\n`), whereas Linux (Render/CI) checked them out with LF (`\n`). This caused SHA-256 checksum mismatches on text configuration files during boot.
*   **Resolution**: Added `.gitattributes` enforcing `*.json text eol=lf` and updated `compute_file_sha256` in `src/inference/runtime_resolver.py` to normalize JSON text to LF before hashing.

### 19.4 Dependency Isolation
Dependencies are segregated into distinct requirement files to optimize container build times and memory footprints:
*   `requirements.txt`: Production runtime dependencies for Flask/Gunicorn.
*   `requirements-dev.txt`: Development and testing dependencies (pytest, pytest-cov).
*   `requirements-feature-pipeline.txt`: Hopsworks and streaming dependencies (`hopsworks[python]==5.0.6`, `confluent-kafka`, `pyarrow`).
*   `src/dashboard/requirements.txt`: Lightweight UI subset (~40 MB footprint) for Streamlit Community Cloud.

---

## 20. Comprehensive Challenges and Debugging

Engineering an end-to-end forecasting system uncovered numerous analytical and operational hurdles:

### 1. The Model Complexity Paradox
*   **Problem**: Initial hypothesis assumed non-linear tree ensembles and deep neural networks would outperform linear baselines.
*   **Investigation**: Random Forest achieved RMSE 89.18 (losing to zero-parameter Naive Persistence at 85.35), while TensorFlow DNN achieved RMSE 85.01.
*   **Resolution**: Regularized linear models (Ridge $\alpha=1.0$) provided superior multi-step stability on tabular time-series with collinear predictors. LightGBM was isolated strictly to short horizons ($h1..6$).
*   **Result**: Hybrid architecture (EXP-019) achieved champion RMSE of 75.91.

### 2. Temporal Distribution Shift
*   **Problem**: Aggregate test metrics masked severe seasonal failure modes.
*   **Investigation**: Seasonal error segmentation and Wasserstein distance analysis revealed errors were concentrated in the winter smog regime.
*   **Resolution**: Walk-forward validation across 4 seasonal folds was instituted to evaluate models across distinct climatological regimes.
*   **Result**: Identified regime-specific error profiles (Winter RMSE 85–104 vs Summer RMSE 71–72).

### 3. Winter Feature Ablation Rejection
*   **Problem**: Hypothesized that surface winter proxies (fog proxy, inversion index) would reduce winter error.
*   **Investigation**: Evaluated 6 ablation configurations across 4 walk-forward folds; mean RMSE improved by only 0.11 points ($83.44 \to 83.33$).
*   **Resolution**: Followed strict software engineering principles; rejected the added feature complexity since statistical gains were negligible.
*   **Result**: Preserved canonical 114-column feature space and documented the need for vertical atmospheric data.

### 4. Canonical Feature Count & Predictor Reconciliation
*   **Problem**: Confusion between 114 stored columns and 113 fitted regressors.
*   **Investigation**: `dt` is an integral timestamp necessary for temporal ordering and provenance, but using it as a linear regressor causes temporal overfitting.
*   **Resolution**: Standardized terminology: 114 canonical production model-input columns received by the frozen scaler and model; 113 fitted predictors when `dt` is excluded during walk-forward and candidate training.
*   **Result**: Exact schema parity across scaler, model, Hopsworks feature group, and inference resolver.

### 5. Observation Timestamp Semantics
*   **Problem**: Early API iterations anchored forecast horizons to request generation time (`generated_at`).
*   **Investigation**: Network delays or stale caches caused forecast horizon timestamps to drift away from physical observation times.
*   **Resolution**: Anchored `forecast_origin` strictly to `input_observed_at`.
*   **Result**: Forecast trajectories represent true physical time steps ($T+1\text{h} \dots T+72\text{h}$).

### 6. Serialized Hybrid Blend Weight State
*   **Problem**: Hybrid model class constructor defaulted to `min_blend_weight=0.6`, whereas EXP-019 was serialized with `0.7`.
*   **Investigation**: Re-instantiating the class without deserializing saved attributes loaded the constructor default.
*   **Resolution**: Updated `load()` method to extract the serialized `min_blend_weight` and dynamically reconstruct the 35-step decay array.
*   **Result**: Exact mathematical prediction parity verified across development, CI, and production.

### 7. SHAP Dependency Compatibility
*   **Problem**: `shap>=0.50` required Python 3.11+, conflicting with Python 3.10 deployment environments.
*   **Investigation**: Dependency resolver failed during deployment builds.
*   **Resolution**: Pinned `shap==0.49.1` in production requirements.
*   **Result**: Stable explainability generation on Python 3.10.

### 8. GitHub Actions Runner Ephemerality
*   **Problem**: Generated evaluation reports and artifacts disappeared when runner jobs concluded.
*   **Investigation**: GitHub Actions runners are ephemeral containers that destroy local filesystems on completion.
*   **Resolution**: Added `actions/upload-artifact@v4` steps with explicit retention policies (7–14 days).
*   **Result**: Persistent, auditable CI/CD and training evaluation artifacts.

### 9. Hopsworks Streaming Kafka Writes vs Asynchronous Offline Materialization
*   **Problem**: Synchronous feature store writes blocked waiting for offline Spark materialization jobs (10–15 minutes), causing CI timeouts.
*   **Investigation**: Online-enabled Hopsworks feature groups write synchronously to RonDB via Kafka in ~2 seconds, while offline Hudi materialization runs asynchronously.
*   **Resolution**: Configured `wait_for_job=False` / `wait=False` for offline storage in automated pipelines.
*   **Result**: Live feature pipeline workflow execution time reduced to 43–76 seconds.

### 10. Clean-Clone Runtime Asset Absence
*   **Problem**: Cloud servers building from clean Git checkouts crashed because `data/models/` was gitignored.
*   **Investigation**: Deployment platforms have no access to untracked local developer directories.
*   **Resolution**: Created versioned `data/runtime/` package with explicit `.gitignore` whitelist rules (`!data/runtime/**`).
*   **Result**: Clean clones boot deterministically without manual file transfers.

### 11. Initial Live Cloud Feature Store State
*   **Problem**: Live Hopsworks instance initially contained zero feature groups and zero registered models.
*   **Investigation**: Development had utilized mock integration tests prior to cloud provisioning.
*   **Resolution**: Seeded production feature group `aqi_weather_features_v2` (v1) with historical observations (verified audit snapshot: 48,718 rows through Sep 7) and registered `pearls_aqi_production_champion` (v1).
*   **Result**: Fully populated cloud feature store and model registry.

### 12. Inability of Point Observations to Reconstruct Lookback Windows
*   **Problem**: Naive approach attempted to fetch a single current API reading on boot to generate forecasts.
*   **Investigation**: EXP-019 requires 24-hour backward lag and rolling features that cannot be computed from a single instantaneous reading.
*   **Resolution**: Live pipeline ingests 72-hour historical windows; cold-start boot utilizes committed canonical bootstrap vector.
*   **Result**: Cold starts succeed immediately without feature distortion.

### 13. Frontend Heavy Dependency Bloat
*   **Problem**: Streamlit Cloud build failed when `data_client.py` imported `AQIPredictor`, pulling in `scikit-learn`, `lightgbm`, and `shap`.
*   **Investigation**: Unused local fallback code caused frontend container to install heavy ML frameworks.
*   **Resolution**: Lazy-imported `AQIPredictor` inside fallback branch, configured `ENABLE_LOCAL_FALLBACK=false`, and created lightweight `src/dashboard/requirements.txt` (~40 MB).
*   **Result**: Streamlit Cloud container deploys in seconds.

### 14. Render Memory Allocation & Concurrency Sizing
*   **Problem**: Multi-worker Gunicorn configuration risked Out-Of-Memory termination on Render Free tier (512 MB).
*   **Investigation**: Continuous 5ms sampling showed baseline memory of ~178 MB with transient SHAP peaks reaching 360.81 MB, establishing that a single worker (`--workers 1`) is the safest operational baseline.
*   **Resolution**: In the production Blueprint (`render.yaml`), Gunicorn is configured with `-w 2` (`gunicorn -w 2 -b 0.0.0.0:$PORT src.api.app:app`) to handle request concurrency. However, concurrent execution of multiple heavy SHAP evaluations could potentially duplicate transient memory allocations and approach the 512 MB threshold; multi-worker concurrency under continuous heavy load remains an operational capacity risk that should be monitored.
*   **Result**: Functional WSGI serving with monitored memory boundaries.

### 15. Render Python Version & Build Timeouts
*   **Problem**: Render defaulted to Python 3.14 without binary wheels, triggering 19+ minute C++ source compilations.
*   **Investigation**: Build logs showed pip compiling `grpcio` and `google-pasta` from source.
*   **Resolution**: Pinned Python `3.10.14` via `.python-version` and separated runtime requirements from developer tools.
*   **Result**: Render build completed in under 45 seconds using pre-compiled wheels.

### 16. WSGI Start Command Shell Parsing
*   **Problem**: Start command `gunicorn "src.api.app:create_app()"` failed on Linux with status 2.
*   **Investigation**: Linux shell parsed parentheses as subshell operators.
*   **Resolution**: Exported module-level `app = create_app()` and updated start command to `src.api.app:app`.
*   **Result**: Clean Gunicorn WSGI startup on Linux.

### 17. Cross-Platform CRLF vs LF Line-Ending Hashes
*   **Problem**: SHA256 checksum of `feature_schema_v2_weather.json` mismatched between Windows development and Render Linux.
*   **Investigation**: Windows Git checked out files with CRLF (`\r\n`), altering file hashes.
*   **Resolution**: Added `.gitattributes` enforcing `*.json text eol=lf` and updated runtime resolver to normalize JSON text before hashing.
*   **Result**: SHA256 checksums match identically across Windows and Linux.

### 18. Clean CI Empty Directory Structure
*   **Problem**: Tests failed in CI runners because Git does not track empty directories (`data/raw/`, `data/processed/`).
*   **Investigation**: File handlers threw `FileNotFoundError` when attempting to write logs or temp files.
*   **Resolution**: Added tracked `.gitkeep` files across all required directories.
*   **Result**: CI test runner executes cleanly on fresh clones.

### 19. GitHub Native Scheduled Workflow Reliability Gaps
*   **Problem**: Expected hourly schedule-triggered GitHub Actions executions on cron `17 * * * *` exhibited multi-hour gaps during live monitoring.
*   **Investigation**: GitHub documentation notes that scheduled workflows are executed on a best-effort basis and that delays or dropped jobs can occur during periods of high platform load.
*   **Resolution**: Implemented `repository_dispatch` trigger (`event_type: hourly_ingest`) and configured an independent Render Cron Job (`aqi-hourly-ingest-trigger` at `15 * * * *` UTC) using `trigger_dispatch.py` to dispatch GitHub Actions externally, keeping native GitHub cron as backup.
*   **Result**: Resilient dual-trigger scheduling with idempotent ingestion.

### 20. Inability of Frontend Refresh to Generate Upstream Data
*   **Problem**: Users clicking "Refresh" on the dashboard expected newly ingested physical data when upstream pipelines had not run.
*   **Investigation**: Frontend refresh intentionally does not launch upstream ingestion because triggering live provider calls and feature store writes on user web requests would unacceptably couple presentation traffic to external provider latency, rate limits, write locks, and backend credentials.
*   **Resolution**: Stale-data warning banners transparently display observation age and explain when telemetry is historical.
*   **Result**: Honest, transparent data freshness communication.

### 21. Rejection of In-Process Flask Scheduler Daemon
*   **Problem**: Running an in-process background thread scheduler (`run_scheduler_daemon()`) inside `src/api/app.py` was initially considered for hourly dispatches.
*   **Investigation**: On Render Free tier, web services spin down after 15 minutes of inactivity, terminating background threads. Furthermore, scheduler lifecycle was coupled to Gunicorn worker restarts.
*   **Resolution**: Removed in-process scheduler daemon from Flask; returned Flask to a purely stateless REST service; deployed dedicated Render Cron Job.
*   **Result**: Independent scheduler execution decoupled from web traffic and worker restarts.

### 22. Standalone Zero-Dependency Dispatcher for Cron Jobs
*   **Problem**: Running `python -m src.feature_pipeline.scheduler --dispatch-now` in the Render Cron container failed with `ModuleNotFoundError: No module named 'numpy'`.
*   **Investigation**: `src/feature_pipeline/__init__.py` imported modules requiring numpy, but the cron container needed only standard HTTP dispatch capabilities.
*   **Resolution**: Created standalone `trigger_dispatch.py` using Python standard library (`urllib.request`, `json`, `os`, `sys`) with zero external dependencies, configured as the `startCommand` in `render.yaml`.
*   **Result**: Cron job executes in $<1$ second with zero package installation overhead.

### 23. Backend URL Migration to Blueprint Service
*   **Problem**: Migration to Render Blueprint provisioned web service `aqi-forecasting-api`, superseding legacy URL `aqi-forecasting-xyyb`.
*   **Investigation**: Streamlit frontend configuration needed to be synchronized with the new Blueprint service endpoint.
*   **Resolution**: Updated frontend configuration to `FLASK_API_URL = "https://aqi-forecasting-api.onrender.com/api"` and updated repository documentation.
*   **Result**: Live frontend communicates with active Blueprint REST backend.

---

## 21. Architectural Design Decisions

Key design choices governing the system:

1.  **Targeting US EPA AQI**: Predicting the standard 0–500 EPA AQI scale provides direct public health utility compared to raw pollutant concentrations.
2.  **72-Hour Continuous Horizon**: 3-day forecast window balances predictive capability with actionable planning lead-time.
3.  **Chronological Splitting with Embargo**: Enforcing strict temporal embargoes ($>72\text{h}$) prevents data leakage in sequential forecasting.
4.  **Hybrid Modeling Strategy**: Combining LightGBM for non-linear short-term transitions ($h1..6$), Ridge Regression for mid-range stability ($h7..37$), and Persistence Blending for long-range regularization ($h38..72$) achieved the lowest aggregate error.
5.  **Empirical Residual Uncertainty**: Quantile-based residual intervals ($Q_{0.10}$ to $Q_{0.90}$) provide realistic, non-parametric uncertainty bounds without assuming Gaussian errors.
6.  **Observation-Anchored Forecasting**: Anchoring forecast origin strictly to physical observation timestamps ensures temporal integrity regardless of network latency.
7.  **Fail-Closed Runtime Integrity**: Cryptographic SHA256 verification of models, scalers, and schemas on startup prevents silent execution of corrupted artifacts.
8.  **Decoupled Frontend & Backend**: Strict physical separation allows independent deployment, optimizes memory footprints, and protects backend credentials.

---

## 22. Lessons Learned

1.  **Complexity vs Simplicity**: Complex architectures (Random Forests, Deep Neural Networks) do not automatically outperform simple linear models on noisy time-series data. Regularized linear regression remains a strong baseline.
2.  **Data Engineering over Model Tuning**: Adding meteorological features (expanding from 64 to 114 features) produced a significantly larger performance improvement (RMSE $82.97 \to 78.38$) than extensive hyperparameter tuning on pollutant-only data.
3.  **Importance of Walk-Forward Validation**: Single-split benchmarks can mask regime-specific degradation. Walk-forward testing revealed that winter smog forecasting is fundamentally more difficult than summer forecasting.
4.  **Surface Data Limitations**: Tested surface-level weather proxy features yielded negligible incremental benefit in the winter smog ablation study, suggesting that richer vertical atmospheric profiles (such as Planetary Boundary Layer height and vertical temperature lapse rates) and satellite-derived emission telemetry may be necessary for additional forecasting gains.
5.  **Production Discipline**: Transitioning from experimental code to deployed software requires strict dependency isolation, clean-clone validation, fail-closed runtime checks, and automated CI test gates.

---

## 23. Future Work

*   **Vertical Atmospheric Profiling**: Incorporating Planetary Boundary Layer (PBL) height and vertical temperature lapse rates to model winter temperature inversions.
*   **Satellite Emission Telemetry**: Integrating active fire hotspot counts from satellite sensors (MODIS/VIIRS) to capture agricultural crop burning spikes.
*   **Multi-Station Spatial Ingestion**: Integrating additional monitoring stations (e.g. AQICN network) across Lahore and Punjab to capture spatial dispersion.
*   **Multi-Season Winter History**: Accumulating additional winter seasons in the Hopsworks feature store to expand training diversity under extreme smog events.
*   **Automated Model Governance**: Implementing statistically gated, canary-style champion promotion pipelines after establishing formal safety boundaries.

---

## 24. Analytical & Modeling Notebooks

1.  `notebooks/01_api_investigation.ipynb` — OpenWeather API rate limits and response structures.
2.  `notebooks/02_raw_data_exploration.ipynb` — Historical data ingestion validation.
3.  `notebooks/03_eda_and_aqi_conversion.ipynb` — EPA AQI conversion and pollutant distributions.
4.  `notebooks/04_feature_engineering.ipynb` — Temporal, lag, and rolling feature generation.
5.  `notebooks/05_dataset_preparation.ipynb` — Multi-output target alignment and chronological split.
6.  `notebooks/06_model_training_ridge.ipynb` — Ridge baseline model training.
7.  `notebooks/07_model_training_rf.ipynb` — Random Forest regressor evaluation.
8.  `notebooks/08_model_training_tf.ipynb` — TensorFlow DNN architecture and training.
9.  `notebooks/09_error_and_distribution_diagnostics.ipynb` — Distribution shift and seasonal error analysis.
10. `notebooks/10_weather_enrichment.ipynb` — Open-Meteo meteorological feature enrichment.
11. `notebooks/11_systematic_model_tuning.ipynb` — Cross-validation sweeps and experiment tracking.
12. `notebooks/12_forecasting_architecture_experiments.ipynb` — Hybrid architecture development.
13. `notebooks/13_final_test_benchmark.ipynb` — 7-model benchmark on held-out test partition.
14. `notebooks/14_walk_forward_stability.ipynb` — 4-fold temporal walk-forward validation.
15. `notebooks/15_winter_smog_ablation.ipynb` — Winter smog feature ablation study.

---

## 25. Live Deployment & Operational Endpoints

### 25.1 Production Endpoints
*   **Frontend Web Dashboard**: [https://aqi-forecasting.streamlit.app](https://aqi-forecasting.streamlit.app) (Streamlit Community Cloud)
*   **Backend REST API**: [https://aqi-forecasting-api.onrender.com/api](https://aqi-forecasting-api.onrender.com/api) (Render Web Service)

### 25.2 CORS Configuration
The Flask REST API is configured via `render.yaml` to permit cross-origin requests from the production dashboard and local development interfaces:
```yaml
CORS_ORIGINS: "https://aqi-forecasting.streamlit.app,http://localhost:8501,http://127.0.0.1:8501"
```

### 25.3 Streamlit Production Configuration
```toml
FLASK_API_URL = "https://aqi-forecasting-api.onrender.com/api"
ENABLE_LOCAL_FALLBACK = "false"
```

---

## 26. High-Severity & Hazardous AQI Alert System

### 26.1 Centralized Classification Architecture & Single Source of Truth
The alerting layer was implemented as an authoritative, centralized module in `src/inference/alerting.py`. A core design tenet was eliminating divergent threshold implementations between backend API responses, frontend visualizations, and domain category definitions.

To avoid contradictions between floating-point predictions and integer EPA category breakpoints, the alerting system enforces the single-path pipeline:
$$\text{numeric AQI} \longrightarrow \text{get\_aqi\_category()} \longrightarrow \text{category} \longrightarrow \text{classify\_category\_alert()}$$

Because `get_aqi_category()` rounds raw float AQI values to the nearest integer prior to range evaluation, evaluating alerts directly from the resulting category guarantees that the displayed category, hex color, alert level, and severity rank are mathematically locked across all system interfaces.

### 26.2 Exact Category Boundaries & Severity Mapping
The centralized schema (`CATEGORY_TO_ALERT_CONFIG`) maps each official EPA category to a normalized severity rank (0 to 5) and alert level:

| EPA AQI Range | EPA Category | Alert Level | Severity Rank | Active Alert | Threshold | Official Reference Color |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0 – 50** | Good | `none` | 0 | False | None | `#00E400` |
| **51 – 100** | Moderate | `none` | 1 | False | None | `#FFFF00` |
| **101 – 150** | Unhealthy for Sensitive Groups | `advisory` | 2 | True | 101.0 | `#FF7E00` |
| **151 – 200** | Unhealthy | `warning` | 3 | True | 151.0 | `#FF0000` |
| **201 – 300** | Very Unhealthy | `severe` | 4 | True | 201.0 | `#8F3F97` |
| **301 – 500+** | Hazardous | `hazardous` | 5 | True | 301.0 | `#7E0023` |

*Color System Note: The table lists standard EPA reference regulatory colors. The redesigned Streamlit UI applies a harmonized frontend design palette optimized for accessibility and contrast in dark/light themes while maintaining strict numerical parity on thresholds.*

Extreme predicted AQI values exceeding 500 are preserved as unclipped floating-point numbers and mapped into the Hazardous tier (severity rank 5).

### 26.3 Current Observation vs Forecast Trajectory Alert Distinction
To provide transparent public risk communication, the system strictly distinguishes between *current observed conditions* and *predicted future conditions*:
*   **Current Observations**: Evaluated by `evaluate_current_alert()`, returning:
    `"Current observed air quality in Lahore is [X] AQI ([Category]). [Category-level health advisory text]"`
*   **Forecast Trajectory**: Evaluated by `evaluate_forecast_alerts()` scanning all 72 horizons, phrasing alerts with scientific calibration:
    `"The model forecasts [Level] air quality conditions within the next 72 hours (Peak AQI [X] at +[H]h). First enters [Level] range at +[H]h. A total of [N] forecast hours are in the [Level] range."`

This ensures that model predictions are never conflated with observed sensor measurements.

### 26.4 Multi-Horizon Scanning: First Threshold Crossings & Horizon Counts
The multi-horizon scanner sequentially examines the 72-hour forecast sequence, capturing:
*   **Peak Event**: `peak_aqi`, `peak_horizon`, `peak_timestamp`, and `peak_category`.
*   **First Crossings**: Captures the earliest horizon and timestamp where the forecast enters or exceeds each severity tier:
    *   `first_advisory_horizon` & `first_advisory_timestamp` ($\text{rank} \ge 2$, AQI $\ge 101$)
    *   `first_unhealthy_horizon` & `first_unhealthy_timestamp` ($\text{rank} \ge 3$, AQI $\ge 151$)
    *   `first_very_unhealthy_horizon` & `first_very_unhealthy_timestamp` ($\text{rank} \ge 4$, AQI $\ge 201$)
    *   `first_hazardous_horizon` & `first_hazardous_timestamp` ($\text{rank} \ge 5$, AQI $\ge 301$)
*   **Category-Specific Counts**: Computes non-overlapping hourly duration counts for each tier:
    *   `advisory_horizon_count` ($\text{rank} == 2$)
    *   `unhealthy_horizon_count` ($\text{rank} == 3$)
    *   `very_unhealthy_horizon_count` ($\text{rank} == 4$)
    *   `hazardous_horizon_count` ($\text{rank} \ge 5$)
    *   `severe_or_higher_horizon_count`: Combined sum of Very Unhealthy and Hazardous hours ($\text{rank} \ge 4$).

### 26.5 Stale Telemetry and Bootstrap Fallback Handling
Alert evaluations independently preserve and propagate data provenance:
*   Both `evaluate_current_alert()` and `evaluate_forecast_alerts()` accept `data_is_stale`, `feature_source`, and `fallback_active`.
*   When telemetry is historical ($>3$ hours old), alerts remain numerically valid and categorized, but the system concurrently renders prominent telemetry freshness notices.
*   Dynamic cache refresh updates `data_is_stale` on every cached response, ensuring that aging cached forecasts automatically reflect staleness without requiring re-inference.

### 26.6 Empirical Upper-Bound Uncertainty Auditing
In addition to point forecasts, the scanner audits the 90th percentile empirical prediction error upper bounds (`error_upper` derived from walk-forward residual quantiles):
*   If `error_upper >= 301.0` at any horizon while the expected point forecast remains below Very Unhealthy ($<201$), `upper_interval_crosses_hazardous` is set to `True`.
*   The dashboard displays an uncertainty notice:
    `"Uncertainty Notice: The 90th percentile empirical error interval crosses the Hazardous threshold (>300 AQI) at one or more horizons, indicating extreme air pollution tail risk. Monitor ongoing hourly telemetry updates."`

### 26.7 REST API Contract Integration
The Flask API incorporates the alert contract without route proliferation:
*   **`GET /api/current`**: Enriched with `alert_level`, `severity_rank`, and the complete `alert` dictionary.
*   **`GET /api/forecast`**: Enriched with `forecast_alert` at the root response level and within `summary`, while each point in `forecasts` includes `alert_level` and `severity_rank`.
*   **Legacy Backward Compatibility**: Retains `high_severity` ($>200$), `hazardous` ($>300$), `has_high_severity`, `has_hazardous`, and `highest_alert_level` to ensure existing API consumers experience zero breaking changes.

### 26.8 Streamlit Dashboard Integration
*   **Prioritized Top-Level Banners (`render_alert_banners`)**:
    1.  Current Severe / Hazardous emergency banner (`st.error`)
    2.  Forecast Severe / Hazardous trajectory banner (`st.error`)
    3.  Stale telemetry notice (`st.warning`, displayed alongside severe alerts when data is historical)
    4.  Unhealthy warning / Advisory banners (`st.warning` / `st.info`)
    5.  Empirical error upper bound tail risk notice (`st.info`)
*   **Interactive Chart Reference Lines**: `build_forecast_figure` renders dashed horizontal lines at exact EPA boundaries: `151` (Unhealthy, red), `201` (Very Unhealthy, purple), and `301` (Hazardous, maroon).

### 26.9 Automated Alert Testing & Validation Suite
The alerting layer is protected by a dedicated regression testing suite:
*   `tests/test_alerting.py` (33 unit tests, 99% module coverage): Validates boundary transitions across 0 to 650 AQI, floating-point rounding parity, sequence simulations, severity rank assertions, and provenance flag propagation.
*   `tests/test_dashboard_alerts.py` (7 tests): Tests alert banner priority hierarchy, sidebar status rendering, and reference threshold lines.
*   `tests/test_dashboard.py` (17 tests): Tests component rendering, layout composition, and graceful error presentation.

---

## 27. Comprehensive Project Requirements Compliance Matrix

The table below maps the complete project implementation against all requirements in the original project specification:

| Requirement Area | Specification / Expectation | Implementation Status | Evidence / Verification |
| :--- | :--- | :---: | :--- |
| **Language & Core Frameworks** | Python 3.10+, Scikit-learn, LightGBM, TensorFlow | **COMPLIANT ✅** | EXP-019 hybrid model, DNN baseline, full scikit-learn preprocessing pipeline |
| **Raw External Ingestion** | OpenWeather Air Pollution & Open-Meteo Weather APIs | **COMPLIANT ✅** | `src/data_ingestion/` with atomic writes, exponential retry, and data validation |
| **Feature Engineering** | Lags, rolling aggregates, cyclical signals, ratios, weather | **COMPLIANT ✅** | 114 canonical model-input columns; exact schema in `feature_schema_v2_weather.json` |
| **Exploratory Data Analysis** | In-depth EDA notebooks, distributions, correlations | **COMPLIANT ✅** | 15 notebooks covering EDA, diagnostics, tuning, benchmarking, and ablations |
| **Historical Data Backfill** | Multi-year historical data ingestion and storage | **COMPLIANT ✅** | `src/feature_pipeline/backfill.py` retrieving history from Nov 2020 to present |
| **Feature Store Integration** | Hopsworks Feature Store integration | **COMPLIANT ✅** | `aqi_weather_features_v2` (v1, ID 52526), 48,718 offline records (Sep 7 audit snapshot), RonDB online store |
| **Feature-Store Training** | Direct model training from offline Feature Store | **COMPLIANT ✅** | `src/training_pipeline/dataset_builder.py` querying offline Hudi store via Arrow Flight |
| **Multiple ML Experiments** | Ridge, Random Forest, Deep Neural Network, LightGBM, Hybrid | **COMPLIANT ✅** | Experiments EXP-001 through EXP-019 tracked across linear, tree, DNN, and hybrid families |
| **Model Evaluation** | Multi-metric evaluation (RMSE, MAE, $R^2$, horizon limits) | **COMPLIANT ✅** | Evaluated across 72 horizons with walk-forward validation and out-of-time test benchmarks |
| **Forecast Horizon** | 72 continuous hourly predictions (3-day forecast window) | **COMPLIANT ✅** | Multi-output $h=1..72$ sequence anchored to physical observation timestamp $T$ |
| **Model Registry** | Cloud model registry registration and provenance tracking | **COMPLIANT ✅** | Hopsworks Model Registry `pearls_aqi_production_champion` (v1, ID `pearls_aqi_production_champion_1`) |
| **Daily Candidate Retraining** | Automated candidate training from Feature Store | **COMPLIANT ✅** | GitHub Actions workflow (`training_pipeline.yml` at `45 2 * * *` UTC), candidate Ridge +28.20% gain |
| **Web REST API** | Flask REST API serving predictions, health, and metadata | **COMPLIANT ✅** | 5 Blueprint endpoints (`/health`, `/current`, `/forecast`, `/model/info`, `/explain`) on Render |
| **Interactive Dashboard** | Streamlit web UI with interactive charts and alerts | **COMPLIANT ✅** | Streamlit Community Cloud app with Plotly trajectories, hero gauge, and SHAP explorer |
| **Explainable AI** | Model interpretability via SHAP attributions | **COMPLIANT ✅** | Exact mathematical decomposition for hybrid model; `/api/explain` endpoint and UI explorer |
| **Hazardous AQI Alerts** | Public health alerting and extreme event tracking | **COMPLIANT ✅** | Centralized alerting engine with multi-horizon scanning, first crossings, and tail risk notices |
| **Version Control & CI/CD** | Git repository hygiene, automated testing, coverage gates | **COMPLIANT ✅** | GitHub Actions CI workflow enforcing 70% coverage gate (**540 passed tests, 75% coverage**) |
| **Hourly Ingestion Automation** | Independent hourly scheduler calling GitHub Actions | **OPERATIONAL VERIFICATION IN PROGRESS 🔄** | Render Cron Job (`aqi-hourly-ingest-trigger`, `15 * * * *` UTC) configured; initial triggered execution verified; multi-cycle scheduled cadence logging in progress |
| **100% Serverless Architecture** | Original educational specification | **PARTIAL / TERMINOLOGY CAVEAT ⚠️** | Zero self-managed infrastructure intent is satisfied via managed platforms (Streamlit Cloud, Render, GitHub Actions, Hopsworks). However, Render WSGI web service is a managed container rather than a strict FaaS architecture. |

---

## 28. Authoritative Production Hashes & Asset Integrity

The production system enforces fail-closed cryptographic verification of all runtime assets in `data/runtime/production/`:

```text
================================ Authoritative Production SHA-256 Hashes ================================
Asset File                              Exact Cryptographic SHA-256 Checksum
---------------------------------------------------------------------------------------------------------
production_hybrid_model.joblib          f51d2eff53b8eadaf7ddb615f1dc76aeed30526038c79cc68ae758e52d8412ee
feature_scaler_v2_weather.joblib        9ce7e9fcc4fe65c2ac3b52c48c1215622112c9e3827d2c42d9d2c36fa246f063
feature_schema_v2_weather.json          38a5fdea89b0c5ed9b4e233ab1c46de1837a4dccdfccf77e58b74d06d5348538
empirical_error_intervals.json          dba4571ebe0599aac7cef8c8982e88b7a7b6f5d656eeb318de960f40483dab83
=========================================================================================================
```

### Verified Verification Environments
*   **Local Repository Disk**: Hashed directly on disk in `data/runtime/production/` using SHA-256.
*   **Continuous Integration Runners**: Pre- and post-execution checksum assertions execute in `.github/workflows/training_pipeline.yml`.
*   **Model Registry Parity Verification**: Clean-download parity tests (`python -m src.inference.hopsworks_registry --all`) verify exact checksum equality with live registered assets.

---

## 29. Conclusion and Final System Status

The Pearls AQI Predictor project delivers an end-to-end, automated air quality forecasting and MLOps system for Lahore, Pakistan.

### Final Verification Status
*   **Core Forecasting & MLOps Pipeline**: **COMPLETE & OPERATIONAL ✅** (EXP-019 hybrid champion, Hopsworks Feature Store & Model Registry, Flask Blueprint REST API, Streamlit Cloud dashboard, centralized alert system, 540 automated tests with 75% coverage).
*   **Asset & Governance Integrity**: **VERIFIED & FROZEN ✅** (authoritative SHA-256 hashes locked, zero candidate auto-promotions, candidate isolation enforced).
*   **Hourly Trigger Automation**: **OPERATIONAL VERIFICATION IN PROGRESS 🔄** (independent Render Cron Job configured at `15 * * * *` UTC, zero-dependency dispatcher verified, multi-cycle scheduled cadence observation ongoing).
*   **Primary Scientific Limitation**: The winter smog season (November–February) remains significantly more difficult to forecast (RMSE 85–104) than summer/monsoon periods (RMSE 71–72). The winter ablation study demonstrated that surface-level weather proxies provide negligible incremental benefit, indicating that future forecasting breakthroughs will require vertical atmospheric profiles (Planetary Boundary Layer dynamics, lapse rates) and satellite-derived agricultural fire emissions.
