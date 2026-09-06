# Pearls-AQI-Predictor Project Report

## 1. Introduction and Problem Statement

### 1.1 Problem Context
Air quality in Lahore, Pakistan (31.55°N, 74.34°E) is consistently ranked among the worst globally, posing severe public health risks. The situation becomes particularly acute during the winter smog season, which typically spans from November to February. While real-time monitoring infrastructure exists and provides current pollutant levels, there is a critical gap in actionable, hour-ahead forecasting. Public health planning, school closures, and personal mitigation strategies require reliable advance warning rather than just reactive monitoring.

### 1.2 Project Goal
The primary objective of this project was to architect and build a serverless, 72-hour air quality forecasting system. The system predicts the US Environmental Protection Agency (EPA) Air Quality Index (AQI), which operates on a standardized 0 to 500 scale. The forecast horizon was defined as 72 continuous hourly predictions to provide a practical three-day operational window.

### 1.3 Data Sources
The forecasting system relies on two primary data streams:
*   **OpenWeather Air Pollution History API**: Provided historical concentrations from November 27, 2020 to the present for seven key pollutants: Particulate Matter 2.5 (PM2.5), Particulate Matter 10 (PM10), Nitrogen Dioxide (NO2), Sulfur Dioxide (SO2), Carbon Monoxide (CO), Ozone (O3), and Ammonia (NH3).
*   **Open-Meteo Historical Weather Archive**: Provided essential meteorological covariates, including 2-meter temperature, 2-meter relative humidity, surface pressure, 10-meter wind speed, 10-meter wind direction, and precipitation.

## 2. Preparation and Learning

Before diving into code implementation, I spent approximately my first week focused entirely on preparation, reviewing resources shared on the course Discord server. This dedicated study period covered Git and GitHub workflows, Python project structuring, CI/CD with GitHub Actions, and machine learning pipelines.

While core machine learning, Python programming, and data manipulation concepts were familiar from prior university coursework, I recognized the need for a practical refresh before applying them to a robust, end-to-end system. The Git and GitHub resources were particularly valuable in establishing strict version control habits right from the project's inception. 

Conversely, GitHub Actions and CI/CD pipelines were comparatively new topics to me. The Discord materials provided a crucial foundational understanding of these systems. I engaged with the material actively, ensuring I understood the mechanics of how CI pipelines execute, the secure management of repository secrets, the behavior of workflow triggers, and the lifecycle of pipeline artifacts. This upfront investment paid massive dividends later in the project; when it came time to implement the automation pipelines in Phase 15, the process was significantly smoother than I anticipated because the conceptual framework was already in place.

## 3. Data Collection and Exploration

### 3.1 Historical Data Ingestion
The foundational dataset was acquired from the OpenWeather Air Pollution History API, starting from the earliest available Unix timestamp (1606482000, corresponding to Nov 27, 2020). To handle the volume and network constraints, I implemented a robust historical backfill mechanism. The download was executed in monthly chunks using Python's `ThreadPoolExecutor` with `max_workers=5`. 

Network resilience was built in through an exponential backoff retry strategy, allowing 3 retries with a 1.0-second base delay. Furthermore, to prevent corrupted partial files in the event of failure, the data was written using atomic writes via a `.tmp` file rename pattern.

### 3.2 Dataset Validation
Upon collection, the raw data underwent a rigorous completeness audit. I enforced a `MIN_COMPLETENESS_RATIO` of 0.95 (95%), ensuring that the temporal integrity of the time series was sufficient for sequential modeling. The final corpus comprised approximately 50,000 hourly records. I also implemented preprocessing to handle specific missing sensor sentinel values (recorded as -9999).

### 3.3 Meteorological Covariates
To supplement the pollutant data, I integrated the Open-Meteo weather archive. I retrieved historical temperature (2m), relative humidity (2m), surface pressure, wind speed (10m), wind direction (10m), and precipitation. These variables were subjected to strict atmospheric validation bounds:
*   Temperature bounded from -50°C to 65°C
*   Relative humidity bounded from 0% to 100%
*   Wind speed constrained to non-negative values

The initial investigative work is documented in `01_api_investigation.ipynb` and `02_raw_data_exploration.ipynb`.

## 4. Feature Engineering

The feature space evolved significantly over the project lifecycle, expanding from an initial set of 64 pollutant-only features to a comprehensive 114-feature set inclusive of meteorological data.

### 4.1 Pollutant Features (Phase 6)
The initial feature set focused entirely on the intrinsic patterns within the pollutant time series:
1.  **Temporal Cyclical**: Trigonometric transformations of time components (hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos) and a boolean `is_weekend` flag.
2.  **Historical Lags**: Specific historical observations at [1, 3, 6, 12, 24] hours prior for PM2.5, PM10, NO2, O3, and the overall EPA AQI.
3.  **Rolling Aggregates**: Statistical summaries including mean and standard deviation over [6, 12, 24] hour windows for PM2.5 and EPA AQI, along with the 24-hour minimum and maximum values.
4.  **Chemical Ratios**: Interaction features capturing chemical relationships, including `pm_ratio` (PM2.5/PM10), `nitrogen_ozone_ratio` (NO2/O3), and a `combustion_index` (CO/NO2).
5.  **Rate of Change**: Short-term (1h) and diurnal (24h) deltas for PM2.5 and EPA AQI.

### 4.2 Weather Features (Phase 10.5B)
Recognizing that air pollution is heavily influenced by atmospheric physics, I enriched the dataset with Open-Meteo covariates:
1.  **Trigonometric Wind Decomposition**: `wind_dir_sin` and `wind_dir_cos` to handle the circular nature of wind direction.
2.  **Barometric Pressure Tendency**: 1-hour and 24-hour pressure differentials (`pressure_diff_1h`, `pressure_diff_24h`).
3.  **Thermal Moisture Index**: A composite variable: `temperature_2m * (relative_humidity_2m / 100.0)`.
4.  **Weather Lags**: Historical values at [1, 3, 6, 12, 24] hours for temperature, humidity, wind speed, and pressure.
5.  **Weather Rolling Stats**: Rolling mean and standard deviation over [6, 12, 24] hours for temperature, humidity, and wind speed.
6.  **Atmospheric Stagnation Index**: A custom formulation defined as `PM2.5 / (wind_speed + 0.5)` to capture accumulation under low-wind conditions.

### 4.3 EPA AQI Calculation
The target variable, EPA AQI, is not a raw measurement but a synthetic index. I implemented the official US EPA piecewise linear interpolation formula:
`I_p = ((I_high - I_low) / (C_high - C_low)) * (C_p - C_low) + I_low`

This was computed across six criteria pollutants (PM2.5, PM10, O3, NO2, SO2, CO). The overall AQI for a given hour is defined as the maximum of these sub-indices, clamped strictly to the [0, 500] range. All pollutant concentrations were truncated and rounded according to specific EPA rules prior to the calculation.

### 4.4 Winter/Smog Features (Phase 11.5 - Exploratory)
Later in the project, I explored highly specialized features targeting the severe winter smog regime:
1.  **Family 1: Thermal/Inversion Proxy**: Diurnal temperature range (24h), inversion risk proxy, and 6-hour temperature drop.
2.  **Family 2: Fog/Mist Indicator**: A boolean `fog_proxy` (defined as RH>85% AND Temp<12°C AND Wind<2.0 m/s) and a rolling count of fog hours over 24 hours.
3.  **Family 3: Stagnation Enhancement**: Rolling count of wind stagnation hours (12h/24h) and pressure stability over 24 hours.
4.  **Family 4: Seasonal Emission Proxy**: A binary flag for the crop burning season (Oct 15-Nov 30) and a general winter emission intensity metric.
*(Note: As discussed in Section 10, these features were ultimately not adopted into the champion model.)*

These efforts are detailed in `03_eda_and_aqi_conversion.ipynb`, `04_feature_engineering.ipynb`, and `10_weather_enrichment.ipynb`.

## 5. Dataset Preparation

Formulating the dataset for sequence prediction required careful temporal alignment. 
*   **Multi-Output Target Construction**: The target matrix was constructed by shifting the AQI series forward: `y_{t+h} = shift(-h)` for `h=1..72`.
*   **Data Splitting**: I employed a strict chronological 80/20 train/test split. Random sampling cannot be used in time-series forecasting due to future-to-past data leakage.
*   **Temporal Embargo**: To further prevent data leakage, a 72-hour embargo gap was enforced at the split boundary. Any training samples falling within 72 hours prior to the start of the test set were completely purged.
*   **Scaling**: `StandardScaler` was fitted strictly on the training partition. The test set was subsequently transformed using these locked parameters without any refitting.

The final held-out test partition consisted of 9,311 samples spanning from 2025-06-07 to 2026-08-28 UTC. This process is documented in `05_dataset_preparation.ipynb`.

## 6. Model Development Journey

The modeling phase was structured as a progression of increasingly complex architectures, benchmarked against rigorous baselines.

### 6.1 Phase 8: Ridge Regression + Naive Baseline
I established two critical baselines:
*   **Naive Persistence**: The simplest possible assumption: whatever the AQI is now, it will remain exactly the same for the next 72 hours (`y_hat_{t+h} = y_t`). This model has zero trainable parameters.
*   **Ridge Regression**: Using scikit-learn's `MultiOutputRegressor` wrapped around `Ridge(alpha=1.0)`. This instantiated 72 independent linear regressors. The outputs were clamped to [0, 500].

Using the initial 64-feature set (Ridge v1), the Ridge model achieved an RMSE of 82.97 and an R² of 0.3741. The Naive Persistence model yielded an RMSE of 85.35 and an R² of 0.3499. The linear model comfortably beat the naive assumption. (See `06_model_training_ridge.ipynb`).

### 6.2 Phase 9: Random Forest
Seeking to capture non-linear relationships, I trained a `RandomForestRegressor` with 100 estimators. The results were highly informative: the Random Forest achieved an RMSE of 89.18 and an R² of 0.2768. This performance was demonstrably worse than both the Ridge model and the zero-parameter Naive baseline. The tree-based ensemble was severely overfitting to the specific temporal distribution of the training set and failing to generalize. (See `07_model_training_rf.ipynb`).

### 6.3 Phase 10: TensorFlow DNN
I then designed a Deep Neural Network architecture:
`Input(n) -> Dense(128, ReLU) + Dropout(0.3) -> Dense(64, ReLU) + Dropout(0.2) -> Dense(32, ReLU) -> Dense(72, Linear)`
The network was optimized using Adam and MSE loss, with `EarlyStopping` (patience=10, restoring best weights) and `ReduceLROnPlateau` (factor=0.5). To maintain strict chronological integrity, a chronological tail (the final 15% of `X_train`) was reserved for validation; the true test set remained completely isolated. 
The DNN achieved an RMSE of 85.01 and an R² of 0.3429. While a marginal improvement over Naive Persistence, it still underperformed the simple Ridge linear model. (See `08_model_training_tf.ipynb`).

### 6.4 Phase 10.5A: Diagnostics
To understand why complex models were failing to generalize, I conducted deep diagnostic analysis. I analyzed distribution shifts between train and test sets by comparing median, 90th, and 99th percentiles, quantified by Wasserstein distance. Crucially, I segmented model performance by season (Winter Smog, Pre-Winter, Spring/Summer, Monsoon) and by AQI severity tiers, which isolated the most catastrophic errors (the top 1%) predominantly to the severe winter smog episodes. (See `09_error_and_distribution_diagnostics.ipynb`).

### 6.5 Phase 10.5B: Weather Enrichment
The diagnostics strongly indicated that the intrinsic pollutant history was insufficient. By integrating the Open-Meteo features, the feature count expanded from 64 to 114. Retraining the Ridge model on this expanded dataset (Ridge v2 / EXP-005) resulted in a massive leap in performance: RMSE dropped to 78.38, and R² climbed to 0.4518. Meteorology was the missing link. (See `10_weather_enrichment.ipynb`).

### 6.6 Phase 10.5C: Systematic Tuning & Experiment Registry
With weather data integrated, I formalized an experiment registry to track iterations from EXP-001 through EXP-019. Model evaluation utilized an expanding-window 3-fold chronological cross-validation procedure, executed solely on the training partition.
*   **EXP-001 to EXP-009**: Grid search over Ridge alpha values (1e-4 to 1e4).
*   **EXP-010 to EXP-012**: ElasticNet regularization variations.
*   **EXP-013**: LightGBM employing a direct multi-output strategy (72 distinct models).
*   **EXP-014**: LightGBM utilizing a horizon-as-feature reformulation.
(See `11_systematic_model_tuning.ipynb`).

### 6.7 Phase 10.5D: Architecture Experiments
Recognizing that different forecast horizons exhibit different statistical behaviors, I explored hybrid architectures:
*   **EXP-015**: Multi-Pollutant Ridge (predicting 6 raw pollutant concentrations and applying the EPA piecewise function post-hoc).
*   **EXP-016**: Grouped Horizon Ridge, utilizing separate feature sets for h1-6, h7-24, and h25-72.
*   **EXP-017**: Hybrid AQI Specialist, utilizing LightGBM for short horizons (h1-6) and Ridge for the remainder (h7-72).
*   **EXP-018**: Hybrid Multi-Pollutant Specialist.
*   **EXP-019**: Persistence-Aware Hybrid. This architecture emerged as the champion.
(See `12_forecasting_architecture_experiments.ipynb`).

## 7. Champion Model — EXP-019

The system's final champion is `EXP-019`, implemented as the `PersistenceAwareHybridModel`. It divides the 72-hour horizon into three distinct strategic zones:
1.  **Short Horizon (h=1..6)**: Driven by LightGBM Direct Multi-Output (40 estimators, lr=0.1, num_leaves=20). Gradient boosting effectively captures non-linear, immediate-term patterns and interactions.
2.  **Medium Horizon (h=7..37)**: Driven by Ridge Regression (alpha=1.0). Regularized linear models provide stable, un-overfitted predictions for the mid-range.
3.  **Long Horizon (h=38..72)**: Driven by a blended approach: `y_hat_h = w_h * y_Ridge_h + (1 - w_h) * y_current`. The blend weights (`w_h`) decay linearly using `np.linspace(1.0, min_blend_weight, 35)`.

The constructor sets a default `min_blend_weight=0.6`. To ensure exact reproducibility across environments, the serialized production artifact directly stores the `min_blend_weight` attribute. The class's `load()` method relies on this serialized value to dynamically recompute the precise `blend_weights` array. The model accesses the current AQI via index 9 (`epa_aqi_lag_1h` in the canonical schema).

**Rationale**: Short horizons require the flexibility of tree-based methods to capture rapid local dynamics. Medium horizons are best served by linear regression to suppress variance. At long horizons (days 2 and 3), predictive uncertainty grows dramatically; mathematically anchoring the forecast back toward the most recently observed true value acts as a powerful regularizer against extreme divergence.

## 8. Final Test Benchmark (Phase 10.5E)

I evaluated 7 frozen, finalized models against the strict 9,311-sample held-out test partition (2025-06-07 to 2026-08-28 UTC).

| Rank | Model | Features | RMSE | MAE | R² | h+1 RMSE | h+72 RMSE |
|------|-------|----------|------|-----|----|---------|---------|
| 1 | EXP-019 Persistence-Aware Hybrid | 114 | 75.91 | 53.55 | 0.4858 | 50.43 | 77.43 |
| 2 | EXP-017 Hybrid Specialist | 114 | 78.20 | 55.98 | 0.4543 | 50.43 | 84.53 |
| 3 | Ridge v2 Weather-Enriched | 114 | 78.38 | 56.17 | 0.4518 | 54.41 | 84.53 |
| 4 | Ridge v1 Pollutants-only | 64 | 82.97 | 63.14 | 0.3741 | 53.18 | 96.57 |
| 5 | TensorFlow DNN v1 | 64 | 85.01 | 65.94 | 0.3429 | 55.59 | 102.26 |
| 6 | Naive Persistence | 1 | 85.35 | 46.64 | 0.3499 | 68.95 | 89.68 |
| 7 | Random Forest v1 | 64 | 89.18 | 65.06 | 0.2768 | 54.28 | 99.64 |

EXP-019 demonstrated decisive superiority across the aggregate RMSE and particularly at the outer limit (h+72). (See `13_final_test_benchmark.ipynb`).

## 9. Walk-Forward Validation (Phase 11)

To ensure the model wasn't simply tuned to one specific temporal split, I subjected it to rigorous walk-forward validation across 4 temporally embargoed folds. Each fold maintained the strict >72h gap between training and validation data. All preprocessing (including the scaler) and model weights were refit entirely from scratch for every single fold.

I utilized a custom `LahoreSeasonClassifier` mapping months to prevailing climatology:
*   `winter_smog`: Nov, Dec, Jan, Feb
*   `transition`: Mar, Apr, Oct
*   `summer`: May, Jun
*   `monsoon`: Jul, Aug, Sep

| Fold | Season | Samples | EXP-019 | EXP-017 | Ridge v2 | Naive | EXP-019 R² |
|------|--------|---------|---------|---------|----------|-------|-----------|
| F1 | Winter/Smog 2021 | 2,159 | 104.23 | 102.76 | 103.08 | 139.04 | 0.1567 |
| F2 | Transition+Summer 2022 | 2,183 | 72.43 | 72.67 | 72.95 | 94.28 | 0.1403 |
| F3 | Monsoon 2022 | 2,135 | 71.88 | 71.15 | 71.42 | 96.03 | 0.1715 |
| F4 | Winter/Smog 2023 | 2,159 | 85.21 | 85.47 | 85.79 | 112.49 | 0.3813 |

**Analysis**:
*   EXP-019 achieved a mean RMSE of 83.44 vs Naive's 110.46, representing a massive 24.5% relative gain.
*   EXP-019 outperformed the naive baseline in all 4 out of 4 folds, with a worst-case margin of +21.8 RMSE points.
*   The fold standard deviation was tight at 4.9, proving the performance margin is structurally sound and not regime-dependent.
*   Crucially, this diagnostic confirmed the core challenge: the Winter/Smog regime is vastly harder to predict (RMSE 85-104) compared to the stable Summer/Monsoon seasons (RMSE 71-72).
(See `14_walk_forward_stability.ipynb`).

## 10. Winter/Smog Ablation (Phase 11.5)

To address the high error rates in winter folds, I tested the 4 candidate feature families outlined in Section 4.4. I kept the EXP-019 architecture completely frozen and evaluated 6 configurations (ABL-000 baseline through ABL-005 all families) across all 4 walk-forward folds.

Adoption gates were strict: the new features required demonstrated winter improvement, no overall regression >2.0, no summer/monsoon regression >3.0, no specific horizon group regression >5.0, no extreme event regression, and the model must maintain a >15.0 RMSE advantage over the naive baseline.

| Ablation | Description | F1 | F2 | F3 | F4 | Mean RMSE |
|----------|-------------|-----|-----|-----|-----|-----------|
| ABL-000 | Baseline (113 predictors) | 104.23 | 72.43 | 71.88 | 85.21 | 83.44 |
| ABL-001 | + Thermal/Inversion Proxy | 104.38 | 72.29 | 71.81 | 85.20 | 83.42 |
| ABL-002 | + Fog/Mist Indicator | 104.30 | 72.40 | 71.84 | 84.94 | 83.37 |
| ABL-003 | + Stagnation Enhancement | 104.24 | 72.78 | 72.24 | 85.37 | 83.66 |
| ABL-004 | + Seasonal Emission Proxy | 103.86 | 72.40 | 71.75 | 85.31 | 83.33 |
| ABL-005 | + All 4 Families | 104.28 | 72.55 | 72.00 | 85.16 | 83.50 |

**Conclusion**: The full spread of mean RMSE across all ablations was barely 0.33 points (83.33 to 83.66), with a maximum observed improvement of just 0.11 RMSE points. Consequently, none of the ablation features were adopted, and the original EXP-019 feature set was retained. This yielded a key scientific finding: simple surface-level meteorological proxies are near their predictive limit for this specific problem; they fundamentally cannot represent vertical atmospheric phenomena (like Planetary Boundary Layer height or specific inversion lapse rates) or real-time stochastic emission shocks (like agricultural fire counts).
(See `15_winter_smog_ablation.ipynb`).

## 11. Inference Pipeline (Phase 12)

The inference architecture was designed for robustness and safety in production.
*   **ModelLoader**: Provides thread-safe, lazy instantiation of the model, the feature scaler (which strictly validates `n_features_in_ == 114`), and the schema specification (which ensures exactly 114 features are presented in the correct order).
*   **AQIPostProcessor**: Handles translation of raw AQI predictions into EPA categories, hex colors, and severity flags (e.g., >200 triggers 'high', >300 triggers 'hazardous'). It also calculates the empirical prediction error intervals.
    *   **Empirical Intervals**: These bands are derived from 8,636 out-of-fold residuals calculated during the walk-forward validation process. For each horizon *h*, the lower bound is `L_h = max(0, aqi + Q_0.10(e_h))` and the upper bound is `U_h = max(0, aqi + Q_0.90(e_h))`. I must stress: these are empirical residual quantile ranges, NOT parametric statistical confidence intervals.
*   **PredictionCache**: Implemented a file-based JSON caching layer with a 1-hour TTL to prevent redundant computation.
*   **AQIPredictor**: Orchestrates the process. Crucially, it manages timestamp semantics:
    *   `input_observed_at`: The physical time the data was measured.
    *   `forecast_origin`: Anchored strictly to `input_observed_at`.
    *   `generated_at`: The system time the API processed the request.
    *   It implements staleness detection (flagging inputs older than 3 hours) and deliberately does *not* artificially clip upper-bound predictions at 500, preserving the model's unconstrained view of extreme events.

## 12. REST API (Phase 13)

The serving layer is a Flask application structured using the Blueprint pattern. It exposes five primary endpoints:
*   `/api/health`: Service liveness verification.
*   `/api/current`: Retrieves the latest observed state.
*   `/api/forecast`: Generates the 72-hour prediction array.
*   `/api/model/info`: Exposes metadata about the currently loaded champion model.
*   `/api/explain`: Provides SHAP-based attribution arrays.

CORS was configured to accept specific origins, defaulting to `localhost:8501` for dashboard integration. I implemented robust HTTP error handling (400, 404, 503, 500). A critical security requirement was ensuring 500-level errors returned sanitized JSON payloads that never leaked internal filesystem paths or Python tracebacks to the client.

## 13. Dashboard (Phase 14)

The user interface was constructed using Streamlit and Plotly. Key components include:
*   A data freshness banner warning of stale ingestion.
*   A "Current Observation" summary card.
*   An interactive 72-hour forecast chart displaying the primary prediction line alongside the empirical error bands. These bands are explicitly labeled "Empirical prediction error interval"—they are never misrepresented as "confidence intervals."
*   Milestone cards detailing specific threshold crossings.
*   A SHAP attribution explorer for model transparency.
*   Under the hood, the `DashboardDataClient` communicates with the Flask API but implements an automatic local inference fallback mechanism if the REST service is unresponsive.

## 14. CI/CD and Automation (Phase 15)

This phase directly leveraged the preparation work completed in the first week. I established three distinct GitHub Actions workflows to govern the repository:

1.  **Continuous Integration (`ci.yml`)**: Triggered on push or PR to the `main` branch. It sets up Python 3.10, executes the `pytest` suite enforcing `--cov-fail-under=70`, and uploads both JUnit XML results and coverage reports as workflow artifacts with a 14-day retention policy.
2.  **Feature Pipeline (`feature_pipeline.yml`)**: A cron-driven job scheduled at `17 * * * *` (the 17th minute of every hour). It executes `src.feature_pipeline.run_hourly_ingestion`. It supports a `dry_run` input parameter for manual testing and uploads a telemetry snapshot artifact retained for 7 days.
3.  **Training Pipeline (`training_pipeline.yml`)**: A weekly cron job executing at `23 2 * * 0` (Sundays at 02:23 UTC). It orchestrates candidate evaluation. Crucially, it enforces production model immutability by validating the SHA256 hash of the production model before and after execution to ensure silent overwrites cannot occur.

Because GitHub Actions runners are ephemeral, pipeline outputs and logs are explicitly saved as GitHub Actions artifacts. The fundamental understanding of runner lifecycles and secret management gained from the Discord resources made implementing these workflows straightforward.

## 15. Hopsworks Integration (Phase 16)

To elevate the system architecture to MLOps standards, I integrated Hopsworks.
*   **Feature Store**: Designed a 115-column cloud storage schema where the primary key is `location_id`, coupled with an event time `dt`, and the 113 predictor columns.
*   **Online Serving**: Configured low-latency entity-key lookups by `location_id`.
*   **Inference Projection**: Ensured strict extraction routines that project the 115-column store data down to the exact ordered 114-column canonical schema required by the model.
*   **Model Registry**: Packaged the production assets into a versioned bundle containing a `manifest.json` with SHA256 hashes for integrity verification. The bundle includes `production_hybrid_model.joblib`, `feature_scaler_v2_weather.joblib`, `feature_schema_v2_weather.json`, and `empirical_error_intervals.json`.

## 16. Explainability (Phase 17)

To demystify model behavior, I implemented SHAP (SHapley Additive exPlanations) attribution mapped directly onto the exact architecture of the `PersistenceAwareHybridModel`. The implementation uses three distinct routing zones:
1.  **h=1..6**: Evaluates individual LightGBM estimators using `TreeExplainer` via interventional perturbation.
2.  **h=7..37**: Computes Interventional Linear SHAP on the Ridge weights: `phi_i = coef_i * (x_scaled_i - mean_bg_i)`.
3.  **h=38..72**: Executes the exact mathematical blended decomposition: `base_value = w_h * b_specialist`, `phi_final = w_h * phi_unweighted`, and `persistence_component = (1-w_h) * y_current`.

Global feature importance was defined as `I_i = (1/72) * sum over h of mean|phi_final_i,h|`, evaluated rigorously on a 500-sample seasonal stratified cohort. The SHAP background reference distribution utilized a separate, independent 100-sample set.
I built in mathematical additivity validation to verify: `|sum(base + phis + persistence) - explained_output_preclip| < 1e-6`.
The API reports attributions in scaled z-space (providing both raw and scaled values) and strictly uses labels like "increases_prediction" or "decreases_prediction" to avoid implying causality.

## 17. Testing

The codebase is protected by 346 individual tests spanning 27 test modules and a central `conftest.py`. Overall test coverage is 75%, but coverage for critical path inference modules runs between 88% and 100%. The CI pipeline enforces a strict 70% minimum coverage gate. The test suite comprehensively covers the API layer, model internals, data ingestion logic, feature engineering routines, training pipelines, inference paths, dashboard logic, workflow scripts, and the SHAP explainability engine.

## 18. Challenges and Debugging

Building this system involved significant debugging and navigating analytical challenges:

*   **Model Complexity Paradox**: My initial assumption was that a complex model would easily outperform simpler methods. This was proven false rapidly. The Random Forest achieved a dismal RMSE of 89.18, losing outright to a naive baseline. Even the TensorFlow DNN (RMSE 85.01) could not match a basic Ridge Regression (RMSE 82.97). This reinforced the reality that regularized linear models are exceptionally powerful for tabular time-series data with relatively limited sample sizes.
*   **Temporal Distribution Shift**: The composition of the training data differed markedly from the test data due to prevailing seasonal factors. Diagnosing this shift required stepping away from aggregate metrics and analyzing segment-specific performance, ultimately revealing that errors were heavily concentrated in the winter smog periods.
*   **The Winter/Smog Crucible**: The walk-forward validation laid bare the difficulty of predicting extreme events. Fold 1 (Winter 2021) showed an RMSE of 104, while Summer folds sat comfortably at 71. The physical mechanics of smog formulation during winter inversions are simply harder to forecast using only surface-level data.
*   **Feature Schema Misalignment**: I spent considerable time debugging the feature count definitions. The canonical schema is 114 features, which includes the `dt` (datetime) column. However, during model training and walk-forward validation, `dt` is strictly excluded as a predictor, leaving 113 functional features. Reconciling this definition across the inference pipeline, scaling transforms, and Hopsworks integration required careful refactoring.
*   **Timestamp Semantics**: In early API iterations, I mistakenly anchored the `forecast_origin` to the time the API request was received (`generated_at`). This caused misalignment; the forecast origin must strictly match the timestamp of the last actual physical observation (`input_observed_at`).
*   **Serialization State**: A subtle bug occurred in the Hybrid model loading. The constructor sets a default `min_blend_weight`. During serialization, the specific learned weight is saved. However, I realized the `load()` method had to explicitly extract this serialized `min_blend_weight` and dynamically reconstruct the full 35-step `blend_weights` array, rather than relying on the constructor default.
*   **Dependency Management Hell**: Integrating SHAP proved surprisingly difficult due to underlying Python version constraints. SHAP version 0.50+ requires Python 3.11+, and 0.52+ requires Python 3.12+. Because my environment was Python 3.10, I had to identify and strictly pin the dependency to version `0.49.1` to maintain compatibility.
*   **CI Ephemerality**: When initially writing the GitHub Actions workflows, I assumed files generated during a job step would persist indefinitely. Debugging failing pipelines taught me about ephemeral runners—if a report or model artifact is generated, it will vanish the moment the job concludes unless explicitly captured and uploaded using the `actions/upload-artifact` action.

## 19. Design Decisions

Throughout the project, several core architectural choices defined the system's character:
*   **Targeting EPA AQI**: Rather than just predicting raw ug/m3 concentrations, targeting the synthetic EPA AQI scale provided immediate, universally understood public health context.
*   **72-Hour Horizon**: A 3-day forecast window was chosen as the optimal balance between predictive feasibility and providing enough lead time for municipal or personal health planning.
*   **Chronological Split with Embargo**: Traditional randomized train/test splits destroy time-series integrity. The strict chronological split, coupled with a 72-hour embargo gap matching the forecast horizon, aggressively prevented temporal information leakage.
*   **Ridge Baseline Superiority**: The selection of Ridge (alpha=1.0) was not arbitrary; it was derived from the systematic parameter grid sweeps executed in experiments EXP-001 through EXP-009.
*   **Persistence Blending Architecture**: The decision to blend the outer horizons (h=38..72) back toward the current observation was driven by the reality of compounding uncertainty. In the absence of high-confidence predictions three days out, the most recently observed value acts as the safest, most stable anchor point.
*   **Empirical Error vs Parametric CI**: I explicitly chose to report empirical residual ranges calculated from walk-forward holdout data rather than calculating standard parametric confidence intervals. Machine learning error distributions on environmental data are rarely Gaussian. Empirical residuals provide a more honest, battle-tested reflection of actual model behavior, and preventing the UI from labeling them "confidence intervals" maintains that honesty.
*   **Rejecting Ablation Features**: The decision not to adopt the winter-specific features (despite a trivial 0.11 RMSE improvement) was rooted in software engineering pragmatism. The added code complexity, data dependencies, and testing surface area were not justified by a marginal statistical gain.
*   **Observation-Anchored Forecating**: Tying the `forecast_origin` to the observation timestamp rather than system time ensures that forecasts are physically meaningful, regardless of network latency or API processing delays.

## 20. Lessons Learned

Executing this project end-to-end offered profound personal and professional insights. First and foremost, the critical importance of robust baselines cannot be overstated. Watching my complex TensorFlow network and Random Forest ensemble lose outright to a simple Ridge regression (Ridge v1) was a humbling and clarifying moment. It taught me that complexity is not a proxy for capability. 

I also learned that data engineering often yields far higher returns than algorithm selection. The single largest leap in system performance came not from model tuning, but from integrating meteorology. Moving from 64 to 114 features by pulling Open-Meteo data drove the RMSE down from 82.97 to 78.38—a far larger gain than any architecture tweak provided.

The implementation of the walk-forward validation framework was a watershed moment. It revealed that a single chronological train/test split can mask regime-specific failures. By splitting the evaluation temporally, I could clearly see that summer predictions were reliable, while winter smog episodes were severely stretching the model's capabilities. It showed me the limits of the data; the ablation study demonstrated that no amount of clever feature engineering on surface-level metrics could fully compensate for the lack of vertical atmospheric data.

On the software engineering side, the upfront time spent on the Discord resources reviewing CI/CD pipelines paid off immensely. When I reached Phase 15, establishing the automation workflows was relatively frictionless because I understood the core mechanics of GitHub Actions. 

Finally, I learned the value of operational discipline. Building a system that strictly freezes production artifacts, rigorously tests new candidates against held-out baselines, and reports empirical—rather than theoretical—errors fundamentally shifted my perspective from building ML scripts to engineering reliable ML systems.

## 21. Future Work

Several avenues exist for significant future enhancement:
*   **AQICN Integration (Phase 18)**: Expanding the ingestion layer to pull multi-source data from the AQICN network to improve spatial robustness.
*   **Vertical Atmospheric Profiling**: The ablation study strongly suggested that surface data is insufficient for severe winter inversion modeling. Incorporating planetary boundary layer (PBL) height and specific temperature lapse rates is a high priority.
*   **Real-time Emission Dynamics**: Integrating active fire count data from satellite telemetry could provide crucial early warning signals for agricultural burning impacts.
*   **Expanded Winter Histories**: The model simply needs to observe more winter smog seasons to learn the underlying dynamics better. Time will organically solve this data starvation issue.
*   **Automated Champion Promotion**: Maturing the CI/CD pipeline to include fully automated, statistically gated model promotion rather than relying on manual registry updates.

## 22. Complete List of Notebooks

1.  `01_api_investigation.ipynb` — OpenWeather API validation & rate limits
2.  `02_raw_data_exploration.ipynb` — Historical data ingestion validation
3.  `03_eda_and_aqi_conversion.ipynb` — EDA and EPA AQI conversion
4.  `04_feature_engineering.ipynb` — Temporal, lag, rolling features
5.  `05_dataset_preparation.ipynb` — Multi-output target alignment & chronological split
6.  `06_model_training_ridge.ipynb` — Ridge baseline training
7.  `07_model_training_rf.ipynb` — Random Forest training
8.  `08_model_training_tf.ipynb` — TensorFlow DNN training
9.  `09_error_and_distribution_diagnostics.ipynb` — Distribution shift & error analysis
10. `10_weather_enrichment.ipynb` — Open-Meteo feature extraction
11. `11_systematic_model_tuning.ipynb` — Cross-validation & experiment tracking
12. `12_forecasting_architecture_experiments.ipynb` — Hybrid architectures
13. `13_final_test_benchmark.ipynb` — 7-model benchmark on held-out test
14. `14_walk_forward_stability.ipynb` — 4-fold walk-forward validation
15. `15_winter_smog_ablation.ipynb` — Winter/smog feature ablation
