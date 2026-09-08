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

While the source class constructor sets a default `min_blend_weight=0.6`, the frozen EXP-019 production champion was validated and serialized with `min_blend_weight=0.7` (with blend weights decaying from 1.0 at h38 to 0.7 at h72). To ensure exact reproducibility across environments, the serialized production artifact directly stores its fitted `min_blend_weight=0.7` attribute. The class's `load()` method relies on this serialized value to dynamically reconstruct the precise `blend_weights` array ($w_{38}=1.0 \to w_{72}=0.7$). The model accesses the current AQI via index 9 (`epa_aqi_lag_1h` in the canonical schema).

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

To elevate the system architecture to MLOps standards, I integrated Hopsworks as a cloud feature store and model registry layer.
*   **Feature Group Architecture**: Created and populated production feature group `aqi_weather_features_v2` (version 1, ID `52526`) on project `aqi_predictor_by_Waleed` (`https://eu-west.cloud.hopsworks.ai`). Configured `primary_key=["location_id"]`, `event_time="dt"`, `online_enabled=True`, and `time_travel_format="HUDI"`.
*   **Offline Storage**: Ingested 48,716 historical hourly observations (spanning 2020-11-28 13:00:00 UTC through 2026-09-07 15:00:00 UTC for Lahore) across 115 columns (`location_id` + 114 canonical features). Idempotency is verified: re-inserting historical records maintains row count without inflation.
*   **Scheduled Hourly Live Feature Pipeline**: Configured to run automatically every hour via GitHub Actions (`.github/workflows/feature_pipeline.yml`, cron schedule `17 * * * *`). The exact live production workflow path has been successfully verified through `workflow_dispatch` in workflow run [34152458548](https://github.com/hwaleedkhalid/aqi-forecasting-system/actions/runs/34152458548), completing in 43 seconds.
    *   *Mathematical Lookback Necessity*: EXP-019 requires 114 canonical model inputs: 25 pollutant lags, 20 weather lags, 34 rolling aggregates, 6 differentials, 5 chemical interaction ratios, 7 cyclical/calendar signals, 9 base pollutants + AQI, 7 base meteorology, and 1 `dt`. A single instantaneous API reading at time $t$ cannot construct 24-hour backward lag and rolling windows. The pipeline therefore mandates a continuous $[T-24\text{h}, T]$ historical lookback context.
    *   *Dual Telemetry Providers*: Combines OpenWeather Air Pollution History API (`fetch_historical_air_quality` retrieving 72 hours of hourly criteria pollutants: CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃) with Open-Meteo Weather API (`past_days=3, forecast_days=1` retrieving hourly temperature, humidity, surface pressure, wind speed, wind direction, and precipitation).
    *   *Hourly Grid Continuity & Dropout Toleration*: Reindexes both series to an exact 1-hour UTC frequency grid. Adheres strictly to the established production preprocessing contract: short sensor dropouts $\le 3$ hours are linearly/time interpolated, while unbridgeable gaps $> 3$ hours or missing $T-24\text{h}$ boundaries fail closed with `ValidationError`.
    *   *Explicit Production Timestamp $T$*: Derived as $T = \min(T_{\text{AQ}}, T_{\text{weather}})$ floored to the hour boundary ($T \pmod{3600} == 0$), strictly $\le \text{now\_utc}$. Future forecast hours from Open-Meteo are discarded so future timestamps never become the observation origin.
    *   *Authoritative Current AQI*: Computed via standard US EPA piecewise linear interpolation (`calculate_overall_aqi`), cleanly distinguishing current observation $T$ (`epa_aqi=131`) from $T-1\text{h}$ lag (`epa_aqi_lag_1h=128`).
    *   *Dual-Store Ingestion Semantics & Asynchronous Materialization*: Historical event is ingested with streaming Kafka writes to online RonDB using server-side `upsert_if_newer=True` (synchronous, completing in ~2 seconds), while offline Hudi materialization is triggered asynchronously (`wait_for_job=False`), preventing GitHub Actions runner timeouts.
    *   *Automated Orchestration & Security*: Scheduled via GitHub Actions (`.github/workflows/feature_pipeline.yml`) on cron `17 * * * *` with encrypted repository secrets (`OPENWEATHER_API_KEY`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`), concurrency control (`cancel-in-progress: false`), least privilege permissions (`contents: read`), fail-fast secret validation, and structured snapshot export to `data/snapshots/latest_ingestion_report.json`. Dependency isolation is guaranteed via `requirements-feature-pipeline.txt` (`hopsworks[python]==5.0.6` with `confluent-kafka` and `pyarrow`), preserving the clean deployment footprint of Render and Streamlit.
    *   *Live Cloud State Verified*: Queried live Hopsworks RonDB online store, verifying latest observation $T = 1788804000$ (2026-09-07 18:00:00 UTC), EPA AQI 131.0, 114 features in exact canonical schema order, and `cloud_active: true`.
*   **Model Registry (Live Champion Registration)**: 
    *   *Rationale for Initial Decoupling*: The Model Registry connector was originally maintained in an integration-ready state to decouple the live production deployment on Render and Streamlit from external cloud availability, authentication latencies, and cold-start download overheads.
    *   *Live Registration*: Successfully registered the frozen EXP-019 production champion under the canonical identity `pearls_aqi_production_champion` (version 1, ID `pearls_aqi_production_champion_1`) in the live Hopsworks Model Registry (`aqi_predictor_by_Waleed`).
    *   *Bundle Integrity & Assets*: Packaged the complete runtime bundle comprising `production_hybrid_model.joblib` (SHA256 `f51d2eff53b8...`), `feature_scaler_v2_weather.joblib` (SHA256 `9ce7e9fcc4fe...`), `feature_schema_v2_weather.json` (SHA256 `38a5fdea89b0...`), `empirical_error_intervals.json` (SHA256 `dba4571ebe05...`), and companion explainability assets (`shap_background.npy`, `global_shap_importance.json`, `explainer_manifest.json`) alongside an LF-normalized `manifest.json`.
    *   *SDK & Schema Binding*: Constructed the 114-input / 72-output `ModelSchema` using the Hopsworks 5.0.6 `hsml` API and bound verified benchmarks explicitly distinguished by provenance: chronological out-of-time holdout test metrics (9,311 samples, 2025-06-07 to 2026-08-28 UTC: `overall_rmse: 75.91, overall_mae: 53.55, overall_r2: 0.4858, h1_rmse: 50.43, h72_rmse: 77.43`) alongside the 4-fold temporal walk-forward validation results (`mean_rmse: 83.44`, 4/4 fold wins vs persistence, 24.46% relative gain).
    *   *Idempotency & Fail-Closed Protection*: Engineered safe re-run semantics that download and verify checksum parity on existing version 1 before skipping redundant registration, preventing unwanted version inflation (v2, v3). Mismatched existing versions fail closed with `ValidationError`.
    *   *Clean-Download & Parity Verification*: Downloaded the registered champion into an isolated temporary environment, verified exact SHA256 equality, validated standalone loadability, and proved complete mathematical prediction parity (0.00000000 maximum absolute difference across all 72 horizons, with boundary check parity on h1, h6, h7, h24, h37, h38, h39, h72) and explainability parity on h1, h24, and h72 with exact numerical alignment between predictor and explainer (e.g. h72 unrounded 158.9892 rounding to 159.0 for both).
    *   *Production Boundary*: Runtime deployment on Render and Streamlit continues to execute self-contained Git runtime assets, guaranteeing zero latency regressions while fulfilling full Model Registry reproducibility via `python -m src.inference.hopsworks_registry --all`.

## 16. Explainability (Phase 17)

To demystify model behavior, I implemented SHAP (SHapley Additive exPlanations) attribution mapped directly onto the exact architecture of the `PersistenceAwareHybridModel`. The implementation uses three distinct routing zones:
1.  **h=1..6**: Evaluates individual LightGBM estimators using `TreeExplainer` via interventional perturbation.
2.  **h=7..37**: Computes Interventional Linear SHAP on the Ridge weights: `phi_i = coef_i * (x_scaled_i - mean_bg_i)`.
3.  **h=38..72**: Executes the exact mathematical blended decomposition: `base_value = w_h * b_specialist`, `phi_final = w_h * phi_unweighted`, and `persistence_component = (1-w_h) * y_current`.

Global feature importance was defined as `I_i = (1/72) * sum over h of mean|phi_final_i,h|`, evaluated rigorously on a 500-sample seasonal stratified cohort. The SHAP background reference distribution utilized a separate, independent 100-sample set.
I built in mathematical additivity validation to verify: `|sum(base + phis + persistence) - explained_output_preclip| < 1e-6`.
The API reports attributions in scaled z-space (providing both raw and scaled values) and strictly uses labels like "increases_prediction" or "decreases_prediction" to avoid implying causality.

## 17. Testing

The codebase is protected by 389 individual automated tests spanning 30 test modules and a central `conftest.py`. Overall test coverage is 73%, with critical-path inference, model loading, and explainability modules achieving between 88% and 100% coverage. The CI pipeline enforces a strict 70% minimum coverage gate. The test suite comprehensively covers the API layer, model internals, data ingestion logic, feature engineering routines, training pipelines, inference paths, runtime asset integrity resolution, clean-clone bootstrap parity, dashboard logic, workflow scripts, Hopsworks integration and backfill audit rules, Model Registry bundle integrity and idempotency, the scheduled hourly live feature pipeline (with grid continuity and out-of-order protection verification), and the SHAP explainability engine. All tests run in isolated environments with external cloud dependencies 100% mocked, ensuring CI remains fast, reliable, and secret-independent.


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
*   **Streaming Ingestion and Materialization Latency in CI**: When automating the live feature pipeline in GitHub Actions, Hopsworks feature groups with `online_enabled=True` operate as streaming Kafka feature groups requiring `confluent-kafka` and `pyarrow` (provided via `hopsworks[python]==5.0.6`). Furthermore, default synchronous insert calls block polling for cloud Spark materialization jobs, which can take 10-15 minutes or timeout. Setting `wait_for_job=False` / `wait=False` for offline storage ensures the Kafka online write completes synchronously in ~2 seconds while offline Hudi materialization runs asynchronously in the cloud, bringing total GitHub Actions workflow duration down to 43 seconds.

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

Executing this project end-to-end offered profound personal and professional insights. In the initial phase, I did not immediately begin implementation; I first reviewed the preparatory resources shared in the course Discord server. This dedicated review refreshed my understanding of Python project structuring and machine learning fundamentals, while solidifying strict Git and GitHub version control practices. Furthermore, GitHub Actions and CI/CD concepts were relatively new to me; studying workflow triggers, runner lifecycles, secrets management, and artifact persistence proved directly beneficial later in Phase 15 and during production deployment.

First and foremost, the critical importance of robust baselines cannot be overstated. Watching my complex TensorFlow network and Random Forest ensemble lose outright to a simple Ridge regression (Ridge v1) was a humbling and clarifying moment. It taught me that complexity is not a proxy for capability.

I also learned that data engineering often yields far higher returns than algorithm selection. The single largest leap in system performance came not from model tuning, but from integrating meteorology. Moving from 64 to 114 features by pulling Open-Meteo data drove the RMSE down from 82.97 to 78.38—a far larger gain than any architecture tweak provided.

The implementation of the walk-forward validation framework was a watershed moment. It revealed that a single chronological train/test split can mask regime-specific failures. By splitting the evaluation temporally, I could clearly see that summer predictions were reliable, while winter smog episodes were severely stretching the model's capabilities. It showed me the limits of the data; the ablation study demonstrated that no amount of clever feature engineering on surface-level metrics could fully compensate for the lack of vertical atmospheric data.

Finally, navigating the deployment journey taught me the value of operational discipline. Moving from local notebooks to a deployed system required rigorous dependency isolation, runtime artifact whitelisting, fail-closed integrity validation, continuous memory profiling, and cross-platform consistency. Building a system that strictly freezes production artifacts, rigorously tests candidates against held-out baselines, and reports empirical—rather than theoretical—errors fundamentally shifted my perspective from building standalone ML scripts to engineering reliable, production-grade ML systems.

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

## 23. Deployment and Production Hardening

### 23.1 Deployment Architecture

The final deployment architecture enforces a strict physical separation between user interface rendering and backend model inference:

```text
GitHub Repository
       │
       ├── Streamlit Community Cloud
       │       └── Streamlit Frontend (src/dashboard/app.py)
       │                │
       │                │ HTTPS REST API Requests
       │                ▼
       └── Render (Free Web Service)
               └── Flask REST API (src/api/app.py)
                    ├── RuntimeAssetResolver
                    ├── EXP-019 Production Hybrid Model
                    ├── Empirical Prediction Error Intervals
                    ├── SHAP Explainability Engine
                    └── Bootstrap Feature Vector
```

This decoupled topology provides significant operational advantages:
1.  **Separation of Concerns**: Presentation logic in Streamlit remains completely decoupled from model execution and data ingestion.
2.  **Resource and Dependency Optimization**: The Streamlit frontend installs only 41 lightweight UI packages (~40 MB footprint) from `src/dashboard/requirements.txt`, avoiding heavy machine learning frameworks on the frontend container.
3.  **Credential and Data Isolation**: The Flask API manages all internal runtime assets, schemas, and credentials on the backend server, exposing only validated JSON contracts.
4.  **Independent Lifecycle**: The frontend and backend deploy and scale independently. If the frontend restarts, backend inference caches remain intact; if the backend sleeps on standby, the frontend cleanly presents service status notices.

The system is deployed using **Streamlit Community Cloud** for the dashboard and **Render** for the Flask API, with Railway identified as a viable secondary backend alternative if additional memory or CPU resources become necessary.

### 23.2 Deployment Problems and Investigation

Transitioning from local development to cloud hosting revealed several critical architectural hurdles:

#### 1. Ignored Runtime Artifacts and Clean-Clone Divergence
In local development, the model loader and feature pipeline read from `data/models/` and `data/processed/`. However, standard `.gitignore` rules correctly exclude these directories to prevent committing multi-megabyte training caches and raw datasets to Git. On a clean Git clone, the backend initially crashed because the model files did not exist.
*   **Investigation**: I recognized that cloud deployment platforms (Render, Railway) build directly from clean Git checkouts. They have no access to untracked local developer directories.
*   **Solution**: I created a dedicated, versioned `data/runtime/` package containing only the frozen champion model, scaler, schema, empirical error intervals, explainability reference matrices, and bootstrap vector. I updated `.gitignore` with explicit whitelist rules (`!data/runtime/`, `!data/runtime/**`) while keeping large training datasets excluded.

#### 2. Hopsworks Cloud Audit and Architectural Realignment
Phase 16 implemented Hopsworks integration using local mock tests. When auditing the live cloud environment, I encountered DNS deprecation issues with the legacy `c.app.hopsworks.ai` endpoint in Hopsworks 3.4.0. Testing with the modern Hopsworks 5.0.6 client successfully authenticated against the cloud project `aqi_predictor_by_Waleed` (ID: 44159) via `https://eu-west.cloud.hopsworks.ai:443`.
*   **Findings**: The initial live cloud Feature Store contained zero feature groups and zero registered models.
*   **Resolution**: To prevent unpopulated remote infrastructure from blocking deployment, I established the self-contained `data/runtime/` package as the authoritative production source. Subsequently, the live Feature Store was successfully populated with feature group `aqi_weather_features_v2` (version 1), ingesting all 48,715 historical observations into offline storage and synchronizing the latest observation into online storage with verified 114-feature canonical parity. The deployed Streamlit and Render services continue to operate independently from self-contained runtime assets.

#### 3. Bootstrap Feature Vector and Ingestion Hierarchy
A naive assumption was that the backend could reconstruct input features on the fly by querying current OpenWeather observations.
*   **Investigation**: EXP-019 requires 114 engineered features, including 24-hour pollutant lags, rolling statistics, cross-pollutant chemical ratios, and multi-hour weather differentials. A single point-in-time API response lacks the 24+ hours of unbroken historical context required to compute these features.
*   **Solution**: I established a committed canonical bootstrap feature vector (`data/runtime/bootstrap/latest_feature_vector.json`) representing a validated 114-column observation row. The runtime resolution hierarchy prioritizes: (1) verified fresh feature pipeline outputs, (2) Hopsworks Feature Store vectors when populated, and (3) committed bootstrap vectors.

#### 4. Data Freshness and Observation Anchoring
To avoid misleading users, the application implements strict data freshness transparency:
*   The committed bootstrap vector is timestamped `2026-08-31T07:00:00+00:00`.
*   The API calculates input age (`now - input_observed_at`) and marks `is_stale=true` whenever age exceeds 3.0 hours.
*   The Streamlit dashboard prominently renders a warning banner explaining that telemetry is historical.
*   Crucially, forecast timestamps (h1 through h72) are anchored to `forecast_origin = input_observed_at`, spanning `2026-08-31T08:00:00+00:00` to `2026-09-03T07:00:00+00:00`, rather than shifting dynamically with the client request time.

#### 5. Frontend Dependency Leakage
Initial frontend code imported `from src.inference.predictor import AQIPredictor` at the top of `src/dashboard/data_client.py`. When Streamlit Cloud attempted to build the dashboard with a lightweight dependency list, it failed because `AQIPredictor` pulled in `scikit-learn`, `lightgbm`, and `shap`.
*   **Solution**: I refactored `DashboardDataClient` to remove top-level inference imports, configured `ENABLE_LOCAL_FALLBACK=false` for cloud deployment, lazy-imported `AQIPredictor` strictly inside local fallback branches, and created `src/dashboard/requirements.txt` containing only UI packages.

#### 6. Render Memory Profiling and Concurrency Sizing
Render's free tier provides 512 MB RAM and 0.1 CPU. I conducted live process memory profiling across all endpoint states using a continuous high-frequency background RSS sampler (5ms interval):
*   **Flask Startup & EXP-019 Load**: 176.95 MB RSS
*   **72-Hour Forecast Execution**: 178.62 MB to 185.08 MB RSS
*   **SHAP Multi-Horizon Explainability Peak**: 360.81 MB transient peak RSS (settling back to 190.06 MB after garbage collection)
*   **Render Free Allocation**: 512.00 MB
*   **Available Headroom at Transient Peak**: 151.19 MB (29.5% free headroom)

*Analysis*: The backend fits comfortably within Render's 512 MB allocation for a single worker. However, because SHAP creates transient allocation matrices during multi-horizon tree and linear evaluation, running multiple Gunicorn workers would duplicate memory and risk out-of-memory termination. While the deployment was configured with `gunicorn src.api.app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`, memory profiling confirms that a single worker (`--workers 1`) is the conservative and stable operational model.

#### 7. Build Environment and Dependency Resolution
During initial deployment on Render, two build issues occurred:
*   **Default Python Version**: Render defaulted to Python 3.14, which lacked pre-compiled binary wheels for machine learning libraries. I pinned Python `3.10.14` via a root `.python-version` file.
*   **Pip Backtracking**: Including offline baseline tools (`tensorflow>=2.13`) and legacy client constraints (`hopsworks<4.0`) in the root `requirements.txt` caused pip to backtrack across 10 years of C++ source tarballs (`grpcio`, `google-pasta`), causing build timeouts. I separated runtime dependencies into `requirements.txt` (installing in <30 seconds) and developer dependencies into `requirements-dev.txt`.

#### 8. Cross-Platform Line-Ending Hashes
On Windows, Git checked out JSON files with CRLF (`\r\n`), whereas Linux (Render) checked them out with LF (`\n`). This caused SHA256 checksum mismatches on text configuration files during boot.
*   **Solution**: I added `.gitattributes` enforcing `*.json text eol=lf` and updated `compute_file_sha256` in `src/inference/runtime_resolver.py` to normalize JSON text to LF before hashing.

#### 9. Clean CI Runner Directory Structure and Offline Mocking
When running the full test suite in ephemeral GitHub Actions runners, several setup tests failed because Git ignores empty directories (`data/raw/`, `data/processed/`, `data/models/`). Furthermore, clean-clone test jobs lack external API credentials and local training CSVs.
*   **Solution**: I introduced tracked `.gitkeep` markers across all data subdirectories (`!data/**/.gitkeep` in `.gitignore`), committed the canonical `feature_schema_v2_weather.json` to Git, and wired fallback resolution to committed bootstrap vectors. All external cloud and API endpoints remain strictly mocked in test suites, allowing CI workflow runs to complete 100% green (389 passed tests, 72.43% coverage) without requiring secrets or network access.

### 23.3 Runtime Integrity and Validation

The runtime layer implements a fail-closed integrity system managed by `RuntimeAssetResolver`:
*   **Cryptographic SHA256 Checksums**: Validates production artifacts against `manifest.json` on boot. Any tampered model file or corrupted byte triggers an immediate `ValidationError` and returns HTTP 503.
*   **Model-Explainer Binding**: `explainer_manifest.json` binds the SHA256 hashes of the model, schema, and background matrix to ensure explainability artifacts cannot be used with mismatched models.
*   **Bootstrap Schema Binding**: `latest_feature_vector.json` encodes `schema_sha256`, which must match the active production schema.
*   **Finite Numeric Enforcement**: Validates that all 114 features are present in canonical order and rejects non-finite numeric entries (`NaN`, `Infinity`, `-Infinity`).
*   **Timestamp Parsing**: Strictly validates ISO 8601 UTC timestamps.

These checks serve as robust internal corruption-detection mechanisms to guarantee runtime stability.

### 23.4 Clean-Clone Verification

To prove production readiness, I conducted a genuine clean-clone test in an isolated temporary directory:
1.  Cloned the repository into a fresh directory completely devoid of local files.
2.  Verified that `.env`, `data/models/`, `data/processed/`, `data/raw/`, and `data/logs/` were completely absent.
3.  Booted Flask using only committed `data/runtime/` assets.
4.  Executed endpoint verification:
    *   `GET /api/health` -> HTTP 200 OK (`service_ready: true`, `model_id: "EXP-019"`).
    *   `GET /api/current` -> HTTP 200 OK (`current_aqi: 100.0`, `is_stale: true`, exact parity with historical August 31 row).
    *   `GET /api/forecast` -> HTTP 200 OK (72 horizons, Origin `2026-08-31T07:00:00+00:00`, H1 `2026-08-31T08:00:00+00:00`, H72 `2026-09-03T07:00:00+00:00`).
    *   `GET /api/model/info` -> HTTP 200 OK (`EXP-019`, 114 features, R² 0.4858).
    *   `GET /api/explain?horizon=24` -> HTTP 200 OK (Predicted AQI `154.15`, matching forecast H24; additivity error `< 1e-6`).


### 23.5 CORS and Frontend/API Communication

The Streamlit dashboard communicates with the Flask REST API via server-side Python `requests`. Because the HTTP requests originate from the Streamlit Cloud server rather than the end-user's browser, browser-enforced Cross-Origin Resource Sharing (CORS) restrictions do not apply to the primary dashboard data path. Nevertheless, CORS middleware is configured on the Flask API (`CORS_ORIGINS=*`) to support future browser-based single-page applications or third-party client integrations.

### 23.6 Final Live Deployment

The system is publicly deployed and operational:
*   **Live Frontend**: [https://aqi-forecasting.streamlit.app](https://aqi-forecasting.streamlit.app) (Streamlit Community Cloud)
*   **Live Backend REST API**: [https://aqi-forecasting-xyyb.onrender.com/api](https://aqi-forecasting-xyyb.onrender.com/api) (Render Free Web Service)

*(Note: These represent the verified deployment endpoints at final project testing time. Third-party hosting availability is not permanently guaranteed.)*

**Verified User-Facing Capabilities**:
*   **Telemetry Overview**: Displays bootstrap baseline AQI (100.0, Moderate) with a prominent stale data warning banner.
*   **72-Hour Forecast Trajectory**: Interactive Plotly curve displaying predictions anchored to `2026-08-31T07:00:00+00:00`.
*   **Empirical Prediction Error Intervals**: Shaded residual bands reflecting walk-forward empirical uncertainty (10th to 90th percentiles).
*   **Multi-Horizon Milestone Cards**: Summaries for key milestones (+1h, +12h, +24h, +48h, +72h).
*   **SHAP Feature Attribution Explorer**: Interactive slider allowing users to inspect feature impact across any horizon from 1 to 72.
*   **Model Provenance**: Sidebar detailing architecture, training span, test benchmarks, and active REST API source mode.

### 23.7 Deployment Challenges Summary

| Challenge | Cause | Investigation | Solution | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Ignored Runtime Artifacts** | `data/models/` and `data/processed/` excluded by `.gitignore`. | Clean Git clone on cloud servers failed due to missing model files. | Created self-contained `data/runtime/` package and added explicit `.gitignore` whitelist rules. | Clean clones boot deterministically without untracked files. |
| **Hopsworks Cloud Dependency** | Remote Hopsworks instance contained 0 feature groups and 0 models. | Live cloud audit revealed Phase 16 was developed using integration mocks. | Made `data/runtime/` the primary source of truth; preserved Hopsworks as optional integration. | Deployment operates independently of remote feature store status. |
| **Feature Reconstruction on Boot** | EXP-019 requires 114 engineered lag and rolling features. | Single OpenWeather API call cannot reconstruct 24+ hours of historical context. | Created canonical bootstrap feature vector (`latest_feature_vector.json`). | Cold-start boot reliably generates valid 72-hour forecasts. |
| **Frontend Dependency Bloat** | `data_client.py` imported `AQIPredictor` at module scope. | Streamlit Cloud attempted to install heavy ML packages (`shap`, `lightgbm`). | Lazy-imported predictor inside fallback branch, set `ENABLE_LOCAL_FALLBACK=false`, and created `src/dashboard/requirements.txt`. | Frontend container installs in 1.6s with only 41 lightweight UI packages. |
| **Render Memory Constraints** | Free tier allocates 512 MB RAM. | Continuous 5ms sampling revealed transient SHAP attribution peak of 360.81 MB. | Restricted Gunicorn concurrency to a single application worker (`--workers 1`). | API runs stably with 151.19 MB (29.5%) headroom during peak calculation. |
| **Build Dependency Conflicts** | `tensorflow` and legacy `hopsworks<4.0` in root `requirements.txt`. | Pip backtracked across 10 years of C++ source tarballs, causing 19+ min build timeouts. | Separated runtime requirements into `requirements.txt` and dev tools into `requirements-dev.txt`. | Render build completed in under 45 seconds using pre-compiled wheels. |
| **WSGI Start Command Syntax** | Unquoted parentheses `create_app()` in Render start command. | Linux bash parsed parentheses as shell subshell operators, exiting with status 2. | Exported module-level `app = create_app()` in `src/api/app.py` and updated start command to `src.api.app:app`. | Gunicorn boots cleanly on Render startup. |
| **Cross-Platform Checksum Mismatches** | Windows CRLF (`\r\n`) vs Linux LF (`\n`) line endings in JSON text files. | Hash of `feature_schema_v2_weather.json` diverged between development and Render Linux. | Added `.gitattributes` enforcing `eol=lf` and updated resolver to normalize JSON text before hashing. | SHA256 checksums match identically across Windows and Linux. |

### 23.8 Final Project Outcome

The Pearls AQI Predictor project is fully implemented, thoroughly tested, and publicly deployed. The application is reproducibly bootable from a clean Git clone, hosted across Streamlit Community Cloud and Render, and accessible through an interactive web dashboard backed by a high-speed Flask REST API. Operating on the frozen champion model EXP-019, the system delivers 72 continuous hourly predictions with empirical prediction error intervals, multi-horizon SHAP feature attributions, and transparent data freshness indicators.

