"""Tests for Phase 11.5: Winter/Smog Candidate Feature Families.

Coverage:
  - Feature construction correctness for all 4 families
  - Backward-looking constraint (no future information)
  - Proxy naming: 'inversion_risk_proxy' not 'inversion_risk_score'
  - Absence of 'winter_month_index' (removed per user review)
  - crop_burning_season calendar logic
  - fog_proxy threshold conditions
  - Adoption gate logic (criterion 1: mean winter improvement)
  - Ablation matrix: exactly 6 configs (ABL-000 to ABL-005)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline.winter_smog_features import (
    ALL_WINTER_FEATURE_COLS,
    CROP_BURNING_END_MONTH,
    CROP_BURNING_START_DAY,
    CROP_BURNING_START_MONTH,
    FAMILY_NAMES,
    FOG_HUMIDITY_THRESHOLD,
    FOG_TEMPERATURE_THRESHOLD,
    FOG_WIND_THRESHOLD,
    add_fog_indicators,
    add_seasonal_emission_proxy,
    add_stagnation_enhancement,
    add_thermal_inversion_proxy,
    enrich_with_winter_features,
)
from src.training_pipeline.run_winter_ablation import (
    ABLATION_CONFIGS,
    check_adoption_criteria,
    WINTER_FOLDS,
    WINTER_MAX_REGRESSION_RMSE,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def _make_feature_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic feature dataframe covering multiple months."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-10-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "datetime_utc": dates,
        "temperature_2m": rng.uniform(5.0, 35.0, n),
        "relative_humidity_2m": rng.uniform(30.0, 95.0, n),
        "wind_speed_10m": rng.uniform(0.5, 15.0, n),
        "surface_pressure": rng.uniform(1005.0, 1025.0, n),
        "pm2_5": rng.uniform(10.0, 300.0, n),
        "pm_ratio": rng.uniform(0.3, 1.0, n),
        "epa_aqi": rng.uniform(30.0, 400.0, n),
    })


# =============================================================================
# Naming / Schema Guards
# =============================================================================

class TestSchemaConstraints:
    """Critical: verify correct naming and absence of removed features."""

    def test_inversion_risk_proxy_name_is_correct(self):
        """Must use 'inversion_risk_proxy' not 'inversion_risk_score'."""
        df = _make_feature_df()
        result = add_thermal_inversion_proxy(df)
        assert "inversion_risk_proxy" in result.columns

    def test_inversion_risk_score_name_absent(self):
        """Old name 'inversion_risk_score' must not appear in outputs."""
        df = _make_feature_df()
        result = add_thermal_inversion_proxy(df)
        assert "inversion_risk_score" not in result.columns

    def test_winter_month_index_absent_from_all_families(self):
        """'winter_month_index' was explicitly removed — must not exist in any family."""
        df = _make_feature_df()
        result = enrich_with_winter_features(df)
        assert "winter_month_index" not in result.columns

    def test_winter_month_index_absent_from_family_names(self):
        """FAMILY_NAMES constants must not include 'winter_month_index'."""
        for family, cols in FAMILY_NAMES.items():
            assert "winter_month_index" not in cols, f"Found in {family}"

    def test_all_winter_feature_cols_excludes_winter_month_index(self):
        assert "winter_month_index" not in ALL_WINTER_FEATURE_COLS

    def test_family1_produces_exactly_three_features(self):
        df = _make_feature_df()
        before_cols = set(df.columns)
        result = add_thermal_inversion_proxy(df)
        new_cols = set(result.columns) - before_cols
        assert new_cols == {"diurnal_temp_range_24h", "inversion_risk_proxy", "temp_drop_6h"}

    def test_family2_produces_exactly_two_features(self):
        df = _make_feature_df()
        before_cols = set(df.columns)
        result = add_fog_indicators(df)
        new_cols = set(result.columns) - before_cols
        assert new_cols == {"fog_proxy", "fog_hours_rolling_24h"}

    def test_family3_produces_exactly_three_features(self):
        df = _make_feature_df()
        before_cols = set(df.columns)
        result = add_stagnation_enhancement(df)
        new_cols = set(result.columns) - before_cols
        assert new_cols == {"wind_stagnation_hours_12h", "wind_stagnation_hours_24h", "pressure_stability_24h"}

    def test_family4_produces_exactly_two_features(self):
        df = _make_feature_df()
        before_cols = set(df.columns)
        result = add_seasonal_emission_proxy(df)
        new_cols = set(result.columns) - before_cols
        assert new_cols == {"crop_burning_season", "winter_emission_intensity"}


# =============================================================================
# Family 1: Thermal / Inversion-Risk Proxy
# =============================================================================

class TestThermalInversionProxy:
    def test_inversion_risk_proxy_range(self):
        """Score must be in [0, 1]."""
        df = _make_feature_df(1000)
        result = add_thermal_inversion_proxy(df)
        scores = result["inversion_risk_proxy"].dropna()
        assert scores.min() >= 0.0 - 1e-6
        assert scores.max() <= 1.0 + 1e-6

    def test_diurnal_temp_range_is_non_negative(self):
        df = _make_feature_df()
        result = add_thermal_inversion_proxy(df)
        vals = result["diurnal_temp_range_24h"].dropna()
        assert (vals >= 0.0).all()

    def test_temp_drop_6h_is_backward_looking(self):
        """temp_drop_6h at index i uses temperature at i and i-6. Row 0-5 should be NaN."""
        df = _make_feature_df(100)
        result = add_thermal_inversion_proxy(df)
        # First 6 values must be NaN (no 6h history)
        assert result["temp_drop_6h"].iloc[:6].isna().all()
        # Row 6 onward must not all be NaN
        assert result["temp_drop_6h"].iloc[6:].notna().any()

    def test_high_wind_lowers_inversion_risk_proxy(self):
        """Hours with near-calm wind should have a higher inversion_risk_proxy
        than hours with strong wind, when evaluated within the same dataframe
        so normalization is shared.
        """
        n = 200
        df = _make_feature_df(n)
        # Make first half calm, second half windy (same df → same normalization scale)
        df["wind_speed_10m"] = np.where(np.arange(n) < n // 2, 0.5, 15.0)
        df["temperature_2m"] = 15.0         # constant → DTR contribution equal
        df["relative_humidity_2m"] = 70.0   # constant → RH contribution equal

        result = add_thermal_inversion_proxy(df)["inversion_risk_proxy"].dropna()
        first_half = result.iloc[24 : n // 2].mean()    # skip warm-up (24h DTR)
        second_half = result.iloc[n // 2 :].mean()

        assert first_half > second_half, (
            f"Calm hours ({first_half:.4f}) should have higher proxy than windy ({second_half:.4f})"
        )


# =============================================================================
# Family 2: Fog/Mist Indicators
# =============================================================================

class TestFogIndicators:
    def test_fog_proxy_is_binary(self):
        df = _make_feature_df()
        result = add_fog_indicators(df)
        unique_vals = set(result["fog_proxy"].unique())
        assert unique_vals.issubset({0.0, 1.0})

    def test_fog_proxy_fires_under_threshold_conditions(self):
        """Row meeting all 3 fog conditions must receive fog_proxy = 1."""
        df = _make_feature_df(10)
        df["relative_humidity_2m"] = FOG_HUMIDITY_THRESHOLD + 1.0   # >85
        df["temperature_2m"] = FOG_TEMPERATURE_THRESHOLD - 1.0      # <12
        df["wind_speed_10m"] = FOG_WIND_THRESHOLD - 0.5             # <2.0
        result = add_fog_indicators(df)
        assert (result["fog_proxy"] == 1.0).all()

    def test_fog_proxy_off_when_humidity_below_threshold(self):
        df = _make_feature_df(10)
        df["relative_humidity_2m"] = FOG_HUMIDITY_THRESHOLD - 1.0   # <85 = no fog
        df["temperature_2m"] = 5.0
        df["wind_speed_10m"] = 0.5
        result = add_fog_indicators(df)
        assert (result["fog_proxy"] == 0.0).all()

    def test_fog_hours_rolling_is_non_decreasing_within_consecutive_fog(self):
        """If fog_proxy is 1 for all rows, rolling count should be non-decreasing."""
        df = _make_feature_df(50)
        df["relative_humidity_2m"] = 95.0
        df["temperature_2m"] = 5.0
        df["wind_speed_10m"] = 0.5
        result = add_fog_indicators(df)
        rolling = result["fog_hours_rolling_24h"].values
        # Should be monotonically non-decreasing up to window size
        for i in range(1, min(25, len(rolling))):
            assert rolling[i] >= rolling[i - 1]

    def test_fog_hours_rolling_backward_looking(self):
        """fog_hours_rolling_24h must only use past 24h, never future rows."""
        df = _make_feature_df(100)
        result = add_fog_indicators(df)
        # At row 0 (no history), rolling count must be at most fog_proxy[0]
        assert result["fog_hours_rolling_24h"].iloc[0] == result["fog_proxy"].iloc[0]


# =============================================================================
# Family 3: Stagnation Enhancement
# =============================================================================

class TestStagnationEnhancement:
    def test_wind_stagnation_hours_non_negative(self):
        df = _make_feature_df()
        result = add_stagnation_enhancement(df)
        assert (result["wind_stagnation_hours_12h"] >= 0).all()
        assert (result["wind_stagnation_hours_24h"] >= 0).all()

    def test_wind_stagnation_hours_max_bounded_by_window(self):
        """12h window count cannot exceed 12."""
        df = _make_feature_df(200)
        df["wind_speed_10m"] = 0.0   # always stagnant
        result = add_stagnation_enhancement(df)
        assert result["wind_stagnation_hours_12h"].max() <= 12.0
        assert result["wind_stagnation_hours_24h"].max() <= 24.0

    def test_pressure_stability_non_negative(self):
        df = _make_feature_df()
        result = add_stagnation_enhancement(df)
        assert (result["pressure_stability_24h"] >= 0.0).all()

    def test_constant_pressure_gives_zero_stability(self):
        """Perfectly constant pressure → std = 0."""
        df = _make_feature_df(100)
        df["surface_pressure"] = 1013.25
        result = add_stagnation_enhancement(df)
        # After warmup, stability should be 0
        assert result["pressure_stability_24h"].iloc[25:].max() < 1e-6


# =============================================================================
# Family 4: Seasonal-Emission Proxy
# =============================================================================

class TestSeasonalEmissionProxy:
    def test_crop_burning_flag_in_october_after_15th(self):
        df = pd.DataFrame({
            "datetime_utc": pd.to_datetime(["2021-10-14 00:00+00:00",
                                            "2021-10-15 00:00+00:00",
                                            "2021-10-16 00:00+00:00"]),
            "pm_ratio": [0.7, 0.7, 0.7],
        })
        result = add_seasonal_emission_proxy(df)
        assert result["crop_burning_season"].iloc[0] == 0.0   # Oct 14 — before start
        assert result["crop_burning_season"].iloc[1] == 1.0   # Oct 15 — start day
        assert result["crop_burning_season"].iloc[2] == 1.0   # Oct 16

    def test_crop_burning_flag_in_november(self):
        df = pd.DataFrame({
            "datetime_utc": pd.to_datetime(["2021-11-01 00:00+00:00",
                                            "2021-11-30 23:00+00:00"]),
            "pm_ratio": [0.7, 0.7],
        })
        result = add_seasonal_emission_proxy(df)
        assert (result["crop_burning_season"] == 1.0).all()

    def test_crop_burning_flag_off_in_december(self):
        df = pd.DataFrame({
            "datetime_utc": pd.to_datetime(["2021-12-01 00:00+00:00"]),
            "pm_ratio": [0.7],
        })
        result = add_seasonal_emission_proxy(df)
        assert result["crop_burning_season"].iloc[0] == 0.0

    def test_crop_burning_flag_off_in_summer(self):
        df = pd.DataFrame({
            "datetime_utc": pd.to_datetime(["2021-06-15 00:00+00:00"]),
            "pm_ratio": [0.7],
        })
        result = add_seasonal_emission_proxy(df)
        assert result["crop_burning_season"].iloc[0] == 0.0

    def test_winter_emission_intensity_is_pm_ratio_during_burning(self):
        """winter_emission_intensity = pm_ratio × 1 during burning season."""
        df = pd.DataFrame({
            "datetime_utc": pd.to_datetime(["2021-11-15 00:00+00:00"]),
            "pm_ratio": [0.75],
        })
        result = add_seasonal_emission_proxy(df)
        assert abs(result["winter_emission_intensity"].iloc[0] - 0.75) < 1e-5

    def test_winter_emission_intensity_zero_outside_burning(self):
        df = pd.DataFrame({
            "datetime_utc": pd.to_datetime(["2021-07-15 00:00+00:00"]),
            "pm_ratio": [0.8],
        })
        result = add_seasonal_emission_proxy(df)
        assert abs(result["winter_emission_intensity"].iloc[0]) < 1e-5


# =============================================================================
# enrich_with_winter_features — Unified Entry Point
# =============================================================================

class TestEnrichWithWinterFeatures:
    def test_all_families_adds_all_feature_cols(self):
        df = _make_feature_df()
        result = enrich_with_winter_features(df)
        for col in ALL_WINTER_FEATURE_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_selective_family_only_adds_requested_features(self):
        df = _make_feature_df()
        result = enrich_with_winter_features(df, families=["family1_inversion"])
        # Family 1 features present
        for col in FAMILY_NAMES["family1_inversion"]:
            assert col in result.columns
        # Family 2 features absent
        for col in FAMILY_NAMES["family2_fog"]:
            assert col not in result.columns

    def test_invalid_family_raises(self):
        df = _make_feature_df()
        with pytest.raises(ValueError, match="Unknown family"):
            enrich_with_winter_features(df, families=["family99_invalid"])

    def test_original_df_not_mutated(self):
        df = _make_feature_df()
        original_cols = set(df.columns)
        _ = enrich_with_winter_features(df)
        assert set(df.columns) == original_cols


# =============================================================================
# Adoption Gate Tests
# =============================================================================

def _make_fold_result(fold_id: str, overall_rmse: float,
                      h1_6: float = 50.0, h7_24: float = 75.0,
                      h25_48: float = 85.0, h49_72: float = 90.0,
                      gt200_rmse: float = 120.0, gt300_rmse: float = 150.0) -> dict:
    return {
        "fold_id": fold_id,
        "diagnostics": {
            "overall": {"rmse": overall_rmse, "r2": 0.4},
            "horizon_groups": {
                "short_h1_6":          {"rmse": h1_6},
                "medium_h7_24":        {"rmse": h7_24},
                "medium_long_h25_48":  {"rmse": h25_48},
                "long_h49_72":         {"rmse": h49_72},
            },
            "extreme_events": {
                "high_severity_gt200": {"rmse": gt200_rmse, "n": 100, "low_sample_warning": False},
                "hazardous_gt300":     {"rmse": gt300_rmse, "n": 40, "low_sample_warning": False},
            },
        },
    }


FOLD_IDS = ["fold_1_winter2021", "fold_2_transition_summer2022",
            "fold_3_monsoon2022", "fold_4_winter2023"]

BASELINE = [
    _make_fold_result("fold_1_winter2021",         104.23),
    _make_fold_result("fold_2_transition_summer2022", 72.43),
    _make_fold_result("fold_3_monsoon2022",          71.88),
    _make_fold_result("fold_4_winter2023",           85.21),
]


class TestAdoptionGate:
    def test_criterion1_passes_when_mean_winter_improves_and_no_regression(self):
        """Both winter folds improve → criterion 1 passes."""
        candidate = [
            _make_fold_result("fold_1_winter2021", 100.0),          # -4.23
            _make_fold_result("fold_2_transition_summer2022", 72.43),
            _make_fold_result("fold_3_monsoon2022", 71.88),
            _make_fold_result("fold_4_winter2023", 83.0),            # -2.21
        ]
        decision = check_adoption_criteria(candidate, BASELINE)
        assert decision["criteria"]["1_winter_mean_improvement"]["passed"] is True

    def test_criterion1_fails_when_one_winter_regresses_more_than_threshold(self):
        """One winter fold regresses by more than WINTER_MAX_REGRESSION_RMSE → fail."""
        candidate = [
            _make_fold_result("fold_1_winter2021", 100.0),           # improves
            _make_fold_result("fold_2_transition_summer2022", 72.43),
            _make_fold_result("fold_3_monsoon2022", 71.88),
            _make_fold_result("fold_4_winter2023",
                              85.21 + WINTER_MAX_REGRESSION_RMSE + 0.1),  # over threshold
        ]
        decision = check_adoption_criteria(candidate, BASELINE)
        assert decision["criteria"]["1_winter_mean_improvement"]["passed"] is False

    def test_criterion1_fails_when_mean_winter_does_not_improve(self):
        """Net winter improvement = 0 → criterion 1 fails (mean delta not > 0)."""
        candidate = [
            _make_fold_result("fold_1_winter2021", 102.0),   # improves by 2.23
            _make_fold_result("fold_2_transition_summer2022", 72.43),
            _make_fold_result("fold_3_monsoon2022", 71.88),
            _make_fold_result("fold_4_winter2023", 87.44),   # worsens by 2.23
        ]
        decision = check_adoption_criteria(candidate, BASELINE)
        assert decision["criteria"]["1_winter_mean_improvement"]["passed"] is False

    def test_criterion2_fails_when_overall_regresses_more_than_threshold(self):
        """Overall mean RMSE worsens by more than 2.0 → fails."""
        candidate = [
            _make_fold_result("fold_1_winter2021", 107.0),   # +2.77
            _make_fold_result("fold_2_transition_summer2022", 75.0),  # +2.57
            _make_fold_result("fold_3_monsoon2022", 74.0),    # +2.12
            _make_fold_result("fold_4_winter2023", 87.5),     # +2.29
        ]
        decision = check_adoption_criteria(candidate, BASELINE)
        assert decision["criteria"]["2_overall_no_regression"]["passed"] is False

    def test_all_criteria_passed_returns_model_v2_candidate(self):
        """All criteria satisfied → MODEL_V2_CANDIDATE."""
        # Each fold improves across the board
        candidate = [
            _make_fold_result("fold_1_winter2021", 98.0, gt200_rmse=115.0, gt300_rmse=145.0),
            _make_fold_result("fold_2_transition_summer2022", 71.0, gt200_rmse=115.0, gt300_rmse=145.0),
            _make_fold_result("fold_3_monsoon2022", 70.0, gt200_rmse=115.0, gt300_rmse=145.0),
            _make_fold_result("fold_4_winter2023", 83.0, gt200_rmse=115.0, gt300_rmse=145.0),
        ]
        decision = check_adoption_criteria(candidate, BASELINE)
        assert decision["all_criteria_passed"] is True
        assert decision["recommendation"] == "MODEL_V2_CANDIDATE"

    def test_failing_any_criterion_gives_reject(self):
        """Failing criterion 3 (summer regression) → REJECT."""
        candidate = [
            _make_fold_result("fold_1_winter2021", 98.0),
            _make_fold_result("fold_2_transition_summer2022", 80.0),  # +7.57 regression
            _make_fold_result("fold_3_monsoon2022", 71.88),
            _make_fold_result("fold_4_winter2023", 83.0),
        ]
        decision = check_adoption_criteria(candidate, BASELINE)
        assert decision["recommendation"] == "REJECT"


# =============================================================================
# Ablation Config Structural Tests
# =============================================================================

class TestAblationMatrix:
    def test_exactly_six_ablation_configs(self):
        assert len(ABLATION_CONFIGS) == 6

    def test_abl_000_has_no_families(self):
        abl000 = next(c for c in ABLATION_CONFIGS if c["id"] == "ABL-000")
        assert abl000["families"] == []

    def test_abl_005_has_all_four_families(self):
        abl005 = next(c for c in ABLATION_CONFIGS if c["id"] == "ABL-005")
        assert set(abl005["families"]) == set(FAMILY_NAMES.keys())

    def test_single_family_ablations_each_test_one_family(self):
        single_family_abls = [c for c in ABLATION_CONFIGS
                               if c["id"] in {"ABL-001", "ABL-002", "ABL-003", "ABL-004"}]
        for abl in single_family_abls:
            assert len(abl["families"]) == 1, f"{abl['id']} should test exactly 1 family"

    def test_ablation_ids_are_unique(self):
        ids = [c["id"] for c in ABLATION_CONFIGS]
        assert len(ids) == len(set(ids))
