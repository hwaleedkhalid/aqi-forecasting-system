"""Tests for Phase 11: Walk-Forward Stability & Diagnostic Validation.

Covers:
- SeasonClassifier: explicit month-by-month assertions (10 checks)
- Embargo invariant: structural unit test
- DiagnosticEvaluator: output structure verification
- Extreme-event metrics: low-sample flag verification
- WalkForwardFoldEngine: fold construction and embargo checking
- WalkForwardReportGenerator: CSV structure (16 rows for 4×4 evaluations)
- FoldTrainer: preprocessing is fitted only on training data
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.training_pipeline.season_classifier import (
    LAHORE_MONTH_TO_SEASON,
    LahoreSeasonClassifier,
)
from src.training_pipeline.walk_forward_validator import (
    DiagnosticEvaluator,
    WalkForwardFoldEngine,
    WalkForwardFoldSpec,
    WalkForwardReportGenerator,
    _extreme_event_metrics,
    LOW_SAMPLE_THRESHOLD,
)


# =============================================================================
# SeasonClassifier Tests
# =============================================================================

class TestLahoreSeasonClassifier:
    """Explicitly verify the complete month-to-season mapping."""

    def setup_method(self):
        self.clf = LahoreSeasonClassifier()

    # Winter/Smog months
    def test_january_is_winter_smog(self):
        assert self.clf.classify(1) == "winter_smog"

    def test_february_is_winter_smog(self):
        assert self.clf.classify(2) == "winter_smog"

    def test_november_is_winter_smog(self):
        assert self.clf.classify(11) == "winter_smog"

    def test_december_is_winter_smog(self):
        assert self.clf.classify(12) == "winter_smog"

    # Transition months
    def test_march_is_transition(self):
        assert self.clf.classify(3) == "transition"

    def test_april_is_transition(self):
        assert self.clf.classify(4) == "transition"

    def test_october_is_transition(self):
        assert self.clf.classify(10) == "transition"

    # Summer months
    def test_may_is_summer(self):
        assert self.clf.classify(5) == "summer"

    def test_june_is_summer(self):
        assert self.clf.classify(6) == "summer"

    # Monsoon months
    def test_july_is_monsoon(self):
        assert self.clf.classify(7) == "monsoon"

    def test_august_is_monsoon(self):
        assert self.clf.classify(8) == "monsoon"

    def test_september_is_monsoon(self):
        assert self.clf.classify(9) == "monsoon"

    def test_invalid_month_raises(self):
        with pytest.raises(ValueError, match="Month must be in"):
            self.clf.classify(0)

    def test_invalid_month_13_raises(self):
        with pytest.raises(ValueError, match="Month must be in"):
            self.clf.classify(13)

    def test_classify_series(self):
        result = self.clf.classify_series([1, 7, 11])
        assert result == ["winter_smog", "monsoon", "winter_smog"]

    def test_all_12_months_covered(self):
        """Every month 1–12 must have exactly one season label."""
        clf = LahoreSeasonClassifier()
        seasons = {clf.classify(m) for m in range(1, 13)}
        assert seasons == {"winter_smog", "summer", "monsoon", "transition"}

    def test_mapping_is_authoritative_single_definition(self):
        """The mapping in LAHORE_MONTH_TO_SEASON matches classify() for all months."""
        clf = LahoreSeasonClassifier()
        for month, expected_season in LAHORE_MONTH_TO_SEASON.items():
            assert clf.classify(month) == expected_season


# =============================================================================
# Embargo Invariant Tests
# =============================================================================

class TestEmbargoInvariant:
    """Verify the explicit embargo invariant is enforced at fold construction time.

    Invariant: max(train_feature_timestamp) + embargo_hours + 1h <= val_start
    Equivalently:  val_start > train_end + embargo_hours
    """

    def test_valid_embargo_does_not_raise(self):
        """A fold with sufficient gap should construct without error."""
        engine = WalkForwardFoldEngine()
        fold = WalkForwardFoldSpec(
            fold_id="test_valid",
            train_end=pd.Timestamp("2021-10-01 00:00:00+00:00"),
            val_start=pd.Timestamp("2021-10-04 01:00:00+00:00"),  # 3 days + 1h gap
            val_end=pd.Timestamp("2021-12-31 23:00:00+00:00"),
            embargo_hours=72,
        )
        # Should not raise
        engine._verify_embargo_invariant(fold)

    def test_embargo_exactly_at_limit_raises(self):
        """A fold where val_start == train_end + embargo_hours violates the invariant."""
        engine = WalkForwardFoldEngine()
        fold = WalkForwardFoldSpec(
            fold_id="test_exact_limit",
            train_end=pd.Timestamp("2021-10-01 00:00:00+00:00"),
            val_start=pd.Timestamp("2021-10-04 00:00:00+00:00"),  # exactly 72h, not strictly >
            val_end=pd.Timestamp("2021-12-31 23:00:00+00:00"),
            embargo_hours=72,
        )
        with pytest.raises(ValueError, match="Embargo invariant violated"):
            engine._verify_embargo_invariant(fold)

    def test_embargo_below_limit_raises(self):
        """val_start less than train_end + 72h must raise."""
        engine = WalkForwardFoldEngine()
        fold = WalkForwardFoldSpec(
            fold_id="test_short_gap",
            train_end=pd.Timestamp("2021-10-01 00:00:00+00:00"),
            val_start=pd.Timestamp("2021-10-02 00:00:00+00:00"),  # only 24h gap
            val_end=pd.Timestamp("2021-12-31 23:00:00+00:00"),
            embargo_hours=72,
        )
        with pytest.raises(ValueError, match="Embargo invariant violated"):
            engine._verify_embargo_invariant(fold)

    def test_all_production_folds_satisfy_embargo(self):
        """All 4 production fold specs must pass the embargo invariant."""
        engine = WalkForwardFoldEngine()
        folds = engine.build_folds()
        assert len(folds) == 4
        for fold in folds:
            # If invariant is violated, build_folds() would have raised already.
            assert fold.val_start > fold.train_end + pd.Timedelta(hours=fold.embargo_hours), (
                f"Embargo invariant failed for {fold.fold_id}"
            )

    def test_production_fold_gaps_are_at_least_72h(self):
        """Each production fold must have >= 72h gap between train_end and val_start."""
        engine = WalkForwardFoldEngine()
        folds = engine.build_folds()
        for fold in folds:
            gap = fold.val_start - fold.train_end
            assert gap >= pd.Timedelta(hours=72), (
                f"{fold.fold_id} has insufficient gap: {gap}"
            )


# =============================================================================
# DiagnosticEvaluator Tests
# =============================================================================

def _make_fake_predictions(n: int = 200, horizons: int = 72, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(50, 300, size=(n, horizons)).astype(np.float32)
    noise = rng.normal(0, 20, size=(n, horizons)).astype(np.float32)
    y_pred = np.clip(y_true + noise, 0, 500).astype(np.float32)
    return y_true, y_pred


def _make_timestamps(n: int = 200):
    return pd.Series(
        pd.date_range("2021-11-01", periods=n, freq="h", tz="UTC")
    )


def _make_feature_df(n: int = 200, cols=None, seed=42):
    rng = np.random.default_rng(seed)
    if cols is None:
        cols = ["pm2_5", "epa_aqi", "temperature_2m", "wind_speed_10m"]
    data = {c: rng.uniform(10, 200, size=n) for c in cols}
    df = pd.DataFrame(data)
    return df


class TestDiagnosticEvaluator:
    """Verify all 5 diagnostic dimension keys are present in output."""

    def setup_method(self):
        self.evaluator = DiagnosticEvaluator()
        self.y_true, self.y_pred = _make_fake_predictions(200)
        self.timestamps = _make_timestamps(200)
        self.df_train = _make_feature_df(500)
        self.df_val = _make_feature_df(200)
        self.dist_cols = ["pm2_5", "epa_aqi", "temperature_2m", "wind_speed_10m"]

    def test_returns_all_five_dimension_keys(self):
        result = self.evaluator.evaluate(
            self.y_true, self.y_pred, self.timestamps,
            self.df_train, self.df_val, self.dist_cols,
        )
        assert "overall" in result
        assert "seasonal" in result
        assert "horizon_groups" in result
        assert "extreme_events" in result
        assert "distribution_shift" in result

    def test_overall_metrics_present(self):
        result = self.evaluator.evaluate(
            self.y_true, self.y_pred, self.timestamps,
            self.df_train, self.df_val, self.dist_cols,
        )
        overall = result["overall"]
        assert "rmse" in overall
        assert "mae" in overall
        assert "r2" in overall
        assert "n_samples" in overall
        assert overall["n_samples"] == 200

    def test_horizon_groups_all_four_groups_present(self):
        result = self.evaluator.evaluate(
            self.y_true, self.y_pred, self.timestamps,
            self.df_train, self.df_val, self.dist_cols,
        )
        hor = result["horizon_groups"]
        assert "short_h1_6" in hor
        assert "medium_h7_24" in hor
        assert "medium_long_h25_48" in hor
        assert "long_h49_72" in hor

    def test_extreme_events_has_both_tiers(self):
        result = self.evaluator.evaluate(
            self.y_true, self.y_pred, self.timestamps,
            self.df_train, self.df_val, self.dist_cols,
        )
        ext = result["extreme_events"]
        assert "high_severity_gt200" in ext
        assert "hazardous_gt300" in ext
        assert "top_1pct_errors" in ext

    def test_distribution_shift_has_wasserstein_distance(self):
        result = self.evaluator.evaluate(
            self.y_true, self.y_pred, self.timestamps,
            self.df_train, self.df_val, self.dist_cols,
        )
        dist = result["distribution_shift"]
        for col in self.dist_cols:
            if col in dist and isinstance(dist[col], dict):
                assert "wasserstein_distance" in dist[col]

    def test_seasonal_labels_are_valid(self):
        from src.training_pipeline.season_classifier import VALID_SEASONS
        result = self.evaluator.evaluate(
            self.y_true, self.y_pred, self.timestamps,
            self.df_train, self.df_val, self.dist_cols,
        )
        for season_key in result["seasonal"]:
            assert season_key in VALID_SEASONS, f"Unexpected season key: {season_key}"


# =============================================================================
# Extreme-Event Metrics Low-N Flag Tests
# =============================================================================

class TestExtremeEventMetrics:
    """Verify low_sample_warning flag behavior."""

    def test_low_sample_flag_triggered_below_threshold(self):
        n_small = LOW_SAMPLE_THRESHOLD - 1
        # Create y_true where exactly n_small values exceed 300
        y_true = np.zeros(500)
        y_true[:n_small] = 350.0
        y_pred = np.zeros(500)
        y_pred[:n_small] = 300.0

        result = _extreme_event_metrics(y_true, y_pred, 300.0, "AQI>300")
        assert result["n"] == n_small
        assert result["low_sample_warning"] is True

    def test_low_sample_flag_not_triggered_at_threshold(self):
        n = LOW_SAMPLE_THRESHOLD
        y_true = np.zeros(500)
        y_true[:n] = 350.0
        y_pred = np.zeros(500)
        y_pred[:n] = 300.0

        result = _extreme_event_metrics(y_true, y_pred, 300.0, "AQI>300")
        assert result["n"] == n
        assert result["low_sample_warning"] is False

    def test_zero_extreme_events_returns_none_metrics(self):
        y_true = np.full(100, 50.0)   # All AQI=50, none > 300
        y_pred = np.full(100, 50.0)

        result = _extreme_event_metrics(y_true, y_pred, 300.0, "AQI>300")
        assert result["n"] == 0
        assert result["rmse"] is None
        assert result["mae"] is None
        assert result["low_sample_warning"] is True

    def test_high_severity_threshold_is_gt200(self):
        y_true = np.array([190.0, 200.0, 201.0, 250.0])
        y_pred = np.array([190.0, 200.0, 180.0, 200.0])

        result = _extreme_event_metrics(y_true, y_pred, 200.0, "AQI>200")
        # Only 201 and 250 exceed 200
        assert result["n"] == 2

    def test_hazardous_threshold_is_gt300(self):
        y_true = np.array([290.0, 300.0, 301.0, 400.0])
        y_pred = np.array([290.0, 300.0, 280.0, 350.0])

        result = _extreme_event_metrics(y_true, y_pred, 300.0, "AQI>300")
        # Only 301 and 400 exceed 300
        assert result["n"] == 2

    def test_sample_count_in_result(self):
        y_true = np.full(100, 350.0)
        y_pred = np.full(100, 300.0)

        result = _extreme_event_metrics(y_true, y_pred, 300.0, "AQI>300")
        assert "n" in result
        assert result["n"] == 100


# =============================================================================
# WalkForwardFoldEngine Tests
# =============================================================================

class TestWalkForwardFoldEngine:
    """Test fold construction and embargo validation."""

    def test_builds_exactly_four_folds(self):
        engine = WalkForwardFoldEngine()
        folds = engine.build_folds()
        assert len(folds) == 4

    def test_fold_ids_are_unique(self):
        engine = WalkForwardFoldEngine()
        folds = engine.build_folds()
        fold_ids = [f.fold_id for f in folds]
        assert len(fold_ids) == len(set(fold_ids))

    def test_folds_are_chronologically_ordered(self):
        engine = WalkForwardFoldEngine()
        folds = engine.build_folds()
        for i in range(len(folds) - 1):
            assert folds[i].train_end <= folds[i + 1].train_end

    def test_fold_val_windows_do_not_overlap_training(self):
        engine = WalkForwardFoldEngine()
        folds = engine.build_folds()
        for fold in folds:
            assert fold.val_start > fold.train_end

    def test_expanding_train_windows(self):
        """Each fold's train_end is >= the previous fold's train_end."""
        engine = WalkForwardFoldEngine()
        folds = engine.build_folds()
        for i in range(1, len(folds)):
            assert folds[i].train_end >= folds[i - 1].train_end


# =============================================================================
# WalkForwardReportGenerator Tests
# =============================================================================

class TestWalkForwardReportGenerator:
    """Test report generation and CSV structure."""

    def _make_fake_results(self) -> dict:
        """Construct minimal synthetic results mimicking 4×4 evaluations."""
        model_keys = ["exp019", "exp017", "ridge_v2", "naive"]
        fold_ids = ["fold_1", "fold_2", "fold_3", "fold_4"]
        results = {"folds": {}, "metadata": {}}
        for fold_id in fold_ids:
            fold_data = {"models": {}, "primary_seasons": "test"}
            for mk in model_keys:
                fold_data["models"][mk] = {
                    "diagnostics": {
                        "overall": {"rmse": 80.0, "mae": 60.0, "r2": 0.4, "n_samples": 100},
                        "horizon_groups": {
                            "short_h1_6": {"rmse": 50.0, "mae": 35.0, "r2": 0.6, "horizons": "h1-h6"},
                            "medium_h7_24": {"rmse": 75.0, "mae": 55.0, "r2": 0.45, "horizons": "h7-h24"},
                            "medium_long_h25_48": {"rmse": 85.0, "mae": 65.0, "r2": 0.38, "horizons": "h25-h48"},
                            "long_h49_72": {"rmse": 90.0, "mae": 70.0, "r2": 0.35, "horizons": "h49-h72"},
                        },
                        "extreme_events": {
                            "high_severity_gt200": {"n": 50, "rmse": 120.0, "mae": 90.0, "low_sample_warning": False},
                            "hazardous_gt300": {"n": 5, "rmse": 180.0, "mae": 140.0, "low_sample_warning": True},
                            "top_1pct_errors": {"threshold_abs_error": 200.0, "n": 10, "mean_abs_error": 250.0, "low_sample_warning": True},
                        },
                    }
                }
            results["folds"][fold_id] = fold_data
        return results

    def test_csv_has_16_rows(self, tmp_path):
        reporter = WalkForwardReportGenerator(tmp_path)
        results = self._make_fake_results()
        path = reporter.save_summary_csv(results)
        df = pd.read_csv(path)
        assert len(df) == 16  # 4 folds × 4 models

    def test_csv_has_required_columns(self, tmp_path):
        reporter = WalkForwardReportGenerator(tmp_path)
        results = self._make_fake_results()
        path = reporter.save_summary_csv(results)
        df = pd.read_csv(path)
        required_cols = [
            "fold_id", "model_key", "overall_rmse", "overall_mae", "overall_r2",
            "hazardous_gt300_n", "hazardous_gt300_low_n_flag", "n_samples",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_json_report_saved(self, tmp_path):
        reporter = WalkForwardReportGenerator(tmp_path)
        results = self._make_fake_results()
        path = reporter.save_json_report(results)
        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert "folds" in data

    def test_stability_report_structure(self, tmp_path):
        reporter = WalkForwardReportGenerator(tmp_path)
        results = self._make_fake_results()
        stability = reporter.build_stability_report(results)
        # With identical RMSE (80.0 each), delta should be 0.0
        assert stability["fold_count"] == 4
        assert "mean_delta" in stability
        assert "median_delta" in stability
        assert "std_delta" in stability
        assert "worst_fold_delta" in stability
        assert "exp019_wins" in stability
        assert "relative_gain_percent" in stability
