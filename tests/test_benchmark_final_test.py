"""Unit tests for Phase 10.5E benchmark suite."""

from pathlib import Path
import numpy as np
import pytest

from src.training_pipeline.benchmark_final_test import compute_comprehensive_metrics


class TestBenchmarkMetrics:
    """Test metric computation mechanics."""

    def test_compute_comprehensive_metrics_shape_and_checkpoints(self) -> None:
        N, H = 50, 72
        np.random.seed(42)
        y_true = np.random.uniform(50, 350, (N, H)).astype(np.float32)
        y_pred = y_true + np.random.normal(0, 10, (N, H)).astype(np.float32)

        res = compute_comprehensive_metrics(y_true, y_pred)

        assert "overall_rmse" in res
        assert "overall_mae" in res
        assert "overall_r2" in res
        assert "checkpoints" in res
        assert "per_horizon_rmse" in res
        assert len(res["per_horizon_rmse"]) == 72
        assert len(res["per_horizon_mae"]) == 72
        assert len(res["per_horizon_r2"]) == 72

        chk = res["checkpoints"]
        assert "h1_rmse" in chk
        assert "h6_rmse" in chk
        assert "h24_rmse" in chk
        assert "h72_rmse" in chk
        assert chk["h1_rmse"] > 0
        assert res["overall_r2"] > 0.8  # Strong signal
