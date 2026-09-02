"""Unit tests for DatasetDiagnostics module."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.training_pipeline.diagnostics import DatasetDiagnostics


class TestDatasetDiagnostics:
    """Test distribution shift, seasonal segmentation, and report generation."""

    def test_generate_diagnostic_report_runs_and_saves(self, tmp_path: Path) -> None:
        diagnostics = DatasetDiagnostics()
        report_path = tmp_path / "test_report.json"
        report = diagnostics.generate_diagnostic_report(output_path=report_path)

        assert report_path.exists()
        assert "distribution_shift" in report
        assert "segmented_performance" in report

        # Verify distribution shift contents
        dist = report["distribution_shift"]
        assert "epa_aqi_distribution" in dist
        assert "train" in dist["epa_aqi_distribution"]
        assert "test" in dist["epa_aqi_distribution"]
        assert dist["epa_aqi_distribution"]["train"]["mean"] > 0

        # Verify segmented performance contents
        seg = report["segmented_performance"]
        assert "seasonal_performance" in seg
        assert "Winter_Smog (Nov-Feb)" in seg["seasonal_performance"]
        assert "current_aqi_severity_performance" in seg
        assert "top_catastrophic_events" in seg
        assert len(seg["top_catastrophic_events"]) > 0
