"""Unit and verification tests for GitHub Actions workflows and automation runners."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import yaml

from src.exceptions import DataIngestionError
from src.feature_pipeline.run_hourly_ingestion import run_ingestion
from src.training_pipeline.run_candidate_evaluation import run_candidate_evaluation

WORKFLOWS_DIR = Path(".github/workflows")


@pytest.fixture
def workflow_files() -> dict[str, dict]:
    """Parse and return all workflow YAML documents."""
    workflows = {}
    for yml_path in WORKFLOWS_DIR.glob("*.yml"):
        with open(yml_path, "r", encoding="utf-8") as f:
            workflows[yml_path.name] = yaml.safe_load(f)
    return workflows


class TestWorkflowSchemaAndSecurity:
    """Test suite ensuring all GitHub Actions workflows adhere to security, syntax, and operational rules."""

    def test_all_expected_workflows_exist(self, workflow_files):
        assert "ci.yml" in workflow_files
        assert "feature_pipeline.yml" in workflow_files
        assert "training_pipeline.yml" in workflow_files

    def test_all_workflows_have_least_privilege_permissions(self, workflow_files):
        for name, wf in workflow_files.items():
            assert "permissions" in wf, f"Workflow {name} is missing top-level permissions block"
            assert wf["permissions"] == {"contents": "read"}, f"Workflow {name} permissions must be 'contents: read'"

    def test_ci_enforces_coverage_threshold(self, workflow_files):
        ci_raw = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
        assert "--cov-fail-under=70" in ci_raw
        assert "pytest" in ci_raw

    def test_ci_requires_no_external_api_secrets(self, workflow_files):
        ci_raw = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
        assert "secrets.OPENWEATHER_API_KEY" not in ci_raw
        assert "secrets." not in ci_raw

    def test_scheduled_crons_avoid_minute_zero(self, workflow_files):
        for name in ["feature_pipeline.yml", "training_pipeline.yml"]:
            wf = workflow_files[name]
            on_block = wf.get("on") or wf.get(True) or {}
            schedule = on_block.get("schedule", [])
            assert len(schedule) > 0, f"Workflow {name} must have a schedule trigger"
            for item in schedule:
                cron = item["cron"]
                minute = cron.split()[0]
                assert minute != "0", f"Workflow {name} cron '{cron}' must avoid minute 0 to prevent GitHub contention"

    def test_concurrency_configured_on_scheduled_workflows(self, workflow_files):
        for name in ["feature_pipeline.yml", "training_pipeline.yml"]:
            wf = workflow_files[name]
            assert "concurrency" in wf, f"Workflow {name} must configure concurrency group"
            assert wf["concurrency"]["cancel-in-progress"] is False

    def test_current_action_majors_used(self, workflow_files):
        for name in ["ci.yml", "feature_pipeline.yml", "training_pipeline.yml"]:
            raw = (WORKFLOWS_DIR / name).read_text(encoding="utf-8")
            assert "actions/checkout@v6" in raw
            assert "actions/setup-python@v6" in raw
            assert "actions/upload-artifact@v4" in raw

    def test_artifacts_uploaded_in_all_workflows(self, workflow_files):
        for name in ["ci.yml", "feature_pipeline.yml", "training_pipeline.yml"]:
            raw = (WORKFLOWS_DIR / name).read_text(encoding="utf-8")
            assert "actions/upload-artifact@v4" in raw

    def test_scheduled_feature_pipeline_references_secret(self):
        feat_raw = (WORKFLOWS_DIR / "feature_pipeline.yml").read_text(encoding="utf-8")
        assert "secrets.OPENWEATHER_API_KEY" in feat_raw

    def test_production_model_is_never_output_target_of_training_workflow(self):
        train_raw = (WORKFLOWS_DIR / "training_pipeline.yml").read_text(encoding="utf-8")
        assert "data/models/candidate" in train_raw
        assert "production_hybrid_model.joblib >" not in train_raw.replace("sha256sum data/models/production_hybrid_model.joblib > /tmp/prod_model_hash_", "")


class TestAutomationRunners:
    """Test suite for CLI automation runners executed by workflows."""

    def test_hourly_ingestion_dry_run(self, tmp_path):
        report = run_ingestion(dry_run=True, output_dir=tmp_path)
        assert report["status"] == "dry_run_success"
        assert report["mode"] == "dry_run"
        assert (tmp_path / "latest_ingestion_report.json").exists()

    def test_hourly_ingestion_live_missing_key_raises_error(self, tmp_path):
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": ""}, clear=True):
            with pytest.raises(DataIngestionError, match="OPENWEATHER_API_KEY is required"):
                run_ingestion(dry_run=False, api_key="", output_dir=tmp_path)

    def test_hourly_ingestion_live_success_mocked(self, tmp_path):
        mock_aq = {"list": [{"main": {"aqi": 3}}]}
        mock_weather = {"main": {"temp": 28.5}}

        with patch("src.data_ingestion.openweather_provider.OpenWeatherProvider.fetch_current_air_quality", return_value=mock_aq), \
             patch("src.data_ingestion.openweather_provider.OpenWeatherProvider.fetch_current_weather", return_value=mock_weather):
            report = run_ingestion(dry_run=False, api_key="valid_key", output_dir=tmp_path)
            assert report["status"] == "live_ingestion_success"
            assert report["mode"] == "live"
            assert (tmp_path / "latest_ingestion_report.json").exists()

    def test_candidate_evaluation_preserves_production_champion(self, tmp_path):
        prod_path = Path("data/models/production_hybrid_model.joblib")
        if not prod_path.exists():
            prod_path = Path("data/runtime/production/production_hybrid_model.joblib")
        assert prod_path.exists(), "Production model artifact must exist"
        initial_mtime = prod_path.stat().st_mtime

        report = run_candidate_evaluation(output_dir=tmp_path)

        # Assert report was written to candidate output directory
        assert report["status"] == "candidate_evaluation_completed"
        assert report["production_champion"]["model_id"] == "EXP-019"
        assert (tmp_path / "candidate_evaluation_report.json").exists()

        # Assert production champion artifact was untouched
        assert prod_path.stat().st_mtime == initial_mtime
