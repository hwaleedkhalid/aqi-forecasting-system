"""Unit tests for independent hourly scheduler and repository_dispatch trigger."""

from unittest.mock import MagicMock, patch
import pytest

from src.feature_pipeline.scheduler import (
    run_scheduled_iteration,
    trigger_github_repository_dispatch,
)


class TestIndependentScheduler:
    """Test suite for repository_dispatch trigger and scheduler execution."""

    def test_trigger_dispatch_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        with patch("src.feature_pipeline.scheduler.resolve_github_token", return_value=None):
            with pytest.raises(ValueError) as exc_info:
                trigger_github_repository_dispatch(github_token=None)
            assert "GITHUB_TOKEN is required" in str(exc_info.value)

    @patch("src.feature_pipeline.scheduler.requests.post")
    def test_trigger_dispatch_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        mock_post.return_value = mock_resp

        res = trigger_github_repository_dispatch(
            github_token="ghp_mock_token_12345",
            repo="hwaleedkhalid/aqi-forecasting-system",
            event_type="hourly_ingest",
        )

        assert res["status"] == "dispatch_success"
        assert res["status_code"] == 204
        mock_post.assert_called_once()
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer ghp_mock_token_12345"

    @patch("src.feature_pipeline.scheduler.requests.post")
    def test_trigger_dispatch_failure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Bad credentials"
        mock_post.return_value = mock_resp

        res = trigger_github_repository_dispatch(
            github_token="ghp_invalid_token",
            repo="hwaleedkhalid/aqi-forecasting-system",
        )

        assert res["status"] == "dispatch_failure"
        assert res["status_code"] == 401
        assert "Bad credentials" in res["error"]

    @patch("src.feature_pipeline.scheduler.trigger_github_repository_dispatch")
    def test_run_scheduled_iteration_with_token(self, mock_dispatch):
        mock_dispatch.return_value = {"status": "dispatch_success"}
        res = run_scheduled_iteration(github_token="ghp_valid")
        assert res["status"] == "dispatch_success"
        mock_dispatch.assert_called_once()

    @patch("src.feature_pipeline.scheduler.run_hourly_pipeline")
    def test_run_scheduled_iteration_fallback_to_live_pipeline(self, mock_pipeline, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        mock_pipeline.return_value = {"status": "live_ingestion_success", "hopsworks_mutated": True}

        res = run_scheduled_iteration(github_token=None, live_fallback=True)
        assert res["status"] == "live_ingestion_success"
        mock_pipeline.assert_called_once_with(live=True, dry_run=False)
