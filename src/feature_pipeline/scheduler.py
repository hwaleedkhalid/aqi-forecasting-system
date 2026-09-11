"""Pearls AQI Predictor - Independent Hourly Scheduler.

Provides independent, resilient hourly scheduling for the feature ingestion pipeline.
Can operate either by:
1. Dispatching a repository_dispatch event (event_type='hourly_ingest') to GitHub Actions, or
2. Executing the canonical live ingestion pipeline in-process when run in a container/sidecar.

Enforces strict idempotency and dual-store integrity via Hopsworks upsert_if_newer semantics.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import requests

from src.logger import logger
from src.feature_pipeline.run_hourly_ingestion import run_hourly_pipeline

DEFAULT_REPO = os.environ.get("GITHUB_REPOSITORY", "hwaleedkhalid/aqi-forecasting-system")


def trigger_github_repository_dispatch(
    github_token: str | None = None,
    repo: str = DEFAULT_REPO,
    event_type: str = "hourly_ingest",
    client_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send repository_dispatch event to trigger GitHub Actions feature pipeline.

    Args:
        github_token: GitHub Personal Access Token (PAT) with Actions write permissions.
        repo: Target GitHub repository (owner/repo).
        event_type: Event type identifier matching workflow trigger.
        client_payload: Optional JSON payload for the dispatch.

    Returns:
        Status report dictionary.
    """
    if github_token is None:
        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if not github_token:
        raise ValueError("GITHUB_TOKEN is required to send repository_dispatch to GitHub Actions.")

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Pearls-Hourly-Scheduler",
    }
    payload = {
        "event_type": event_type,
        "client_payload": client_payload or {"triggered_by": "independent_hourly_scheduler"},
    }

    logger.info(f"Dispatching '{event_type}' event to GitHub repository {repo}...")
    resp = requests.post(url, headers=headers, json=payload, timeout=20)

    if resp.status_code == 204:
        logger.info(f"Successfully dispatched '{event_type}' event to GitHub Actions.")
        return {
            "status": "dispatch_success",
            "status_code": 204,
            "event_type": event_type,
            "repo": repo,
            "dispatched_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        logger.error(f"GitHub repository_dispatch failed: HTTP {resp.status_code} - {resp.text}")
        return {
            "status": "dispatch_failure",
            "status_code": resp.status_code,
            "error": resp.text,
            "event_type": event_type,
            "repo": repo,
            "dispatched_at": datetime.now(timezone.utc).isoformat(),
        }


def run_scheduled_iteration(
    github_token: str | None = None,
    live_fallback: bool = True,
) -> dict[str, Any]:
    """Execute a single hourly scheduler iteration.

    Attempts repository_dispatch first if GITHUB_TOKEN is present; otherwise
    executes the canonical live ingestion pipeline directly.

    Args:
        github_token: Optional GitHub PAT.
        live_fallback: Whether to execute live pipeline directly if dispatch token is unavailable.

    Returns:
        Execution summary dictionary.
    """
    token = github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        try:
            return trigger_github_repository_dispatch(github_token=token)
        except Exception as e:
            logger.warning(f"Repository dispatch trigger failed ({e}); evaluating live fallback...")
            if not live_fallback:
                raise

    logger.info("Executing canonical hourly ingestion pipeline via local/direct runner...")
    return run_hourly_pipeline(live=True, dry_run=False)


def run_scheduler_daemon(target_minute: int = 15, poll_interval_sec: int = 30) -> None:
    """Run an always-running hourly scheduler daemon that triggers at target_minute.

    Args:
        target_minute: Minute of the hour to trigger (default: 15).
        poll_interval_sec: Polling interval in seconds to check system clock.
    """
    import threading
    logger.info(f"Starting hourly scheduler daemon (target: minute :{target_minute:02d})...")
    last_executed_hour: int | None = None

    while True:
        now = datetime.now(timezone.utc)
        current_hour = now.hour
        current_minute = now.minute

        if current_minute == target_minute and last_executed_hour != current_hour:
            logger.info(f"Scheduler daemon alarm triggered at {now.isoformat()} (hour {current_hour}, min {current_minute})")
            try:
                result = run_scheduled_iteration()
                logger.info(f"Scheduled iteration completed: {result.get('status', 'unknown')}")
                last_executed_hour = current_hour
            except Exception as e:
                logger.error(f"Scheduler daemon iteration encountered error: {e}", exc_info=True)

        time.sleep(poll_interval_sec)


def start_background_scheduler(target_minute: int = 15) -> Any:
    """Start the hourly scheduler daemon in a background daemon thread.

    Args:
        target_minute: Minute of the hour to trigger (default: 15).

    Returns:
        The started Thread instance.
    """
    import threading
    thread = threading.Thread(
        target=run_scheduler_daemon,
        args=(target_minute,),
        daemon=True,
        name="Pearls-Hourly-Scheduler-Thread",
    )
    thread.start()
    logger.info(f"Launched background scheduler thread (target minute :{target_minute:02d}).")
    return thread


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pearls Independent Hourly Scheduler")
    parser.add_argument("--daemon", action="store_true", help="Run continuously in background daemon loop")
    parser.add_argument("--minute", type=int, default=15, help="Minute of the hour to trigger (default: 15)")
    parser.add_argument("--dispatch-now", action="store_true", help="Execute single dispatch immediately")
    args = parser.parse_args()

    if args.dispatch_now:
        res = run_scheduled_iteration()
        print(res)
    elif args.daemon:
        run_scheduler_daemon(target_minute=args.minute)
    else:
        res = run_scheduled_iteration()
        print(res)

