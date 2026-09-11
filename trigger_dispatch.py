"""Zero-dependency standalone GitHub Actions repository_dispatch trigger for Render Cron."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


def resolve_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_PAT")
    if token:
        return token
    try:
        import subprocess
        p = subprocess.Popen(
            ["git", "credential", "fill"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        out, _ = p.communicate(input="protocol=https\nhost=github.com\n\n", timeout=5)
        creds = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        return creds.get("password")
    except Exception:
        return None


def main() -> int:
    token = resolve_token()
    repo = os.environ.get("GITHUB_REPOSITORY", "hwaleedkhalid/aqi-forecasting-system")
    event_type = os.environ.get("DISPATCH_EVENT_TYPE", "hourly_ingest")
    timestamp = datetime.now(timezone.utc).isoformat()

    if not token:
        print(f"[{timestamp}] Error: GITHUB_TOKEN is required for repository_dispatch trigger", file=sys.stderr)
        return 1

    url = f"https://api.github.com/repos/{repo}/dispatches"
    payload = json.dumps({
        "event_type": event_type,
        "client_payload": {"triggered_by": "render_cron_job", "timestamp": timestamp},
    }).encode("utf-8")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Pearls-Render-Cron-Dispatcher",
        "Content-Type": "application/json",
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    print(f"[{timestamp}] Dispatching '{event_type}' to GitHub Actions repository: {repo}...")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            status_code = resp.status
            if status_code in (200, 204):
                print(f"[{timestamp}] Successfully dispatched '{event_type}' (HTTP {status_code}). Trigger accepted by GitHub Actions.")
                return 0
            else:
                print(f"[{timestamp}] Unexpected status response: HTTP {status_code}", file=sys.stderr)
                return 1
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="ignore")
        print(f"[{timestamp}] HTTP Error {e.code}: {err_text}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[{timestamp}] Dispatch failed with exception: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
