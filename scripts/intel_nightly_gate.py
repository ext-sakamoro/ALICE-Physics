#!/usr/bin/env python3
"""Decide whether the nightly CI runs the Mac Intel leg.

Every commit that reaches main already runs the 5-OS CI on its push, so the
nightly is a safety net: it runs the `Test (macos-15-intel)` leg only when
main's HEAD has no successful run of that job (the push run failed, was
cancelled, or never ran it). The CI runs of HEAD are read with the GitHub API
(read-only, the workflow's GITHUB_TOKEN):

  | the `Test (macos-15-intel)` jobs of HEAD's ci.yml runs | nightly |
  |---|---|
  | any one concluded `success`                             | skip    |
  | none succeeded, one queued / in progress                | skip (leave it to finish) |
  | only failed / cancelled / timed out, or none at all     | run     |
  | the API could not be read                               | run (fail toward running) |

The run that calls this script is left out (its own Intel job does not exist
yet). Writes `run=true` / `run=false` to $GITHUB_OUTPUT and prints why.

  python3 scripts/intel_nightly_gate.py      # in the gate job, on the schedule
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Callable

JOB = "Test (macos-15-intel)"
WORKFLOW = "ci.yml"
API = "https://api.github.com"


def decide(runs: list[dict], jobs_of: Callable[[int], list[dict]], current_run: int) -> tuple[bool, str]:
    """(run the leg, reason) from HEAD's ci.yml runs and a function giving a run's jobs."""
    looked = 0
    pending = False
    for r in runs:
        if r.get("id") == current_run:
            continue
        for job in jobs_of(r["id"]):
            if job.get("name") != JOB:
                continue
            looked += 1
            if job.get("status") == "completed" and job.get("conclusion") == "success":
                return False, f"run {r['id']} already passed {JOB} on this commit"
            if job.get("status") != "completed":
                pending = True
    if pending:
        return False, f"{JOB} is still running for this commit; leaving it to finish"
    if looked == 0:
        return True, f"no {JOB} job was found for this commit"
    return True, f"{JOB} did not succeed in any of the {looked} runs of this commit"


def fetch(path: str, token: str) -> dict:
    req = urllib.request.Request(
        f"{API}{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main() -> int:
    repo = os.environ["GITHUB_REPOSITORY"]
    sha = os.environ["GITHUB_SHA"]
    current = int(os.environ.get("GITHUB_RUN_ID", "0"))
    token = os.environ.get("GITHUB_TOKEN", "")
    try:
        runs = fetch(f"/repos/{repo}/actions/workflows/{WORKFLOW}/runs?head_sha={sha}&per_page=50",
                     token)["workflow_runs"]
        run, why = decide(
            runs,
            lambda rid: fetch(f"/repos/{repo}/actions/runs/{rid}/jobs?per_page=100", token)["jobs"],
            current,
        )
    except Exception as e:  # noqa: BLE001 — any failure to read the API runs the leg
        run, why = True, f"could not read the CI runs of this commit ({e}); running the leg"
    print(f"{'run' if run else 'skip'}: {why}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"run={'true' if run else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
