#!/usr/bin/env python3
"""Predict a Test job's time before the push, from what the last run on main took.

The Test jobs of ci.yml run every test binary twice (the default pass and the
parallel pass) inside one time limit. A new or changed test binary that takes
minutes there is found only when the job is cut at its limit, two hours later.
This script answers before the push:

  ledger   `scripts/test-time-ledger.tsv`, written from a finished CI run on main
           (`--write-ledger RUN`): per Test job its limit, its total time and, per
           test binary, the seconds of each pass ("finished in" of the job's log).
  check    `check BASE`: the test binaries the commits BASE..HEAD add or change are
           timed here (`cargo test --profile ci-test --test <name>`), together with
           one calibration binary of the ledger; the ratio runner / here of that
           binary, per job, scales the local times. Projected job time = the
           ledger's total + (scaled new time - the ledger's time of that binary)
           over its passes. A job whose projection is above its limit minus MARGIN
           minutes is refused, with the binaries that cost most.

The ratio is measured each time from the calibration binary (the machine's load
changes), not written down. Nothing compared is an error: no job read from the
ledger, no changed binary timed when some changed, or a calibration time of 0.

Usage:
  python3 scripts/test_time_budget.py check BASE [--timings FILE]
  python3 scripts/test_time_budget.py --write-ledger RUN_ID [--repo OWNER/NAME]
`--timings FILE` reads local times (`<binary>\t<seconds>` rows, the calibration
binary included) instead of running cargo: the CI's own tests of this script and a
caller that has already timed the binaries use it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

LEDGER = os.path.join("scripts", "test-time-ledger.tsv")
# minutes kept free under a job's limit: the same code's job time moves by several
# minutes from run to run (34 to 39 minutes for one pass on the slowest runner)
MARGIN = 15
# the calibration binary: in every Test job, tens of seconds, no I/O
CALIBRATION = "tests/trimesh_raycast_bruteforce.rs"
# the ledger keeps the binaries that take at least this many seconds in a pass
KEEP = 2.0
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RUNNING = re.compile(r"Running (unittests )?(\S+)")
RESULT = re.compile(r"test result: \w+\. \d+ passed.*finished in ([\d.]+)s")


class BudgetError(Exception):
    """A reason the budget cannot be read or is exceeded."""


def binary_times(log: str) -> dict[str, list[float]]:
    """Seconds of each pass of each test binary in a Test job's log."""
    out: dict[str, list[float]] = {}
    current = None
    for raw in log.splitlines():
        line = ANSI.sub("", raw)
        m = RUNNING.search(line)
        if m:
            # the Windows job prints its paths with backslashes
            current = ("unit " if m.group(1) else "") + m.group(2).replace("\\", "/")
            continue
        m = RESULT.search(line)
        if m and current:
            out.setdefault(current, []).append(float(m.group(1)))
            current = None
    return out


def format_ledger(sha: str, jobs: list[dict]) -> str:
    """The ledger's text: `job` rows, then `bin` rows."""
    lines = [
        "# test-time ledger: what the Test jobs of ci.yml took on main (scripts/test_time_budget.py --write-ledger)",
        f"# sha\t{sha}",
        "# job\t<name>\t<limit minutes>\t<total seconds>",
        "# bin\t<job>\t<binary>\t<seconds of each pass, comma separated>",
    ]
    for job in jobs:
        lines.append(f"job\t{job['name']}\t{job['limit']}\t{job['total']:.0f}")
    for job in jobs:
        for name in sorted(job["bins"]):
            # binaries under KEEP seconds a pass are left out (a binary without a row counts
            # as 0 s before, which errs on the side of refusing)
            if name != CALIBRATION and max(job["bins"][name]) < KEEP:
                continue
            passes = ",".join(f"{s:.2f}" for s in job["bins"][name])
            lines.append(f"bin\t{job['name']}\t{name}\t{passes}")
    return "\n".join(lines) + "\n"


def parse_ledger(text: str) -> dict[str, dict]:
    """job name -> {limit (minutes), total (seconds), bins {binary: [seconds]}}."""
    jobs: dict[str, dict] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        cells = line.split("\t")
        if cells[0] == "job" and len(cells) == 4:
            jobs[cells[1]] = {"limit": float(cells[2]), "total": float(cells[3]), "bins": {}}
        elif cells[0] == "bin" and len(cells) == 4:
            if cells[1] not in jobs:
                raise BudgetError(f"ledger: a bin row of {cells[1]!r} before its job row")
            jobs[cells[1]]["bins"][cells[2]] = [float(s) for s in cells[3].split(",") if s]
        else:
            raise BudgetError(f"ledger: unreadable row {line!r}")
    if not jobs:
        raise BudgetError("ledger: no job row (nothing to compare with)")
    return jobs


def project(
    ledger: dict[str, dict], local: dict[str, float], changed: list[str], margin: float = MARGIN
) -> tuple[list[str], list[str]]:
    """(report lines, refusals) for the changed binaries timed in `local`."""
    if CALIBRATION not in local or local[CALIBRATION] <= 0:
        raise BudgetError(f"no local time of the calibration binary {CALIBRATION}")
    missing = [b for b in changed if b not in local]
    if missing:
        raise BudgetError(f"changed test binaries not timed: {', '.join(missing)}")
    report: list[str] = []
    refusals: list[str] = []
    compared = 0
    for name, job in ledger.items():
        runner = job["bins"].get(CALIBRATION)
        if not runner:
            raise BudgetError(f"ledger: job {name!r} has no time of {CALIBRATION}")
        ratio = (sum(runner) / len(runner)) / local[CALIBRATION]
        passes = len(runner)
        delta = 0.0
        rows = []
        for binary in changed:
            scaled = local[binary] * ratio * passes
            before = sum(job["bins"].get(binary, []))
            delta += scaled - before
            rows.append((scaled - before, binary, before, scaled))
            compared += 1
        projected = (job["total"] + delta) / 60.0
        budget = job["limit"] - margin
        report.append(
            f"{name}: {job['total'] / 60.0:.0f} min on main, {delta / 60.0:+.1f} min from "
            f"{len(changed)} changed binaries (ratio {ratio:.2f}, {passes} passes) = "
            f"{projected:.0f} min of {job['limit']:.0f} (budget {budget:.0f})"
        )
        if projected > budget:
            worst = sorted(rows, reverse=True)[:5]
            cost = "; ".join(f"{b}: {before:.0f} s -> {after:.0f} s" for _, b, before, after in worst)
            refusals.append(f"{name}: projected {projected:.0f} min is over {budget:.0f} min ({cost})")
    if changed and compared == 0:
        raise BudgetError("no job compared although test binaries changed")
    return report, refusals


def changed_binaries(base: str) -> list[str]:
    """tests/*.rs the commits base..HEAD add or change (not deletions)."""
    r = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AMR", f"{base}...HEAD", "--", "tests"],
        capture_output=True, text=True, check=True,
    )
    return sorted(p for p in r.stdout.split() if re.fullmatch(r"tests/[^/]+\.rs", p))


def time_locally(binaries: list[str]) -> dict[str, float]:
    """Seconds of each binary's run here (ci-test profile, as the Test jobs)."""
    out: dict[str, float] = {}
    for path in binaries:
        name = os.path.splitext(os.path.basename(path))[0]
        r = subprocess.run(
            ["cargo", "test", "--profile", "ci-test", "--test", name],
            capture_output=True, text=True, check=False,
        )
        times = binary_times(r.stdout + r.stderr)
        got = [sum(v) for k, v in times.items() if k.endswith(f"{name}.rs")]
        if r.returncode != 0 or not got:
            raise BudgetError(f"{path}: the test run failed or reported no time (exit {r.returncode})")
        out[path] = got[0]
    return out


def read_timings(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                name, seconds = line.rstrip("\n").split("\t")
                out[name] = float(seconds)
    return out


def write_ledger(run: str, repo: str | None) -> int:
    base = ["gh"] + (["--repo", repo] if repo else [])
    view = subprocess.run(
        ["gh", "run", "view", run] + (["--repo", repo] if repo else [])
        + ["--json", "headSha,jobs"], capture_output=True, text=True, check=True,
    )
    data = json.loads(view.stdout)
    slug = repo or subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    limits = {"macos-15-intel": 360, "windows-latest": 160}
    jobs = []
    for job in data["jobs"]:
        m = re.fullmatch(r"Test \(((?:macos|ubuntu|windows)[^)]*)\)", job["name"])
        if not m or job.get("conclusion") != "success":
            continue
        log = subprocess.run(
            base[:1] + ["api", f"repos/{slug}/actions/jobs/{job['databaseId']}/logs"],
            capture_output=True, text=True, check=True,
        ).stdout
        bins = binary_times(log)
        if CALIBRATION not in bins:
            raise BudgetError(f"{job['name']}: the log has no time of {CALIBRATION}")
        started = _seconds(job["startedAt"])
        ended = _seconds(job["completedAt"])
        jobs.append({
            "name": job["name"], "limit": limits.get(m.group(1), 120),
            "total": ended - started, "bins": bins,
        })
    if not jobs:
        raise BudgetError(f"run {run}: no successful Test job (nothing to write)")
    with open(LEDGER, "w", encoding="utf-8") as f:
        f.write(format_ledger(data["headSha"], jobs))
    print(f"test_time_budget: ledger written from run {run}: {len(jobs)} jobs, "
          f"{sum(len(j['bins']) for j in jobs)} binaries read, those of {KEEP:.0f} s or more kept")
    return 0


def _seconds(stamp: str) -> float:
    import datetime
    return datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", nargs="?", choices=["check"])
    ap.add_argument("base", nargs="?")
    ap.add_argument("--timings")
    ap.add_argument("--write-ledger", metavar="RUN_ID")
    ap.add_argument("--repo")
    args = ap.parse_args(argv)
    try:
        if args.write_ledger:
            return write_ledger(args.write_ledger, args.repo)
        if args.command != "check" or not args.base:
            ap.error("check BASE, or --write-ledger RUN_ID")
        with open(LEDGER, encoding="utf-8") as f:
            ledger = parse_ledger(f.read())
        changed = changed_binaries(args.base)
        if not changed:
            print(f"test_time_budget: 0 test binaries changed since {args.base}; "
                  f"{len(ledger)} jobs in the ledger, nothing to project")
            return 0
        local = read_timings(args.timings) if args.timings else time_locally(changed + [CALIBRATION])
        report, refusals = project(ledger, local, changed)
        for line in report:
            print(f"test_time_budget: {line}")
        for line in refusals:
            print(f"error: {line}", file=sys.stderr)
        return 1 if refusals else 0
    except (BudgetError, subprocess.CalledProcessError, OSError) as e:
        print(f"error: test_time_budget: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
