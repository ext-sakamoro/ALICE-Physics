#!/usr/bin/env python3
"""Every job of every workflow has a `timeout-minutes` within the cap.

A job without one runs until GitHub's default of 360 minutes when it hangs and
holds a runner the whole time; the runners are shared by every workflow of the
account. The limits in .github/workflows/*.yml are set from measured run times
(about twice the 95th percentile); this check keeps every job carrying one.

Rules, per job under the top-level `jobs:` of each workflow file:

  * a job that runs steps needs a job-level `timeout-minutes` (indented one
    level below the job key; a step's own `timeout-minutes` does not count)
  * the value is an integer, or a `${{ }}` expression whose integer literals are
    the limits it picks from (`matrix.os == 'macos-15-intel' && 205 || 125`):
    there must be at least one, and every one must be in 1..CAP
  * a job that calls a reusable workflow (`uses:` at job level) cannot carry
    `timeout-minutes` (GitHub rejects it); it is counted, and the called
    workflow's own jobs are checked when it is a file of this repository
  * zero jobs found fails (the check compared nothing)

  python3 scripts/workflow_timeouts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP = 360
JOB_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s*(?:#.*)?$")
TIMEOUT_RE = re.compile(r"^    timeout-minutes:\s*(.+?)\s*(?:#.*)?$")
USES_RE = re.compile(r"^    uses:\s*(\S+)")


def jobs(text: str) -> list[tuple[str, int, list[str]]]:
    """`(name, line number, body lines)` of each job under the top-level `jobs:`."""
    lines = text.splitlines()
    out: list[tuple[str, int, list[str]]] = []
    in_jobs = False
    current: tuple[str, int, list[str]] | None = None
    for n, line in enumerate(lines, 1):
        if line and not line[0].isspace() and not line.startswith("#"):
            if current:
                out.append(current)
                current = None
            in_jobs = line.split("#")[0].strip() == "jobs:"
            continue
        if not in_jobs:
            continue
        m = JOB_RE.match(line)
        if m:
            if current:
                out.append(current)
            current = (m.group(1), n, [])
        elif current:
            current[2].append(line)
    if current:
        out.append(current)
    return out


def limits(value: str) -> list[int]:
    """The minute values a `timeout-minutes` can take."""
    if re.fullmatch(r"\d+", value):
        return [int(value)]
    m = re.fullmatch(r"\$\{\{(.*)\}\}", value)
    if not m:
        return []
    # the literals the expression picks from (strings like 'macos-15' are skipped)
    expr = re.sub(r"'[^']*'", "", m.group(1))
    return [int(x) for x in re.findall(r"\b\d+\b", expr)]


def check(root: Path = ROOT) -> tuple[list[str], int, int]:
    """(errors, jobs with a limit checked, reusable-workflow calls)."""
    errors: list[str] = []
    checked = calls = 0
    for wf in sorted((root / ".github" / "workflows").glob("*.y*ml")):
        rel = wf.relative_to(root).as_posix()
        for name, line, body in jobs(wf.read_text(encoding="utf-8")):
            uses = next((m.group(1) for b in body if (m := USES_RE.match(b))), None)
            if uses:
                calls += 1
                if uses.startswith("./") and not (root / uses).is_file():
                    errors.append(f"{rel}:{line}: job `{name}` calls {uses}, which does not exist")
                continue
            values = [m.group(1) for b in body if (m := TIMEOUT_RE.match(b))]
            if not values:
                errors.append(f"{rel}:{line}: job `{name}` has no job-level timeout-minutes")
                continue
            picks = limits(values[0])
            if not picks:
                errors.append(f"{rel}:{line}: job `{name}` timeout-minutes {values[0]!r} names no minutes")
            for v in picks:
                if not 1 <= v <= CAP:
                    errors.append(f"{rel}:{line}: job `{name}` timeout-minutes {v} is outside 1..{CAP}")
            checked += 1
    if checked + calls == 0:
        errors.append("no job found under .github/workflows (checked nothing)")
    return errors, checked, calls


def main() -> int:
    errors, checked, calls = check()
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: jobs {checked}, reusable-workflow calls {calls}, cap {CAP} min")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
