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
  * a workflow file in which no job is found fails, file by file (the check
    compared nothing there), and so does a `runs-on:` key, or a `uses:` key
    that is not a step's, outside every job found: the parser missed a job
  * keys may be quoted, spaced (`jobs :`) or flow-style
    (`build: {runs-on: x, timeout-minutes: 10}`); the job indent is taken
    from the first job, the job-level key indent from each job's first key

  python3 scripts/workflow_timeouts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP = 360
KEY_RE = re.compile(
    r"""^(?P<indent>[ ]*)(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<plain>[A-Za-z0-9_.$-][^:#'"]*?))"""
    r"""[ ]*:(?:[ ]+(?P<value>.*?))?[ ]*$""")
FLOW_KEY_RE = re.compile(r"""(?:^|[{,])\s*["']?([A-Za-z0-9_-]+)["']?\s*:\s*([^,}]*)""")


class Job:
    """A job under the top-level `jobs:` key: its name, the lines it spans
    (1-based, inclusive) and its job-level keys (block or flow style)."""

    def __init__(self, name: str, start: int) -> None:
        self.name, self.start, self.end = name, start, start
        self.keys: dict[str, str] = {}
        self.key_indent: int | None = None


def strip_comment(line: str) -> str:
    """The line without a trailing `# comment` outside quotes."""
    out, quote = [], ""
    for k, ch in enumerate(line):
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (k == 0 or line[k - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def key_of(line: str) -> tuple[int, str, str] | None:
    """(indent, key, value) of a mapping line, or None."""
    m = KEY_RE.match(line)
    if not m:
        return None
    key = m["dq"] if m["dq"] is not None else m["sq"] if m["sq"] is not None else m["plain"]
    return len(m["indent"]), key.strip(), (m["value"] or "").strip()


def parse(text: str) -> tuple[list[Job], list[str]]:
    """The jobs of a workflow file, and the parse errors (lines the parser could
    not place: an indent below the jobs, or a flow-style job left open)."""
    jobs: list[Job] = []
    errors: list[str] = []
    in_jobs, child_indent, job = False, None, None
    flow: list[str] | None = None
    for n, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw)
        if not line.strip():
            continue
        if flow is not None and job is not None:
            flow.append(line)
            job.end = n
            if "".join(flow).count("{") <= "".join(flow).count("}"):
                job.keys.update((k, v.strip()) for k, v in FLOW_KEY_RE.findall(" ".join(flow)))
                flow = None
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            kv = key_of(line)
            in_jobs = kv is not None and kv[1] == "jobs"
            child_indent, job = None, None
            continue
        if not in_jobs:
            continue
        if child_indent is None:
            child_indent = indent
        if indent < child_indent:
            errors.append(f"line {n}: indented less than the jobs above it")
            continue
        kv = key_of(line)
        if indent == child_indent:
            if kv is None:
                errors.append(f"line {n}: a job key was expected")
                continue
            job = Job(kv[1], n)
            jobs.append(job)
            if kv[2].startswith("{"):
                flow = [kv[2]]
                if kv[2].count("{") <= kv[2].count("}"):
                    job.keys.update((k, v.strip()) for k, v in FLOW_KEY_RE.findall(kv[2]))
                    flow = None
            continue
        if job is None:
            continue
        job.end = n
        if job.key_indent is None:
            job.key_indent = indent
        if indent == job.key_indent and kv is not None:
            job.keys[kv[1]] = kv[2]
    if flow is not None:
        errors.append("a flow-style job is not closed")
    return jobs, errors


def orphans(text: str, jobs: list[Job]) -> list[int]:
    """Lines with a `runs-on:` key, or a `uses:` key that is not a step's, outside
    every parsed job: a job the parser did not see."""
    spans = [(j.start, j.end) for j in jobs]
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw)
        kv = key_of(line)
        flow_runs_on = re.search(r"""[{,]\s*["']?runs-on["']?\s*:""", line)
        if (kv and kv[1] in ("runs-on", "uses")) or flow_runs_on:
            if not any(a <= n <= b for a, b in spans):
                out.append(n)
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
        text = wf.read_text(encoding="utf-8")
        found, parse_errors = parse(text)
        errors.extend(f"{rel}: {e}" for e in parse_errors)
        if not found:
            errors.append(f"{rel}: no job found (checked nothing in this file)")
        for line in orphans(text, found):
            errors.append(f"{rel}:{line}: a runs-on / uses key outside every job the check found")
        for job in found:
            uses = job.keys.get("uses")
            if uses:
                calls += 1
                if uses.startswith("./") and not (root / uses).is_file():
                    errors.append(f"{rel}:{job.start}: job `{job.name}` calls {uses}, which does not exist")
                continue
            value = job.keys.get("timeout-minutes")
            if value is None:
                errors.append(f"{rel}:{job.start}: job `{job.name}` has no job-level timeout-minutes")
                continue
            picks = limits(value)
            if not picks:
                errors.append(f"{rel}:{job.start}: job `{job.name}` timeout-minutes {value!r} names no minutes")
            for v in picks:
                if not 1 <= v <= CAP:
                    errors.append(f"{rel}:{job.start}: job `{job.name}` timeout-minutes {v} is outside 1..{CAP}")
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
