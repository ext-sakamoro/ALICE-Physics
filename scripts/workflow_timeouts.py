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
  * the files are read with PyYAML (scripts/requirements-ci.txt, pinned by
    hash), so any valid layout is read (indent, quoted or spaced keys, flow
    style); without PyYAML the check fails

  python3 scripts/workflow_timeouts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP = 360
TOKEN_RE = re.compile(r"""["']?\b(runs-on|uses)\b["']?\s*:""")


def load_yaml():
    """PyYAML, installed from scripts/requirements-ci.txt (pinned by hash). No
    fallback: without the parser the check cannot see the jobs, so it fails."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            "error: PyYAML is required: python -m pip install --require-hashes "
            "-r scripts/requirements-ci.txt") from e
    return yaml


class Job:
    """A job under the top-level `jobs` key: its name, its value (a mapping, as
    parsed) and the lines it spans (1-based, inclusive)."""

    def __init__(self, name: str, value, start: int, end: int) -> None:
        self.name, self.value, self.start, self.end = name, value, start, end


def parse(text: str) -> tuple[list[Job], list[tuple[int, int]], list[str]]:
    """(jobs, line spans of block scalars, errors) of a workflow file, by PyYAML."""
    yaml = load_yaml()
    try:
        data = yaml.safe_load(text)
        root = yaml.compose(text)
    except yaml.YAMLError as e:
        return [], [], [f"not valid YAML: {e}".replace("\n", " ")]
    jobs: list[Job] = []
    blocks: list[tuple[int, int]] = []

    def walk(node) -> None:
        if isinstance(node, yaml.ScalarNode):
            if node.style in ("|", ">"):
                blocks.append((node.start_mark.line + 2, node.end_mark.line))
        elif isinstance(node, yaml.SequenceNode):
            for child in node.value:
                walk(child)
        elif isinstance(node, yaml.MappingNode):
            for k, v in node.value:
                walk(k)
                walk(v)

    if root is not None:
        walk(root)
    if not isinstance(data, dict) or not isinstance(root, yaml.MappingNode):
        return [], blocks, ["the file is not a mapping"]
    values = data.get("jobs")
    nodes = next((v for k, v in root.value if isinstance(k, yaml.ScalarNode) and k.value == "jobs"), None)
    if not isinstance(values, dict) or not isinstance(nodes, yaml.MappingNode):
        return [], blocks, []
    for k, v in nodes.value:
        end = v.end_mark.line + (0 if v.end_mark.column == 0 else 1)
        jobs.append(Job(str(k.value), values.get(k.value), k.start_mark.line + 1, max(end, k.start_mark.line + 1)))
    return jobs, blocks, []


def orphans(text: str, jobs: list[Job], blocks: list[tuple[int, int]]) -> list[int]:
    """Lines of a plain text scan holding a `runs-on:` / `uses:` token outside
    every parsed job and outside every block scalar: a job the parser did not
    see. Fail-closed: a quoted value containing `runs-on:` outside the jobs also
    counts (rename the text rather than loosen the check)."""
    spans = [(j.start, j.end) for j in jobs] + blocks
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        code = line.split(" #")[0] if not line.lstrip().startswith("#") else ""
        if TOKEN_RE.search(code) and not any(a <= n <= b for a, b in spans):
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
        found, blocks, parse_errors = parse(text)
        errors.extend(f"{rel}: {e}" for e in parse_errors)
        if not found:
            errors.append(f"{rel}: no job found (checked nothing in this file)")
        for line in orphans(text, found, blocks):
            errors.append(f"{rel}:{line}: a runs-on / uses key outside every job the check found")
        for job in found:
            if not isinstance(job.value, dict):
                errors.append(f"{rel}:{job.start}: job `{job.name}` is not a mapping")
                continue
            uses = job.value.get("uses")
            if uses:
                calls += 1
                if str(uses).startswith("./") and not (root / str(uses)).is_file():
                    errors.append(f"{rel}:{job.start}: job `{job.name}` calls {uses}, which does not exist")
                continue
            value = job.value.get("timeout-minutes")
            if value is None:
                errors.append(f"{rel}:{job.start}: job `{job.name}` has no job-level timeout-minutes")
                continue
            picks = [value] if isinstance(value, int) and not isinstance(value, bool) else limits(str(value))
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
