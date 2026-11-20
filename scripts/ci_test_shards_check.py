#!/usr/bin/env python3
"""The test job of ci.yml runs every step once per runner, whether or not the runner is sharded.

    python3 scripts/ci_test_shards_check.py [path/to/ci.yml]

On Linux and Windows the test job's matrix value is `<runner>, <shard>` (four shards per
runner); a macOS value is the runner alone and runs every step in one job. Each step says
where it runs with an `if:` over `matrix.os`. This reads the matrix of every event (with and
without the macos input of a manual run) and every step's condition, and fails (exit 1) unless:

  - every sharded runner has exactly the four shards (checks, default, parallel, features) in
    every event that runs it;
  - every step that is not shared runs in exactly one shard of each sharded runner, and on
    every unsharded value; a step named for one value (`matrix.os == '<runner>, <shard>'`)
    runs there and nowhere else;
  - a shared step (checkout, Python, the Rust toolchain, the cargo cache) runs in at least one
    shard of each runner and on every unsharded value;
  - every shard runs at least one step that is not shared;
  - something was compared (no step or no sharded runner read is a failure).

A condition this file cannot read (anything but `!contains(matrix.os, ', ')`,
`endsWith(matrix.os, '<x>')`, `!endsWith(matrix.os, '<x>')` and `matrix.os == '<x>'` joined by
`||`) is a failure, not a guess.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import changed_paths as cp  # noqa: E402
import ci_triggers_check as ct  # noqa: E402

SHARDS = ("checks", "default", "parallel", "features")
SEP = ", "
# shared setup: run in more than one shard (each shard is its own runner)
SHARED = ("actions/checkout", "actions/setup-python", "Install Rust toolchain", "Swatinem/rust-cache")
GATE_OUTPUT = "inputs.macos"

TERM = re.compile(r"""^\s*(!?)\s*(contains|endsWith)\(\s*matrix\.os\s*,\s*'([^']*)'\s*\)\s*$|"""
                  r"""^\s*matrix\.os\s*==\s*'([^']*)'\s*$""")


class CondError(ValueError):
    pass


def runs(cond: str | None, value: str) -> bool:
    """whether a step with this `if:` runs for this matrix value"""
    if cond is None:
        return True
    c = cond.strip()
    if c.startswith("${{") and c.endswith("}}"):
        c = c[3:-2]
    result = False
    for term in c.split("||"):
        m = TERM.match(term)
        if not m:
            raise CondError(f"cannot read the condition {cond!r}")
        if m.group(4) is not None:
            hit = value == m.group(4)
        else:
            neg, fn, lit = m.group(1), m.group(2), m.group(3)
            hit = (lit in value) if fn == "contains" else value.endswith(lit)
            hit = not hit if neg else hit
        result = result or hit
    return result


def step_label(s: dict) -> str:
    return str(s.get("name") or s.get("uses") or s.get("run", "")[:40])


def shared(s: dict) -> bool:
    return any(str(s.get(k, "")).startswith(x) for x in SHARED for k in ("uses", "name"))


def named_value(cond: str | None) -> str | None:
    """the one value a step is pinned to (`matrix.os == '<x>'`), if any"""
    m = re.search(r"matrix\.os\s*==\s*'([^']*)'", cond or "")
    return m.group(1) if m else None


def matrix_values(text: str) -> list[str]:
    job = ct.jobs(text)["test"]
    vals: list[str] = []
    for out in (None, GATE_OUTPUT):
        for _label, ev, ref, _key in ct.EVENTS:
            for v in ct.matrix_values(job, "os", ct.event_ctx(ev, ref, out)) or []:
                if v not in vals:
                    vals.append(v)
    return vals


def per_event_shards(text: str) -> list[str]:
    """events whose matrix has a sharded runner without all four shards"""
    job = ct.jobs(text)["test"]
    bad = []
    for out in (None, GATE_OUTPUT):
        for label, ev, ref, _key in ct.EVENTS:
            vals = ct.matrix_values(job, "os", ct.event_ctx(ev, ref, out)) or []
            runners = {v.split(SEP)[0] for v in vals if SEP in v}
            for r in sorted(runners):
                got = sorted(v.split(SEP, 1)[1] for v in vals if v.startswith(r + SEP))
                if got != sorted(SHARDS):
                    bad.append(f"{label}{' (landing scripts)' if out else ''}: {r} has shards {got}, not {list(SHARDS)}")
    return bad


def check(text: str) -> tuple[list[str], int]:
    """(errors, number of (step, runner) placements compared)"""
    tree = cp.load_yaml(text)
    steps = tree["jobs"]["test"].get("steps") or []
    values = matrix_values(text)
    sharded = sorted({v.split(SEP)[0] for v in values if SEP in v})
    whole = [v for v in values if SEP not in v]
    errors = per_event_shards(text)
    if not steps:
        return ["the test job has no steps"], 0
    if not sharded:
        return errors + ["no sharded runner in the test job's matrix"], 0
    compared = 0
    own = {(r, k): 0 for r in sharded for k in SHARDS}
    for s in steps:
        label, cond = step_label(s), s.get("if")
        try:
            pinned = named_value(cond)
            for r in sharded:
                where = [k for k in SHARDS if runs(cond, f"{r}{SEP}{k}")]
                compared += 1
                if shared(s):
                    if not where:
                        errors.append(f"{label}: shared, but runs in no shard of {r}")
                    continue
                want = 1 if pinned is None or pinned.split(SEP)[0] == r else 0
                if len(where) != want:
                    errors.append(f"{label}: runs in {len(where)} shards of {r} ({where}), not {want}")
                for k in where:
                    own[(r, k)] += 1
            if pinned is not None and pinned not in values:
                errors.append(f"{label}: pinned to {pinned!r}, which no event's matrix has")
            for v in whole:
                compared += 1
                if pinned is None and not runs(cond, v):
                    errors.append(f"{label}: does not run on {v} (an unsharded runner runs every step)")
                if pinned is not None and runs(cond, v) != (v == pinned):
                    errors.append(f"{label}: pinned to {pinned} but {'runs' if runs(cond, v) else 'does not run'} on {v}")
        except CondError as e:
            errors.append(f"{label}: {e}")
    for (r, k), n in own.items():
        if n == 0:
            errors.append(f"shard {r}{SEP}{k} runs no step of its own")
    return errors, compared


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    errors, compared = check(path.read_text(encoding="utf-8"))
    for e in errors:
        print(f"ci_test_shards_check: {e}", file=sys.stderr)
    if compared == 0 and not errors:
        print("ci_test_shards_check: compared nothing", file=sys.stderr)
        return 1
    print(f"ci_test_shards_check: {path.name}: {compared} step placements compared, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
