#!/usr/bin/env python3
"""Group the mutated files of an in-diff run by the integration tests that execute them.

The in-diff run passes one set of integration test targets (the union the plan selects,
scripts/mutants_in_diff_plan.py) to every mutant, whatever file the mutant is in. A mutant
can only be killed by a test that executes its file, so for each file the run needs only
the union targets whose coverage includes that file. This script reads a coverage map
(per test target: the source files it executes, and which files have instrumented code at
all) and prints the groups as JSON:

  * each group is `{"files": [...], "tests": [...], "fallback": reason-or-null}`;
  * `tests` = the union targets that execute at least one line of every file in the group
    (files with the same covering set share a group, so one build serves them);
  * a file falls back to the whole union, never to fewer tests, when the map cannot speak
    for it: no union target executes it (`not covered`), it has no instrumented line in any
    target (`compile-time only`: const evaluation, macro-only modules), or it is not a
    `src/` Rust file (`not src`: build.rs, a manifest);
  * `--lib` is not part of the selection: the run always passes it.

The map must be complete: every union target has an entry (a target missing from the map
is an error, not a fallback, so a lost coverage job cannot shrink a run silently), and a map
with no target at all is an error.

Map format (JSON): {"targets": {"<target>": {"covered": [files], "instrumented": [files]}}}
with paths relative to the repository root.

  python3 scripts/mutants_select_groups.py --map covmap.json --union tests.txt --files files.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NOT_COVERED = "not covered"
COMPILE_TIME = "compile-time only"
NOT_SRC = "not src"


class MapError(ValueError):
    """The coverage map cannot be used for this union (a fallback would hide a lost job)."""


def buckets(union: list[str], seconds: dict[str, float], k: int) -> list[list[str]]:
    """Split the union targets into k coverage jobs of about equal time (longest first, each
    target to the lightest bucket). A target with no measured time counts as the median of the
    measured ones, so a new test file is still placed; every target is in exactly one bucket."""
    if k < 1:
        raise ValueError("k must be at least 1")
    known = sorted(seconds[t] for t in union if t in seconds)
    default = known[len(known) // 2] if known else 1.0
    cost = {t: seconds.get(t, default) for t in union}
    bins: list[tuple[float, int, list[str]]] = [(0.0, i, []) for i in range(min(k, len(union)) or 1)]
    for t in sorted(set(union), key=lambda t: (-cost[t], t)):
        load, i, members = min(bins)
        bins.remove((load, i, members))
        members.append(t)
        bins.append((load + cost[t], i, members))
    return [sorted(m) for _, _, m in sorted(bins, key=lambda b: b[1]) if m]


def merge(parts: list[dict], union: list[str]) -> dict:
    """Join the maps of the coverage jobs into one. A target in two parts, or a union target in
    none, is an error: a lost or doubled job must not change which tests run."""
    merged: dict[str, dict] = {}
    for part in parts:
        for t, entry in part.get("targets", {}).items():
            if t in merged:
                raise MapError(f"target {t} is in two coverage parts")
            merged[t] = entry
    missing = sorted(set(union) - set(merged))
    if not merged or missing:
        raise MapError(f"{len(missing)} of {len(set(union))} union targets have no coverage part"
                       + (f" (first: {missing[0]})" if missing else ""))
    return {"targets": merged}


def groups(files: list[str], union: list[str], cmap: dict) -> list[dict]:
    targets = cmap.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise MapError("the coverage map has no target")
    missing = sorted(t for t in union if t not in targets)
    if missing:
        raise MapError(f"{len(missing)} of {len(union)} union targets are not in the coverage map "
                       f"(first: {missing[0]})")
    covered_by: dict[str, set[str]] = {}
    instrumented: set[str] = set()
    for t in union:
        for f in targets[t].get("covered", []):
            covered_by.setdefault(f, set()).add(t)
    # whether a file has instrumented code is a property of the file, so every target of the map speaks for it
    for entry in targets.values():
        instrumented.update(entry.get("instrumented", []))
        instrumented.update(entry.get("covered", []))
    selected: dict[tuple[str, ...], list[str]] = {}
    fallback: dict[str, list[str]] = {}
    for f in sorted(set(files)):
        if not (f.startswith("src/") and f.endswith(".rs")):
            fallback.setdefault(NOT_SRC, []).append(f)
        elif f not in instrumented:
            fallback.setdefault(COMPILE_TIME, []).append(f)
        elif not covered_by.get(f):
            fallback.setdefault(NOT_COVERED, []).append(f)
        else:
            selected.setdefault(tuple(sorted(covered_by[f])), []).append(f)
    out = [{"files": fs, "tests": list(ts), "fallback": None} for ts, fs in sorted(selected.items(), key=lambda kv: kv[1])]
    whole = sorted(union)
    out += [{"files": fs, "tests": whole, "fallback": why} for why, fs in sorted(fallback.items())]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True, help="coverage map JSON")
    ap.add_argument("--union", required=True, help="file with the union test targets, one per line")
    ap.add_argument("--files", required=True, help="file with the mutated files, one per line")
    args = ap.parse_args(argv)
    union = [t for t in Path(args.union).read_text(encoding="utf-8").split() if t]
    files = [f for f in Path(args.files).read_text(encoding="utf-8").split() if f]
    if not union:
        print("error: the union has no test target", file=sys.stderr)
        return 1
    if not files:
        print("error: no mutated file", file=sys.stderr)
        return 1
    try:
        gs = groups(files, union, json.loads(Path(args.map).read_text(encoding="utf-8")))
    except MapError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"groups": gs}))
    fb = sum(len(g["files"]) for g in gs if g["fallback"])
    print(f"note: {len(files)} file(s) in {len(gs)} group(s); {fb} file(s) run the whole union "
          f"({len(union)} targets)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
