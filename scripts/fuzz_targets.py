#!/usr/bin/env python3
"""The fuzz targets of fuzz/Cargo.toml, for the Fuzz workflow's matrix.

Every `[[bin]]` of fuzz/Cargo.toml is a fuzz target. The workflow builds all of
them on every push and pull request (`build-all`), runs the CORE set on those
events, and runs every target on the nightly schedule and on a manual run, so a
target added to fuzz/Cargo.toml is built and run without editing the workflow.

Errors (exit 1): no `[[bin]]` at all, a target whose `path` does not exist, or a
CORE target that is not a `[[bin]]` (a renamed target would silently drop out).

Usage: `python3 scripts/fuzz_targets.py --check` prints the counts;
`python3 scripts/fuzz_targets.py --matrix EVENT` prints the JSON list of targets
to run for the GitHub event name EVENT.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = ["fuzz_step", "fuzz_collision", "fuzz_deterministic_roundtrip", "fuzz_step_parity"]
ALL_EVENTS = {"schedule", "workflow_dispatch"}


def targets(root: str) -> list[tuple[str, str]]:
    """`(name, path)` of every `[[bin]]` in fuzz/Cargo.toml, in file order."""
    text = open(os.path.join(root, "fuzz", "Cargo.toml"), encoding="utf-8").read()
    out = []
    for block in re.split(r"^\[\[bin\]\]\s*$", text, flags=re.M)[1:]:
        block = re.split(r"^\[", block, flags=re.M)[0]
        name = re.search(r'^name\s*=\s*"([^"]+)"', block, re.M)
        path = re.search(r'^path\s*=\s*"([^"]+)"', block, re.M)
        if name:
            out.append((name.group(1), path.group(1) if path else f"fuzz_targets/{name.group(1)}.rs"))
    return out


def check(root: str) -> list[str]:
    errors = []
    found = targets(root)
    if not found:
        errors.append("fuzz/Cargo.toml has no [[bin]] (no fuzz target to build or run)")
    names = {n for n, _ in found}
    for name, path in found:
        if not os.path.exists(os.path.join(root, "fuzz", path)):
            errors.append(f"fuzz target {name}: {path} does not exist")
    for name in CORE:
        if name not in names:
            errors.append(f"core fuzz target {name} is not a [[bin]] of fuzz/Cargo.toml")
    # a source file with no [[bin]] is neither built nor run: cargo fuzz only
    # knows the bins of fuzz/Cargo.toml
    listed = {os.path.normpath(path) for _, path in found}
    src = os.path.join(root, "fuzz", "fuzz_targets")
    for f in sorted(os.listdir(src)) if os.path.isdir(src) else []:
        rel = os.path.normpath(os.path.join("fuzz_targets", f))
        if f.endswith(".rs") and rel not in listed:
            errors.append(f"fuzz/{rel} has no [[bin]] in fuzz/Cargo.toml (never built or run)")
    return errors


def matrix(root: str, event: str) -> list[str]:
    names = [n for n, _ in targets(root)]
    return names if event in ALL_EVENTS else [n for n in names if n in CORE]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.path.dirname(HERE))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--matrix", metavar="EVENT")
    args = ap.parse_args(argv)
    errors = check(args.root)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if errors:
        return 1
    if args.matrix:
        print(json.dumps(matrix(args.root, args.matrix)))
    else:
        print(f"compared: fuzz targets {len(targets(args.root))}, core {len(CORE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
