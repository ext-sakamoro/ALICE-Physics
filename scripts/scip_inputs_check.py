#!/usr/bin/env python3
"""Every file the SCIP index's readers open is an input scripts/land.py knows about.

    python3 scripts/scip_inputs_check.py --scip <index dir> [--patterns scripts/land-scip-inputs.txt]

scripts/land.py skips the SCIP index when a landing's range touches none of the patterns of
scripts/land-scip-inputs.txt. That is only sound if the patterns cover every file the readers of
the index depend on. This runs each reader (the checks land.py runs with --scip, and the
integration-status writer into a temporary file) in its own interpreter under an audit hook,
collects the files of the repository each one opens, and fails (exit 1) when:
  - a file one of them opened is matched by no pattern (add it to the patterns file);
  - a reader opened no file of the repository at all (it read nothing: the check compared nothing);
  - the patterns file cannot be read or holds no pattern.
Byte-compiled caches (__pycache__) and .git are not inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import changed_paths as cp  # noqa: E402

TRACER = r"""
import json, os, runpy, sys
root, out, script, *args = sys.argv[1:]
root = os.path.realpath(root)
seen = set()
def hook(event, a):
    if event == "open" and a and isinstance(a[0], (str, bytes, os.PathLike)):
        p = os.path.realpath(os.fsdecode(a[0]))
        if p.startswith(root + os.sep):
            seen.add(os.path.relpath(p, root).replace(os.sep, "/"))
sys.addaudithook(hook)
sys.argv = [script, *args]
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
with open(out, "w", encoding="utf-8") as f:
    json.dump({"exit": code, "read": sorted(seen)}, f)
"""


def readers(scip: str, tmp: str) -> list[list[str]]:
    """the readers of the index, as land.py and CI run them (the writer into a temporary file)"""
    return [["scripts/integration_levels.py", "--check", "--scip", scip],
            ["scripts/scip_reach.py", "--check-baseline", "--scip", scip],
            ["scripts/scip_reach.py", "--write", str(Path(tmp) / "integration-status.md"), "--scip", scip]]


def trace(root: Path, argv: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "trace.json"
        subprocess.run([sys.executable, "-c", TRACER, str(root), str(out), *argv], cwd=root,
                       capture_output=True, text=True)
        if not out.exists():
            return {"exit": None, "read": []}
        return json.loads(out.read_text(encoding="utf-8"))


def is_input(path: str) -> bool:
    parts = path.split("/")
    return "__pycache__" not in parts and parts[0] != ".git"


def problems(read_by: dict[str, list[str]], patterns: list[str]) -> list[str]:
    """why the patterns do not cover what the readers read (empty: they do)"""
    out = []
    if not patterns:
        return ["the patterns file holds no pattern"]
    rules = cp.compile_patterns(patterns)
    for reader, files in read_by.items():
        files = [f for f in files if is_input(f)]
        if not files:
            out.append(f"{reader}: opened no file of the repository (compared nothing)")
        for f in files:
            if not cp.selected(f, rules):
                out.append(f"{reader}: reads {f}, which no pattern of the inputs file matches")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scip", required=True)
    ap.add_argument("--patterns", default="scripts/land-scip-inputs.txt")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    try:
        patterns = cp.read_patterns(str(root / a.patterns), [])
    except (OSError, UnicodeDecodeError) as e:
        print(f"scip_inputs_check: {a.patterns} cannot be read: {e}", file=sys.stderr)
        return 1
    read_by: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for argv_ in readers(a.scip, tmp):
            t = trace(root, argv_)
            read_by[" ".join(argv_[:2])] = t["read"]
    errs = problems(read_by, patterns)
    for e in errs:
        print(f"scip_inputs_check: {e}", file=sys.stderr)
    n = sum(len([f for f in v if is_input(f)]) for v in read_by.values())
    print(f"scip_inputs_check: {len(read_by)} readers, {n} files read, {len(patterns)} patterns, {len(errs)} problems")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
