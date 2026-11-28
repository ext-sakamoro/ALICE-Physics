#!/usr/bin/env python3
"""Build the coverage map the in-diff mutation run selects tests from (scripts/mutants_select_groups.py).

The map says, for each integration test target of the run's union, which source files it executes
and which files have instrumented code at all. It is built with cargo-llvm-cov, one target at a
time, in K coverage jobs that run side by side:

  buckets  split the union targets into K jobs of about equal time (from a table of measured times)
  run      one job: for each target of its bucket, clean the profiles, run the target instrumented,
           and read the per-file summary; writes that bucket's part of the map
  join     merge the parts into one map; a target in two parts, a union target in none, or a target
           whose own test run failed is an error (a lost job never shrinks the selection silently)

  python3 scripts/mutants_covmap.py buckets --union tests.txt --times times.tsv --k 8 --out buckets.json
  python3 scripts/mutants_covmap.py run --buckets buckets.json --bucket 3 --features parallel --out part-3.json
  python3 scripts/mutants_covmap.py join --union tests.txt --parts parts/ --out covmap.json

A target named `__lib__` stands for the library's unit tests (`--lib`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_select_groups as sg  # noqa: E402

LIB = "__lib__"


def read_times(text: str) -> dict[str, float]:
    """A tab-separated table `target<TAB>seconds[<TAB>...]`; lines that do not parse are skipped."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) >= 2:
            try:
                out[cols[0]] = float(cols[1])
            except ValueError:
                continue
    return out


def files_of(summary: dict, root: str) -> dict[str, list[str]]:
    """The covered and the instrumented source files of one `cargo llvm-cov report --json
    --summary-only` output, as paths relative to `root` (files outside it are dropped)."""
    prefix = root.rstrip("/") + "/"
    covered, instrumented = [], []
    for entry in summary.get("data", [{}])[0].get("files", []):
        name = entry.get("filename", "")
        if not name.startswith(prefix):
            continue
        rel = name[len(prefix):]
        lines = entry.get("summary", {}).get("lines", {})
        if lines.get("count", 0) > 0:
            instrumented.append(rel)
        if lines.get("covered", 0) > 0:
            covered.append(rel)
    return {"covered": sorted(covered), "instrumented": sorted(instrumented)}


def target_args(target: str) -> list[str]:
    return ["--lib"] if target == LIB else ["--test", target]


def run_target(target: str, features: str, root: Path) -> dict:
    base = ["cargo", "llvm-cov"]
    feat = ["--features", features] if features else []
    subprocess.run(base + ["clean", "--profraw-only"], cwd=root, check=True, capture_output=True)
    test = subprocess.run(base + ["--no-report", "--locked", "--profile", "ci-test", *feat, *target_args(target)],
                          cwd=root, capture_output=True, text=True)
    if test.returncode != 0:
        tail = "\n".join((test.stdout + test.stderr).splitlines()[-20:])
        raise RuntimeError(f"target {target}: its instrumented test run failed (exit {test.returncode}):\n{tail}")
    report = subprocess.run(base + ["report", "--profile", "ci-test", "--json", "--summary-only"],
                            cwd=root, check=True, capture_output=True, text=True)
    return files_of(json.loads(report.stdout), str(root))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("buckets")
    b.add_argument("--union", required=True)
    b.add_argument("--times", help="tab-separated measured times (target, seconds); missing targets count as the median")
    b.add_argument("--k", type=int, required=True)
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--buckets", required=True)
    r.add_argument("--bucket", type=int, required=True)
    r.add_argument("--features", default="")
    r.add_argument("--root", default=str(HERE.parent))
    r.add_argument("--out", required=True)
    j = sub.add_parser("join")
    j.add_argument("--union", required=True)
    j.add_argument("--parts", required=True, help="directory of the parts (*.json)")
    j.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "buckets":
        union = Path(args.union).read_text(encoding="utf-8").split()
        if not union:
            print("error: the union has no test target", file=sys.stderr)
            return 1
        times = read_times(Path(args.times).read_text(encoding="utf-8")) if args.times else {}
        bs = sg.buckets(union, times, args.k)
        Path(args.out).write_text(json.dumps({"buckets": bs}), encoding="utf-8")
        print(f"note: {len(union)} target(s) in {len(bs)} bucket(s); {sum(t in times for t in union)} with a measured time",
              file=sys.stderr)
        return 0

    if args.cmd == "run":
        bs = json.loads(Path(args.buckets).read_text(encoding="utf-8"))["buckets"]
        if not 0 <= args.bucket < len(bs):
            print(f"error: bucket {args.bucket} is not one of the {len(bs)} buckets", file=sys.stderr)
            return 1
        mine = bs[args.bucket]
        if not mine:
            print(f"error: bucket {args.bucket} is empty", file=sys.stderr)
            return 1
        part: dict[str, dict] = {}
        for t in mine:
            try:
                part[t] = run_target(t, args.features, Path(args.root))
            except RuntimeError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(f"note: {t}: {len(part[t]['covered'])} file(s) executed", file=sys.stderr)
        Path(args.out).write_text(json.dumps({"targets": part}), encoding="utf-8")
        return 0

    union = Path(args.union).read_text(encoding="utf-8").split()
    parts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.parts).glob("*.json"))]
    try:
        merged = sg.merge(parts, union)
    except sg.MapError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    Path(args.out).write_text(json.dumps(merged), encoding="utf-8")
    print(f"note: {len(parts)} part(s), {len(merged['targets'])} target(s) mapped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
