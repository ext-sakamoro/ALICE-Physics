#!/usr/bin/env python3
"""Build every example in release and run each one, failing on a non-zero
exit, a timeout, or a run that executed no example at all.

Background: CI compiled the examples (clippy --all-targets) but never ran
them, so three examples kept asserting a behaviour that a library fix had
changed, and one ran for more than ten minutes, without any job turning red.

The example list comes from `cargo metadata` on every run, and the feature
set is the union of every example's `required-features`, so a new example
(or a new required feature) is picked up without editing this script,
ci.yml or preflight.sh. Both call this script without arguments, so their
argument sets cannot drift apart.

Guards (each one turns the run red):
- no example discovered, or fewer examples than `examples/*.rs` files
  (metadata did not see them)
- an example binary is missing after the build
- an example exits non-zero (an `assert!` panics with 101) or exceeds the
  per-example timeout
- the number of examples executed is zero or differs from the number
  discovered

run: python3 scripts/run_examples.py [--timeout SECONDS] [--list]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Per-example wall-clock limit. The slowest example takes about two minutes
# on a laptop core in release; the limit leaves headroom for a slower runner.
DEFAULT_TIMEOUT_S = 300.0


def discover(metadata: dict) -> list[tuple[str, list[str]]]:
    """(name, required-features) of every example target of the root package."""
    examples = []
    for pkg in metadata.get("packages", []):
        if Path(pkg["manifest_path"]).resolve().parent != ROOT:
            continue
        for target in pkg.get("targets", []):
            if "example" in target.get("kind", []):
                examples.append((target["name"], list(target.get("required-features", []))))
    return sorted(examples)


def feature_union(examples) -> list[str]:
    feats = set()
    for _, req in examples:
        feats.update(req)
    feats.discard("std")  # default feature
    return sorted(feats)


def run_one(binary: Path, timeout_s: float) -> tuple[str, float, str]:
    """(status, seconds, output tail); status is "ok", "exit <code>" or "timeout"."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [str(binary)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace")
        return "timeout", time.monotonic() - start, out[-2000:]
    elapsed = time.monotonic() - start
    out = proc.stdout.decode("utf-8", "replace")
    status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
    return status, elapsed, out[-2000:]


def run_all(binaries: dict[str, Path], timeout_s: float) -> tuple[list, list[str]]:
    """Run every binary; return (rows, problems). Empty problems = green."""
    rows = []
    problems = []
    if not binaries:
        problems.append("no example to run (0 executed)")
        return rows, problems
    executed = 0
    for name, binary in sorted(binaries.items()):
        if not binary.is_file():
            problems.append(f"{name}: binary {binary} missing after the build")
            continue
        status, elapsed, tail = run_one(binary, timeout_s)
        executed += 1
        rows.append((name, status, elapsed))
        print(f"{status:>8}  {elapsed:7.1f} s  {name}", flush=True)
        if status != "ok":
            problems.append(f"{name}: {status} after {elapsed:.1f} s\n{tail}")
    if executed == 0:
        problems.append("0 examples executed")
    elif executed != len(binaries):
        problems.append(f"executed {executed} of {len(binaries)} examples")
    return rows, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--list", action="store_true", help="print the examples and exit")
    args = ap.parse_args()

    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
    )
    examples = discover(meta)
    features = feature_union(examples)
    files = sorted((ROOT / "examples").glob("*.rs"))
    if args.list:
        for name, req in examples:
            print(name, ",".join(req))
        print(f"{len(examples)} examples, features: {','.join(features) or '(default)'}")
        return 0
    if not examples:
        print("run_examples: no example target discovered", file=sys.stderr)
        return 1
    if len(examples) < len(files):
        print(
            f"run_examples: {len(examples)} example targets but {len(files)} examples/*.rs files",
            file=sys.stderr,
        )
        return 1

    build = ["cargo", "build", "--release", "--examples"]
    if features:
        build += ["--features", ",".join(features)]
    print("+", " ".join(build), flush=True)
    subprocess.run(build, cwd=ROOT, check=True)

    target_dir = Path(meta.get("target_directory", ROOT / "target"))
    exe = ".exe" if os.name == "nt" else ""
    binaries = {
        name: target_dir / "release" / "examples" / f"{name}{exe}" for name, _ in examples
    }
    rows, problems = run_all(binaries, args.timeout)
    total = sum(r[2] for r in rows)
    print(f"\n{len(rows)} examples executed in {total:.1f} s (timeout {args.timeout:.0f} s each)")
    if problems:
        print("\nrun_examples: FAILED", file=sys.stderr)
        for p in problems:
            print("-", p, file=sys.stderr)
        return 1
    print("run_examples: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
