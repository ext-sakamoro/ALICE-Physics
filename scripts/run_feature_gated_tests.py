#!/usr/bin/env python3
"""Run the integration tests gated on features that `cargo test` and
`cargo test --features parallel` never enable, and fail if nothing ran.

Background: CI ran these features only with `--lib`, so a test file under
`#![cfg(feature = "replay")]` (or a `#[cfg(feature = "gpu-solver-bridge")]`
module inside a file) compiled to zero tests and stayed green. The file list
is derived from the source on every run, so a new gated file is picked up
without editing this script, ci.yml or preflight.sh.

Guards (each one turns the run red):
- no test file selected, or a listed feature selects no file
- a selected target executed zero tests (the gate did not open)
- the executed total is zero, or any test failed

Both ci.yml and scripts/preflight.sh call this script without arguments, so
their feature sets cannot drift apart.

run: python3 scripts/run_feature_gated_tests.py [--list]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Features that neither `cargo test` nor `cargo test --features parallel` enable.
FEATURES = ("neural", "replay", "analytics", "gpu-solver-bridge")

ROOT = Path(__file__).resolve().parent.parent
FEATURE_RE = re.compile(r'feature\s*=\s*"([A-Za-z0-9_-]+)"')
RUNNING_RE = re.compile(r"^\s*Running (?:tests[/\\])?(\S+?)\.rs\b")
RESULT_RE = re.compile(
    r"^test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored"
)


def select(tests_dir: Path, features=FEATURES) -> dict[str, set[str]]:
    """Map each test target (file stem) to the listed features it mentions."""
    picked: dict[str, set[str]] = {}
    for path in sorted(tests_dir.glob("*.rs")):
        text = path.read_text(encoding="utf-8")
        hit = {f for f in FEATURE_RE.findall(text) if f in features}
        if hit:
            picked[path.stem] = hit
    return picked


def parse(output: str) -> dict[str, tuple[int, int, int]]:
    """Per target: (passed, failed, ignored) from `cargo test` output."""
    counts: dict[str, tuple[int, int, int]] = {}
    current = None
    for line in output.splitlines():
        m = RUNNING_RE.match(line)
        if m:
            current = Path(m.group(1)).name
            continue
        m = RESULT_RE.match(line)
        if m and current is not None:
            counts[current] = tuple(int(g) for g in m.groups())
            current = None
    return counts


def check(picked: dict[str, set[str]], counts, features=FEATURES) -> list[str]:
    """Problems that make the run red (empty list = green)."""
    problems = []
    if not picked:
        problems.append("no test file mentions any of " + ", ".join(features))
    for f in features:
        if not any(f in fs for fs in picked.values()):
            problems.append(f"feature {f!r} selects no test file (stale list?)")
    total = 0
    for target in picked:
        if target not in counts:
            problems.append(f"{target}: no test result line (did it run?)")
            continue
        passed, failed, _ignored = counts[target]
        total += passed + failed
        if passed + failed == 0:
            problems.append(f"{target}: executed 0 tests (gate not opened)")
        if failed:
            problems.append(f"{target}: {failed} failed")
    if picked and total == 0:
        problems.append("executed 0 tests in total")
    return problems


def main(argv: list[str]) -> int:
    picked = select(ROOT / "tests")
    if "--list" in argv:
        for target, fs in picked.items():
            print(f"{target}: {', '.join(sorted(fs))}")
        return 0
    cmd = ["cargo", "test", "--no-fail-fast", "--features", ",".join(FEATURES)]
    for target in picked:
        cmd += ["--test", target]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    sys.stdout.write(proc.stdout)
    counts = parse(proc.stdout)
    problems = check(picked, counts)
    if proc.returncode != 0 and not problems:
        problems.append(f"cargo test exited {proc.returncode}")
    executed = sum(p + f for p, f, _ in counts.values())
    print(f"\nfeature-gated targets: {len(picked)}, executed tests: {executed}")
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
