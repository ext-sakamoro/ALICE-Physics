#!/usr/bin/env python3
"""The determinism tests, optimised: every filter term and every test binary runs a test.

The sign of a zero result can differ between an unoptimised and an optimised build on a
target with a fused multiply-add instruction (the optimiser may rewrite a negated fused
multiply-add), so the determinism tests also run with `--release`. A filter that matches
nothing passes with "0 passed", so this check counts what ran:

  - each library filter term runs on its own (`cargo test --release --lib -- <term>`) and
    must pass at least one test: with the terms in one invocation, a term that stopped
    matching (a renamed test) drops out while the other term keeps the count above zero;
  - the integration test binaries run in one invocation, and each of them must report a
    result with at least one passed test.

usage: determinism_release.py
Exit 1 when a term or a binary ran no test, a run failed, or a result is missing.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB_TERMS = ("max_min_tests", "windward_alignment")
TEST_BINARIES = (
    "determinism_golden",
    "determinism_golden_f32",
    "determinism_golden_contacts",
    "determinism_golden_paths",
    "determinism_zero_tie",
    # the Law outputs (a golden per src file calling the Law max / min / clamp)
    "determinism_law_outputs",
    # the semantics, step digests and golden coverage (the rest of the determinism suite)
    "determinism_semantic",
    "determinism_physics2d_step_digest",
    "determinism_cloth_step_digest",
    "determinism_golden_coverage",
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RESULT_RE = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed", re.M)


def results(text: str) -> list[tuple[str, int]]:
    """(ok | FAILED, passed) for each `test result` line, in order"""
    return [(m.group(1), int(m.group(2))) for m in RESULT_RE.finditer(ANSI_RE.sub("", text))]


def judge(label: str, text: str, rc: int, expected: int) -> list[str]:
    """errors for one run that should report `expected` test results, each passing ≥ 1"""
    errors = []
    found = results(text)
    if rc != 0:
        errors.append(f"{label}: cargo test exited {rc}")
    if len(found) != expected:
        errors.append(f"{label}: {len(found)} test results, expected {expected}")
    for i, (status, passed) in enumerate(found):
        if status != "ok" or passed == 0:
            errors.append(f"{label}: result {i + 1} is `{status}. {passed} passed` (a filter or binary ran no test)")
    return errors


def run(args: list[str], log: Path) -> tuple[str, int]:
    env = dict(os.environ, CARGO_TERM_COLOR="never")
    proc = subprocess.run(["cargo", "test", "--release", *args], cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    log.write_text(proc.stdout, encoding="utf-8")
    sys.stdout.write(proc.stdout)
    return proc.stdout, proc.returncode


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")
    out_dir = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for term in LIB_TERMS:
        text, rc = run(["--lib", "--", term], out_dir / f"determinism-release-lib-{term}.log")
        errors += judge(f"--lib -- {term}", text, rc, 1)
    args = [a for name in TEST_BINARIES for a in ("--test", name)]
    text, rc = run(args, out_dir / "determinism-release-tests.log")
    errors += judge("integration tests", text, rc, len(TEST_BINARIES))
    print(f"determinism (release): {len(LIB_TERMS)} library filter terms, {len(TEST_BINARIES)} test binaries, errors {len(errors)}")
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
