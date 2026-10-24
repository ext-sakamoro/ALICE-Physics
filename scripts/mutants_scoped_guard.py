#!/usr/bin/env python3
"""Whether a scoped cargo-mutants run measured anything.

    python3 scripts/mutants_scoped_guard.py mutants.out

The scoped job of quality-deep.yml mutates a few functions and runs the tests
its `test_args` input selects (default `--lib`). When those tests never reach
the mutated code, every viable mutant is "missed" and the run looks like a
weak score instead of a run that compared nothing (2026-10-24: 72 of 76
missed, because the oracles of the scope were an integration test and the run
used `--lib`). This fails the job when:

- there is no outcomes.json, or no unmutated baseline in it, or the baseline
  did not pass;
- the baseline log reports 0 tests run (the selected test target is empty);
- no mutant was viable (caught + missed + timeout = 0);
- no mutant was caught while some were missed (the selected tests do not reach
  the scope: pick the test target that holds its oracles).

A run with caught mutants and some missed ones is a measurement and passes;
the missed ones are in the job summary.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PASSED = re.compile(r"^test result: \w+\. (\d+) passed", re.M)


def check(out: Path) -> tuple[list[str], str]:
    problems: list[str] = []
    path = out / "outcomes.json"
    if not path.is_file():
        return [f"{path}: missing (cargo-mutants wrote no outcomes)"], ""
    doc = json.loads(path.read_text(encoding="utf-8"))
    caught, missed = int(doc.get("caught", 0)), int(doc.get("missed", 0))
    timeout, unviable = int(doc.get("timeout", 0)), int(doc.get("unviable", 0))
    base = [o for o in doc.get("outcomes", []) if o.get("scenario") == "Baseline"]
    tests = 0
    if not base:
        problems.append("no unmutated baseline in outcomes.json: nothing to compare the mutants with")
    else:
        if base[0].get("summary") != "Success":
            problems.append(f"the unmutated baseline did not pass ({base[0].get('summary')})")
        log = out / base[0].get("log_path", "log/baseline.log")
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        tests = sum(int(n) for n in PASSED.findall(text))
        if tests == 0:
            problems.append("the unmutated baseline ran 0 tests: the selected test target is empty")
    viable = caught + missed + timeout
    if viable == 0:
        problems.append(f"no viable mutant ({unviable} unviable): the run compared nothing")
    elif caught == 0 and missed > 0:
        problems.append(f"no mutant was caught ({missed} missed, {timeout} timeout): "
                        "the selected tests do not reach the scope; choose the test target that holds its oracles")
    report = (f"mutants_scoped_guard: baseline ran {tests} tests; caught {caught} / missed {missed} / "
              f"timeout {timeout} / unviable {unviable}")
    return problems, report


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    problems, report = check(Path(argv[0]))
    if report:
        print(report)
    for p in problems:
        print(f"::error::mutants_scoped_guard: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
