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
- nothing was measured: no mutant was caught, missed or timed out in its test
  phase (a timeout in the build phase, `--build-timeout`, measured nothing);
- no mutant was caught (only missed ones, or only test-phase timeouts): the
  selected tests do not reach the scope; pick the test target that holds its
  oracles;
- more mutants timed out building than were measured: the run measured too
  little of the scope to stand for it (2026-10-24: 72 of 76 mutants of
  `maxwell_fdtd` hit the build timeout, 1 was measured);
- outcomes.json is one directory deeper than the directory given
  (`cargo mutants --output X` writes `X/mutants.out/`): the job reads the
  wrong directory and names the right one.

The counts come from each mutant's outcome, not from the totals, since the
totals do not say in which phase a timeout happened. A run with caught
mutants and some missed ones is a measurement and passes; the missed ones are
in the job summary.
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
        nested = out / "mutants.out" / "outcomes.json"
        if nested.is_file():
            return [f"{path}: missing, but {nested} exists: cargo-mutants was given --output {out} "
                    f"(it writes <output>/mutants.out); give it the parent directory"], ""
        return [f"{path}: missing (cargo-mutants wrote no outcomes)"], ""
    doc = json.loads(path.read_text(encoding="utf-8"))
    outcomes = doc.get("outcomes", [])
    caught = missed = unviable = test_timeout = build_timeout = 0
    for o in outcomes:
        if o.get("scenario") == "Baseline":
            continue
        summary = o.get("summary")
        phases = o.get("phase_results") or []
        last = phases[-1].get("phase") if phases else None
        if summary == "CaughtMutant":
            caught += 1
        elif summary == "MissedMutant":
            missed += 1
        elif summary == "Unviable":
            unviable += 1
        elif summary == "Timeout":
            if last == "Test":
                test_timeout += 1
            else:
                build_timeout += 1
    base = [o for o in outcomes if o.get("scenario") == "Baseline"]
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
    measured = caught + missed + test_timeout
    if measured == 0:
        problems.append(f"no mutant was measured ({unviable} unviable, {build_timeout} build timeouts): "
                        "the run compared nothing")
    elif caught == 0:
        problems.append(f"no mutant was caught ({missed} missed, {test_timeout} test timeouts): "
                        "the selected tests do not reach the scope; choose the test target that holds its oracles")
    if measured and build_timeout > measured:
        problems.append(f"{build_timeout} mutants timed out building and {measured} were measured: "
                        "the run measured too little of the scope (raise the build timeout or narrow the scope)")
    report = (f"mutants_scoped_guard: baseline ran {tests} tests; caught {caught} / missed {missed} / "
              f"test timeouts {test_timeout} / {build_timeout} build timeouts / unviable {unviable}")
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
