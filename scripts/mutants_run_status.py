#!/usr/bin/env python3
"""What a cargo-mutants run measured, in one line: complete, killed or not measured.

Reads one `mutants.out` directory (its `mutants.json` plan and `outcomes.json`) and
says which of three things the run is:

  * killed: the run has no end time or fewer outcomes than planned mutants (it was
    stopped by its deadline or cancelled). The line carries how many of the planned
    mutants were tested, and the survivors seen before the stop are counted from the
    outcomes (cargo-mutants writes `missed.txt` only at the end, so a killed run's
    `missed.txt` reads 0).
  * not measured: more than `--build-timeout-bound` of the outcomes are timeouts in
    the build phase. Such a run measured its own build deadline, not the tests: a
    mutant whose build did not finish was not tested.
  * complete: every planned mutant has an outcome and the build timeouts are within
    the bound.

A run that planned no mutant, or has no plan / outcomes file, is an error: it compared
nothing. The line goes to stdout (and to `--summary`, the step summary, when given);
the exit code is 0 for complete, 3 for killed, 4 for not measured, 2 for an error.

    mutants_run_status.py mutants.out [--build-timeout-bound 0.10] [--summary "$GITHUB_STEP_SUMMARY"]

Author: Moroya Sakamoto
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# a normal run has no build timeout (2026-11-14 measurement: the parallel axis 0 of 164 outcomes);
# the default axis had 131 of 137 while its builds sat at the 240 s limit
BUILD_TIMEOUT_BOUND = 0.10

COMPLETE, KILLED, NOT_MEASURED, ERROR = 0, 3, 4, 2


def status(out: Path, bound: float = BUILD_TIMEOUT_BOUND) -> tuple[int, str]:
    root = out / "mutants.out" if (out / "mutants.out").is_dir() else out
    plan, outcomes = root / "mutants.json", root / "outcomes.json"
    if not plan.is_file() or not outcomes.is_file():
        return ERROR, f"no run: {root} has no mutants.json / outcomes.json (compared nothing)"
    try:
        planned = len(json.loads(plan.read_text(encoding="utf-8")))
        result = json.loads(outcomes.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return ERROR, f"no run: mutants.json / outcomes.json is not readable ({e})"
    if planned == 0:
        return ERROR, "no run: no mutant was planned (compared nothing)"
    done = [o for o in result.get("outcomes", []) if o.get("scenario") != "Baseline"]
    kinds = {k: 0 for k in ("CaughtMutant", "MissedMutant", "Unviable", "Timeout", "Success", "Failure")}
    build_timeouts = 0
    for o in done:
        kinds[o.get("summary", "Failure")] = kinds.get(o.get("summary", "Failure"), 0) + 1
        if o.get("summary") == "Timeout" and any(
                p.get("phase") == "Build" and p.get("process_status") == "Timeout" for p in o.get("phase_results", [])):
            build_timeouts += 1
    counts = (f"caught {kinds['CaughtMutant']} / missed {kinds['MissedMutant']} / unviable {kinds['Unviable']}"
              f" / timeout {kinds['Timeout']} (build {build_timeouts})")
    if result.get("end_time") is None or len(done) < planned:
        return KILLED, (f"killed ({len(done)}/{planned} tested): the run stopped before every planned mutant had an "
                        f"outcome; seen before the stop: {counts}")
    if build_timeouts > bound * len(done):
        return NOT_MEASURED, (f"not measured ({build_timeouts} of {len(done)} outcomes are build timeouts, over "
                              f"{bound:.0%}): the run measured its build deadline, not the tests; {counts}")
    return COMPLETE, f"complete ({len(done)}/{planned} tested): {counts}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="the mutants.out directory (or its parent)")
    ap.add_argument("--build-timeout-bound", type=float, default=BUILD_TIMEOUT_BOUND,
                    help="largest share of outcomes that may be build timeouts in a measured run")
    ap.add_argument("--summary", help="append the line to this file (the step summary)")
    args = ap.parse_args(argv)
    code, line = status(Path(args.out), args.build_timeout_bound)
    print(line)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(f"- run: {line}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
