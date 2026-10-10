#!/usr/bin/env python3
"""Derive the in-diff mutants baseline --timeout from measured per-target times.

A fixed --timeout (300s) does not scale with how many test targets a diff's
plan selects (scripts/mutants_in_diff_plan.py): on a day that touches several
heavy modules at once, the planned targets' combined wall-clock legitimately
exceeds any fixed small budget, and cargo-mutants reports the baseline itself
as an incomplete run (the real defect this script exists for: 2026-10-09
nightly run 37995588561, 0/68 mutants tested).

budget = sum(measured time of every selected target, "lib" always included) *
margin, capped at --hard-cap. A selected target absent from the timing table
(scripts/mutants-baseline-timings.txt, see mutants_baseline_timings.py) is
treated as the SLOWEST known target, not zero: an unmeasured new test must
not silently shrink the budget.

Concentration of several genuinely slow targets does not parallelize away
even with `cargo mutants --test-tool nextest` (measured 2026-10-10: nextest
on just the 5 hyperelastic-family targets alone still timed out at 300s) --
the sum-based budget here is the complement to using nextest, not a
replacement for it.

Prints the computed integer number of seconds to stdout (nothing else), so a
workflow step can do `--timeout $(scripts/mutants_baseline_budget.py ...)`.

  scripts/mutants_baseline_budget.py --targets lib,analytic_cubic_hyperelastic,...
  scripts/mutants_baseline_budget.py --targets-file /tmp/targets.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "scripts" / "mutants-baseline-timings.txt"


def read_table(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, secs = line.rpartition(" ")
        if not name:
            continue
        out[name] = float(secs)
    return out


def budget_seconds(
    targets: list[str], table: dict[str, float], margin: float, hard_cap: float
) -> float:
    if not targets:
        raise ValueError("no targets given (compared nothing)")
    if not table:
        raise ValueError("timing table is empty or missing (compared nothing)")
    unknown_cost = max(table.values())
    total = sum(table.get(t, unknown_cost) for t in targets)
    return min(total * margin, hard_cap)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", help="comma or space separated target names")
    ap.add_argument("--targets-file", help="file with one target name per line")
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--margin", type=float, default=1.5,
                     help="multiply the summed measured time by this (default 1.5)")
    ap.add_argument("--hard-cap", type=float, default=2700.0,
                     help="never exceed this many seconds (default 45m)")
    ap.add_argument("--min", type=float, default=60.0,
                     help="never go below this many seconds (default 60s, build/startup floor)")
    args = ap.parse_args(argv)

    targets: list[str] = ["lib"]
    if args.targets:
        targets += [t for t in args.targets.replace(",", " ").split() if t]
    if args.targets_file:
        targets += [
            line.strip() for line in Path(args.targets_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    targets = sorted(set(targets))

    table = read_table(Path(args.table))
    try:
        seconds = budget_seconds(targets, table, args.margin, args.hard_cap)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    seconds = max(seconds, args.min)
    print(int(round(seconds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
