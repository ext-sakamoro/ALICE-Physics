#!/usr/bin/env python3
"""Line-coverage ratchet: coverage may not fall below the recorded baseline.

Reads the table `cargo llvm-cov --summary-only` prints and compares it with
scripts/line-coverage-baseline.txt (one `<file> <lines> <missed>` row per
source file, plus a `TOTAL` row). It fails when

  * the total line coverage falls by more than TOTAL_SLACK percentage points, or
  * a file that had at least MIN_LINES lines falls by more than FILE_SLACK points
    (a regression in one module is not hidden by growth elsewhere), or
  * nothing was compared (no file row parsed, or no baseline row matched).

New files are reported, not failed; a removed file is dropped silently. When the
coverage has risen, `--write` records the new numbers (the ratchet only moves up
when someone commits the baseline).

  cargo llvm-cov --lib --summary-only > coverage-summary.txt
  (the weekly job measures `--lib --tests` against line-coverage-baseline-full.txt)
  python3 scripts/line_coverage_ratchet.py --summary coverage-summary.txt
  python3 scripts/line_coverage_ratchet.py --summary coverage-summary.txt --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "line-coverage-baseline.txt"
TOTAL_SLACK = 0.10
FILE_SLACK = 2.0
MIN_LINES = 50


def parse_summary(text: str) -> dict[str, tuple[int, int]]:
    """`{file: (lines, missed lines)}` from the llvm-cov summary table, `TOTAL` included."""
    rows: dict[str, tuple[int, int]] = {}
    for line in text.splitlines():
        f = line.split()
        # Filename Regions Missed Cover Functions Missed Executed Lines Missed Cover ...
        if len(f) < 10 or not f[1].isdigit() or not f[7].isdigit() or not f[8].isdigit():
            continue
        rows[f[0]] = (int(f[7]), int(f[8]))
    return rows


def parse_baseline(text: str) -> dict[str, tuple[int, int]]:
    rows = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) == 3 and not line.startswith("#"):
            rows[f[0]] = (int(f[1]), int(f[2]))
    return rows


def pct(lines: int, missed: int) -> float:
    return 100.0 * (lines - missed) / lines if lines else 100.0


def compare(now: dict[str, tuple[int, int]], base: dict[str, tuple[int, int]]) -> tuple[list[str], list[str], int]:
    """(errors, notes, files compared)"""
    errors, notes = [], []
    if "TOTAL" not in now:
        errors.append("the summary has no TOTAL row (compared nothing)")
    if "TOTAL" not in base:
        errors.append("the baseline has no TOTAL row (compared nothing)")
    compared = 0
    if "TOTAL" in now and "TOTAL" in base:
        a, b = pct(*base["TOTAL"]), pct(*now["TOTAL"])
        if b < a - TOTAL_SLACK:
            errors.append(f"total line coverage {b:.2f}% is below the baseline {a:.2f}% (slack {TOTAL_SLACK} pt)")
        elif b > a + TOTAL_SLACK:
            notes.append(f"total line coverage rose {a:.2f}% -> {b:.2f}%: record it with --write")
    for name, (lines, missed) in sorted(now.items()):
        if name == "TOTAL":
            continue
        if name not in base:
            notes.append(f"new file {name}: {pct(lines, missed):.2f}% of {lines} lines")
            continue
        compared += 1
        blines, bmissed = base[name]
        if blines < MIN_LINES:
            continue
        a, b = pct(blines, bmissed), pct(lines, missed)
        if b < a - FILE_SLACK:
            errors.append(f"{name}: line coverage {b:.2f}% is below the baseline {a:.2f}% (slack {FILE_SLACK} pt)")
    if compared == 0:
        errors.append("no file of the summary is in the baseline (compared nothing)")
    return errors, notes, compared


def baseline_text(now: dict[str, tuple[int, int]]) -> str:
    out = ["# line coverage of `cargo llvm-cov --summary-only` per file: <file> <lines> <missed lines>",
           "# written by scripts/line_coverage_ratchet.py --write; coverage may not fall below it"]
    out += [f"{name} {lines} {missed}" for name, (lines, missed) in sorted(now.items())]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summary", required=True, help="the output of `cargo llvm-cov --summary-only`")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--write", action="store_true", help="record the summary as the new baseline")
    args = ap.parse_args(argv)
    now = parse_summary(Path(args.summary).read_text(encoding="utf-8", errors="replace"))
    if args.write:
        if "TOTAL" not in now:
            print("error: the summary has no TOTAL row", file=sys.stderr)
            return 1
        Path(args.baseline).write_text(baseline_text(now), encoding="utf-8", newline="\n")
        print(f"wrote {len(now) - 1} files and TOTAL {pct(*now['TOTAL']):.2f}% to {args.baseline}")
        return 0
    bp = Path(args.baseline)
    base = parse_baseline(bp.read_text(encoding="utf-8")) if bp.is_file() else {}
    errors, notes, compared = compare(now, base)
    for n in notes:
        print(f"note: {n}")
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if "TOTAL" in now:
        print(f"compared: files {compared}, total {pct(*now['TOTAL']):.2f}%"
              + (f" (baseline {pct(*base['TOTAL']):.2f}%)" if "TOTAL" in base else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
