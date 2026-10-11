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
coverage has risen, `--write` records the new numbers. A write never lowers an
entry: for every file and the total it keeps the better of the baseline and the
summary (the slack absorbs noise in the check, never in the write). Lowering one
takes `--allow-worse <file> --reason "<12 characters or more>"`, and the reason
is written into the baseline as `# allow-worse <file>: <reason>` above the entry.

`--diff-against <ref>` checks a baseline edited by hand: every entry that is
lower than in the baseline at git revision `<ref>` (CI: `origin/main`) has to
carry such a comment.

  cargo llvm-cov --lib --summary-only > coverage-summary.txt
  (the weekly job measures `--lib --tests` against line-coverage-baseline-full.txt)
  python3 scripts/line_coverage_ratchet.py --summary coverage-summary.txt
  python3 scripts/line_coverage_ratchet.py --summary coverage-summary.txt --write
  python3 scripts/line_coverage_ratchet.py --diff-against origin/main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ratchet_merge as rm

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


def lower(old: tuple[int, int], new: tuple[int, int]) -> bool:
    """`new` covers a smaller fraction of its lines than `old`."""
    return pct(*new) < pct(*old)


def baseline_text(now: dict[str, tuple[int, int]], reasons: dict[str, str] | None = None) -> str:
    out = ["# line coverage of `cargo llvm-cov --summary-only` per file: <file> <lines> <missed lines>",
           "# written by scripts/line_coverage_ratchet.py --write; coverage may not fall below it"]
    for name, (lines, missed) in sorted(now.items()):
        if reasons and name in reasons:
            out.append(f"# allow-worse {name}: {reasons[name]}")
        out.append(f"{name} {lines} {missed}")
    return "\n".join(out) + "\n"


def write(now: dict[str, tuple[int, int]], path: Path, allowed: dict[str, str]) -> int:
    """Record `now` in `path`, keeping the old value of every entry it would lower."""
    if "TOTAL" not in now:
        print("error: the summary has no TOTAL row", file=sys.stderr)
        return 1
    old_text = path.read_text(encoding="utf-8") if path.is_file() else ""
    old = parse_baseline(old_text)
    unknown = sorted(k for k in allowed if k not in old or k not in now)
    if unknown:
        print(f"error: --allow-worse names entries that are not in both the baseline and the summary: {unknown}",
              file=sys.stderr)
        return 1
    merged, kept, used = rm.merge_floors(old, now, lower, allowed)
    unused = sorted(k for k in allowed if k not in used)
    if unused:
        print(f"error: --allow-worse names entries that did not get lower: {unused}", file=sys.stderr)
        return 1
    # an earlier allowance stays beside its entry while the entry keeps that value
    reasons = {k: r for k, r in rm.allow_comments(old_text).items() if k in merged and merged[k] == old.get(k)}
    reasons.update(used)
    path.write_text(baseline_text(merged, reasons), encoding="utf-8", newline="\n")
    for k in kept:
        print(f"note: kept the baseline of {k} ({pct(*old[k]):.2f}% > measured {pct(*now[k]):.2f}%)")
    for k, r in sorted(used.items()):
        print(f"note: lowered {k} to {pct(*now[k]):.2f}% ({r})")
    print(f"wrote {len(merged) - 1} files and TOTAL {pct(*merged['TOTAL']):.2f}% to {path} "
          f"(kept {len(kept)}, lowered {len(used)})")
    return 0


def diff_against(path: Path, ref: str) -> int:
    """A hand edit may not lower an entry against `ref` without an allow-worse comment."""
    main_text = rm.text_at(ref, path)
    if main_text is None:
        print(f"error: {path} does not exist at {ref} (compared nothing)", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    errors, compared = rm.diff_check(parse_baseline(main_text), parse_baseline(text), lower,
                                     rm.allow_comments(text))
    if compared == 0:
        errors.append(f"no entry of {path} is in the baseline at {ref} (compared nothing)")
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: {compared} entries of {path.name} against {ref}")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summary", help="the output of `cargo llvm-cov --summary-only`")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--write", action="store_true", help="record the summary, never lowering an entry")
    ap.add_argument("--allow-worse", action="append", default=[], metavar="FILE",
                    help="with --write: lower this entry to the measured value (needs a --reason)")
    ap.add_argument("--reason", action="append", default=[], help="why the matching --allow-worse is lower")
    ap.add_argument("--diff-against", metavar="REF",
                    help="check the baseline file against the one at git revision REF")
    args = ap.parse_args(argv)
    if args.diff_against:
        return diff_against(Path(args.baseline), args.diff_against)
    if not args.summary:
        ap.error("--summary is required unless --diff-against is given")
    allowed, problems = rm.allowances(args.allow_worse, args.reason)
    if problems or (allowed and not args.write):
        for p in problems or ["--allow-worse only applies to --write"]:
            print(f"error: {p}", file=sys.stderr)
        return 1
    now = parse_summary(Path(args.summary).read_text(encoding="utf-8", errors="replace"))
    if args.write:
        return write(now, Path(args.baseline), allowed)
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
