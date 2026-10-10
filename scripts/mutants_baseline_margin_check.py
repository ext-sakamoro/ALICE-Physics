#!/usr/bin/env python3
"""Warn (fail-soft) when cargo-mutants' baseline test phase is already close
to the derived --timeout budget.

scripts/mutants-baseline-timings.txt is a single sample from one CI runner,
not a statistical estimate (see mutants_baseline_budget.py's module doc): the
1.5x margin is headroom against normal runner-to-runner noise, not a
confidence interval. If a baseline's own measured test time already exceeds
60% of the derived budget, this runner is slower than when the table was
recorded, and a future run on a similarly slow runner could time out before
the margin would normally cover it.

Reads cargo-mutants' own log and looks for its "Unmutated baseline in Ns
build + Ms test" line. The log is coloured when CARGO_TERM_COLOR=always (CI
sets it workflow-wide): ANSI escapes are stripped with scripts/ansi.py before
parsing, not with a one-off sed over the raw bytes (that silently matched
nothing on a real CI run, 2026-10-10: budget 1325s, baseline test 274s --
well over 60% -- no warning and no notice, because the coloured line never
matched the plain-text pattern). A line found but not matching the expected
shape prints its own ::notice (never silently skipped).

  scripts/mutants_baseline_margin_check.py --log /tmp/mutants-run.log --budget 300
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ansi  # noqa: E402

BASELINE_RE = re.compile(r"Unmutated baseline in (\d+)s build \+ (\d+)s test")


def find_baseline_test_seconds(log_text: str) -> int | None:
    """The baseline's measured test-phase seconds, read from `log_text`
    (already stripped of ANSI escapes). None if the line isn't present at all
    (nothing to check -- cargo-mutants didn't reach the baseline, handled
    elsewhere). Raises ValueError if a line is present but doesn't match the
    expected shape, so a format change surfaces instead of silently skipping."""
    for line in log_text.splitlines():
        if "Unmutated baseline in" not in line:
            continue
        m = BASELINE_RE.search(line)
        if not m:
            raise ValueError(f"baseline line found but unparsable: {line!r}")
        return int(m.group(2))
    return None


def check(log_text: str, budget: float, fraction: float) -> str | None:
    """`log_text` is the RAW cargo-mutants log (ANSI escapes included if the
    environment coloured it); stripped here before parsing. Returns the
    warning message if the baseline test phase exceeds `fraction` of
    `budget`, else None. Raises ValueError (see find_baseline_test_seconds)
    if a baseline line is present but unparsable -- the caller turns that
    into a ::notice rather than swallowing it."""
    test_s = find_baseline_test_seconds(ansi.strip(log_text))
    if test_s is None:
        return None
    threshold = budget * fraction
    if test_s <= threshold:
        return None
    return (
        f"baseline test phase took {test_s}s, over {int(fraction * 100)}% "
        f"of the derived budget ({int(budget)}s) -- this runner may be "
        "slower than when scripts/mutants-baseline-timings.txt was recorded; "
        "consider re-running scripts/mutants_baseline_timings.py --regenerate"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True, help="path to cargo-mutants' captured stdout/stderr")
    ap.add_argument("--budget", type=float, required=True, help="the derived --timeout value, in seconds")
    ap.add_argument("--fraction", type=float, default=0.6)
    args = ap.parse_args(argv)

    log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    try:
        message = check(log_text, args.budget, args.fraction)
    except ValueError as e:
        print(f"::notice::{e}")
        return 0
    if message is not None:
        print(f"::warning::{message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
