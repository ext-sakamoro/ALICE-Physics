#!/usr/bin/env python3
"""Read a `cargo semver-checks` log and print how many checks it ran.

The semver step must fail when the tool compared nothing (it failed to build
rustdoc, or printed no `Checked [..] N checks` line), and must say so instead
of stopping silently. cargo colours its status words when CARGO_TERM_COLOR is
`always` (the workflow sets it at the top level), which puts an escape code
between `Checked` and the timing, so the log is stripped of ANSI codes first.

Usage: semver_checked.py <log> [--exit-code N]
Prints the number of checks and exits 0, or prints a `::error::` line and exits 1.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ansi import ANSI_RE as ANSI  # noqa: E402  (CSI, charset selectors, OSC)
CHECKED = re.compile(r"\bChecked\s+\[[^\]]*\]\s+(\d+)\s+checks?\b")


def checked_count(log: str) -> int | None:
    """The last `Checked [..] N checks` count in `log`, or None when there is none."""
    found = CHECKED.findall(ANSI.sub("", log))
    return int(found[-1]) if found else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--exit-code", type=int, default=0, help="exit code of cargo semver-checks")
    args = ap.parse_args(argv)
    n = checked_count(Path(args.log).read_text(encoding="utf-8", errors="replace"))
    if not n:
        print(
            f"::error::cargo semver-checks compared nothing (exit {args.exit_code}): "
            f"no `Checked [..] N checks` line with N > 0 in {args.log}; see the log above"
        )
        return 1
    print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
