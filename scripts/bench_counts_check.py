#!/usr/bin/env python3
"""Fail unless every gungraun benchmark in the output measured a non-zero
instruction count. Callgrind reports 0 when it found nothing to measure (for
example, stripped symbols), and a comparison of 0 with 0 reads as "no change",
so an empty measurement would otherwise pass the bench gate.

  python3 scripts/bench_counts_check.py bench.txt
"""

from __future__ import annotations

import re
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
COUNT_RE = re.compile(r"^\s*Instructions:\s*([0-9][0-9,]*)\|")


def counts(text: str) -> list[int]:
    out = []
    for line in ANSI_RE.sub("", text).splitlines():
        m = COUNT_RE.match(line)
        if m:
            out.append(int(m.group(1).replace(",", "")))
    return out


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8", errors="replace") as f:
        cs = counts(f.read())
    if not cs:
        print("error: no `Instructions:` line in the output (compared nothing)", file=sys.stderr)
        return 1
    zero = sum(1 for c in cs if c == 0)
    if zero:
        print(f"error: {zero} of {len(cs)} benchmarks measured 0 instructions (Callgrind collected nothing)",
              file=sys.stderr)
        return 1
    print(f"benchmarks measured: {len(cs)}, instructions {min(cs)}..{max(cs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
