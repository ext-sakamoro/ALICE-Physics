#!/usr/bin/env python3
"""Count the tests of each test binary in a `cargo test` log, for
scripts/downstream_check.sh.

Every binary that cargo runs (a `Running <target>` section, and a `Doc-tests
<crate>` section when doctests are run) must pass at least one test: a binary
whose tests are all compiled out by `cfg` or filtered away would otherwise hide
behind the other binaries of the same step. The steps of downstream_check.sh
name their targets (`--lib`, `--test x`) so that each binary in a log is one
that is meant to run tests; doctests are not run there (they are tested in the
downstream repositories), and if a step does run them, the `Doc-tests` section
is held to the same rule.

  python3 scripts/downstream_test_counts.py LOG

prints `<binary>: <n> passed` per binary on stderr and the total on stdout;
the exit status is 1 when a binary passed 0 tests or the log has no binary.
"""

from __future__ import annotations

import re
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
HEADER_RE = re.compile(r"^\s*(?:Running\s+(.+?)(?:\s+\([^()]*\))?|Doc-tests\s+(\S+))\s*$")
RESULT_RE = re.compile(r"^test result: \w+\. (\d+) passed;")


def count(log: str) -> list[tuple[str, int]]:
    """(binary, passed) for each test binary in the log, in order."""
    out: list[tuple[str, int]] = []
    for line in ANSI_RE.sub("", log).splitlines():
        h = HEADER_RE.match(line)
        if h:
            out.append((h.group(1) or f"doctests {h.group(2)}", 0))
            continue
        r = RESULT_RE.match(line)
        if r and out:
            name, n = out[-1]
            out[-1] = (name, n + int(r.group(1)))
    return out


def check(log: str) -> tuple[int, list[str]]:
    """(total passed, errors)."""
    bins = count(log)
    errors = [f"{name} ran 0 tests" for name, n in bins if n == 0]
    if not bins:
        errors.append("no test binary ran")
    return sum(n for _, n in bins), errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} LOG", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8", errors="replace") as f:
        log = f.read()
    for name, n in count(log):
        print(f"  {name}: {n} passed", file=sys.stderr)
    total, errors = check(log)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(total)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
