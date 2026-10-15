#!/usr/bin/env python3
"""No bullet of CHANGELOG.md's [Unreleased] section is written twice.

A branch whose later commit rewrites a bullet that an earlier commit added
leaves both versions behind when a rebase conflict is resolved by keeping both
sides: the old bullet and the new one, which starts the same and differs by a
clause. Within [Unreleased] this check fails when two bullets start with the
same OPENING characters (every duplicate of that kind seen so far, in this file
and in released sections, shares at least its first 40 characters; a
similarity ratio was tried and rejected, since templated bullets such as
numbered validation batches score above 0.9 while being distinct).

Released sections are not judged (they are history), but their pairs are
counted and printed so a release cut can see them. A file without an
[Unreleased] heading fails (the check would compare nothing); an empty
[Unreleased], as right after a release cut, passes.

    python3 scripts/changelog_dup_check.py [CHANGELOG.md]
"""
from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENING = 40
HEADING = re.compile(r"(?m)^## \[(.+?)\].*$")


def sections(text: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """(name, [(line, bullet text)]) per `## [..]` section, top-level bullets only."""
    marks = list(HEADING.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        first = text.count("\n", 0, m.end()) + 2
        body = text[m.end():end].split("\n")[1:]
        bullets = [(first + k, line[2:].strip()) for k, line in enumerate(body) if line.startswith("- ")]
        out.append((m.group(1), bullets))
    return out


def pairs(bullets: list[tuple[int, str]]) -> list[str]:
    found = []
    for (la, a), (lb, b) in itertools.combinations(bullets, 2):
        if a[:OPENING] == b[:OPENING]:
            found.append(f"lines {la} and {lb} start the same: {a[:OPENING]!r}")
    return found


def check(text: str) -> tuple[list[str], int, list[str]]:
    """(problems in [Unreleased], bullets compared there, notes on released sections)."""
    secs = sections(text)
    unreleased = [b for name, b in secs if name.lower() == "unreleased"]
    if not unreleased:
        return ["no [Unreleased] section"], 0, []
    problems = pairs(unreleased[0])
    notes = [f"[{name}] {p}" for name, b in secs if name.lower() != "unreleased" for p in pairs(b)]
    return problems, len(unreleased[0]), notes


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "CHANGELOG.md"
    problems, compared, notes = check(path.read_text(encoding="utf-8"))
    for n in notes:
        print(f"note (released, not judged): {n}")
    for p in problems:
        print(f"error: CHANGELOG [Unreleased] {p}", file=sys.stderr)
    print(f"changelog_dup_check: {compared} bullets in [Unreleased], {len(problems)} problems, "
          f"{len(notes)} same-opening pairs in released sections (not judged)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
