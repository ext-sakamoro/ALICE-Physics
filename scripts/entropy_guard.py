#!/usr/bin/env python3
"""No deterministic path of the crate draws OS entropy or reads the wall clock.

The crate's results are a function of its inputs (bit-for-bit on every
platform), so production code in src/ must not call an OS entropy source or a
wall clock. Randomness that a caller wants (privacy noise, for example) comes
from a key or seed the caller passes in, or from the `alice-crypto`
constructors re-exported by `privacy` (`SecureRng::from_entropy`,
`DpNoise::try_from_entropy`), which the caller chooses to call: the crate
itself never calls them.

Checked in every src/**/*.rs file, outside `#[cfg(test)]` modules:

    getrandom  OsRng  thread_rng  rand::random  from_entropy(  try_from_entropy(
    SystemTime::now  Instant::now

A definition (`fn from_entropy`) is not a call. ALLOWED lists the call sites
that stay, each with its reason and exact count: a count that drifts either way
fails, so a new call cannot hide behind an old entry and a removed one must be
removed here too. Scanning no file, or no pattern, fails.

    python3 scripts/entropy_guard.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "getrandom": re.compile(r"\bgetrandom\b"),
    "OsRng": re.compile(r"\bOsRng\b"),
    "thread_rng": re.compile(r"\bthread_rng\b"),
    "rand::random": re.compile(r"\brand::random\b"),
    "from_entropy(": re.compile(r"(?<!fn )\b(?:try_)?from_entropy\s*\("),
    "SystemTime::now": re.compile(r"\bSystemTime::now\b"),
    "Instant::now": re.compile(r"\bInstant::now\b"),
}
# (file, pattern) -> (count, reason)
ALLOWED: dict[tuple[str, str], tuple[int, str]] = {
    ("src/privacy.rs", "SystemTime::now"): (
        1, "deprecated XorShift64::from_entropy seeds from the clock; documented as not private",
    ),
    ("src/privacy.rs", "from_entropy("): (
        4, "deprecated LaplaceNoise / RandomizedResponse / Rappor constructors call that clock seed",
    ),
}
STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')


def production_lines(text: str) -> list[tuple[int, str]]:
    """(line number, code) outside `#[cfg(test)]` modules, strings and `//` comments removed."""
    out = []
    skip_depth = None  # brace depth at which the test module started
    pending_test = False
    depth = 0
    for n, raw in enumerate(text.splitlines(), 1):
        code = STRING_RE.sub('""', raw).split("//")[0]
        stripped = code.strip()
        if skip_depth is None and stripped.startswith("#[cfg(test)]"):
            pending_test = True
            continue
        opens, closes = code.count("{"), code.count("}")
        if pending_test and stripped:
            if re.match(r"(pub(\([^)]*\))?\s+)?mod\s+\w+\s*\{", stripped) and opens > closes:
                skip_depth = depth
            pending_test = False
        depth += opens - closes
        if skip_depth is not None:
            if depth <= skip_depth:
                skip_depth = None
            continue
        out.append((n, code))
    return out


def check(root: Path) -> tuple[list[str], int, int]:
    problems: list[str] = []
    counts: dict[tuple[str, str], list[int]] = {}
    files = sorted((root / "src").rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        for n, code in production_lines(path.read_text(encoding="utf-8")):
            for name, rx in PATTERNS.items():
                if rx.search(code):
                    counts.setdefault((rel, name), []).append(n)
    for key, lines in sorted(counts.items()):
        allowed = ALLOWED.get(key)
        where = ", ".join(f"{key[0]}:{n}" for n in lines)
        if allowed is None:
            problems.append(f"{key[1]} in production code: {where}")
        elif allowed[0] != len(lines):
            problems.append(f"{key[1]} in {key[0]}: {len(lines)} call sites, ALLOWED says {allowed[0]} ({where})")
    for key, (count, _) in sorted(ALLOWED.items()):
        if key not in counts:
            problems.append(f"ALLOWED entry {key[1]} in {key[0]} ({count}) matches nothing: remove it")
    return problems, len(files), len(PATTERNS)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    problems, files, patterns = check(ROOT)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if files == 0 or patterns == 0:
        print("error: compared nothing (no src file or no pattern)", file=sys.stderr)
        return 1
    print(f"entropy_guard: {files} files, {patterns} patterns, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
