#!/usr/bin/env python3
"""No deterministic path of the crate draws OS entropy or reads the wall clock.

The crate's results are a function of its inputs (bit-for-bit on every
platform), so production code in src/ must not call an OS entropy source or a
wall clock. Randomness that a caller wants (privacy noise, for example) comes
from a key or seed the caller passes in, or from the `alice-crypto`
constructors re-exported by `privacy` (`SecureRng::from_entropy`,
`DpNoise::try_from_entropy`), which the caller chooses to call: the crate
itself never calls them.

Checked in every src/**/*.rs file, outside `#[cfg(test)]` modules, with the
code of a file joined (a call split over lines still matches) and `use .. as`
aliases of `Instant` / `SystemTime` / `RandomState` / `HashMap` / `HashSet`
resolved:

    getrandom  OsRng  thread_rng  rand::random  from_entropy(  try_from_entropy(
    SystemTime::now  Instant::now  UNIX_EPOCH  RandomState  HashMap  HashSet

(`HashMap` / `HashSet` because their default hasher is seeded from OS
randomness; the crate uses `det_hash::DetHashMap`.) clippy's
`disallowed-types` / `disallowed-methods` in clippy.toml is the first layer;
this script is the second, readable without compiling.

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
    "getrandom": r"\bgetrandom\b",
    "OsRng": r"\bOsRng\b",
    "thread_rng": r"\bthread_rng\b",
    "rand::random": r"\brand\s*::\s*random\b",
    "from_entropy(": r"(?<!fn )\b(?:try_)?from_entropy\s*\(",
    "SystemTime::now": r"\bSystemTime\s*::\s*now\b",
    "Instant::now": r"\bInstant\s*::\s*now\b",
    "UNIX_EPOCH": r"\bUNIX_EPOCH\b",
    "RandomState": r"\bRandomState\b",
    "HashMap": r"\bHashMap\b",
    "HashSet": r"\bHashSet\b",
}
# `use std::time::Instant as Clock;` makes `Clock::now()` the same call
ALIASABLE = {
    "Instant": ("Instant::now", r"\b{}\s*::\s*now\b"),
    "SystemTime": ("SystemTime::now", r"\b{}\s*::\s*now\b"),
    "RandomState": ("RandomState", r"\b{}\b"),
    "HashMap": ("HashMap", r"\b{}\b"),
    "HashSet": ("HashSet", r"\b{}\b"),
}
ALIAS_RE = re.compile(r"\b(Instant|SystemTime|RandomState|HashMap|HashSet)\s+as\s+(\w+)")
# (file, pattern) -> (count, reason)
ALLOWED: dict[tuple[str, str], tuple[int, str]] = {
    ("src/privacy.rs", "SystemTime::now"): (
        1, "deprecated XorShift64::from_entropy seeds from the clock; documented as not private",
    ),
    ("src/privacy.rs", "UNIX_EPOCH"): (2, "the same deprecated clock seed"),
    ("src/privacy.rs", "RandomState"): (2, "the same deprecated seed, hashed with a randomly keyed hasher"),
    ("src/privacy.rs", "from_entropy("): (
        4, "deprecated LaplaceNoise / RandomizedResponse / Rappor constructors call that clock seed",
    ),
    ("src/det_hash.rs", "HashMap"): (1, "the DetHashMap alias names the std map with the fixed hasher"),
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
            if stripped.startswith("#["):
                continue  # more attributes on the test module (`#[allow(..)]`)
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


def matches(lines: list[tuple[int, str]]) -> dict[str, list[int]]:
    """Line numbers of each pattern in the production lines of one file, with
    the code joined so a call split over lines still matches, and aliases
    (`use std::time::Instant as Clock`) resolved"""
    text = "\n".join(code for _, code in lines)
    starts = []
    pos = 0
    for _, code in lines:
        starts.append(pos)
        pos += len(code) + 1
    patterns = {name: [re.compile(rx)] for name, rx in PATTERNS.items()}
    for original, alias in ALIAS_RE.findall(text):
        name, rx = ALIASABLE[original]
        patterns[name].append(re.compile(rx.format(re.escape(alias))))
    found: dict[str, list[int]] = {}
    for name, rxs in patterns.items():
        hits = set()
        for rx in rxs:
            for m in rx.finditer(text):
                idx = max(i for i, st in enumerate(starts) if st <= m.start())
                hits.add(lines[idx][0])
        if hits:
            found[name] = sorted(hits)
    return found


def check(root: Path) -> tuple[list[str], int, int]:
    problems: list[str] = []
    counts: dict[tuple[str, str], list[int]] = {}
    files = sorted((root / "src").rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        for name, hits in matches(production_lines(path.read_text(encoding="utf-8"))).items():
            counts[(rel, name)] = hits
    for key, lines in sorted(counts.items()):
        allowed = ALLOWED.get(key)
        where = ", ".join(f"{key[0]}:{n}" for n in lines)
        if allowed is None:
            problems.append(f"{key[1]} in production code: {where}")
        elif allowed[0] != len(lines):
            problems.append(f"{key[1]} in {key[0]}: {len(lines)} call sites, ALLOWED says {allowed[0]} ({where})")
    for key, (count, reason) in sorted(ALLOWED.items()):
        if not reason.strip():
            problems.append(f"ALLOWED entry {key[1]} in {key[0]} has no reason")
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
