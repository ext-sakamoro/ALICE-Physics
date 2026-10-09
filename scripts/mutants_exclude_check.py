#!/usr/bin/env python3
"""Check that every `.cargo/mutants.toml` `exclude_re` entry still matches
a mutant, and how many.

An `exclude_re` entry is a proof: "this specific mutant (or these specific
mutants) is equivalent, verified by reading the code at this line". The
proof is only still true if the entry still matches the line it was
written about. A line number shifts whenever code is added or removed
above it in the same file (adding a function, inserting a doc comment),
silently making a line-anchored entry match nothing — the mutant it used
to exclude is back in the unfiltered list, exactly as "missed" as if the
proof had never been written, and the next ratchet run (comparing against
`scripts/mutants-missed-baseline.txt`, which no longer lists it) reports
it as newly missed.

This compares each entry's match count (`cargo mutants --list --no-config`,
which bypasses `.cargo/mutants.toml` entirely, against every entry's
regex) with the tracked count in `scripts/mutants-exclude-baseline.txt`.
Fails when:

  * an entry matches 0 mutants (its proof covers nothing any more);
  * an entry's count differs from its tracked count (something about the
    code at those sites changed — a new mutation appeared there, one
    disappeared, or a line shifted onto an unrelated one with the same
    description text); or
  * it compared 0 entries at all (its own empty-input case).

  python3 scripts/mutants_exclude_check.py          # check
  python3 scripts/mutants_exclude_check.py --write  # record the current counts
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MUTANTS_TOML = REPO_ROOT / ".cargo" / "mutants.toml"
BASELINE = HERE / "mutants-exclude-baseline.txt"


def load_exclude_re(mutants_toml: Path) -> list[str]:
    with mutants_toml.open("rb") as f:
        doc = tomllib.load(f)
    return list(doc.get("exclude_re", []))


def full_mutant_list() -> list[str]:
    """`cargo mutants --list --no-config`: the full list, bypassing
    `.cargo/mutants.toml`'s own `exclude_re` entirely, so a regex's count
    here is exactly how many mutants it would remove."""
    proc = subprocess.run(
        ["cargo", "mutants", "--list", "--no-config"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def match_counts(patterns: list[str], lines: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    compiled = [(p, re.compile(p)) for p in patterns]
    for line in lines:
        for pattern, regex in compiled:
            if regex.search(line):
                counts[pattern] += 1
    return counts


def read_baseline(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        count_str, pattern = line.split("\t", 1)
        out[pattern] = int(count_str)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mutants-toml", type=Path, default=MUTANTS_TOML)
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--write", action="store_true", help="record the current counts")
    args = ap.parse_args(argv)

    patterns = load_exclude_re(args.mutants_toml)
    if not patterns:
        print("error: compared nothing (exclude_re is empty)", file=sys.stderr)
        return 1

    lines = full_mutant_list()
    counts = match_counts(patterns, lines)

    if args.write:
        text = "# pattern's current match count against `cargo mutants --list --no-config`\n"
        text += "# written by scripts/mutants_exclude_check.py --write\n"
        text += "".join(f"{counts[p]}\t{p}\n" for p in patterns)
        args.baseline.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {len(patterns)} entries to {args.baseline}")
        return 0

    baseline = read_baseline(args.baseline)
    errors = []
    for pattern in patterns:
        got = counts[pattern]
        if got == 0:
            errors.append(f"matches 0 mutants (no proof left to exclude anything): {pattern!r}")
            continue
        want = baseline.get(pattern)
        if want is None:
            errors.append(f"not in baseline (run --write): {pattern!r}")
        elif got != want:
            errors.append(f"match count changed from {want} to {got}: {pattern!r}")
    for stale in set(baseline) - set(patterns):
        errors.append(f"baseline entry no longer in exclude_re (stale): {stale!r}")

    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: {len(patterns)} entries, {len(lines)} mutants in the full list")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
