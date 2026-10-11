#!/usr/bin/env python3
"""Equivalent mutants: missed by cargo-mutants, but no test can see them.

A mutant is equivalent when the mutated program computes the same observable
result as the original (for example a sign flip of a vector whose only use is
its length). cargo-mutants reports such a mutant as missed on every run. This
file lists them in scripts/mutants-equivalent.txt so the in-diff run and the
weekly ratchet do not report them, without letting a code move hide a real
survivor:

  * an entry is keyed by `file:line:col` and the exact mutation text, as
    cargo-mutants writes it in `missed.txt`
  * an entry records the SHA-256 (first 16 hex digits) of the source line with
    surrounding whitespace removed; an entry whose line no longer hashes to it
    is stale: `--check` fails and the mutant is not filtered (it reads missed
    again until someone re-reads the argument and updates the entry)
  * for `replace <op> with ...` the text at `col` must start with `<op>`
  * every entry carries a non-empty reason; an empty list fails `--check`

Format (tab separated, `#` starts a comment line):

  src/joint.rs:955:23<TAB><hash>:<TAB>replace + with - in solve_ball_joint<TAB><reason>

Usage:

  python3 scripts/mutants_equivalent.py --check
  python3 scripts/mutants_equivalent.py --filter mutants.out/missed.txt   # prints the real survivors
  python3 scripts/mutants_equivalent.py --hash src/joint.rs:955           # hash of a line, for a new entry
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIST = HERE / "mutants-equivalent.txt"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
MISSED_RE = re.compile(r"^(?P<file>[^:\s]+):(?P<line>\d+):(?P<col>\d+): (?P<mutation>.+)$")
REPLACE_RE = re.compile(r"^replace (?P<op>\S+) with \S+ in ")


@dataclass(frozen=True)
class Entry:
    file: str
    line: int
    col: int
    digest: str
    mutation: str
    reason: str
    lineno: int  # line of the entry in the list, for messages

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (self.file, self.line, self.col, self.mutation)


def line_digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def parse(text: str) -> tuple[list[Entry], list[str]]:
    entries, errors = [], []
    seen: set[tuple[str, int, int, str]] = set()
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 4:
            errors.append(f"line {n}: expected 4 tab-separated fields, found {len(parts)}")
            continue
        pos, digest, mutation, reason = (p.strip() for p in parts)
        m = re.fullmatch(r"(?P<file>[^:\s]+):(?P<line>\d+):(?P<col>\d+)", pos)
        if not m:
            errors.append(f"line {n}: position {pos!r} is not file:line:col")
            continue
        if not re.fullmatch(r"[0-9a-f]{16}", digest):
            errors.append(f"line {n}: hash {digest!r} is not 16 lower-case hex digits")
            continue
        if not mutation:
            errors.append(f"line {n}: empty mutation")
            continue
        if not reason:
            errors.append(f"line {n}: empty reason (say why no test can see this mutant)")
            continue
        e = Entry(m["file"], int(m["line"]), int(m["col"]), digest, mutation, reason, n)
        if e.key in seen:
            errors.append(f"line {n}: duplicate entry {pos} {mutation}")
            continue
        seen.add(e.key)
        entries.append(e)
    return entries, errors


def stale_reason(e: Entry, root: Path) -> str | None:
    """Why the entry no longer describes the source, or None when it does."""
    path = root / e.file
    if not path.is_file():
        return f"{e.file} does not exist"
    lines = path.read_text(encoding="utf-8").splitlines()
    if not 1 <= e.line <= len(lines):
        return f"{e.file} has {len(lines)} lines, entry points at line {e.line}"
    text = lines[e.line - 1]
    if line_digest(text) != e.digest:
        return (f"{e.file}:{e.line} changed (hash {line_digest(text)}, entry {e.digest}): "
                f"re-read the argument and update the entry")
    m = REPLACE_RE.match(e.mutation)
    if m and not text[e.col - 1:].startswith(m["op"]):
        return f"{e.file}:{e.line}:{e.col} does not start with {m['op']!r}"
    return None


def load(path: Path = LIST) -> tuple[list[Entry], list[str]]:
    if not path.is_file():
        return [], [f"{path} does not exist"]
    return parse(path.read_text(encoding="utf-8"))


def check(path: Path = LIST, root: Path = ROOT) -> list[str]:
    entries, errors = load(path)
    if not entries and not errors:
        errors.append(f"{path.name} lists no entry (checked nothing)")
    for e in entries:
        why = stale_reason(e, root)
        if why:
            errors.append(f"line {e.lineno}: stale: {why}")
    return errors


def valid_keys(path: Path = LIST, root: Path = ROOT) -> set[tuple[str, int, int, str]]:
    """Keys of the entries that still describe the source (stale ones are left out)."""
    entries, _ = load(path)
    return {e.key for e in entries if stale_reason(e, root) is None}


def missed_key(line: str) -> tuple[str, int, int, str] | None:
    m = MISSED_RE.match(ANSI_RE.sub("", line).strip())
    if not m:
        return None
    return (m["file"], int(m["line"]), int(m["col"]), m["mutation"].strip())


def split_missed(lines: list[str], keys: set[tuple[str, int, int, str]]) -> tuple[list[str], list[str]]:
    """(real survivors, equivalent) of the non-empty lines of a missed.txt."""
    real, equivalent = [], []
    for line in lines:
        if not line.strip():
            continue
        k = missed_key(line)
        (equivalent if k is not None and k in keys else real).append(line)
    return real, equivalent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="every entry has a reason and still matches the source")
    g.add_argument("--filter", metavar="MISSED_TXT", help="print the missed mutants that are not listed")
    g.add_argument("--hash", metavar="FILE:LINE", help="print the hash of a source line")
    ap.add_argument("--list", default=str(LIST))
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    path, root = Path(args.list), Path(args.root)

    if args.hash:
        file, _, line = args.hash.rpartition(":")
        text = (root / file).read_text(encoding="utf-8").splitlines()[int(line) - 1]
        print(line_digest(text))
        return 0

    if args.check:
        errors = check(path, root)
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        if errors:
            return 1
        print(f"checked {len(load(path)[0])} equivalent mutants")
        return 0

    missed = Path(args.filter)
    lines = missed.read_text(encoding="utf-8", errors="replace").splitlines() if missed.is_file() else []
    real, equivalent = split_missed(lines, valid_keys(path, root))
    for line in real:
        print(line)
    print(f"{len(equivalent)} equivalent (scripts/mutants-equivalent.txt), {len(real)} missed",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
