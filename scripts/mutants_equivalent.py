#!/usr/bin/env python3
"""Equivalent mutants: missed by cargo-mutants, but no test can see them.

A mutant is equivalent when every bit of every observable result of the mutated
program equals the original's (for example a sign flip of a vector whose only use
is its length, when the negation is exact). Close is not equal: a mutant that
changes a result only in its low bits is a survivor, and a test should pin it. cargo-mutants reports such a mutant as missed on every run. This
file lists them in scripts/mutants-equivalent.txt so the in-diff run and the
weekly ratchet do not report them, without letting a code move hide a real
survivor:

  * an entry is keyed by `file:line:col` and the exact mutation text, as
    cargo-mutants writes it in `missed.txt`
  * an entry records the SHA-256 (first 16 hex digits) of the source line with
    surrounding whitespace removed, and of the whole function that holds it
    (found by the name at the end of the mutation text, its lines stripped);
    an entry whose line or function no longer hashes to them is stale:
    `--check` fails and the mutant is not filtered (it reads missed again until
    someone re-reads the argument and updates the entry). The function hash
    covers arguments about the rest of the body ("the only line that adds to
    impulse"): a second row added later changes it
  * for `replace <op> with ...` the text at `col` must start with `<op>`
  * every entry carries a non-empty reason, and the reason may not argue
    approximately (`up to rounding`, `approximately`, `low bits`, `≈`, ...);
    this word list is only a hint, the definition above and the required
    tests field are the guard
  * every entry names, in a fifth field, the tests that pin its argument
    (`tests/<file>.rs::<fn>` separated by commas); `--check` fails when a named
    file or `#[test] fn` is gone, so the mutant comes back when its argument does
  * an empty list fails `--check`

Format (tab separated, `#` starts a comment line):

  src/joint.rs:955:23<TAB><line hash><TAB><fn hash><TAB>replace + with - in solve_ball_joint<TAB><reason><TAB><tests>

Usage:

  python3 scripts/mutants_equivalent.py --check
  python3 scripts/mutants_equivalent.py --filter mutants.out/missed.txt   # prints the real survivors
  python3 scripts/mutants_equivalent.py --hash src/joint.rs:955:solve_ball_joint   # line and fn hash, for a new entry
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
APPROX_RE = re.compile(
    r"round(?:ing|ed|s)?\b(?! trip)|\bulps?\b|low(?:er)?[- ]order bits|\blow bits|last bits?|"
    r"approx|roughly|nearly|almost|practically|virtually|close to|negligible|modulo|"
    r"within [0-9]|tolerance|≈|~=", re.I)
TEST_RE = re.compile(r"tests/[A-Za-z0-9_/]+\.rs::[A-Za-z_][A-Za-z0-9_]*")


def has_test(root: Path, ref: str) -> bool:
    """`tests/x.rs::name` names a `#[test] fn name` (attributes may sit between)."""
    file, _, name = ref.partition("::")
    path = root / file
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    pat = rf"#\[test\]\s*(?:#\[[^\]]*\]\s*)*(?:pub\s+)?fn\s+{re.escape(name)}\s*\("
    return re.search(pat, text) is not None


@dataclass(frozen=True)
class Entry:
    file: str
    line: int
    col: int
    digest: str
    mutation: str
    reason: str
    lineno: int  # line of the entry in the list, for messages
    tests: tuple[str, ...] = ()  # `tests/<file>.rs::<fn>` that pin the invariant
    fn_digest: str = ""  # hash of the whole enclosing function

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (self.file, self.line, self.col, self.mutation)


def line_digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def mask(src: str) -> str:
    """The source with comments, strings, raw strings and char literals blanked
    (newlines kept), so braces are counted only in code."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == "r" and (m := re.match(r'r(#*)"', src[i:])) and not (
                i and (src[i - 1].isalnum() or src[i - 1] == "_")):
            close = '"' + m.group(1)
            j = src.find(close, i + len(m.group(0)))
            j = n if j < 0 else j + len(close)
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == "'" and (m := re.match(r"'(\\.[^']*|[^\\'])'", src[i:])):
            out.append(" " * len(m.group(0)))
            i += len(m.group(0))
        else:
            out.append(c)
            i += 1
    return "".join(out)


def fn_name(mutation: str) -> str | None:
    """`replace + with - in solve_ball_joint` / `... in PhysicsWorld::f` -> the fn name."""
    m = re.search(r" in ([A-Za-z_][A-Za-z0-9_:<>, ]*?)(?: -> .*)?$", mutation)
    return m.group(1).split("::")[-1].strip() if m else None


def fn_text(src: str, line: int, name: str) -> str | None:
    """The lines of `fn name` that holds `line` (1-based), stripped and joined, or None."""
    lines, code = src.splitlines(), mask(src).splitlines()
    head = re.compile(rf"\bfn\s+{re.escape(name)}\b")
    for start in range(min(line, len(code)) - 1, -1, -1):
        if not head.search(code[start]):
            continue
        depth, opened = 0, False
        for end in range(start, len(code)):
            for ch in code[end]:
                if ch == "{":
                    depth, opened = depth + 1, True
                elif ch == "}":
                    depth -= 1
            if opened and depth == 0:
                if start < line <= end + 1:
                    return "\n".join(x.strip() for x in lines[start:end + 1])
                return None
        return None
    return None


def fn_digest_at(src: str, line: int, name: str) -> str | None:
    text = fn_text(src, line, name)
    return None if text is None else line_digest(text)


def parse(text: str) -> tuple[list[Entry], list[str]]:
    entries, errors = [], []
    seen: set[tuple[str, int, int, str]] = set()
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 6:
            errors.append(f"line {n}: expected 6 tab-separated fields (position, line hash, "
                          f"function hash, mutation, reason, tests), found {len(parts)}")
            continue
        pos, digest, fn_digest, mutation, reason = (p.strip() for p in parts[:5])
        tests = tuple(t.strip() for t in parts[5].split(","))
        if not all(TEST_RE.fullmatch(t) for t in tests):
            errors.append(f"line {n}: tests {parts[4].strip()!r} are not tests/<file>.rs::<fn>")
            continue
        m = re.fullmatch(r"(?P<file>[^:\s]+):(?P<line>\d+):(?P<col>\d+)", pos)
        if not m:
            errors.append(f"line {n}: position {pos!r} is not file:line:col")
            continue
        if not re.fullmatch(r"[0-9a-f]{16}", digest) or not re.fullmatch(r"[0-9a-f]{16}", fn_digest):
            errors.append(f"line {n}: hashes {digest!r} / {fn_digest!r} are not 16 lower-case hex digits")
            continue
        if not mutation:
            errors.append(f"line {n}: empty mutation")
            continue
        if not reason:
            errors.append(f"line {n}: empty reason (say why no test can see this mutant)")
            continue
        m_approx = APPROX_RE.search(reason)
        if m_approx:
            errors.append(f"line {n}: the reason argues approximately ({m_approx.group(0)!r}): "
                          f"equivalent means every bit of every observable result is equal")
            continue
        e = Entry(m["file"], int(m["line"]), int(m["col"]), digest, mutation, reason, n, tests,
                  fn_digest)
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
    name = fn_name(e.mutation)
    if name is None:
        return f"cannot read the function name from {e.mutation!r}"
    now = fn_digest_at(path.read_text(encoding="utf-8"), e.line, name)
    if now is None:
        return f"{e.file}:{e.line} is not inside fn {name}"
    if now != e.fn_digest:
        return (f"fn {name} in {e.file} changed (hash {now}, entry {e.fn_digest}): "
                f"re-read the argument and update the entry")
    m = REPLACE_RE.match(e.mutation)
    if m and not text[e.col - 1:].startswith(m["op"]):
        return f"{e.file}:{e.line}:{e.col} does not start with {m['op']!r}"
    gone = [t for t in e.tests if not has_test(root, t)]
    if gone:
        return f"the test that pins the argument is gone: {', '.join(gone)}"
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
    g.add_argument("--hash", metavar="FILE:LINE:FN", help="print the line and function hashes")
    ap.add_argument("--list", default=str(LIST))
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    path, root = Path(args.list), Path(args.root)

    if args.hash:
        file, line, name = args.hash.rsplit(":", 2)
        src = (root / file).read_text(encoding="utf-8")
        print(line_digest(src.splitlines()[int(line) - 1]), fn_digest_at(src, int(line), name))
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
