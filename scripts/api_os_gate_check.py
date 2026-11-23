#!/usr/bin/env python3
"""No public item of the crates is gated on the operating system.

security-audit.yml checks the public API snapshot (docs/PUBLIC_API_SNAPSHOT.txt) on an
aarch64 Linux runner, and the developers' machines are aarch64 Macs. That is the same
snapshot only while no public item depends on the OS: an item under
`#[cfg(target_os = ...)]`, `target_family`, `target_vendor`, `target_env`, `unix` or
`windows` (as `cfg` or `cfg_attr`) would exist on one and not the other. This script reads
every `.rs` file under src/ and crates/*/src/, with comments removed, and reads each
attribute up to the `]` that balances it (rustfmt writes a long `cfg` over several lines).
It fails on
  * an OS attribute whose item is `pub` (plain `pub`, not `pub(crate)` / `pub(super)` /
    `pub(in ...)`): the first code after the attribute and any further attributes, on the
    same line or a later one;
  * a file-level `#![cfg(...)]` naming the OS in a file that has a plain `pub` item.
An OS gate on a private item (a test, a helper) does not change the public surface and
passes; `cfg!(...)` inside a body is an expression, not a gate, and passes.
Exit 1 on a finding, 2 when no file was read.
    python3 scripts/api_os_gate_check.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OS_WORDS = re.compile(r"\b(?:target_os|target_family|target_vendor|target_env|unix|windows)\b")
GATE = re.compile(r"^#!?\[\s*(?:cfg|cfg_attr)\s*\(")
PUBLIC = re.compile(r"pub\s")
PUBLIC_LINE = re.compile(r"^\s*pub\s", re.M)


def sources(root: Path) -> list[Path]:
    dirs = [root / "src", *sorted((root / "crates").glob("*/src"))]
    return sorted(p for d in dirs if d.is_dir() for p in d.rglob("*.rs"))


def strip_comments(src: str) -> str:
    """`//` and `/* */` comments replaced by spaces (newlines kept, so line numbers hold);
    string and char literals are kept as they are"""
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
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif c == "'" and re.match(r"'(?:\\.|[^\\'])'", src[i:i + 4]):
            m = re.match(r"'(?:\\.|[^\\'])'", src[i:i + 4])
            out.append(m.group(0))
            i += len(m.group(0))
        else:
            out.append(c)
            i += 1
    return "".join(out)


def attribute_end(text: str, start: int) -> int:
    """the index after the `]` that closes the attribute opened at `start` (`#[` / `#![`)"""
    depth, j, n = 0, start, len(text)
    while j < n:
        c = text[j]
        if c == '"':
            j += 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return n


def findings(path: Path, rel: str) -> list[str]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    has_public = bool(PUBLIC_LINE.search(text))
    out: list[str] = []
    pending: list[int] = []  # line numbers of OS attributes waiting for their item
    pos, n = 0, len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        if text.startswith("#[", pos) or text.startswith("#![", pos):
            end = attribute_end(text, pos)
            attr = text[pos:end]
            line = text.count("\n", 0, pos) + 1
            if GATE.match(attr) and OS_WORDS.search(attr):
                if attr.startswith("#!["):
                    if has_public:
                        out.append(f"{rel}:{line}: a file-level OS cfg on a file with public items")
                else:
                    pending.append(line)
            pos = end
            continue
        eol = text.find("\n", pos)
        eol = n if eol < 0 else eol
        head = text[pos:eol]
        if pending and PUBLIC.match(head):
            out.extend(f"{rel}:{line}: an OS cfg on a public item ({head.strip()[:60]})" for line in pending)
        pending = []
        pos = eol + 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(argv)
    files = sources(args.root)
    if not files:
        print("error: no .rs file under src/ or crates/*/src/ (compared nothing)", file=sys.stderr)
        return 2
    found = [f for p in files for f in findings(p, p.relative_to(args.root).as_posix())]
    for f in found:
        print(f"error: {f}", file=sys.stderr)
    print(f"compared: rust files {len(files)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
