#!/usr/bin/env python3
"""Rewrite the line references of docs/coverage/*.toml as symbol references.

A `file:line` reference moves whenever lines are inserted above it, so
scripts/coverage_check.py refuses it. This script converts each one in place:

  * evidence `src/x.rs:123`          -> `src/x.rs::name`
  * limitation `src/x.rs:123 'quote'` -> `src/x.rs::name 'quote'`

`name` is the item the line belongs to: the next definition when the line is a
doc comment, an attribute or blank (it documents what follows), otherwise the
nearest definition above it. A line in the module documentation (`//!` before
any definition) becomes the bare file. A limitation is converted only when its
quote is inside the item the new reference names; otherwise it is reported and
left as it was, to be fixed by hand. The line numbers are read against the
current tree, so run it on the tree the tables were written for.

  python3 scripts/coverage_refs_to_symbols.py            # rewrite, report what is left
  python3 scripts/coverage_refs_to_symbols.py --check    # report only, exit 1 if anything would change
  python3 scripts/coverage_refs_to_symbols.py --root DIR
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coverage_check as cc  # noqa: E402

LINE_REF_RE = re.compile(r"(?<![\w/.-])((?:src|tests)/[\w/.-]+?\.rs):(\d+)")
FIELD_RE = re.compile(r'^(evidence|limitation)(\s*=\s*)"((?:[^"\\]|\\.)*)"\s*$')


def symbol_for(lines: list[str], n: int) -> str | None:
    """The item line `n` (1-based) belongs to, or None for the module documentation."""
    i = n - 1
    text = lines[i].strip() if 0 <= i < len(lines) else ""
    if text.startswith("//!"):
        return None
    m = cc.DEF_RE.match(lines[i]) if i < len(lines) else None
    if m:
        return m.group(1)
    if text == "" or text.startswith(("///", "//", "#[", "#!")):
        for k in range(i + 1, len(lines)):
            m = cc.DEF_RE.match(lines[k])
            if m:
                return m.group(1)
            if lines[k].strip() and not lines[k].strip().startswith(("///", "//", "#[", "#!")):
                break
    for k in range(i, -1, -1):
        m = cc.DEF_RE.match(lines[k])
        if m:
            return m.group(1)
    return None


def convert(root: Path, write: bool) -> tuple[int, list[str]]:
    cache: dict[str, list[str] | None] = {}

    def lines_of(rel: str) -> list[str] | None:
        if rel not in cache:
            p = root / rel
            cache[rel] = p.read_text(encoding="utf-8").splitlines() if p.is_file() else None
        return cache[rel]

    def repl_ref(m: re.Match) -> str:
        lines = lines_of(m.group(1))
        if lines is None or not 1 <= int(m.group(2)) <= len(lines):
            return m.group(0)
        sym = symbol_for(lines, int(m.group(2)))
        return m.group(1) if sym is None else f"{m.group(1)}::{sym}"

    changed, left = 0, []
    tables = sorted((root / "docs" / "coverage").glob("*.toml"))
    if not tables:
        return 0, ["no table: docs/coverage/*.toml matched nothing"]
    for t in tables:
        rel = t.relative_to(root).as_posix()
        out = []
        for n, ln in enumerate(t.read_text(encoding="utf-8").split("\n"), 1):
            m = FIELD_RE.match(ln)
            if m and LINE_REF_RE.search(m.group(3)):
                key, eq, val = m.groups()
                new = LINE_REF_RE.sub(repl_ref, val) if key == "evidence" else val
                if key == "limitation":
                    lm = re.match(r"^((?:src|tests)/[\w/.-]+?\.rs):(\d+)( '.*)$", val, re.S)
                    lines = lines_of(lm.group(1)) if lm else None
                    if lm and lines is not None and 1 <= int(lm.group(2)) <= len(lines):
                        sym = symbol_for(lines, int(lm.group(2)))
                        quote = " ".join(lm.group(3)[2:-1].replace('\\"', '"').split())
                        spans = [(0, len(lines) - 1)] if sym is None else cc.item_spans(lines, sym)
                        if any(quote in cc._flatten("\n".join(lines[a:b + 1])) for a, b in spans):
                            new = (lm.group(1) if sym is None else f"{lm.group(1)}::{sym}") + lm.group(3)
                        elif quote in cc._flatten("\n".join(lines)):
                            # documentation of an unnamed item (an `impl` block): the file is the
                            # narrowest reference that holds the quote
                            new = lm.group(1) + lm.group(3)
                            left.append(f"{rel}:{n}: limitation widened to the whole file (the quote is "
                                        f"not inside a named item): {new[:100]}")
                if LINE_REF_RE.search(new):
                    left.append(f"{rel}:{n}: {key} still holds a line reference: {new[:100]}")
                if new != val:
                    changed += 1
                    ln = f'{key}{eq}"{new}"'
            out.append(ln)
        if write:
            t.write_text("\n".join(out), encoding="utf-8", newline="\n")
    return changed, left


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if anything would change")
    args = ap.parse_args(argv)
    changed, left = convert(Path(args.root), write=not args.check)
    for msg in left:
        print(f"coverage refs: {msg}", file=sys.stderr)
    print(f"coverage refs: {changed} field(s) {'would change' if args.check else 'rewritten'}, "
          f"{len(left)} left for hand")
    return 1 if (args.check and changed) or left else 0


if __name__ == "__main__":
    sys.exit(main())
