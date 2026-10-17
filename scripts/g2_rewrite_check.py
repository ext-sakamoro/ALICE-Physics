#!/usr/bin/env python3
"""ALICE-Law G2: comparison rewrites of max / min in library code each need a reviewed entry.

clippy's disallowed-methods bans `f32::max` / `f32::min` / `f64::max` / `f64::min`, but a
max or min written as a comparison is not a method call and the lint does not see it.
This check finds the two shapes such a rewrite takes in `src/` (test modules excluded):

* `expr`:   `if a > b { a } else { b }` (any of `<`, `>`, `<=`, `>=`, either branch order)
* `assign`: `if x < lo { x = lo; }` (the assignment on the same line or the next one)
* `clamp`:  a std `.clamp(lo, hi)` call with a float literal or a float path in its
  bounds. `f32::clamp` is banned by clippy in library code, so a match is an `f64::clamp`:
  alice-det-math, the owner of the clamp Law, has no 64-bit clamp yet. An integer clamp
  (no float in its bounds) is not counted

A comparison of floats gives a different result from the Law's maximumNumber /
minimumNumber for a pair of signed zeros or a NaN operand, so every match must be listed
in the reviewed table `scripts/g2-rewrite-sites.tsv`, keyed by file, enclosing function,
shape and operands (no line numbers):

* `fixed-point`: the operands are `Fix128` (an integer representation: no signed zero,
  no NaN), so the comparison is the Law's result
* `integer`: the operands are a primitive integer type, for the same reason
* `routed`: the call is a crate method of the same name that goes through the Law helper
  (the reason names it), not the std method
* `pending`: a float site that still has to go through the Law helpers
  (`crate::det_math::max` / `min` / `clamp`); the reason names what it waits for

The run fails on a match that is not listed, on a listed entry that no longer matches,
and when no source file was scanned.

usage: g2_rewrite_check.py [--root DIR] [--table TSV]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OPS = r"(<=|>=|<|>)"
OPERAND = r"(\*?[A-Za-z_][\w.]*(?:\([^()]*\))?(?:\[[^\[\]]*\])?)"
EXPR = re.compile(r"\bif\s+" + OPERAND + r"\s*" + OPS + r"\s*" + OPERAND +
                  r"\s*\{\s*" + OPERAND + r"\s*\}\s*else\s*\{\s*" + OPERAND + r"\s*\}")
ASSIGN_HEAD = re.compile(r"\bif\s+" + OPERAND + r"\s*" + OPS + r"\s*([-\w.()\[\]]+)\s*\{(.*)$")
CLAMP = re.compile(r"\.clamp\(((?:[^()]|\([^()]*\))*)\)")
FLOAT_BOUND = re.compile(r"\d\.\d|\bf(?:32|64)::")
FN = re.compile(r"\bfn\s+([A-Za-z_]\w*)")
CFG_TEST = re.compile(r"^\s*#\[cfg\(test\)\]")
STATUSES = ("fixed-point", "integer", "routed", "pending")


def code_of(line: str) -> str:
    """The line without a trailing `//` comment (a `//` inside a string is rare in src and only
    shortens what is scanned)"""
    return line.split("//", 1)[0]


def test_module_lines(lines: list[str]) -> set[int]:
    """Line indexes inside a `#[cfg(test)]` item (counted by braces from the item's first `{`)"""
    skip: set[int] = set()
    i = 0
    while i < len(lines):
        if CFG_TEST.match(lines[i]):
            depth, opened, j = 0, False, i
            while j < len(lines):
                code = code_of(lines[j])
                skip.add(j)
                depth += code.count("{") - code.count("}")
                opened = opened or "{" in code
                if opened and depth <= 0:
                    break
                if not opened and code.rstrip().endswith(";"):
                    break
                j += 1
            i = j + 1
            continue
        i += 1
    return skip


def scan_text(path: str, text: str) -> list[tuple[str, str, str, str]]:
    """(path, enclosing fn, shape, operands) for each rewrite in one file"""
    lines = text.split("\n")
    skip = test_module_lines(lines)
    found = []
    current_fn = "<module>"
    for i, line in enumerate(lines):
        code = code_of(line)
        m = FN.search(code)
        if m:
            current_fn = m.group(1)
        if i in skip:
            continue
        for m in EXPR.finditer(code):
            a, op, b, t, e = m.groups()
            if {t, e} == {a, b} and t != e:
                found.append((path, current_fn, "expr", f"{a} {op} {b}"))
        for m in CLAMP.finditer(code):
            bounds = " ".join(m.group(1).split())
            if FLOAT_BOUND.search(bounds):
                found.append((path, current_fn, "clamp", bounds))
        m = ASSIGN_HEAD.search(code)
        if m:
            x, op, y, rest = m.groups()
            body = rest + " " + (code_of(lines[i + 1]) if i + 1 < len(lines) else "")
            if re.search(r"(^|[^\w.])" + re.escape(x) + r"\s*=\s*" + re.escape(y) + r"\s*;", body):
                found.append((path, current_fn, "assign", f"{x} {op} {y}"))
    return found


def load_table(path: Path) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    table = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 6:
            raise SystemExit(f"{path}:{n}: expected 6 tab-separated columns, got {len(cols)}")
        file, fn, shape, operands, status, reason = cols
        if status not in STATUSES:
            raise SystemExit(f"{path}:{n}: status must be one of {STATUSES}, got {status!r}")
        if len(reason.strip()) < 12:
            raise SystemExit(f"{path}:{n}: the reason must say why (12 characters or more)")
        key = (file, fn, shape, operands)
        if key in table:
            raise SystemExit(f"{path}:{n}: duplicate entry {key}")
        table[key] = (status, reason)
    return table


def check(root: Path, table_path: Path) -> int:
    files = sorted(p for p in (root / "src").rglob("*.rs"))
    if not files:
        print(f"g2_rewrite_check: FAIL: no source file under {root / 'src'} (0 scanned)")
        return 1
    found: dict[tuple[str, str, str, str], int] = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        for key in scan_text(rel, p.read_text(encoding="utf-8")):
            found[key] = found.get(key, 0) + 1
    table = load_table(table_path)
    problems = []
    for key in sorted(found):
        if key not in table:
            problems.append("not reviewed: " + "\t".join(key))
    for key in sorted(table):
        if key not in found:
            problems.append("listed but not found (remove it): " + "\t".join(key))
    by = {s: sum(found[k] for k in found if k in table and table[k][0] == s) for s in STATUSES}
    print(f"g2_rewrite_check: {len(files)} files scanned, {sum(found.values())} rewrites "
          f"({by['fixed-point']} fixed-point, {by['integer']} integer, {by['routed']} routed, "
          f"{by['pending']} pending, "
          f"{len(problems)} problems)")
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--table", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    table = Path(args.table) if args.table else root / "scripts" / "g2-rewrite-sites.tsv"
    return check(root, table)


if __name__ == "__main__":
    sys.exit(main())
