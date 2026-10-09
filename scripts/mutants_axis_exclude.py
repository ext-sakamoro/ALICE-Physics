#!/usr/bin/env python3
"""Which source lines a mutation run on one feature axis leaves to another axis.

A line inside `#[cfg(feature = "parallel")]` is not compiled by a default
build, so mutating it changes nothing there and every such mutant reads
"missed" on the default axis, although the parallel axis compiles and tests
that line. This script reads the `#[cfg(...)]` attributes of the given files,
evaluates them for each axis, and prints the lines an axis does not compile
but another axis does. The weekly workflow passes them to cargo-mutants as
`--exclude-re`, so those mutants are measured on the axis that compiles them
instead of being counted missed on the one that cannot.

Lines that no axis compiles (`not(feature = "std")`, another target
architecture, release only) are not excluded: no axis would measure them.

An axis is a cargo feature list (`""` is the default features); features are
expanded through `[features]` of Cargo.toml. The evaluation also fixes the
target the workflow runs on: `target_arch = "x86_64"`, no `avx2`, a test build
(`test`, `debug_assertions`), no `loom`.

Fails (exit 1) rather than excluding nothing: a predicate it cannot evaluate,
an attribute whose item it cannot delimit, or no `cfg` attribute in the files.

  scripts/mutants_axis_exclude.py --axis "" --other "parallel,gpu-solver-bridge,simd" \
      --file src/solver.rs ... [--format args|lines]
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = {"target_arch": "x86_64", "target_os": "linux", "target_feature": set()}
FLAGS = {"test": True, "debug_assertions": True, "loom": False}
CFG_RE = re.compile(r"^\s*#\[cfg\((.*)\)\]\s*$")
ITEM_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:const|async|unsafe|extern(?:\s+\"[^\"]*\")?|default)\s+)*"
    r"(?:fn|impl|struct|enum|union|trait|mod|type|use|static|const|macro_rules!)\b"
)


class Unknown(Exception):
    """A cfg predicate this script does not evaluate."""


def expand_features(axis: str, table: dict[str, list[str]]) -> set[str]:
    """The features an axis enables: its list (or `default`), closed under `[features]`."""
    todo = [f for f in axis.split(",") if f] or ["default"]
    out: set[str] = set()
    while todo:
        f = todo.pop()
        if f in out or f not in table:
            continue
        out.add(f)
        todo += [d for d in table[f] if "/" not in d and not d.startswith("dep:")]
    out.discard("default")
    return out


def _split_args(s: str) -> list[str]:
    """Top-level comma-separated arguments of a cfg list."""
    parts, depth, cur, in_str = [], 0, "", False
    for ch in s:
        if ch == '"':
            in_str = not in_str
        if not in_str and ch == "(":
            depth += 1
        elif not in_str and ch == ")":
            depth -= 1
        if not in_str and ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts]


def evaluate(pred: str, features: set[str]) -> bool:
    pred = pred.strip()
    m = re.fullmatch(r"(all|any|not)\((.*)\)", pred, re.S)
    if m:
        args = [evaluate(a, features) for a in _split_args(m.group(2))]
        if m.group(1) == "all":
            return all(args)
        if m.group(1) == "any":
            return any(args)
        if len(args) != 1:
            raise Unknown(pred)
        return not args[0]
    m = re.fullmatch(r'(\w+)\s*=\s*"([^"]*)"', pred)
    if m:
        key, value = m.groups()
        if key == "feature":
            return value in features
        if key == "target_feature":
            return value in TARGET["target_feature"]
        if key in TARGET:
            return TARGET[key] == value
        raise Unknown(pred)
    if pred in FLAGS:
        return FLAGS[pred]
    raise Unknown(pred)


def regions(text: str) -> list[tuple[int, int, str]]:
    """`(first line, last line, predicate)` of every item or statement under an
    outer `#[cfg(...)]`, 1-based and inclusive. The item starts at the first
    line after the attributes and comments; it ends where its first `{` block
    closes, or, if no block opens first, at a line ending in `;` (or, for a
    statement or match arm, `,`) outside parentheses."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        m = CFG_RE.match(line)
        if not m:
            continue
        j = i + 1
        while j < len(lines) and (
            not lines[j].strip() or lines[j].strip().startswith(("#[", "//"))
        ):
            j += 1
        if j == len(lines):
            raise Unknown(f"line {i + 1}: #[cfg] with no item after it")
        # an item (fn / impl / struct / mod / ...) ends with its block or a `;`;
        # only a statement or a match arm ends with `,` (a where clause or a
        # multi-line signature has commas before the body opens)
        is_item = bool(ITEM_RE.match(lines[j]))
        depth, parens, opened, k = 0, 0, False, j
        while k < len(lines):
            code = lines[k].split("//")[0]
            for ch in code:
                if ch == "{":
                    depth += 1
                    opened = True
                elif ch == "}":
                    depth -= 1
                elif ch in "([":
                    parens += 1
                elif ch in ")]":
                    parens -= 1
            if opened and depth <= 0:
                break
            # a `;` / `,` ends the item only outside parentheses (a multi-line
            # signature or argument list has commas before its body opens)
            ends = (";",) if is_item else (";", ",")
            if not opened and parens == 0 and code.rstrip().endswith(ends):
                break
            k += 1
        if k == len(lines):
            raise Unknown(f"line {i + 1}: the item under #[cfg] does not end")
        out.append((j + 1, k + 1, m.group(1)))
    return out


def compiled_lines(text: str, features: set[str]) -> dict[int, bool]:
    """`{line: compiled?}` for every line inside some cfg region."""
    out: dict[int, bool] = {}
    for first, last, pred in regions(text):
        ok = evaluate(pred, features)
        for ln in range(first, last + 1):
            out[ln] = out.get(ln, True) and ok
    return out


def moved_lines(text: str, axis: set[str], others: list[set[str]]) -> list[int]:
    """Lines `axis` does not compile and some other axis does."""
    mine = compiled_lines(text, axis)
    theirs = [compiled_lines(text, o) for o in others]
    return sorted(
        ln for ln, ok in mine.items() if not ok and any(t.get(ln, True) for t in theirs)
    )


def patterns(files: dict[str, list[int]]) -> list[str]:
    """One `--exclude-re` per file with moved lines (cargo-mutants names a
    mutant `file:line:column: ...`)."""
    out = []
    for f, lines in sorted(files.items()):
        if lines:
            alt = "|".join(str(n) for n in lines)
            out.append(f"^{re.escape(f)}:({alt}):")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--axis", required=True, help='feature list of this axis ("" = default)')
    ap.add_argument("--other", action="append", default=[], help="feature list of another axis")
    ap.add_argument("--file", action="append", required=True)
    ap.add_argument("--format", choices=("args", "lines"), default="args")
    ap.add_argument("--allow-no-cfg", action="store_true",
                    help="files without any #[cfg] are fine (the in-diff run: a changed "
                         "file may have none); the weekly run reads fixed files and needs one")
    args = ap.parse_args(argv)
    table = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["features"]
    axis = expand_features(args.axis, table)
    others = [expand_features(o, table) for o in args.other]
    if not others:
        print("error: no other axis given (nothing could take the lines over)", file=sys.stderr)
        return 1
    moved: dict[str, list[int]] = {}
    found = 0
    for f in args.file:
        text = (ROOT / f).read_text(encoding="utf-8")
        try:
            found += len(regions(text))
            moved[f] = moved_lines(text, axis, others)
        except Unknown as e:
            print(f"error: {f}: cannot evaluate cfg: {e}", file=sys.stderr)
            return 1
    if found == 0 and not args.allow_no_cfg:
        print("error: no #[cfg] attribute in the files (the analysis read nothing)", file=sys.stderr)
        return 1
    if args.format == "lines":
        for f, lines in sorted(moved.items()):
            for n in lines:
                print(f"{f}:{n}")
    else:
        for p in patterns(moved):
            print(f"--exclude-re={p}")
    total = sum(len(v) for v in moved.values())
    print(f"axis {sorted(axis)}: {total} line(s) left to another axis, {found} cfg region(s) read",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
