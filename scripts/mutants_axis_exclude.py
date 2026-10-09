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


def blank_non_code(text: str) -> str:
    r"""`text` with string, raw string, char literal and comment contents replaced
    by spaces (newlines kept), so braces and parentheses inside them are not
    counted. A `'` starts a char literal only when it closes within a few
    characters (`'a'`, `'\n'`, `'\u{1F600}'`), otherwise it is a lifetime."""
    out = list(text)
    n, i = len(text), 0

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            e = text.find("\n", i)
            e = n if e < 0 else e
            blank(i, e)
            i = e
        elif text.startswith("/*", i):
            depth, k = 1, i + 2
            while k < n and depth:
                if text.startswith("/*", k):
                    depth, k = depth + 1, k + 2
                elif text.startswith("*/", k):
                    depth, k = depth - 1, k + 2
                else:
                    k += 1
            blank(i, k)
            i = k
        elif c == "r" and re.match(r'r#*"', text[i:]) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            hashes = re.match(r'r(#*)"', text[i:]).group(1)
            start = i + 2 + len(hashes)
            e = text.find('"' + hashes, start)
            e = n if e < 0 else e + 1 + len(hashes)
            blank(start, e - 1 - len(hashes))
            i = e
        elif c == '"':
            k = i + 1
            while k < n and text[k] != '"':
                k += 2 if text[k] == "\\" else 1
            blank(i + 1, k)
            i = k + 1
        elif c == "'":
            m = re.match(r"'(\\u\{[0-9a-fA-F]+\}|\\.|[^\\'\n])'", text[i:])
            if m:
                blank(i + 1, i + m.end() - 1)
                i += m.end()
            else:
                i += 1  # a lifetime
        else:
            i += 1
    return "".join(out)


def _matching(code: str, open_at: int, pair: str = "()") -> int:
    """Index just past the bracket matching `code[open_at]`, or -1."""
    depth = 0
    for k in range(open_at, len(code)):
        if code[k] == pair[0]:
            depth += 1
        elif code[k] == pair[1]:
            depth -= 1
            if depth == 0:
                return k + 1
    return -1


def regions(text: str) -> list[tuple[int, int, str]]:
    """`(first line, last line, predicate)` of every item or statement under a
    `#[cfg(...)]`, 1-based and inclusive, read with strings, chars and comments
    blanked. The attribute may share its line with the item or span several
    lines; other attributes between it and the item are skipped. The item ends
    where its first `{` block closes (an `if` keeps its `else` branches), or,
    if no block opens first, at a `;` (or, for a statement or match arm, `,`)
    outside parentheses. `#![cfg(...)]` covers the rest of the file.
    `#[cfg_attr(...)]` removes no code and is not a region."""
    code = blank_non_code(text)
    line_of = lambda off: text.count("\n", 0, off) + 1  # noqa: E731
    out = []
    for m in re.finditer(r"#(!?)\[\s*cfg\s*\(", code):
        open_paren = m.end() - 1
        close = _matching(code, open_paren)
        if close < 0 or not re.match(r"\s*\]", code[close:]):
            raise Unknown(f"line {line_of(m.start())}: #[cfg( ... without a closing )]")
        pred = " ".join(text[open_paren + 1 : close - 1].split())
        after = close + re.match(r"\s*\]", code[close:]).end()
        if m.group(1):  # inner attribute: the rest of the file
            first = after + re.match(r"\s*", code[after:]).end()
            if first < len(text):
                out.append((line_of(first), line_of(len(text.rstrip()) - 1), pred))
            continue
        # skip whitespace and further attributes to the item itself
        k = after
        while True:
            ws = re.match(r"\s*", code[k:]).end()
            k += ws
            if code.startswith("#[", k):
                e = _matching(code, k + 1, "[]")
                if e < 0:
                    raise Unknown(f"line {line_of(k)}: an attribute does not close")
                k = e
                continue
            break
        if k >= len(code):
            raise Unknown(f"line {line_of(m.start())}: #[cfg] with no item after it")
        start = k
        line_text = code[start : code.find("\n", start) if "\n" in code[start:] else len(code)]
        is_item = bool(ITEM_RE.match(line_text))
        depth = parens = 0
        opened = False
        end = None
        while k < len(code):
            ch = code[k]
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    rest = re.match(r"\s*else\b", code[k + 1 :])
                    if rest:  # if ... else: the region goes on
                        k += 1 + rest.end()
                        opened = False
                        continue
                    end = k
                    break
            elif ch in "([":
                parens += 1
            elif ch in ")]":
                parens -= 1
            elif not opened and parens == 0 and depth == 0 and (ch == ";" or (ch == "," and not is_item)):
                end = k
                break
            k += 1
        if end is None:
            raise Unknown(f"line {line_of(m.start())}: the item under #[cfg] does not end")
        out.append((line_of(start), line_of(end), pred))
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
