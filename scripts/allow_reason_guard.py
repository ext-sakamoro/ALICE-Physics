#!/usr/bin/env python3
"""Every allow of `clippy::disallowed_methods` states why, and no wide allow
hides a clock read.

clippy.toml forbids the platform libm and the wall clock (`disallowed-methods`).
Tests, examples and benches may still call them where the result is not part
of the stepping law (a closed form in f64 to compare with, a printed timing),
but the allow that lets them through must say so, and must not be wider than
it has to be. Checked in src/, tests/, examples/ and benches/:

  * every `#[allow(..)]` / `#![allow(..)]` that names
    `clippy::disallowed_methods` carries the built-in `reason = "…"`;
  * the reason starts with one of KINDS and a colon, is at least
    MIN_REASON characters, and is not a placeholder after the kind
    (`todo`, `fixme`, `n/a`, `allow`, `oracle` alone and the like): it names
    the concrete reference, harness or table;
  * in src/, an allow outside a `#[cfg(test)]` module is production code and
    fails, except the counted entries of PRODUCTION (each with its reason);
  * an allow whose scope is a whole file, module, function or impl fails when
    that scope reads the clock (`Instant::now`, `SystemTime::now`,
    `.elapsed(`, `UNIX_EPOCH`): a clock read gets an allow of its own on the
    statement, so a wide libm allow never covers it;
  * scanning no file fails.

    python3 scripts/allow_reason_guard.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("src", "tests", "examples", "benches")
LINT = "clippy::disallowed_methods"
MIN_REASON = 30
KINDS = (
    "analytic reference",
    "independent reference",
    "timing harness",
    "measurement table",
    "measured quantity",
    "test input",
    "deprecated API",
)
PLACEHOLDERS = {"todo", "fixme", "n/a", "na", "tbd", "allow", "allowed", "oracle", "reference", "-", "x"}
# (file) -> (count, reason): allows in non-test src code
PRODUCTION: dict[str, tuple[int, str]] = {
    "src/privacy.rs": (2, "the deprecated clock-seeded XorShift64::from_entropy, documented as not private"),
}

ATTR = re.compile(r"#(!?)\s*\[\s*allow\s*\(")
REASON = re.compile(r'\breason\s*=\s*"((?:\\.|[^"\\])*)"')
CLOCK = re.compile(r"\b(?:Instant|SystemTime)\s*::\s*now\b|\.elapsed\s*\(|\bUNIX_EPOCH\b")
CFG_TEST = re.compile(r"#\s*\[\s*cfg\s*\(\s*(?:test\b|all\s*\(\s*test\b)")


def _strip(text: str, keep_strings: bool) -> str:
    """Comments (line and nested block) replaced by spaces, newlines kept;
    string and char literals kept or blanked. One left-to-right pass, so a
    quote inside a comment or a `//` inside a string is read as what it is."""
    out = list(text)
    n, i = len(text), 0

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
            continue
        if text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
            continue
        prev_ident = i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_")
        raw = re.match(r'b?r(#*)"', text[i:i + 260]) if c in "br" and not prev_ident else None
        if raw:
            close = '"' + raw.group(1)
            j = text.find(close, i + raw.end())
            j = n if j == -1 else j + len(close)
            if not keep_strings:
                blank(i, j)
            i = j
            continue
        if c == '"' or (c == "b" and not prev_ident and text.startswith('b"', i)):
            j = i + (2 if c == "b" else 1)
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            if not keep_strings:
                blank(i, j)
            i = j
            continue
        if c == "'":
            ch = re.match(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'", text[i:i + 16])
            if ch:
                if not keep_strings:
                    blank(i, i + ch.end())
                i += ch.end()
                continue
        i += 1
    return "".join(out)


def blank_comments(text: str) -> str:
    """Comments replaced by spaces; strings kept (they hold the reasons)."""
    return _strip(text, keep_strings=True)


def blank_all(text: str) -> str:
    """Comments and string / char literals replaced by spaces."""
    return _strip(text, keep_strings=False)


def matching(code: str, i: int, open_c: str, close_c: str) -> int:
    """Index of the bracket closing the one at `i` (code with strings blanked)."""
    depth = 0
    for j in range(i, len(code)):
        if code[j] == open_c:
            depth += 1
        elif code[j] == close_c:
            depth -= 1
            if depth == 0:
                return j
    return len(code) - 1


def line_of(code: str, i: int) -> int:
    return code.count("\n", 0, i) + 1


def test_ranges(code: str) -> list[tuple[int, int]]:
    """Character ranges of `#[cfg(test)]` modules (code fully blanked)."""
    out = []
    for m in CFG_TEST.finditer(code):
        rest = code[m.end():]
        mm = re.match(r"[^\n]*\)\s*\]\s*(?:#\s*\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{", rest)
        if mm:
            start = m.end() + mm.end() - 1
            out.append((m.start(), matching(code, start, "{", "}")))
    return out


def scope(code: str, attr_start: int, attr_end: int, inner: bool) -> tuple[int, int, bool]:
    """(start, end, wide) of what the attribute covers: the enclosing block for
    an inner attribute, else the next item; `wide` if that is a file, module,
    function or impl rather than a statement."""
    if inner:
        depth, j = 0, attr_start
        while j > 0:
            j -= 1
            if code[j] == "}":
                depth += 1
            elif code[j] == "{":
                if depth == 0:
                    return j, matching(code, j, "{", "}"), True
                depth -= 1
        return 0, len(code), True
    after = code[attr_end:]
    m = re.match(r"\s*(?:#\s*!?\s*\[[^\]]*\]\s*)*", after)
    head = after[m.end():]
    wide = bool(re.match(r"(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|async\s+|unsafe\s+)*(?:fn|mod|impl|trait)\b", head))
    k = attr_end + m.end()
    brace = code.find("{", k)
    semi = code.find(";", k)
    if brace == -1 or (semi != -1 and semi < brace and not wide):
        return k, semi if semi != -1 else len(code), wide
    return k, matching(code, brace, "{", "}"), wide


def judge_reason(reason: str | None) -> str | None:
    """Why `reason` is not acceptable, or None."""
    if reason is None:
        return "no `reason = \"…\"`"
    text = reason.strip()
    if len(text) < MIN_REASON:
        return f"reason shorter than {MIN_REASON} characters: {text!r}"
    kind = next((k for k in KINDS if text.lower().startswith(k.lower() + ":")), None)
    if kind is None:
        return f"reason does not start with a kind ({', '.join(KINDS)}) and a colon: {text!r}"
    rest = text[len(kind) + 1:].strip().lower().rstrip(".")
    if rest in PLACEHOLDERS or len(rest) < 8:
        return f"reason names no concrete reference after the kind: {text!r}"
    return None


def check(root: Path, production: dict[str, tuple[int, str]] | None = None) -> tuple[list[str], int, int]:
    production = PRODUCTION if production is None else production
    problems: list[str] = []
    files = sorted(p for d in DIRS if (root / d).is_dir() for p in (root / d).rglob("*.rs"))
    allows = 0
    prod_counts: dict[str, int] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        attr_code = blank_comments(text)
        code = blank_all(text)
        tests = test_ranges(code) if rel.startswith("src/") else []
        for m in ATTR.finditer(attr_code):
            paren = m.end() - 1
            end = matching(code, paren, "(", ")")
            inner_text = attr_code[paren:end + 1]
            if LINT not in inner_text:
                continue
            allows += 1
            where = f"{rel}:{line_of(code, m.start())}"
            r = REASON.search(inner_text)
            why = judge_reason(r.group(1) if r else None)
            if why:
                problems.append(f"{where}: {why}")
            close = code.find("]", end)
            s, e, wide = scope(code, m.start(), close + 1, m.group(1) == "!")
            if wide and CLOCK.search(code[s:e + 1]):
                problems.append(
                    f"{where}: this allow covers a whole file / module / function and a clock read "
                    f"(line {line_of(code, s + CLOCK.search(code[s:e + 1]).start())}): give the clock "
                    "statement an allow of its own"
                )
            if rel.startswith("src/") and not any(a <= m.start() <= b for a, b in tests):
                prod_counts[rel] = prod_counts.get(rel, 0) + 1
    for rel, n in sorted(prod_counts.items()):
        entry = production.get(rel)
        if entry is None:
            problems.append(f"{rel}: {n} allow(s) of {LINT} in production code (outside #[cfg(test)])")
        elif entry[0] != n:
            problems.append(f"{rel}: {n} production allow(s), PRODUCTION says {entry[0]}")
    for rel, (n, why) in sorted(production.items()):
        if not why.strip():
            problems.append(f"PRODUCTION entry {rel} has no reason")
        if rel not in prod_counts:
            problems.append(f"PRODUCTION entry {rel} ({n}) matches nothing: remove it")
    return problems, len(files), allows


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    problems, files, allows = check(ROOT)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if files == 0:
        print("error: compared nothing (no .rs file under src/ tests/ examples/ benches/)", file=sys.stderr)
        return 1
    print(f"allow_reason_guard: {files} files, {allows} allows of {LINT}, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
