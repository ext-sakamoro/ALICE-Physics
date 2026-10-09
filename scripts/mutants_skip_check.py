#!/usr/bin/env python3
"""Check every `#[mutants::skip]` against its ledger, and that none is on code
the library itself compiles.

cargo-mutants does not filter "delete field X from struct ... expression"
mutants by `--re`, `--exclude-re` or `exclude_re` (measured with 27.1.0), so a
test helper whose struct literal cannot be observed on an axis keeps reporting
missed there. `#[mutants::skip]` on the function does remove them. It is a
hole in the measurement, so each one is allowed only on test-only code (inside
a `#[cfg(test)]` or `#[cfg(all(test, ...))]` item or module) and must be listed
with its reason in `scripts/mutants-skip-ledger.txt`:

    src/solver.rs::build_one_contact_world<TAB>reason (12 characters or more)

Fails when a skip is on code a non-test build compiles, when a skip is not in
the ledger or a ledger line names no skip, when a reason is too short, and
when it compared nothing (no skip and no ledger line).

  python3 scripts/mutants_skip_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "scripts" / "mutants-skip-ledger.txt"
SCANNED = ("src", "tests/lib")
SKIP_RE = re.compile(r"#\[\s*(?:cfg_attr\s*\([^\]]*?,\s*)?mutants\s*::\s*skip\s*\)?\s*\]")
CFG_RE = re.compile(r"#\[\s*cfg\s*\((.*)\)\s*\]\s*$")
FN_RE = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")
MIN_REASON = 12


def blank_non_code(text: str) -> str:
    """`text` with comment, string and char contents replaced by spaces."""
    out = list(text)
    n, i = len(text), 0

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
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
        elif text[i] == '"':
            k = i + 1
            while k < n and text[k] != '"':
                k += 2 if text[k] == "\\" else 1
            blank(i + 1, k)
            i = k + 1
        elif text[i] == "'":
            m = re.match(r"'(\\u\{[0-9a-fA-F]+\}|\\.|[^\\'\n])'", text[i:])
            if m:
                blank(i + 1, i + m.end() - 1)
                i += m.end()
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def _split_args(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts]


def needs_test(pred: str) -> bool:
    """True when the cfg predicate is false in every build without `test`:
    `test`, or `all(...)` with an argument that needs `test`."""
    pred = pred.strip()
    if pred == "test":
        return True
    m = re.fullmatch(r"all\((.*)\)", pred, re.S)
    if m:
        return any(needs_test(a) for a in _split_args(m.group(1)))
    m = re.fullmatch(r"any\((.*)\)", pred, re.S)
    if m:
        args = _split_args(m.group(1))
        return bool(args) and all(needs_test(a) for a in args)
    return False


def test_only_lines(text: str) -> set[int]:
    """1-based lines inside a block (`mod`, `fn`, `impl`, ...) whose own
    attributes include a `cfg` that needs `test`."""
    code = blank_non_code(text)
    lines = code.split("\n")
    out: set[int] = set()
    pending = False  # a test-only cfg seen on the attributes above
    stack: list[bool] = []  # per open brace: inside a test-only block
    for ln, line in enumerate(lines, 1):
        stripped = line.strip()
        inside = any(stack)
        if inside:
            out.add(ln)
        m = CFG_RE.match(stripped)
        if m and needs_test(m.group(1)):
            pending = True
            out.add(ln)
            continue
        if stripped.startswith("#[") or not stripped:
            if pending:
                out.add(ln)
            continue
        for ch in line:
            if ch == "{":
                stack.append(pending or (stack[-1] if stack else False))
                if pending:
                    out.add(ln)
                pending = False
            elif ch == "}":
                if stack:
                    stack.pop()
            elif ch == ";" and pending:
                # an item without a block (`use ...;`, `mod x;`)
                pending = False
    return out


def skips(path: Path, rel: str) -> list[tuple[str, int, bool]]:
    """`(file::fn, line, test_only)` for each skip attribute in `path`."""
    text = path.read_text(encoding="utf-8")
    code = blank_non_code(text)
    test_only = test_only_lines(text)
    found = []
    for m in SKIP_RE.finditer(code):
        line = code.count("\n", 0, m.start()) + 1
        fn = FN_RE.search(code, m.end())
        name = fn.group(1) if fn else "?"
        fn_line = code.count("\n", 0, fn.start()) + 1 if fn else line
        found.append((f"{rel}::{name}", line, line in test_only and fn_line in test_only))
    return found


def read_ledger(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.is_file():
        return entries
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        key, _, reason = raw.partition("\t")
        entries[key.strip()] = reason.strip()
    return entries


def check(root: Path, ledger_path: Path) -> list[str]:
    errors: list[str] = []
    found: dict[str, tuple[int, bool]] = {}
    for top in SCANNED:
        for path in sorted((root / top).rglob("*.rs")):
            rel = path.relative_to(root).as_posix()
            for key, line, test_only in skips(path, rel):
                found[key] = (line, test_only)
                if not test_only:
                    errors.append(f"{key} (line {line}): #[mutants::skip] on code a non-test build "
                                  "compiles; observe it with a test instead")
    ledger = read_ledger(ledger_path)
    if not found and not ledger:
        errors.append("no #[mutants::skip] and no ledger line: compared nothing")
    for key in sorted(found.keys() - ledger.keys()):
        errors.append(f"{key}: #[mutants::skip] not in {ledger_path.name}")
    for key in sorted(ledger.keys() - found.keys()):
        errors.append(f"{key}: ledger line names no #[mutants::skip]")
    for key, reason in sorted(ledger.items()):
        if key in found and len(reason) < MIN_REASON:
            errors.append(f"{key}: reason shorter than {MIN_REASON} characters")
    print(f"compared: {len(found)} skip(s), {len(ledger)} ledger line(s)")
    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    errors = check(ROOT, LEDGER)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
