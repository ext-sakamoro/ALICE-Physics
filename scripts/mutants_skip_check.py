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

The files read are the crate's module tree, walked from `src/lib.rs` the
way the compiler does: `mod x;` (`x.rs` / `x/mod.rs`, under inline modules),
`#[path = "..."] mod x;` and `include!("...")`, wherever the target lies (the
mutation tool follows them out of `src/` too). A file declared from a
test-only line is test-only as a whole; one reached from any production line
is not. A declaration whose file is missing fails.

Attributes are read by bracket matching, not by one pattern: `#[mutants::skip]`,
`#[::mutants::skip]` and `#[cfg_attr(<any predicate>, ..., mutants::skip)]`
(nested `cfg_attr` too) are skips. An inner `#![...]` skip covers the whole
enclosing module or function and is refused. Every `mutants::skip` token in
the code has to belong to an attribute read this way, so a spelling this check
does not parse fails instead of passing unseen.

Fails when a skip is on code a non-test build compiles, when a skip is inner,
when a `mutants::skip` token is outside the attributes it reads, when a skip
is not in the ledger or a ledger line names no skip, when a reason is too
short, and when it compared nothing (no skip and no ledger line).

  python3 scripts/mutants_skip_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "scripts" / "mutants-skip-ledger.txt"
CRATE_ROOT = "src/lib.rs"
MOD_RE = re.compile(r"(?<![A-Za-z0-9_])mod\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*([;{])|[{}]")
VIS_MOD_DECL_RE = re.compile(r"(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*;")
PATH_META_RE = re.compile(r"\s*path\s*=\s*\"([^\"]*)\"\s*", re.S)
INCLUDE_RE = re.compile(r"(?<![A-Za-z0-9_])include\s*!\s*\(\s*\"")
ATTR_START_RE = re.compile(r"#\s*(!?)\s*\[")
SKIP_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])mutants\s*::\s*skip(?![A-Za-z0-9_])")
SKIP_PATH_RE = re.compile(r"(?:::\s*)?mutants\s*::\s*skip")
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
                out.add(ln)
                pending = False
    return out


def _close(code: str, open_at: int) -> int:
    """Index of the `]` that closes the `[` at `open_at` (-1 if none)."""
    depth = 0
    for k in range(open_at, len(code)):
        if code[k] in "[(":
            depth += 1
        elif code[k] in "])":
            depth -= 1
            if depth == 0:
                return k if code[k] == "]" else -1
    return -1


def is_skip_meta(meta: str) -> bool:
    """True when an attribute's content is `mutants::skip` (optionally `::`
    first) or a `cfg_attr(<any predicate>, ...)` whose attribute list holds
    one, at any depth of nested `cfg_attr`."""
    meta = meta.strip()
    if SKIP_PATH_RE.fullmatch(meta):
        return True
    m = re.fullmatch(r"cfg_attr\s*\((.*)\)", meta, re.S)
    if m:
        args = _split_args(m.group(1))
        return any(is_skip_meta(a) for a in args[1:])
    return False


def attributes(code: str) -> list[tuple[int, int, bool, str]]:
    """`(start, end, inner, content)` for every `#[...]` / `#![...]`."""
    found = []
    for m in ATTR_START_RE.finditer(code):
        open_at = m.end() - 1
        close = _close(code, open_at)
        if close < 0:
            continue
        found.append((m.start(), close + 1, m.group(1) == "!", code[open_at + 1 : close]))
    return found


def skips(path: Path, rel: str, whole_test_only: bool = False) -> tuple[list[tuple[str, int, bool]], list[str]]:
    """`(file::fn, line, test_only)` for each outer skip attribute in `path`,
    and the problems found while reading it.

    An inner `#![...]` skip covers the whole enclosing module or function, so
    it is refused outright: put `#[mutants::skip]` on the function instead.
    Every `mutants::skip` token in the code must belong to an attribute this
    reader recognised; one that does not (a form it does not parse) is a
    problem, so an unrecognised spelling fails instead of passing unseen.
    """
    text = path.read_text(encoding="utf-8")
    code = blank_non_code(text)
    test_only = test_only_lines(text)
    found, problems, recognised = [], [], 0
    for start, end, inner, meta in attributes(code):
        if not is_skip_meta(meta):
            continue
        recognised += len(SKIP_TOKEN_RE.findall(meta))
        line = code.count("\n", 0, start) + 1
        if inner:
            problems.append(f"{rel} (line {line}): inner #![...] mutants::skip skips the whole enclosing "
                            "module or function; put #[mutants::skip] on the function")
            continue
        fn = FN_RE.search(code, end)
        name = fn.group(1) if fn else "?"
        fn_line = code.count("\n", 0, fn.start()) + 1 if fn else line
        found.append((f"{rel}::{name}", line, whole_test_only or (line in test_only and fn_line in test_only)))
    tokens = len(SKIP_TOKEN_RE.findall(code))
    if tokens != recognised:
        problems.append(f"{rel}: {tokens} `mutants::skip` token(s) in the code, {recognised} in attributes "
                        "this check understands; spell the skip as #[mutants::skip] or "
                        "#[cfg_attr(<pred>, mutants::skip)]")
    return found, problems


def _skip_attrs(code: str, pos: int, attrs_at: dict[int, int]) -> int:
    """`pos` moved past whitespace and whole attributes."""
    while True:
        while pos < len(code) and code[pos].isspace():
            pos += 1
        if pos in attrs_at:
            pos = attrs_at[pos]
            continue
        return pos


def children(path: Path, text: str, code: str) -> tuple[list[tuple[Path, int]], list[str]]:
    """The files `path` brings into the build, each with the 1-based line of
    its declaration: `mod x;` (`x.rs` or `x/mod.rs` in the module directory,
    under any inline `mod a { ... }` it sits in), `#[path = "..."] mod x;`,
    and `include!("...")`. A declaration whose file is missing is a problem,
    so the walk cannot skip code the mutation tool reads."""
    problems: list[str] = []
    mod_rs = path.name in ("lib.rs", "main.rs", "mod.rs")
    module_dir = path.parent if mod_rs else path.parent / path.stem
    attrs = attributes(code)
    attrs_at = {start: end for start, end, inner, _ in attrs if not inner}
    paths_for: dict[int, str] = {}
    for start, end, inner, _ in attrs:
        m = PATH_META_RE.fullmatch(text[start:end].split("[", 1)[1][:-1])
        if inner or not m:
            continue
        at = _skip_attrs(code, end, attrs_at)
        decl = VIS_MOD_DECL_RE.match(code, at)
        if not decl:
            line = code.count("\n", 0, start) + 1
            problems.append(f"{path}: #[path] at line {line} is not on a `mod x;` this check can follow")
            continue
        paths_for[decl.start(1)] = m.group(1)
    found: list[tuple[Path, int]] = []
    stack: list[str | None] = []
    for m in MOD_RE.finditer(code):
        if m.group(0) == "{":
            stack.append(None)
        elif m.group(0) == "}":
            if stack:
                stack.pop()
        elif m.group(2) == "{":
            stack.append(m.group(1))
        else:
            line = code.count("\n", 0, m.start()) + 1
            inline = [n for n in stack if n is not None]
            base = module_dir.joinpath(*inline)
            if m.start(1) in paths_for:
                here = path.parent if not inline else base
                candidates = [here / paths_for[m.start(1)]]
            else:
                candidates = [base / f"{m.group(1)}.rs", base / m.group(1) / "mod.rs"]
            target = next((c for c in candidates if c.is_file()), None)
            if target is None:
                problems.append(f"{path} (line {line}): `mod {m.group(1)};` names no file "
                                f"({' or '.join(str(c) for c in candidates)})")
            else:
                found.append((target.resolve(), line))
    for m in INCLUDE_RE.finditer(code):
        close = code.find('"', m.end())
        line = code.count("\n", 0, m.start()) + 1
        target = path.parent / text[m.end():close]
        if not target.is_file():
            problems.append(f"{path} (line {line}): include! names no file ({target})")
        else:
            found.append((target.resolve(), line))
    return found, problems


def walk(root: Path) -> tuple[list[tuple[Path, bool]], list[str]]:
    """Every file of the crate's module tree from `src/lib.rs`, with whether
    the whole file is compiled only under `test` (declared from a test-only
    line, or inside a file that is)."""
    start = root / CRATE_ROOT
    if not start.is_file():
        return [], [f"{CRATE_ROOT} is missing: no module tree to read"]
    seen: dict[Path, bool] = {}
    problems: list[str] = []
    queue = [(start.resolve(), False)]
    while queue:
        path, test_only = queue.pop()
        # visit again only to downgrade test-only to production (any non-test
        # route compiles the file in a normal build)
        if path in seen and not (seen[path] and not test_only):
            continue
        seen[path] = test_only
        text = path.read_text(encoding="utf-8")
        code = blank_non_code(text)
        lines = test_only_lines(text)
        found, more = children(path, text, code)
        problems += more
        for child, line in found:
            queue.append((child, test_only or line in lines))
    return sorted(seen.items()), problems


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
    files, walk_problems = walk(root)
    errors.extend(walk_problems)
    resolved_root = root.resolve()
    for path, whole_test_only in files:
        try:
            rel = path.relative_to(resolved_root).as_posix()
        except ValueError:
            rel = path.as_posix()
        found_here, problems = skips(path, rel, whole_test_only)
        errors.extend(problems)
        for key, line, test_only in found_here:
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
