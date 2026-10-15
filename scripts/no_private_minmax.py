#!/usr/bin/env python3
"""The crate's float max / min come from alice-det-math, not from a local copy.

`max` / `min` / `max64` / `min64` with a defined zero sign and NaN policy are
alice-det-math's law; `src/lib.rs` re-exports them into `det_math`. This
check denies the names: every `fn max` / `min` / `max64` / `min64` (also
written `r#max` etc.) anywhere in src/ fails, whatever its parameter types
are spelled as (`f32`, `core::primitive::f32`, an alias, a generic), unless

  * its first parameter is `self` / `&self` / `&mut self` (a method of a
    type, another law: `sketch` and `coupled_field` bounds), or
  * its parameter list names a type listed in OTHER_LAWS (today `Fix128`:
    the fixed-point max / min of `euler_fv`).

Matching the names rather than the float types is deliberate: every spelling
of a float parameter is caught without listing them. The parameter list is
read by bracket depth (generics with `Fn(..) -> ..` bounds and parameter
attributes included); a list that cannot be read fails.

Comments and string literals are blanked first. The re-export in `src/lib.rs`
must name all four (any order), and scanning no file fails.

    python3 scripts/no_private_minmax.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("max", "min", "max64", "min64")
# every function of these names, whatever its visibility, generics or
# parameter spelling; `is_allowed` lets methods and other laws through
NAME = re.compile(r"\bfn\s+(?:r#)?(" + "|".join(NAMES) + r")\b")
# non-float types whose own max / min is a different law
OTHER_LAWS = ("Fix128",)
SELF_FIRST = re.compile(r"^(?:mut\s+)?(?:&\s*(?:'\w+\s+)?(?:mut\s+)?)?self\b")
ATTRIBUTE = re.compile(r"#\s*\[[^\]]*\]")
REEXPORT = re.compile(r"pub\(crate\)\s+use\s+alice_det_math::\{([^}]*)\}")
STRING = re.compile(r'r(#*)"(?:.|\n)*?"\1|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])\'', re.S)
LINE_COMMENT = re.compile(r"//[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*(?:.|\n)*?\*/", re.S)


def blank(text: str) -> str:
    """Comments and string / char literals replaced by spaces (newlines kept)."""
    def keep_lines(m: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    text = BLOCK_COMMENT.sub(keep_lines, text)
    text = STRING.sub(keep_lines, text)
    return LINE_COMMENT.sub(keep_lines, text)


def params(code: str, pos: int) -> str | None:
    """The text inside the parameter list that follows `pos` (after the
    function name), skipping a generic list; None if it cannot be read."""
    i, n = pos, len(code)
    while i < n and code[i].isspace():
        i += 1
    if i < n and code[i] == "<":
        depth = 0
        while i < n:
            if code.startswith("->", i):
                i += 2
                continue
            ch = code[i]
            if ch in "<([":
                depth += 1
            elif ch in ">)]":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        while i < n and code[i].isspace():
            i += 1
    if i >= n or code[i] != "(":
        return None
    start, depth = i + 1, 0
    while i < n:
        if code.startswith("->", i):
            i += 2
            continue
        ch = code[i]
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
            if depth == 0:
                return code[start:i]
        i += 1
    return None


def is_allowed(plist: str) -> bool:
    """A method (first parameter self) or a function over another law's type."""
    first = ATTRIBUTE.sub(" ", plist).strip()
    if SELF_FIRST.match(first):
        return True
    return any(re.search(rf"\b{t}\b", plist) for t in OTHER_LAWS)


def check(root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    files = sorted((root / "src").rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        code = blank(path.read_text(encoding="utf-8"))
        for m in NAME.finditer(code):
            plist = params(code, m.end())
            if plist is not None and is_allowed(plist):
                continue
            line = code.count("\n", 0, m.start()) + 1
            why = "its parameter list could not be read" if plist is None else "not a method and no other-law type"
            problems.append(
                f"{rel}:{line}: a local `fn {m.group(1)}` ({why}): "
                "use the alice-det-math one re-exported in crate::det_math"
            )
    lib = root / "src" / "lib.rs"
    reexported: set[str] = set()
    if lib.is_file():
        for m in REEXPORT.finditer(blank(lib.read_text(encoding="utf-8"))):
            reexported |= {n.strip() for n in m.group(1).split(",")}
    if not set(NAMES) <= reexported:
        problems.append("src/lib.rs: the re-export of alice_det_math::{max, max64, min, min64} is missing")
    return problems, len(files)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    problems, files = check(ROOT)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if files == 0:
        print("error: compared nothing (no file under src/)", file=sys.stderr)
        return 1
    print(f"no_private_minmax: {files} files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
