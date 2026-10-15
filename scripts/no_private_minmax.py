#!/usr/bin/env python3
"""The crate's float max / min come from alice-det-math, not from a local copy.

`max` / `min` / `max64` / `min64` with a defined zero sign and NaN policy are
alice-det-math's law; `src/lib.rs` re-exports them into `det_math`. A local
`fn max(a: f32, b: f32)` (or `min`, `max64`, `min64`, any parameter names,
`f32` or `f64`) anywhere in src/ is a second copy of that law that can drift
from it, so this check fails on one. Functions of the same name over other
types (`Fix128`, methods taking `&self`) are other laws and pass.

Comments and string literals are blanked first, so a copy inside them is not
counted and a `{` inside them does not hide a real one. The re-export in
`src/lib.rs` must be present, and scanning no file fails.

It matches by name and parameter type, so a renamed copy (`fn fmax`), a
closure, a `macro_rules!`, a tuple parameter or a fully generic
`fn max<T>(a: T, b: T)` is not caught here; the lib clippy pass still forbids
the std `f32::max` / `f64::min` those would have to call.

    python3 scripts/no_private_minmax.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("max", "min", "max64", "min64")
# `fn max(a: f32, b: f32)` / `fn min64 ( x : f64 , y : f64 )`, generics and
# visibility in front do not matter; a float type on the first parameter is
# what makes it a copy of the law
COPY = re.compile(
    # `r#max` is the same name; the generic list runs to the `(` so a nested
    # bound (`<T: Into<f32>>`) does not end it early
    r"\bfn\s+(?:r#)?(" + "|".join(NAMES) + r")\s*(<[^(]*>)?\s*\(\s*(mut\s+)?\w+\s*:\s*(?:&\s*(?:mut\s+)?)?(f32|f64)\b"
)
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


def check(root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    files = sorted((root / "src").rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        code = blank(path.read_text(encoding="utf-8"))
        for m in COPY.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            problems.append(
                f"{rel}:{line}: a local `fn {m.group(1)}` over {m.group(4)}: "
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
