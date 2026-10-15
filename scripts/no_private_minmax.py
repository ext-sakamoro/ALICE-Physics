#!/usr/bin/env python3
"""The crate's float max / min come from alice-det-math, not from a local copy.

`max` / `min` / `max64` / `min64` with a defined zero sign and NaN policy are
alice-det-math's law; `src/lib.rs` re-exports them into `det_math`. This
check denies the names: every `fn max` / `min` / `max64` / `min64` (also
written `r#max` etc.) in src/ is counted per file, and each (file, name)
must match an entry of ALLOWED exactly, with a reason. A new function of
those names, in any form (free, method, trait impl, any parameter types),
therefore fails until the list is edited and reviewed; a count that drifts
either way fails, an entry that matches nothing fails, an entry without a
reason fails, and scanning no file fails.

No parameter or receiver is inspected: the decision is the name and the
reviewed list, not the spelling of a type.

Comments and string literals are blanked first. The re-export in `src/lib.rs`
must name all four (any order).

    python3 scripts/no_private_minmax.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("max", "min", "max64", "min64")
# every function of these names, whatever its visibility, receiver, generics
# or parameter spelling
NAME = re.compile(r"\bfn\s+(?:r#)?(" + "|".join(NAMES) + r")\b")
# (file, name) -> (count, reason): the functions of these names that are not
# a copy of alice-det-math's law
ALLOWED: dict[tuple[str, str], tuple[int, str]] = {
    ("src/euler_fv.rs", "min"): (1, "fixed-point Fix128 min of the finite-volume limiter, another law"),
    ("src/euler_fv.rs", "max"): (1, "fixed-point Fix128 max of the finite-volume limiter, another law"),
    ("src/coupled_field.rs", "min"): (1, "getter of the field's lower bound (a Fix128 triple), not a comparison"),
    ("src/coupled_field.rs", "max"): (1, "getter of the field's upper bound (a Fix128 triple), not a comparison"),
    ("src/sketch.rs", "min"): (1, "getter of the smallest value the sketch has seen, not a comparison"),
    ("src/sketch.rs", "max"): (1, "getter of the largest value the sketch has seen, not a comparison"),
}
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


def check(root: Path, allowed: dict[tuple[str, str], tuple[int, str]] | None = None) -> tuple[list[str], int]:
    allowed = ALLOWED if allowed is None else allowed
    problems: list[str] = []
    counts: dict[tuple[str, str], list[int]] = {}
    files = sorted((root / "src").rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        code = blank(path.read_text(encoding="utf-8"))
        for m in NAME.finditer(code):
            counts.setdefault((rel, m.group(1)), []).append(code.count("\n", 0, m.start()) + 1)
    for key, lines in sorted(counts.items()):
        where = ", ".join(f"{key[0]}:{n}" for n in lines)
        entry = allowed.get(key)
        if entry is None:
            problems.append(
                f"{where}: a local `fn {key[1]}`: use the alice-det-math one re-exported in "
                "crate::det_math (or, if it is another law, add a reviewed ALLOWED entry)"
            )
        elif entry[0] != len(lines):
            problems.append(f"`fn {key[1]}` in {key[0]}: {len(lines)} found, ALLOWED says {entry[0]} ({where})")
    for key, (count, reason) in sorted(allowed.items()):
        if not reason.strip():
            problems.append(f"ALLOWED entry `fn {key[1]}` in {key[0]} has no reason")
        if key not in counts:
            problems.append(f"ALLOWED entry `fn {key[1]}` in {key[0]} ({count}) matches nothing: remove it")
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
