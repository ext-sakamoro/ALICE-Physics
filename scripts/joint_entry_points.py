#!/usr/bin/env python3
"""Every public PhysicsWorld method that reaches the joints is classified.

A step checks all joints at its start (`validate_joints`, through `check_step`);
a public method that reaches the joint solve, the island rebuilds or `joints`
some other way must either do the same or say why it needs no check. This scan
keeps that list complete, so a new entry point cannot skip the check unnoticed.

It reads the `impl PhysicsWorld` blocks of src/solver.rs and src/solver/*.rs
(comments and strings masked), builds the call graph of `self.<method>(` calls
between their methods, and marks a method as reaching the joints when its body
touches them directly (`self.joints`, `solve_joints*`, `break_overloaded_joints`,
`islands.union`) or calls a method that does. Every `pub fn` that reaches them
must be listed in scripts/joint-entry-points.txt as

  * `validated-at-entry`: its body calls `validate_joints`, `check_step`, or a
    method listed as `validated-at-entry` (checked), or
  * `n/a`: it reaches the joints without solving them by index (adding,
    removing, counting, serialising, restoring a checked snapshot), with the
    reason.

It fails on a reaching `pub fn` that is not listed, a listed method that no
longer exists or no longer reaches the joints, a `validated-at-entry` method
whose body calls no validator, an `n/a` without a reason, and when no method
was found (it compared nothing).

  python3 scripts/joint_entry_points.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIST = Path(__file__).resolve().parent / "joint-entry-points.txt"
TOUCH_RE = re.compile(r"self\.joints\b|\bsolve_joints\w*\s*\(|\bbreak_overloaded_joints\s*\(|islands\.union\s*\(")
CALL_RE = re.compile(r"\bself\.([a-z_][a-z0-9_]*)\s*\(")
FN_RE = re.compile(r"\b(pub(?:\([^)]*\))?\s+)?(?:const\s+)?fn\s+([a-z_][a-z0-9_]*)")
VALIDATORS = {"validate_joints", "check_step"}


def mask(src: str) -> str:
    """Comments, strings, raw strings and char literals blanked (newlines kept)."""
    out, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == "r" and (m := re.match(r'r(#*)"', src[i:])) and not (
                i and (src[i - 1].isalnum() or src[i - 1] == "_")):
            close = '"' + m.group(1)
            j = src.find(close, i + len(m.group(0)))
            j = n if j < 0 else j + len(close)
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c == "'" and (m := re.match(r"'(\\.[^']*|[^\\'])'", src[i:])):
            out.append(" " * len(m.group(0)))
            i += len(m.group(0))
        else:
            out.append(c)
            i += 1
    return "".join(out)


def block_end(code: str, brace: int) -> int:
    depth = 0
    for k in range(brace, len(code)):
        if code[k] == "{":
            depth += 1
        elif code[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    return len(code) - 1


def body_start(code: str, i: int) -> int:
    """Index of the `{` that opens a fn body after its signature, or -1 for a
    declaration ending in `;` (a `;` or `{` inside `( )` / `[ ]` does not count:
    `[u8; 32]` in a signature)."""
    depth = 0
    for k in range(i, len(code)):
        ch = code[k]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth == 0 and ch == "{":
            return k
        elif depth == 0 and ch == ";":
            return -1
    return -1


def methods(code: str) -> dict[str, tuple[bool, str]]:
    """`{name: (is pub, body)}` of the methods in the `impl PhysicsWorld` blocks."""
    out: dict[str, tuple[bool, str]] = {}
    for m in re.finditer(r"\bimpl\s+PhysicsWorld\s*\{", code):
        start = m.end() - 1
        end = block_end(code, start)
        inner = code[start + 1:end]
        k = 0
        while (f := FN_RE.search(inner, k)) is not None:
            brace = body_start(inner, f.end())
            if brace < 0:
                k = f.end()
                continue
            stop = block_end(inner, brace)
            is_pub = bool(f.group(1)) and f.group(1).strip() == "pub"
            out[f.group(2)] = (is_pub or out.get(f.group(2), (False, ""))[0],
                               out.get(f.group(2), (False, ""))[1] + inner[brace:stop + 1])
            k = stop + 1
    return out


def reaching(found: dict[str, tuple[bool, str]]) -> set[str]:
    """Methods whose body touches the joints or calls one that does."""
    reach = {name for name, (_, body) in found.items() if TOUCH_RE.search(body)}
    changed = True
    while changed:
        changed = False
        for name, (_, body) in found.items():
            if name not in reach and any(c in reach for c in CALL_RE.findall(body)):
                reach.add(name)
                changed = True
    return reach


def load_list(path: Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    entries: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = [p.strip() for p in raw.split("\t")]
        if len(parts) != 3 or parts[1] not in ("validated-at-entry", "n/a"):
            errors.append(f"line {n}: expected `name<TAB>validated-at-entry|n/a<TAB>reason`")
            continue
        if not parts[2]:
            errors.append(f"line {n}: `{parts[0]}` has no reason")
            continue
        entries[parts[0]] = (parts[1], parts[2])
    return entries, errors


def check(root: Path = ROOT, path: Path = LIST) -> tuple[list[str], int]:
    sources = [root / "src" / "solver.rs", *sorted((root / "src" / "solver").glob("*.rs"))]
    code = "\n".join(mask(p.read_text(encoding="utf-8")) for p in sources if p.is_file())
    found = methods(code)
    reach = reaching(found)
    public = sorted(n for n in reach if found[n][0])
    entries, errors = load_list(path)
    if not found:
        errors.append("no method of PhysicsWorld was found (checked nothing)")
    for name in public:
        if name not in entries:
            errors.append(f"pub fn `{name}` reaches the joints and is not classified in {path.name}")
    for name, (kind, _) in entries.items():
        if name not in found:
            errors.append(f"`{name}` is listed but no such method exists")
        elif name not in reach or not found[name][0]:
            errors.append(f"`{name}` is listed but is not a pub fn reaching the joints (remove it)")
        elif kind == "validated-at-entry":
            calls = set(CALL_RE.findall(found[name][1]))
            ok = calls & VALIDATORS or any(entries.get(c, ("",))[0] == "validated-at-entry" for c in calls)
            if not ok:
                errors.append(f"`{name}` is listed as validated-at-entry but calls no validator")
    return errors, len(public)


def main() -> int:
    errors, compared = check()
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: {compared} pub fn reaching the joints")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
