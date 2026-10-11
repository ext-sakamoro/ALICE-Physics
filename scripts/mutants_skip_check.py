#!/usr/bin/env python3
"""Check every `#[mutants::skip]` against its ledger, and that none is on code
the library itself compiles.

cargo-mutants does not filter "delete field X from struct ... expression"
mutants by `--re`, `--exclude-re` or `exclude_re` (measured with 27.1.0), so a
test helper whose struct literal cannot be observed on an axis keeps reporting
missed there. `#[mutants::skip]` on the function does remove them. It is a
hole in the measurement, so each one is allowed only on test-only code (inside
a `#[cfg(test)]` or `#[cfg(all(test, ...))]` item or module, or a file declared
from one) and must be listed with its reason in `scripts/mutants-skip-ledger.txt`:

    src/solver.rs::build_one_contact_world<TAB>reason (12 characters or more)

The source is read by a Rust parser (`syn`, `scripts/cfg-items --mutants-skips`),
the same one the in-diff mutation plan uses, so comments, raw strings, byte and
C strings and char literals are lexed as the compiler lexes them. It reports
every `mutants::skip` attribute (`::` first allowed, and `cfg_attr(<any
predicate>, ...)` holding one at any depth) with the `cfg(...)` of the node and
everything around it, every `mod x;` with its `#[path]`, every `include!`, and
the count of `mutants :: skip` token runs anywhere in the file.

The files read are the crate's module tree, walked from `src/lib.rs` the way
the compiler does, wherever a `#[path]` or `include!` leads (the mutation tool
follows them out of `src/` too). A file declared from a test-only route is
test-only as a whole; one reached from any production route is not.

Fails when a file does not parse, when a declaration names no file, when a
skip is on code a non-test build compiles, when a skip is inner (`#![...]`
covers a whole module or function), when the file holds more `mutants :: skip`
token runs than skip attributes (a spelling or place the parser does not report,
such as a macro body), when a skip is not in the ledger or a ledger line names
no skip, when a reason is too short, and when it compared nothing.

  python3 scripts/mutants_skip_check.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "scripts" / "mutants-skip-ledger.txt"
CRATE_ROOT = "src/lib.rs"
MIN_REASON = 12
ROUTE_SEP = "\x1e"


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
    `test`, or `all(...)` with an argument that needs `test`, or `any(...)`
    whose every argument does. Whitespace from the parser's token printing
    (`all (test , feature = "std")`) is ignored."""
    pred = re.sub(r"\s+", "", pred)
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


def route_needs_test(route: str) -> bool:
    return any(needs_test(p) for p in route.split(ROUTE_SEP) if p.strip())


@dataclass
class Parsed:
    """What `cfg-items --mutants-skips` reported for one file."""
    skips: list[tuple[int, str, bool, str]] = field(default_factory=list)  # line, name, inner, route
    mods: list[tuple[int, str, str, str, str]] = field(default_factory=list)  # line, name, path, inline, route
    includes: list[tuple[int, str, str]] = field(default_factory=list)  # line, literal, route
    inline_paths: list[tuple[int, str, str]] = field(default_factory=list)  # line, name, path
    tokens: int = -1


def helper() -> Path:
    """The parser binary (built once by the in-diff plan's builder)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mutants_in_diff_plan  # noqa: PLC0415

    return mutants_in_diff_plan.helper()


def parse(exe: Path, path: Path) -> tuple[Parsed | None, str]:
    r = subprocess.run([str(exe), "--mutants-skips", str(path)], capture_output=True, text=True,
                       encoding="utf-8", check=False)
    if r.returncode != 0:
        return None, f"{path}: the parser refused the file: {r.stderr.strip()}"
    out = Parsed()
    # split on "\n" only: str.splitlines() also breaks at ROUTE_SEP (U+001E)
    for line in r.stdout.split("\n"):
        if not line:
            continue
        kind, rest = line.split("\t", 1)
        f = rest.split("\t")
        if kind == "skip":
            out.skips.append((int(f[1]), f[2], f[3] == "1", f[4] if len(f) > 4 else ""))
        elif kind == "mod":
            out.mods.append((int(f[1]), f[2], f[3], f[4], f[5] if len(f) > 5 else ""))
        elif kind == "include":
            out.includes.append((int(f[1]), f[2], f[3] if len(f) > 3 else ""))
        elif kind == "pathinline":
            out.inline_paths.append((int(f[1]), f[2], f[3]))
        elif kind == "tokens":
            out.tokens = int(f[1])
    if out.tokens < 0:
        return None, f"{path}: the parser printed no token count"
    return out, ""


def children(path: Path, parsed: Parsed) -> tuple[list[tuple[Path, bool]], list[str]]:
    """The files `path` brings into the build, each with whether its
    declaration's route needs `test`: `mod x;` (`x.rs` or `x/mod.rs` in the
    module directory, under any inline `mod a { ... }`), `#[path = "..."] mod x;`
    (relative to the file's directory, or to the inline module's directory),
    and `include!("...")` (relative to the file's directory). A declaration
    whose file is missing is a problem."""
    problems: list[str] = []
    mod_rs = path.name in ("lib.rs", "main.rs", "mod.rs")
    module_dir = path.parent if mod_rs else path.parent / path.stem
    found: list[tuple[Path, bool]] = []
    for line, name, path_attr, inline, route in parsed.mods:
        parts = [p for p in inline.split("/") if p]
        base = module_dir.joinpath(*parts)
        if path_attr:
            candidates = [(base if parts else path.parent) / path_attr]
        else:
            candidates = [base / f"{name}.rs", base / name / "mod.rs"]
        target = next((c for c in candidates if c.is_file()), None)
        if target is None:
            problems.append(f"{path} (line {line}): `mod {name};` names no file "
                            f"({' or '.join(str(c) for c in candidates)})")
        else:
            found.append((target.resolve(), route_needs_test(route)))
    for line, name, path_attr in parsed.inline_paths:
        problems.append(f"{path} (line {line}): #[path = \"{path_attr}\"] on the inline `mod {name} {{ }}` "
                        "moves where its modules are looked up, which this check does not follow")
    for line, literal, route in parsed.includes:
        target = path.parent / literal if literal else None
        if target is None or not target.is_file():
            problems.append(f"{path} (line {line}): include! names no file ({literal or 'not one string literal'})")
        else:
            found.append((target.resolve(), route_needs_test(route)))
    return found, problems


def walk(root: Path, exe: Path) -> tuple[list[tuple[Path, bool, Parsed]], list[str]]:
    """Every file of the crate's module tree from `src/lib.rs`, whether the whole
    file is compiled only under `test`, and what the parser reported for it."""
    start = root / CRATE_ROOT
    if not start.is_file():
        return [], [f"{CRATE_ROOT} is missing: no module tree to read"]
    seen: dict[Path, bool] = {}
    parsed: dict[Path, Parsed] = {}
    problems: list[str] = []
    queue = [(start.resolve(), False)]
    while queue:
        path, test_only = queue.pop()
        # visit again only to downgrade test-only to production (any non-test
        # route compiles the file in a normal build)
        if path in seen and not (seen[path] and not test_only):
            continue
        seen[path] = test_only
        if path not in parsed:
            got, problem = parse(exe, path)
            if got is None:
                problems.append(problem)
                continue
            parsed[path] = got
        found, more = children(path, parsed[path])
        problems += more
        for child, declared_test_only in found:
            queue.append((child, test_only or declared_test_only))
    return [(p, t, parsed[p]) for p, t in sorted(seen.items()) if p in parsed], problems


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


def check(root: Path, ledger_path: Path, exe: Path | None = None) -> list[str]:
    errors: list[str] = []
    found: dict[str, tuple[int, bool]] = {}
    files, walk_problems = walk(root, exe or helper())
    errors.extend(walk_problems)
    resolved_root = root.resolve()
    for path, whole_test_only, parsed in files:
        try:
            rel = path.relative_to(resolved_root).as_posix()
        except ValueError:
            rel = path.as_posix()
        if parsed.tokens != len(parsed.skips):
            errors.append(f"{rel}: {parsed.tokens} `mutants :: skip` token run(s), {len(parsed.skips)} skip "
                          "attribute(s); spell the skip as #[mutants::skip] or "
                          "#[cfg_attr(<pred>, mutants::skip)] on an item")
        for line, name, inner, route in parsed.skips:
            key = f"{rel}::{name}"
            if inner:
                errors.append(f"{key} (line {line}): inner #![...] mutants::skip skips the whole enclosing "
                              "module or function; put #[mutants::skip] on the function")
                continue
            test_only = whole_test_only or route_needs_test(route)
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
    print(f"compared: {len(found)} skip(s), {len(ledger)} ledger line(s), {len(files)} file(s) read")
    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    errors = check(ROOT, LEDGER)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
