#!/usr/bin/env python3
"""Before landing: new source files must bring tests that the lib coverage counts.

The per-push coverage ratchet (security-audit.yml) measures `cargo llvm-cov
--lib`, which runs only the unit tests inside src/. A new module tested only
from tests/ lands at 0 % lib coverage and turns the ratchet red after the land.
This gate catches it in seconds, without running the coverage:

  * new file   every `src/**/*.rs` added by the commits being landed must hold a
               `#[cfg(test)]` module, or bring a tests/ oracle into the lib tests
               with `#[path = "../tests/…"]`; otherwise it is an error. A file
               that needs neither (a `mod.rs` that only re-exports) is listed in
               scripts/lib-test-exempt.txt with a reason.
  * new fn     a `pub fn` / `pub(crate) fn` the commits add to an existing file
               whose `#[cfg(test)]` part never names it is a warning (indirectly
               tested helpers would make an error too noisy).

When the commits add no source file the gate says so and passes; when they add
some, each one is examined (the count is printed), so an empty discovery cannot
pass by comparing nothing.

Usage: `python3 scripts/lib_test_gate.py BASE` (BASE = the commit the landed
commits sit on, e.g. `origin/main`); exit 1 on an error.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

EXEMPT = os.path.join("scripts", "lib-test-exempt.txt")
TEST_MOD = re.compile(r"#\[cfg\(\s*(?:all\(\s*)?test\b")
PATH_ORACLE = re.compile(r'#\[path\s*=\s*"[^"]*\btests/[^"]+\.rs"\]')
NEW_FN = re.compile(r"^\+\s*pub(?:\(crate\))?\s+(?:const\s+)?(?:unsafe\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")


def git(root: str, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def exempt(root: str) -> dict[str, str]:
    path = os.path.join(root, EXEMPT)
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            file, _, reason = line.partition(" ")
            out[file] = reason.strip()
    return out


def has_lib_tests(text: str) -> bool:
    return bool(TEST_MOD.search(text) or PATH_ORACLE.search(text))


def test_part(text: str) -> str:
    m = TEST_MOD.search(text)
    return text[m.start():] if m else ""


def check(root: str, base: str) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    counts = {"new_files": 0, "new_fns": 0}
    ex = exempt(root)
    for f, reason in ex.items():
        if not reason:
            errors.append(f"{EXEMPT}: {f} has no reason")
    status = git(root, "diff", "--name-status", f"{base}...HEAD", "--", "src")
    added = [l.split("\t", 1)[1] for l in status.splitlines() if l.startswith("A\t") and l.endswith(".rs")]
    modified = [l.split("\t", 1)[1] for l in status.splitlines() if l.startswith("M\t") and l.endswith(".rs")]
    for rel in added:
        counts["new_files"] += 1
        if rel in ex:
            continue
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            text = f.read()
        if not has_lib_tests(text):
            errors.append(
                f"{rel}: new source file without lib tests: add a #[cfg(test)] module, or "
                f'bring its tests/ oracle in with #[path = "../tests/<file>.rs"] (cargo llvm-cov '
                f"--lib does not run tests/, so the coverage ratchet fails after the land), "
                f"or list it in {EXEMPT} with a reason"
            )
    for rel in modified:
        diff = git(root, "diff", "-U0", f"{base}...HEAD", "--", rel)
        names = [m.group(1) for line in diff.splitlines() if (m := NEW_FN.match(line))]
        if not names:
            continue
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            tests = test_part(f.read())
        for name in names:
            counts["new_fns"] += 1
            if not re.search(r"\b" + re.escape(name) + r"\b", tests):
                warnings.append(f"{rel}: new `{name}` is not named in the file's #[cfg(test)] part")
    return errors, warnings, counts


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", help="the commit the landed commits sit on (e.g. origin/main)")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args(argv)
    errors, warnings, counts = check(args.root, args.base)
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error: {e}")
    if counts["new_files"] == 0:
        print("lib-test gate: the commits add no source file (new-file check skipped); "
              f"new pub fns examined: {counts['new_fns']}")
    else:
        print(f"lib-test gate: new source files examined: {counts['new_files']}, "
              f"new pub fns examined: {counts['new_fns']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
