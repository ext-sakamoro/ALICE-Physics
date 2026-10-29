#!/usr/bin/env python3
"""The commit a CI check compares with ("the base"), and the rule that named it.

Shared file: the canonical copy is ALICE-DetMath scripts/ci_base.py; every other repository carries it
byte for byte (the shared-files check compares the copies), so change it there and copy it out.

On a push to main the base is the commit main stood at before the push (`github.event.before`), so a
push of several commits is judged against what main had, not against an earlier commit of the same push
(those never stood on main). It must be a full SHA, not all zeros, a commit this clone has and an
ancestor of HEAD; anything else fails (exit 2), with no fallback. Anywhere else (ci/ branches, pull
requests, a local run) the base is the merge-base of HEAD with origin/main, or HEAD's parent when HEAD is
origin/main itself. The rule is one of `before`, `merge-base` and `HEAD~1`.

    python3 scripts/ci_base.py [--event E --ref R --before SHA]

prints the base SHA on stdout and `base <SHA> (rule <rule>, <event ref>)` on stderr; a base that cannot
be named prints the reason on stderr and exits 2. Run it after `git fetch origin main`, in a checkout
with the history (`fetch-depth: 0`).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

ZERO = "0" * 40


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8")


def ci_base(event: str, ref: str, before: str) -> tuple[str, str]:
    """The base and the rule that named it (`before`, `merge-base` or `HEAD~1`, see the module doc);
    raises ValueError when it cannot be named"""
    if event == "push" and ref == "refs/heads/main":
        if not re.fullmatch(r"[0-9a-f]{40}", before or ""):
            raise ValueError(f"a push to main needs github.event.before as a full SHA, got {before!r}")
        if before == ZERO:
            raise ValueError("a push to main with an all-zero before (main has no previous commit to compare with)")
        if git("cat-file", "-e", f"{before}^{{commit}}").returncode != 0:
            raise ValueError(f"before {before} is not in this clone (fetch it, or check out with fetch-depth 0)")
        if git("merge-base", "--is-ancestor", before, "HEAD").returncode != 0:
            raise ValueError(f"before {before} is not an ancestor of HEAD (main was not fast-forwarded)")
        return before, "before"
    mb = git("merge-base", "HEAD", "origin/main")
    if mb.returncode != 0:
        raise ValueError(f"no merge-base of HEAD with origin/main: {mb.stderr.strip()}")
    base = mb.stdout.strip()
    if base == git("rev-parse", "HEAD").stdout.strip():
        parent = git("rev-parse", "HEAD~1")
        if parent.returncode != 0:
            raise ValueError("HEAD is origin/main and has no parent to compare with")
        return parent.stdout.strip(), "HEAD~1"
    return base, "merge-base"


def describe(base: str, rule: str, event: str, ref: str) -> str:
    where = " ".join(x for x in (event, ref) if x) or "local"
    return f"base {base} (rule {rule}, {where})"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--event", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--before", default="")
    a = ap.parse_args(argv)
    try:
        base, rule = ci_base(a.event, a.ref, a.before)
    except ValueError as e:
        print(f"ci_base: {e}", file=sys.stderr)
        return 2
    print(f"ci_base: {describe(base, rule, a.event, a.ref)}", file=sys.stderr)
    print(base)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
