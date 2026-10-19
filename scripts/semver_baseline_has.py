#!/usr/bin/env python3
"""Exit 0 when <package> is a package of the workspace at git revision <rev>, 1 when it is
not, 2 when the revision's root manifest cannot be read.

The workspace members are read from the root Cargo.toml at <rev> (`[workspace] members`,
literal paths) plus the root package itself. Used by security-audit.yml's semver job: a
package that does not exist at the baseline (the commit that introduces it) has nothing to
be compared with.

usage: semver_baseline_has.py <rev> <package>
"""
from __future__ import annotations

import subprocess
import sys
import tomllib


def show(rev: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def packages(rev: str) -> set[str] | None:
    root = show(rev, "Cargo.toml")
    if root is None:
        return None
    doc = tomllib.loads(root)
    names = {doc["package"]["name"]} if "package" in doc else set()
    for member in doc.get("workspace", {}).get("members", []):
        if member in (".", ""):
            continue
        text = show(rev, f"{member.rstrip('/')}/Cargo.toml")
        if text is not None:
            m = tomllib.loads(text)
            if "package" in m:
                names.add(m["package"]["name"])
    return names


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    names = packages(argv[0])
    if names is None:
        print(f"semver_baseline_has: no Cargo.toml at {argv[0]}", file=sys.stderr)
        return 2
    return 0 if argv[1] in names else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
