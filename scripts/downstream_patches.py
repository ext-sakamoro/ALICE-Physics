#!/usr/bin/env python3
"""The `--config` patches that point a fetched downstream copy's alice-physics packages at this
checkout, one for every source its Cargo.lock takes them from.

A downstream crate may take alice-physics from crates.io or from a git repository (ALICE-SDF pins
the ALICE-Physics repository at a rev), and cargo applies a patch only to the source it names:
`patch.crates-io.<name>` leaves a git package as it is, and `patch."<url>".<name>` leaves a
registry one. So the patches are read from the lock, not written for one source: every package of
the given names, per source, gets one line

    patch.crates-io.alice-physics.path="<dir>"
    patch."https://github.com/ext-sakamoro/ALICE-Physics".alice-physics-core.path="<dir>"

(the git key is the source's URL without `git+`, the query and the fragment). A source that is
neither a registry nor git, no Cargo.lock, a lock that does not parse, or no package of the given
names at all (compared nothing) fails with exit 1. With `--path-ok` a lock whose packages of the
given names are all path packages (a consumer that takes alice-physics by path, already placed next
to this checkout) prints nothing and exits 0; a lock with none of the names still fails.

usage: downstream_patches.py <downstream dir> [--path-ok] --package <name>=<dir> [--package <name>=<dir> ...]
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path


def source_key(source: str) -> str:
    """The patch table a lock source belongs to: `crates-io` for the registry, the quoted URL for git"""
    if source.startswith("registry+") or source.startswith("sparse+"):
        url = source.split("+", 1)[1]
        if url not in ("https://github.com/rust-lang/crates.io-index", "https://index.crates.io/"):
            raise ValueError(f"a registry other than crates.io: {source}")
        return "crates-io"
    if source.startswith("git+"):
        url = source[len("git+"):].split("#", 1)[0].split("?", 1)[0]
        if not url or '"' in url:
            raise ValueError(f"an unreadable git source: {source}")
        return f'"{url}"'
    raise ValueError(f"neither a registry nor a git source: {source}")


def path_packages(lock_text: str, packages: dict[str, str]) -> int:
    """How many packages of the given names the lock takes by path (no `source`)"""
    lock = tomllib.loads(lock_text)
    return sum(1 for pkg in lock.get("package", []) if pkg.get("name") in packages and "source" not in pkg)


def patches(lock_text: str, packages: dict[str, str]) -> list[str]:
    lock = tomllib.loads(lock_text)
    out: list[str] = []
    for pkg in lock.get("package", []):
        name = pkg.get("name")
        if name not in packages or "source" not in pkg:
            continue  # another package, or a path package (nothing to patch)
        line = f'patch.{source_key(pkg["source"])}.{name}.path="{packages[name]}"'
        if line not in out:
            out.append(line)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--package", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--path-ok", action="store_true",
                    help="a lock that takes the packages only by path needs no patch (exit 0, no output)")
    a = ap.parse_args(argv)
    packages = {}
    for spec in a.package:
        name, sep, directory = spec.partition("=")
        if not sep or not name or not directory:
            print(f"downstream_patches: --package must be NAME=DIR, got {spec!r}", file=sys.stderr)
            return 1
        packages[name] = directory
    if not packages:
        print("downstream_patches: no --package given (compared nothing)", file=sys.stderr)
        return 1
    lock = Path(a.dir) / "Cargo.lock"
    try:
        text = lock.read_text(encoding="utf-8")
        found = patches(text, packages)
        by_path = path_packages(text, packages)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as e:
        print(f"downstream_patches: {lock}: {e}", file=sys.stderr)
        return 1
    if not found and a.path_ok and by_path:
        return 0
    if not found:
        print(f"downstream_patches: {lock} takes none of {sorted(packages)} from a registry or git "
              "(compared nothing)", file=sys.stderr)
        return 1
    print("\n".join(found))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
