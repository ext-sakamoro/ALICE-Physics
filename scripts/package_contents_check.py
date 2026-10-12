"""Checks the file list of the published .crate (`cargo package --list`).

Cargo.toml ships an `include` allowlist; this check fails when the list
  - is empty (nothing was compared),
  - contains a path under repository tooling or host-binding directories, or the
    toolchain pin, or
  - lacks a file the crate needs (manifest, lib root, licence, README).

run: python3 scripts/package_contents_check.py            (runs cargo package --list)
     python3 scripts/package_contents_check.py --list F   (reads a saved list)
"""
from __future__ import annotations

import argparse
import subprocess
import sys

# first path component that must not be shipped
FORBIDDEN_DIRS = (
    ".github",
    ".cargo",
    "scripts",
    "docs",
    "fuzz",
    "web",
    "unreal-plugin",
    "python",
    "bindings",
    "target",
)
FORBIDDEN_FILES = ("rust-toolchain.toml", "rust-toolchain")
REQUIRED = (
    "Cargo.toml",
    "src/lib.rs",
    "README.md",
    "LICENSE-AGPL",
    "LICENSE-COMMERCIAL.md",
)


def check(paths: list[str]) -> list[str]:
    errors: list[str] = []
    paths = [p.strip() for p in paths if p.strip()]
    if not paths:
        return ["package list is empty (0 files compared)"]
    for p in paths:
        top = p.split("/", 1)[0]
        if "/" in p and top in FORBIDDEN_DIRS:
            errors.append(f"shipped but not allowed: {p}")
        elif p in FORBIDDEN_FILES:
            errors.append(f"shipped but not allowed: {p}")
    have = set(paths)
    for req in REQUIRED:
        if req not in have:
            errors.append(f"required file missing from the package: {req}")
    return errors


def cargo_list() -> list[str]:
    out = subprocess.run(
        ["cargo", "package", "--list", "--locked", "--allow-dirty"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.splitlines()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", help="file holding `cargo package --list` output")
    args = ap.parse_args()
    if args.list:
        with open(args.list, encoding="utf-8") as f:
            paths = f.read().splitlines()
    else:
        paths = cargo_list()
    errors = check(paths)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    if errors:
        return 1
    print(f"package contents: {len([p for p in paths if p.strip()])} files, none forbidden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
