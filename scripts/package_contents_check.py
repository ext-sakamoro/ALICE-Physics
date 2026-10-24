"""Checks the file list of the published .crate (`cargo package --list`).

Cargo.toml ships an `include` allowlist; this check fails when the list
  - is empty (nothing was compared), or no Cargo target or no Rust source was
    compared,
  - contains a path under repository tooling or host-binding directories, or the
    toolchain pin,
  - lacks a file the crate needs (manifest, lib root, licence, README),
  - lacks a Cargo target (lib, test, bench, example) that `cargo metadata` reports,
  - lacks a file that a packaged Rust source pulls in with `include!` /
    `include_str!` / `include_bytes!` or `#[path = "..."]` (resolved against the
    source's directory; the crate or its tests would not compile from crates.io),
  - lacks a `tests/fixtures/...` file that a packaged source names in a string
    literal, or any git-tracked file under a `tests/fixtures/...` directory it names

Both published packages are checked: the engine (alice-physics-core, the repository root)
and the facade (alice-physics, crates/alice-physics).

run: python3 scripts/package_contents_check.py                      (cargo package --list, both)
     python3 scripts/package_contents_check.py --package P --list F (reads a saved list of P)
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the published packages and their directories (relative to ROOT)
PACKAGES = {"alice-physics-core": ".", "alice-physics": "crates/alice-physics"}

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
    "LICENSE-Proprietary",
)

EMBED_RE = re.compile(
    r'(?:\binclude(?:_str|_bytes)?!\s*\(\s*"([^"]+)"\s*\))|(?:#\[path\s*=\s*"([^"]+)"\s*\])'
)
# fixtures that tests open at run time (`CARGO_MANIFEST_DIR` + this path)
FIXTURE_RE = re.compile(r'"/?(tests/fixtures(?:/[^"]*[^"/])?)/?"')


def embedded_paths(rel: str, text: str) -> list[str]:
    """Package-relative paths that `rel` embeds; a path that leaves the package
    keeps its leading `../`, so it can never match a listed file."""
    out = []
    for m in EMBED_RE.finditer(text):
        target = m.group(1) or m.group(2)
        out.append(posixpath.normpath(posixpath.join(posixpath.dirname(rel), target)))
    return out


def check(
    paths: list[str],
    targets: list[str],
    sources: dict[str, str],
    tracked: list[str],
) -> tuple[list[str], dict[str, int]]:
    """(errors, counts compared)

    paths: `cargo package --list` lines; targets: package-relative source path of
    every Cargo target; sources: text of each packaged `.rs` file; tracked: files
    git tracks (to expand a fixture directory).
    """
    errors: list[str] = []
    counts = {"files": 0, "targets": 0, "sources": 0, "embeds": 0, "fixtures": 0}
    paths = [p.strip() for p in paths if p.strip()]
    if not paths:
        return ["package list is empty (0 files compared)"], counts
    counts["files"] = len(paths)
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
    for t in targets:
        counts["targets"] += 1
        if t not in have:
            errors.append(f"Cargo target missing from the package: {t}")
    if counts["targets"] == 0:
        errors.append("no Cargo target was compared")
    for rel, text in sorted(sources.items()):
        counts["sources"] += 1
        for dep in embedded_paths(rel, text):
            counts["embeds"] += 1
            if dep not in have:
                errors.append(f"{rel} embeds {dep}, which is not in the package")
        for fixture in FIXTURE_RE.findall(text):
            under = [f for f in tracked if f.startswith(fixture + "/")]
            for f in under or [fixture]:
                counts["fixtures"] += 1
                if f not in have:
                    errors.append(f"{rel} reads {f}, which is not in the package")
    if counts["sources"] == 0:
        errors.append("no packaged Rust source was scanned")
    return errors, counts


def cargo_list(package: str) -> list[str]:
    out = subprocess.run(
        ["cargo", "package", "--list", "--locked", "--allow-dirty", "-p", package],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return [line.replace("\\", "/") for line in out.stdout.splitlines()]


def cargo_targets(package: str) -> list[str]:
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    pkg = next(p for p in json.loads(out)["packages"] if p["name"] == package)
    base = os.path.dirname(pkg["manifest_path"])
    return sorted(
        os.path.relpath(t["src_path"], base).replace(os.sep, "/") for t in pkg["targets"]
    )


def git_tracked(directory: str) -> list[str]:
    """tracked files, relative to the package directory"""
    return subprocess.run(
        ["git", "ls-files"], cwd=os.path.join(ROOT, directory), check=True, capture_output=True, text=True
    ).stdout.splitlines()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", help="file holding `cargo package --list` output (needs --package)")
    ap.add_argument("--package", choices=sorted(PACKAGES), help="check this package only")
    args = ap.parse_args()
    if args.list and not args.package:
        ap.error("--list needs --package")
    failed = False
    for package in [args.package] if args.package else list(PACKAGES):
        directory = PACKAGES[package]
        if args.list:
            with open(args.list, encoding="utf-8") as f:
                paths = f.read().splitlines()
        else:
            paths = cargo_list(package)
        sources = {}
        for rel in (p.strip() for p in paths):
            if rel.endswith(".rs"):
                with open(os.path.join(ROOT, directory, rel), encoding="utf-8") as f:
                    sources[rel] = f.read()
        errors, counts = check(paths, cargo_targets(package), sources, git_tracked(directory))
        for e in errors:
            print(f"ERROR: {package}: {e}", file=sys.stderr)
        if errors:
            failed = True
            continue
        print(
            f"package contents ({package}): {counts['files']} files, none forbidden; "
            f"{counts['targets']} Cargo targets present; {counts['sources']} sources scanned, "
            f"{counts['embeds']} embedded paths and {counts['fixtures']} fixture paths present"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
