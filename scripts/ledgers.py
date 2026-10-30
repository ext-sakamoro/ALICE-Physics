#!/usr/bin/env python3
"""The generated files of the repository and the commands that write them, in
one place, plus a check that reports every stale one at once.

GENERATED files are written whole by a generator; SEMI_GENERATED files are
hand-written with a generated column. scripts/land.py regenerates them when it
lands a branch (it imports this module), and the preflight runs `--check` near
its start, so a branch that adds a test file and a pub item learns about every
ledger it must regenerate from one run instead of one failed preflight per
ledger.

    python3 scripts/ledgers.py --write    regenerate every file
    python3 scripts/ledgers.py --check    regenerate, list every file that
                                          changed and every failing check-only
                                          script, restore the tree, fail if any
    --allow-missing-tools                 skip (and say so) a generator whose
                                          tool is not installed, instead of
                                          failing (the quick / fast preflight)

`--check` fails when it compared nothing.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NIGHTLY = "nightly-2026-09-26"  # the security-audit pin
NATIVE = "std,simd,parallel,ffi,gpu-solver-bridge"

GENERATED = (
    "docs/integration-status.md",
    "docs/integration-levels.md",
    "docs/oracle-status.md",
    "docs/wiring-status.md",
    "docs/PUBLIC_API_SNAPSHOT.txt",
    "docs/PUBLIC_API_SNAPSHOT.x86_64.txt",
    "docs/coverage/status.md",
)
# hand-written, with a generated column (Integration) that the regeneration
# rewrites: a conflict is merged with that column blanked, never dropped
SEMI_GENERATED = ("docs/MODULES.md",)

PY = sys.executable
# (name, argv, needs the SCIP index), in the order land.py has always run them
WRITERS: tuple[tuple[str, list[str], bool], ...] = (
    ("integration levels", [PY, "scripts/integration_levels.py", "--write"], True),
    ("integration status", [PY, "scripts/scip_reach.py", "--write", "docs/integration-status.md"], True),
    ("oracle status", [PY, "scripts/gen-oracle-status.py"], False),
    ("wiring status", [PY, "scripts/gen-wiring-status.py"], False),
    ("coverage status", [PY, "scripts/coverage_check.py", "--write-status"], False),
)
# the files each writer produces (a file counts as compared only when its writer ran)
OUTPUTS: dict[str, tuple[str, ...]] = {
    "integration levels": ("docs/integration-levels.md", "docs/MODULES.md"),
    "integration status": ("docs/integration-status.md",),
    "oracle status": ("docs/oracle-status.md",),
    "wiring status": ("docs/wiring-status.md",),
    "coverage status": ("docs/coverage/status.md",),
    "public API": ("docs/PUBLIC_API_SNAPSHOT.txt", "docs/PUBLIC_API_SNAPSHOT.x86_64.txt"),
}
INDEX = ["bash", "scripts/scip_index.sh"]
# the public API of each architecture, in its own file: `#[cfg(target_arch = ...)]` items (the x86_64
# SIMD methods) are in one and not the other. Both are generated with --target on any host (rustdoc's
# JSON needs the target's standard library, not a linker), so every machine compares both
PUBLIC_API_SNAPSHOTS: tuple[tuple[str, str], ...] = (
    ("aarch64-unknown-linux-gnu", "docs/PUBLIC_API_SNAPSHOT.txt"),
    ("x86_64-unknown-linux-gnu", "docs/PUBLIC_API_SNAPSHOT.x86_64.txt"),
)


def public_api_argv(target: str) -> list[str]:
    return ["cargo", f"+{NIGHTLY}", "public-api", "--target", target, "--features", NATIVE, "--simplified"]
# generated parts checked in place (README tables, version strings, target lists)
CHECK_ONLY: tuple[tuple[str, list[str]], ...] = (
    ("README f32 module row", [PY, "scripts/f32_modules.py", "--check"]),
    ("README / MODULES sync", [PY, "scripts/readme_sync.py", "--check"]),
    ("version sync", [PY, "scripts/version_sync.py", "--check"]),
    ("fuzz targets", [PY, "scripts/fuzz_targets.py", "--check"]),
    ("binding build commands", [PY, "scripts/binding_build_commands.py", "--check"]),
)


def have_index_tool() -> bool:
    if shutil.which("rust-analyzer") is None:
        return False
    return subprocess.run(["rust-analyzer", "--version"], capture_output=True).returncode == 0


def have_public_api_tool(root: Path) -> bool:
    """cargo-public-api runs on the pinned nightly, and that nightly has every snapshot's target"""
    r = subprocess.run(["cargo", f"+{NIGHTLY}", "public-api", "--version"], cwd=root, capture_output=True)
    if r.returncode != 0:
        return False
    t = subprocess.run(["rustup", "target", "list", "--installed", "--toolchain", NIGHTLY],
                       capture_output=True, text=True)
    return t.returncode == 0 and {target for target, _ in PUBLIC_API_SNAPSHOTS} <= set(t.stdout.split())


def write_public_api(root: Path) -> None:
    for target, path in PUBLIC_API_SNAPSHOTS:
        r = subprocess.run(public_api_argv(target), cwd=root, capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            raise RuntimeError(f"cargo +{NIGHTLY} public-api --target {target} failed "
                               f"(rustup target add --toolchain {NIGHTLY} {target}): {r.stderr.strip()[-300:]}")
        (root / path).write_text(r.stdout, encoding="utf-8")


def regenerate(root: Path, allow_missing: bool) -> tuple[list[str], list[str], list[str]]:
    """Run every writer. (errors, skipped, names of the writers that ran)"""
    errors: list[str] = []
    skipped: list[str] = []
    ran: list[str] = []
    index = have_index_tool()
    if index:
        r = subprocess.run(INDEX, cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            errors.append(f"SCIP index: {r.stderr.strip()[-300:]}")
            index = False
    elif allow_missing:
        skipped.append("SCIP index (rust-analyzer not installed)")
    else:
        errors.append("SCIP index: rust-analyzer not installed")
    for name, argv, needs_index in WRITERS:
        if needs_index and not index:
            skipped.append(f"{name} (no SCIP index)")
            continue
        r = subprocess.run(argv, cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            errors.append(f"{name}: {(r.stderr or r.stdout).strip()[-300:]}")
        else:
            ran.append(name)
    if have_public_api_tool(root):
        try:
            write_public_api(root)
            ran.append("public API")
        except RuntimeError as e:
            errors.append(f"public API: {e}")
    elif allow_missing:
        skipped.append(f"public API (cargo +{NIGHTLY} public-api or its targets not installed: "
                       + f"rustup target add --toolchain {NIGHTLY} " + " ".join(t for t, _ in PUBLIC_API_SNAPSHOTS) + ")")
    else:
        errors.append(f"public API: cargo +{NIGHTLY} public-api or its targets not installed ("
                      + f"rustup target add --toolchain {NIGHTLY} " + " ".join(t for t, _ in PUBLIC_API_SNAPSHOTS) + ")")
    return errors, skipped, ran


def check(root: Path, allow_missing: bool) -> tuple[list[str], list[str], int]:
    """(problems, skipped, files compared); the tree is restored afterwards."""
    listed = GENERATED + SEMI_GENERATED
    missing = [p for p in listed if not (root / p).is_file()]
    before = {p: (root / p).read_bytes() for p in listed if p not in missing}
    problems = [f"{p} is missing: generate it (python3 scripts/ledgers.py --write)" for p in missing]
    compared = 0
    try:
        errors, skipped, ran = regenerate(root, allow_missing)
        problems += errors
        produced = {f for name in ran for f in OUTPUTS.get(name, ())}
        for p in before:
            if p not in produced:
                continue  # its writer was skipped or failed: not compared
            compared += 1
            if (root / p).read_bytes() != before[p]:
                problems.append(f"{p} is stale: regenerate it (python3 scripts/ledgers.py --write)")
    finally:
        for p, data in before.items():
            (root / p).write_bytes(data)
        for p in missing:
            (root / p).unlink(missing_ok=True)
    for name, argv in CHECK_ONLY:
        r = subprocess.run(argv, cwd=root, capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            problems.append(f"{name} ({' '.join(argv[1:])}): " + " / ".join(tail))
    return problems, skipped, compared + len(CHECK_ONLY)


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--allow-missing-tools", action="store_true")
    a = ap.parse_args()
    if a.write:
        errors, skipped, _ = regenerate(ROOT, a.allow_missing_tools)
        for s in skipped:
            print(f"skipped: {s}", file=sys.stderr)
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1 if errors else 0
    problems, skipped, compared = check(ROOT, a.allow_missing_tools)
    for s in skipped:
        print(f"skipped: {s}", file=sys.stderr)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if compared == 0:
        print("error: compared nothing (no generated file and no check)", file=sys.stderr)
        return 1
    print(f"ledgers: {compared} generated files and checks compared, {len(problems)} stale or failing")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
