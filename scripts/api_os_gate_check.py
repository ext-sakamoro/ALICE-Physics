#!/usr/bin/env python3
"""No public item of the crates is gated on the operating system.

security-audit.yml checks the public API snapshot (docs/PUBLIC_API_SNAPSHOT.txt) on an
aarch64 Linux runner, and the developers' machines are aarch64 Macs. That is the same
snapshot only while no public item depends on the OS: an item under
`#[cfg(target_os = ...)]`, `target_family`, `target_vendor`, `target_env`, `unix` or
`windows` (as `cfg` or `cfg_attr`) would exist on one and not the other. This script reads
every `.rs` file under src/ and crates/*/src/ and fails on
  * an OS `cfg` attribute whose item is `pub` (plain `pub`, not `pub(crate)` /
    `pub(super)` / `pub(in ...)`): the next line that is not another attribute, a comment
    or blank;
  * a file-level `#![cfg(...)]` naming the OS in a file that has a plain `pub` item.
An OS gate on a private item (a test, a helper) does not change the public surface and
passes; `cfg!(...)` inside a body is an expression, not a gate, and passes.
Exit 1 on a finding, 2 when no file was read.
    python3 scripts/api_os_gate_check.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OS_WORDS = r"(?:target_os|target_family|target_vendor|target_env|\bunix\b|\bwindows\b)"
OUTER = re.compile(r"^\s*#\[\s*(?:cfg|cfg_attr)\s*\(.*" + OS_WORDS)
INNER = re.compile(r"^\s*#!\[\s*(?:cfg|cfg_attr)\s*\(.*" + OS_WORDS)
SKIP = re.compile(r"^\s*(?:#\[|///|//|/\*|\*|$)")
PUBLIC = re.compile(r"^\s*pub\s")


def sources(root: Path) -> list[Path]:
    dirs = [root / "src", *sorted((root / "crates").glob("*/src"))]
    return sorted(p for d in dirs if d.is_dir() for p in d.rglob("*.rs"))


def findings(path: Path, rel: str) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    if any(INNER.match(l) for l in lines) and any(PUBLIC.match(l) for l in lines):
        out.append(f"{rel}: a file-level OS cfg on a file with public items")
    for i, line in enumerate(lines):
        if not OUTER.match(line):
            continue
        j = i + 1
        while j < len(lines) and SKIP.match(lines[j]):
            j += 1
        if j < len(lines) and PUBLIC.match(lines[j]):
            out.append(f"{rel}:{i + 1}: an OS cfg on a public item ({lines[j].strip()[:60]})")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args(argv)
    files = sources(args.root)
    if not files:
        print("error: no .rs file under src/ or crates/*/src/ (compared nothing)", file=sys.stderr)
        return 2
    found = [f for p in files for f in findings(p, p.relative_to(args.root).as_posix())]
    for f in found:
        print(f"error: {f}", file=sys.stderr)
    print(f"compared: rust files {len(files)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
