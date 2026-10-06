#!/usr/bin/env python3
"""Plan an in-diff mutation run so that the changed code is compiled and tested.

`cargo mutants --in-diff` mutates the changed lines, but a module behind
`#[cfg(feature = "...")]` in src/lib.rs is not compiled unless that feature is
on: its mutants change nothing and every one is reported missed. And `-- --lib`
runs only the unit tests, so the integration oracles in tests/ never see the
mutants. This script reads the diff and prints, as JSON:

  * `features`: the features to add so every changed module compiles (taken
    from the `cfg` attributes on its `mod` line in src/lib.rs, plus the
    `required-features` of the tests below);
  * `tests`: the integration test targets whose file name contains the name of
    a changed module (`tests/audit_replay.rs` for `src/replay.rs`), to run
    besides `--lib`;
  * `skipped`: changed files whose module cannot be compiled on the host at
    all (it requires `not(feature = "std")`, or a wasm / python build); they
    are removed from the diff written to `--out`, and listed so the run says so.

  python3 scripts/mutants_in_diff_plan.py change.diff --features "parallel" --out planned.diff
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# features that cannot be built for the host test run
HOST_UNBUILDABLE = {"wasm", "python"}
FILE_RE = re.compile(r"^\+\+\+ b/(\S+)$", re.M)
MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z0-9_]+)\s*;")
CFG_RE = re.compile(r"^\s*#\[cfg\((.*)\)\]\s*$")
FEATURE_RE = re.compile(r'feature\s*=\s*"([^"]+)"')


def module_gates(lib_rs: str) -> dict[str, str]:
    """`{module: cfg expression}` for every `mod` line of lib.rs with a cfg above it."""
    gates: dict[str, str] = {}
    pending: list[str] = []
    for line in lib_rs.splitlines():
        m = CFG_RE.match(line)
        if m:
            pending.append(m.group(1))
            continue
        mm = MOD_RE.match(line)
        if mm and pending:
            gates[mm.group(1)] = " , ".join(pending)
        if line.strip() and not line.strip().startswith(("#[", "//")):
            pending = []
    return gates


def features_for(cfg: str) -> tuple[list[str], bool]:
    """(features to turn on, buildable on the host) for a cfg expression.

    Positive `feature = ".."` terms are collected; inside `any(..)` the first
    is enough. A `not(feature = ...)` or a host-unbuildable feature makes the
    module unbuildable for the host test run."""
    if "not(" in cfg:
        return [], False
    if cfg.strip().startswith("any("):
        names = FEATURE_RE.findall(cfg)[:1]
    else:
        names = FEATURE_RE.findall(cfg)
    if any(n in HOST_UNBUILDABLE for n in names):
        return [], False
    return [n for n in names if n != "std"], True


def required_features(cargo_toml: str) -> dict[str, list[str]]:
    """`{test name: required-features}` from the `[[test]]` tables of Cargo.toml."""
    out: dict[str, list[str]] = {}
    for block in re.split(r"^\[\[test\]\]\s*$", cargo_toml, flags=re.M)[1:]:
        block = re.split(r"^\[", block, maxsplit=1, flags=re.M)[0]
        name = re.search(r'^\s*name\s*=\s*"([^"]+)"', block, re.M)
        req = re.search(r"^\s*required-features\s*=\s*\[([^\]]*)\]", block, re.M)
        if name and req:
            out[name.group(1)] = re.findall(r'"([^"]+)"', req.group(1))
    return out


def module_of(path: str) -> str | None:
    """The top-level module of a src/ file (`src/a.rs` and `src/a/b.rs` -> `a`)."""
    p = Path(path)
    if p.parts[:1] != ("src",) or p.suffix != ".rs" or len(p.parts) < 2:
        return None
    name = p.parts[1]
    if len(p.parts) == 2:
        name = p.stem
    return None if name in ("lib", "main") else name


def split_diff(diff: str) -> list[tuple[str, str]]:
    """`[(path, text of that file's diff)]` for a unified diff."""
    parts = re.split(r"(?=^diff --git )", diff, flags=re.M)
    out = []
    for part in parts:
        m = FILE_RE.search(part)
        if m:
            out.append((m.group(1), part))
    return out


def plan(diff: str, base_features: list[str], root: Path = ROOT) -> dict:
    lib_rs = (root / "src/lib.rs").read_text(encoding="utf-8")
    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
    gates = module_gates(lib_rs)
    req = required_features(cargo)
    tests_dir = root / "tests"
    test_names = sorted(p.stem for p in tests_dir.glob("*.rs")) if tests_dir.is_dir() else []
    features = set(base_features)
    tests: set[str] = set()
    kept, skipped = [], []
    for path, text in split_diff(diff):
        module = module_of(path)
        if module is None:
            kept.append(text)
            continue
        feats, buildable = features_for(gates.get(module, ""))
        if not buildable:
            skipped.append(path)
            continue
        features.update(feats)
        kept.append(text)
        token = re.compile(rf"(^|_){re.escape(module)}(_|$)")
        for t in test_names:
            if token.search(t):
                need = req.get(t, [])
                if any(n in HOST_UNBUILDABLE for n in need):
                    continue
                features.update(n for n in need if n != "std")
                tests.add(t)
    return {
        "features": sorted(f for f in features if f),
        "tests": sorted(tests),
        "skipped": skipped,
        "diff": "".join(kept),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("diff")
    ap.add_argument("--features", default="", help="comma-separated features of the matrix entry")
    ap.add_argument("--out", required=True, help="where to write the diff with unbuildable files removed")
    args = ap.parse_args(argv)
    base = [f for f in args.features.split(",") if f.strip()]
    p = plan(Path(args.diff).read_text(encoding="utf-8"), base)
    Path(args.out).write_text(p.pop("diff"), encoding="utf-8")
    print(json.dumps(p))
    for f in p["skipped"]:
        print(f"note: {f} is not buildable on the host test run; not mutated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
