#!/usr/bin/env python3
"""ALICE-Law G2: every max / min written as a comparison in library code has a reviewed entry.

clippy's disallowed-methods bans `f32::max` / `f32::min` / `f64::max` / `f64::min` and
`f32::clamp` / `f64::clamp` in library code, but a max or min written as a comparison is not
a method call and the lint does not see it. `scripts/g2-rewrite-scan` (a syntax-tree scan,
so formatting, line breaks, comments, expression operands and a `!` around the condition do
not hide a site; `#[cfg(test)]` items are skipped) reports in `src/`:

* `expr`:   `if a > b { a } else { b }` (`<`, `>`, `<=`, `>=`, either branch order)
* `assign`: `if x < lo { x = lo; }` (the assignment anywhere in the block)
* `raw`:    a function named `raw_*`, the named helpers that call a platform operation

Every finding has an entry in the reviewed table `scripts/g2-rewrite-sites.tsv`, keyed by
file, enclosing function, shape and operands (no line numbers), with one of:

* `fixed-point`: the operands are `Fix128` (an integer representation: no signed zero, no
  NaN), so the comparison is the Law's result
* `integer`: the operands are a primitive integer type, for the same reason
* `equivalent`: a float comparison that gives the Law's result for every operand the site
  can see (the reason says why)
* `selection`: an argmax / argmin that picks an element (and its payload), not a value; the
  rule is written in alice-law-conformance.toml [G2]
* `pending`: a float site that still waits for a Law helper (the reason names it)

The operand types were proven with the compiler: each operand wrapped in an identity
function taking `Fix128` and `cargo check --lib` with the CI feature set naming every
operand that is not one. Out of scope: orderings (`partial_cmp` / `sort_by` / `min_by`),
which are ordering rules, not a max or min value.

The run fails on a finding that is not listed, on a listed entry that no longer matches,
when the scan read no file (whatever `--min-sites` says), and when it found fewer sites than
`--min-sites` (default 1: a library with comparison sites that reports none has a broken
scan). `--min-sites 0` is accepted only when alice-law-conformance.toml has
`[G2] no_comparison_sites_reason` (why the library has no float min / max site by
construction); without it the run fails, so 0 is not a quiet way out of the check.

usage: g2_rewrite_check.py [--root DIR] [--src DIR]... [--table TSV] [--scan-bin PATH]
                           [--min-sites N]
"""
from __future__ import annotations

import argparse
import os
import tomllib
import subprocess
import sys
from pathlib import Path

STATUSES = ("fixed-point", "integer", "equivalent", "selection", "pending")
SHAPES = ("expr", "assign", "raw")
HERE = Path(__file__).resolve().parent


def scan(root: Path, scan_bin: str | None, dirs: list[str]) -> tuple[list[tuple[str, str, str, str]], str]:
    paths = [str(root / d) for d in dirs]
    if scan_bin:
        cmd = [scan_bin, *paths]
    else:
        cmd = ["cargo", "run", "-q", "--release", "--locked",
               "--manifest-path", str(HERE / "g2-rewrite-scan" / "Cargo.toml"), "--", *paths]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise SystemExit(f"g2_rewrite_check: FAIL: the scan exited {p.returncode}: {p.stderr.strip()}")
    rows = []
    prefix = str(root).replace("\\", "/").rstrip("/") + "/"
    for line in p.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) != 6 or cols[2] not in SHAPES:
            raise SystemExit(f"g2_rewrite_check: FAIL: unexpected scan line {line!r}")
        file = cols[0].replace("\\", "/")
        if file.startswith(prefix):
            file = file[len(prefix):]
        rows.append((file, cols[1], cols[2], cols[3]))
    return rows, p.stderr.strip()


def load_table(path: Path) -> dict[tuple[str, str, str, str], tuple[str, str]]:
    table = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 6:
            raise SystemExit(f"{path}:{n}: expected 6 tab-separated columns, got {len(cols)}")
        file, fn, shape, operands, status, reason = cols
        if shape not in SHAPES:
            raise SystemExit(f"{path}:{n}: shape must be one of {SHAPES}, got {shape!r}")
        if status not in STATUSES:
            raise SystemExit(f"{path}:{n}: status must be one of {STATUSES}, got {status!r}")
        if len(reason.strip()) < 12:
            raise SystemExit(f"{path}:{n}: the reason must say why (12 characters or more)")
        key = (file, fn, shape, operands)
        if key in table:
            raise SystemExit(f"{path}:{n}: duplicate entry {key}")
        table[key] = (status, reason)
    return table


def no_sites_reason(root: Path) -> str:
    """[G2] no_comparison_sites_reason in the repository's alice-law-conformance.toml, or ''"""
    manifest = root / "alice-law-conformance.toml"
    if not manifest.is_file():
        return ""
    with manifest.open("rb") as f:
        g2 = tomllib.load(f).get("G2", {})
    return str(g2.get("no_comparison_sites_reason", "")).strip()


def check(root: Path, table_path: Path, scan_bin: str | None, dirs: list[str], min_sites: int) -> int:
    if min_sites < 0:
        print("g2_rewrite_check: FAIL: --min-sites must be 0 or more")
        return 1
    if min_sites == 0 and len(no_sites_reason(root)) < 12:
        print("g2_rewrite_check: FAIL: --min-sites 0 needs [G2] no_comparison_sites_reason "
              "(12 characters or more) in alice-law-conformance.toml: why the library has no "
              "float min / max site by construction")
        return 1
    rows, summary = scan(root, scan_bin, dirs)
    parts = summary.split()
    files = int(parts[1]) if len(parts) == 3 and parts[0] == "scanned" else 0
    if files == 0:
        print(f"g2_rewrite_check: FAIL: the scan read no file under {dirs} (0 scanned)")
        return 1
    if len(rows) < min_sites:
        print(f"g2_rewrite_check: FAIL: {files} files scanned and {len(rows)} sites found, "
              f"fewer than --min-sites {min_sites} (the scan is broken or the tree moved)")
        return 1
    found: dict[tuple[str, str, str, str], int] = {}
    for key in rows:
        found[key] = found.get(key, 0) + 1
    table = load_table(table_path)
    problems = [("not reviewed: " + "\t".join(k)) for k in sorted(found) if k not in table]
    problems += [("listed but not found (remove it): " + "\t".join(k)) for k in sorted(table) if k not in found]
    by = {s: sum(found[k] for k in found if k in table and table[k][0] == s) for s in STATUSES}
    print(f"g2_rewrite_check: {files} files scanned; {len(rows)} sites ("
          + ", ".join(f"{by[s]} {s}" for s in STATUSES) + f", {len(problems)} problems)")
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--table", default=None)
    ap.add_argument("--scan-bin", default=os.environ.get("G2_SCAN_BIN"))
    ap.add_argument("--src", action="append", default=None,
                    help="directory under --root to scan (repeatable, default: src)")
    ap.add_argument("--min-sites", type=int, default=1,
                    help="fail when fewer sites are found (default 1); 0 only with [G2] "
                         "no_comparison_sites_reason in alice-law-conformance.toml")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    table = Path(args.table) if args.table else root / "scripts" / "g2-rewrite-sites.tsv"
    return check(root, table, args.scan_bin, args.src or ["src"], args.min_sites)


if __name__ == "__main__":
    sys.exit(main())
