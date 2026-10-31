#!/usr/bin/env python3
"""Every float to `Fix128` conversion in library code has a reviewed entry.

`Fix128::from_f64` / `from_f32` and `Vec3Fix::from_f32` saturate a finite value outside
`[-2^63, 2^63)` and refuse a non-finite one only in debug builds (NaN gives zero in release),
so a site that can see a NaN or an infinity has to check first (`try_from_*`), propagate it,
or show it cannot arise. `scripts/g2-rewrite-scan --float-to-fix` (a syntax-tree scan; calls,
the constructors named as a function value and calls inside macro arguments; `#[cfg(test)]`
items are skipped) lists them, and every one has an entry in the reviewed table
`scripts/fix-conversion-sites.tsv`, keyed by file, enclosing function, callee, form, arguments
and kind (no line numbers), with the number of such sites and one of:

* `core`: the conversion functions themselves (`src/math.rs`)
* `constant`: every argument is a literal or a named constant, so the value is finite
* `checked`: the argument is shown finite (and in range where it matters) just before
* `finite-by-construction`: the value is finite for every input the site can see (the
  reason says why)
* `pending-binding`, `pending-sdf`, `pending-input`, `pending-computed`: a site that still
  converts an unchecked float: a binding argument, an SDF distance or normal, data handed
  in by the caller, or a value computed in the crate (the reason names the change)

The check fails on a site with no entry, an entry whose site is gone, a count that differs,
a bad row, and a scan that read no file or found fewer sites than `--min-sites`.

usage: fix_conversion_sites.py [--root DIR] [--src DIR]... [--table TSV] [--scan-bin PATH]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

STATUSES = ("core", "constant", "checked", "finite-by-construction",
            "pending-binding", "pending-sdf", "pending-input", "pending-computed")
FORMS = ("call", "fn-value", "macro-body", "macro-unparsed")
KINDS = ("constant", "value")
HERE = Path(__file__).resolve().parent

Key = tuple[str, str, str, str, str, str]


def scan(root: Path, scan_bin: str | None, dirs: list[str]) -> tuple[list[Key], str]:
    paths = [str(root / d) for d in dirs]
    if scan_bin:
        cmd = [scan_bin, "--float-to-fix", *paths]
    else:
        cmd = ["cargo", "run", "-q", "--release", "--locked",
               "--manifest-path", str(HERE / "g2-rewrite-scan" / "Cargo.toml"), "--",
               "--float-to-fix", *paths]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise SystemExit(f"fix_conversion_sites: FAIL: the scan exited {p.returncode}: {p.stderr.strip()}")
    rows = []
    prefix = str(root).replace("\\", "/").rstrip("/") + "/"
    for line in p.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) != 7 or cols[3] not in FORMS or cols[5] not in KINDS:
            raise SystemExit(f"fix_conversion_sites: FAIL: unexpected scan line {line!r}")
        file = cols[0].replace("\\", "/")
        if file.startswith(prefix):
            file = file[len(prefix):]
        rows.append((file, cols[1], cols[2], cols[3], cols[4], cols[5]))
    return rows, p.stderr.strip()


def load_table(path: Path) -> dict[Key, tuple[int, str, str]]:
    table = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 9:
            raise SystemExit(f"{path}:{n}: expected 9 tab-separated columns, got {len(cols)}")
        file, fn, callee, form, args, kind, count, status, reason = cols
        if form not in FORMS:
            raise SystemExit(f"{path}:{n}: form must be one of {FORMS}, got {form!r}")
        if kind not in KINDS:
            raise SystemExit(f"{path}:{n}: kind must be one of {KINDS}, got {kind!r}")
        if not count.isdigit() or int(count) < 1:
            raise SystemExit(f"{path}:{n}: count must be a positive integer, got {count!r}")
        if status not in STATUSES:
            raise SystemExit(f"{path}:{n}: status must be one of {STATUSES}, got {status!r}")
        if status == "constant" and kind != "constant":
            raise SystemExit(f"{path}:{n}: status constant needs kind constant (the scan saw a value)")
        if len(reason.strip()) < 12:
            raise SystemExit(f"{path}:{n}: the reason must say why (12 characters or more)")
        key = (file, fn, callee, form, args, kind)
        if key in table:
            raise SystemExit(f"{path}:{n}: duplicate entry {key}")
        table[key] = (int(count), status, reason)
    return table


def check(root: Path, table_path: Path, scan_bin: str | None, dirs: list[str], min_sites: int) -> int:
    if min_sites < 1:
        print("fix_conversion_sites: FAIL: --min-sites must be 1 or more")
        return 1
    rows, summary = scan(root, scan_bin, dirs)
    parts = summary.split()
    files = int(parts[1]) if len(parts) == 3 and parts[0] == "scanned" else 0
    if files == 0:
        print(f"fix_conversion_sites: FAIL: the scan read no file under {dirs} (0 scanned)")
        return 1
    if len(rows) < min_sites:
        print(f"fix_conversion_sites: FAIL: {files} files scanned and {len(rows)} sites found, "
              f"fewer than --min-sites {min_sites} (the scan is broken or the tree moved)")
        return 1
    found: dict[Key, int] = {}
    for key in rows:
        found[key] = found.get(key, 0) + 1
    table = load_table(table_path)
    problems = [("not reviewed: " + "\t".join(k)) for k in sorted(found) if k not in table]
    problems += [("listed but not found (remove it): " + "\t".join(k)) for k in sorted(table) if k not in found]
    problems += [(f"count {table[k][0]} in the table, {found[k]} found: " + "\t".join(k))
                 for k in sorted(found) if k in table and table[k][0] != found[k]]
    by = {s: sum(found[k] for k in found if k in table and table[k][1] == s) for s in STATUSES}
    print(f"fix_conversion_sites: {files} files scanned; {len(rows)} sites ("
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
                    help="fail when fewer sites are found (default 1)")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    table = Path(args.table) if args.table else root / "scripts" / "fix-conversion-sites.tsv"
    return check(root, table, args.scan_bin, args.src or ["src"], args.min_sites)


if __name__ == "__main__":
    sys.exit(main())
