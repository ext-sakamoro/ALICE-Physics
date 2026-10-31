#!/usr/bin/env python3
"""The table of binding parameters that carry floats equals the bindings' code.

Every C, WebAssembly and Python entry point refuses a NaN or an infinity explicitly
(`binding_api::checked*`), so it never reaches the debug assertion of `Fix128::from_f64`.
Three tests pass each such parameter NaN, +inf and -inf and check the refusal: the Rust test
in `src/ffi.rs` (`every_float_parameter_refuses_a_non_finite_value`), `tests/js` (the wasm
package built with `--dev`) and `python/tests` (a dev-profile wheel). Each reads
`scripts/binding-float-entries.tsv` and fails on a row it has no case for.

This check keeps that table equal to `scripts/g2-rewrite-scan --binding-floats src`, a
syntax-tree scan of the entry points (`extern` functions, `#[wasm_bindgen]`, `#[pyfunction]`,
`#[pymethods]`) and their parameters whose type names a float type: a new parameter without a
row, a row whose parameter is gone, a scan that read no file or found fewer rows than
`--min-rows`, and a bad row fail.

usage: binding_float_entries.py [--root DIR] [--table TSV] [--scan-bin PATH] [--min-rows N]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

Row = tuple[str, str, str, str]


def scan(root: Path, scan_bin: str | None) -> tuple[list[Row], str]:
    target = str(root / "src")
    if scan_bin:
        cmd = [scan_bin, "--binding-floats", target]
    else:
        cmd = ["cargo", "run", "-q", "--release", "--locked",
               "--manifest-path", str(HERE / "g2-rewrite-scan" / "Cargo.toml"), "--",
               "--binding-floats", target]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise SystemExit(f"binding_float_entries: FAIL: the scan exited {p.returncode}: {p.stderr.strip()}")
    rows = []
    prefix = str(root).replace("\\", "/").rstrip("/") + "/"
    for line in p.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) != 4:
            raise SystemExit(f"binding_float_entries: FAIL: unexpected scan line {line!r}")
        file = cols[0].replace("\\", "/")
        if file.startswith(prefix):
            file = file[len(prefix):]
        rows.append((file, cols[1], cols[2], cols[3]))
    return rows, p.stderr.strip()


def load_table(path: Path) -> list[Row]:
    rows = []
    seen = set()
    for n, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 4 or not all(c.strip() for c in cols):
            raise SystemExit(f"{path}:{n}: expected 4 non-empty tab-separated columns, got {cols!r}")
        key = tuple(cols)
        if key in seen:
            raise SystemExit(f"{path}:{n}: duplicate row {key}")
        seen.add(key)
        rows.append(key)
    return rows


def check(root: Path, table_path: Path, scan_bin: str | None, min_rows: int) -> int:
    if min_rows < 1:
        print("binding_float_entries: FAIL: --min-rows must be 1 or more")
        return 1
    found, summary = scan(root, scan_bin)
    parts = summary.split()
    files = int(parts[1]) if len(parts) == 3 and parts[0] == "scanned" else 0
    if files == 0:
        print("binding_float_entries: FAIL: the scan read no file under src (0 scanned)")
        return 1
    if len(found) < min_rows:
        print(f"binding_float_entries: FAIL: {files} files scanned and {len(found)} rows found, "
              f"fewer than --min-rows {min_rows} (the scan is broken or the tree moved)")
        return 1
    table = load_table(table_path)
    problems = [("no row: " + "\t".join(k)) for k in sorted(set(found) - set(table))]
    problems += [("row not found (remove it): " + "\t".join(k)) for k in sorted(set(table) - set(found))]
    by_file: dict[str, int] = {}
    for f, *_ in found:
        by_file[f] = by_file.get(f, 0) + 1
    print(f"binding_float_entries: {files} files scanned; {len(found)} parameters ("
          + ", ".join(f"{n} {f}" for f, n in sorted(by_file.items())) + f", {len(problems)} problems)")
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--table", default=None)
    ap.add_argument("--scan-bin", default=os.environ.get("G2_SCAN_BIN"))
    ap.add_argument("--min-rows", type=int, default=1,
                    help="fail when the scan finds fewer rows (default 1)")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    table = Path(args.table) if args.table else root / "scripts" / "binding-float-entries.tsv"
    return check(root, table, args.scan_bin, args.min_rows)


if __name__ == "__main__":
    sys.exit(main())
