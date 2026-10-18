#!/usr/bin/env python3
"""ALICE-Law G2: every float comparison in library code is typed by the compiler and reviewed.

`scripts/g2-rewrite-scan --comparisons` lists every ordering comparison of the non-test code
under `src/` (a binary `<` `>` `<=` `>=`, the methods `lt` `gt` `le` `ge` `partial_cmp`
`total_cmp`, a path call such as `PartialOrd::lt(&a, &b)`, and the same inside a macro
invocation's arguments), whatever its context. This script asks the compiler for the type of
each one: in a scratch copy of the tree it wraps one operand of every comparison in a call
`crate::__cmp_probe::s<N>(..)` to a generic function bounded by a trait that has no impls,
runs `cargo check --lib` in each configuration of CONFIGS, and reads the type from each E0277
error (`the trait bound `<type>: Probe` is not satisfied`, required by `s<N>`). The probe is
type checking only: nothing is built, and the source tree is not touched.

A comparison whose operand is `f32` / `f64` (through references), a SIMD float vector, or a
type parameter (which could be a float at a use) must have a row in
`scripts/g2-comparisons.tsv`, keyed by file, enclosing function and operands (no line
numbers; a repeated key carries its count), with the type the probe found and a class:

* `select`: the result chooses between values (a max / min / clamp written as a comparison,
  an argmax): the reason says how the Law's tie rule (ALICE-Law G2) is met there
* `predicate`: the result is only a condition (a branch, a filter, an early exit): no value
  of the operands is chosen by it

A row may say `predicate` only where the rewrite scan (`scripts/g2-rewrite-scan`, the
selection shapes `expr` / `return` / `assign`) finds no selection between the operands; a
selection shape the scan finds must be `select`. A `predicate` the scan cannot disprove (a
selection written in a form it does not recognise, an argmin through a closure) is the
reviewer's reading, and the reason says what the result decides.

Every other comparison (Fix128, integers, ...) is counted per type in the report.

Fails on: a float comparison with no row; a row that matches no comparison (stale) or whose
count differs; a row whose type is not the probed type (misclassified); a comparison that no
configuration typed (unprobed: a cfg no configuration compiles, a site inside a
`macro_rules!` body), unless it has a row with the type `unprobed` and its reason; a probe run
that did not type every wrapped comparison, or typed none (the probe was skipped or broken);
no comparison found (compared nothing); a row with an unknown class or a reason under 12
characters; a `predicate` row on a comparison the rewrite scan finds as a selection.

usage: g2_comparisons.py [--root DIR] [--table TSV] [--scan-bin PATH] [--write-new]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROBE = "__cmp_probe"
# the configurations the probe type checks: every feature but wasm (the documentation set of
# scripts/id_move.toml), no default features, x86_64 (the SIMD code), and wasm32
CONFIGS = [
    ["--features", "std,simd,parallel,ffi,python,privacy-crypto,gpu-solver-bridge,neural,replay,analytics"],
    ["--no-default-features"],
    ["--target", "x86_64-unknown-linux-gnu", "--features", "std,simd,parallel"],
    ["--target", "wasm32-unknown-unknown", "--features", "wasm"],
]
CLASSES = ("select", "predicate")
FLOAT = re.compile(r"^(?:&\s*(?:mut\s+)?)*(?:f32|f64|wide::f32x\d+|wide::f64x\d+|f32x\d+|f64x\d+)$")
BOUND = re.compile(r"the trait bound `(.+): (?:[\w:]*::)?Probe` is not satisfied")
REQUIRED = re.compile(r"required by a bound in `(?:[\w:]*::)?s(\d+)`")


def run_scan(root: Path, scan_bin: str | None, *args: str) -> str:
    if scan_bin:
        cmd = [scan_bin, *args, "src"]
    else:
        cmd = ["cargo", "run", "--quiet", "--release", "--manifest-path",
               str(HERE / "g2-rewrite-scan" / "Cargo.toml"), "--", *args, "src"]
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"g2_comparisons: the scan failed: {r.stderr.strip()}")
    return r.stdout


def parse_sites(text: str) -> list[dict]:
    sites = []
    for n, line in enumerate(text.splitlines()):
        f = line.split("\t")
        sites.append({"id": n, "file": f[0], "fn": f[1], "kind": f[2], "ops": f[3],
                      "line": int(f[4]), "span": f[5] if len(f) > 5 else "",
                      "ctx": f[6] if len(f) > 6 else ""})
    return sites


def scan(root: Path, scan_bin: str | None) -> list[dict]:
    return parse_sites(run_scan(root, scan_bin, "--comparisons"))


def selection_keys(text: str) -> set[tuple[str, str, str]]:
    """(file, fn, operands) of the rewrite scan's selection shapes"""
    out = set()
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) > 3 and f[2] in ("expr", "return", "assign"):
            out.add((f[0], f[1], f[3]))
    return out


def parse_span(s: str) -> tuple[int, int, int, int] | None:
    m = re.match(r"^(\d+):(\d+)-(\d+):(\d+)$", s)
    return tuple(int(x) for x in m.groups()) if m else None


def offset(lines: list[str], line: int, col: int) -> int:
    """character offset of (1-based line, 0-based char column) in the joined text"""
    return sum(len(l) + 1 for l in lines[: line - 1]) + col


def wrap(text: str, spans: list[tuple[int, tuple[int, int, int, int]]]) -> str:
    """the text with each span wrapped in `crate::__cmp_probe::s<id>(..)`; nested spans nest"""
    lines = text.split("\n")
    events = []  # (pos, order, text): closes before opens at a position, outer opens first
    for sid, (l1, c1, l2, c2) in spans:
        a, b = offset(lines, l1, c1), offset(lines, l2, c2)
        events.append((a, 1, -(b - a), f"crate::{PROBE}::s{sid}("))
        events.append((b, 0, (b - a), ")"))
    events.sort()
    out, pos = [], 0
    for at, _, _, ins in events:
        out.append(text[pos:at])
        out.append(ins)
        pos = at
    out.append(text[pos:])
    return "".join(out)


def tracked(root: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def probe(root: Path, sites: list[dict], configs: list[list[str]] | None = None) -> tuple[dict[int, set[str]], int, list[str]]:
    """(site id -> types, wrapped count, configuration problems)"""
    by_file: dict[str, list] = defaultdict(list)
    for s in sites:
        sp = parse_span(s["span"])
        if sp:
            by_file[s["file"]].append((s["id"], sp))
    wrapped = sum(len(v) for v in by_file.values())
    types: dict[int, set[str]] = defaultdict(set)
    problems = []
    with tempfile.TemporaryDirectory(prefix="g2-cmp-probe-") as tmp:
        dst = Path(tmp)
        for rel in tracked(root):
            src = root / rel
            if not src.is_file():
                continue
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if rel in by_file:
                out.write_text(wrap(src.read_text(encoding="utf-8"), by_file[rel]), encoding="utf-8")
            else:
                shutil.copyfile(src, out)
        if (root / ".cargo").is_dir():
            shutil.copytree(root / ".cargo", dst / ".cargo", dirs_exist_ok=True)
        fns = "".join(f"#[inline(always)]\npub const fn s{s['id']}<T: Probe>(x: T) -> T {{\n    x\n}}\n"
                      for s in sites if parse_span(s["span"]))
        (dst / "src" / f"{PROBE}.rs").write_text(
            "//! the type probe of scripts/g2_comparisons.py (a scratch copy only)\n"
            "#![allow(clippy::all, dead_code, missing_docs)]\npub trait Probe {}\n" + fns, encoding="utf-8")
        lib = dst / "src" / "lib.rs"
        lib.write_text(lib.read_text(encoding="utf-8") + f"\n#[doc(hidden)]\npub mod {PROBE};\n", encoding="utf-8")
        target = os.environ.get("CARGO_TARGET_DIR") or str(root / "target" / "g2-cmp-probe")
        for cfg in (CONFIGS if configs is None else configs):
            env = dict(os.environ, CARGO_TERM_COLOR="never", CARGO_TARGET_DIR=target)
            r = subprocess.run(["cargo", "check", "--lib", "--message-format=json", *cfg], cwd=dst,
                               env=env, capture_output=True, text=True)
            got = 0
            for line in r.stdout.splitlines():
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if m.get("reason") != "compiler-message":
                    continue
                msg = m["message"]
                bm = BOUND.search(msg.get("message", ""))
                if not bm:
                    continue
                text = msg.get("rendered") or ""
                ids = REQUIRED.findall(text)
                if not ids:
                    continue
                types[int(ids[0])].add(bm.group(1))
                got += 1
            if got == 0:
                problems.append(f"configuration {' '.join(cfg)}: the probe typed no comparison "
                                f"(cargo exit {r.returncode}; {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''})")
    return types, wrapped, problems


def key(s: dict) -> tuple[str, str, str]:
    return (s["file"], s["fn"], s["ops"])


def load_table(path: Path) -> tuple[dict, list[str]]:
    rows, problems = {}, []
    if not path.is_file():
        return rows, [f"{path}: missing"]
    for n, raw in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        f = raw.split("\t")
        if len(f) != 7:
            problems.append(f"{path.name}:{n}: expected 7 columns, got {len(f)}")
            continue
        file, fn, ops, count, ty, cls, reason = f
        if cls not in CLASSES and ty != "unprobed":
            problems.append(f"{path.name}:{n}: class must be one of {CLASSES}, got {cls!r}")
        if len(reason.strip()) < 12:
            problems.append(f"{path.name}:{n}: the reason must say why (12 characters or more)")
        if (file, fn, ops) in rows:
            problems.append(f"{path.name}:{n}: duplicate row {(file, fn, ops)}")
        rows[(file, fn, ops)] = (int(count), ty, cls, reason)
    return rows, problems


def is_float(ty: str) -> bool:
    t = ty.strip()
    return bool(FLOAT.match(t)) or bool(re.fullmatch(r"(?:&\s*(?:mut\s+)?)*[A-Z]\w*", t)) and t.lstrip("&mut ").strip() not in KNOWN_TYPES


# capitalised names that are concrete non-float types, not type parameters
KNOWN_TYPES = {"Fix128", "Vec3Fix", "QuatFix", "Ordering"}


def judge(sites: list[dict], types: dict[int, set[str]], rows: dict,
          selections: set[tuple[str, str, str]]) -> tuple[list[str], Counter, list[str]]:
    """Pure core: (problems, comparisons per type, rows to add for the unreviewed)"""
    problems: list[str] = []
    if not sites:
        return ["no comparison found under src/ (compared nothing)"], Counter(), []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in sites:
        groups[key(s)].append(s)
    per_type: Counter = Counter()
    new_rows = []
    for k, ss in sorted(groups.items()):
        tys: set[str] = set()
        unprobed = False
        for s in ss:
            t = types.get(s["id"])
            if t:
                tys |= t
            else:
                unprobed = True
        ty = "unprobed" if unprobed and not tys else " | ".join(sorted(tys))
        for t in tys or {"unprobed"}:
            per_type[t] += len(ss)
        if not (unprobed or any(is_float(t) for t in tys)):
            if k in rows:
                problems.append(f"row for a non-float comparison {k} (type {ty}): remove it")
            continue
        if k not in rows:
            problems.append(f"float comparison without a row: {k} x{len(ss)} (type {ty}, line {ss[0]['line']})")
            new_rows.append("\t".join([*k, str(len(ss)), ty, "?", "?"]))
            continue
        count, rty, cls, _ = rows[k]
        if count != len(ss):
            problems.append(f"{k}: the row counts {count}, the scan found {len(ss)}")
        if rty != ty:
            problems.append(f"{k}: the row says type {rty!r}, the probe found {ty!r} (misclassified)")
        if cls == "predicate" and k in selections:
            problems.append(f"{k}: the row says predicate, the rewrite scan finds a selection between the operands")
    for k in rows:
        if k not in groups:
            problems.append(f"row matches no comparison (stale): {k}")
    return problems, per_type, new_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--table", default=None)
    ap.add_argument("--scan-bin", default=None)
    ap.add_argument("--write-new", action="store_true", help="print rows for the unreviewed float comparisons")
    a = ap.parse_args(argv)
    root = Path(a.root)
    sites = scan(root, a.scan_bin)
    if not sites:
        print("g2_comparisons: no comparison found under src/ (compared nothing)")
        return 1
    types, wrapped, problems = probe(root, sites)
    if wrapped and not any(s["id"] in types for s in sites):
        problems.append("the probe typed none of the wrapped comparisons (skipped or broken)")
    selections = selection_keys(run_scan(root, a.scan_bin))
    rows, table_problems = load_table(Path(a.table) if a.table else HERE / "g2-comparisons.tsv")
    found, per_type, new_rows = judge(sites, types, rows, selections)
    problems += table_problems + found
    top = ", ".join(f"{n} {t}" for t, n in per_type.most_common(8))
    print(f"g2_comparisons: {len(sites)} comparisons ({wrapped} wrapped for the probe, {len(types)} typed) "
          f"in {len({s['file'] for s in sites})} files; {len(rows)} reviewed rows "
          f"({sum(1 for r in rows.values() if r[2] == 'select')} select); by type: {top}; {len(problems)} problems")
    for p in problems:
        print("  " + p)
    if a.write_new:
        print("\n".join(new_rows))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
