#!/usr/bin/env python3
"""Plan an in-diff mutation run so that the changed code is compiled and tested.

`cargo mutants --in-diff` mutates the changed lines, but a module behind
`#[cfg(feature = "...")]` in src/lib.rs is not compiled unless that feature is
on: its mutants change nothing and every one is reported missed. And `-- --lib`
runs only the unit tests, so the integration oracles in tests/ never see the
mutants. This script reads the diff and prints, as JSON:

  * `features`: the features to add so every changed module compiles (taken
    from the `cfg` attributes on the `mod` declarations on the way from
    src/lib.rs to the file, from the `cfg` attributes of every node (item,
    impl / trait item, field, variant, match arm, statement) whose span holds a
    changed line, plus the `required-features` of the tests below);
  * `tests`: the integration test targets whose file name contains the name of
    a changed module (`tests/audit_analytics_bridge.rs` for `src/analytics_bridge.rs`), to run
    besides `--lib`;
  * `skipped`: changed files whose module cannot be compiled on the host at
    all (it requires `not(feature = "std")`, or a wasm / python build); they
    are removed from the diff written to `--out`, and listed so the run says so;
  * `not_exercised`: changed lines inside an item (fn, impl, statement, ...) of
    a buildable file whose own cfg cannot hold on the host (`#[cfg(feature =
    "wasm")] fn ...`). The item is not compiled in the host test run, so a
    mutant of it changes nothing and would read missed; the lines are turned
    into context lines in the diff written to `--out` (no mutant is made there)
    and listed by file, line count and cfg, so the run reports them as not
    exercised instead of missed.

The spans and attributes come from a Rust parser (`syn`, scripts/cfg-items), not
from a line scanner: an item header that spans several lines (a long signature,
a `where` clause, a brace on its own line, a multi-line `#[cfg(all(...))]`) keeps
its attribute. The helper is built with cargo on first use; a file that does not
parse fails the plan.

  python3 scripts/mutants_in_diff_plan.py change.diff --features "parallel" --out planned.diff

`--features host-union` takes every feature of Cargo.toml that the host test run can
build (all but `default`, `std` and the host-unbuildable ones), so one job compiles
everything the host can; `--features host-union-no-simd` is the same set without
`simd` and keeps only the changed files whose code differs with `simd` (some cfg-items
span names the feature): the scalar fallbacks the first job does not compile. The
feature sets are read from Cargo.toml, not listed by hand: a new feature joins the
union, and a file that gains a `simd`-gated node joins the second job. A crate in
which no src file names `simd` in a cfg fails the second plan (the name is stale).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# features that cannot be built for the host test run
HOST_UNBUILDABLE = {"wasm", "python"}
FILE_RE = re.compile(r"^\+\+\+ b/(\S+)$", re.M)
MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z0-9_]+)\s*;")
FEATURE_RE = re.compile(r'feature\s*=\s*"([^"]+)"')


CFG_ITEMS = Path(__file__).resolve().parent / "cfg-items"


@dataclass(frozen=True)
class Span:
    """A node with a `#[cfg(...)]` attribute: lines `start..=end` (1-based)."""
    start: int
    end: int
    kind: str  # "file" (inner attribute), "mod", or "item"
    name: str  # the module name for "mod"
    cfg: str


_helper: Path | None = None


def helper() -> Path:
    """Build scripts/cfg-items once and return its binary."""
    global _helper
    if _helper is None:
        base = os.environ.get("CARGO_TARGET_DIR")
        target = Path(base) / "cfg-items" if base else CFG_ITEMS / "target"
        subprocess.run(
            ["cargo", "build", "--release", "--quiet", "--locked",
             "--manifest-path", str(CFG_ITEMS / "Cargo.toml"), "--target-dir", str(target)],
            check=True,
        )
        exe = "cfg-items.exe" if os.name == "nt" else "cfg-items"
        _helper = target / "release" / exe
    return _helper


def cfg_spans(path: Path) -> list[Span]:
    """The cfg-gated nodes of one Rust file, from the parser. A file that does not
    parse raises (the plan must not guess which features a line needs)."""
    r = subprocess.run([str(helper()), str(path)], capture_output=True, text=True,
                       encoding="utf-8", check=False)
    if r.returncode != 0:
        raise RuntimeError(f"cfg-items failed on {path}: {r.stderr.strip()}")
    spans = []
    for line in r.stdout.splitlines():
        _, start, end, kind, name, cfg = line.split("\t", 5)
        spans.append(Span(int(start), int(end), kind, name, cfg))
    return spans


def cfg_spans_many(paths: list[Path]) -> dict[str, list[Span]]:
    """`{path: cfg-gated nodes}` for many files in one run of the parser."""
    out: dict[str, list[Span]] = {str(p): [] for p in paths}
    if not paths:
        return out
    r = subprocess.run([str(helper()), *map(str, paths)], capture_output=True, text=True,
                       encoding="utf-8", check=False)
    if r.returncode != 0:
        raise RuntimeError(f"cfg-items failed: {r.stderr.strip()}")
    for line in r.stdout.splitlines():
        path, start, end, kind, name, cfg = line.split("\t", 5)
        out.setdefault(path, []).append(Span(int(start), int(end), kind, name, cfg))
    return out


def names_feature(cfg: str, feature: str) -> bool:
    """Whether a cfg expression has a `feature = "<feature>"` term anywhere in it."""
    def walk(node) -> bool:
        if node[0] == "pred":
            return node[1] == "feature" and node[2] == feature
        return any(walk(k) for k in node[1])
    return walk(parse_cfg(cfg))


def host_union(cargo_toml: str) -> list[str]:
    """Every feature of the `[features]` table the host test run can build: all but
    `default`, `std` and the host-unbuildable ones."""
    m = re.search(r"^\[features\]\s*$(.*?)(?=^\[)", cargo_toml + "\n[", re.M | re.S)
    if not m:
        return []
    names = re.findall(r"^([A-Za-z0-9_-]+)\s*=", m.group(1), re.M)
    return sorted(n for n in names if n not in ("default", "std") and n not in HOST_UNBUILDABLE)


def feature_sensitive_files(root: Path, feature: str) -> list[str]:
    """The src files with a cfg-gated node naming `feature` (relative, `/`-separated).
    Raises when there is none: a feature no cfg names is a stale name."""
    files = sorted((root / "src").rglob("*.rs"))
    spans = cfg_spans_many(files)
    hits = sorted({Path(p).relative_to(root).as_posix() for p, ss in spans.items()
                   if any(names_feature(s.cfg, feature) for s in ss)})
    if not hits:
        raise RuntimeError(f"no src file has a cfg naming feature {feature!r} ({len(files)} files read)")
    return hits


def module_gates(spans: list[Span]) -> dict[str, str]:
    """`{module: cfg expression}` for every `mod` declaration with a cfg on it."""
    gates: dict[str, list[str]] = {}
    for s in spans:
        if s.kind == "mod":
            gates.setdefault(s.name, []).append(s.cfg)
    return {m: " , ".join(c) for m, c in gates.items()}


def module_path_gates(path: str, root: Path) -> list[str]:
    """The cfgs on the `mod` declarations from src/lib.rs down to a src/ file:
    `src/a/b.rs` is gated by `mod a` in lib.rs and by `mod b` in src/a.rs or
    src/a/mod.rs."""
    parts = Path(path).parts
    if parts[:1] != ("src",) or len(parts) < 2:
        return []
    names = list(parts[1:-1]) + [Path(parts[-1]).stem]
    if names[-1] == "mod":
        names.pop()
    out: list[str] = []
    parent = root / "src" / "lib.rs"
    for k, name in enumerate(names):
        if name in ("lib", "main") or not parent.is_file():
            break
        out.extend(s.cfg for s in cfg_spans(parent) if s.kind == "mod" and s.name == name)
        here = root / "src" / Path(*names[: k + 1])
        parent = here / "mod.rs" if (here / "mod.rs").is_file() else here.with_suffix(".rs")
    return out


CFG_TOKEN_RE = re.compile(r'\s*(?:([A-Za-z_][A-Za-z0-9_]*)|("(?:\\.|[^"\\])*")|([=(),]))')
# what the host test run (Linux, `cargo test`, std on) makes true; any other
# configuration predicate is taken as true so the code is still compiled
HOST_FALSE = {("target_arch", "wasm32"), ("target_family", "wasm"), ("target_os", "windows"),
              ("target_os", "macos"), ("windows", None), ("miri", None)}


def parse_cfg(cfg: str):
    """A cfg expression as a tree: ("all"|"any"|"not", [children]) or ("pred", name, value)."""
    toks = []
    pos = 0
    while pos < len(cfg):
        m = CFG_TOKEN_RE.match(cfg, pos)
        if not m or m.end() == pos:
            if cfg[pos:].strip():
                raise ValueError(f"cannot read cfg {cfg!r} at {pos}")
            break
        toks.append(m.group(1) or m.group(2) or m.group(3))
        pos = m.end()
    k = 0

    def expr():
        nonlocal k
        name = toks[k]
        k += 1
        if k < len(toks) and toks[k] == "(":
            k += 1
            kids = []
            while toks[k] != ")":
                kids.append(expr())
                if toks[k] == ",":
                    k += 1
            k += 1
            return (name, kids)
        if k < len(toks) and toks[k] == "=":
            value = toks[k + 1][1:-1]
            k += 2
            return ("pred", name, value)
        return ("pred", name, None)

    tree = expr()
    if k != len(toks):
        raise ValueError(f"trailing tokens in cfg {cfg!r}")
    return tree


_HELD_OFF: frozenset[str] = frozenset()


def _solve(node, positive: bool):
    """(features that must be on, features that must be off) for `node` to be
    `positive` on the host, or None when it cannot be."""
    if node[0] == "pred":
        _, name, value = node
        if name == "feature":
            if positive:
                return None if value in HOST_UNBUILDABLE or value in _HELD_OFF else ({value}, set())
            return None if value == "std" else (set(), {value})
        truth = (name, value) not in HOST_FALSE
        return (set(), set()) if truth == positive else None
    op, kids = node
    if op == "not":
        return _solve(kids[0], not positive)
    # all(...) true / any(...) false need every child; the other two need one
    if (op == "all") == positive:
        on, off = set(), set()
        for kid in kids:
            r = _solve(kid, positive)
            if r is None:
                return None
            on |= r[0]
            off |= r[1]
        return None if on & off else (on, off)
    for kid in kids:
        r = _solve(kid, positive)
        if r is not None:
            return r
    return None


def features_for(cfg: str, held_off: frozenset[str] = frozenset()) -> tuple[list[str], bool]:
    """(features to turn on, buildable on the host) for a cfg expression, or for
    several joined with " , " (all of them must hold).

    The expression is evaluated, not matched as text: `feature = ".."` terms are
    chosen as needed (`any(..)` takes its first satisfiable alternative), `test`
    is true, a wasm / Windows target is false, `not(..)` follows De Morgan. A
    cfg that needs a host-unbuildable feature or `not(feature = "std")` makes
    the code unbuildable for the host test run."""
    if not cfg.strip():
        return [], True
    global _HELD_OFF
    saved, _HELD_OFF = _HELD_OFF, frozenset(held_off)
    try:
        r = _solve(parse_cfg(f"all({cfg})"), True)
    finally:
        _HELD_OFF = saved
    if r is None:
        return [], False
    return sorted(n for n in r[0] if n != "std"), True


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


# a group of changed files shares one cargo-mutants invocation (one baseline build and test run);
# more groups than this are merged, the pair that adds the fewest test targets first
MAX_GROUPS = 4


def reach_name(path: str) -> str | None:
    """The module name a test target names to reach a src/ file: the file stem, or the directory
    of a `mod.rs` (`src/solver/world_snapshot.rs` -> `world_snapshot`)."""
    p = Path(path)
    if p.parts[:1] != ("src",) or p.suffix != ".rs":
        return None
    name = p.parent.name if p.stem == "mod" else p.stem
    return None if name in ("lib", "main", "src") else name


def reaches(name: str, test_source: str) -> bool:
    """A test target reaches a module when its source names the module path: `<name>::` or
    `use …<name>` (2026-11-16: on the 11-14 nightly artifact this map kept all 59 mutants an
    integration target caught, and builds 2,431 targets a night against 20,140)."""
    n = re.escape(name)
    return re.search(rf"(::|\buse\s+[\w:]*){n}\b|\b{n}::", test_source) is not None


def group_files(subsets: dict[str, frozenset[str]], max_groups: int = MAX_GROUPS) -> list[dict]:
    """Files with the same reaching targets share a group; a group whose targets another group
    already contains is merged into it (no extra target, one baseline less), then groups are
    merged pairwise, the pair adding the fewest targets first, until at most `max_groups` remain."""
    by: dict[frozenset[str], list[str]] = {}
    for f, sub in sorted(subsets.items()):
        by.setdefault(sub, []).append(f)
    groups = [(set(k), sorted(v)) for k, v in by.items()]

    def cost(a, b):
        u = a[0] | b[0]
        return (len(u) - max(len(a[0]), len(b[0])), len(u))

    while len(groups) > 1:
        pairs = [(cost(groups[i], groups[j]), i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
        (added, _), i, j = min(pairs)
        if added > 0 and len(groups) <= max_groups:
            break
        a, b = groups[i], groups[j]
        groups[i] = (a[0] | b[0], sorted(a[1] + b[1]))
        del groups[j]
    return [{"files": f, "tests": sorted(t)} for t, f in sorted(groups, key=lambda g: g[1])]


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


HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", re.M)


def changed_lines(file_diff: str) -> list[int]:
    """1-based line numbers, in the new file, of the lines a file's diff adds."""
    out: list[int] = []
    line = 0
    for raw in file_diff.splitlines():
        m = HUNK_RE.match(raw)
        if m:
            line = int(m.group(1))
            continue
        if line == 0 or raw.startswith(("+++", "---")):
            continue
        if raw.startswith("+"):
            out.append(line)
            line += 1
        elif raw.startswith(" "):
            line += 1
    return out


def item_gates(spans: list[Span], lines: list[int]) -> list[str]:
    """The `cfg` expressions of the nodes whose span holds one of `lines` (1-based).

    A module that is not gated in lib.rs can still gate one function, impl,
    field, match arm or statement (`#[cfg(feature = "gpu-solver-bridge")] pub fn ...`);
    without that feature the node is not compiled and its mutants all read missed."""
    found: list[str] = []
    for s in spans:
        if any(s.start <= n <= s.end for n in lines):
            found.append(s.cfg)
    return found


def unbuildable_lines(spans: list[Span], lines: list[int],
                      held_off: frozenset[str] = frozenset()) -> dict[str, list[int]]:
    """`{cfg: [changed lines]}` for the changed lines inside a node whose cfg cannot
    hold on the host test run (every such node counts, nested or not, and of any
    kind: an item, a `mod` declaration, or a file's own `#![cfg(...)]`)."""
    out: dict[str, list[int]] = {}
    for s in spans:
        if not features_for(s.cfg, held_off)[1]:
            hit = [n for n in lines if s.start <= n <= s.end]
            if hit:
                out.setdefault(s.cfg, []).extend(hit)
    return out


def neutralise(file_diff: str, lines: set[int]) -> str:
    """The file's diff with the added lines at `lines` (new-file numbers) turned into
    context lines, each hunk's old-side count raised to match: those lines are no
    longer changed lines, so cargo-mutants makes no mutant there."""
    if not lines:
        return file_diff
    out: list[str] = []
    hunk_at = -1
    old_start = old_len = new_start = new_len = 0
    added_ctx = 0
    line = 0
    for raw in file_diff.splitlines(keepends=True):
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$", raw.rstrip("\n"))
        if m:
            if hunk_at >= 0 and added_ctx:
                out[hunk_at] = f"@@ -{old_start},{old_len + added_ctx} +{new_start},{new_len} @@{tail}\n"
            old_start, new_start = int(m.group(1)), int(m.group(3))
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            new_len = int(m.group(4)) if m.group(4) is not None else 1
            tail = m.group(5)
            hunk_at, added_ctx, line = len(out), 0, new_start
            out.append(raw)
            continue
        if hunk_at < 0 or raw.startswith(("+++", "---")):
            out.append(raw)
            continue
        if raw.startswith("+"):
            if line in lines:
                raw = " " + raw[1:]
                added_ctx += 1
            line += 1
        elif raw.startswith(" "):
            line += 1
        out.append(raw)
    if hunk_at >= 0 and added_ctx:
        out[hunk_at] = f"@@ -{old_start},{old_len + added_ctx} +{new_start},{new_len} @@{tail}\n"
    return "".join(out)


def plan(diff: str, base_features: list[str], root: Path = ROOT, only_sensitive: str | None = None) -> dict:
    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
    req = required_features(cargo)
    tests_dir = root / "tests"
    test_sources = ({p.stem: p.read_text(encoding="utf-8", errors="replace") for p in sorted(tests_dir.glob("*.rs"))}
                    if tests_dir.is_dir() else {})
    subsets: dict[str, frozenset[str]] = {}
    features = set(base_features)
    tests: set[str] = set()
    kept, skipped = [], []
    not_exercised: list[dict] = []
    outside: list[str] = []
    other_job: list[dict] = []
    held = frozenset({only_sensitive}) if only_sensitive else frozenset()
    sensitive = set(feature_sensitive_files(root, only_sensitive)) if only_sensitive else None
    for path, text in split_diff(diff):
        path = path.replace("\\", "/")  # a diff written on Windows by another tool
        if sensitive is not None and path not in sensitive:
            # this job compiles the code that differs without the feature; the others are the first job's
            outside.append(path)
            continue
        module = module_of(path)
        if module is None:
            # lib.rs / main.rs have no module gate, but their items can carry one
            # (`#[cfg(feature = "privacy-crypto")] pub use ...`)
            source = root / path
            if path.startswith("src/") and path.endswith(".rs") and source.is_file():
                for cfg in item_gates(cfg_spans(source), changed_lines(text)):
                    item_feats, item_buildable = features_for(cfg, held)
                    if item_buildable:
                        features.update(item_feats)
            kept.append(text)
            continue
        feats, buildable = features_for(" , ".join(module_path_gates(path, root)), held)
        if not buildable:
            skipped.append(path)
            continue
        features.update(feats)
        source = root / path
        if source.is_file():
            spans = cfg_spans(source)
            lines = changed_lines(text)
            for cfg in item_gates(spans, lines):
                item_feats, item_buildable = features_for(cfg, held)
                if item_buildable:
                    features.update(item_feats)
            dropped = unbuildable_lines(spans, lines)
            for cfg, hit in sorted(dropped.items()):
                not_exercised.append({"file": path, "lines": len(set(hit)), "cfg": cfg})
            gone = {n for hit in dropped.values() for n in hit}
            if held:
                # code that needs the held-off feature is compiled by the job that has it, not here
                other = unbuildable_lines(spans, lines, held)
                elsewhere = {n for hit in other.values() for n in hit} - gone
                if elsewhere:
                    other_job.append({"file": path, "lines": len(elsewhere)})
                gone |= elsewhere
            text = neutralise(text, gone)
        kept.append(text)
        if not changed_lines(text):
            continue
        name = reach_name(path)
        sub = set()
        for t, src in test_sources.items():
            if name and reaches(name, src):
                need = req.get(t, [])
                if any(n in HOST_UNBUILDABLE or n in held for n in need):
                    continue
                features.update(n for n in need if n != "std")
                sub.add(t)
        subsets[path] = frozenset(sub)
        tests |= sub
    return {
        "features": sorted(f for f in features if f),
        "tests": sorted(tests),
        "skipped": skipped,
        "not_exercised": not_exercised,
        "outside_this_job": outside,
        "other_job_lines": other_job,
        "groups": group_files(subsets),
        "lib_tests_only": sorted(f for f, sub in subsets.items() if not sub),
        "diff": "".join(kept),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("diff")
    ap.add_argument("--features", default="", help="comma-separated features of the matrix entry")
    ap.add_argument("--out", required=True, help="where to write the diff with unbuildable files removed")
    ap.add_argument("--summary", help="append the not-exercised lines (by file and in total) to this file")
    args = ap.parse_args(argv)
    only = None
    if args.features in ("host-union", "host-union-no-simd"):
        base = host_union((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
        if args.features == "host-union-no-simd":
            base = [f for f in base if f != "simd"]
            only = "simd"
    else:
        base = [f for f in args.features.split(",") if f.strip()]
    p = plan(Path(args.diff).read_text(encoding="utf-8"), base, ROOT, only)
    Path(args.out).write_text(p.pop("diff"), encoding="utf-8")
    print(json.dumps(p))
    for f in p["skipped"]:
        print(f"note: {f} is not buildable on the host test run; not mutated", file=sys.stderr)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            for n in p["not_exercised"]:
                f.write(f"- not exercised by this job: {n['file']}: {n['lines']} changed line(s) under "
                        f"#[cfg({n['cfg']})]\n")
            total = sum(n["lines"] for n in p["not_exercised"])
            f.write(f"- not exercised (host-unbuildable cfg): {total} changed line(s)\n")
            if only:
                other = sum(n["lines"] for n in p["other_job_lines"])
                f.write(f"- left to the {only} job: {len(p['outside_this_job'])} file(s) whose code does not differ "
                        f"without {only}, {other} changed line(s) that need {only}\n")
    for n in p["not_exercised"]:
        print(f"note: {n['file']}: {n['lines']} changed line(s) under #[cfg({n['cfg']})] are not compiled "
              "on the host test run; not exercised by this job (not mutated)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
