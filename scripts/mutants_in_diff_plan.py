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
  * `tests`: the integration test targets to run besides `--lib`, and `why`, the
    reason each one is there: its file name contains the name of a changed module
    (`tests/audit_analytics_bridge.rs` for `src/analytics_bridge.rs`), the diff
    changes the test file itself (the oracle that came with the change, whatever
    its name), or the file names a changed module in a path of this crate
    (`alice_physics_core::cloth::Cloth`, or a name src/lib.rs re-exports from it:
    `alice_physics_core::Cloth`). A module named by more than USE_LIMIT
    test files adds none of them by use; `not_selected` gives the module and the
    count, so the run says what it did not run;
  * `skip`: the tests of a selected file that are left out of the mutation run.
    A file whose whole run is too long for one mutant marks the tests that are
    quick and exact with a `// mutants: fast` line above them; once a file has
    such a line, its other tests are skipped (`-- --exact --skip <name>`), and
    `fast` gives marked / all per file. A file with no such line runs whole;
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


def _solve(node, positive: bool):
    """(features that must be on, features that must be off) for `node` to be
    `positive` on the host, or None when it cannot be."""
    if node[0] == "pred":
        _, name, value = node
        if name == "feature":
            if positive:
                return None if value in HOST_UNBUILDABLE else ({value}, set())
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


def features_for(cfg: str) -> tuple[list[str], bool]:
    """(features to turn on, buildable on the host) for a cfg expression, or for
    several joined with " , " (all of them must hold).

    The expression is evaluated, not matched as text: `feature = ".."` terms are
    chosen as needed (`any(..)` takes its first satisfiable alternative), `test`
    is true, a wasm / Windows target is false, `not(..)` follows De Morgan. A
    cfg that needs a host-unbuildable feature or `not(feature = "std")` makes
    the code unbuildable for the host test run."""
    if not cfg.strip():
        return [], True
    r = _solve(parse_cfg(f"all({cfg})"), True)
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


def module_of(path: str) -> str | None:
    """The top-level module of a src/ file (`src/a.rs` and `src/a/b.rs` -> `a`)."""
    p = Path(path)
    if p.parts[:1] != ("src",) or p.suffix != ".rs" or len(p.parts) < 2:
        return None
    name = p.parts[1]
    if len(p.parts) == 2:
        name = p.stem
    return None if name in ("lib", "main") else name


# a module named by more test files than this adds none of them by use: every selected
# target is built and run for every mutant (2026-11-23, of 581 files: `math` 447, `solver` 189,
# then 40 and fewer; `cloth` 18, `heightfield` 19, three modules at 24)
USE_LIMIT = 24
FAST_MARK_RE = re.compile(r"^\s*//\s*mutants:\s*fast\s*$")
FN_RE = re.compile(r"^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")
TEST_ATTR_RE = re.compile(r"^\s*#\[(?:[A-Za-z_][A-Za-z0-9_]*::)*test\b")


def crate_names(cargo_toml: str) -> list[str]:
    """The names a test file can call this crate by: the `[lib]` name and the
    package name with `_` for `-`."""
    names = set()
    for table in ("package", "lib"):
        m = re.search(rf'^\[{table}\]\s*$(.*?)(?=^\[|\Z)', cargo_toml, re.M | re.S)
        n = re.search(r'^\s*name\s*=\s*"([^"]+)"', m.group(1), re.M) if m else None
        if n:
            names.add(n.group(1).replace("-", "_"))
    return sorted(names)


def test_target(path: str) -> str | None:
    """The integration test target of a tests/ file (`tests/a.rs` and `tests/a/main.rs` -> `a`)."""
    parts = Path(path).parts
    if len(parts) == 2 and parts[0] == "tests" and parts[1].endswith(".rs"):
        return parts[1][:-3]
    if len(parts) == 3 and parts[0] == "tests" and parts[2] == "main.rs":
        return parts[1]
    return None


REEXPORT_RE = re.compile(
    r"^pub\s+use\s+(?:crate::)?([a-z_][a-z0-9_]*)::(\{[^{}]*\}|\*|[A-Za-z_][A-Za-z0-9_]*(?:\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?)\s*;",
    re.M)


PUB_USE_RE = re.compile(r"^pub\s+use\b[^;]*;", re.M)


def reexports(lib_text: str) -> tuple[dict[str, set[str]], list[str], list[str]]:
    """(`{name at the crate root: modules it comes from}`, modules re-exported with `*`, the
    column-0 `pub use` statements this cannot read) from the
    `pub use <module>::...;` lines at column 0 of src/lib.rs. Most test files reach a type through
    the root (`krate::Cloth`), not through its module. A `*` re-export gives no names to read: its
    module is returned so the plan says that users by a root name are not found for it."""
    names: dict[str, set[str]] = {}
    globs: list[str] = []
    for m in REEXPORT_RE.finditer(lib_text):
        module, what = m.group(1), m.group(2)
        if what == "*":
            globs.append(module)
            continue
        for item in what.strip("{}").split(","):
            words = item.split()
            if words and "::" not in words[0]:
                names.setdefault(words[-1], set()).add(module)
    read = {m.group(0) for m in REEXPORT_RE.finditer(lib_text)}
    # a form the pattern does not cover (nested braces `m::{a::{B, C}}`, `self::`, a path item
    # inside braces is dropped above): its names are not in the map, so the plan says how many
    unread = [" ".join(s.split()) for s in (m.group(0) for m in PUB_USE_RE.finditer(lib_text)) if s not in read]
    for m in REEXPORT_RE.finditer(lib_text):
        if m.group(2).startswith("{") and any("::" in item for item in m.group(2).strip("{}").split(",")):
            unread.append(" ".join(m.group(0).split()))
    return names, sorted(set(globs)), unread


def modules_used(text: str, crates: list[str], exports: dict[str, set[str]] | None = None) -> set[str]:
    """The top-level modules of this crate a test file names in a path:
    `krate::cloth::Cloth`, the heads of `krate::{cloth::Cloth, math::Fix128}`, and, through the
    re-exports of the crate root (`exports`), `krate::Cloth` and `krate::{Cloth, Grab}`."""
    if not crates:
        return set()
    exports = exports or {}
    alt = "|".join(re.escape(c) for c in crates)
    heads = set(re.findall(rf"\b(?:{alt})::([A-Za-z_][A-Za-z0-9_]*)\b", text))
    used = {h for h in heads if re.fullmatch(r"[a-z_][a-z0-9_]*", h)}
    for h in heads:
        used |= exports.get(h, set())
    for m in re.finditer(rf"\b(?:{alt})::\{{", text):
        depth, k, start = 1, m.end(), m.end()
        group = []
        while k < len(text) and depth:
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    group.append(text[start:k])
            elif c == "," and depth == 1:
                group.append(text[start:k])
                start = k + 1
            k += 1
        for h in group:
            hm = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\b", h)
            if hm:
                if re.fullmatch(r"[a-z_][a-z0-9_]*", hm.group(1)):
                    used.add(hm.group(1))
                used |= exports.get(hm.group(1), set())
    return used


def test_fns(text: str) -> list[tuple[str, bool, bool]]:
    """`[(name, marked fast, nested)]` for the `#[test]` functions of a test file.
    The attributes and line comments directly above a `fn` (no blank line between)
    are its header; `nested` is a function that is not at column 0 (inside a `mod`).
    A header with an attribute that spans several lines (`#[cfg_attr(\n ...)]`) is cut at
    the continuation line: a mark or `#[test]` above it is not seen, so such a test is
    not skipped and not counted (it runs)."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        m = FN_RE.match(line)
        if not m:
            continue
        k = i - 1
        header = []
        while k >= 0 and lines[k].strip().startswith(("#[", "//")):
            header.append(lines[k])
            k -= 1
        if any(TEST_ATTR_RE.match(h) for h in header):
            out.append((m.group(2), any(FAST_MARK_RE.match(h) for h in header), bool(m.group(1))))
    return out


def fast_subset(texts: dict[str, str]) -> tuple[list[str], dict[str, list[int]]]:
    """(tests to skip, `{target: [marked, all]}`) for the selected test files
    `{target: text}`. A skip is an exact name for every test binary of the run, so
    a marked file must keep its tests at the top level, and a skipped name must not
    be a test another selected file runs."""
    skip: dict[str, str] = {}
    fast: dict[str, list[int]] = {}
    runs: dict[str, str] = {}
    for target, text in sorted(texts.items()):
        fns = test_fns(text)
        if not any(f for _, f, _ in fns):
            for name, _, nested in fns:
                if not nested:
                    runs[name] = target
            continue
        nested = sorted(n for n, _, deep in fns if deep)
        if nested:
            raise ValueError(f"tests/{target}.rs marks tests `// mutants: fast` but {nested} are inside a "
                             "module: a skip is an exact top-level name, keep the tests of a marked file at the top level")
        fast[target] = [sum(1 for _, f, _ in fns if f), len(fns)]
        for name, f, _ in fns:
            if f:
                runs[name] = target
            else:
                skip[name] = target
    clash = sorted(n for n in skip if n in runs)
    if clash:
        raise ValueError("a test skipped in one selected file is run by another under the same name (the skip "
                         "would drop both): " + ", ".join(f"{n} (skipped in {skip[n]}, run in {runs[n]})" for n in clash))
    return sorted(skip), fast


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


def unbuildable_lines(spans: list[Span], lines: list[int]) -> dict[str, list[int]]:
    """`{cfg: [changed lines]}` for the changed lines inside a node whose cfg cannot
    hold on the host test run (every such node counts, nested or not, and of any
    kind: an item, a `mod` declaration, or a file's own `#![cfg(...)]`)."""
    out: dict[str, list[int]] = {}
    for s in spans:
        if not features_for(s.cfg)[1]:
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


def plan(diff: str, base_features: list[str], root: Path = ROOT) -> dict:
    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
    req = required_features(cargo)
    tests_dir = root / "tests"
    test_names = sorted(p.stem for p in tests_dir.glob("*.rs")) if tests_dir.is_dir() else []
    features = set(base_features)
    tests: set[str] = set()
    why: dict[str, set[str]] = {}
    crates = crate_names(cargo)
    lib_rs = root / "src" / "lib.rs"
    exports, globs, unread = reexports(lib_rs.read_text(encoding="utf-8")) if lib_rs.is_file() else ({}, [], [])
    test_texts: dict[str, str] = {}

    def text_of(t: str) -> str:
        if t not in test_texts:
            f = tests_dir / f"{t}.rs"
            if not f.is_file():
                f = tests_dir / t / "main.rs"
            test_texts[t] = f.read_text(encoding="utf-8") if f.is_file() else ""
        return test_texts[t]

    def select(t: str, reason: str) -> None:
        need = req.get(t, [])
        if any(n in HOST_UNBUILDABLE for n in need):
            return
        features.update(n for n in need if n != "std")
        tests.add(t)
        why.setdefault(t, set()).add(reason)

    users: dict[str, list[str]] | None = None
    not_selected: list[dict] = []
    seen_modules: set[str] = set()
    kept, skipped = [], []
    not_exercised: list[dict] = []
    for path, text in split_diff(diff):
        path = path.replace("\\", "/")  # a diff written on Windows by another tool
        if path.startswith("tests/"):
            # a test file is never mutated; the one the diff changes is run (the oracle of the change)
            target = test_target(path)
            if target is not None and (root / path).is_file():
                select(target, "changed")
            continue
        module = module_of(path)
        if module is None:
            # lib.rs / main.rs have no module gate, but their items can carry one
            # (`#[cfg(feature = "privacy-crypto")] pub use ...`)
            source = root / path
            if path.startswith("src/") and path.endswith(".rs") and source.is_file():
                for cfg in item_gates(cfg_spans(source), changed_lines(text)):
                    item_feats, item_buildable = features_for(cfg)
                    if item_buildable:
                        features.update(item_feats)
            kept.append(text)
            continue
        feats, buildable = features_for(" , ".join(module_path_gates(path, root)))
        if not buildable:
            skipped.append(path)
            continue
        features.update(feats)
        source = root / path
        if source.is_file():
            spans = cfg_spans(source)
            lines = changed_lines(text)
            for cfg in item_gates(spans, lines):
                item_feats, item_buildable = features_for(cfg)
                if item_buildable:
                    features.update(item_feats)
            dropped = unbuildable_lines(spans, lines)
            for cfg, hit in sorted(dropped.items()):
                not_exercised.append({"file": path, "lines": len(set(hit)), "cfg": cfg})
            text = neutralise(text, {n for hit in dropped.values() for n in hit})
        kept.append(text)
        token = re.compile(rf"(^|_){re.escape(module)}(_|$)")
        for t in test_names:
            if token.search(t):
                select(t, "name")
        if module in seen_modules:
            continue
        seen_modules.add(module)
        if users is None:
            users = {}
            for t in test_names:
                for used in modules_used(text_of(t), crates, exports):
                    users.setdefault(used, []).append(t)
        mine = users.get(module, [])
        if module in globs:
            not_selected.append({"module": module, "glob": True})
        if len(mine) > USE_LIMIT:
            not_selected.append({"module": module, "files": len(mine), "limit": USE_LIMIT})
        else:
            for t in mine:
                select(t, "uses")
    if unread and seen_modules:
        not_selected.append({"unread_reexports": len(unread), "example": unread[0][:80]})
    skip, fast = fast_subset({t: text_of(t) for t in tests})
    return {
        "features": sorted(f for f in features if f),
        "tests": sorted(tests),
        "why": {t: sorted(r) for t, r in sorted(why.items())},
        "skip": skip,
        "fast": fast,
        "not_selected": not_selected,
        "skipped": skipped,
        "not_exercised": not_exercised,
        "diff": "".join(kept),
    }


def not_selected_line(n: dict) -> str:
    if "unread_reexports" in n:
        return (f"{n['unread_reexports']} `pub use` statement(s) of src/lib.rs are in a form the plan does not read "
                f"(e.g. `{n['example']}`): test files that reach those names through the crate root are not found by use")
    if n.get("glob"):
        return (f"module {n['module']} is re-exported from the crate root with `*`: test files that reach it "
                "through a root name are not found by use")
    return (f"the {n['files']} test files that name module {n['module']} (more than {n['limit']}) are not added "
            "by use; only those selected by name or changed by the diff run")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("diff")
    ap.add_argument("--features", default="", help="comma-separated features of the matrix entry")
    ap.add_argument("--out", required=True, help="where to write the diff with unbuildable files removed")
    ap.add_argument("--summary", help="append the not-exercised lines (by file and in total) to this file")
    ap.add_argument("--root", default=str(ROOT),
                    help="the tree the diff is of (default: the one this script is in); a range tested at another "
                         "head than the run's is planned by the run's script on that head's tree")
    args = ap.parse_args(argv)
    base = [f for f in args.features.split(",") if f.strip()]
    p = plan(Path(args.diff).read_text(encoding="utf-8"), base, Path(args.root).resolve())
    Path(args.out).write_text(p.pop("diff"), encoding="utf-8")
    print(json.dumps(p))
    for f in p["skipped"]:
        print(f"note: {f} is not buildable on the host test run; not mutated", file=sys.stderr)
    for n in p["not_selected"]:
        print(f"note: {not_selected_line(n)}", file=sys.stderr)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            for t in p["tests"]:
                part = p["fast"].get(t)
                f.write(f"- test target {t} ({', '.join(p['why'][t])})"
                        + (f": {part[0]} of {part[1]} tests (marked fast); a mutant missed here is missed against "
                           "this subset, run the file without skips before listing it as equivalent" if part else "")
                        + "\n")
            for n in p["not_selected"]:
                f.write(f"- not run: {not_selected_line(n)}\n")
            for n in p["not_exercised"]:
                f.write(f"- not exercised by this job: {n['file']}: {n['lines']} changed line(s) under "
                        f"#[cfg({n['cfg']})]\n")
            total = sum(n["lines"] for n in p["not_exercised"])
            f.write(f"- not exercised (host-unbuildable cfg): {total} changed line(s)\n")
    for n in p["not_exercised"]:
        print(f"note: {n['file']}: {n['lines']} changed line(s) under #[cfg({n['cfg']})] are not compiled "
              "on the host test run; not exercised by this job (not mutated)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
