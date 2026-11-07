#!/usr/bin/env python3
"""Every physics module declares its model, and the declaration names tests that exist and run.

A module that computes physics (rigid bodies, contact, friction, cloth, soft bodies, fluids,
wind, buoyancy, heat, sound, …) carries a `# MODEL` section at the top of its module doc
with seven required fields (and an optional eighth):

  equations      what is solved (the equations, the constraint, the law)
  units          SI units of every quantity (or the scale when non-dimensional)
  conservation   which of mass / momentum / angular momentum / energy is conserved, which is
                 dissipated, by how much and why
  range          the valid regime (Reynolds / Courant numbers, time step, resolution, 2D / 3D)
                 and the behaviour outside it (error / clamp / warning / nothing)
  approximations what is left out
  validation     which test checks what, by test name: `tests/<file>.rs::<fn>` (several `::<fn>`
                 may follow one file; a bare file names all of its tests) or a bare `fn` name
                 in backticks that exists in tests/ or src/
  tiers          the precision tiers of the same model: `L0` is the reference; every lighter
                 tier (`L1`, `L2`, …, the 2D tier included) names the test that measures its
                 error against L0 in the same field (`L0 only` when there is no lighter tier)
  known defects  (optional, eighth) the oracles kept red on purpose: each named test exists, is
                 `#[ignore = "known defect: …"]` and is listed in docs/oracle-status.md; a red
                 oracle never counts as validation (an ignored test named there fails)

A field is a bullet or a paragraph whose bold label starts with the field name in English or
Japanese (`**Equations.**`, `- **方程式 (equations)**:`). The section ends at the next `#`
heading of the module doc.

The module list is the code, not a search by words: every `src/**/*.rs` is either in
scripts/model-modules.txt (`<group>\\t<path>`, groups A fluid, B wind and drag, C rigid and
contact, D cloth, soft body and solid, E heat, electromagnetism, sound and the rest) or in
scripts/model-exempt.txt (`<path>\\t<reason>`: no physics model in it, shapes, queries, bindings,
infrastructure), never both, and a path listed must exist. A listed module without a MODEL
section must be in scripts/model-section-baseline.txt, the ratchet seeded from the state before
this check: a module missing from the baseline fails, a baseline line whose module now has the
section fails (shrink the file), and `--write-baseline` regenerates it for an intended change.

Every test a section names must exist, must not be `#[ignore]`d, and, for a file under
tests/ gated by `#![cfg(feature = …)]`, must be reached by a `cargo test` command of
.github/workflows/ci.yml whose features (default features included, unless
`--no-default-features`) cover the gate and whose `--test` filter (if any) selects the file.
The check fails when it classified no module, read no MODEL section, or resolved no test:
a check that compares nothing passes nothing.

Usage: python3 scripts/model_section_check.py [--root DIR] [--write-baseline]
Prints `MODEL-SECTION-CHECK: {...}` with the counts, and one `model-section-check: ...` line
per finding; exit 1 on any finding, 2 when a list file cannot be read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES_FILE = "scripts/model-modules.txt"
EXEMPT_FILE = "scripts/model-exempt.txt"
BASELINE_FILE = "scripts/model-section-baseline.txt"
CI_FILE = ".github/workflows/ci.yml"
GROUPS = {"A", "B", "C", "D", "E"}

# canonical field -> labels accepted at the start of the bold label (lower-cased, prefix match)
FIELDS: dict[str, tuple[str, ...]] = {
    "equations": ("equation", "方程式"),
    "units": ("unit", "単位"),
    "conservation": ("conservation", "conserv", "保存則", "保存"),
    "range": ("range", "valid range", "validity", "適用範囲", "範囲"),
    "approximations": ("approximation", "approx", "近似"),
    "validation": ("validation", "verification", "検証"),
    "tiers": ("tier", "lod", "段", "精度段"),
    "known defects": ("known defect", "既知の欠陥", "欠陥"),
}
REQUIRED = tuple(f for f in FIELDS if f != "known defects")
ORACLE_STATUS = "docs/oracle-status.md"
KNOWN_DEFECT_REASON = "known defect:"

HEADING_RE = re.compile(r"^#\s*MODEL(\s*$|\s*[—–:\-])")
ANY_HEADING_RE = re.compile(r"^#\s+")
LABEL_RE = re.compile(r"^(?:-\s+)?\*\*([^*]+)\*\*")
TEST_FILE_RE = re.compile(r"tests/([A-Za-z0-9_][A-Za-z0-9_/]*)\.rs")
FN_AFTER_RE = re.compile(r"::\s*([A-Za-z_][A-Za-z0-9_]*)")
BARE_FN_RE = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`")
TIER_RE = re.compile(r"\bL(\d+)\b")
CFG_FEATURE_RE = re.compile(r'feature\s*=\s*"([^"]+)"')
CARGO_TEST_RE = re.compile(r"\bcargo\s+test\b([^\n]*)")


@dataclass
class Report:
    findings: list[str] = field(default_factory=list)
    modules: int = 0
    with_model: int = 0
    baseline: int = 0
    exempt: int = 0
    tests_resolved: int = 0

    def add(self, what: str) -> None:
        self.findings.append(what)


def read_rows(path: str, columns: int, what: str) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != columns:
                raise ValueError(f"{what} line {n}: expected {columns} tab-separated columns: {line!r}")
            rows.append(tuple(p.strip() for p in parts))
    return rows


def src_files(root: str) -> list[str]:
    out: list[str] = []
    base = os.path.join(root, "src")
    for dirpath, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".rs"):
                out.append(os.path.relpath(os.path.join(dirpath, f), root).replace(os.sep, "/"))
    return sorted(out)


def module_doc(text: str) -> list[str]:
    """the inner doc comment lines (`//!` stripped) at the top of a file, before the first code line"""
    doc: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("//!"):
            doc.append(s[3:].lstrip(" ").rstrip())
        elif s == "" or s.startswith("//") or s.startswith("#!["):
            continue
        else:
            break
    return doc


def model_section(doc: list[str]) -> list[str] | None:
    start = None
    for i, line in enumerate(doc):
        if HEADING_RE.match(line):
            start = i + 1
            break
    if start is None:
        return None
    body: list[str] = []
    for line in doc[start:]:
        if ANY_HEADING_RE.match(line):
            break
        body.append(line)
    return body


def canonical_field(label: str) -> str | None:
    low = label.strip().lower().lstrip("*").strip()
    for name, prefixes in FIELDS.items():
        for p in prefixes:
            if low.startswith(p):
                return name
    return None


def split_fields(body: list[str]) -> tuple[dict[str, str], list[str]]:
    """field -> joined text; plus the labels that matched no field"""
    fields: dict[str, list[str]] = {}
    unknown: list[str] = []
    current: str | None = None
    for line in body:
        m = LABEL_RE.match(line)
        if m:
            name = canonical_field(m.group(1))
            if name is None:
                unknown.append(m.group(1).strip())
                current = None
                continue
            current = name
            fields.setdefault(name, []).append(line[m.end():])
            continue
        if current is not None:
            fields[current].append(line)
    return {k: " ".join(x.strip() for x in v).strip() for k, v in fields.items()}, unknown


def test_fn_lines(text: str) -> dict[str, str | None]:
    """fn name -> None when not ignored, else the `#[ignore]` reason ("" for a bare `#[ignore]`);
    the attribute lines just above the fn are read"""
    out: dict[str, str | None] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*[<(]", line)
        if not m:
            continue
        ignored: str | None = None
        j = i - 1
        while j >= 0:
            s = lines[j].strip()
            if s.startswith("#[") or s.startswith("#!["):
                ignored = ignore_reason(s, ignored)
                j -= 1
            elif s == "" or s.startswith("//"):
                j -= 1
            elif s.endswith(")]") or s.endswith("]"):
                # the tail of a multi-line attribute: scan back to its head
                k = j
                while k >= 0 and not lines[k].strip().startswith("#["):
                    k -= 1
                if k >= 0:
                    ignored = ignore_reason(" ".join(x.strip() for x in lines[k:j + 1]), ignored)
                j = k - 1
            else:
                break
        prev = out.get(m.group(1))
        out[m.group(1)] = ignored if ignored is not None else prev
    return out


def ignore_reason(attr: str, current: str | None) -> str | None:
    """the reason of an `#[ignore …]` attribute text ("" when bare), or `current` when it is no ignore"""
    m = re.search(r"#\[\s*ignore\b(?:\s*=\s*\"((?:[^\"\\\\]|\\\\.)*)\")?", attr)
    if not m:
        return current
    return (m.group(1) or "").strip()


def file_features(text: str) -> set[str]:
    """the features a `#![cfg(...)]` file gate requires (features under `not(...)` are not required)"""
    req: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#![cfg("):
            if s.startswith("//!") or s.startswith("//") or s == "" or s.startswith("#!["):
                continue
            break
        inner = re.sub(r"not\s*\([^)]*\)", "", s)
        req.update(CFG_FEATURE_RE.findall(inner))
    return req


def default_features(root: str) -> set[str]:
    try:
        with open(os.path.join(root, "Cargo.toml"), encoding="utf-8") as fh:
            toml = fh.read()
    except OSError:
        return set()
    m = re.search(r"^default\s*=\s*\[([^\]]*)\]", toml, re.M)
    if not m:
        return set()
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def ci_test_commands(root: str, defaults: set[str]) -> list[tuple[set[str], set[str] | None]]:
    """(features enabled, test stems selected or None for all) per `cargo test` run of ci.yml
    that compiles integration tests in this package"""
    try:
        with open(os.path.join(root, CI_FILE), encoding="utf-8") as fh:
            ci = fh.read()
    except OSError:
        return []
    out: list[tuple[set[str], set[str] | None]] = []
    for m in CARGO_TEST_RE.finditer(ci):
        args = m.group(1)
        if "--lib" in args or "--doc" in args or "--no-run" in args or "--manifest-path" in args:
            continue
        if re.search(r"\s-p\s+(?!alice-physics-core\b)\S+", args) or re.search(r"--package[= ]\s*(?!alice-physics-core\b)\S+", args):
            continue
        feats: set[str] = set() if "--no-default-features" in args else set(defaults)
        if "--all-features" in args:
            feats.add("*")
        for fm in re.finditer(r"--features[= ]+(\"[^\"]*\"|'[^']*'|\S+)", args):
            feats.update(x for x in re.split(r"[,\s]+", fm.group(1).strip("\"'")) if x)
        stems = set(re.findall(r"--test[= ]+(\S+)", args))
        out.append((feats, stems or None))
    return out


def test_runs_in_ci(stem: str, required: set[str], commands: list[tuple[set[str], set[str] | None]]) -> bool:
    for feats, stems in commands:
        if stems is not None and stem not in stems:
            continue
        if "*" in feats or required <= feats:
            return True
    return False


class Resolver:
    def __init__(self, root: str, rep: Report) -> None:
        self.root = root
        self.rep = rep
        self.commands = ci_test_commands(root, default_features(root))
        self._fns: dict[str, dict[str, str | None]] = {}
        self._oracle_status: str | None = None
        self._all_fns: set[str] | None = None

    def fns_of(self, rel: str) -> dict[str, str | None] | None:
        if rel not in self._fns:
            p = os.path.join(self.root, rel)
            if not os.path.isfile(p):
                return None
            with open(p, encoding="utf-8") as fh:
                self._fns[rel] = test_fn_lines(fh.read())
        return self._fns[rel]

    def all_fns(self) -> set[str]:
        if self._all_fns is None:
            names: set[str] = set()
            for sub in ("tests", "src"):
                for dirpath, _d, files in os.walk(os.path.join(self.root, sub)):
                    for f in files:
                        if f.endswith(".rs"):
                            with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                                names.update(test_fn_lines(fh.read()))
            self._all_fns = names
        return self._all_fns

    def resolve(self, module: str, fieldname: str, text: str) -> int:
        """check every test reference of a field; returns how many resolved"""
        n = 0
        # a test fn is `::fn` right after its file or after white space / a backtick; `Type::item` (an
        # identifier right before the `::`) is a path in the prose, not a test
        tokens = re.compile(r"(tests/[A-Za-z0-9_][A-Za-z0-9_/]*\.rs)(?:\s*::\s*([A-Za-z_][A-Za-z0-9_]*))?"
                            r"|(?<![A-Za-z0-9_>\]])::\s*([A-Za-z_][A-Za-z0-9_]*)")
        current: str | None = None
        seen_fn_for_file: dict[str, int] = {}
        for m in tokens.finditer(text):
            fn = m.group(2) or m.group(3)
            if m.group(1):
                current = m.group(1)
                seen_fn_for_file.setdefault(current, 0)
                fns = self.fns_of(current)
                if fns is None:
                    self.rep.add(f"{module}: {fieldname} names {current}, which does not exist")
                    current = None
                    continue
                if not fns:
                    self.rep.add(f"{module}: {fieldname} names {current}, which has no fn")
                    current = None
                    continue
                self.check_ci(module, fieldname, current)
                if fn is None:
                    continue
            if current is None:
                self.rep.add(f"{module}: {fieldname} names `::{fn}` with no tests/<file>.rs before it")
                continue
            fns = self.fns_of(current) or {}
            if fn not in fns:
                self.rep.add(f"{module}: {fieldname} names {current}::{fn}, which is not a fn of that file")
                continue
            if fns[fn] is not None:
                self.rep.add(f"{module}: {fieldname} names {current}::{fn}, which is #[ignore]d (not run)")
                continue
            seen_fn_for_file[current] += 1
            n += 1
        for f, k in seen_fn_for_file.items():
            if k == 0:
                fns = self.fns_of(f) or {}
                live = [name for name, ign in fns.items() if ign is None]
                if not live:
                    self.rep.add(f"{module}: {fieldname} names {f}, whose tests are all #[ignore]d")
                else:
                    n += len(live)
        # bare fn names in backticks (not paths, not `Type::item`)
        no_paths = TEST_FILE_RE.sub(" ", text)
        no_paths = re.sub(r"`[^`]*::[^`]*`", " ", no_paths)
        for m in BARE_FN_RE.finditer(no_paths):
            name = m.group(1)
            if name in self.all_fns():
                n += 1
            else:
                self.rep.add(f"{module}: {fieldname} names `{name}`, which is not a fn in tests/ or src/")
        return n

    def oracle_status(self) -> str:
        if self._oracle_status is None:
            try:
                with open(os.path.join(self.root, ORACLE_STATUS), encoding="utf-8") as fh:
                    self._oracle_status = fh.read()
            except OSError:
                self._oracle_status = ""
        return self._oracle_status

    def resolve_known_defects(self, module: str, text: str) -> int:
        """every test the `known defects` field names exists, is `#[ignore = "known defect: …"]`
        and is listed in docs/oracle-status.md; returns how many resolved"""
        n = 0
        fieldname = "known defects"
        tokens = re.compile(r"(tests/[A-Za-z0-9_][A-Za-z0-9_/]*\.rs)(?:\s*::\s*([A-Za-z_][A-Za-z0-9_]*))?"
                            r"|(?<![A-Za-z0-9_>\]])::\s*([A-Za-z_][A-Za-z0-9_]*)")
        current: str | None = None
        named: list[tuple[str, str]] = []
        for m in tokens.finditer(text):
            fn = m.group(2) or m.group(3)
            if m.group(1):
                current = m.group(1)
                if self.fns_of(current) is None:
                    self.rep.add(f"{module}: {fieldname} names {current}, which does not exist")
                    current = None
                if fn is None:
                    continue
            if current is None:
                self.rep.add(f"{module}: {fieldname} names `::{fn}` with no tests/<file>.rs before it")
                continue
            named.append((current, fn))
        no_paths = TEST_FILE_RE.sub(" ", text)
        no_paths = re.sub(r"`[^`]*::[^`]*`", " ", no_paths)
        for m in BARE_FN_RE.finditer(no_paths):
            name = m.group(1)
            files = [rel for rel in self.test_files() if name in (self.fns_of(rel) or {})]
            if not files:
                self.rep.add(f"{module}: {fieldname} names `{name}`, which is not a fn in tests/")
                continue
            named.append((files[0], name))
        for rel, fn in named:
            fns = self.fns_of(rel) or {}
            if fn not in fns:
                self.rep.add(f"{module}: {fieldname} names {rel}::{fn}, which is not a fn of that file")
                continue
            reason = fns[fn]
            if reason is None:
                self.rep.add(f"{module}: {fieldname} names {rel}::{fn}, which is not #[ignore]d (a known defect is a red oracle)")
                continue
            if not reason.lower().startswith(KNOWN_DEFECT_REASON):
                self.rep.add(f"{module}: {fieldname} names {rel}::{fn}, whose #[ignore] reason does not start with "
                             f"\"{KNOWN_DEFECT_REASON}\" ({reason[:40]!r})")
                continue
            if f"`{fn}`" not in self.oracle_status():
                self.rep.add(f"{module}: {fieldname} names {rel}::{fn}, which is not listed in {ORACLE_STATUS}")
                continue
            n += 1
        return n

    def test_files(self) -> list[str]:
        out: list[str] = []
        for dirpath, _d, files in os.walk(os.path.join(self.root, "tests")):
            for f in sorted(files):
                if f.endswith(".rs"):
                    out.append(os.path.relpath(os.path.join(dirpath, f), self.root).replace(os.sep, "/"))
        return sorted(out)

    def check_ci(self, module: str, fieldname: str, rel: str) -> None:
        if not rel.startswith("tests/"):
            return
        with open(os.path.join(self.root, rel), encoding="utf-8") as fh:
            required = file_features(fh.read())
        stem = rel[len("tests/"):-len(".rs")].replace("/", "_")
        if not self.commands:
            self.rep.add(f"{module}: {fieldname} names {rel}, but {CI_FILE} has no cargo test run that builds tests/")
            return
        if not test_runs_in_ci(stem, required, self.commands):
            self.rep.add(f"{module}: {fieldname} names {rel}, which needs features {sorted(required)} "
                         f"that no cargo test run of {CI_FILE} enables for it")


def check_module(root: str, rel: str, resolver: Resolver, rep: Report) -> bool:
    """True when the module has a MODEL section (findings are recorded either way)"""
    with open(os.path.join(root, rel), encoding="utf-8") as fh:
        text = fh.read()
    body = model_section(module_doc(text))
    if body is None:
        return False
    fields, unknown = split_fields(body)
    for u in unknown:
        rep.add(f"{rel}: MODEL section has a label that is no field: **{u}**")
    for name in REQUIRED:
        if name not in fields:
            rep.add(f"{rel}: MODEL section lacks the field `{name}`")
        elif not fields[name].strip():
            rep.add(f"{rel}: MODEL section field `{name}` is empty")
    if "validation" in fields:
        n = resolver.resolve(rel, "validation", fields["validation"])
        if n == 0:
            rep.add(f"{rel}: MODEL section field `validation` names no test")
        rep.tests_resolved += n
    if "known defects" in fields:
        n = resolver.resolve_known_defects(rel, fields["known defects"])
        if n == 0:
            rep.add(f"{rel}: MODEL section field `known defects` names no ignored oracle (drop the field when there is none)")
        rep.tests_resolved += n
    if "tiers" in fields:
        tiers = {int(t) for t in TIER_RE.findall(fields["tiers"])}
        if not tiers:
            rep.add(f"{rel}: MODEL section field `tiers` names no tier (`L0 only` at least)")
        elif 0 not in tiers:
            rep.add(f"{rel}: MODEL section field `tiers` has no `L0` (the reference tier)")
        elif tiers - {0}:
            n = resolver.resolve(rel, "tiers", fields["tiers"])
            if n == 0:
                rep.add(f"{rel}: MODEL section field `tiers` lists a lighter tier than L0 "
                        f"({', '.join('L%d' % t for t in sorted(tiers - {0}))}) but names no error test against L0")
            rep.tests_resolved += n
    return True


def run(root: str, write_baseline: bool) -> Report:
    rep = Report()
    try:
        module_rows = read_rows(os.path.join(root, MODULES_FILE), 2, MODULES_FILE)
        exempt_rows = read_rows(os.path.join(root, EXEMPT_FILE), 2, EXEMPT_FILE)
        baseline_rows = read_rows(os.path.join(root, BASELINE_FILE), 1, BASELINE_FILE)
    except (OSError, ValueError) as e:
        raise SystemExit(f"model-section-check: {e}")
    modules: dict[str, str] = {}
    for group, path in module_rows:
        if group not in GROUPS:
            rep.add(f"{MODULES_FILE}: {path} has group {group!r}, not one of {''.join(sorted(GROUPS))}")
        if path in modules:
            rep.add(f"{MODULES_FILE}: {path} is listed twice")
        modules[path] = group
    exempt: dict[str, str] = {}
    for path, reason in exempt_rows:
        if path in exempt:
            rep.add(f"{EXEMPT_FILE}: {path} is listed twice")
        if len(reason) < 8:
            rep.add(f"{EXEMPT_FILE}: {path} has no reason")
        exempt[path] = reason
    baseline: set[str] = set()
    for (path,) in baseline_rows:
        if path in baseline:
            rep.add(f"{BASELINE_FILE}: {path} is listed twice")
        baseline.add(path)

    files = src_files(root)
    present = set(files)
    for path in sorted(set(modules) & set(exempt)):
        rep.add(f"{path} is both a model module and exempt")
    for path in sorted(set(modules) - present):
        rep.add(f"{MODULES_FILE}: {path} does not exist")
    for path in sorted(set(exempt) - present):
        rep.add(f"{EXEMPT_FILE}: {path} does not exist")
    for path in files:
        if path not in modules and path not in exempt:
            rep.add(f"{path} is neither in {MODULES_FILE} nor in {EXEMPT_FILE}: classify it")
    for path in sorted(baseline - set(modules)):
        rep.add(f"{BASELINE_FILE}: {path} is not a model module")

    resolver = Resolver(root, rep)
    missing: list[str] = []
    for path in sorted(modules):
        if path not in present:
            continue
        rep.modules += 1
        if check_module(root, path, resolver, rep):
            rep.with_model += 1
            if path in baseline and not write_baseline:
                rep.add(f"{BASELINE_FILE}: {path} now has a MODEL section, remove the line")
        else:
            missing.append(path)
            if path not in baseline and not write_baseline:
                rep.add(f"{path}: no MODEL section (a {modules[path]}-group physics module) and not in {BASELINE_FILE}")
    rep.exempt = len(set(exempt) & present)
    rep.baseline = len(baseline)

    if write_baseline:
        with open(os.path.join(root, BASELINE_FILE), "w", encoding="utf-8") as fh:
            fh.write("# model-section-check baseline: the physics modules that still have no MODEL section\n"
                     "# (ratchet: a module not listed fails, a listed module that gained the section fails until\n"
                     "# its line is removed). Shrink this file; --write-baseline only for an intended change.\n")
            for path in missing:
                fh.write(path + "\n")
        rep.baseline = len(missing)

    if rep.modules == 0:
        rep.add("no model module was classified (compared 0)")
    elif rep.with_model == 0:
        rep.add("no MODEL section was read (compared 0 sections)")
    elif rep.tests_resolved == 0:
        rep.add("no test named by a MODEL section was resolved (compared 0 tests)")
    return rep


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args(argv)
    rep = run(args.root, args.write_baseline)
    print("MODEL-SECTION-CHECK: " + json.dumps({"modules": rep.modules, "with_model": rep.with_model,
                                                "baseline": rep.baseline, "exempt": rep.exempt,
                                                "tests_resolved": rep.tests_resolved}))
    for f in rep.findings:
        print(f"model-section-check: {f}")
    return 1 if rep.findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
