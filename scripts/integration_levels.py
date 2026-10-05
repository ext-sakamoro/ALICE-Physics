#!/usr/bin/env python3
"""How each part of the crate can actually be used: world, bindings, library, example.

The name-based wiring guard counts an example as a caller, so an item that only
an example calls passes it without being integrated anywhere. This script
measures integration directly and does not count examples as evidence of it:

  module levels    every pub item is reached (SCIP, scripts/scip_reach.py) from
                   `PhysicsWorld::step*` (step), another `PhysicsWorld` pub method
                   (world API), a binding file (binding), or none of these; a module
                   takes the highest level any of its items has:
                     step         runs when the world steps
                     world API    used through a `PhysicsWorld` method
                     binding      reached only from src/ffi.rs / python.rs / wasm.rs
                     standalone   a Rust API that only examples call: usable
                                  directly, not integrated into the world or a binding
                     unused       no caller outside tests
  C ABI coverage   for every `extern "C"` function in src/ffi.rs, whether the C
                   header, the second header, the Unity C# bindings and the
                   Unreal Engine plugin declare or call it (read from source:
                   C#, C and C++ are not in the SCIP index). A function a
                   consumer lacks fails the check unless scripts/abi-consumer-gaps.txt
                   lists it for that consumer with a reason; a listed gap that
                   is closed fails too, so the list cannot go stale

The ledger is docs/integration-levels.md (generated whole; scripts/scip_reach.py
owns docs/integration-status.md). `--check` fails when the ledger or the README
/ MODULES.md figures disagree with the source, or when a table would be empty.

Usage:
  python3 scripts/integration_levels.py --write
  python3 scripts/integration_levels.py --check
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scip_reach  # noqa: E402

FFI_FILE = "src/ffi.rs"
WORLD_PREFIX = "src/solver.rs::PhysicsWorld"
LEVELS = ("step", "world API", "binding", "standalone", "unused")
BINDING_LABELS = {"src/python.rs": "Python", "src/ffi.rs": "C ABI", "src/wasm.rs": "WebAssembly"}
SUMMARY_MARK = "<!-- integration-levels: summary -->"
MODULES_DOC = "docs/MODULES.md"
FFI_FN_RE = re.compile(r'pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(alice_physics_[a-z0-9_]+)')
SYMBOL_RE = re.compile(r"\b(alice_physics_[a-z0-9_]+)\b")
# consumers of the C ABI: (key used in the gap list, label, glob patterns relative to the root)
C_ABI_CONSUMERS = [
    ("c-header", "C header", ["include/alice_physics.h"]),
    ("bindings-header", "bindings header", ["bindings/AlicePhysics.h"]),
    ("unity", "Unity C#", ["bindings/AlicePhysics.cs"]),
    ("unreal", "Unreal Engine plugin", ["unreal-plugin/Source/**/*.cpp", "unreal-plugin/Source/**/*.h"]),
]
GAPS_FILE = "scripts/abi-consumer-gaps.txt"
LEDGER = "docs/integration-levels.md"
MIN_REASON = 12


def _strip_comments(text: str) -> str:
    """Drop // and /* */ comments so a commented-out call does not count."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def ffi_functions(root: Path) -> list[str]:
    p = root / FFI_FILE
    if not p.exists():
        return []
    return sorted(set(FFI_FN_RE.findall(p.read_text(encoding="utf-8"))))


def consumer_symbols(root: Path, patterns: list[str]) -> tuple[set[str], int]:
    """Symbols referenced outside comments, and the number of files read."""
    found: set[str] = set()
    files = 0
    for pat in patterns:
        for p in sorted(glob.glob(str(root / pat), recursive=True)):
            files += 1
            text = Path(p).read_text(encoding="utf-8", errors="replace")
            found |= set(SYMBOL_RE.findall(_strip_comments(text)))
    return found, files


def read_gaps(root: Path) -> tuple[dict[tuple[str, str], str], list[str]]:
    """`<consumer key> <function> <reason>` lines (`#` starts a comment)."""
    gaps: dict[tuple[str, str], str] = {}
    errors: list[str] = []
    p = root / GAPS_FILE
    if not p.exists():
        return gaps, errors
    keys = {k for k, _, _ in C_ABI_CONSUMERS}
    for n, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3 or len(parts[2]) < MIN_REASON:
            errors.append(f"{GAPS_FILE}:{n}: expected `<consumer> <function> <reason of {MIN_REASON}+ chars>`")
            continue
        key, fn, reason = parts
        if key not in keys:
            errors.append(f"{GAPS_FILE}:{n}: unknown consumer `{key}` (one of {', '.join(sorted(keys))})")
            continue
        if (key, fn) in gaps:
            errors.append(f"{GAPS_FILE}:{n}: `{key} {fn}` listed twice")
        gaps[(key, fn)] = reason
    return gaps, errors


class CAbi:
    def __init__(self, root: Path):
        self.functions = ffi_functions(root)
        self.consumers: list[tuple[str, str, set[str], int]] = []
        for key, label, pats in C_ABI_CONSUMERS:
            syms, files = consumer_symbols(root, pats)
            self.consumers.append((key, label, syms, files))
        self.gaps, self.gap_errors = read_gaps(root)

    def problems(self) -> list[str]:
        out = list(self.gap_errors)
        if not self.functions:
            out.append(f"no extern \"C\" functions found in {FFI_FILE} (compared nothing)")
        fn = set(self.functions)
        for key, label, syms, files in self.consumers:
            if files == 0:
                out.append(f"{label}: no files found (compared nothing)")
                continue
            for s in sorted(syms - fn):
                out.append(f"{label} declares or calls `{s}`, which src/ffi.rs does not export (link error)")
            for f in self.functions:
                listed = (key, f) in self.gaps
                if f not in syms and not listed:
                    out.append(f"{label} does not declare or call `{f}`: add it, or list it in {GAPS_FILE} with a reason")
                elif f in syms and listed:
                    out.append(f"{label} now uses `{f}`: remove its line from {GAPS_FILE}")
        for key, f in sorted(self.gaps):
            if f not in fn:
                out.append(f"{GAPS_FILE}: `{key} {f}` names a function src/ffi.rs does not export")
        return out

    def markdown(self) -> list[str]:
        fn = self.functions
        lines = [
            "### C ABI (`--features ffi`)",
            "",
            f"`src/ffi.rs` exports {len(fn)} `extern \"C\"` functions. Which consumers declare or call each one:",
            "",
            "| Consumer | Functions used | Missing |",
            "|----------|---------------:|---------|",
        ]
        for _key, label, syms, _files in self.consumers:
            used = [f for f in fn if f in syms]
            missing = [f"`{f}`" for f in fn if f not in syms]
            lines.append(f"| {label} | {len(used)} / {len(fn)} | {', '.join(missing) if missing else '—'} |")
        lines.append("")
        listed = [(k, f, r) for (k, f), r in sorted(self.gaps.items())]
        if listed:
            labels = {k: label for k, label, _ in C_ABI_CONSUMERS}
            lines += [f"Why the missing functions are not wrapped (`{GAPS_FILE}`):", "",
                      "| Consumer | Function | Reason |", "|----------|----------|--------|"]
            lines += [f"| {labels[k]} | `{f}` | {r} |" for k, f, r in listed]
            lines.append("")
        return lines


def module_of(key: str) -> str:
    """`src/a.rs::X` and `src/a/b.rs::X` -> `a`."""
    rel = key.split("::", 1)[0]
    parts = rel[len("src/"):].split("/")
    return parts[0][:-3] if len(parts) == 1 else parts[0]


class Levels:
    def __init__(self, root: Path, scip_paths: list[Path]):
        a = scip_reach.analyze(root, scip_paths, keep_graph=True)
        self.analysis = a
        world = [k for k in a.items if k == WORLD_PREFIX or k.startswith(WORLD_PREFIX + "::")]
        step = [k for k in world if k.rsplit("::", 1)[-1].startswith("step")]
        syms = lambda keys: set().union(*(a.items[k] for k in keys)) if keys else set()  # noqa: E731
        self.step_entries = sorted(k.rsplit("::", 1)[-1] for k in step)
        r_step = a.reach(syms(step)) if step else set()
        r_world = a.reach(syms(world)) if world else set()
        self.by_binding = {f: a.reach(r) for f, r in sorted(a.roots_by_binding.items())}
        r_bind = set().union(*self.by_binding.values()) if self.by_binding else set()
        self.item_level: dict[str, str] = {}
        for k, ss in a.items.items():
            if ss & r_step:
                lv = "step"
            elif ss & r_world:
                lv = "world API"
            elif ss & r_bind:
                lv = "binding"
            elif a.level[k] in ("live", "L1"):
                lv = "standalone"  # live without world/binding is crate-internal plumbing of a standalone API
            else:
                lv = "unused"
            self.item_level[k] = lv
        self.module_items: dict[str, dict[str, int]] = {}
        self.module_bindings: dict[str, set[str]] = {}
        for k, lv in self.item_level.items():
            m = module_of(k)
            self.module_items.setdefault(m, {x: 0 for x in LEVELS})[lv] += 1
            for f, r in self.by_binding.items():
                if a.items[k] & r:
                    self.module_bindings.setdefault(m, set()).add(BINDING_LABELS.get(f, f))
        # public modules only (src/lib.rs `pub mod`): the documents list those. The C ABI
        # module is the binding surface itself; re-exports without items get no level
        lib = (root / "src" / "lib.rs").read_text(encoding="utf-8")
        self.public = sorted(set(re.findall(r"^\s*pub mod (\w+)", lib, re.M)))
        # A module's label is the level most of its items have (ties go to the higher
        # level). "Any item reaches step" overstates: a helper type used by the
        # solver made a whole unrelated module look integrated (measured: `privacy`
        # through one RNG type). Items above the label are reported next to it.
        self.module_level = {m: max(LEVELS, key=lambda x: (c[x], -LEVELS.index(x)))
                             for m, c in self.module_items.items() if m in self.public}
        if "ffi" in self.public:
            self.module_level["ffi"] = "binding"
            self.module_items.setdefault("ffi", {x: 0 for x in LEVELS})

    def label(self, m: str) -> str:
        """`standalone`, or `standalone (step 9, world API 1 of 25 items)` when some
        items reach a level above the label (partly integrated)."""
        lv = self.module_level[m]
        n = self.module_items[m]
        above = [f"{x} {n[x]}" for x in LEVELS[:LEVELS.index(lv)] if n[x]]
        return f"{lv} ({', '.join(above)} of {sum(n.values())} items)" if above else lv

    def counts(self) -> dict[str, int]:
        return {x: sum(1 for v in self.module_level.values() if v == x) for x in LEVELS}

    def problems(self) -> list[str]:
        out = []
        if not self.step_entries:
            out.append(f"no `{WORLD_PREFIX}::step*` item found (compared nothing)")
        if not self.by_binding:
            out.append("no binding roots (compared nothing)")
        if not self.module_level:
            out.append("no modules classified (compared nothing)")
        return out

    def markdown(self) -> list[str]:
        c = self.counts()
        lines = [
            "### Modules by how they are used",
            "",
            "A module is labelled with the level most of its public items have; items that reach a higher",
            "level are listed next to the label (partly integrated). Examples do not count: an item that",
            "only an example calls is *standalone*, usable from Rust but not wired into",
            f"`PhysicsWorld` or a binding. Step entry points: {', '.join(f'`{e}`' for e in self.step_entries)}.",
            "",
            "| Level | Meaning | Modules |",
            "|-------|---------|--------:|",
            f"| step | runs when `PhysicsWorld` steps | {c['step']} |",
            f"| world API | used through another `PhysicsWorld` method | {c['world API']} |",
            f"| binding | reached only from the C ABI, Python or WebAssembly bindings | {c['binding']} |",
            f"| standalone | a Rust API that only examples call | {c['standalone']} |",
            f"| unused | no caller outside tests | {c['unused']} |",
            "",
            "| Module | Level | Items: step / world API / binding / standalone / unused | Reached from bindings |",
            "|--------|-------|----------------------------------------------------------|-----------------------|",
        ]
        for m in sorted(self.module_level, key=lambda m: (LEVELS.index(self.module_level[m]), m)):
            n = self.module_items[m]
            b = ", ".join(sorted(self.module_bindings.get(m, set()))) or "—"
            lines.append(f"| `{m}` | {self.label(m)} | {' / '.join(str(n[x]) for x in LEVELS)} | {b} |")
        lines.append("")
        return lines


def ledger_text(abi: CAbi, levels: "Levels | None" = None) -> str:
    body = [
        "# How each part can be used",
        "",
        "_Generated by `scripts/integration_levels.py --write`; `--check` fails CI when this file is stale._",
        "_Examples are not counted as integration._",
        "",
    ] + (levels.markdown() if levels else []) + abi.markdown()
    return "\n".join(body).rstrip("\n") + "\n"


def doc_problems(root: Path, levels: "Levels") -> list[str]:
    """README / README_JP summary tables and the MODULES.md Integration column."""
    out = []
    want = levels.counts()
    for rel in ("README.md", "README_JP.md"):
        p = root / rel
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        if SUMMARY_MARK not in text:
            out.append(f"{rel}: no `{SUMMARY_MARK}` table")
            continue
        rows = []
        for line in text[text.index(SUMMARY_MARK):].splitlines()[1:]:
            if line.startswith("|"):
                rows.append(line)
            elif rows:
                break
        got = {}
        for r in rows:
            cells = [c.strip() for c in r.strip("|").split("|")]
            m = re.search(r"\b(step|world API|binding|standalone|unused)\b", cells[0]) if cells else None
            n = re.fullmatch(r"\d+", cells[-1]) if cells else None
            if m and n:
                got[m.group(1)] = int(n.group(0))
        if got != want:
            out.append(f"{rel}: summary table {got} != measured {want}")
    p = root / MODULES_DOC
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    seen = 0
    col = None  # index of the Integration column in the current table
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if not cells:
            col = None
            continue
        if cells[0] == "Module":
            col = cells.index("Integration") if "Integration" in cells else None
            continue
        mm = re.fullmatch(r"`(\w+)`", cells[0])
        if col is None or not mm or mm.group(1) not in levels.module_level or col >= len(cells):
            continue
        cell = cells[col]
        seen += 1
        want_label = levels.label(mm.group(1))
        if cell != want_label:
            out.append(f"{MODULES_DOC}: `{mm.group(1)}` is listed as `{cell}`, measured `{want_label}`")
    if seen == 0:
        out.append(f"{MODULES_DOC}: no Integration column found (compared nothing)")
    return out


def write_modules_column(root: Path, levels: "Levels") -> None:
    """Add or refresh the Integration column of every module table in docs/MODULES.md."""
    p = root / MODULES_DOC
    out = []
    for line in p.read_text(encoding="utf-8").split("\n"):
        if line.startswith("| Module |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if "Integration" not in cells:
                line = line.rstrip() + " Integration |"
        elif re.match(r"^\|-+\|", line) and out and out[-1].startswith("| Module |") and line.count("|") < out[-1].count("|"):
            line = line.rstrip() + "-------------|"
        else:
            m = re.match(r"^\|\s*`(\w+)`\s*\|", line)
            if m and out and any(o.startswith("| Module |") for o in out[-60:]):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                header = next(o for o in reversed(out) if o.startswith("| Module |"))
                hcells = [c.strip() for c in header.strip().strip("|").split("|")]
                value = levels.label(m.group(1)) if m.group(1) in levels.module_level else "—"
                if len(cells) == len(hcells):
                    cells[-1] = value
                else:
                    cells.append(value)
                line = "| " + " | ".join(cells) + " |"
        out.append(line)
    p.write_text("\n".join(out), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--write", action="store_true", help=f"regenerate {LEDGER} (and the MODULES.md column with an index)")
    ap.add_argument("--check", action="store_true", help="fail on problems, a stale section or docs that disagree")
    ap.add_argument("--scip", default="target/scip")
    ap.add_argument("--no-index", action="store_true", help="C ABI part only (no SCIP index needed)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    levels = None
    if not args.no_index:
        sdir = Path(args.scip) if Path(args.scip).is_absolute() else root / args.scip
        paths = [sdir / n for n in ("native.scip", "wasm.scip", "fuzz.scip") if (sdir / n).exists()]
        if not (sdir / "native.scip").exists():
            print(f"error: SCIP index missing in {sdir} (run scripts/scip_index.sh, or use --no-index)", file=sys.stderr)
            return 1
        levels = Levels(root, paths)
        print("modules: " + ", ".join(f"{k} {v}" for k, v in levels.counts().items()))
    abi = CAbi(root)
    problems = abi.problems() + (levels.problems() if levels else [])
    print("compared: c-abi functions %d, consumers %s, listed gaps %d" % (
        len(abi.functions), ", ".join(f"{label} {len(s & set(abi.functions))}" for _, label, s, _ in abi.consumers),
        len(abi.gaps)))
    text = ledger_text(abi, levels)
    ledger = root / LEDGER
    if args.write:
        if levels is None:
            print("error: --write needs the SCIP index (the ledger has the module tables)", file=sys.stderr)
            return 1
        ledger.write_text(text, encoding="utf-8")
        write_modules_column(root, levels)
    if args.check and levels is not None:
        problems += doc_problems(root, levels)
        current = ledger.read_text(encoding="utf-8") if ledger.exists() else ""
        if current != text:
            problems.append(f"{LEDGER} is stale (run scripts/integration_levels.py --write)")
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
