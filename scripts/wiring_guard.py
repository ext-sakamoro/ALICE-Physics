#!/usr/bin/env python3
"""配線ガード: 実装したが production から呼ばれていない item を CI で止める検査器.

検査 A (dead_code): `#[allow(dead_code)]` / `#![allow(dead_code)]` は
    `// ALLOW-DEAD: <12 字以上の理由>` を直前に置くか baseline に載せる
検査 B (unwired): `src/` の `pub` / `pub(crate)` な fn / struct / enum / const / static /
    trait / type / union で、test と自身の宣言と `use` 行を除いた production code
    (src / examples / benches / fuzz / bindings 等) に 1 度も現れないものは未配線.
    `// ALLOW-UNWIRED: <12 字以上の理由>` を直前に置くか baseline に載せる
検査 C: 検査対象が 0 件なら fail (検査器が空振りして green になるのを防ぐ)

baseline (`scripts/wiring-baseline.txt`) は既存の違反を記録するラチェットで、
新規の違反だけが fail する 解消された entry が残っていても fail (stale_baseline).

限界 (fail-open 側): 名前の字面一致で数えるので、他の item / field / method と
同名なら「配線済」と誤判定する (`new` 等) 偽陽性より偽陰性を選んでいる
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MIN_REASON = 12
SKIP_DIRS = {"target", ".git", ".claude", "tests", "node_modules"}
EXEMPT_ATTRS = re.compile(r"no_mangle|export_name|wasm_bindgen|pyfunction|pyclass|pymethods|napi|uniffi")
DEF_RE = re.compile(
    r"\bpub(?:\([^)]*\))?\s+"
    r"(?:(?:const|unsafe|async|extern(?:\s+\"[^\"]*\")?|default)\s+)*"
    r"(fn|struct|enum|const|static|trait|type|union)\s+([A-Za-z_]\w*)"
)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DECL_RE = re.compile(r"\b(?:fn|struct|enum|const|static|trait|type|mod|union)\s+([A-Za-z_][A-Za-z0-9_]*)")
USE_RE = re.compile(r"\b(?:pub(?:\([^)]*\))?\s+)?use\s+[^;]*;")
CFG_TEST_RE = re.compile(r"#\s*\[\s*cfg\s*\(\s*(?:all\s*\(\s*)?test\b[^\]]*\]")
ALLOW_DEAD_RE = re.compile(r"#\s*!?\s*\[[^\]]*\ballow\s*\([^)]*\bdead_code\b")
MARKER_RE = re.compile(r"//\s*ALLOW-(UNWIRED|DEAD)\s*:(.*)$")


@dataclass(frozen=True)
class Violation:
    kind: str
    key: str
    message: str


def strip_rust(src: str) -> str:
    """コメントと文字列 / 文字 literal を空白に置換する (改行と長さは保つ)."""
    out = list(src)
    n = len(src)
    i = 0

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            blank(i, j)
            i = j
        elif c == "r" and re.match(r'r#*"', src[i : i + 40]) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            m = re.match(r'r(#*)"', src[i:])
            close = '"' + m.group(1)
            j = src.find(close, i + m.end())
            j = n if j < 0 else j + len(close)
            blank(i, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            blank(i, j)
            i = j
        elif c == "'":
            if i + 1 < n and src[i + 1] == "\\":
                j = src.find("'", i + 2)
                j = n if j < 0 else j + 1
                blank(i, j)
                i = j
            elif i + 2 < n and src[i + 2] == "'":
                blank(i, i + 3)
                i += 3
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def blank_span(code: str, a: int, b: int) -> str:
    return code[:a] + "".join("\n" if ch == "\n" else " " for ch in code[a:b]) + code[b:]


def remove_cfg_test(code: str) -> str:
    """`#[cfg(test)]` が付いた item (mod / fn / use) を丸ごと空白にする."""
    while True:
        m = CFG_TEST_RE.search(code)
        if not m:
            return code
        j = m.end()
        n = len(code)
        while j < n and code[j] not in "{;":
            j += 1
        if j >= n:
            return blank_span(code, m.start(), n)
        if code[j] == ";":
            code = blank_span(code, m.start(), j + 1)
            continue
        depth, k = 0, j
        while k < n:
            if code[k] == "{":
                depth += 1
            elif code[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        code = blank_span(code, m.start(), min(k + 1, n))


def rs_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*.rs"):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts[:-1]):
            continue
        out.append(p)
    return sorted(out)


def lineno(code: str, pos: int) -> int:
    return code.count("\n", 0, pos)


def parse_baseline(text: str) -> tuple[set[str], dict[str, int]]:
    unwired: set[str] = set()
    dead: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "unwired" and len(parts) == 2:
            unwired.add(parts[1])
        elif parts[0] == "dead_code" and len(parts) == 3:
            dead[parts[1]] = int(parts[2])
    return unwired, dead


def check(root: Path, baseline_text: str = "") -> list[Violation]:
    root = Path(root)
    base_unwired, base_dead = parse_baseline(baseline_text)
    vs: list[Violation] = []

    src_dir = root / "src"
    files = rs_files(root)
    stripped: dict[Path, str] = {}
    raw: dict[Path, list[str]] = {}
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        raw[p] = text.split("\n")
        stripped[p] = remove_cfg_test(strip_rust(text))

    # 参照 corpus: use 文を除いた production code 全体
    corpus = "\n".join(USE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), c) for c in stripped.values())

    defs: list[tuple[str, str, int, Path]] = []  # (name, key, line, path)
    exempt: set[str] = set()
    for p in files:
        if src_dir not in p.parents and p.parent != src_dir:
            continue
        code = stripped[p]
        lines = code.split("\n")
        rel = p.relative_to(root).as_posix()
        for m in DEF_RE.finditer(code):
            ln = lineno(code, m.start())
            above = " ".join(lines[max(0, ln - 4) : ln + 1])
            key = f"{rel}::{m.group(2)}"
            if EXEMPT_ATTRS.search(above):
                exempt.add(key)
            defs.append((m.group(2), key, ln, p))

    if not src_dir.is_dir() or not defs:
        return [Violation("empty_scan", str(root), "検査対象の pub item が 0 件 (src/ が無いか、検査器が何も見ていない)")]

    ident_count = Counter(IDENT_RE.findall(corpus))
    decl_count = Counter(m.group(1) for m in DECL_RE.finditer(corpus))

    def referenced(name: str) -> bool:
        return ident_count[name] - decl_count[name] > 0

    # --- markers ---
    unwired_markers: dict[str, tuple[Path, int, str]] = {}  # key -> (file, line, reason)
    dead_markers: list[tuple[Path, int, str]] = []
    for p in files:
        code_lines = stripped[p].split("\n")
        rel = p.relative_to(root).as_posix()
        for i, rawline in enumerate(raw[p]):
            m = MARKER_RE.search(rawline)
            if not m:
                continue
            kind, reason = m.group(1), m.group(2).strip()
            if len(reason) < MIN_REASON:
                vs.append(Violation("bad_marker", f"{rel}:{i + 1}", f"ALLOW-{kind} の理由が {MIN_REASON} 字未満: {reason!r}"))
                continue
            if code_lines[i].strip():
                target = i  # 同一行にコードがあればその行
            else:
                j = i + 1
                target = None
                while j < len(code_lines):
                    c = code_lines[j]
                    if not c.strip():
                        j += 1
                        continue
                    if kind == "DEAD" and ALLOW_DEAD_RE.search(c):
                        target = j
                        break
                    if c.lstrip().startswith("#"):
                        j += 1
                        continue
                    target = j
                    break
            if kind == "UNWIRED":
                names = [(dm.group(2)) for dm in DEF_RE.finditer(code_lines[target])] if target is not None else []
                if not names:
                    vs.append(Violation("stale_marker", f"{rel}:{i + 1}", "ALLOW-UNWIRED の直後に pub item が無い"))
                for nm in names:
                    unwired_markers[f"{rel}::{nm}"] = (p, i + 1, reason)
            else:
                if target is None or not ALLOW_DEAD_RE.search(code_lines[target]):
                    vs.append(Violation("stale_marker", f"{rel}:{i + 1}", "ALLOW-DEAD の直後に allow(dead_code) が無い"))
                else:
                    dead_markers.append((p, target, reason))

    # --- 検査 B: unwired ---
    current_unwired: set[str] = set()
    for name, key, ln, p in defs:
        if key in exempt or referenced(name):
            continue
        current_unwired.add(key)
        if key in unwired_markers or key in base_unwired:
            continue
        vs.append(Violation("unwired", key, f"{name}: production code から 1 度も参照されていない (test / doc / use のみ)"))
    for key, (p, ln, _r) in unwired_markers.items():
        if key not in current_unwired:
            vs.append(Violation("stale_marker", f"{key}", f"ALLOW-UNWIRED があるが配線済 ({p.name}:{ln}) マーカーを消す"))
    for key in sorted(base_unwired - current_unwired):
        vs.append(Violation("stale_baseline", key, "baseline にあるが既に配線済 / 消えている 行を消す"))

    # --- 検査 A: dead_code ---
    marked_lines = {(p, ln) for p, ln, _ in dead_markers}
    actual_unmarked: dict[str, int] = {}
    for p in files:
        rel = p.relative_to(root).as_posix()
        for i, c in enumerate(stripped[p].split("\n")):
            if ALLOW_DEAD_RE.search(c) and (p, i) not in marked_lines:
                actual_unmarked[rel] = actual_unmarked.get(rel, 0) + 1
    for rel, n in sorted(actual_unmarked.items()):
        allowed = base_dead.get(rel, 0)
        if n > allowed:
            vs.append(Violation("dead_code", rel, f"理由マーカーの無い allow(dead_code) が {n} 件 (baseline {allowed})"))
    for rel, allowed in sorted(base_dead.items()):
        n = actual_unmarked.get(rel, 0)
        if allowed > n:
            vs.append(Violation("stale_baseline", rel, f"baseline の dead_code {allowed} 件に対し実際は {n} 件 数を減らす"))
    return vs


def build_baseline(root: Path) -> str:
    vs = check(root, "")
    lines = ["# wiring-guard baseline: 既存の違反 (ラチェット) 新規は fail、解消したら行を消す"]
    dead: dict[str, int] = {}
    for v in vs:
        if v.kind == "unwired":
            lines.append(f"unwired {v.key}")
        elif v.kind == "dead_code":
            dead[v.key] = int(re.search(r"(\d+) 件", v.message).group(1))
    lines += [f"dead_code {k} {n}" for k, n in sorted(dead.items())]
    return "\n".join([lines[0]] + sorted(lines[1:])) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--baseline", default="scripts/wiring-baseline.txt")
    ap.add_argument("--update-baseline", action="store_true", help="現在の違反を baseline として書き出す (レビュー必須)")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    bpath = root / a.baseline
    if a.update_baseline:
        bpath.write_text(build_baseline(root), encoding="utf-8")
        print(f"wrote {bpath}")
        return 0
    text = bpath.read_text(encoding="utf-8") if bpath.exists() else ""
    vs = check(root, text)
    if vs:
        for v in vs:
            print(f"{v.kind}: {v.key}: {v.message}", file=sys.stderr)
        print(f"wiring-guard: {len(vs)} violation(s)", file=sys.stderr)
        return 1
    print("wiring-guard: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
