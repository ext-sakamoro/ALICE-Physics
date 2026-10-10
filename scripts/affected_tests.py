#!/usr/bin/env python3
"""preflight --fast: 変更した src/<module>.rs に関係する test だけを選んで走らせる.

CI は全 test を 5 OS で走らせるので、ローカルの push 前検査は「静的検査 + 変更に関係する test」で足りる
(2026-10-04: 全 test の実行で初めて出た指摘は 0 件だった、指摘は全て静的検査で出ている)
選択の漏れは CI が補う 過剰な選択 (単語が一致するだけの test) は安全側なので許す

選び方:
  * 変更した `src/<module>.rs` (または `src/<module>/…`) の module 名を、`alice_physics` を含む tests/*.rs が
    単語として参照していれば、その integration test target を選ぶ
  * 変更した tests/*.rs 自身も選ぶ
  * `determinism_golden` / `determinism_golden_f32` / `determinism_golden_contacts` は常に選ぶ
  * lib の unit test は `<module>::` の filter で選ぶ
  * `src/lib.rs` を変えた場合: 追加した行が `mod` / `use` / cfg / comment だけなら、追加された module を変更した module として扱う
    (新 module の追加で全 test に退避しない) それ以外 (行の削除・式の変更) は絞れないので全 test に退避する
  * `required-features` が実行する feature に含まれない test target は選ばない (cargo が明示指定を拒むため)
  * src を変えたのに、golden 以外に 1 本も選ばれなければ失敗する (空振りで green にしない)
  * 実行した test が 0 本なら失敗する
  * 選んだ target の crate 冒頭の `#![cfg(...)]` を実行する feature で評価する 偽なら、その cfg を真にする
    最も近い feature 集合 (cfg が名指す feature だけを反転、required-features も満たす) の追加の回で走らせる
    (例: `not(feature = "parallel")` は parallel を外した回、`feature = "neural"` は neural を足した回)
    host で真にできない cfg (他 OS の target_os など) は「host では対象外」と報告して CI に任せる
    feature を反転した回が排他な feature の組 (例: wasm と ffi) になると compile_error で回が失敗する
    (赤になるので fail closed は保たれる)
    解釈できない cfg の形は失敗にする (fail closed)
  * どの回でも、cfg が真の target が passed + ignored = 0 なら失敗する (全 test が `#[ignore]` の target は
    0 passed / N ignored で正当)

usage: scripts/affected_tests.py [--base REV] [--features CSV] [--list | --run]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALWAYS = ("determinism_golden", "determinism_golden_f32", "determinism_golden_contacts")


def changed_files(base: str) -> list[str]:
    """base との merge-base からの差分 (commit 済み + 作業 tree)."""
    mb = subprocess.run(
        ["git", "merge-base", "HEAD", base], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    out = subprocess.run(
        ["git", "diff", "--name-only", mb], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    # 未 commit の新規 file (untracked) は `git diff` に出ないので別に拾う
    # (拾わないと commit 前の preflight --fast が新しい test を一度も走らせない)
    new = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return sorted({l for l in (out + new).splitlines() if l})


def module_of(path: str) -> str | None:
    """`src/foo.rs` と `src/foo/bar.rs` は module `foo`、`src/lib.rs` は None (絞れない)."""
    p = Path(path)
    if p.parts[0] != "src" or p.suffix != ".rs":
        return None
    if len(p.parts) == 2:
        return None if p.stem == "lib" else p.stem
    return p.parts[1]


_LIB_ADD_OK = re.compile(
    r"^\+\s*((pub(\([a-z]+\))?\s+)?(mod\s+(\w+)|use\b[^;]*|extern\s+crate\b[^;]*)\s*;?"
    r"|#\[cfg[^\n]*\]|#!?\[[^\n]*\]|//[^\n]*|)\s*$"
)
# `pub use` の括弧の中身だけの行 (`SolverBackend, WorldSnapshotError,`): 再 export の増減で、挙動を変えない
_IDENT_LIST = re.compile(r"^[-+]\s*(?:[A-Za-z_][\w:]*(?:\s+as\s+\w+)?\s*,?\s*)+$")
_MOD_DECL = re.compile(r"^\+\s*(?:pub(?:\([a-z]+\))?\s+)?mod\s+(\w+)\s*;")


def lib_rs_added_modules(diff_text: str) -> list[str] | None:
    """src/lib.rs の diff が「module の宣言・再 export (括弧の中の名前の増減を含む)・属性・comment の追加だけ」なら、追加された module 名を返す.

    行の削除、式 / 定数 / 関数の変更が 1 行でもあれば None (絞れない = 全 test に退避).
    """
    mods: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith(("+++", "---", "@@", "diff ", "index ")):
            continue
        if _IDENT_LIST.match(line):
            continue
        if line.startswith("-"):
            return None
        if line.startswith("+"):
            if not _LIB_ADD_OK.match(line):
                return None
            m = _MOD_DECL.match(line)
            if m:
                mods.append(m.group(1))
    return sorted(set(mods))


def required_features(cargo_toml: str) -> dict[str, set[str]]:
    """[[test]] の name -> required-features."""
    data = tomllib.loads(cargo_toml)
    return {
        t["name"]: set(t.get("required-features", []))
        for t in data.get("test", [])
        if "name" in t
    }


class UnknownCfg(ValueError):
    """A cfg form the evaluator does not understand (fail closed)."""


_CFG_TOKEN = re.compile(r'\s*(?:(?P<str>"[^"]*")|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<p>[(),=]))')


def crate_cfg(text: str) -> str | None:
    """The crate-level `#![cfg(...)]` predicates of a test file, joined as
    `all(...)` when there are several; `None` when the file has none.

    Only the crate header counts: blank lines, `//` comments and inner
    attributes (`#![...]`, possibly over several lines) before the first item.
    A `#![cfg(...)]` inside a `mod` or quoted in a comment is not crate-level
    and is not read."""
    preds = []
    i, n = 0, len(text)
    while i < n:
        # skip whitespace and line comments
        while i < n and text[i] in " \t\r\n":
            i += 1
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if not text.startswith("#![", i):
            break
        depth, k = 0, i + 2
        while k < n:
            depth += {"[": 1, "]": -1}.get(text[k], 0)
            k += 1
            if depth == 0:
                break
        if depth:
            raise UnknownCfg("unbalanced inner attribute in the crate header")
        attr = text[i + 3 : k - 1].strip()
        if attr.startswith("cfg(") and attr.endswith(")"):
            preds.append(attr[4:-1].strip())
        elif attr.startswith("cfg") and not attr.startswith("cfg_attr"):
            raise UnknownCfg(f"unreadable crate cfg {attr!r}")
        i = k
    if not preds:
        return None
    return preds[0] if len(preds) == 1 else "all(" + ", ".join(preds) + ")"


def _tokens(expr: str) -> list[tuple[str, str]]:
    out, i = [], 0
    while i < len(expr):
        m = _CFG_TOKEN.match(expr, i)
        if not m or m.end() == i:
            if expr[i:].strip() == "":
                break
            raise UnknownCfg(f"cannot tokenize cfg at {expr[i:]!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        i = m.end()
    return out


def parse_cfg(expr: str):
    """cfg predicate -> nested tuples: ("all"|"any", [..]) / ("not", x) /
    ("kv", key, value) / ("id", name)."""
    toks = _tokens(expr)
    pos = 0

    def pred():
        nonlocal pos
        if pos >= len(toks) or toks[pos][0] != "id":
            raise UnknownCfg(f"expected a cfg name in {expr!r}")
        name = toks[pos][1]
        pos += 1
        if pos < len(toks) and toks[pos] == ("p", "="):
            pos += 1
            if pos >= len(toks) or toks[pos][0] != "str":
                raise UnknownCfg(f"expected a string after {name} = in {expr!r}")
            val = toks[pos][1][1:-1]
            pos += 1
            return ("kv", name, val)
        if pos < len(toks) and toks[pos] == ("p", "("):
            if name not in ("all", "any", "not"):
                raise UnknownCfg(f"unknown cfg operator {name}() in {expr!r}")
            pos += 1
            args = []
            while pos < len(toks) and toks[pos] != ("p", ")"):
                args.append(pred())
                if pos < len(toks) and toks[pos] == ("p", ","):
                    pos += 1
            if pos >= len(toks):
                raise UnknownCfg(f"unclosed {name}( in {expr!r}")
            pos += 1
            if name == "not":
                if len(args) != 1:
                    raise UnknownCfg(f"not() takes one predicate in {expr!r}")
                return ("not", args[0])
            return (name, args)
        return ("id", name)

    tree = pred()
    if pos != len(toks):
        raise UnknownCfg(f"trailing tokens in cfg {expr!r}")
    return tree


# bare names and key=value pairs that hold in a `cargo test` build besides
# what `rustc --print cfg` reports for the host
TEST_BUILD_CFG = {("id", "test"), ("id", "debug_assertions")}
KNOWN_KEYS = {
    "feature", "target_os", "target_family", "target_arch", "target_env",
    "target_pointer_width", "target_endian", "target_vendor", "target_has_atomic",
    "panic", "target_feature",
}


def eval_cfg(tree, features: set[str], host: set[tuple]) -> bool:
    kind = tree[0]
    if kind == "all":
        return all(eval_cfg(t, features, host) for t in tree[1])
    if kind == "any":
        return any(eval_cfg(t, features, host) for t in tree[1])
    if kind == "not":
        return not eval_cfg(tree[1], features, host)
    if kind == "kv":
        _, key, val = tree
        if key == "feature":
            return val in features
        if key not in KNOWN_KEYS:
            raise UnknownCfg(f"unknown cfg key {key}")
        return ("kv", key, val) in host
    name = tree[1]
    if ("id", name) in host or ("id", name) in TEST_BUILD_CFG:
        return True
    if name in ("unix", "windows", "miri", "doc", "doctest", "proc_macro"):
        return False
    raise UnknownCfg(f"unknown cfg name {name}")


def cfg_features(tree) -> set[str]:
    """Feature names a cfg predicate mentions."""
    if tree[0] in ("all", "any"):
        return set().union(*(cfg_features(t) for t in tree[1])) if tree[1] else set()
    if tree[0] == "not":
        return cfg_features(tree[1])
    if tree[0] == "kv" and tree[1] == "feature":
        return {tree[2]}
    return set()


def host_cfg() -> set[tuple]:
    out = subprocess.run(["rustc", "--print", "cfg"], capture_output=True, text=True, check=True).stdout
    host: set[tuple] = set()
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            host.add(("kv", k, v.strip('"')))
        elif line:
            host.add(("id", line))
    return host


def plan_passes(
    targets: list[str],
    tests: dict[str, str],
    features: set[str],
    host: set[tuple],
    req: dict[str, set[str]],
    known_features: set[str],
) -> tuple[list[tuple[frozenset[str], list[str]]], list[str]]:
    """Group targets into cargo test passes so every target runs with a feature
    set that makes its crate-level cfg true. The first pass is `features`;
    a cfg-false target goes to the nearest feature set that makes it true
    (fewest features flipped among those its cfg names, required-features
    added). Returns (passes, host_skipped): targets no feature set can enable
    on this host. Raises UnknownCfg on a form the evaluator cannot read."""
    passes: dict[frozenset[str], list[str]] = {frozenset(features): []}
    host_skipped: list[str] = []
    for t in targets:
        expr = crate_cfg(tests.get(t, ""))
        tree = parse_cfg(expr) if expr else None
        if tree is None or eval_cfg(tree, features, host):
            passes[frozenset(features)].append(t)
            continue
        atoms = sorted(cfg_features(tree))
        unknown = [f for f in atoms if f not in known_features]
        if unknown:
            raise UnknownCfg(f"{t}: cfg names feature(s) Cargo.toml does not define: {unknown}")
        best = None
        for mask in range(1 << len(atoms)):
            flipped = {atoms[i] for i in range(len(atoms)) if mask >> i & 1}
            cand = (features ^ flipped) | req.get(t, set())
            if eval_cfg(tree, cand, host):
                key = (len(flipped), sorted(flipped))
                if best is None or key < best[0]:
                    best = (key, frozenset(cand))
        if best is None:
            host_skipped.append(t)
        else:
            passes.setdefault(best[1], []).append(t)
    return [(f, ts) for f, ts in passes.items() if ts or f == frozenset(features)], host_skipped


def select(
    changed: list[str],
    tests: dict[str, str],
    req: dict[str, set[str]],
    features: set[str],
    lib_rs_modules: list[str] | None = None,
) -> dict:
    """純関数 (git / cargo を呼ばない): 選択結果を返す.

    tests: target 名 -> file の中身.
    返り値: {"all": bool, "modules": [...], "targets": [...], "skipped": [...], "src_changed": bool}
    """
    src = [f for f in changed if f.startswith("src/") and f.endswith(".rs")]
    src_changed = bool(src)
    lib_narrowed = lib_rs_modules is not None
    if any(module_of(f) is None and not (f == "src/lib.rs" and lib_narrowed) for f in src):
        return {"all": True, "modules": [], "targets": [], "skipped": [], "src_changed": True}
    modules = sorted(
        {module_of(f) for f in src if module_of(f) is not None} | set(lib_rs_modules or [])
    )
    targets: set[str] = set()
    for name, text in tests.items():
        if "alice_physics" not in text:
            continue
        if any(re.search(rf"\b{re.escape(m)}\b", text) for m in modules):
            targets.add(name)
    for f in changed:
        p = Path(f)
        if p.parts[0] == "tests" and len(p.parts) == 2 and p.suffix == ".rs" and p.stem in tests:
            targets.add(p.stem)
    chosen, skipped = [], []
    for name in sorted(targets):
        if req.get(name, set()) <= features:
            chosen.append(name)
        else:
            skipped.append(name)
    return {
        "all": False,
        "modules": modules,
        "targets": chosen,
        "skipped": skipped,
        "src_changed": src_changed,
    }


def load_tests() -> dict[str, str]:
    return {
        p.stem: p.read_text(encoding="utf-8", errors="replace")
        for p in sorted((ROOT / "tests").glob("*.rs"))
    }


def passed_count(output: str) -> int:
    return sum(int(m.group(1)) for m in re.finditer(r"test result: \w+\. (\d+) passed", output))


RUNNING = re.compile(r"^\s*Running tests/([A-Za-z0-9_]+)\.rs\b", re.M)
RESULT = re.compile(r"test result: \w+\. (\d+) passed; \d+ failed; (\d+) ignored")
# ANSI SGR sequences: with CARGO_TERM_COLOR=always cargo colours "Running", and
# the header no longer matches, so every target would read as 0 tests
from ansi import ANSI_RE  # noqa: E402  (CSI, charset selectors, OSC)


def per_target_passed(output: str) -> dict[str, int]:
    """Passed + ignored tests per integration target, from the `Running
    tests/<t>.rs` section headers of cargo's output (a section's result line
    follows it). A target whose tests are all `#[ignore]` (known defects, src
    gaps) counts its ignored tests: it compiled and was run. ANSI colour codes
    are stripped first."""
    output = ANSI_RE.sub("", output)
    counts: dict[str, int] = {}
    heads = list(RUNNING.finditer(output))
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(output)
        m = RESULT.search(output, h.end(), end)
        counts[h.group(1)] = int(m.group(1)) + int(m.group(2)) if m else 0
    return counts


def empty_targets(counts: dict[str, int], selected: list[str]) -> list[str]:
    """Selected targets (cfg true for the pass) with no passed and no ignored
    test: compiled to nothing or missing from the run. The goldens that always
    run must not hide them in the total."""
    return [t for t in selected if counts.get(t, 0) == 0]


def run(cmd: list[str], out: list[str] | None = None) -> tuple[int, int]:
    """cmd を実行し (rc, passed 数) を返す (出力はそのまま流す、`out` に出力を残す)."""
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    buf = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        buf.append(line)
    proc.wait()
    if out is not None:
        out.extend(buf)
    return proc.returncode, passed_count("".join(buf))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--features", default="")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true")
    g.add_argument("--run", action="store_true")
    a = ap.parse_args()
    features = {f for f in a.features.split(",") if f}
    feat_args = ["--features", a.features] if a.features else []

    changed = changed_files(a.base)
    req = required_features((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    lib_mods = None
    if "src/lib.rs" in changed:
        mb = subprocess.run(
            ["git", "merge-base", "HEAD", a.base], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", mb, "--", "src/lib.rs"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
        lib_mods = lib_rs_added_modules(diff)
    sel = select(changed, load_tests(), req, features, lib_mods)

    if sel["all"]:
        print("affected_tests: src/lib.rs changed beyond module declarations -> cannot narrow, running every test", file=sys.stderr)
        if a.list:
            print("ALL")
            return 0
        rc, n = run(["cargo", "test", "--no-fail-fast", *feat_args])
        if rc != 0 or n == 0:
            print(f"affected_tests: FAILED (rc={rc}, passed={n})", file=sys.stderr)
            return 1
        return 0

    always = [t for t in ALWAYS if (ROOT / "tests" / f"{t}.rs").exists()]
    targets = sorted(set(sel["targets"]) | set(always))
    print(
        f"affected_tests: modules={sel['modules']} integration targets={len(sel['targets'])} "
        f"(+{len(always)} golden) skipped(required-features)={sel['skipped']}",
        file=sys.stderr,
    )
    if sel["src_changed"] and not sel["targets"]:
        print(
            "affected_tests: src changed but no test references the changed module(s) "
            "(add a test, or run scripts/preflight.sh without --fast)",
            file=sys.stderr,
        )
        return 3
    if a.list:
        for m in sel["modules"]:
            print(f"LIB {m}")
        for t in targets:
            print(f"TEST {t}")
        return 0

    total = 0
    if sel["modules"]:
        filters = [f"{m}::" for m in sel["modules"]]
        rc, n = run(["cargo", "test", "--lib", *feat_args, "--", *filters])
        if rc != 0:
            return 1
        total += n
    if targets:
        cargo_doc = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
        known = set(cargo_doc.get("features", {})) | {
            d for d, v in cargo_doc.get("dependencies", {}).items()
            if isinstance(v, dict) and v.get("optional")
        }
        try:
            passes, host_skipped = plan_passes(
                targets, load_tests(), features, host_cfg(), req, known
            )
        except UnknownCfg as e:
            print(f"affected_tests: {e} (fail closed)", file=sys.stderr)
            return 1
        if host_skipped:
            print(
                f"affected_tests: not runnable on this host (cfg false for every feature set, "
                f"CI runs them on their target): {', '.join(host_skipped)}",
                file=sys.stderr,
            )
        for i, (pf, pts) in enumerate(passes):
            if not pts:
                continue
            feats = ",".join(sorted(pf))
            cmd = ["cargo", "test", "--no-fail-fast", *(["--features", feats] if feats else [])]
            for t in pts:
                cmd += ["--test", t]
            out: list[str] = []
            rc, n = run(cmd, out)
            counts = per_target_passed("".join(out))
            empty = empty_targets(counts, pts)
            print(
                f"affected_tests: pass {i + 1} [{feats}]: targets {len(pts)}, "
                f"ran {len(pts) - len(empty)}, empty {len(empty)}, passed {n}",
                file=sys.stderr,
            )
            if rc != 0:
                return 1
            total += n
            if empty:
                print(
                    f"affected_tests: target(s) whose cfg is true for [{feats}] ran no test "
                    f"(0 passed, 0 ignored): {', '.join(empty)}",
                    file=sys.stderr,
                )
                return 1
    if total == 0:
        print("affected_tests: executed 0 tests (the selection ran nothing)", file=sys.stderr)
        return 1
    print(f"affected_tests: OK ({total} tests passed)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
