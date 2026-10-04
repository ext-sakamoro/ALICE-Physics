#!/usr/bin/env python3
"""preflight --fast: 変更した src/<module>.rs に関係する test だけを選んで走らせる.

CI は全 test を 5 OS で走らせるので、ローカルの push 前検査は「静的検査 + 変更に関係する test」で足りる
(2026-10-04: 全 test の実行で初めて出た指摘は 0 件だった、指摘は全て静的検査で出ている)
選択の漏れは CI が補う 過剰な選択 (単語が一致するだけの test) は安全側なので許す

選び方:
  * 変更した `src/<module>.rs` (または `src/<module>/…`) の module 名を、`alice_physics` を含む tests/*.rs が
    単語として参照していれば、その integration test target を選ぶ
  * 変更した tests/*.rs 自身も選ぶ
  * `determinism_golden` / `determinism_golden_f32` は常に選ぶ
  * lib の unit test は `<module>::` の filter で選ぶ
  * `src/lib.rs` を変えた場合: 追加した行が `mod` / `use` / cfg / comment だけなら、追加された module を変更した module として扱う
    (新 module の追加で全 test に退避しない) それ以外 (行の削除・式の変更) は絞れないので全 test に退避する
  * `required-features` が実行する feature に含まれない test target は選ばない (cargo が明示指定を拒むため)
  * src を変えたのに、golden 以外に 1 本も選ばれなければ失敗する (空振りで green にしない)
  * 実行した test が 0 本なら失敗する

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
ALWAYS = ("determinism_golden", "determinism_golden_f32")


def changed_files(base: str) -> list[str]:
    """base との merge-base からの差分 (commit 済み + 作業 tree)."""
    mb = subprocess.run(
        ["git", "merge-base", "HEAD", base], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    out = subprocess.run(
        ["git", "diff", "--name-only", mb], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [l for l in out.splitlines() if l]


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


def run(cmd: list[str]) -> tuple[int, int]:
    """cmd を実行し (rc, passed 数) を返す (出力はそのまま流す)."""
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    buf = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        buf.append(line)
    proc.wait()
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
        cmd = ["cargo", "test", "--no-fail-fast", *feat_args]
        for t in targets:
            cmd += ["--test", t]
        rc, n = run(cmd)
        if rc != 0:
            return 1
        total += n
    if total == 0:
        print("affected_tests: executed 0 tests (the selection ran nothing)", file=sys.stderr)
        return 1
    print(f"affected_tests: OK ({total} tests passed)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
