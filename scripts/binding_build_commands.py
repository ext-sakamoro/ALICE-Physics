#!/usr/bin/env python3
"""The binding build commands the README documents are the ones that build the bindings.

`[lib] crate-type` is rlib only (3.0.1), so `cargo build --features ffi` builds no C
library at all, silently: a README that still says it would send a reader to a command
that produces nothing. The libraries, the extension and the package are built from the facade
crate (`-p alice-physics` / `-m crates/alice-physics/Cargo.toml`), so their file and module names
are `alice_physics` (the engine's own lib is `alice_physics_core`). The README (and README_JP, with the same commands) keeps the
commands in one marked block:

    <!-- binding-build-commands ... -->
    ```sh
    ...
    ```
    <!-- /binding-build-commands -->

This script reads that block and

  --check   the two READMEs list the same commands (comments aside), at least one; the
            WebAssembly command is what `web/package.json` runs for `build:wasm`; the
            Python command is a `maturin build` with the `python` feature, and the Python
            workflow runs `maturin build` with that feature too; every one of them, and the
            release workflow's `cargo rustc` / `maturin build`, builds the facade
  --run c   runs the README's C ABI commands (the `cargo rustc ... --crate-type` lines)
            and fails unless the shared and the static library both exist afterwards

usage: binding_build_commands.py --check | --run c
Exit 1 on a failed check, 2 when the block holds no command (compared nothing).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCK_RE = re.compile(r"<!-- binding-build-commands\b[^>]*-->\s*```sh\n(.*?)```\s*<!-- /binding-build-commands -->", re.S)
SHARED = ("libalice_physics.so", "libalice_physics.dylib", "alice_physics.dll")
STATIC = ("libalice_physics.a", "alice_physics.lib")
FACADE_P = "-p alice-physics"
FACADE_M = "-m crates/alice-physics/Cargo.toml"


def commands(text: str) -> list[str]:
    m = BLOCK_RE.search(text)
    if not m:
        return []
    return [line.strip() for line in m.group(1).splitlines() if line.strip() and not line.lstrip().startswith("#")]


def check(root: Path) -> tuple[list[str], int]:
    errors: list[str] = []
    en = commands((root / "README.md").read_text(encoding="utf-8"))
    ja = commands((root / "README_JP.md").read_text(encoding="utf-8"))
    if en != ja:
        errors.append(f"README.md and README_JP.md list different binding commands: {en} vs {ja}")
    c = [x for x in en if x.startswith("cargo rustc") and "--features ffi" in x]
    kinds = {k for x in c for k in ("cdylib", "staticlib") if f"--crate-type {k}" in x}
    if kinds != {"cdylib", "staticlib"}:
        errors.append(f"the C ABI commands build {sorted(kinds)}, not both cdylib and staticlib")
    wasm = [x for x in en if "build_wasm_pkg.sh" in x]
    pkg = json.loads((root / "web" / "package.json").read_text(encoding="utf-8"))
    build_wasm = pkg.get("scripts", {}).get("build:wasm", "")
    if len(wasm) != 1 or not build_wasm.endswith(wasm[0]):
        errors.append(f"web/package.json build:wasm ({build_wasm!r}) does not run the README's WebAssembly command {wasm}")
    py = [x for x in en if x.startswith("maturin build")]
    if len(py) != 1 or "--features python" not in py[0]:
        errors.append(f"the README's Python command is not one `maturin build ... --features python`: {py}")
    workflow = (root / ".github" / "workflows" / "python-binding.yml").read_text(encoding="utf-8")
    if not re.search(r"maturin build\b[^\n]*--features python", workflow):
        errors.append("python-binding.yml does not run `maturin build` with --features python")
    for x in c:
        if not re.search(r"(?<!\S)-p alice-physics(?!\S)", x):
            errors.append(f"the README's C ABI command does not build the facade ({FACADE_P}): {x}")
    for x in py:
        if FACADE_M not in x:
            errors.append(f"the README's Python command does not build the facade ({FACADE_M}): {x}")
    release = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    for name, text in (("python-binding.yml", workflow), ("release.yml", release)):
        for m in re.finditer(r"maturin build\b[^\n]*", text):
            if FACADE_M not in m.group(0):
                errors.append(f"{name}: `{m.group(0).strip()}` does not build the facade ({FACADE_M})")
    rustc = re.findall(r"cargo rustc\b(?:[^\n]*\\\n)*[^\n]*", release)
    if not rustc:
        errors.append("release.yml runs no `cargo rustc` (compared nothing)")
    for x in rustc:
        if not re.search(r"(?<!\S)-p alice-physics(?!\S)", x):
            errors.append(f"release.yml: `{' '.join(x.split())}` does not build the facade ({FACADE_P})")
    wasm_script = (root / "scripts" / "build_wasm_pkg.sh").read_text(encoding="utf-8")
    if not re.search(r"cargo rustc -p alice-physics --lib --target wasm32-unknown-unknown", wasm_script):
        errors.append(f"scripts/build_wasm_pkg.sh does not build the facade ({FACADE_P})")
    return errors, len(en)


def run_c(root: Path) -> list[str]:
    en = commands((root / "README.md").read_text(encoding="utf-8"))
    c = [x for x in en if x.startswith("cargo rustc") and "--features ffi" in x]
    if not c:
        return ["the README holds no C ABI command"]
    for line in c:
        print(f"$ {line}", flush=True)
        subprocess.run(shlex.split(line), cwd=root, check=True)
    profile = "release" if all("--release" in x for x in c) else "debug"
    out = Path(os.environ.get("CARGO_TARGET_DIR", root / "target")) / profile
    errors = []
    if not any((out / f).is_file() for f in SHARED):
        errors.append(f"no shared library ({', '.join(SHARED)}) in {out}")
    if not any((out / f).is_file() for f in STATIC):
        errors.append(f"no static library ({', '.join(STATIC)}) in {out}")
    return errors


def main(argv: list[str]) -> int:
    # Windows runners print in cp1252: a compiler message quoting a non-ASCII line must not crash the check
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")
    if argv == ["--check"]:
        errors, n = check(ROOT)
        print(f"binding build commands: {n} in the README, errors {len(errors)}")
        if n == 0:
            print("error: no binding-build-commands block with a command (compared nothing)", file=sys.stderr)
            return 2
    elif argv == ["--run", "c"]:
        errors = run_c(ROOT)
        print(f"binding build commands (C ABI): errors {len(errors)}")
    else:
        print(__doc__, file=sys.stderr)
        return 1
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
