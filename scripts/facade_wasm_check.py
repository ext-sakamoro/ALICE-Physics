#!/usr/bin/env python3
"""The WebAssembly package built from the facade (alice-physics) is the one the engine
(alice-physics-core) gives.

scripts/build_wasm_pkg.sh builds the published package from the facade (so the module and
the files are `alice_physics`); the `#[wasm_bindgen]` items live in the engine. This script
builds both crates the way build_wasm_pkg.sh does (`cargo rustc -p <crate> --lib --target
wasm32-unknown-unknown --features wasm --crate-type cdylib`, release), runs the wasm-bindgen
CLI on each with the same `--out-name alice_physics`, and compares

  * the export names of the two .wasm files (read from the export section), and
  * the generated JavaScript glue and the TypeScript declarations, byte for byte.

Exit 1 on a difference, 2 when a build produced no export or no glue (compared nothing) or a
tool is missing.

    python3 scripts/facade_wasm_check.py [--target-dir DIR] [--js-target web|nodejs|bundler]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATES = {"alice-physics-core": "alice_physics_core", "alice-physics": "alice_physics"}
GLUE = ("alice_physics.js", "alice_physics.d.ts", "alice_physics_bg.wasm.d.ts")


def leb128(data: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        b = data[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7


def wasm_exports(data: bytes) -> list[str]:
    """names in the export section (id 7) of a wasm module"""
    if data[:4] != b"\0asm":
        raise ValueError("not a wasm module")
    i, out = 8, []
    while i < len(data):
        sid = data[i]
        size, i = leb128(data, i + 1)
        end = i + size
        if sid == 7:
            count, j = leb128(data, i)
            for _ in range(count):
                ln, j = leb128(data, j)
                out.append(data[j:j + ln].decode("utf-8"))
                j += ln + 1          # name, then the kind byte
                _, j = leb128(data, j)  # the index
        i = end
    return out


def build(package: str, target_dir: Path, out: Path, js_target: str) -> tuple[list[str], dict[str, bytes]]:
    cmd = ["cargo", "rustc", "-p", package, "--lib", "--target", "wasm32-unknown-unknown", "--release",
           "--features", "wasm", "--crate-type", "cdylib", "--target-dir", str(target_dir)]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    wasm = target_dir / "wasm32-unknown-unknown" / "release" / f"{CRATES[package]}.wasm"
    if not wasm.is_file():
        raise FileNotFoundError(f"{wasm} was not produced")
    exports = wasm_exports(wasm.read_bytes())
    cmd = ["wasm-bindgen", "--target", js_target, "--out-dir", str(out), "--out-name", "alice_physics", str(wasm)]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return exports, {n: (out / n).read_bytes() for n in GLUE if (out / n).is_file()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-dir", default=os.environ.get("CARGO_TARGET_DIR") or str(ROOT / "target"))
    ap.add_argument("--js-target", default="web", choices=["web", "nodejs", "bundler"])
    a = ap.parse_args(argv)
    base = Path(a.target_dir) / "facade-wasm"
    try:
        with tempfile.TemporaryDirectory(prefix="facade-wasm-") as tmp:
            core_x, core_g = build("alice-physics-core", base / "core", Path(tmp) / "core", a.js_target)
            fac_x, fac_g = build("alice-physics", base / "facade", Path(tmp) / "facade", a.js_target)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
        print(f"facade_wasm_check: {e}", file=sys.stderr)
        return 2
    if not core_x or not fac_x or not core_g or not fac_g:
        print("facade_wasm_check: a build gave no export or no glue (compared nothing)", file=sys.stderr)
        return 2
    problems = [f"wasm export {n} only in the engine's build" for n in sorted(set(core_x) - set(fac_x))]
    problems += [f"wasm export {n} only in the facade's build" for n in sorted(set(fac_x) - set(core_x))]
    for n in sorted(set(core_g) | set(fac_g)):
        if core_g.get(n) != fac_g.get(n):
            problems.append(f"{n} differs between the engine's and the facade's package")
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    print(f"facade_wasm_check: {len(fac_x)} exports, {len(fac_g)} glue files compared, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
