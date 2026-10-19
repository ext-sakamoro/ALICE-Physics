#!/usr/bin/env python3
"""The C ABI libraries built from the facade (alice-physics) export what the engine
(alice-physics-core) exports.

The published libraries are built from the facade so that their names stay `alice_physics`
(release.yml, README); the `#[no_mangle]` functions live in the engine. This script builds
both crates with the release profile and `--features ffi`, as release.yml does:

    cargo rustc -p <crate> --lib --release --features ffi --crate-type cdylib
    cargo rustc -p <crate> --lib --release --features ffi --crate-type staticlib

each into its own target directory, and compares

  * the shared library: the exported symbols (ELF dynamic symbols, Mach-O external
    symbols, the PE export table);
  * the static library: the defined external symbols that are not Rust-mangled (the C
    ABI and the C symbols of the dependencies; a Rust-mangled name carries a hash of the
    crate graph, which differs between the two builds by construction).

Both sets must be equal, non-empty, and contain every function the C header declares
(include/alice_physics.h). Symbols are read with the toolchain's llvm-nm / llvm-readobj
(`rustup component add llvm-tools`), so the same reader runs on Linux, macOS and Windows.

Exit 1 on a difference, 2 when a set is empty or a tool is missing (compared nothing).

    python3 scripts/facade_artifacts.py [--target-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "alice_physics.h"
CRATES = {"alice-physics-core": "alice_physics_core", "alice-physics": "alice_physics"}
HEADER_FN = re.compile(r"\b(alice_[A-Za-z0-9_]+)\s*\(")
MANGLED = re.compile(r"^_{0,2}(ZN|R[A-Z0-9_])")


def llvm_tool(name: str) -> Path:
    sysroot = subprocess.run(["rustc", "--print", "sysroot"], capture_output=True, text=True, check=True).stdout.strip()
    host = next(ln.split(":", 1)[1].strip() for ln in
                subprocess.run(["rustc", "-vV"], capture_output=True, text=True, check=True).stdout.splitlines()
                if ln.startswith("host:"))
    exe = name + (".exe" if os.name == "nt" else "")
    p = Path(sysroot) / "lib" / "rustlib" / host / "bin" / exe
    if not p.is_file():
        raise FileNotFoundError(f"{p} (rustup component add llvm-tools)")
    return p


def artifact_names(lib: str, kind: str) -> list[str]:
    if kind == "cdylib":
        return [f"lib{lib}.so", f"lib{lib}.dylib", f"{lib}.dll"]
    return [f"lib{lib}.a", f"{lib}.lib"]


def build(package: str, kind: str, target_dir: Path) -> Path:
    cmd = ["cargo", "rustc", "-p", package, "--lib", "--release", "--features", "ffi",
           "--crate-type", kind, "--target-dir", str(target_dir)]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    for name in artifact_names(CRATES[package], kind):
        p = target_dir / "release" / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"{package} {kind}: none of {artifact_names(CRATES[package], kind)} in {target_dir / 'release'}")


def strip_macho(names: set[str], path: Path) -> set[str]:
    # Mach-O prefixes every C symbol with an underscore
    return {n[1:] if n.startswith("_") else n for n in names} if path.suffix in (".dylib", ".a") and sys.platform == "darwin" else names


def shared_symbols(path: Path) -> set[str]:
    if path.suffix == ".dll":
        out = subprocess.run([str(llvm_tool("llvm-readobj")), "--coff-exports", str(path)],
                             capture_output=True, text=True, check=True).stdout
        return set(re.findall(r"^\s*Name:\s*(\S+)\s*$", out, re.M))
    args = ["-D"] if path.suffix == ".so" else ["--extern-only"]
    out = subprocess.run([str(llvm_tool("llvm-nm")), *args, "--defined-only", "--format=just-symbols", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return strip_macho({ln.strip() for ln in out.splitlines() if ln.strip()}, path)


def static_symbols(path: Path) -> set[str]:
    r = subprocess.run([str(llvm_tool("llvm-nm")), "--extern-only", "--defined-only", "--format=just-symbols", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0 and not r.stdout:
        raise RuntimeError(f"llvm-nm {path}: {r.stderr.strip()[-300:]}")
    names = {ln.strip() for ln in r.stdout.splitlines() if ln.strip() and not ln.strip().endswith(":")}
    return {n for n in strip_macho(names, path) if not MANGLED.match(n)}


def compare(label: str, core: set[str], facade: set[str], header: set[str]) -> list[str]:
    out = []
    for n in sorted(core - facade):
        out.append(f"{label}: {n} is exported by the engine's build but not by the facade's")
    for n in sorted(facade - core):
        out.append(f"{label}: {n} is exported by the facade's build but not by the engine's")
    for n in sorted(header - facade):
        out.append(f"{label}: {n} is declared in include/alice_physics.h but the facade's build does not export it")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-dir", default=os.environ.get("CARGO_TARGET_DIR") or str(ROOT / "target"),
                    help="the two builds go to <dir>/facade-artifacts/{core,facade}")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")
    base = Path(a.target_dir) / "facade-artifacts"
    header = set(HEADER_FN.findall(HEADER.read_text(encoding="utf-8")))
    if not header:
        print("facade_artifacts: include/alice_physics.h declares no function (compared nothing)", file=sys.stderr)
        return 2
    problems, counts = [], []
    try:
        for kind, read in (("cdylib", shared_symbols), ("staticlib", static_symbols)):
            core = read(build("alice-physics-core", kind, base / "core"))
            facade = read(build("alice-physics", kind, base / "facade"))
            counts.append(f"{kind} {len(core)}/{len(facade)}")
            if not core or not facade:
                print(f"facade_artifacts: {kind}: an empty symbol set (compared nothing)", file=sys.stderr)
                return 2
            problems += compare(kind, core, facade, header)
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"facade_artifacts: {e}", file=sys.stderr)
        return 2
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    print(f"facade_artifacts: symbols (engine/facade) {', '.join(counts)}, {len(header)} header functions, "
          f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
