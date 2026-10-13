#!/usr/bin/env python3
"""alice-physics builds as a dependency of a no_std crate.

A downstream crate builds every crate type a dependency declares in `[lib] crate-type`,
not only the rlib, so a `cdylib` or `staticlib` there is built in no_std mode too and fails
for want of a `#[global_allocator]` and a `#[panic_handler]`. The library's own build cannot
see this: `cargo rustc --crate-type rlib` overrides the declared types, and bare-metal
targets drop a cdylib with only a warning. This check builds `consumers/no_std` (a
`#![no_std]` crate with `alice-physics = { path = "../..", default-features = false }`) the
way a user would, with warnings denied, and reads cargo's JSON messages:

  - the build exits 0;
  - alice-physics was built, and only as an rlib (no cdylib / staticlib artifact);
  - the consumer itself was built.

A build that produced no alice-physics artifact (nothing compared) fails as well.

usage: no_std_consumer.py [--target TRIPLE]
Exit 1 when a condition above fails, 2 when nothing was built.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONSUMER = ROOT / "consumers" / "no_std" / "Cargo.toml"
LIB = "alice-physics"


def artifacts(messages: list[dict], package: str) -> list[dict]:
    """compiler-artifact messages of `package` (by the name in its package id)"""
    out = []
    for m in messages:
        if m.get("reason") != "compiler-artifact":
            continue
        pid = m.get("package_id", "")
        # `path+file:///…/ALICE-Physics#alice-physics@3.0.1` or `alice-physics 3.0.1 (path+…)`
        name = pid.split("#", 1)[1].split("@", 1)[0] if "#" in pid else pid.split(" ", 1)[0]
        if name == package:
            out.append(m)
    return out


def judge(messages: list[dict], rc: int) -> tuple[list[str], int]:
    """errors, and the number of alice-physics artifacts read"""
    errors: list[str] = []
    lib = artifacts(messages, LIB)
    kinds = sorted({k for m in lib for k in m.get("target", {}).get("crate_types", [])})
    if rc != 0:
        errors.append(f"the no_std consumer build failed (exit {rc})")
    if lib and kinds != ["rlib"]:
        errors.append(f"{LIB} was built as {kinds}, a no_std dependent can only use an rlib")
    if lib and not any(f.endswith(".rlib") for m in lib for f in m.get("filenames", [])):
        errors.append(f"no {LIB} .rlib was produced")
    if rc == 0 and not artifacts(messages, "alice-physics-no-std-consumer"):
        errors.append("the consumer crate itself was not built")
    return errors, len(lib)


def main(argv: list[str]) -> int:
    # Windows runners print in cp1252: a compiler message quoting a non-ASCII line must not crash the check
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="backslashreplace")
    cmd = ["cargo", "build", "--manifest-path", str(CONSUMER), "--message-format=json"]
    if "--target" in argv:
        cmd += ["--target", argv[argv.index("--target") + 1]]
    env = dict(os.environ)
    env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " -D warnings").strip()
    env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target" / "no-std-consumer"))
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    messages = []
    for line in proc.stdout.splitlines():
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    for m in messages:
        if m.get("reason") == "compiler-message":
            rendered = m.get("message", {}).get("rendered")
            if rendered:
                sys.stderr.write(rendered)
    sys.stderr.write(proc.stderr[-4000:])
    errors, n = judge(messages, proc.returncode)
    print(f"no_std consumer: {LIB} artifacts {n}, errors {len(errors)}")
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    code = exit_code(errors, n, proc.returncode)
    if code == 2:
        print(f"error: no {LIB} artifact was read (compared nothing)", file=sys.stderr)
    return code


def exit_code(errors: list[str], n: int, rc: int) -> int:
    """2 when a successful build produced no alice-physics artifact (nothing compared),
    1 on any error, 0 otherwise"""
    if n == 0 and rc == 0:
        return 2
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
