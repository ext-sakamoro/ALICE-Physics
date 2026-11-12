#!/usr/bin/env python3
"""A world that never turns on continuous collision nor makes a cloth does not link their code.

The world step reaches the continuous-collision sweep (and the shape casts it runs) and the cloth
step only through what set_continuous_collision / restore_world and a cloth's constructor set, so a
program that never calls them gets neither under whole-program optimisation. A page that steps a
world of rigid bodies (the size of what it ships) depends on that.

This script builds scripts/lazy-link-probe for wasm32-unknown-unknown (release: opt-level z, LTO,
one unit, names kept), reads the function names from the name section of its two programs, and
checks each watched function:

  * bare (a world stepped and digested, nothing else): absent
  * full (the same with continuous collision on and a cloth added): present

The second is the control: a watched name that the full program does not link either means the
check would pass whatever the code did, so it fails too. Exit 1 on a failed check, 2 when a build
failed or nothing was compared.

    python3 scripts/lazy_link_check.py [--target-dir DIR]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "scripts" / "lazy-link-probe"
TARGET = "wasm32-unknown-unknown"

# (what it is, the identifiers of its path in the mangled name, each as its length-prefixed form)
WATCHED: list[tuple[str, list[str]]] = [
    ("the continuous-collision sweep", ["world_ccd", "ccd_sweep"]),
    # the shape casts the sweep runs: the bulk of what it pulls in, and named even when the sweep is inlined
    ("the sphere cast", ["world_shape_query", "cast_sphere"]),
    ("the sphere overlap", ["overlap_sphere"]),
    ("the cloth's own step (awake)", ["cloth", "step_awake"]),
    ("the cloth's self contact over a frame", ["cloth", "resolve_self_contact_over_frame"]),
]


def leb(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        s += 7
        if x < 0x80:
            return r, i


def function_names(wasm: bytes) -> list[str]:
    """the function names of the name section (subsection 1), in index order"""
    if wasm[:4] != b"\0asm":
        raise ValueError("not a wasm module")
    i, names = 8, []
    while i < len(wasm):
        sid = wasm[i]
        i += 1
        n, i = leb(wasm, i)
        end = i + n
        if sid == 0:
            ln, j = leb(wasm, i)
            if wasm[j:j + ln] == b"name":
                j += ln
                while j < end:
                    sub = wasm[j]
                    j += 1
                    sl, j = leb(wasm, j)
                    se = j + sl
                    if sub == 1:
                        c, k = leb(wasm, j)
                        for _ in range(c):
                            _, k = leb(wasm, k)
                            nl, k = leb(wasm, k)
                            names.append(wasm[k:k + nl].decode("utf-8", "replace"))
                            k += nl
                    j = se
        i = end
    return names


def links(names: list[str], path: list[str]) -> bool:
    """some function's mangled name holds every identifier of the path, in order (as `<len><ident>`)"""
    tokens = [f"{len(p)}{p}" for p in path]
    for n in names:
        at = 0
        for t in tokens:
            k = n.find(t, at)
            if k < 0:
                break
            at = k + len(t)
        else:
            return True
    return False


def build(target_dir: Path) -> dict[str, list[str]]:
    env = {k: v for k, v in os.environ.items() if k not in ("RUSTFLAGS", "CARGO_TARGET_DIR")}
    env["CARGO_TARGET_DIR"] = str(target_dir)
    r = subprocess.run(["cargo", "build", "--release", "--target", TARGET, "--bins"], cwd=PROBE, env=env)
    if r.returncode != 0:
        print("lazy_link_check: the probe did not build", file=sys.stderr)
        sys.exit(2)
    out = {}
    for b in ("bare", "full"):
        p = target_dir / TARGET / "release" / f"{b}.wasm"
        if not p.is_file():
            print(f"lazy_link_check: no {p}", file=sys.stderr)
            sys.exit(2)
        out[b] = function_names(p.read_bytes())
    return out


def check(names: dict[str, list[str]]) -> tuple[int, list[str]]:
    """(compared, problems)"""
    problems, compared = [], 0
    for what, path in WATCHED:
        in_full, in_bare = links(names["full"], path), links(names["bare"], path)
        if not in_full:
            problems.append(f"{what} ({'::'.join(path)}) is not linked even with continuous collision on and a cloth: "
                            "the check would compare nothing (renamed?)")
            continue
        compared += 1
        if in_bare:
            problems.append(f"{what} ({'::'.join(path)}) is linked into a world that never asked for it")
    return compared, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target-dir", type=Path, default=PROBE / "target")
    args = ap.parse_args()
    names = build(args.target_dir)
    for b, ns in names.items():
        if not ns:
            print(f"lazy_link_check: {b}.wasm has no name section (stripped?)", file=sys.stderr)
            return 2
    compared, problems = check(names)
    for p in problems:
        print(f"FAIL {p}")
    print(f"lazy_link_check: {compared} of {len(WATCHED)} watched functions compared "
          f"(bare {len(names['bare'])} functions, full {len(names['full'])})")
    if compared == 0:
        return 2
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
