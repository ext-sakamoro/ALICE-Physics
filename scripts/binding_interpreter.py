#!/usr/bin/env python3
"""The interpreter the Python binding step of scripts/preflight.sh builds the wheel for, checked before
anything else runs.

The step makes a venv under the target directory, builds the wheel with the venv's maturin and installs
it with the venv's pip. maturin is given the venv's interpreter (`-i`); a `PYO3_PYTHON` in the environment
that names another version makes the extension be built for that one, and the install into the venv then
fails: on 2026-11-24 after 5.7 hours of a full run. So the full run asks here first, in seconds:

  * the interpreter the step will use runs: the venv's when the venv exists, else the `python3` that
    will make it;
  * a `PYO3_PYTHON` in the environment runs and is the same implementation and major.minor.

    binding_interpreter.py --venv TARGET/python-binding-venv     # exit 1 with the reason; 0 with one line

Author: Moroya Sakamoto
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROBE = "import platform, sys; print(platform.python_implementation(), sys.version_info[0], sys.version_info[1])"


class Mismatch(Exception):
    pass


def probe(exe: str) -> tuple[str, int, int]:
    """(implementation, major, minor) of an interpreter, or Mismatch when it does not run"""
    try:
        out = subprocess.run([exe, "-c", PROBE], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise Mismatch(f"{exe} does not run ({e})") from e
    parts = out.stdout.split()
    if out.returncode != 0 or len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
        raise Mismatch(f"{exe} does not answer as a Python interpreter (exit {out.returncode}: "
                       f"{(out.stderr or out.stdout).strip()[:200]})")
    return parts[0], int(parts[1]), int(parts[2])


def name(v: tuple[str, int, int]) -> str:
    return f"{v[0]} {v[1]}.{v[2]}"


def check(venv: Path, env: dict[str, str], which=shutil.which) -> str:
    """the line to print, or Mismatch"""
    mine = venv / "bin" / "python"
    if mine.exists():
        exe, role = str(mine), "the venv's interpreter"
        try:
            version = probe(exe)
        except Mismatch as e:
            raise Mismatch(f"{e}: remove {venv} (the step makes it again)") from e
    else:
        found = which("python3")
        if not found:
            raise Mismatch("no python3 on PATH to make the binding venv with")
        exe, role = found, "python3, which will make the venv"
        version = probe(exe)
    asked = env.get("PYO3_PYTHON")
    if not asked:
        return f"binding interpreter: {exe} ({name(version)}; {role}); PYO3_PYTHON is not set"
    try:
        other = probe(asked)
    except Mismatch as e:
        raise Mismatch(f"PYO3_PYTHON: {e}: unset it for this run") from e
    if other != version:
        raise Mismatch(f"PYO3_PYTHON={asked} is {name(other)} and the binding step uses {exe} ({name(version)}; "
                       f"{role}): the wheel would not install into the venv. Unset PYO3_PYTHON for this run")
    return f"binding interpreter: {exe} ({name(version)}; {role}); PYO3_PYTHON={asked} is the same version"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--venv", required=True, help="the venv of the binding step (it need not exist yet)")
    args = ap.parse_args(argv)
    try:
        print(check(Path(args.venv), dict(os.environ)))
    except Mismatch as e:
        print(f"binding_interpreter: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
