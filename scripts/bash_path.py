"""The bash to start a shell script with: Git's bash on Windows, the first `bash` on PATH elsewhere.

On Windows, CreateProcess and often PATH find System32's `bash.exe` first: that is the WSL launcher, which runs no
script ("WSL has no installed distributions" on a machine without one). Git's bash sits next to git
(`<git>/../../bin/bash.exe` or `usr/bin/bash.exe`); a `bash` found under System32 is refused (None), never used.
Used by scripts/land.py. scripts/test_run_tests.py keeps its own copy (it is a shared file whose bytes the shared-files
table pins across repositories); scripts/test_land.py BashParity holds the two to the same verdicts.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable


def find_bash(git: str | None = None, is_windows: bool | None = None,
              which: Callable[[str], str | None] = shutil.which) -> str | None:
    """the bash to run a script with, or None when only System32's WSL launcher (or nothing) is there; `git` is the
    git whose bash to use on Windows (default: the first git on PATH)"""
    if is_windows is None:
        is_windows = os.name == "nt"
    if is_windows:
        git = git or which("git")
        if git:
            root = Path(git).parents[1]
            for cand in (root / "bin" / "bash.exe", root / "usr" / "bin" / "bash.exe"):
                if cand.exists():
                    return str(cand)
        found = which("bash")
        return None if found is None or "system32" in found.lower() else found
    return which("bash")
