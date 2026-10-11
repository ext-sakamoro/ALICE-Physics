#!/usr/bin/env python3
"""Shared rules for writing a ratchet baseline: a write may only move it the good way.

A ratchet baseline records a floor (or a list of tolerated violations) that the
check compares against, with some slack for noise. Writing the *current*
measurement as the new baseline lets a value that fell within the slack lower
the floor without anyone seeing it: the check stays green, and the next write
records the drop. These helpers make a write keep the better of the old and the
new value for every entry, and lower an entry only when the caller names it
with a reason, which is written into the baseline next to the entry:

    # allow-worse <entry>: <reason, 12 characters or more>
    <entry> ...

`diff_check` applies the same rule to a baseline file edited by hand: compared
with the baseline on the main branch, an entry that got worse must carry such a
comment.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

MIN_REASON = 12
ALLOW_RE = re.compile(r"^#\s*allow-worse\s+(\S+):\s*(.*?)\s*$")

V = TypeVar("V")


def allowances(keys: list[str], reasons: list[str]) -> tuple[dict[str, str], list[str]]:
    """`{entry: reason}` from paired `--allow-worse` / `--reason` options, and the
    problems with them (unpaired, or a reason shorter than MIN_REASON)."""
    problems = []
    if len(keys) != len(reasons):
        problems.append(f"{len(keys)} --allow-worse but {len(reasons)} --reason: give one reason per entry")
    out = {}
    for key, reason in zip(keys, reasons):
        if len(reason.strip()) < MIN_REASON:
            problems.append(f"--allow-worse {key}: reason shorter than {MIN_REASON} characters")
        out[key] = reason.strip()
    return out, problems


def allow_comments(text: str) -> dict[str, str]:
    """`{entry: reason}` from the `# allow-worse` comment lines of a baseline."""
    out = {}
    for line in text.splitlines():
        m = ALLOW_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def merge_floors(
    old: dict[str, V],
    now: dict[str, V],
    worse: Callable[[V, V], bool],
    allowed: dict[str, str],
) -> tuple[dict[str, V], list[str], dict[str, str]]:
    """The baseline a write records: for an entry in both, `now` unless it is
    worse than `old` (then `old` stays, unless the entry is allowed); entries
    only in `now` are new and recorded; entries only in `old` are gone (their
    source was removed) and dropped. Returns (merged, entries that kept the old
    value, allowances that were used)."""
    merged: dict[str, V] = {}
    kept: list[str] = []
    used: dict[str, str] = {}
    for key, value in now.items():
        if key in old and worse(old[key], value):
            if key in allowed:
                merged[key] = value
                used[key] = allowed[key]
            else:
                merged[key] = old[key]
                kept.append(key)
        else:
            merged[key] = value
    return merged, sorted(kept), used


def diff_check(
    main: dict[str, V],
    current: dict[str, V],
    worse: Callable[[V, V], bool],
    comments: dict[str, str],
) -> tuple[list[str], int]:
    """(errors, entries compared): every entry of `current` that is worse than
    the same entry on main has to carry an `# allow-worse` comment with a
    reason of MIN_REASON characters or more."""
    errors = []
    compared = 0
    for key, value in sorted(current.items()):
        if key not in main:
            continue
        compared += 1
        if worse(main[key], value):
            reason = comments.get(key, "")
            if len(reason) < MIN_REASON:
                errors.append(f"{key}: worse than on main without an `# allow-worse {key}: <reason>` line")
    return errors, compared


def text_at(ref: str, path: Path) -> str | None:
    """The file at `path` in git revision `ref`, or None when it does not exist there."""
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                         cwd=path.parent, check=True).stdout.strip()
    rel = path.resolve().relative_to(Path(top).resolve()).as_posix()
    r = subprocess.run(["git", "show", f"{ref}:{rel}"], capture_output=True, cwd=top, check=False)
    return r.stdout.decode("utf-8") if r.returncode == 0 else None
