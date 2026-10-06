#!/usr/bin/env python3
"""Keep the heavy CI analyses off the path the landing script waits on.

Runners are shared: when the weekly mutation shards and per-push analyses ran
together they held the runners for hours, and the CI of the ci/** branches that
scripts/land.py waits on stayed queued. The heavy workflows below must therefore

  * not run on ci/** pushes (only main, a schedule or a manual dispatch),
  * run at most once at a time: their `concurrency` group must not include the
    ref (a group per ref lets every branch start its own copy),
  * and quality-deep.yml must not trigger on push at all, and must cap its
    mutation matrix with `max-parallel` of at most MAX_PARALLEL.

The workflows are read as text (no YAML library on the runners): only the `on:`
block and the top-level `concurrency:` block are inspected.

  python3 scripts/ci_load_check.py            # exit 1 on a finding
  python3 scripts/ci_load_check.py --root DIR
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEAVY = ("quality-deep.yml", "unsafe-and-parallel.yml", "bench-gate.yml")
NO_PUSH = ("quality-deep.yml",)
MAX_PARALLEL = 4


def top_block(text: str, key: str) -> str:
    """The lines of a top-level `key:` block (until the next top-level key)."""
    out, inside = [], False
    for line in text.splitlines():
        if re.match(rf"^{re.escape(key)}:\s*(#.*)?$", line):
            inside = True
            continue
        if inside and re.match(r"^[A-Za-z_]", line):
            break
        if inside:
            out.append(line)
    return "\n".join(out)


def check(root: Path) -> tuple[list[str], int]:
    errors, seen = [], 0
    wf = root / ".github" / "workflows"
    for name in HEAVY:
        p = wf / name
        if not p.is_file():
            errors.append(f"{name}: missing (the check compares nothing for it)")
            continue
        seen += 1
        text = p.read_text(encoding="utf-8")
        on = top_block(text, "on")
        if not on:
            errors.append(f"{name}: no `on:` block found")
        if re.search(r"ci/\*\*", on):
            errors.append(f"{name}: triggers on ci/** pushes (the landing path); run it on main / schedule only")
        if name in NO_PUSH and re.search(r"^  push:", on, re.M):
            errors.append(f"{name}: triggers on push; it runs on a schedule or dispatch only")
        conc = top_block(text, "concurrency")
        group = re.search(r"^\s*group:\s*(.+)$", conc, re.M)
        if not group:
            errors.append(f"{name}: no top-level concurrency group (two runs can hold runners at once)")
        elif "github.ref" in group.group(1):
            errors.append(f"{name}: concurrency group {group.group(1).strip()} is per ref; use one group per workflow")
        if name == "quality-deep.yml":
            mp = [int(x) for x in re.findall(r"^\s*max-parallel:\s*(\d+)", text, re.M)]
            if not mp:
                errors.append(f"{name}: the mutation matrix has no max-parallel")
            elif max(mp) > MAX_PARALLEL:
                errors.append(f"{name}: max-parallel {max(mp)} exceeds {MAX_PARALLEL}")
    if seen == 0:
        errors.append("no heavy workflow found (compared nothing)")
    return errors, seen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args(argv)
    errors, seen = check(Path(args.root))
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: heavy workflows {seen}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
