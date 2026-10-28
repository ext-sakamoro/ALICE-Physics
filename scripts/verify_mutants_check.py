#!/usr/bin/env python3
"""The pass / fail rules of the mutant jobs of .github/workflows/verify.yml.

  shard: one cargo-mutants run (one shard of one feature axis) of a diff that changed
         src/; fails when cargo-mutants exited with a code that is not a result
         (0 all caught, 2 a mutant missed, 3 a timeout; 1 = usage or a bad --features,
         4 = the unmutated tree failed), and when the run is the only shard (--shards 1)
         of a non-empty planned diff but generated no mutant (a reading of a source change
         that holds nothing); writes `<axis> <shard> <generated> <tested>` to --count.
  total: the counts of every shard; fails when no count was read, and when an axis
         (each of --axes) tested no mutant in all its shards (one empty axis is not
         hidden by the other).

A missed mutant and an unfinished run are decided by the workflow
(scripts/mutants_equivalent.py, scripts/mutants_ratchet.py --complete-only).

  python3 scripts/verify_mutants_check.py shard --rc 0 --out mutants.out --shards 1 \\
      --planned /tmp/planned.diff --axis default --shard 0 --count counts/0-default
  python3 scripts/verify_mutants_check.py total --dir counts --axes default,features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULT_CODES = {0, 2, 3}


def outcome_root(out: Path) -> Path:
    """cargo-mutants writes into --output, or into a mutants.out inside it"""
    return out / "mutants.out" if (out / "mutants.out").is_dir() else out


def counts(out: Path) -> tuple[int, int]:
    """(generated, tested) of one run: the planned mutants (mutants.json) and the
    outcomes other than the unmutated baseline (outcomes.json); 0 when a file is missing"""
    root = outcome_root(out)
    try:
        generated = len(json.loads((root / "mutants.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        generated = 0
    try:
        outcomes = json.loads((root / "outcomes.json").read_text(encoding="utf-8")).get("outcomes", [])
        tested = sum(1 for o in outcomes if o.get("scenario") != "Baseline")
    except (OSError, ValueError, AttributeError):
        tested = 0
    return generated, tested


def shard(rc: int, out: Path, shards: int, planned: Path, axis: str, index: int, count: Path) -> list[str]:
    errors = []
    if rc not in RESULT_CODES:
        errors.append(f"cargo-mutants exited {rc}, not a result (0 / 2 / 3): a usage error, a bad "
                      "--features or a failing unmutated tree")
    generated, tested = counts(out)
    planned_lines = planned.is_file() and planned.stat().st_size > 0
    if shards == 1 and planned_lines and generated == 0:
        errors.append("the planned diff is not empty but no mutant was generated")
    count.parent.mkdir(parents=True, exist_ok=True)
    count.write_text(f"{axis} {index} {generated} {tested}\n", encoding="utf-8")
    print(f"shard {index}/{shards} ({axis}): exit {rc}, {generated} generated, {tested} tested")
    return errors


def total(d: Path, axes: list[str]) -> list[str]:
    tested = {a: 0 for a in axes}
    files = sorted(p for p in d.rglob("*") if p.is_file()) if d.is_dir() else []
    errors = []
    for f in files:
        parts = f.read_text(encoding="utf-8").split()
        if len(parts) != 4 or not all(p.isdigit() for p in parts[1:]):
            errors.append(f"{f.name}: not `<axis> <shard> <generated> <tested>`")
            continue
        if parts[0] not in tested:
            errors.append(f"{f.name}: axis {parts[0]} is not one of {', '.join(axes)}")
            continue
        tested[parts[0]] += int(parts[3])
    if not files:
        errors.append("no shard count was read (compared 0)")
    for a, n in tested.items():
        print(f"axis {a}: {n} tested")
        if files and n == 0:
            errors.append(f"axis {a} tested no mutant")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shard")
    s.add_argument("--rc", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--planned", type=Path, required=True)
    s.add_argument("--axis", required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--count", type=Path, required=True)
    t = sub.add_parser("total")
    t.add_argument("--dir", type=Path, required=True)
    t.add_argument("--axes", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "shard":
        errors = shard(a.rc, a.out, a.shards, a.planned, a.axis, a.shard, a.count)
    else:
        errors = total(a.dir, [x for x in a.axes.split(",") if x])
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
