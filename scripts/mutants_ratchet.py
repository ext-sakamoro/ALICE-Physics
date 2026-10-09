#!/usr/bin/env python3
"""Mutation-testing ratchet: no mutant may be missed that was not missed before.

Reads the `mutants.out` directories of a cargo-mutants run (one per shard and
feature axis, as quality-deep.yml uploads them) and compares the missed mutants
with scripts/mutants-missed-baseline.txt. It fails when

  * a mutant is missed that the baseline does not list (a test lost its teeth,
    or new code came without one), or
  * a baseline entry was tested and caught (it is fixed: remove it, so it cannot
    come back unnoticed), or
  * nothing was tested at all (unviable mutants, which did not build, are not
    tested), or
  * a run is incomplete: fewer output directories than `--expect-dirs`, a
    directory without `mutants.json` / `outcomes.json`, a run without an end
    time (cancelled, or killed by its timeout), or fewer outcomes than mutants
    it planned. A shard that did not finish is a failure, not a skipped shard:
    its mutants were not checked, or
  * with `--check-moved`, a mutant one feature axis left to the other (lines
    it does not compile, see scripts/mutants_axis_exclude.py) is not planned on
    that other axis.

A baseline entry that was not tested in a complete run (the code is gone) is
left alone. Mutants are compared without their line and column
(`src/x.rs:12:5: replace f -> T with ...` becomes `src/x.rs: replace f -> T with ...`)
so that edits elsewhere in the file do not move them; repeated identical
mutants are counted. Each entry carries its feature axis, taken from the
directory name (`...-default` / `...-parallel`).

  python3 scripts/mutants_ratchet.py --out mutants-out-*          # check
  python3 scripts/mutants_ratchet.py --out mutants-out-* --write  # record the missed set
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "mutants-missed-baseline.txt"
POS_RE = re.compile(r"^(\S+?\.rs):\d+(?::\d+)?: ")
OUTCOMES = ("caught", "missed", "timeout", "unviable")


def axis_of(d: Path) -> str:
    return "parallel" if d.name.endswith("parallel") else "default"


def normalise(line: str) -> str:
    return POS_RE.sub(r"\1: ", line.strip())


def read_run(dirs: list[Path]) -> dict[str, Counter]:
    """`{outcome: Counter("[axis] mutant")}` over every output directory."""
    out = {k: Counter() for k in OUTCOMES}
    for d in dirs:
        root = d / "mutants.out" if (d / "mutants.out").is_dir() else d
        for k in OUTCOMES:
            f = root / f"{k}.txt"
            if f.is_file():
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.strip():
                        out[k][f"[{axis_of(d)}] {normalise(line)}"] += 1
    return out


def read_baseline(path: Path) -> Counter:
    c = Counter()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                c[line.strip()] += 1
    return c


def completeness(dirs: list[Path], expect_dirs: int) -> list[str]:
    """Errors for a run that is not complete (see the module documentation)."""
    errors = []
    distinct = {d.resolve() for d in dirs}
    if len(distinct) < expect_dirs:
        errors.append(f"{len(distinct)} of {expect_dirs} expected output directories")
    for d in dirs:
        root = d / "mutants.out" if (d / "mutants.out").is_dir() else d
        plan, outcomes = root / "mutants.json", root / "outcomes.json"
        if not plan.is_file() or not outcomes.is_file():
            errors.append(f"{d.name}: no mutants.json / outcomes.json (the run did not start or was cut off)")
            continue
        try:
            planned = len(json.loads(plan.read_text(encoding="utf-8")))
            result = json.loads(outcomes.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            errors.append(f"{d.name}: mutants.json / outcomes.json is not readable ({e})")
            continue
        if result.get("end_time") is None:
            errors.append(f"{d.name}: the run did not finish (no end time: cancelled or timed out)")
        done = sum(1 for o in result.get("outcomes", []) if o.get("scenario") != "Baseline")
        if done < planned:
            errors.append(f"{d.name}: {done} of {planned} planned mutants have an outcome")
    return errors


def moved_check(dirs: list[Path]) -> list[str]:
    """Every mutant one axis left to another (`axis-moved.txt`, written by the
    workflow from scripts/mutants_axis_exclude.py) is planned on that other axis
    (the union of its directories' `mutants.json`). A moved mutant no axis plans
    is a mutant no axis measures."""
    errors = []
    moved: dict[str, set[str]] = {"default": set(), "parallel": set()}
    planned: dict[str, set[str]] = {"default": set(), "parallel": set()}
    for d in dirs:
        root = d / "mutants.out" if (d / "mutants.out").is_dir() else d
        axis = axis_of(d)
        f = root / "axis-moved.txt"
        if not f.is_file():
            errors.append(f"{d.name}: no axis-moved.txt (the mutants left to the other axis are unknown)")
            continue
        moved[axis].update(l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip())
        plan = root / "mutants.json"
        if plan.is_file():
            try:
                planned[axis].update(m.get("name", "") for m in json.loads(plan.read_text(encoding="utf-8")))
            except ValueError:
                pass  # reported by completeness()
    for axis, other in (("default", "parallel"), ("parallel", "default")):
        for name in sorted(moved[axis] - planned[other]):
            errors.append(f"left by the {axis} axis but not planned on the {other} axis: {name}")
    return errors


def compare(run: dict[str, Counter], base: Counter) -> tuple[list[str], int]:
    # unviable mutants did not build: nothing was tested for them
    tested = sum(sum(run[k].values()) for k in ("caught", "missed", "timeout"))
    errors = []
    if tested == 0:
        return ["no mutant was tested (compared nothing)"], 0
    new = run["missed"] - base
    for m, n in sorted(new.items()):
        errors.append(f"newly missed{f' x{n}' if n > 1 else ''}: {m}")
    # an entry is fixed only when this run tested it and did not miss it
    fixed = (base - run["missed"]) & (run["caught"] + run["timeout"])
    for m, n in sorted(fixed.items()):
        errors.append(f"now caught (remove it from the baseline){f' x{n}' if n > 1 else ''}: {m}")
    return errors, tested


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", nargs="+", required=True, help="mutants.out directories (or their parents)")
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--write", action="store_true", help="record the missed mutants as the baseline")
    ap.add_argument("--expect-dirs", type=int, default=1,
                    help="number of output directories a complete run has (shards x feature axes)")
    ap.add_argument("--check-moved", action="store_true",
                    help="every mutant an axis left to the other axis is planned there")
    ap.add_argument("--complete-only", action="store_true",
                    help="check only that every run finished and tested what it planned "
                         "(no baseline; the in-diff run, where any missed mutant already fails)")
    args = ap.parse_args(argv)
    dirs = [Path(p) for p in args.out]
    incomplete = completeness(dirs, args.expect_dirs)
    if incomplete:
        for e in incomplete:
            print(f"error: incomplete run: {e}", file=sys.stderr)
        return 1
    if args.check_moved:
        unmoved = moved_check(dirs)
        if unmoved:
            for e in unmoved:
                print(f"error: {e}", file=sys.stderr)
            return 1
    if args.complete_only:
        print(f"complete: {len(dirs)} output director{'y' if len(dirs) == 1 else 'ies'}")
        return 0
    run = read_run(dirs)
    if args.write:
        lines = sorted(run["missed"].elements())
        Path(args.baseline).write_text(
            "# missed mutants of the weekly run, without line numbers: [axis] file: mutation\n"
            "# written by scripts/mutants_ratchet.py --write; a new missed mutant fails\n"
            + "".join(f"{m}\n" for m in lines), encoding="utf-8", newline="\n")
        print(f"wrote {len(lines)} missed mutants to {args.baseline}")
        return 0
    errors, tested = compare(run, read_baseline(Path(args.baseline)))
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: tested {tested}, missed {sum(run['missed'].values())}, caught {sum(run['caught'].values())}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
