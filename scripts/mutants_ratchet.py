#!/usr/bin/env python3
"""Mutation-testing ratchet: no mutant may be missed that was not missed before.

Reads the `mutants.out` directories of a cargo-mutants run (one per shard and
feature axis, as quality-deep.yml uploads them) and compares the missed mutants
with scripts/mutants-missed-baseline.txt. It fails when

  * a mutant is missed that the baseline does not list (a test lost its teeth,
    or new code came without one), or
  * a baseline entry was tested and caught (it is fixed: remove it, so it cannot
    come back unnoticed), or
  * nothing was tested at all.

A baseline entry that was not tested in this run (its shard timed out, or the
code is gone) is left alone. Mutants are compared without their line and column
(`src/x.rs:12:5: replace f -> T with ...` becomes `src/x.rs: replace f -> T with ...`)
so that edits elsewhere in the file do not move them; repeated identical
mutants are counted. Each entry carries its feature axis, taken from the
directory name (`...-default` / `...-parallel`).

  python3 scripts/mutants_ratchet.py --out mutants-out-*          # check
  python3 scripts/mutants_ratchet.py --out mutants-out-* --write  # record the missed set
"""

from __future__ import annotations

import argparse
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


def compare(run: dict[str, Counter], base: Counter) -> tuple[list[str], int]:
    tested = sum(sum(c.values()) for c in run.values())
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
    args = ap.parse_args(argv)
    run = read_run([Path(p) for p in args.out])
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
