#!/usr/bin/env python3
"""Run one unittest file's tests in N processes and prove every test ran once.

    python scripts/run_unittest_shards.py scripts/test_land.py [--shards N] [--shuffle SEED]

The parent loads the file's tests (the discovered list, sorted by id) and gives test i to
shard i mod N: a total function over the list, so no test can fall outside every shard and
none can be in two. Each shard is its own process (`--shard-index K`): it loads the same file,
takes its part of the list, runs it with the standard unittest runner and writes the ids it
executed and their outcomes to a json report.

The parent then requires, and fails otherwise (exit 1):
  - the discovered list is not empty, and every shard was given and executed at least 1 test;
  - the executed ids of all shards, taken together, are exactly the discovered list (none
    missing, none extra, none twice) and each shard process loaded the same list as the parent;
  - every shard process exited 0 and reported no failure and no error (skips are reported, as the
    plain runner reports them).

--shuffle SEED runs each shard's tests in a seeded random order (an isolation check: a test
that only passes after another one shows up as a failure). Shards default to the CPU count,
at most 4 and at least 1. Each shard process is a separate interpreter, so module-level state
(temporary directories, os.environ, monkeypatched module attributes) is not shared.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


def load_module(path: Path):
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"run_unittest_shards: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def flatten(suite) -> list[unittest.TestCase]:
    out = []
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            out.extend(flatten(t))
        else:
            out.append(t)
    return out


def discover(path: Path) -> list[unittest.TestCase]:
    """the file's tests, sorted by id (a load error shows as a _FailedTest, which then fails its shard)"""
    mod = load_module(path)
    tests = flatten(unittest.defaultTestLoader.loadTestsFromModule(mod))
    return sorted(tests, key=lambda t: t.id())


def part(ids: list[str], k: int, n: int) -> list[str]:
    """shard k of n: the ids at positions k, k+n, k+2n, ... of the sorted list"""
    return [x for i, x in enumerate(ids) if i % n == k]


class Recorder(unittest.TextTestResult):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.executed: list[str] = []

    def startTest(self, test):
        self.executed.append(test.id())
        super().startTest(test)


def run_shard(path: Path, k: int, n: int, seed: int | None, report: Path) -> int:
    tests = discover(path)
    by_id = {t.id(): t for t in tests}
    mine = part(sorted(by_id), k, n)
    if seed is not None:
        random.Random(seed * 1000 + k).shuffle(mine)
    runner = unittest.TextTestRunner(resultclass=Recorder, verbosity=1)
    result = runner.run(unittest.TestSuite(by_id[i] for i in mine))
    report.write_text(json.dumps({
        "loaded": sorted(by_id), "given": mine, "executed": result.executed,
        "failures": [t.id() for t, _ in result.failures], "errors": [t.id() for t, _ in result.errors],
        "unexpected_successes": [t.id() for t in result.unexpectedSuccesses],
        "skipped": len(result.skipped), "ran": result.testsRun,
    }), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


def verdict(discovered: list[str], reports: list[dict | None], codes: list[int]) -> list[str]:
    """why the sharded run does not equal one plain run of the discovered list (empty: it does)"""
    why: list[str] = []
    if not discovered:
        return ["no test was discovered"]
    n = len(reports)
    seen: dict[str, int] = {}
    for k, (r, code) in enumerate(zip(reports, codes)):
        if r is None:
            why.append(f"shard {k} wrote no report (exit {code})")
            continue
        if r["loaded"] != discovered:
            why.append(f"shard {k} loaded another test list ({len(r['loaded'])} tests, the parent {len(discovered)})")
        if sorted(r["given"]) != part(discovered, k, n):
            why.append(f"shard {k} was not given its part of the list")
        if not r["executed"]:
            why.append(f"shard {k} executed 0 tests")
        for t in r["executed"]:
            seen[t] = seen.get(t, 0) + 1
        for key in ("failures", "errors", "unexpected_successes"):
            if r[key]:
                why.append(f"shard {k}: {len(r[key])} {key.replace('_', ' ')} (first: {r[key][0]})")
        if code != 0 and not any(r[key] for key in ("failures", "errors", "unexpected_successes")):
            why.append(f"shard {k} exited {code}")
    missing = [t for t in discovered if t not in seen]
    extra = sorted(t for t in seen if t not in set(discovered))
    twice = sorted(t for t, c in seen.items() if c > 1)
    if missing:
        why.append(f"{len(missing)} discovered tests ran in no shard (first: {missing[0]})")
    if extra:
        why.append(f"{len(extra)} executed tests were not discovered (first: {extra[0]})")
    if twice:
        why.append(f"{len(twice)} tests ran in more than one shard (first: {twice[0]})")
    return why


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--shards", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--shuffle", type=int, default=None, metavar="SEED")
    ap.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--report", type=Path, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.shards < 1:
        print("run_unittest_shards: --shards must be at least 1", file=sys.stderr)
        return 2
    if a.shard_index is not None:
        return run_shard(a.file, a.shard_index, a.shards, a.shuffle, a.report)
    t0 = time.monotonic()
    discovered = [t.id() for t in discover(a.file)]
    n = max(1, min(a.shards, len(discovered))) if discovered else a.shards
    with tempfile.TemporaryDirectory(prefix="unittest-shards-") as d:
        procs = []
        for k in range(n):
            rep = Path(d) / f"shard{k}.json"
            cmd = [sys.executable, os.path.abspath(__file__), str(a.file), "--shards", str(n),
                   "--shard-index", str(k), "--report", str(rep)]
            if a.shuffle is not None:
                cmd += ["--shuffle", str(a.shuffle)]
            log = open(Path(d) / f"shard{k}.log", "w+", encoding="utf-8")
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, rep))
        codes, reports = [], []
        for k, (p, log, rep) in enumerate(procs):
            codes.append(p.wait())
            log.seek(0)
            text = log.read()
            log.close()
            print(f"--- shard {k} of {n} (exit {codes[-1]}) ---")
            print(text.rstrip())
            reports.append(json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else None)
    why = verdict(discovered, reports, codes)
    ran = sum(len(r["executed"]) for r in reports if r)
    skipped = sum(r["skipped"] for r in reports if r)
    print(f"run_unittest_shards: {a.file}: discovered {len(discovered)}, executed {ran} in {n} shards "
          f"({', '.join(str(len(r['executed'])) if r else '-' for r in reports)}), skipped {skipped}, "
          f"{time.monotonic() - t0:.1f}s")
    for w in why:
        print(f"run_unittest_shards: FAIL: {w}", file=sys.stderr)
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
