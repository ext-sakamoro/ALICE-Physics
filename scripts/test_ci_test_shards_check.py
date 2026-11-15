#!/usr/bin/env python3
"""Tests for scripts/ci_test_shards_check.py: a test job whose shards do not run every step
exactly once per runner fails, and the real ci.yml passes."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci_test_shards_check as cs  # noqa: E402

HERE = Path(__file__).resolve().parent
S = "${{ !contains(matrix.os, ', ') || endsWith(matrix.os, ', %s') }}"


def shard(k: str) -> str:
    return S.replace("%s", k)


def wf(values: list[str], steps: list[tuple[str, str | None]]) -> str:
    """a workflow with a test job over `values` and steps (name, if)"""
    body = "".join(f"      - name: {n}\n" + (f"        if: {c}\n" if c else "") + "        run: echo\n" for n, c in steps)
    # fromJSON, as ci.yml writes it (a plain YAML list would split the values on their commas)
    vals = ", ".join(f'"{v}"' for v in values)
    return ("name: CI\non:\n  push:\n    branches: [main]\njobs:\n  test:\n    name: Test (${{ matrix.os }})\n"
            "    runs-on: ubuntu-latest\n    strategy:\n      matrix:\n"
            f"        os: ${{{{ fromJSON('[{vals}]') }}}}\n    steps:\n      - uses: actions/checkout@v5\n{body}")


VALUES = ["macos-latest"] + [f"ubuntu-latest, {k}" for k in cs.SHARDS]
GOOD = [("Install Rust toolchain", None), ("py check", shard("checks")), ("build", shard("default")),
        ("par", shard("parallel")), ("feat", shard("features")),
        ("mutants list", "matrix.os == 'ubuntu-latest, default'")]


class Check(unittest.TestCase):
    def errors(self, values, steps):
        return cs.check(wf(values, steps))[0]

    def test_a_partition_passes(self):
        errs, n = cs.check(wf(VALUES, GOOD))
        self.assertEqual(errs, [])
        self.assertEqual(n, (len(GOOD) + 1) * 2)

    def test_a_step_in_no_shard_fails(self):
        steps = GOOD + [("orphan", "${{ endsWith(matrix.os, ', nowhere') }}")]
        self.assertTrue(any("orphan: runs in 0 shards" in e for e in self.errors(VALUES, steps)))

    def test_a_step_in_two_shards_fails(self):
        steps = GOOD + [("twice", "${{ endsWith(matrix.os, ', checks') || endsWith(matrix.os, ', default') }}")]
        self.assertTrue(any("twice: runs in 2 shards" in e for e in self.errors(VALUES, steps)))

    def test_an_unconditioned_step_that_is_not_shared_fails(self):
        self.assertTrue(any("every: runs in 4 shards" in e for e in self.errors(VALUES, GOOD + [("every", None)])))

    def test_a_step_macos_skips_fails(self):
        steps = GOOD + [("linux only", "${{ endsWith(matrix.os, ', checks') }}")]
        self.assertTrue(any("does not run on macos-latest" in e for e in self.errors(VALUES, steps)))

    def test_a_missing_shard_value_fails(self):
        vals = [v for v in VALUES if not v.endswith("parallel")]
        errs = self.errors(vals, GOOD)
        self.assertTrue(any("has shards" in e for e in errs), errs)

    def test_a_shard_with_no_step_of_its_own_fails(self):
        steps = [s for s in GOOD if s[0] != "feat"]
        self.assertTrue(any("ubuntu-latest, features runs no step" in e for e in self.errors(VALUES, steps)))

    def test_a_step_pinned_to_macos_runs_there_only(self):
        self.assertEqual(self.errors(VALUES, GOOD + [("mac only", "matrix.os == 'macos-latest'")]), [])

    def test_a_step_pinned_to_a_value_the_matrix_lacks_fails(self):
        # the old unsharded name: after sharding no job has it, so the step would silently never run
        errs = self.errors(VALUES, GOOD + [("stale pin", "matrix.os == 'ubuntu-latest'")])
        self.assertTrue(any("which no event's matrix has" in e for e in errs), errs)

    def test_a_shared_step_in_no_shard_fails(self):
        steps = [s for s in GOOD if s[0] != "Install Rust toolchain"] + [
            ("Install Rust toolchain", "${{ !contains(matrix.os, ', ') || endsWith(matrix.os, ', none') }}")]
        self.assertTrue(any("shared, but runs in no shard" in e for e in self.errors(VALUES, steps)))

    def test_an_unreadable_condition_fails(self):
        steps = GOOD + [("odd", "${{ runner.os == 'Linux' }}")]
        self.assertTrue(any("cannot read the condition" in e for e in self.errors(VALUES, steps)))

    def test_no_sharded_runner_fails(self):
        self.assertTrue(any("no sharded runner" in e for e in self.errors(["macos-latest"], GOOD)))

    def test_the_real_ci_yml_passes(self):
        errs, n = cs.check((HERE.parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
        self.assertEqual(errs, [])
        self.assertGreater(n, 0)


class Runs(unittest.TestCase):
    def test_conditions(self):
        self.assertTrue(cs.runs(shard("checks"), "macos-latest"))
        self.assertTrue(cs.runs(shard("checks"), "windows-latest, checks"))
        self.assertFalse(cs.runs(shard("checks"), "windows-latest, default"))
        self.assertFalse(cs.runs("${{ !endsWith(matrix.os, ', checks') }}", "ubuntu-latest, checks"))
        self.assertTrue(cs.runs("${{ !endsWith(matrix.os, ', checks') }}", "macos-15-intel"))
        self.assertTrue(cs.runs(None, "anything"))


if __name__ == "__main__":
    unittest.main()
