#!/usr/bin/env python3
"""Tests for scripts/run_ignored.py: a run that checked nothing must not pass."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_ignored as ri  # noqa: E402

RUNTIME = {"file": "tests/a.rs", "line": 3, "binary": "a", "name": "slow_one",
           "reason": "runtime: 40 s", "category": ri.CAT_RUNTIME}
GAP = {"file": "tests/a.rs", "line": 9, "binary": "a", "name": "gap_one",
       "reason": "src gap: not yet", "category": ri.CAT_EXPECTED_RED}

OK_RUN = (
    "running 2 tests\n"
    "test slow_one ... ok\n"
    "test gap_one ... FAILED\n"
    "test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 9 filtered out\n"
)


def run_main(entries, binaries, runs):
    """`main()` with the table, the build and the binaries replaced."""
    def fake_run(label, exe, skip):
        rc, out = runs[exe]
        return ri.parse_outcomes(out), out, rc
    with mock.patch.object(ri, "collect_table", return_value=(entries, [])), \
         mock.patch.object(ri, "build_binaries", return_value=binaries), \
         mock.patch.object(ri, "run_binary", side_effect=fake_run), \
         mock.patch.object(ri, "native_features", return_value="std"), \
         mock.patch.object(ri, "emit"), \
         mock.patch.object(sys, "argv", ["run_ignored.py"]):
        return ri.main()


class Crash(unittest.TestCase):
    def test_a_normal_run_with_failures_is_not_a_crash(self):
        self.assertFalse(ri.crashed(101, OK_RUN))
        self.assertFalse(ri.crashed(0, OK_RUN.replace("FAILED. 1 passed; 1", "ok. 2 passed; 0")))

    def test_a_signal_or_a_missing_summary_is_a_crash(self):
        self.assertTrue(ri.crashed(-11, OK_RUN))
        self.assertTrue(ri.crashed(101, "running 2 tests\ntest slow_one ... ok\n"))
        self.assertTrue(ri.crashed(0, ""))


class Main(unittest.TestCase):
    def test_a_documented_run_passes(self):
        self.assertEqual(run_main([RUNTIME, GAP], [("a", "exe-a")], {"exe-a": (101, OK_RUN)}), 0)

    def test_a_crashed_binary_fails(self):
        out = "running 2 tests\ntest slow_one ... ok\n"
        self.assertEqual(run_main([RUNTIME, GAP], [("a", "exe-a")], {"exe-a": (-6, out)}), 1)

    def test_no_test_binary_fails(self):
        self.assertEqual(run_main([RUNTIME, GAP], [], {}), 1)

    def test_an_empty_table_fails(self):
        self.assertEqual(run_main([], [("a", "exe-a")], {"exe-a": (0, OK_RUN)}), 1)

    def test_a_runtime_test_that_did_not_run_fails(self):
        out = ("running 1 test\ntest gap_one ... FAILED\n"
               "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 9 filtered out\n")
        self.assertEqual(run_main([RUNTIME, GAP], [("a", "exe-a")], {"exe-a": (101, out)}), 1)



class Tally(unittest.TestCase):
    def test_every_runtime_test_ran(self):
        ran, expected, problems = ri.runtime_tally([RUNTIME, GAP], {("a", "slow_one"): "ok", ("a", "gap_one"): "FAILED"})
        self.assertEqual((ran, expected, problems), (1, 1, []))

    def test_a_failed_runtime_test_still_counts_as_ran(self):
        ran, _, problems = ri.runtime_tally([RUNTIME], {("a", "slow_one"): "FAILED"})
        self.assertEqual((ran, problems), (1, []))

    def test_a_shortfall_is_a_problem(self):
        other = dict(RUNTIME, name="slow_two")
        ran, expected, problems = ri.runtime_tally([RUNTIME, other], {("a", "slow_one"): "ok"})
        self.assertEqual((ran, expected), (1, 2))
        self.assertEqual(problems, ["only 1 of 2 `runtime:` tests ran"])

    def test_a_same_name_in_another_binary_does_not_count(self):
        ran, _, problems = ri.runtime_tally([RUNTIME], {("b", "slow_one"): "ok"})
        self.assertEqual(ran, 0)
        self.assertTrue(problems)

    def test_no_runtime_entry_is_a_problem(self):
        ran, expected, problems = ri.runtime_tally([GAP], {("a", "gap_one"): "FAILED"})
        self.assertEqual((ran, expected), (0, 0))
        self.assertEqual(problems, ["the table has no `runtime:` test, so this run checked none"])

    def test_a_table_without_runtime_tests_fails_the_job(self):
        self.assertEqual(run_main([GAP], [("a", "exe-a")], {"exe-a": (101, OK_RUN)}), 1)

if __name__ == "__main__":
    unittest.main()
