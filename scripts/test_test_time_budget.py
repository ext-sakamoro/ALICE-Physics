#!/usr/bin/env python3
"""Tests of scripts/test_time_budget.py: the projection refuses what would not fit."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_time_budget as t  # noqa: E402

CAL = t.CALIBRATION
FLUID = "tests/analytic_fluid_sdf_body.rs"
# the slowest per-push job as it was before the fluid file existed: 56 min, of 120
LEDGER = t.format_ledger("0" * 40, [
    {"name": "Test (macos-latest)", "limit": 120, "total": 3360.0,
     "bins": {CAL: [50.0, 42.0], "tests/small.rs": [0.5, 0.5]}},
    {"name": "Test (ubuntu-latest)", "limit": 120, "total": 2400.0,
     "bins": {CAL: [34.0, 34.0]}},
])


class Projection(unittest.TestCase):
    def setUp(self):
        self.ledger = t.parse_ledger(LEDGER)

    def test_the_900_frame_fluid_file_is_refused(self):
        # here the calibration binary takes 23 s (ratio 2.0 on the slowest job) and the
        # 900-frame file 600 s: 2 passes of 1200 s are 40 min on top of 56
        report, refusals = t.project(self.ledger, {CAL: 23.0, FLUID: 600.0}, [FLUID])
        self.assertEqual(len(report), 2)
        self.assertEqual(len(refusals), 0, "56 + 40 = 96 min is inside 105")
        # with the job where main stood after it (106 min) the same file does not fit
        tight = t.parse_ledger(LEDGER.replace("\t3360", "\t6360"))
        _, refusals = t.project(tight, {CAL: 23.0, FLUID: 600.0}, [FLUID])
        self.assertEqual(len(refusals), 1)
        self.assertIn("Test (macos-latest)", refusals[0])
        self.assertIn(FLUID, refusals[0])

    def test_a_file_already_in_the_ledger_counts_only_its_change(self):
        # the ledger holds the 900-frame file (2 x 1200 s) in a 106 min job; cut to 120
        # frames (80 s here, ratio 2.0) the job is projected 35 min shorter, and fits
        with_fluid = LEDGER.replace("\t3360", "\t6360") + (
            f"bin\tTest (macos-latest)\t{FLUID}\t1200.00,1200.00\n")
        ledger = t.parse_ledger(with_fluid)
        report, refusals = t.project(ledger, {CAL: 23.0, FLUID: 80.0}, [FLUID])
        self.assertEqual(refusals, [])
        self.assertIn("-34.7 min", report[0])
        # unchanged (600 s here): still 106 min, over the 105 budget
        _, refusals = t.project(ledger, {CAL: 23.0, FLUID: 600.0}, [FLUID])
        self.assertEqual(len(refusals), 1)

    def test_nothing_compared_is_an_error(self):
        with self.assertRaises(t.BudgetError):
            t.parse_ledger("# only a comment\n")
        with self.assertRaises(t.BudgetError):
            t.project(self.ledger, {FLUID: 600.0}, [FLUID])  # no calibration time
        with self.assertRaises(t.BudgetError):
            t.project(self.ledger, {CAL: 0.0, FLUID: 600.0}, [FLUID])
        with self.assertRaises(t.BudgetError):
            t.project(self.ledger, {CAL: 23.0}, [FLUID])  # the changed binary not timed
        no_cal = t.parse_ledger("job\tTest (x)\t120\t600\nbin\tTest (x)\ttests/a.rs\t3.00\n")
        with self.assertRaises(t.BudgetError):
            t.project(no_cal, {CAL: 23.0, FLUID: 1.0}, [FLUID])

    def test_the_margin_is_what_refuses(self):
        tight = t.parse_ledger(LEDGER.replace("\t3360", "\t6360"))
        _, refusals = t.project(tight, {CAL: 23.0, FLUID: 600.0}, [FLUID], margin=0)
        self.assertEqual(len(refusals), 1, "146 min is over 120 whatever the margin")
        _, refusals = t.project(self.ledger, {CAL: 23.0, FLUID: 600.0}, [FLUID], margin=30)
        self.assertEqual(len(refusals), 1, "96 min is over 120 - 30")


class Logs(unittest.TestCase):
    def test_a_jobs_log_gives_each_pass_of_each_binary(self):
        log = "\n".join([
            "2026-11-18T05:54:00Z \x1b[1m\x1b[92m     Running\x1b[0m tests/a.rs (target/x/a-1)",
            "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.50s",
            "     Running unittests src/lib.rs (target/x/lib-1)",
            "test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 100.00s",
            "     Running tests\\a.rs (target\\x\\a-2)",
            "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 13.00s",
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.00s",
        ])
        self.assertEqual(t.binary_times(log), {"tests/a.rs": [12.5, 13.0], "unit src/lib.rs": [100.0]})

    def test_the_ledger_round_trips_and_drops_the_quick_binaries(self):
        ledger = t.parse_ledger(LEDGER)
        self.assertEqual(ledger["Test (macos-latest)"]["limit"], 120.0)
        self.assertEqual(ledger["Test (macos-latest)"]["bins"][CAL], [50.0, 42.0])
        self.assertNotIn("tests/small.rs", ledger["Test (macos-latest)"]["bins"])
        with self.assertRaises(t.BudgetError):
            t.parse_ledger("bin\tTest (x)\ttests/a.rs\t3.00\n")
        with self.assertRaises(t.BudgetError):
            t.parse_ledger("job\tTest (x)\t120\n")


if __name__ == "__main__":
    unittest.main()
