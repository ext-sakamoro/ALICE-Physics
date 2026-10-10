#!/usr/bin/env python3
"""Tests for scripts/mutants_baseline_budget.py."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_baseline_budget as bb  # noqa: E402

TABLE = {"lib": 10.0, "analytic_cubic_hyperelastic": 121.0,
         "analytic_hyperelastic_mms_order": 105.0, "analytic_euler_fv": 22.0,
         "analytic_cloth_rest_tether": 20.0, "analytic_math_ln": 0.1}


class BudgetSeconds(unittest.TestCase):
    def test_sum_of_known_targets_times_margin(self):
        got = bb.budget_seconds(["lib", "analytic_math_ln"], TABLE, margin=1.5, hard_cap=2700)
        self.assertAlmostEqual(got, (10.0 + 0.1) * 1.5, places=3)

    def test_an_unknown_target_uses_the_slowest_known_time_not_zero(self):
        got = bb.budget_seconds(["lib", "a_brand_new_test"], TABLE, margin=1.0, hard_cap=2700)
        self.assertAlmostEqual(got, 10.0 + max(TABLE.values()), places=3)

    def test_concentration_of_slow_targets_is_additive_not_maxed(self):
        # the real defect this script exists for: several slow targets at once
        # sum (not the max of the four) is what a non-overlapping worst case costs
        slow = ["lib", "analytic_cubic_hyperelastic", "analytic_hyperelastic_mms_order",
                "analytic_euler_fv", "analytic_cloth_rest_tether"]
        got = bb.budget_seconds(slow, TABLE, margin=1.0, hard_cap=2700)
        want = 10.0 + 121.0 + 105.0 + 22.0 + 20.0
        self.assertAlmostEqual(got, want, places=3)
        self.assertGreater(got, max(TABLE[t] for t in slow))  # not just the slowest one

    def test_hard_cap_is_enforced(self):
        slow = ["lib"] + [t for t in TABLE if t != "lib"]
        got = bb.budget_seconds(slow, TABLE, margin=10.0, hard_cap=50.0)
        self.assertEqual(got, 50.0)

    def test_no_targets_is_an_error(self):
        with self.assertRaises(ValueError):
            bb.budget_seconds([], TABLE, margin=1.0, hard_cap=2700)

    def test_empty_table_is_an_error(self):
        with self.assertRaises(ValueError):
            bb.budget_seconds(["lib"], {}, margin=1.0, hard_cap=2700)


class Cli(unittest.TestCase):
    def _table_file(self, d: Path) -> Path:
        p = d / "table.txt"
        p.write_text("# header\n" + "\n".join(f"{k} {v}" for k, v in TABLE.items()) + "\n")
        return p

    def test_prints_an_integer_to_stdout(self):
        with tempfile.TemporaryDirectory() as d:
            table = self._table_file(Path(d))
            out = subprocess.run(
                [sys.executable, str(HERE / "mutants_baseline_budget.py"),
                 "--targets", "analytic_math_ln", "--table", str(table), "--margin", "1.0",
                 "--min", "0"],
                capture_output=True, text=True, check=True,
            )
        self.assertEqual(out.stdout.strip(), str(int(round(10.0 + 0.1))))

    def test_comma_separated_targets_are_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            table = self._table_file(Path(d))
            out = subprocess.run(
                [sys.executable, str(HERE / "mutants_baseline_budget.py"),
                 "--targets", "analytic_math_ln,analytic_euler_fv", "--table", str(table),
                 "--margin", "1.0", "--min", "0"],
                capture_output=True, text=True, check=True,
            )
        self.assertEqual(out.stdout.strip(), str(int(round(10.0 + 0.1 + 22.0))))

    def test_targets_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            table = self._table_file(Path(d))
            tf = Path(d) / "targets.txt"
            tf.write_text("analytic_math_ln\nanalytic_euler_fv\n")
            out = subprocess.run(
                [sys.executable, str(HERE / "mutants_baseline_budget.py"),
                 "--targets-file", str(tf), "--table", str(table), "--margin", "1.0",
                 "--min", "0"],
                capture_output=True, text=True, check=True,
            )
        self.assertEqual(out.stdout.strip(), str(int(round(10.0 + 0.1 + 22.0))))

    def test_missing_table_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run(
                [sys.executable, str(HERE / "mutants_baseline_budget.py"),
                 "--targets", "analytic_math_ln", "--table", str(Path(d) / "none.txt")],
                capture_output=True, text=True,
            )
        self.assertNotEqual(out.returncode, 0)

    def test_floor_applies_when_the_sum_is_tiny(self):
        with tempfile.TemporaryDirectory() as d:
            table = Path(d) / "table.txt"
            table.write_text("lib 0.01\nanalytic_math_ln 0.1\n")
            out = subprocess.run(
                [sys.executable, str(HERE / "mutants_baseline_budget.py"),
                 "--targets", "analytic_math_ln", "--table", str(table), "--margin", "1.0"],
                capture_output=True, text=True, check=True,
            )
        self.assertEqual(out.stdout.strip(), "60")  # default --min floor


if __name__ == "__main__":
    unittest.main()
