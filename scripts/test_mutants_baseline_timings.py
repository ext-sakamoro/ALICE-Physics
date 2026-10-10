#!/usr/bin/env python3
"""Tests for scripts/mutants_baseline_timings.py (the --check staleness gate;
--regenerate needs a real nextest run and is exercised by hand / in CI)."""

from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_baseline_timings as bt  # noqa: E402


class TableRoundTrip(unittest.TestCase):
    def test_write_then_read_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            table_path = Path(d) / "table.txt"
            with unittest.mock.patch.object(bt, "TABLE", table_path):
                bt.write_table({"lib": 10.0, "analytic_math_ln": 0.123})
                got = bt.read_table()
        self.assertEqual(got, {"lib": 10.0, "analytic_math_ln": 0.123})

    def test_missing_table_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as d:
            with unittest.mock.patch.object(bt, "TABLE", Path(d) / "none.txt"):
                self.assertEqual(bt.read_table(), {})


class CurrentTargets(unittest.TestCase):
    def test_lib_is_always_included(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests").mkdir()
            with unittest.mock.patch.object(bt, "ROOT", root):
                self.assertEqual(bt.current_targets(), {"lib"})

    def test_every_tests_rs_file_stem_is_a_target(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "tests").mkdir()
            (root / "tests" / "analytic_math_ln.rs").write_text("")
            (root / "tests" / "analytic_euler_fv.rs").write_text("")
            with unittest.mock.patch.object(bt, "ROOT", root):
                self.assertEqual(
                    bt.current_targets(),
                    {"lib", "analytic_math_ln", "analytic_euler_fv"},
                )


class Check(unittest.TestCase):
    def _repo(self, d: Path, targets: set[str]) -> None:
        (d / "tests").mkdir()
        for t in targets - {"lib"}:
            (d / "tests" / f"{t}.rs").write_text("")

    def test_a_target_missing_from_the_table_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._repo(root, {"lib", "analytic_math_ln", "analytic_euler_fv"})
            table_path = root / "scripts" / "mutants-baseline-timings.txt"
            table_path.parent.mkdir()
            table_path.write_text("lib 10.0\nanalytic_math_ln 0.1\n")  # euler_fv missing
            with unittest.mock.patch.object(bt, "ROOT", root), \
                 unittest.mock.patch.object(bt, "TABLE", table_path):
                self.assertEqual(bt.check(), 1)

    def test_every_target_present_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._repo(root, {"lib", "analytic_math_ln", "analytic_euler_fv"})
            table_path = root / "scripts" / "mutants-baseline-timings.txt"
            table_path.parent.mkdir()
            table_path.write_text("lib 10.0\nanalytic_math_ln 0.1\nanalytic_euler_fv 22.0\n")
            with unittest.mock.patch.object(bt, "ROOT", root), \
                 unittest.mock.patch.object(bt, "TABLE", table_path):
                self.assertEqual(bt.check(), 0)

    def test_an_empty_table_fails_even_with_no_tests(self):
        # a table that is empty/missing must not read as "nothing to check, pass"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._repo(root, {"lib"})
            table_path = root / "scripts" / "mutants-baseline-timings.txt"
            with unittest.mock.patch.object(bt, "ROOT", root), \
                 unittest.mock.patch.object(bt, "TABLE", table_path):
                self.assertEqual(bt.check(), 1)

    def test_an_extra_table_entry_for_a_removed_test_does_not_fail(self):
        # a removed test leaving a stale table row is harmless (no silent
        # under-budgeting); --prune is the opt-in cleanup, --check doesn't require it
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._repo(root, {"lib", "analytic_math_ln"})
            table_path = root / "scripts" / "mutants-baseline-timings.txt"
            table_path.parent.mkdir()
            table_path.write_text("lib 10.0\nanalytic_math_ln 0.1\nanalytic_removed_test 5.0\n")
            with unittest.mock.patch.object(bt, "ROOT", root), \
                 unittest.mock.patch.object(bt, "TABLE", table_path):
                self.assertEqual(bt.check(), 0)


if __name__ == "__main__":
    unittest.main()
