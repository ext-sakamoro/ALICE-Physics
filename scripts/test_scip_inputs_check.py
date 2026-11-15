#!/usr/bin/env python3
"""Tests for scripts/scip_inputs_check.py: a reader's file that no pattern matches, and a reader that read
nothing, fail; caches are not inputs; the tracer records the files a script opens."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scip_inputs_check as sc  # noqa: E402

PATS = ["**/*.rs", "scripts/integration_levels.py", "docs/MODULES.md"]


class Problems(unittest.TestCase):
    def test_covered_reads_pass(self):
        self.assertEqual(sc.problems({"r": ["src/lib.rs", "docs/MODULES.md"]}, PATS), [])

    def test_an_unmatched_read_fails_and_names_the_file(self):
        errs = sc.problems({"r": ["src/lib.rs", "scripts/abi-consumer-gaps.txt"]}, PATS)
        self.assertEqual(len(errs), 1)
        self.assertIn("scripts/abi-consumer-gaps.txt", errs[0])

    def test_a_reader_that_read_nothing_fails(self):
        self.assertTrue(any("opened no file" in e for e in sc.problems({"r": []}, PATS)))

    def test_caches_only_count_as_nothing(self):
        errs = sc.problems({"r": ["scripts/__pycache__/x.cpython-314.pyc", ".git/HEAD"]}, PATS)
        self.assertTrue(any("opened no file" in e for e in errs), errs)

    def test_no_pattern_fails(self):
        self.assertEqual(sc.problems({"r": ["src/lib.rs"]}, []), ["the patterns file holds no pattern"])


class Trace(unittest.TestCase):
    def test_the_tracer_records_what_a_script_opens(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data.txt").write_text("x\n", encoding="utf-8")
            (root / "reader.py").write_text("open('data.txt').read()\n", encoding="utf-8")
            t = sc.trace(root, ["reader.py"])
            self.assertEqual(t["exit"], 0)
            self.assertIn("data.txt", t["read"])

    def test_a_failing_script_still_reports(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "reader.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
            self.assertEqual(sc.trace(root, ["reader.py"])["exit"], 3)


if __name__ == "__main__":
    unittest.main()
