#!/usr/bin/env python3
"""Tests for scripts/mutants_scoped_guard.py (one rule per case)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_scoped_guard as g  # noqa: E402

PASSED = "test result: ok. 2339 passed; 0 failed; 4 ignored; 0 measured; 0 filtered out; finished in 5.70s\n"
NONE = "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"


def out_dir(root: Path, counts: dict, baseline_log: str | None = PASSED, baseline="Success") -> Path:
    d = root / "mutants.out"
    (d / "log").mkdir(parents=True)
    outcomes = []
    if baseline_log is not None:
        (d / "log" / "baseline.log").write_text(baseline_log, encoding="utf-8")
        outcomes.append({"scenario": "Baseline", "summary": baseline, "log_path": "log/baseline.log"})
    doc = {"total_mutants": sum(counts.values()), "success": 0, **counts, "outcomes": outcomes}
    (d / "outcomes.json").write_text(json.dumps(doc), encoding="utf-8")
    return d


class Guard(unittest.TestCase):
    def check(self, counts, **kw):
        with tempfile.TemporaryDirectory() as t:
            return g.check(out_dir(Path(t), counts, **kw))

    def test_a_measurement_with_caught_mutants_passes(self):
        problems, _ = self.check({"caught": 10, "missed": 2, "timeout": 1, "unviable": 3})
        self.assertEqual(problems, [])

    def test_every_viable_mutant_missed_fails(self):
        # the 2026-10-24 scoped run: --lib tests do not reach an integration-tested scope
        problems, _ = self.check({"caught": 0, "missed": 72, "timeout": 1, "unviable": 3})
        self.assertTrue(any("no mutant was caught" in p for p in problems), problems)

    def test_a_baseline_that_ran_no_test_fails(self):
        problems, _ = self.check({"caught": 5, "missed": 0, "timeout": 0, "unviable": 0}, baseline_log=NONE)
        self.assertTrue(any("ran 0 tests" in p for p in problems), problems)

    def test_a_baseline_without_a_test_result_line_fails(self):
        problems, _ = self.check({"caught": 5, "missed": 0, "timeout": 0, "unviable": 0}, baseline_log="no tests here\n")
        self.assertTrue(any("ran 0 tests" in p for p in problems), problems)

    def test_a_red_baseline_fails(self):
        problems, _ = self.check({"caught": 5, "missed": 0, "timeout": 0, "unviable": 0}, baseline="Failure")
        self.assertTrue(any("baseline" in p for p in problems), problems)

    def test_no_baseline_outcome_fails(self):
        problems, _ = self.check({"caught": 5, "missed": 0, "timeout": 0, "unviable": 0}, baseline_log=None)
        self.assertTrue(any("baseline" in p for p in problems), problems)

    def test_nothing_viable_fails(self):
        problems, _ = self.check({"caught": 0, "missed": 0, "timeout": 0, "unviable": 4})
        self.assertTrue(any("compared nothing" in p for p in problems), problems)

    def test_missing_outcomes_fails(self):
        with tempfile.TemporaryDirectory() as t:
            problems, _ = g.check(Path(t) / "mutants.out")
        self.assertTrue(problems)

    def test_baseline_counts_add_up_over_test_binaries(self):
        two = "test result: ok. 3 passed; 0 failed\n" + "test result: ok. 4 passed; 0 failed\n"
        _, report = self.check({"caught": 1, "missed": 0, "timeout": 0, "unviable": 0}, baseline_log=two)
        self.assertIn("baseline ran 7 tests", report)

    def test_the_real_2026_10_24_counts_go_red_through_main(self):
        with tempfile.TemporaryDirectory() as t:
            d = out_dir(Path(t), {"caught": 0, "missed": 72, "timeout": 1, "unviable": 3})
            self.assertEqual(g.main([str(d)]), 1)
            d2 = Path(t) / "ok"
            d2.mkdir()
            self.assertEqual(g.main([str(out_dir(d2, {"caught": 3, "missed": 1, "timeout": 0, "unviable": 0}))]), 0)


if __name__ == "__main__":
    unittest.main()
