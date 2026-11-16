#!/usr/bin/env python3
"""Tests of scripts/mutants_run_status.py: a killed run, a run that measured its build deadline and a run that
compared nothing never read as a result.

Author: Moroya Sakamoto
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutants_run_status as rs  # noqa: E402


def outcome(summary: str, build: str = "Success") -> dict:
    phases = [{"phase": "Build", "duration": 1.0, "process_status": build}]
    if build == "Success":
        phases.append({"phase": "Test", "duration": 1.0,
                       "process_status": "Success" if summary == "MissedMutant" else {"Failure": 101}})
    return {"scenario": {"Mutant": {}}, "summary": summary, "phase_results": phases}


def run_dir(root: Path, planned: int, outcomes: list[dict], finished: bool = True) -> Path:
    d = root / "mutants.out"
    d.mkdir(parents=True)
    (d / "mutants.json").write_text(json.dumps([{}] * planned), encoding="utf-8")
    base = {"scenario": "Baseline", "summary": "Success", "phase_results": []}
    (d / "outcomes.json").write_text(json.dumps({"outcomes": [base, *outcomes],
                                                 "end_time": "2026-11-16T00:00:00Z" if finished else None}),
                                     encoding="utf-8")
    return root


class Status(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_complete_run(self):
        d = run_dir(self.root, 3, [outcome("CaughtMutant"), outcome("CaughtMutant"), outcome("MissedMutant")])
        code, line = rs.status(d)
        self.assertEqual(code, rs.COMPLETE)
        self.assertEqual(line, "complete (3/3 tested): caught 2 / missed 1 / unviable 0 / timeout 0 (build 0)")

    def test_a_run_without_end_time_is_killed_with_its_count(self):
        d = run_dir(self.root, 380, [outcome("CaughtMutant")] * 9 + [outcome("MissedMutant")], finished=False)
        code, line = rs.status(d)
        self.assertEqual(code, rs.KILLED)
        self.assertTrue(line.startswith("killed (10/380 tested)"), line)
        # the survivors seen before the stop are counted from the outcomes, not from missed.txt
        self.assertIn("missed 1", line)

    def test_no_end_time_is_killed_even_with_every_outcome(self):
        code, line = rs.status(run_dir(self.root, 2, [outcome("CaughtMutant")] * 2, finished=False))
        self.assertEqual(code, rs.KILLED)
        self.assertTrue(line.startswith("killed (2/2 tested)"), line)

    def test_fewer_outcomes_than_planned_is_killed_even_with_an_end_time(self):
        code, line = rs.status(run_dir(self.root, 4, [outcome("CaughtMutant")] * 3))
        self.assertEqual(code, rs.KILLED)
        self.assertTrue(line.startswith("killed (3/4 tested)"), line)

    def test_mostly_build_timeouts_is_not_measured(self):
        d = run_dir(self.root, 10, [outcome("Timeout", build="Timeout")] * 9 + [outcome("CaughtMutant")])
        code, line = rs.status(d)
        self.assertEqual(code, rs.NOT_MEASURED)
        self.assertTrue(line.startswith("not measured (9 of 10 outcomes are build timeouts, over 10%)"), line)

    def test_the_bound_is_a_share_of_the_outcomes(self):
        # 1 of 10 is at the bound: measured; 2 of 10 is over it
        one = run_dir(self.root / "a", 10, [outcome("Timeout", build="Timeout")] + [outcome("CaughtMutant")] * 9)
        self.assertEqual(rs.status(one)[0], rs.COMPLETE)
        two = run_dir(self.root / "b", 10, [outcome("Timeout", build="Timeout")] * 2 + [outcome("CaughtMutant")] * 8)
        self.assertEqual(rs.status(two)[0], rs.NOT_MEASURED)

    def test_test_phase_timeouts_are_not_build_timeouts(self):
        test_timeout = {"scenario": {"Mutant": {}}, "summary": "Timeout",
                        "phase_results": [{"phase": "Build", "process_status": "Success"},
                                          {"phase": "Test", "process_status": "Timeout"}]}
        code, line = rs.status(run_dir(self.root, 5, [test_timeout] * 5))
        self.assertEqual(code, rs.COMPLETE)
        self.assertIn("timeout 5 (build 0)", line)

    def test_killed_comes_before_not_measured(self):
        # the 2026-11-14 default axis: 137 of 380, 131 build timeouts, no end time
        d = run_dir(self.root, 380, [outcome("Timeout", build="Timeout")] * 131 + [outcome("Unviable")] * 6,
                    finished=False)
        code, line = rs.status(d)
        self.assertEqual(code, rs.KILLED)
        self.assertIn("timeout 131 (build 131)", line)

    def test_nothing_compared_is_an_error(self):
        self.assertEqual(rs.status(self.root)[0], rs.ERROR)
        self.assertEqual(rs.status(run_dir(self.root / "z", 0, []))[0], rs.ERROR)

    def test_main_writes_the_line_to_the_summary_and_returns_the_code(self):
        d = run_dir(self.root, 2, [outcome("CaughtMutant")], finished=False)
        summary = self.root / "summary.md"
        self.assertEqual(rs.main([str(d), "--summary", str(summary)]), rs.KILLED)
        self.assertTrue(summary.read_text(encoding="utf-8").startswith("- run: killed (1/2 tested)"))


if __name__ == "__main__":
    unittest.main()
