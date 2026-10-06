#!/usr/bin/env python3
"""Tests for scripts/mutants_ratchet.py (one rule per case)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_ratchet as mr  # noqa: E402

A = "src/math.rs:45:9: replace Fix128::add -> Fix128 with Default::default()"
B = "src/solver.rs:120:13: replace < with <= in step"
C = "src/bvh.rs:7:1: replace refit with ()"


def run_dir(root: Path, name: str, **outcomes: list[str]) -> Path:
    d = root / name / "mutants.out"
    d.mkdir(parents=True)
    for k, lines in outcomes.items():
        (d / f"{k}.txt").write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")
    return root / name


class Ratchet(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def check(self, dirs: list[Path], baseline: list[str]) -> tuple[int, list[str]]:
        b = self.root / "baseline.txt"
        b.write_text("# x\n" + "".join(f"{x}\n" for x in baseline), encoding="utf-8")
        errors, _ = mr.compare(mr.read_run(dirs), mr.read_baseline(b))
        return (1 if errors else 0), errors

    def test_line_numbers_do_not_matter(self):
        self.assertEqual(mr.normalise(A), "src/math.rs: replace Fix128::add -> Fix128 with Default::default()")
        d = run_dir(self.root, "mutants-out-0-default", missed=[A.replace(":45:9:", ":80:3:")], caught=[B])
        self.assertEqual(self.check([d], ["[default] " + mr.normalise(A)])[0], 0)

    def test_a_newly_missed_mutant_fails(self):
        d = run_dir(self.root, "mutants-out-0-default", missed=[A, B])
        code, errors = self.check([d], ["[default] " + mr.normalise(A)])
        self.assertEqual(code, 1)
        self.assertTrue(any("newly missed" in e and "solver.rs" in e for e in errors), errors)

    def test_a_baseline_entry_now_caught_fails(self):
        d = run_dir(self.root, "mutants-out-0-default", caught=[A, B])
        code, errors = self.check([d], ["[default] " + mr.normalise(A)])
        self.assertEqual(code, 1)
        self.assertTrue(any("now caught" in e for e in errors), errors)

    def test_an_untested_baseline_entry_is_left_alone(self):
        # its shard did not run (timeout / cancelled): neither fixed nor missed
        d = run_dir(self.root, "mutants-out-0-default", caught=[B])
        self.assertEqual(self.check([d], ["[default] " + mr.normalise(C)])[0], 0)

    def test_the_feature_axis_is_part_of_the_key(self):
        d1 = run_dir(self.root, "mutants-out-0-default", caught=[A])
        d2 = run_dir(self.root, "mutants-out-0-parallel", missed=[A])
        code, errors = self.check([d1, d2], ["[parallel] " + mr.normalise(A)])
        self.assertEqual(code, 0, errors)

    def test_repeated_identical_mutants_are_counted(self):
        d = run_dir(self.root, "mutants-out-0-default", missed=[A, A.replace(":45:9:", ":46:9:")])
        code, errors = self.check([d], ["[default] " + mr.normalise(A)])
        self.assertEqual(code, 1)
        self.assertTrue(any("newly missed" in e for e in errors), errors)

    def test_nothing_tested_fails(self):
        d = run_dir(self.root, "mutants-out-0-default")
        code, errors = self.check([d], [])
        self.assertEqual(code, 1)
        self.assertIn("compared nothing", errors[0])

    def test_write_then_check_round_trips(self):
        d = run_dir(self.root, "mutants-out-0-default", missed=[A, B], caught=[C])
        b = self.root / "b.txt"
        self.assertEqual(mr.main(["--out", str(d), "--baseline", str(b), "--write"]), 0)
        self.assertEqual(mr.main(["--out", str(d), "--baseline", str(b)]), 0)


if __name__ == "__main__":
    unittest.main()
