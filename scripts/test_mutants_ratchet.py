#!/usr/bin/env python3
"""Tests for scripts/mutants_ratchet.py (one rule per case)."""

from __future__ import annotations

import json
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


def run_dir(root: Path, name: str, *, planned: int | None = None, finished: bool = True,
            **outcomes: list[str]) -> Path:
    """A cargo-mutants output directory. `mutants.json` plans `planned` mutants
    (default: as many as have an outcome) and `outcomes.json` has an end time
    unless `finished` is false (a cancelled or timed-out run)."""
    d = root / name / "mutants.out"
    d.mkdir(parents=True)
    for k, lines in outcomes.items():
        (d / f"{k}.txt").write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")
    n = sum(len(v) for v in outcomes.values())
    (d / "mutants.json").write_text(json.dumps([{}] * (n if planned is None else planned)),
                                    encoding="utf-8")
    scenarios = [{"scenario": "Baseline"}] + [{"scenario": {"Mutant": {}}}] * n
    (d / "outcomes.json").write_text(json.dumps({
        "outcomes": scenarios,
        "end_time": "2026-10-09T00:00:00Z" if finished else None,
    }), encoding="utf-8")
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

    def test_a_run_that_did_not_finish_fails(self):
        # a shard cancelled or killed by its timeout leaves no end time
        d = run_dir(self.root, "mutants-out-0-default", finished=False, caught=[A])
        code, errors = self.complete([d], 1)
        self.assertEqual(code, 1)
        self.assertTrue(any("did not finish" in e for e in errors), errors)

    def test_a_run_with_mutants_left_untested_fails(self):
        d = run_dir(self.root, "mutants-out-0-default", planned=5, caught=[A], missed=[B])
        code, errors = self.complete([d], 1)
        self.assertEqual(code, 1)
        self.assertTrue(any("2 of 5" in e for e in errors), errors)

    def test_a_missing_output_directory_fails(self):
        d = run_dir(self.root, "mutants-out-0-default", caught=[A])
        code, errors = self.complete([d], 32)
        self.assertEqual(code, 1)
        self.assertTrue(any("1 of 32" in e for e in errors), errors)

    def test_a_directory_without_the_plan_fails(self):
        d = self.root / "mutants-out-0-default" / "mutants.out"
        d.mkdir(parents=True)
        (d / "caught.txt").write_text(A + "\n", encoding="utf-8")
        code, errors = self.complete([d.parent], 1)
        self.assertEqual(code, 1)
        self.assertTrue(any("mutants.json" in e for e in errors), errors)

    def test_complete_runs_pass(self):
        d1 = run_dir(self.root, "mutants-out-0-default", caught=[A], missed=[B])
        d2 = run_dir(self.root, "mutants-out-0-parallel", caught=[A], unviable=[C])
        self.assertEqual(self.complete([d1, d2], 2), (0, []))

    def test_unviable_mutants_are_not_tested(self):
        # only unviable outcomes: nothing was compared
        d = run_dir(self.root, "mutants-out-0-default", unviable=[A, B])
        code, errors = self.check([d], [])
        self.assertEqual(code, 1)
        self.assertIn("compared nothing", errors[0])

    def test_main_fails_on_an_incomplete_run(self):
        d = run_dir(self.root, "mutants-out-0-default", finished=False, caught=[A])
        b = self.root / "b.txt"
        b.write_text("# x\n", encoding="utf-8")
        self.assertEqual(mr.main(["--out", str(d), "--baseline", str(b)]), 1)
        self.assertEqual(mr.main(["--out", str(d), "--baseline", str(b), "--write"]), 1)
        self.assertEqual(b.read_text(encoding="utf-8"), "# x\n", "an incomplete run overwrote the baseline")

    def test_complete_only_checks_completeness_without_a_baseline(self):
        done = run_dir(self.root, "mutants-out-0-default", missed=[A])
        cut = run_dir(self.root, "mutants-out-1-default", finished=False, caught=[B])
        self.assertEqual(mr.main(["--out", str(done), "--complete-only"]), 0)
        self.assertEqual(mr.main(["--out", str(cut), "--complete-only"]), 1)

    def test_the_same_directory_twice_counts_once(self):
        d = run_dir(self.root, "mutants-out-0-default", caught=[A])
        code, errors = self.complete([d, d], 2)
        self.assertEqual(code, 1)
        self.assertTrue(any("1 of 2" in e for e in errors), errors)

    def test_an_unreadable_outcomes_file_is_reported(self):
        d = run_dir(self.root, "mutants-out-0-default", caught=[A])
        (d / "mutants.out" / "outcomes.json").write_text("", encoding="utf-8")
        code, errors = self.complete([d], 1)
        self.assertEqual(code, 1)
        self.assertTrue(any("not readable" in e for e in errors), errors)

    def moved_dir(self, name: str, moved: list[str], planned: list[str]) -> Path:
        d = run_dir(self.root, name, caught=planned)
        (d / "mutants.out" / "mutants.json").write_text(
            json.dumps([{"name": n} for n in planned]), encoding="utf-8")
        (d / "mutants.out" / "axis-moved.txt").write_text(
            "".join(f"{m}\n" for m in moved), encoding="utf-8")
        return d

    def test_mutants_moved_to_the_other_axis_are_planned_there(self):
        d = self.moved_dir("mutants-out-0-default", [A], [B])
        p = self.moved_dir("mutants-out-0-parallel", [], [A, C])
        self.assertEqual(mr.moved_check([d, p]), [])

    def test_a_moved_mutant_the_other_axis_does_not_plan_fails(self):
        # excluded on the default axis but missing from the parallel plan:
        # no axis measures it
        d = self.moved_dir("mutants-out-0-default", [A], [B])
        p = self.moved_dir("mutants-out-0-parallel", [], [C])
        errors = mr.moved_check([d, p])
        self.assertTrue(any("not planned on the parallel axis" in e and "math.rs" in e
                            for e in errors), errors)

    def test_a_missing_moved_list_fails(self):
        d = run_dir(self.root, "mutants-out-0-default", caught=[B])
        p = self.moved_dir("mutants-out-0-parallel", [], [A])
        self.assertTrue(any("axis-moved.txt" in e for e in mr.moved_check([d, p])))

    def test_moved_lists_are_compared_across_shards(self):
        # the other axis plans the moved mutant in a different shard
        d0 = self.moved_dir("mutants-out-0-default", [A, C], [B])
        p0 = self.moved_dir("mutants-out-0-parallel", [], [A])
        p1 = self.moved_dir("mutants-out-1-parallel", [], [C])
        self.assertEqual(mr.moved_check([d0, p0, p1]), [])

    def complete(self, dirs: list[Path], expect: int) -> tuple[int, list[str]]:
        errors = mr.completeness(dirs, expect)
        return (1 if errors else 0), errors

    def test_write_then_check_round_trips(self):
        d = run_dir(self.root, "mutants-out-0-default", missed=[A, B], caught=[C])
        b = self.root / "b.txt"
        self.assertEqual(mr.main(["--out", str(d), "--baseline", str(b), "--write"]), 0)
        self.assertEqual(mr.main(["--out", str(d), "--baseline", str(b)]), 0)


if __name__ == "__main__":
    unittest.main()
