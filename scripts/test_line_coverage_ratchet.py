#!/usr/bin/env python3
"""Tests for scripts/line_coverage_ratchet.py (one rule per case)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import line_coverage_ratchet as lc  # noqa: E402

HEADER = ("Filename   Regions    Missed Regions     Cover   Functions  Missed Functions  Executed       "
          "Lines      Missed Lines     Cover    Branches   Missed Branches     Cover\n" + "-" * 80 + "\n")


def summary(rows: dict[str, tuple[int, int]]) -> str:
    out = HEADER
    tl = tm = 0
    for name, (lines, missed) in rows.items():
        tl, tm = tl + lines, tm + missed
        out += f"{name}  10  1  90.00%  5  0  100.00%  {lines}  {missed}  {lc.pct(lines, missed):.2f}%  0  0  -\n"
    out += "-" * 80 + f"\nTOTAL  100  10  90.00%  50  5  90.00%  {tl}  {tm}  {lc.pct(tl, tm):.2f}%  0  0  -\n"
    return out


BASE = {"a.rs": (1000, 100), "b.rs": (400, 40), "tiny.rs": (20, 0)}


class Ratchet(unittest.TestCase):
    def run_with(self, now: dict[str, tuple[int, int]], base=BASE) -> tuple[int, str]:
        d = Path(tempfile.mkdtemp())
        (d / "s.txt").write_text(summary(now), encoding="utf-8")
        (d / "b.txt").write_text(lc.baseline_text(lc.parse_summary(summary(base))), encoding="utf-8")
        import io
        from contextlib import redirect_stderr, redirect_stdout
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = lc.main(["--summary", str(d / "s.txt"), "--baseline", str(d / "b.txt")])
        return code, out.getvalue() + err.getvalue()

    def test_the_parser_reads_lines_and_missed_lines(self):
        rows = lc.parse_summary(summary(BASE))
        self.assertEqual(rows["a.rs"], (1000, 100))
        self.assertEqual(rows["TOTAL"], (1420, 140))
        self.assertNotIn("Filename", rows)

    def test_unchanged_coverage_passes(self):
        self.assertEqual(self.run_with(BASE)[0], 0)

    def test_a_total_drop_fails(self):
        code, out = self.run_with({"a.rs": (1000, 140), "b.rs": (400, 40), "tiny.rs": (20, 0)})
        self.assertEqual(code, 1)
        self.assertIn("total line coverage", out)

    def test_a_drop_within_the_slack_passes(self):
        # 1420 lines: one more missed line is 0.07 pt
        self.assertEqual(self.run_with({"a.rs": (1000, 101), "b.rs": (400, 40), "tiny.rs": (20, 0)})[0], 0)

    def test_one_file_falling_is_not_hidden_by_another_rising(self):
        # b.rs 90% -> 80% while a.rs rises enough that the total does not fall
        code, out = self.run_with({"a.rs": (1000, 50), "b.rs": (400, 80), "tiny.rs": (20, 0)})
        self.assertEqual(code, 1)
        self.assertIn("b.rs: line coverage 80.00%", out)
        self.assertNotIn("total line coverage", out.split("error")[-1] if "error" in out else "")

    def test_small_files_are_not_held_to_the_file_slack(self):
        code, out = self.run_with({"a.rs": (1000, 99), "b.rs": (400, 39), "tiny.rs": (20, 2)})
        self.assertEqual(code, 0, out)
        self.assertNotIn("tiny.rs: line coverage", out)

    def test_a_new_file_is_a_note(self):
        code, out = self.run_with({**BASE, "new.rs": (300, 300)}, base={**BASE})
        self.assertIn("new file new.rs", out)

    def test_nothing_compared_fails(self):
        code, out = self.run_with({"other.rs": (100, 0)})
        self.assertEqual(code, 1)
        self.assertIn("compared nothing", out)

    def test_a_summary_without_total_fails(self):
        d = Path(tempfile.mkdtemp())
        (d / "s.txt").write_text(HEADER + "a.rs  10  1  90.00%  5  0  100.00%  1000  100  90.00%  0 0 -\n", encoding="utf-8")
        (d / "b.txt").write_text(lc.baseline_text(lc.parse_summary(summary(BASE))), encoding="utf-8")
        self.assertEqual(lc.main(["--summary", str(d / "s.txt"), "--baseline", str(d / "b.txt")]), 1)

    def test_write_then_check_round_trips(self):
        d = Path(tempfile.mkdtemp())
        (d / "s.txt").write_text(summary(BASE), encoding="utf-8")
        self.assertEqual(lc.main(["--summary", str(d / "s.txt"), "--baseline", str(d / "b.txt"), "--write"]), 0)
        self.assertEqual(lc.main(["--summary", str(d / "s.txt"), "--baseline", str(d / "b.txt")]), 0)

    def test_the_real_baseline_parses_and_has_a_total(self):
        rows = lc.parse_baseline(lc.BASELINE.read_text(encoding="utf-8"))
        self.assertIn("TOTAL", rows)
        self.assertGreater(len(rows), 100)


if __name__ == "__main__":
    unittest.main()
