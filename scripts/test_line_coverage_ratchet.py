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



def run(args: list[str]) -> tuple[int, str]:
    import io
    from contextlib import redirect_stderr, redirect_stdout
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = lc.main(args)
    return code, out.getvalue() + err.getvalue()


class Write(unittest.TestCase):
    """`--write` keeps the better value of every entry; lowering one takes a reason."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.b = self.d / "b.txt"
        self.b.write_text(lc.baseline_text(lc.parse_summary(summary(BASE))), encoding="utf-8")

    def write(self, now, *extra):
        (self.d / "s.txt").write_text(summary(now), encoding="utf-8")
        return run(["--summary", str(self.d / "s.txt"), "--baseline", str(self.b), "--write", *extra])

    def rows(self):
        return lc.parse_baseline(self.b.read_text(encoding="utf-8"))

    def test_a_lower_measurement_keeps_the_floor(self):
        code, out = self.write({"a.rs": (1000, 110), "b.rs": (400, 40), "tiny.rs": (20, 0)})
        self.assertEqual(code, 0, out)
        self.assertEqual(self.rows()["a.rs"], (1000, 100))
        self.assertIn("kept the baseline of a.rs", out)

    def test_a_drop_inside_the_check_slack_still_does_not_lower_the_write(self):
        # a.rs 90.0 % -> 89.9 % and the total -0.07 pt: the check passes (slack
        # 2 pt per file, 0.10 pt in total), the write still keeps 90.0 %
        now = {"a.rs": (1000, 101), "b.rs": (400, 40), "tiny.rs": (20, 0)}
        self.assertEqual(Ratchet.run_with(self, now)[0], 0)
        self.write(now)
        self.assertEqual(self.rows()["a.rs"], (1000, 100))

    def test_a_rise_is_recorded(self):
        self.write({"a.rs": (1000, 50), "b.rs": (400, 40), "tiny.rs": (20, 0)})
        self.assertEqual(self.rows()["a.rs"], (1000, 50))

    def test_a_new_file_is_recorded(self):
        self.write({**BASE, "new.rs": (300, 90)})
        self.assertEqual(self.rows()["new.rs"], (300, 90))

    def test_the_total_never_falls(self):
        # b.rs rises, a.rs falls more: the measured total is lower than the baseline's
        self.write({"a.rs": (1000, 200), "b.rs": (400, 0), "tiny.rs": (20, 0)})
        self.assertGreaterEqual(lc.pct(*self.rows()["TOTAL"]), lc.pct(*lc.parse_summary(summary(BASE))["TOTAL"]))

    def test_an_allowed_lowering_is_recorded_with_its_reason(self):
        reason = "the slow path moved to the release-only test"
        code, out = self.write({"a.rs": (1000, 300), "b.rs": (400, 40), "tiny.rs": (20, 0)},
                               "--allow-worse", "a.rs", "--reason", reason)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.rows()["a.rs"], (1000, 300))
        lines = self.b.read_text(encoding="utf-8").splitlines()
        i = lines.index("a.rs 1000 300")
        self.assertEqual(lines[i - 1], f"# allow-worse a.rs: {reason}")

    def test_the_allowance_stays_while_the_entry_keeps_that_value_and_goes_when_it_rises(self):
        reason = "the slow path moved to the release-only test"
        low = {"a.rs": (1000, 300), "b.rs": (400, 40), "tiny.rs": (20, 0)}
        self.write(low, "--allow-worse", "a.rs", "--reason", reason)
        self.write(low)
        self.assertIn(f"# allow-worse a.rs: {reason}", self.b.read_text(encoding="utf-8"))
        self.write({"a.rs": (1000, 100), "b.rs": (400, 40), "tiny.rs": (20, 0)})
        self.assertNotIn("allow-worse", self.b.read_text(encoding="utf-8"))

    def test_a_short_or_missing_reason_is_refused(self):
        low = {"a.rs": (1000, 300), "b.rs": (400, 40), "tiny.rs": (20, 0)}
        before = self.b.read_text(encoding="utf-8")
        self.assertEqual(self.write(low, "--allow-worse", "a.rs", "--reason", "too short")[0], 1)
        self.assertEqual(self.write(low, "--allow-worse", "a.rs")[0], 1)
        self.assertEqual(self.b.read_text(encoding="utf-8"), before)

    def test_an_allowance_for_an_entry_that_did_not_fall_or_does_not_exist_is_refused(self):
        reason = "a reason long enough to pass the length rule"
        self.assertEqual(self.write(BASE, "--allow-worse", "a.rs", "--reason", reason)[0], 1)
        self.assertEqual(self.write(BASE, "--allow-worse", "nope.rs", "--reason", reason)[0], 1)

    def test_an_allowance_without_write_is_refused(self):
        (self.d / "s.txt").write_text(summary(BASE), encoding="utf-8")
        code, _ = run(["--summary", str(self.d / "s.txt"), "--baseline", str(self.b),
                       "--allow-worse", "a.rs", "--reason", "a reason long enough to pass"])
        self.assertEqual(code, 1)


class DiffAgainst(unittest.TestCase):
    """A baseline edited by hand may not fall below the committed one without a reason."""

    def setUp(self):
        import subprocess
        self.d = Path(tempfile.mkdtemp())
        self.git = lambda *a: subprocess.run(["git", "-C", str(self.d), *a], check=True, capture_output=True)
        self.git("init", "-q")
        self.b = self.d / "b.txt"
        self.b.write_text(lc.baseline_text(lc.parse_summary(summary(BASE))), encoding="utf-8")
        self.git("add", "b.txt")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")

    def check(self, rows, reasons=None):
        self.b.write_text(lc.baseline_text(lc.parse_summary(summary(rows)), reasons), encoding="utf-8")
        return run(["--baseline", str(self.b), "--diff-against", "HEAD"])

    def test_an_unchanged_baseline_passes(self):
        self.assertEqual(self.check(BASE)[0], 0)

    def test_a_hand_lowered_entry_fails(self):
        code, out = self.check({"a.rs": (1000, 300), "b.rs": (400, 40), "tiny.rs": (20, 0)})
        self.assertEqual(code, 1)
        self.assertIn("a.rs: worse than on main", out)

    def test_a_hand_lowered_entry_with_a_reason_passes(self):
        rows = {"a.rs": (1000, 300), "b.rs": (400, 40), "tiny.rs": (20, 0)}
        self.assertEqual(self.check(rows, {"a.rs": "the slow path moved to the release-only test",
                                           "TOTAL": "follows the a.rs allowance above"})[0], 0)

    def test_a_short_reason_does_not_count(self):
        rows = {"a.rs": (1000, 300), "b.rs": (400, 40), "tiny.rs": (20, 0)}
        self.assertEqual(self.check(rows, {"a.rs": "short", "TOTAL": "short"})[0], 1)

    def test_rises_and_new_files_pass(self):
        self.assertEqual(self.check({"a.rs": (1000, 10), "b.rs": (400, 40), "tiny.rs": (20, 0), "n.rs": (5, 5)})[0], 0)

    def test_a_baseline_sharing_no_entry_with_the_ref_fails(self):
        self.b.write_text("x.rs 10 0\n", encoding="utf-8")
        code, out = run(["--baseline", str(self.b), "--diff-against", "HEAD"])
        self.assertEqual(code, 1)
        self.assertIn("compared nothing", out)

    def test_a_baseline_missing_at_the_ref_fails(self):
        self.assertEqual(run(["--baseline", str(self.b), "--diff-against", "HEAD~5"])[0], 1)

if __name__ == "__main__":
    unittest.main()
