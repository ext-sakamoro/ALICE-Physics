#!/usr/bin/env python3
"""Tests for scripts/mutants_baseline_margin_check.py.

scripts/testdata/mutants-baseline-line-colored.txt is the real "Unmutated
baseline in ... build + ... test" line as cargo-mutants wrote it with
CARGO_TERM_COLOR=always, captured verbatim from a real CI run (quality-deep
run 38036840295, job "Mutation testing (cargo-mutants, in-diff, default)",
2026-10-10): baseline test phase 274s. mutants-baseline-line-plain.txt is the
same content with the ANSI escapes removed by hand, for the no-colour case.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_baseline_margin_check as mc  # noqa: E402

COLORED = (HERE / "testdata" / "mutants-baseline-line-colored.txt").read_text(encoding="utf-8")
PLAIN = (HERE / "testdata" / "mutants-baseline-line-plain.txt").read_text(encoding="utf-8")


class RealFixtures(unittest.TestCase):
    def test_colored_fixture_is_actually_coloured(self):
        # guards the fixture itself: if someone "cleans up" the file by hand
        # and strips the escapes, every other test here would pass for the
        # wrong reason (no colour left to fail to parse)
        self.assertIn("\x1b[", COLORED)

    def test_colored_line_under_a_small_budget_warns(self):
        msg = mc.check(COLORED, budget=300, fraction=0.6)
        self.assertIsNotNone(msg)
        self.assertIn("274s", msg)
        self.assertIn("300s", msg)

    def test_colored_line_under_a_large_budget_is_silent(self):
        # 274s is still under 60% of 1325s: not a warning, this is the
        # legitimate "plenty of margin" case, not the bug
        self.assertIsNone(mc.check(COLORED, budget=1325, fraction=0.6))

    def test_plain_line_matches_the_colored_line_exactly(self):
        self.assertEqual(
            mc.check(COLORED, budget=300, fraction=0.6),
            mc.check(PLAIN, budget=300, fraction=0.6),
        )


class ParsingWithoutTheAnsiStripIsBroken(unittest.TestCase):
    """The regression this fix exists for: parsing the raw (unstripped) log
    text finds no match at all, so a bug that reintroduces that -- skipping
    ansi.strip before scanning -- must read as "nothing to check" here, the
    same silent failure mode measured on the real run."""

    def test_the_colored_fixture_does_not_match_without_stripping_first(self):
        with self.assertRaises(ValueError):
            # feeding the raw text straight to find_baseline_test_seconds,
            # bypassing check()'s own ansi.strip, is exactly what the old
            # one-off sed over the raw bytes effectively did: "Unmutated
            # baseline in" is still a substring (colour codes sit inside the
            # numbers, not the label), so the line is found but the digit
            # groups are broken up by escape sequences and the regex can't
            # match them -- unparsable, not merely different
            mc.find_baseline_test_seconds(COLORED)


class DegenerateInputs(unittest.TestCase):
    def test_no_baseline_line_at_all_is_silent_not_an_error(self):
        self.assertIsNone(mc.check("no such line here\n", budget=300, fraction=0.6))

    def test_a_baseline_line_present_but_unparsable_raises_not_silently_skips(self):
        with self.assertRaises(ValueError):
            mc.check("ok Unmutated baseline in a while\n", budget=300, fraction=0.6)

    def test_exactly_at_the_threshold_is_silent(self):
        # 180s is exactly 60% of 300s: "at" the line, not "over" it
        text = "ok Unmutated baseline in 10s build + 180s test\n"
        self.assertIsNone(mc.check(text, budget=300, fraction=0.6))

    def test_one_second_over_the_threshold_warns(self):
        text = "ok Unmutated baseline in 10s build + 181s test\n"
        self.assertIsNotNone(mc.check(text, budget=300, fraction=0.6))


class Cli(unittest.TestCase):
    def test_main_prints_a_warning_annotation_for_the_colored_fixture(self):
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = mc.main([
                "--log", str(HERE / "testdata" / "mutants-baseline-line-colored.txt"),
                "--budget", "300",
            ])
        self.assertEqual(rc, 0)
        self.assertIn("::warning::", buf.getvalue())

    def test_main_prints_a_notice_for_an_unparsable_line_not_silence(self):
        import io
        import contextlib
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.txt"
            log.write_text("ok Unmutated baseline in a while\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = mc.main(["--log", str(log), "--budget", "300"])
        self.assertEqual(rc, 0)
        self.assertIn("::notice::", buf.getvalue())

    def test_main_prints_nothing_when_well_under_budget(self):
        import io
        import contextlib
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "log.txt"
            log.write_text("ok Unmutated baseline in 1s build + 5s test\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = mc.main(["--log", str(log), "--budget", "300"])
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
