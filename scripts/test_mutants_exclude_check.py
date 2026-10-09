#!/usr/bin/env python3
"""Tests for scripts/mutants_exclude_check.py, against a fixture list of
mutant description lines (not a real `cargo mutants --list`, which this
file does not invoke: `full_mutant_list` is monkeypatched everywhere a
test needs its own fixed lines)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_exclude_check as mec  # noqa: E402

FIXTURE_LINES = [
    "src/a.rs:10:5: replace < with <= in f",
    "src/a.rs:10:5: replace < with == in f",
    "src/a.rs:20:5: replace > with >= in g",
    "src/a.rs:20:5: replace > with >= in g",  # same site twice is legitimate (two call sites)
]


class MatchCounts(unittest.TestCase):
    def test_a_pattern_present_once_counts_one(self):
        counts = mec.match_counts(["replace < with <= in f"], FIXTURE_LINES)
        self.assertEqual(counts["replace < with <= in f"], 1)

    def test_a_pattern_present_twice_counts_two(self):
        counts = mec.match_counts(["replace > with >= in g"], FIXTURE_LINES)
        self.assertEqual(counts["replace > with >= in g"], 2)

    def test_a_pattern_present_nowhere_counts_zero(self):
        counts = mec.match_counts(["replace \\+ with - in h"], FIXTURE_LINES)
        self.assertEqual(counts["replace \\+ with - in h"], 0)

    def test_a_line_anchored_pattern_only_matches_its_own_line(self):
        counts = mec.match_counts(["a\\.rs:10:.*replace < with <= in f"], FIXTURE_LINES)
        self.assertEqual(counts["a\\.rs:10:.*replace < with <= in f"], 1)
        counts = mec.match_counts(["a\\.rs:99:.*replace < with <= in f"], FIXTURE_LINES)
        self.assertEqual(counts["a\\.rs:99:.*replace < with <= in f"], 0)


class StripAnsi(unittest.TestCase):
    def test_a_line_with_no_escapes_is_unchanged(self):
        line = "src/a.rs:10:5: replace < with <= in f"
        self.assertEqual(mec.strip_ansi(line), line)

    def test_an_escape_sequence_wrapping_an_identifier_is_removed(self):
        # The exact shape CARGO_TERM_COLOR=always produces (2026-10-09 CI
        # incident): the identifier itself is wrapped, splitting it out of
        # any pattern that names it as a contiguous substring.
        colored = "src/joint.rs:1032:5: replace \x1b[38;5;13msolve_slider_joint\x1b[0m with ()"
        self.assertEqual(
            mec.strip_ansi(colored),
            "src/joint.rs:1032:5: replace solve_slider_joint with ()",
        )

    def test_multiple_escapes_on_one_line_are_all_removed(self):
        colored = "replace \x1b[33m+\x1b[0m with \x1b[38;5;11m-\x1b[0m in \x1b[38;5;13mf\x1b[0m"
        self.assertEqual(mec.strip_ansi(colored), "replace + with - in f")

    def test_a_pattern_naming_a_function_only_matches_once_stripped(self):
        # The exact line `cargo mutants --list --no-config` produced for
        # src/joint.rs:1032 under CARGO_TERM_COLOR=always (captured
        # 2026-10-09 while diagnosing the CI incident): without
        # strip_ansi, a pattern naming the function matches 0 lines even
        # though the line is present -- which is how 37 of 38 exclude_re
        # entries went to 0 matches in CI while the total line count
        # (`compared: N mutants`) stayed unchanged.
        colored_lines = [
            "src/joint.rs:1032:5: replace \x1b[38;5;13msolve_slider_joint\x1b[0m with ()"
        ]
        pattern = "replace solve_slider_joint with \\(\\)"
        raw_counts = mec.match_counts([pattern], colored_lines)
        self.assertEqual(raw_counts[pattern], 0)
        stripped_lines = [mec.strip_ansi(line) for line in colored_lines]
        stripped_counts = mec.match_counts([pattern], stripped_lines)
        self.assertEqual(stripped_counts[pattern], 1)


class Main(unittest.TestCase):
    def run_main(self, mutants_toml_text: str, baseline_text: str | None, lines=FIXTURE_LINES):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mutants_toml = tmp_path / "mutants.toml"
            mutants_toml.write_text(mutants_toml_text, encoding="utf-8")
            baseline = tmp_path / "baseline.txt"
            if baseline_text is not None:
                baseline.write_text(baseline_text, encoding="utf-8")
            with patch.object(mec, "full_mutant_list", return_value=lines):
                rc = mec.main(
                    ["--mutants-toml", str(mutants_toml), "--baseline", str(baseline)]
                )
            return rc, baseline

    def test_empty_exclude_re_is_an_error(self):
        rc, _ = self.run_main("exclude_re = []\n", baseline_text="")
        self.assertEqual(rc, 1)

    def test_a_pattern_matching_zero_is_an_error_even_if_in_baseline(self):
        rc, _ = self.run_main(
            'exclude_re = ["no such pattern anywhere"]\n',
            baseline_text="5\tno such pattern anywhere\n",
        )
        self.assertEqual(rc, 1)

    def test_a_pattern_not_yet_in_baseline_is_an_error(self):
        rc, _ = self.run_main(
            'exclude_re = ["replace < with <= in f"]\n',
            baseline_text="",
        )
        self.assertEqual(rc, 1)

    def test_a_pattern_whose_count_changed_is_an_error(self):
        rc, _ = self.run_main(
            'exclude_re = ["replace > with >= in g"]\n',
            baseline_text="1\treplace > with >= in g\n",  # fixture actually has 2
        )
        self.assertEqual(rc, 1)

    def test_a_pattern_matching_its_tracked_count_passes(self):
        rc, _ = self.run_main(
            'exclude_re = ["replace > with >= in g"]\n',
            baseline_text="2\treplace > with >= in g\n",
        )
        self.assertEqual(rc, 0)

    def test_a_loose_pattern_like_tests_tolerates_a_changed_count(self):
        # baseline's tracked count (99) deliberately does not match how many
        # of FIXTURE_LINES actually say "tests::" below -- a real PR that
        # only adds or removes a #[cfg(test)] helper must not go red here.
        rc, _ = self.run_main(
            'exclude_re = ["tests::"]\n',
            baseline_text="99\ttests::\n",
            lines=[
                "src/a.rs:1:1: replace x in tests::helper_one",
                "src/a.rs:2:1: replace y in tests::helper_two",
                "src/a.rs:3:1: replace z in tests::helper_three",
            ],
        )
        self.assertEqual(rc, 0)

    def test_a_loose_pattern_like_tests_still_errors_on_zero_matches(self):
        rc, _ = self.run_main(
            'exclude_re = ["tests::"]\n',
            baseline_text="3\ttests::\n",
            lines=["src/a.rs:1:1: replace x in production_fn"],
        )
        self.assertEqual(rc, 1)

    def test_a_stale_baseline_entry_no_longer_in_exclude_re_is_an_error(self):
        rc, _ = self.run_main(
            'exclude_re = ["replace > with >= in g"]\n',
            baseline_text="2\treplace > with >= in g\n1\tsome removed entry\n",
        )
        self.assertEqual(rc, 1)

    def test_write_records_the_current_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mutants_toml = tmp_path / "mutants.toml"
            mutants_toml.write_text(
                'exclude_re = ["replace > with >= in g", "replace < with <= in f"]\n',
                encoding="utf-8",
            )
            baseline = tmp_path / "baseline.txt"
            with patch.object(mec, "full_mutant_list", return_value=FIXTURE_LINES):
                rc = mec.main(
                    [
                        "--mutants-toml",
                        str(mutants_toml),
                        "--baseline",
                        str(baseline),
                        "--write",
                    ]
                )
            self.assertEqual(rc, 0)
            written = mec.read_baseline(baseline)
            self.assertEqual(written["replace > with >= in g"], 2)
            self.assertEqual(written["replace < with <= in f"], 1)


if __name__ == "__main__":
    unittest.main()
