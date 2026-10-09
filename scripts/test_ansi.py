#!/usr/bin/env python3
"""Tests for scripts/ansi.py: every escape sequence the tools print is removed."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ansi  # noqa: E402

# libtest with `--color always` (captured): SGR colour, then sgr0 = ESC ( B ESC [ m
LIBTEST = "test result: \x1b[32mok\x1b(B\x1b[m. 3 passed; 0 failed"
# cargo with CARGO_TERM_COLOR=always (captured)
CARGO = "\x1b[1m\x1b[92m     Running\x1b[0m tests/audit_c_math.rs"
SEMVER = "\x1b[1m\x1b[32m     Checked\x1b[0m [   0.580s] 223 checks"
MUTANTS = "replace \x1b[38;5;13msolve_slider_joint\x1b[0m with \x1b[33m()\x1b[0m"


class Strip(unittest.TestCase):
    def test_libtest_sgr0_charset_selector_is_removed(self):
        self.assertEqual(ansi.strip(LIBTEST), "test result: ok. 3 passed; 0 failed")

    def test_cargo_semver_and_mutants_colours_are_removed(self):
        self.assertEqual(ansi.strip(CARGO), "     Running tests/audit_c_math.rs")
        self.assertEqual(ansi.strip(SEMVER), "     Checked [   0.580s] 223 checks")
        self.assertEqual(ansi.strip(MUTANTS), "replace solve_slider_joint with ()")

    def test_cursor_and_erase_sequences_are_removed(self):
        self.assertEqual(ansi.strip("a\x1b[2K\x1b[1Gb"), "ab")

    def test_osc_hyperlinks_are_removed_with_both_terminators(self):
        self.assertEqual(ansi.strip("\x1b]8;;https://x\x07link\x1b]8;;\x07"), "link")
        self.assertEqual(ansi.strip("\x1b]0;title\x1b\\done"), "done")

    def test_plain_text_is_unchanged(self):
        text = "test result: ok. 13 passed; 0 failed; 0 ignored [brackets] (parens)"
        self.assertEqual(ansi.strip(text), text)


if __name__ == "__main__":
    unittest.main()
