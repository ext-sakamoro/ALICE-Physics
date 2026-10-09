#!/usr/bin/env python3
"""Tests for scripts/downstream_test_counts.py (cargo test logs given as text)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import downstream_test_counts as tc  # noqa: E402

TWO_BINARIES = """\
   Compiling alice-sdf v5.0.0 (/x/ALICE-SDF)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 41.20s
     Running tests/test_physics_bridge_determinism.rs (target/debug/deps/test_physics_bridge_determinism-1a2b)

running 9 tests
test a ... ok
test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.20s

     Running tests/test_sim_bridge_oracle.rs (target/debug/deps/test_sim_bridge_oracle-3c4d)

running 4 tests
test result: ok. 4 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.05s

"""

# the second binary has every test compiled out (or filtered away)
ONE_EMPTY = TWO_BINARIES.replace("running 4 tests\ntest result: ok. 4 passed; 0 failed; 1 ignored",
                                 "running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored")

COLOURED = TWO_BINARIES.replace("     Running tests/test_sim", "\x1b[1m\x1b[92m     Running\x1b[0m tests/test_sim")
# libtest with `--color always`: the result word ends with sgr0 = ESC ( B ESC [ m
LIBTEST_COLOURED = TWO_BINARIES.replace("test result: ok.", "test result: \x1b[32mok\x1b(B\x1b[m.")

WITH_DOCTESTS = TWO_BINARIES + """\
   Doc-tests alice_sdf

running 0 tests
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s
"""


class Count(unittest.TestCase):
    def test_libtest_coloured_result_lines_count_like_plain_ones(self):
        self.assertNotEqual(LIBTEST_COLOURED, TWO_BINARIES)
        self.assertEqual(tc.count(LIBTEST_COLOURED), tc.count(TWO_BINARIES))

    def test_each_binary_is_counted(self):
        self.assertEqual(tc.count(TWO_BINARIES), [
            ("tests/test_physics_bridge_determinism.rs", 9), ("tests/test_sim_bridge_oracle.rs", 4)])
        self.assertEqual(tc.check(TWO_BINARIES), (13, []))

    def test_one_binary_with_zero_tests_fails_even_when_the_step_total_is_not_zero(self):
        total, errors = tc.check(ONE_EMPTY)
        self.assertEqual(total, 9)
        self.assertEqual(errors, ["tests/test_sim_bridge_oracle.rs ran 0 tests"])

    def test_binary_without_a_result_line_fails(self):
        log = TWO_BINARIES.split("running 4 tests")[0]
        self.assertEqual(tc.check(log)[1], ["tests/test_sim_bridge_oracle.rs ran 0 tests"])

    def test_colour_codes_are_ignored(self):
        self.assertEqual(tc.count(COLOURED), tc.count(TWO_BINARIES))

    def test_unit_test_binary_name(self):
        log = "     Running unittests src/lib.rs (target/debug/deps/alice_lol-9f)\n\nrunning 3 tests\n" \
              "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 900 filtered out; finished in 0.1s\n"
        self.assertEqual(tc.count(log), [("unittests src/lib.rs", 3)])

    def test_doctests_are_held_to_the_same_rule(self):
        self.assertEqual(tc.check(WITH_DOCTESTS)[1], ["doctests alice_sdf ran 0 tests"])

    def test_failed_result_counts_the_passed_tests(self):
        log = TWO_BINARIES.replace("test result: ok. 4 passed; 0 failed", "test result: FAILED. 3 passed; 1 failed")
        self.assertEqual(tc.check(log), (12, []))

    def test_log_without_binaries_fails(self):
        self.assertEqual(tc.check("error[E0432]: unresolved import\n"), (0, ["no test binary ran"]))

    def test_main_exit_status(self):
        import contextlib
        import io
        import tempfile
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), tempfile.TemporaryDirectory() as d:
            for text, want in ((TWO_BINARIES, 0), (ONE_EMPTY, 1)):
                p = os.path.join(d, "log")
                with open(p, "w", encoding="utf-8") as f:
                    f.write(text)
                self.assertEqual(tc.main(["x", p]), want)


if __name__ == "__main__":
    unittest.main()
