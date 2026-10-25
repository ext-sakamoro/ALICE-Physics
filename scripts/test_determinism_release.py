#!/usr/bin/env python3
"""Tests for determinism_release.py. Usage: python3 scripts/test_determinism_release.py"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import determinism_release as dr  # noqa: E402


def result(passed: int, status: str = "ok") -> str:
    return f"test result: {status}. {passed} passed; 0 failed; 0 ignored; 0 measured; 9 filtered out; finished in 0.01s\n"


class Judge(unittest.TestCase):
    def test_a_term_that_matches_tests_passes(self):
        self.assertEqual(dr.judge("t", "running 2 tests\n" + result(2), 0, 1), [])

    def test_a_term_that_matches_nothing_fails_on_its_own(self):
        # the hole of one invocation with two terms: here each term runs alone, so a dead
        # term reports "0 passed" and fails even when the other term matches
        self.assertTrue(dr.judge("windward", result(0), 0, 1))

    def test_every_binary_must_pass_a_test(self):
        ok = result(3) + result(1) + result(4) + result(2) + result(5)
        self.assertEqual(dr.judge("bins", ok, 0, 5), [])
        one_empty = result(3) + result(0) + result(4) + result(2) + result(5)
        self.assertTrue(dr.judge("bins", one_empty, 0, 5))

    def test_a_missing_binary_result_fails(self):
        self.assertTrue(dr.judge("bins", result(3) * 4, 0, 5))

    def test_a_failed_run_fails(self):
        self.assertTrue(dr.judge("t", result(1, "FAILED"), 101, 1))

    def test_colour_codes_are_ignored(self):
        coloured = "\x1b[1m\x1b[32mtest result: ok\x1b[0m. 3 passed; 0 failed\n"
        self.assertEqual(dr.results(coloured), [("ok", 3)])

    def test_the_terms_and_binaries_are_listed(self):
        self.assertEqual(dr.LIB_TERMS, ("max_min_tests", "windward_alignment"))
        self.assertEqual(dr.TEST_BINARIES, (
            "determinism_golden", "determinism_golden_f32", "determinism_golden_contacts",
            "determinism_golden_paths", "determinism_zero_tie", "determinism_law_outputs",
            "determinism_semantic", "determinism_physics2d_step_digest",
            "determinism_cloth_step_digest", "determinism_golden_coverage"))

    def test_every_determinism_test_file_is_a_release_binary(self):
        tests = sorted(p.stem for p in (dr.ROOT / "tests").glob("determinism_*.rs"))
        self.assertTrue(tests, "no tests/determinism_*.rs read")
        self.assertEqual(sorted(dr.TEST_BINARIES), tests)


if __name__ == "__main__":
    unittest.main()
