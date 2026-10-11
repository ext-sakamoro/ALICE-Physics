#!/usr/bin/env python3
"""Tests for scripts/ratchet_merge.py: a ratchet write may only move the baseline the good way."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ratchet_merge as rm  # noqa: E402


def lower(old: int, new: int) -> bool:
    return new < old


REASON = "a reason that is long enough"


class MergeFloors(unittest.TestCase):
    def test_a_lower_value_keeps_the_old_one(self):
        merged, kept, used = rm.merge_floors({"a": 5}, {"a": 3}, lower, {})
        self.assertEqual((merged, kept, used), ({"a": 5}, ["a"], {}))

    def test_a_higher_or_equal_value_is_taken(self):
        merged, kept, _ = rm.merge_floors({"a": 5, "b": 5}, {"a": 7, "b": 5}, lower, {})
        self.assertEqual((merged, kept), ({"a": 7, "b": 5}, []))

    def test_a_new_entry_is_taken_and_a_gone_entry_is_dropped(self):
        merged, _, _ = rm.merge_floors({"gone": 5}, {"new": 1}, lower, {})
        self.assertEqual(merged, {"new": 1})

    def test_an_allowed_entry_is_lowered_and_reported(self):
        merged, kept, used = rm.merge_floors({"a": 5}, {"a": 3}, lower, {"a": REASON})
        self.assertEqual((merged, kept, used), ({"a": 3}, [], {"a": REASON}))

    def test_an_allowance_for_an_entry_that_did_not_fall_is_not_used(self):
        _, _, used = rm.merge_floors({"a": 5}, {"a": 6}, lower, {"a": REASON})
        self.assertEqual(used, {})


class Allowances(unittest.TestCase):
    def test_paired_options_with_long_reasons_pass(self):
        self.assertEqual(rm.allowances(["a", "b"], [REASON, REASON]), ({"a": REASON, "b": REASON}, []))

    def test_unpaired_options_fail(self):
        self.assertTrue(rm.allowances(["a"], [])[1])
        self.assertTrue(rm.allowances([], [REASON])[1])

    def test_a_short_reason_fails(self):
        self.assertTrue(rm.allowances(["a"], ["x" * (rm.MIN_REASON - 1)])[1])
        self.assertFalse(rm.allowances(["a"], ["x" * rm.MIN_REASON])[1])

    def test_comments_are_read_back(self):
        text = f"# header\n# allow-worse a.rs: {REASON}\na.rs 1 0\n#allow-worse   b.rs:  spaced out reason  \n"
        self.assertEqual(rm.allow_comments(text), {"a.rs": REASON, "b.rs": "spaced out reason"})


class DiffCheck(unittest.TestCase):
    def test_a_worse_entry_without_a_comment_fails(self):
        errors, compared = rm.diff_check({"a": 5, "b": 5}, {"a": 3, "b": 5}, lower, {})
        self.assertEqual((len(errors), compared), (1, 2))

    def test_a_worse_entry_with_a_reason_passes_and_a_short_one_does_not(self):
        self.assertEqual(rm.diff_check({"a": 5}, {"a": 3}, lower, {"a": REASON})[0], [])
        self.assertEqual(len(rm.diff_check({"a": 5}, {"a": 3}, lower, {"a": "short"})[0]), 1)

    def test_new_and_better_entries_pass_and_only_shared_entries_count(self):
        errors, compared = rm.diff_check({"a": 5}, {"a": 9, "new": 0}, lower, {})
        self.assertEqual((errors, compared), ([], 1))


if __name__ == "__main__":
    unittest.main()
