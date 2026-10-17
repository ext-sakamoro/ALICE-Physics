#!/usr/bin/env python3
"""Tests for scripts/changelog_dup_check.py: a bullet written twice in
[Unreleased] (the old and the rewritten version left by a rebase that kept
both sides) fails, distinct bullets and an empty section pass, released
sections and nested bullets are not judged, and a file without [Unreleased]
fails."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import changelog_dup_check as g  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OLD = "- 依存: `alice-det-math` を 0.5.1 にした (要求を `0.5` から `0.5.1` に) 識別子が変わる"
NEW = OLD + " 世代ごとの表で引く"


def doc(unreleased: list[str], released: list[str] = ()) -> str:
    return ("# Changelog\n\n## [Unreleased]\n\n### Changed\n\n" + "\n".join(unreleased)
            + "\n\n## [1.0.0] - 2026-01-01\n\n### Added\n\n" + "\n".join(released) + "\n")


class Duplicates(unittest.TestCase):
    def test_the_repository_passes(self):
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        problems, compared, _ = g.check(text)
        self.assertEqual(problems, [])
        # the bullets of [Unreleased], counted independently: right after a release the section is
        # empty and 0 is the right count; a parser that compares nothing while bullets are there fails
        section = text.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
        bullets = sum(1 for line in section.splitlines() if line.startswith("- "))
        self.assertEqual(compared, bullets)

    def test_the_old_and_the_rewritten_bullet_fail(self):
        problems, compared, _ = g.check(doc([OLD, "- 別の変更", NEW]))
        self.assertEqual(compared, 3)
        self.assertEqual(len(problems), 1)
        self.assertIn("start the same", problems[0])

    def test_three_versions_give_three_pairs(self):
        problems, _, _ = g.check(doc([OLD, NEW, NEW + " さらに足した"]))
        self.assertEqual(len(problems), 3)

    def test_distinct_bullets_pass(self):
        problems, _, _ = g.check(doc(["- 依存: `alice-db` を ^0.5 にした", "- 依存: `alice-analytics` の要求を広げた"]))
        self.assertEqual(problems, [])

    def test_an_empty_unreleased_passes(self):
        problems, compared, _ = g.check(doc([]))
        self.assertEqual((problems, compared), ([], 0))

    def test_released_sections_are_reported_not_judged(self):
        problems, _, notes = g.check(doc(["- 一つ"], [OLD, NEW]))
        self.assertEqual(problems, [])
        self.assertEqual(len(notes), 1)

    def test_nested_bullets_are_not_compared(self):
        problems, compared, _ = g.check(doc(["- 親", "  " + OLD, "  " + NEW]))
        self.assertEqual((problems, compared), ([], 1))

    def test_a_file_without_unreleased_fails(self):
        problems, compared, _ = g.check("# Changelog\n\n## [1.0.0]\n\n- a\n")
        self.assertEqual(compared, 0)
        self.assertEqual(len(problems), 1)


if __name__ == "__main__":
    unittest.main()
