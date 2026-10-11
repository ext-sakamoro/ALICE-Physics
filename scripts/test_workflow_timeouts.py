#!/usr/bin/env python3
"""Tests for scripts/workflow_timeouts.py (one rule per case)."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import workflow_timeouts as wt  # noqa: E402

GOOD = """\
name: x
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: echo hi
        timeout-minutes: 5
  test:
    runs-on: ${{ matrix.os }}
    timeout-minutes: ${{ matrix.os == 'macos-15-intel' && 205 || 125 }}
    steps:
      - run: echo hi
"""


class Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".github" / "workflows").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def workflow(self, text: str, name: str = "ci.yml") -> None:
        (self.root / ".github" / "workflows" / name).write_text(text, encoding="utf-8")

    def errors(self) -> list[str]:
        return wt.check(self.root)[0]


class Rules(Case):
    def test_jobs_with_a_limit_pass(self) -> None:
        self.workflow(GOOD)
        self.assertEqual(wt.check(self.root), ([], 2, 0))

    def test_a_job_without_a_limit_fails(self) -> None:
        self.workflow(GOOD + "  lint:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo\n")
        self.assertTrue(any("job `lint` has no job-level timeout-minutes" in e for e in self.errors()))

    def test_a_step_limit_does_not_count_for_the_job(self) -> None:
        self.workflow("jobs:\n  a:\n    runs-on: x\n    steps:\n      - run: echo\n        timeout-minutes: 5\n")
        self.assertTrue(any("job `a` has no job-level" in e for e in self.errors()))

    def test_a_limit_above_the_cap_fails(self) -> None:
        self.workflow(GOOD.replace("timeout-minutes: 30", "timeout-minutes: 361"))
        self.assertTrue(any("361 is outside 1..360" in e for e in self.errors()))

    def test_zero_minutes_fails(self) -> None:
        self.workflow(GOOD.replace("timeout-minutes: 30", "timeout-minutes: 0"))
        self.assertTrue(any("0 is outside" in e for e in self.errors()))

    def test_every_literal_of_an_expression_is_capped(self) -> None:
        self.workflow(GOOD.replace("&& 205 ||", "&& 400 ||"))
        self.assertTrue(any("400 is outside" in e for e in self.errors()))

    def test_numbers_inside_quoted_strings_are_not_limits(self) -> None:
        self.workflow(GOOD.replace("'macos-15-intel'", "'runner-999'"))
        self.assertEqual(self.errors(), [])

    def test_an_expression_without_minutes_fails(self) -> None:
        self.workflow(GOOD.replace("${{ matrix.os == 'macos-15-intel' && 205 || 125 }}", "${{ inputs.minutes }}"))
        self.assertTrue(any("names no minutes" in e for e in self.errors()))

    def test_a_reusable_workflow_call_is_counted_not_required(self) -> None:
        self.workflow(GOOD + "  call:\n    uses: ./.github/workflows/lib.yml\n")
        self.workflow("on: workflow_call\njobs:\n  inner:\n    runs-on: x\n    timeout-minutes: 10\n"
                      "    steps:\n      - run: echo\n", "lib.yml")
        self.assertEqual(wt.check(self.root), ([], 3, 1))

    def test_a_call_to_a_missing_local_workflow_fails(self) -> None:
        self.workflow(GOOD + "  call:\n    uses: ./.github/workflows/gone.yml\n")
        self.assertTrue(any("gone.yml, which does not exist" in e for e in self.errors()))

    def test_the_called_workflow_s_own_jobs_are_checked(self) -> None:
        self.workflow(GOOD + "  call:\n    uses: ./.github/workflows/lib.yml\n")
        self.workflow("on: workflow_call\njobs:\n  inner:\n    runs-on: x\n    steps:\n      - run: echo\n", "lib.yml")
        self.assertTrue(any("lib.yml" in e and "job `inner`" in e for e in self.errors()))

    def test_no_jobs_at_all_fails(self) -> None:
        self.workflow("name: x\non: push\n")
        self.assertTrue(any("checked nothing" in e for e in self.errors()))

    def test_keys_after_jobs_end_the_jobs_block(self) -> None:
        self.workflow(GOOD + "env:\n  lint:\n    x: 1\n")
        self.assertEqual(self.errors(), [])


class RealRepository(unittest.TestCase):
    def test_every_job_of_this_repository_has_a_limit(self) -> None:
        errors, checked, calls = wt.check()
        self.assertEqual(errors, [])
        self.assertGreater(checked + calls, 30, "the workflows have jobs; none = blind")

    def test_removing_one_limit_from_a_copy_of_the_repository_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shutil.copytree(HERE.parent / ".github", root / ".github")
            ci = root / ".github" / "workflows" / "ci.yml"
            lines = ci.read_text(encoding="utf-8").splitlines(keepends=True)
            first = next(i for i, ln in enumerate(lines) if wt.TIMEOUT_RE.match(ln.rstrip("\n")))
            del lines[first]
            ci.write_text("".join(lines), encoding="utf-8")
            self.assertTrue(any("has no job-level timeout-minutes" in e for e in wt.check(root)[0]))


if __name__ == "__main__":
    unittest.main()
