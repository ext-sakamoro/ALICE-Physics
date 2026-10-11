#!/usr/bin/env python3
"""Tests for scripts/workflow_timeouts.py (one rule per case)."""

from __future__ import annotations

import re
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


LAYOUTS = {
    "four-space indent": "jobs:\n    b:\n      runs-on: x\n      steps:\n        - run: echo\n",
    "quoted job key": "jobs:\n  'b':\n    runs-on: x\n    steps:\n      - run: echo\n",
    "flow-style job": "jobs:\n  b: {runs-on: x, steps: [{run: echo}]}\n",
    "spaced job key": "jobs:\n  b :\n    runs-on: x\n    steps:\n      - run: echo\n",
    "quoted jobs key": '"jobs":\n  b:\n    runs-on: x\n    steps:\n      - run: echo\n',
}


class Layouts(Case):
    def test_a_job_without_a_limit_is_found_in_every_layout(self) -> None:
        for label, text in LAYOUTS.items():
            with self.subTest(layout=label):
                for f in (self.root / ".github" / "workflows").glob("*"):
                    f.unlink()
                self.workflow(GOOD, "good.yml")
                self.workflow("on: push\n" + text, "x.yml")
                self.assertTrue(any("x.yml" in e and "job `b` has no job-level timeout-minutes" in e
                                    for e in self.errors()), self.errors())

    def test_every_layout_with_a_limit_passes(self) -> None:
        with_limit = {
            "four-space indent": LAYOUTS["four-space indent"].replace("      runs-on: x\n", "      runs-on: x\n      timeout-minutes: 5\n"),
            "quoted job key": LAYOUTS["quoted job key"].replace("    runs-on: x\n", "    runs-on: x\n    timeout-minutes: 5\n"),
            "flow-style job": "jobs:\n  b: {runs-on: x, timeout-minutes: 5, steps: [{run: echo}]}\n",
            "spaced job key": LAYOUTS["spaced job key"].replace("    runs-on: x\n", "    runs-on: x\n    timeout-minutes: 5\n"),
            "quoted jobs key": LAYOUTS["quoted jobs key"].replace("    runs-on: x\n", "    runs-on: x\n    timeout-minutes: 5\n"),
        }
        for label, text in with_limit.items():
            with self.subTest(layout=label):
                self.workflow("on: push\n" + text, "x.yml")
                self.assertEqual(self.errors(), [])

    def test_a_file_with_no_job_fails_even_beside_a_good_file(self) -> None:
        self.workflow(GOOD, "good.yml")
        self.workflow("name: x\non: push\n", "empty.yml")
        self.assertTrue(any("empty.yml: no job found" in e for e in self.errors()))

    def test_an_explicit_complex_key_job_is_read(self) -> None:
        # a job hidden from the parser by an unusual layout still leaves its runs-on
        self.workflow(GOOD + "  ? weird\n  : runs-on: x\n", "x.yml")
        self.assertTrue(any("job `weird` has no job-level timeout-minutes" in e for e in self.errors()))

    def test_a_runs_on_under_another_top_level_key_fails_the_check(self) -> None:
        # e.g. a job template defined outside `jobs:` (an anchor) that the check cannot see
        self.workflow(".template:\n  runs-on: x\n" + GOOD)
        self.assertTrue(any("ci.yml:2: a runs-on / uses key outside every job" in e for e in self.errors()))

    def test_a_flow_style_job_over_several_lines_is_read_to_its_end(self) -> None:
        self.workflow("on: push\njobs:\n  b: {runs-on: x,\n      timeout-minutes: 5}\n")
        self.assertEqual(self.errors(), [])
        self.workflow("on: push\njobs:\n  b: {runs-on: x,\n      steps: [{run: a}],\n      timeout-minutes: 5}\n")
        self.assertEqual(self.errors(), [])
        self.workflow("on: push\njobs:\n  b: {runs-on: x,\n      steps: []}\n")
        self.assertTrue(any("job `b` has no job-level timeout-minutes" in e for e in self.errors()))



class Parser(Case):
    def test_a_job_built_from_an_anchor_has_the_anchor_s_limit(self) -> None:
        base = "on: push\nx-base: &base\n  runs-on: x\n  timeout-minutes: 7\njobs:\n  b:\n    <<: *base\n    steps:\n      - run: echo\n"
        self.workflow(base)
        found = [e for e in self.errors() if "job `b`" in e]
        self.assertEqual(found, [])

    def test_a_job_from_an_anchor_without_a_limit_fails(self) -> None:
        self.workflow("on: push\njobs:\n  a: &base\n    runs-on: x\n    timeout-minutes: 7\n"
                      "    steps:\n      - run: echo\n  b:\n    runs-on: x\n    steps:\n      - run: echo\n")
        self.assertTrue(any("job `b` has no job-level" in e for e in self.errors()))

    def test_invalid_yaml_fails_the_file(self) -> None:
        self.workflow("on: push\njobs:\n\tb:\n\t\truns-on: x\n", "x.yml")
        errs = self.errors()
        self.assertTrue(any("x.yml: not valid YAML" in e for e in errs), errs)
        self.assertTrue(any("x.yml: no job found" in e for e in errs))

    def test_runs_on_inside_a_run_block_of_a_job_is_not_an_orphan(self) -> None:
        self.workflow(GOOD.replace("      - run: echo hi\n        timeout-minutes: 5\n",
                                   "      - run: |\n          echo 'runs-on: nowhere'\n        timeout-minutes: 5\n"))
        self.assertEqual(self.errors(), [])

    def test_runs_on_inside_a_top_level_block_scalar_is_not_an_orphan(self) -> None:
        self.workflow("on: push\nx-doc: |\n  runs-on: described here\n" + GOOD.split("on: push\n", 1)[1])
        self.assertEqual(self.errors(), [])

    def test_a_quoted_runs_on_text_outside_the_jobs_fails_closed(self) -> None:
        self.workflow("on: push\nenv:\n  NOTE: 'runs-on: x'\n" + GOOD.split("on: push\n", 1)[1])
        self.assertTrue(any("ci.yml:3: a runs-on / uses key outside every job" in e for e in self.errors()))

    def test_without_pyyaml_the_check_fails(self) -> None:
        self.workflow(GOOD)
        saved = sys.modules.get("yaml")
        sys.modules["yaml"] = None  # import yaml raises ImportError
        try:
            with self.assertRaises(SystemExit):
                wt.check(self.root)
        finally:
            if saved is not None:
                sys.modules["yaml"] = saved
            else:
                del sys.modules["yaml"]

    def test_the_orphan_scan_does_not_use_the_parser(self) -> None:
        text = "jobs:\n  a:\n    runs-on: x\n    steps:\n      - uses: actions/checkout@v5\n"
        self.assertEqual(wt.orphans(text, [], []), [3, 5])


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
            first = next(i for i, ln in enumerate(lines) if re.match(r"^    timeout-minutes:", ln))
            del lines[first]
            ci.write_text("".join(lines), encoding="utf-8")
            self.assertTrue(any("has no job-level timeout-minutes" in e for e in wt.check(root)[0]))


if __name__ == "__main__":
    unittest.main()
