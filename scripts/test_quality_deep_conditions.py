#!/usr/bin/env python3
"""Tests for scripts/quality_deep_conditions.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quality_deep_conditions as qc  # noqa: E402

REAL = (HERE.parent / qc.WORKFLOW).read_text(encoding="utf-8")
SCOPE_TERM = " && inputs.scope == ''"


class Evaluate(unittest.TestCase):
    def test_operators_and_missing_values(self):
        ctx = {"github.event_name": "schedule", "inputs.scope": ""}
        for expr, want in (("github.event_name == 'schedule'", True), ("github.event_name == 'Schedule'", True),
                           ("github.event_name != 'schedule'", False), ("inputs.absent == ''", True),
                           ("inputs.scope != '' || github.event_name == 'schedule'", True),
                           ("inputs.scope == '' && (github.event_name != 'schedule')", False),
                           ("always() && inputs.scope == ''", True), ("!(inputs.scope == '')", False),
                           ("inputs.scope", False), ("github.event_name", True)):
            with self.subTest(expr=expr):
                self.assertEqual(qc.evaluate(expr, ctx), want)

    def test_what_it_does_not_read_is_an_error(self):
        for expr in ("contains(inputs.scope, 'x')", "matrix.os == 'a'", "inputs.scope == 3", "inputs.scope ==",
                     "(inputs.scope == ''", "inputs.scope == '' inputs.label", "inputs.scope == true"):
            with self.subTest(expr=expr):
                with self.assertRaises((qc.Unreadable, TypeError)):
                    qc.evaluate(expr, {})

    def test_a_comment_and_a_wrapper_are_dropped(self):
        self.assertEqual(qc.strip_comment("inputs.a == '#x'  # the nightly"), "inputs.a == '#x'")
        self.assertEqual(qc.strip_comment("${{ inputs.a == '' }}"), "inputs.a == ''")


class Ran(unittest.TestCase):
    def test_a_skipped_need_skips_the_job_unless_it_says_always(self):
        table = {"a": ("inputs.x == '1'", ()), "b": ("", ("a",)), "c": ("always()", ("b",))}
        self.assertEqual(qc.ran(table, {"inputs.x": "0"}), {"a": False, "b": False, "c": True})
        self.assertEqual(qc.ran(table, {"inputs.x": "1"}), {"a": True, "b": True, "c": True})


class RealWorkflow(unittest.TestCase):
    def test_the_file_passes_and_compares_every_job(self):
        errors, compared = qc.check(REAL)
        self.assertEqual(errors, [])
        jobs = len(qc.job_table(REAL))
        self.assertEqual(compared, jobs * len(qc.PLAN_OUTPUTS) * (len(qc.UNSCOPED) + len(qc.SCOPED_RUNS)))
        self.assertGreaterEqual(jobs, len(qc.BEFORE) + len(qc.SCOPED_ONLY))

    def test_each_trigger_runs_something_and_the_sets_differ(self):
        """The contexts are not all-skip: a table that skips everything would agree with anything."""
        now = qc.job_table(REAL)
        plan = qc.PLAN_OUTPUTS[1]
        sets = {name: frozenset(j for j, r in qc.ran(now, {**ctx, **plan}).items() if r)
                for name, ctx in {**qc.UNSCOPED, **qc.SCOPED_RUNS}.items()}
        for name, jobs in sets.items():
            self.assertTrue(jobs, name)
        self.assertEqual(sets["scoped dispatch"], qc.SCOPED)
        self.assertNotEqual(sets["nightly schedule"], sets["weekly schedule"])
        self.assertIn("coverage-full", sets["weekly schedule"])
        self.assertNotIn("coverage-full", sets["nightly schedule"])

    def test_dropping_any_one_scope_term_is_a_finding(self):
        """Without `inputs.scope == ''` on a job, a scoped run starts that job."""
        at = [i for i in range(len(REAL)) if REAL.startswith(SCOPE_TERM, i)]
        self.assertGreaterEqual(len(at), 5, "the scope terms this test removes one by one")
        for i in at:
            line = REAL.count("\n", 0, i) + 1
            with self.subTest(line=line):
                errors, _ = qc.check(REAL[:i] + REAL[i + len(SCOPE_TERM):])
                self.assertTrue(errors)
                self.assertTrue(all("scoped dispatch" in e for e in errors), errors)

    def test_a_changed_unscoped_condition_is_a_finding(self):
        for old, new in (("github.event.schedule != '30 17 * * 1-5')", "github.event.schedule != '30 17 * * 6')"),
                         ("github.event_name != 'workflow_dispatch' || inputs.scope == ''",
                          "github.event_name != 'workflow_dispatch' && inputs.scope == ''"),
                         ("needs.mutants-plan.outputs.jobs != '0'", "needs.mutants-plan.outputs.jobs == '0'")):
            with self.subTest(old=old):
                self.assertIn(old, REAL)
                errors, _ = qc.check(REAL.replace(old, new, 1))
                self.assertTrue(any("scoped dispatch" not in e for e in errors), errors)

    def test_a_new_scoped_job_must_be_skipped_on_unscoped_triggers(self):
        cond = "    if: github.event_name == 'workflow_dispatch' && inputs.scope != ''\n"
        self.assertEqual(REAL.count("  scoped-shards:\n"), 1)
        head, tail = REAL.split("  scoped-shards:\n", 1)
        rest = tail.split(cond, 1)
        self.assertEqual(len(rest), 2)
        errors, _ = qc.check(head + "  scoped-shards:\n" + rest[0] + rest[1])
        self.assertTrue(any("job scoped-shards runs, expected skip" in e for e in errors), errors)

    def test_an_unlisted_job_and_a_missing_job_are_findings(self):
        errors, compared = qc.check(REAL.replace("  loom:\n", "  loom2:\n", 1))
        self.assertEqual(compared, 0)
        self.assertEqual(len(errors), 2, errors)

    def test_an_unreadable_condition_is_a_finding(self):
        old = "    if: needs.mutants-plan.outputs.mode == 'weekly'"
        self.assertIn(old, REAL)
        errors, _ = qc.check(REAL.replace(old, "    if: contains(needs.mutants-plan.outputs.mode, 'weekly')", 1))
        self.assertTrue(errors)

    def test_no_jobs_is_a_finding(self):
        self.assertTrue(qc.check("name: X\non:\n  push:\n")[0])


class Main(unittest.TestCase):
    def test_exit_codes(self):
        self.assertEqual(qc.main([str(HERE.parent)]), 0)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(qc.main([d]), 1)
            wf = Path(d) / qc.WORKFLOW
            wf.parent.mkdir(parents=True)
            wf.write_text(REAL.replace(SCOPE_TERM, "", 1), encoding="utf-8")
            self.assertEqual(qc.main([d]), 1)


if __name__ == "__main__":
    unittest.main()
