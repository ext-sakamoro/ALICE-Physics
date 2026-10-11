#!/usr/bin/env python3
"""Tests for scripts/ci_triggers_check.py (one rule per case)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ci_triggers_check as ct  # noqa: E402

GATED = """name: G
on:
  push:
    branches: [main, 'ci/**']
  pull_request:
    branches: [main]
    paths:
      - 'src/**'
  schedule:
    - cron: '0 3 * * *'
  workflow_dispatch:
    inputs:
      x:
        type: string
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}
jobs:
  changes:
    steps:
      - run: python3 scripts/changed_paths.py decide --patterns-file .github/paths/gated.txt
  heavy:
    needs: changes
  gate:
    name: Gate
    if: always()
    needs: [changes, heavy]
    steps:
      - run: python3 scripts/changed_paths.py gate --needs-json "$NEEDS"
"""
OS = ("${{ fromJSON(github.event_name == 'schedule' && '[\"intel\"]' || "
      "(github.event_name == 'push' && github.ref == 'refs/heads/main') && "
      "'[\"macos-latest\", \"intel\", \"ubuntu-latest\"]' || '[\"macos-latest\", \"ubuntu-latest\"]') }}")
CI = """name: CI
on:
  push:
    branches: [main, 'ci/**']
  pull_request:
    branches: [main]
  schedule:
    - cron: '15 16 * * *'
  workflow_dispatch:
jobs:
  gate:
    outputs:
      run: ${{ steps.decide.outputs.run || 'true' }}
    steps:
      - run: python3 scripts/intel_nightly_gate.py
  test:
    needs: gate
    if: needs.gate.outputs.run == 'true'
    strategy:
      matrix:
        os: OS
  lint:
    if: github.event_name != 'schedule'
""".replace("OS", OS)
WEEKLY = """name: W
on:
  schedule:
    - cron: '0 18 * * 0'
  workflow_dispatch:
concurrency:
  group: ${{ github.workflow }}
  cancel-in-progress: false
jobs:
  m: {}
"""
TAG = """name: R
on:
  push:
    tags:
      - 'v*'
jobs:
  r: {}
"""
EVIDENCE = """[workflows]
"ci.yml" = "landing"
"gated.yml" = "after-landing"
"release.yml" = "tag-only"
"weekly.yml" = "schedule-only"

[[legs]]
workflow = "ci.yml"
job = "test"
matrix = "macos-latest"
class = "after-landing"
events = ["push main", "push ci/**", "pull_request", "workflow_dispatch"]

[[legs]]
workflow = "ci.yml"
job = "test"
matrix = "intel"
class = "main-after-landing"
events = ["push main", "schedule"]
"""
ROWS = """| workflow | push | path gate | pull_request | schedule | dispatch | ci/** push | class |
|---|---|---|---|---|---|---|---|
| `ci.yml` | `main, ci/**` | - | `main` | yes | yes | runs | landing |
| `gated.yml` | `main, ci/**` | `.github/paths/gated.txt` | `main (paths)` | yes | yes | runs when the paths match | after-landing |
| `release.yml` | `tags v*` | - | - | - | - | - | tag-only |
| `weekly.yml` | - | - | - | yes | yes | - | schedule-only |
"""
DISPATCH = """| workflow |
|---|
| `ci.yml` |
| `gated.yml` |
| `weekly.yml` |
"""
WAITS = """| workflow | class | waits for |
|---|---|---|
| `ci.yml` | landing | success; started only: test |
| `gated.yml` | after-landing | started: heavy jobs started, or all skipped by the changes job |
"""
LEGS = """| workflow | job | matrix | class | events |
|---|---|---|---|---|
| `ci.yml` | `test` | `macos-latest` | after-landing | push main, push ci/**, pull_request, workflow_dispatch |
| `ci.yml` | `test` | `intel` | main-after-landing | push main, schedule |
"""


def doc(rows: str = ROWS, legs: str = LEGS, dispatch: str = DISPATCH, waits: str = WAITS) -> str:
    return ("# T\n\n<!-- ci-triggers:workflows:begin -->\n" + rows + "<!-- ci-triggers:workflows:end -->\n\n"
            "<!-- ci-triggers:after-landing-legs:begin -->\n" + legs + "<!-- ci-triggers:after-landing-legs:end -->\n\n"
            "<!-- ci-triggers:dispatchable:begin -->\n" + dispatch + "<!-- ci-triggers:dispatchable:end -->\n\n"
            "<!-- ci-triggers:landing-waits:begin -->\n" + waits + "<!-- ci-triggers:landing-waits:end -->\n")


def tree(evidence: str = EVIDENCE, document: str | None = None, **over: str | None) -> Path:
    d = Path(tempfile.mkdtemp())
    w = d / ".github" / "workflows"
    w.mkdir(parents=True)
    files: dict[str, str | None] = {"ci.yml": CI, "gated.yml": GATED, "weekly.yml": WEEKLY, "release.yml": TAG}
    files.update(over)
    for k, v in files.items():
        if v is not None:
            (w / k).write_text(v, encoding="utf-8")
    (d / ".github" / "ci-evidence.toml").write_text(evidence, encoding="utf-8")
    (d / "docs").mkdir()
    (d / "docs" / "CI_TRIGGERS.md").write_text(doc() if document is None else document, encoding="utf-8")
    return d


class Triggers(unittest.TestCase):
    def errors(self, *a, **k):
        return ct.check(tree(*a, **k))[0]

    def has(self, errors, text):
        self.assertTrue(any(text in e for e in errors), errors)

    def test_the_intended_shape_passes(self):
        self.assertEqual(ct.check(tree()), ([], 4))

    # --- cells -----------------------------------------------------------------------------

    def test_each_cell_is_compared(self):
        row = "| `gated.yml` | `main, ci/**` | `.github/paths/gated.txt` | `main (paths)` | yes | yes | runs when the paths match | after-landing |\n"
        cells = row.strip().strip("|").split("|")
        for i, col in enumerate(ct.COLUMNS[1:], 1):
            changed = list(cells)
            changed[i] = " changed "
            e = self.errors(document=doc(ROWS.replace(row, "|" + "|".join(changed) + "|\n")))
            self.has(e, f"`{col}`")

    def test_a_push_trigger_change_shows(self):
        e = self.errors(**{"ci.yml": CI.replace("[main, 'ci/**']", "[main]")})
        self.has(e, "ci.yml: `push`")
        self.has(e, "does not run on a ci/** push")

    def test_a_removed_path_gate_shows(self):
        e = self.errors(**{"gated.yml": GATED.replace(" --patterns-file .github/paths/gated.txt", "")})
        self.has(e, "`path gate`")
        self.has(e, "`ci/** push`")

    def test_pull_request_paths_show(self):
        body = GATED.replace("    paths:\n      - 'src/**'\n", "")
        self.has(self.errors(**{"gated.yml": body}), "`pull_request`")

    def test_schedule_and_dispatch_show(self):
        e = self.errors(**{"weekly.yml": WEEKLY.replace("  workflow_dispatch:\n", "")})
        self.has(e, "`dispatch`")
        e = self.errors(**{"weekly.yml": WEEKLY.replace("  schedule:\n    - cron: '0 18 * * 0'\n", "")})
        self.has(e, "`schedule`")

    # --- sets ------------------------------------------------------------------------------

    def test_a_workflow_without_a_row_or_class_fails(self):
        extra = CI.replace("name: CI", "name: X")
        e = self.errors(**{"extra.yml": extra})
        self.has(e, "docs/CI_TRIGGERS.md: no entry for workflow extra.yml")
        self.has(e, ".github/ci-evidence.toml: no entry for workflow extra.yml")

    def test_a_row_or_class_without_a_workflow_fails(self):
        e = self.errors(**{"release.yml": None})
        self.has(e, "docs/CI_TRIGGERS.md: entry for release.yml")
        self.has(e, ".github/ci-evidence.toml: entry for release.yml")

    def test_a_duplicate_row_fails(self):
        row = "| `release.yml` | `tags v*` | - | - | - | - | - | tag-only |\n"
        self.has(self.errors(document=doc(ROWS + row)), "listed twice")

    def test_a_short_row_fails(self):
        self.has(self.errors(document=doc(ROWS + "| `x.yml` | - |\n")), "cells, expected")

    # --- classes ---------------------------------------------------------------------------

    def test_class_mismatch_between_doc_and_evidence_fails(self):
        ev = EVIDENCE.replace('"gated.yml" = "after-landing"', '"gated.yml" = "landing"')
        self.has(self.errors(evidence=ev), "`class`")

    def test_unknown_class_fails(self):
        ev = EVIDENCE.replace('"gated.yml" = "after-landing"', '"gated.yml" = "sometimes"')
        self.has(self.errors(evidence=ev), "is not one of")

    def test_schedule_only_with_push_fails(self):
        body = WEEKLY.replace("on:\n", "on:\n  push:\n    branches: [main]\n")
        self.has(self.errors(**{"weekly.yml": body}), "class schedule-only but it triggers on push")

    def test_tag_only_with_branches_fails(self):
        self.has(self.errors(**{"release.yml": TAG.replace("    tags:\n      - 'v*'\n", "    branches: [main]\n")}),
                 "class tag-only")

    def test_landing_off_ci_fails(self):
        ev = EVIDENCE.replace('"release.yml" = "tag-only"', '"release.yml" = "landing"')
        self.has(self.errors(evidence=ev), "release.yml: class landing but it does not run on a ci/** push")

    # --- legs ------------------------------------------------------------------------------

    def test_a_leg_of_a_non_landing_workflow_fails(self):
        ev = EVIDENCE.replace('workflow = "ci.yml"', 'workflow = "gated.yml"', 1)
        self.has(self.errors(evidence=ev), "not a landing workflow")

    def test_a_leg_with_an_unknown_job_fails(self):
        ev = EVIDENCE.replace('job = "test"', 'job = "tests"', 1)
        self.has(self.errors(evidence=ev), "no job 'tests'")

    def test_a_leg_with_an_unknown_matrix_value_fails(self):
        for v in ("macos-15-intel", "macos"):
            ev = EVIDENCE.replace('matrix = "macos-latest"', f'matrix = "{v}"')
            self.has(self.errors(evidence=ev), "not in the job for any event")

    def test_declared_events_must_equal_the_evaluated_matrix(self):
        for wrong in ('["push main", "push ci/**", "pull_request"]',
                      '["push main", "push ci/**", "pull_request", "workflow_dispatch", "schedule"]'):
            ev = EVIDENCE.replace('["push main", "push ci/**", "pull_request", "workflow_dispatch"]', wrong)
            self.has(self.errors(evidence=ev), "macos-latest: declared events")
        ev = EVIDENCE.replace('["push main", "schedule"]', '["push main", "push ci/**", "schedule"]')
        self.has(self.errors(evidence=ev), "intel: declared events")

    def test_a_matrix_change_shows_as_an_event_mismatch(self):
        # the Mac Intel leg added to the ci/** and dispatch runs
        body = CI.replace("|| '[\"macos-latest\", \"ubuntu-latest\"]')", "|| '[\"macos-latest\", \"intel\", \"ubuntu-latest\"]')")
        self.assertNotEqual(body, CI)
        e = self.errors(**{"ci.yml": body})
        self.has(e, "intel: declared events")
        self.has(e, "class main-after-landing needs a main push run and no ci/** push run")

    def test_an_event_that_does_not_trigger_the_workflow_is_not_counted(self):
        # without the schedule trigger the matrix's schedule branch never runs
        body = CI.replace("  schedule:\n    - cron: '15 16 * * *'\n", "")
        e = self.errors(**{"ci.yml": body})
        self.has(e, "intel: declared events ['push main', 'schedule'] but the workflow runs it on ['push main']")

    def test_an_undeclared_main_only_leg_fails(self):
        ev = EVIDENCE.split("[[legs]]")[0] + "[[legs]]" + EVIDENCE.split("[[legs]]")[1]
        legs = "\n".join(LEGS.splitlines()[:3]) + "\n"
        e = self.errors(evidence=ev, document=doc(legs=legs))
        self.has(e, "leg os=intel runs on a main push but not on a ci/** push and is not declared")

    def test_leg_classes(self):
        ev = EVIDENCE.replace('class = "main-after-landing"', 'class = "after-landing"')
        self.has(self.errors(evidence=ev), "class after-landing but it is not started on a ci/** push")
        ev = EVIDENCE.replace('class = "after-landing"\n', 'class = "main-after-landing"\n', 1)
        self.has(self.errors(evidence=ev), "class main-after-landing needs")
        ev = EVIDENCE.replace('class = "after-landing"\n', 'class = "later"\n', 1)
        self.has(self.errors(evidence=ev), "class 'later' is not one of")

    def test_unknown_event_label_and_non_list_fail(self):
        ev = EVIDENCE.replace('"schedule"]', '"nightly"]')
        self.has(self.errors(evidence=ev), "unknown events")
        ev = EVIDENCE.replace('events = ["push main", "schedule"]', 'events = "push main"')
        self.has(self.errors(evidence=ev), "`events` is not a list")

    def test_an_unreadable_matrix_expression_fails(self):
        body = CI.replace("github.event_name == 'schedule'", "contains(github.event_name, 'sched')")
        self.has(self.errors(**{"ci.yml": body}), "ci.yml: job test:")

    def test_a_runtime_matrix_is_skipped_by_the_sweep(self):
        body = CI.replace("  lint:\n", "  fz:\n    strategy:\n      matrix:\n        t: ${{ fromJSON(needs.x.outputs.list) }}\n  lint:\n")
        self.assertEqual(self.errors(**{"ci.yml": body}), [])

    def test_legs_of_the_doc_and_the_evidence_must_agree(self):
        self.has(self.errors(document=doc(legs=LEGS.replace("| after-landing |", "| main-after-landing |"))), "missing from the table")
        self.has(self.errors(document=doc(legs=LEGS.replace("workflow_dispatch |", "schedule |"))), "is not in .github/ci-evidence.toml")

    # --- the guard's gate job versus ci.yml's gate job ---------------------------------------

    def test_a_path_gated_workflow_needs_the_guard_gate_job(self):
        self.has(self.errors(**{"gated.yml": GATED.split("  gate:\n")[0]}), "no `gate` job running")
        body = GATED.replace("changed_paths.py gate", "intel_nightly_gate.py")
        self.has(self.errors(**{"gated.yml": body}), "no `gate` job running")

    def test_the_guard_gate_job_runs_always_and_needs_every_job(self):
        self.has(self.errors(**{"gated.yml": GATED.replace("    if: always()\n", "")}), "not `if: always()`")
        self.has(self.errors(**{"gated.yml": GATED.replace("needs: [changes, heavy]", "needs: [changes]")}), "does not need ['heavy']")

    def test_ci_gate_job_is_not_the_guard(self):
        self.assertEqual(self.errors(), [])  # ci.yml has a `gate` job and no path gate
        body = CI.replace("intel_nightly_gate.py", "changed_paths.py gate")
        self.has(self.errors(**{"ci.yml": body}), "ci.yml: a `gate` job runs changed_paths.py gate, but the workflow has no path gate")

    # --- the expression evaluator ----------------------------------------------------------

    def test_evaluate(self):
        ev = ct.evaluate
        ctx = {"github.event_name": "push", "github.ref": "refs/heads/main"}
        self.assertEqual(ev("'a' && 'b'", ctx), "b")
        self.assertEqual(ev("'' && 'b'", ctx), "")
        self.assertEqual(ev("'' || 'b'", ctx), "b")
        self.assertEqual(ev("'a' || 'b'", ctx), "a")
        self.assertTrue(ev("github.event_name == 'push'", ctx))
        self.assertTrue(ev("github.ref != 'x'", ctx))
        self.assertFalse(ev("!(github.ref == 'refs/heads/main')", ctx))
        self.assertEqual(ev("fromJSON('[\"a\", \"b\"]')", ctx), ["a", "b"])
        self.assertEqual(ev("'it''s'", ctx), "it's")
        self.assertEqual(ev("false || true", ctx), True)
        for bad in ("unknown.x", "'a", "('a'", "'a' 'b'", "fromJSON('[')", "a.b == 'x'", "#"):
            with self.assertRaises(ct.ExprError, msg=bad):
                ev(bad, ctx)

    def test_the_real_ci_matrix(self):
        text = (HERE.parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        job = ct.jobs(text)["test"]
        got = {label: ct.matrix_values(job, "os", {"github.event_name": e, "github.ref": ref})
               for label, e, ref, _ in ct.EVENTS}
        four = ["macos-latest", "ubuntu-latest", "ubuntu-24.04-arm", "windows-latest"]
        self.assertEqual(got["push main"], ["macos-latest", "macos-15-intel", "ubuntu-latest", "ubuntu-24.04-arm", "windows-latest"])
        self.assertEqual(got["push ci/**"], four)
        self.assertEqual(got["pull_request"], four)
        self.assertEqual(got["workflow_dispatch"], four)
        self.assertEqual(got["schedule"], ["macos-15-intel"])

    # --- dispatch --------------------------------------------------------------------------

    def test_a_landing_workflow_without_dispatch_fails(self):
        body = CI.replace("  workflow_dispatch:\n", "")
        rows = ROWS.replace("| `main` | - | yes | runs | landing |", "| `main` | - | - | runs | landing |")
        e = self.errors(document=doc(rows, dispatch=DISPATCH.replace("| `ci.yml` |\n", "")), **{"ci.yml": body})
        self.has(e, "ci.yml: class landing but no workflow_dispatch")

    def test_an_after_landing_workflow_without_dispatch_fails(self):
        body = GATED.replace("  workflow_dispatch:\n    inputs:\n      x:\n        type: string\n", "")
        e = self.errors(**{"gated.yml": body})
        self.has(e, "gated.yml: class after-landing but no workflow_dispatch")
        self.has(e, "`dispatch`")
        self.has(e, "names gated.yml, which has no workflow_dispatch")

    def test_the_dispatchable_list_must_match(self):
        self.has(self.errors(document=doc(dispatch=DISPATCH.replace("| `weekly.yml` |\n", ""))), "misses weekly.yml")
        self.has(self.errors(document=doc(dispatch=DISPATCH + "| `release.yml` |\n")), "names release.yml")
        self.has(self.errors(document=doc(dispatch=DISPATCH + "| `ci.yml` |\n")), "twice")
        self.has(self.errors(document=doc(dispatch="")), "table ci-triggers:dispatchable has no header")
        no_table = doc().split("<!-- ci-triggers:dispatchable:begin -->")[0]
        self.has(self.errors(document=no_table), "no table between <!-- ci-triggers:dispatchable")

    # --- what the landing waits on ---------------------------------------------------------

    def test_the_landing_waits_table_must_match_land_py(self):
        self.has(self.errors(document=doc(waits=WAITS.replace("| landing | success; started only: test |", "| landing | success |"))),
                 "landing-waits row ('ci.yml', 'landing', 'success; started only: test') missing")
        self.has(self.errors(document=doc(waits=WAITS.replace("| landing | success; started only: test |", "| landing | success |"))),
                 "is not what scripts/land.py waits on")
        extra = WAITS + "| `weekly.yml` | landing | success |\n"
        self.has(self.errors(document=doc(waits=extra)), "('weekly.yml', 'landing', 'success') is not what")

    def test_a_missing_landing_waits_table_fails(self):
        no_table = doc().split("<!-- ci-triggers:landing-waits:begin -->")[0]
        self.has(self.errors(document=no_table), "no table between <!-- ci-triggers:landing-waits")

    def test_land_py_failing_to_derive_shows(self):
        body = CI.replace("  workflow_dispatch:\n", "")
        e = self.errors(**{"ci.yml": body})
        self.has(e, "scripts/land.py cannot derive the landing waits")

    # --- concurrency -----------------------------------------------------------------------

    def test_a_shared_group_that_cancels_fails(self):
        body = GATED.replace("${{ github.workflow }}-${{ github.ref }}", "${{ github.workflow }}")
        self.has(self.errors(**{"gated.yml": body}), "a ci/** push can cancel the main run")
        body = body.replace("cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}", "cancel-in-progress: true")
        self.has(self.errors(**{"gated.yml": body}), "a ci/** push can cancel the main run")

    def test_a_shared_group_that_waits_passes(self):
        body = GATED.replace("${{ github.workflow }}-${{ github.ref }}", "${{ github.workflow }}").replace(
            "cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}", "cancel-in-progress: false")
        self.assertEqual(self.errors(**{"gated.yml": body}), [])

    def test_a_shared_cancelling_group_off_ci_is_not_this_check(self):
        body = WEEKLY.replace("cancel-in-progress: false", "cancel-in-progress: true")
        self.assertEqual(self.errors(**{"weekly.yml": body}), [])

    # --- the check itself ------------------------------------------------------------------

    def test_no_row_compares_nothing_and_fails(self):
        header = ROWS.split("\n", 2)
        e, n = ct.check(tree(document=doc(header[0] + "\n" + header[1] + "\n")))
        self.assertEqual(n, 0)
        self.has(e, "compared nothing")

    def test_a_missing_table_fails(self):
        self.has(self.errors(document="# nothing\n"), "no table between")

    def test_wrong_columns_fail(self):
        self.has(self.errors(document=doc(ROWS.replace("| dispatch |", "| manual |", 1))), "columns")

    def test_a_missing_evidence_file_fails(self):
        d = tree()
        (d / ".github" / "ci-evidence.toml").unlink()
        e, n = ct.check(d)
        self.assertEqual(n, 0)
        self.has(e, "ci-evidence.toml")

    def test_the_repository_passes(self):
        errors, n = ct.check(HERE.parent)
        self.assertEqual(errors, [])
        self.assertEqual(n, len(list((HERE.parent / ".github" / "workflows").glob("*.yml"))))


if __name__ == "__main__":
    unittest.main()
