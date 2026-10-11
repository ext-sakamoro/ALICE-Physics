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
"""
CI = """name: CI
on:
  push:
    branches: [main, 'ci/**']
  pull_request:
    branches: [main]
  workflow_dispatch:
jobs:
  test:
    strategy:
      matrix:
        os: [macos-latest, ubuntu-latest]
"""
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

[[after-landing-legs]]
workflow = "ci.yml"
job = "test"
matrix = "macos-latest"
"""
ROWS = """| workflow | push | path gate | pull_request | schedule | dispatch | ci/** push | class |
|---|---|---|---|---|---|---|---|
| `ci.yml` | `main, ci/**` | - | `main` | - | yes | runs | landing |
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
LEGS = """| workflow | job | matrix |
|---|---|---|
| `ci.yml` | `test` | `macos-latest` |
"""


def doc(rows: str = ROWS, legs: str = LEGS, dispatch: str = DISPATCH) -> str:
    return ("# T\n\n<!-- ci-triggers:workflows:begin -->\n" + rows + "<!-- ci-triggers:workflows:end -->\n\n"
            "<!-- ci-triggers:after-landing-legs:begin -->\n" + legs + "<!-- ci-triggers:after-landing-legs:end -->\n\n"
            "<!-- ci-triggers:dispatchable:begin -->\n" + dispatch + "<!-- ci-triggers:dispatchable:end -->\n")


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
        ev = EVIDENCE.replace('workflow = "ci.yml"', 'workflow = "gated.yml"')
        self.has(self.errors(evidence=ev), "not a landing workflow")

    def test_a_leg_with_an_unknown_job_fails(self):
        ev = EVIDENCE.replace('job = "test"', 'job = "tests"')
        self.has(self.errors(evidence=ev), "no job 'tests'")

    def test_a_leg_with_an_unknown_matrix_value_fails(self):
        for v in ("macos-15-intel", "macos"):
            ev = EVIDENCE.replace('matrix = "macos-latest"', f'matrix = "{v}"')
            self.has(self.errors(evidence=ev), "not in the job")

    def test_legs_of_the_doc_and_the_evidence_must_agree(self):
        self.has(self.errors(document=doc(legs=LEGS.replace("macos-latest", "ubuntu-latest"))), "missing from the table")
        self.has(self.errors(document=doc(legs=LEGS.replace("macos-latest", "ubuntu-latest"))), "is not in .github/ci-evidence.toml")

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
