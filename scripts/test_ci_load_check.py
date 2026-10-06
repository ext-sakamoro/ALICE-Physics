#!/usr/bin/env python3
"""Tests for scripts/ci_load_check.py (one rule per case)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ci_load_check as cl  # noqa: E402

QD = """name: Quality Deep
on:
  schedule:
    - cron: '0 18 * * 0'
  workflow_dispatch:
concurrency:
  group: ${{ github.workflow }}
  cancel-in-progress: false
jobs:
  mutants:
    strategy:
      max-parallel: 4
"""
LIGHT = """name: X
on:
  push:
    branches: [main]
concurrency:
  group: ${{ github.workflow }}
  cancel-in-progress: true
jobs: {}
"""


def tree(**over: str) -> Path:
    d = Path(tempfile.mkdtemp())
    wf = d / ".github" / "workflows"
    wf.mkdir(parents=True)
    files = {"quality-deep.yml": QD, "unsafe-and-parallel.yml": LIGHT, "bench-gate.yml": LIGHT}
    files.update(over)
    for k, v in files.items():
        if v is not None:
            (wf / k).write_text(v, encoding="utf-8")
    return d


class Load(unittest.TestCase):
    def errors(self, **over):
        return cl.check(tree(**over))[0]

    def test_the_intended_shape_passes(self):
        self.assertEqual(self.errors(), [])

    def test_a_ci_branch_trigger_fails(self):
        e = self.errors(**{"bench-gate.yml": LIGHT.replace("[main]", "[main, 'ci/**']")})
        self.assertTrue(any("ci/**" in x for x in e), e)

    def test_a_push_trigger_on_quality_deep_fails(self):
        e = self.errors(**{"quality-deep.yml": QD.replace("  workflow_dispatch:\n", "  workflow_dispatch:\n  push:\n    branches: [main]\n")})
        self.assertTrue(any("triggers on push" in x for x in e), e)

    def test_a_per_ref_group_fails(self):
        e = self.errors(**{"unsafe-and-parallel.yml": LIGHT.replace("${{ github.workflow }}", "${{ github.workflow }}-${{ github.ref }}")})
        self.assertTrue(any("per ref" in x for x in e), e)

    def test_a_missing_group_fails(self):
        e = self.errors(**{"bench-gate.yml": LIGHT.split("concurrency:")[0] + "jobs: {}\n"})
        self.assertTrue(any("no top-level concurrency" in x for x in e), e)

    def test_uncapped_or_wide_mutation_matrix_fails(self):
        self.assertTrue(any("no max-parallel" in x for x in self.errors(**{"quality-deep.yml": QD.replace("      max-parallel: 4\n", "")})))
        self.assertTrue(any("exceeds" in x for x in self.errors(**{"quality-deep.yml": QD.replace("max-parallel: 4", "max-parallel: 32")})))

    def test_a_missing_workflow_is_reported(self):
        e = self.errors(**{"bench-gate.yml": None})
        self.assertTrue(any("missing" in x for x in e), e)

    def test_ci_in_a_job_body_is_not_mistaken_for_a_trigger(self):
        body = LIGHT.replace("jobs: {}", "jobs:\n  a:\n    steps:\n      - run: echo ci/**")
        self.assertEqual(self.errors(**{"bench-gate.yml": body}), [])

    def test_the_repository_passes(self):
        errors, seen = cl.check(HERE.parent)
        self.assertEqual(errors, [])
        self.assertEqual(seen, len(cl.HEAVY))


if __name__ == "__main__":
    unittest.main()
