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
TWO = """name: X
on:
  push:
    branches: [main, 'ci/**']
concurrency:
  group: ${{ github.workflow }}-${{ github.ref == 'refs/heads/main' && 'main' || 'ci' }}
  cancel-in-progress: false
jobs:
  changes:
    steps:
      - run: python3 scripts/changed_paths.py decide --patterns-file .github/paths/NAME.txt
  heavy:
    needs: changes
    if: needs.changes.outputs.run == 'true'
    steps:
      - run: echo heavy
  other:
    needs: changes
    if: needs.changes.outputs.run == 'true'
    steps:
      - run: echo other
  gate:
    if: always()
    needs: [changes, heavy, other]
"""
GATE = "src/solver.rs\nCargo.toml\n.github/paths/unsafe-and-parallel.txt\n"


def wf(name: str) -> str:
    return TWO.replace("NAME", name)


def tree(paths: str | None = GATE, **over: str | None) -> Path:
    d = Path(tempfile.mkdtemp())
    w = d / ".github" / "workflows"
    w.mkdir(parents=True)
    files: dict[str, str | None] = {
        "quality-deep.yml": QD,
        "unsafe-and-parallel.yml": wf("unsafe-and-parallel"),
        "bench-gate.yml": wf("bench-gate"),
        "integration-status.yml": wf("integration-status"),
    }
    files.update(over)
    for k, v in files.items():
        if v is not None:
            (w / k).write_text(v, encoding="utf-8")
    if paths is not None:
        (d / ".github" / "paths").mkdir()
        (d / ".github" / "paths" / "unsafe-and-parallel.txt").write_text(paths, encoding="utf-8")
    return d


class Load(unittest.TestCase):
    def errors(self, paths: str | None = GATE, **over):
        return cl.check(tree(paths, **over))[0]

    def has(self, errors, text):
        self.assertTrue(any(text in e for e in errors), errors)

    def test_the_intended_shape_passes(self):
        self.assertEqual(self.errors(), [])

    def test_main_only_two_class_workflows_pass(self):
        main_only = wf("bench-gate").replace("[main, 'ci/**']", "[main]")
        self.assertEqual(self.errors(**{"bench-gate.yml": main_only}), [])

    # --- quality-deep -------------------------------------------------------------------

    def test_quality_deep_on_ci_fails(self):
        e = self.errors(**{"quality-deep.yml": QD.replace("  workflow_dispatch:\n", "  workflow_dispatch:\n  push:\n    branches: [main, 'ci/**']\n")})
        self.has(e, "triggers on ci/**")
        self.has(e, "triggers on push")

    def test_a_push_trigger_on_quality_deep_fails(self):
        e = self.errors(**{"quality-deep.yml": QD.replace("  workflow_dispatch:\n", "  workflow_dispatch:\n  push:\n    branches: [main]\n")})
        self.has(e, "triggers on push")

    def test_a_per_ref_group_on_quality_deep_fails(self):
        e = self.errors(**{"quality-deep.yml": QD.replace("${{ github.workflow }}", "${{ github.workflow }}-${{ github.ref }}")})
        self.has(e, "per ref")

    def test_uncapped_or_wide_mutation_matrix_fails(self):
        self.has(self.errors(**{"quality-deep.yml": QD.replace("      max-parallel: 4\n", "")}), "no max-parallel")
        self.has(self.errors(**{"quality-deep.yml": QD.replace("max-parallel: 4", "max-parallel: 32")}), "exceeds")

    # --- the two-class concurrency --------------------------------------------------------

    def test_a_per_ref_group_fails(self):
        for name in cl.TWO_CLASS:
            body = wf(name.rsplit(".", 1)[0]).replace(cl.TWO_CLASS_GROUP, "${{ github.workflow }}-${{ github.ref }}")
            self.has(self.errors(**{name: body}), "not the two-class group")

    def test_a_single_workflow_group_fails(self):
        body = wf("bench-gate").replace(cl.TWO_CLASS_GROUP, "${{ github.workflow }}")
        self.has(self.errors(**{"bench-gate.yml": body}), "not the two-class group")

    def test_a_missing_group_fails(self):
        for name in cl.TWO_CLASS:
            body = wf(name.rsplit(".", 1)[0]).replace(f"  group: {cl.TWO_CLASS_GROUP}\n", "")
            self.has(self.errors(**{name: body}), "no top-level concurrency group")

    def test_a_missing_concurrency_block_fails(self):
        body = wf("integration-status")
        body = body.split("concurrency:")[0] + "jobs:" + body.split("\njobs:", 1)[1]
        e = self.errors(**{"integration-status.yml": body})
        self.has(e, "no top-level concurrency group")
        self.has(e, "cancel-in-progress is None")

    def test_cancel_in_progress_true_fails(self):
        for name in cl.TWO_CLASS:
            body = wf(name.rsplit(".", 1)[0]).replace("cancel-in-progress: false", "cancel-in-progress: true")
            self.has(self.errors(**{name: body}), "cancel-in-progress is 'true'")

    def test_cancel_in_progress_on_branches_only_fails(self):
        body = wf("bench-gate").replace("cancel-in-progress: false", "cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}")
        self.has(self.errors(**{"bench-gate.yml": body}), "cancel-in-progress is")

    # --- the path gate of unsafe-and-parallel ---------------------------------------------

    def test_unsafe_on_ci_without_the_patterns_file_fails(self):
        body = wf("unsafe-and-parallel").replace(" --patterns-file .github/paths/unsafe-and-parallel.txt", "")
        self.has(self.errors(**{"unsafe-and-parallel.yml": body}), "without its path gate")

    def test_unsafe_gate_file_missing_fails(self):
        self.has(self.errors(paths=None), "is missing")

    def test_unsafe_gate_matching_all_src_fails(self):
        for broad in ("src/**", "**/*.rs", "src/*.rs", "**"):
            self.has(self.errors(paths=GATE + broad + "\n"), "selects src/")

    def test_unsafe_gate_with_a_bad_pattern_fails(self):
        self.has(self.errors(paths="src/{a,b}.rs\n"), "unsupported character")

    def test_unsafe_job_not_waiting_for_the_gate_fails(self):
        body = wf("unsafe-and-parallel").replace("  other:\n    needs: changes\n    if: needs.changes.outputs.run == 'true'\n", "  other:\n")
        self.has(self.errors(**{"unsafe-and-parallel.yml": body}), "job other does not wait")

    def test_unsafe_without_a_changes_job_fails(self):
        body = wf("unsafe-and-parallel").replace("  changes:\n", "  decide:\n")
        self.has(self.errors(**{"unsafe-and-parallel.yml": body}), "without a `changes` job")

    def test_unsafe_with_no_heavy_job_fails(self):
        body = wf("unsafe-and-parallel").split("  heavy:\n")[0]
        self.has(self.errors(**{"unsafe-and-parallel.yml": body}), "compared nothing")

    def test_unsafe_main_only_needs_no_gate(self):
        body = wf("unsafe-and-parallel").replace("[main, 'ci/**']", "[main]").replace(" --patterns-file .github/paths/unsafe-and-parallel.txt", "")
        self.assertEqual(self.errors(paths=None, **{"unsafe-and-parallel.yml": body}), [])

    def test_bench_gate_on_ci_needs_no_narrow_gate(self):
        body = wf("bench-gate").replace(" --patterns-file .github/paths/NAME.txt", "")
        self.assertEqual(self.errors(**{"bench-gate.yml": body}), [])

    # --- coverage of the check itself -----------------------------------------------------

    def test_a_missing_workflow_is_reported(self):
        self.has(self.errors(**{"bench-gate.yml": None}), "missing")

    def test_no_workflow_compares_nothing_and_fails(self):
        errors, seen = cl.check(tree(**{k: None for k in cl.HEAVY}))
        self.assertEqual(seen, 0)
        self.has(errors, "compared nothing")

    def test_ci_in_a_job_body_is_not_mistaken_for_a_trigger(self):
        body = QD.replace("      max-parallel: 4\n", "      max-parallel: 4\n    steps:\n      - run: echo ci/**\n")
        self.assertEqual(self.errors(**{"quality-deep.yml": body}), [])

    def test_the_repository_passes(self):
        errors, seen = cl.check(HERE.parent)
        self.assertEqual(errors, [])
        self.assertEqual(seen, len(cl.HEAVY))


if __name__ == "__main__":
    unittest.main()
