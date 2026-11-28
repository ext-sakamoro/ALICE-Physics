#!/usr/bin/env python3
"""Tests for scripts/mutants_covmap.py (one rule per case)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_covmap as cm  # noqa: E402

ROOT = "/work/ALICE-Physics"
SUMMARY = {"data": [{"files": [
    {"filename": f"{ROOT}/src/cloth.rs", "summary": {"lines": {"count": 120, "covered": 40}}},
    {"filename": f"{ROOT}/src/consts.rs", "summary": {"lines": {"count": 0, "covered": 0}}},
    {"filename": f"{ROOT}/src/joint.rs", "summary": {"lines": {"count": 80, "covered": 0}}},
    {"filename": "/home/runner/.cargo/registry/src/x/lib.rs", "summary": {"lines": {"count": 9, "covered": 9}}},
]}]}


class ReadTimes(unittest.TestCase):
    def test_reads_target_and_seconds_and_skips_bad_lines(self):
        t = cm.read_times("# runner: x\n# total\t99\na\t1.5\t0\nb\t2\nbroken\nc\tnot-a-number\n")
        self.assertEqual(t, {"a": 1.5, "b": 2.0})


class FilesOf(unittest.TestCase):
    def test_covered_files_are_those_with_an_executed_line(self):
        self.assertEqual(cm.files_of(SUMMARY, ROOT)["covered"], ["src/cloth.rs"])

    def test_instrumented_files_are_those_with_a_counted_line(self):
        self.assertEqual(cm.files_of(SUMMARY, ROOT)["instrumented"], ["src/cloth.rs", "src/joint.rs"])

    def test_files_outside_the_root_are_dropped(self):
        got = cm.files_of(SUMMARY, ROOT)
        self.assertFalse(any("registry" in f for f in got["covered"] + got["instrumented"]))

    def test_a_trailing_slash_on_the_root_changes_nothing(self):
        self.assertEqual(cm.files_of(SUMMARY, ROOT + "/"), cm.files_of(SUMMARY, ROOT))


class TargetArgs(unittest.TestCase):
    def test_lib_and_integration_targets(self):
        self.assertEqual(cm.target_args(cm.LIB), ["--lib"])
        self.assertEqual(cm.target_args("analytic_cloth"), ["--test", "analytic_cloth"])


def fake_run(fail_target: str | None = None):
    def run(cmd, cwd=None, check=False, capture_output=False, text=False):
        if "--no-report" in cmd and fail_target and fail_target in cmd:
            return subprocess.CompletedProcess(cmd, 101, "test result: FAILED", "")
        if "report" in cmd:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(SUMMARY), "")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run


class RunTarget(unittest.TestCase):
    def test_a_passing_target_gives_its_files(self):
        with mock.patch.object(cm.subprocess, "run", side_effect=fake_run()):
            self.assertEqual(cm.run_target("analytic_cloth", "", Path(ROOT))["covered"], ["src/cloth.rs"])

    def test_a_failing_target_is_an_error_not_an_empty_entry(self):
        with mock.patch.object(cm.subprocess, "run", side_effect=fake_run("analytic_cloth")):
            with self.assertRaises(RuntimeError):
                cm.run_target("analytic_cloth", "", Path(ROOT))

    def test_the_profiles_are_cleaned_before_each_target(self):
        calls = []
        def rec(cmd, **kw):
            calls.append(cmd)
            return fake_run()(cmd, **kw)
        with mock.patch.object(cm.subprocess, "run", side_effect=rec):
            cm.run_target("analytic_cloth", "parallel", Path(ROOT))
        self.assertIn("clean", calls[0])
        self.assertIn("--profile", calls[1])
        self.assertIn("ci-test", calls[1])
        self.assertIn("parallel", calls[1])


class ListTargets(unittest.TestCase):
    def make(self, d: Path) -> Path:
        (d / "tests" / "dirtest").mkdir(parents=True)
        (d / "tests" / "dirtest" / "main.rs").write_text("", encoding="utf-8")
        for n in ("a", "needs_gpu", "needs_wasm", "plain"):
            (d / "tests" / f"{n}.rs").write_text("", encoding="utf-8")
        (d / "Cargo.toml").write_text('[package]\nname = "x"\n\n[[test]]\nname = "needs_gpu"\nrequired-features = ["std", "gpu"]\n\n'
                                      '[[test]]\nname = "needs_wasm"\nrequired-features = ["wasm"]\n', encoding="utf-8")
        return d

    def test_targets_with_their_features_on_and_lib_last(self):
        with tempfile.TemporaryDirectory() as d:
            got = cm.list_targets(self.make(Path(d)), ["gpu"])
            self.assertEqual(got, ["a", "dirtest", "needs_gpu", "plain", cm.LIB])

    def test_a_target_whose_feature_is_off_is_left_out(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertNotIn("needs_gpu", cm.list_targets(self.make(Path(d)), []))

    def test_a_host_unbuildable_target_is_left_out_even_when_asked(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertNotIn("needs_wasm", cm.list_targets(self.make(Path(d)), ["wasm"]))


class TimeTargets(unittest.TestCase):
    def test_build_once_then_time_each_target(self):
        calls = []
        def run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        ticks = iter([0.0, 2.5, 10.0, 10.4])
        with mock.patch.object(cm.subprocess, "run", side_effect=run):
            rows = cm.time_targets(["a", cm.LIB], "simd", Path(ROOT), clock=lambda: next(ticks))
        self.assertEqual(rows, [("a", 2.5, 0), (cm.LIB, 0.4, 0)])
        self.assertIn("--no-run", calls[0])
        self.assertEqual(sum("--no-run" in c for c in calls), 1)

    def test_a_failed_build_is_an_error(self):
        def run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 101 if "--no-run" in cmd else 0, "", "boom")
        with mock.patch.object(cm.subprocess, "run", side_effect=run):
            with self.assertRaises(RuntimeError):
                cm.time_targets(["a"], "", Path(ROOT))

    def test_times_writes_the_runner_and_fails_on_a_failed_target(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "t.txt").write_text("a\nb\n", encoding="utf-8")
            def run(cmd, **kw):
                return subprocess.CompletedProcess(cmd, 101 if "b" in cmd and "--test" in cmd else 0, "", "")
            with mock.patch.object(cm.subprocess, "run", side_effect=run):
                rc = cm.main(["times", "--targets", str(p / "t.txt"), "--root", ROOT, "--runner", "ubuntu-24.04 4 cpu",
                              "--out", str(p / "o.tsv")])
            self.assertEqual(rc, 1)
            text = (p / "o.tsv").read_text()
            self.assertTrue(text.startswith("# runner: ubuntu-24.04 4 cpu"))
            self.assertEqual(set(cm.read_times(text)), {"a", "b"})


class Main(unittest.TestCase):
    def test_buckets_then_join_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "u.txt").write_text("a\nb\nc\n", encoding="utf-8")
            (p / "t.tsv").write_text("a\t10\nb\t1\n", encoding="utf-8")
            self.assertEqual(cm.main(["buckets", "--union", str(p / "u.txt"), "--times", str(p / "t.tsv"), "--k", "2",
                                      "--out", str(p / "b.json")]), 0)
            bs = json.loads((p / "b.json").read_text())["buckets"]
            self.assertEqual(sorted(t for b in bs for t in b), ["a", "b", "c"])
            parts = p / "parts"
            parts.mkdir()
            for i, b in enumerate(bs):
                (parts / f"{i}.json").write_text(json.dumps({"targets": {t: {"covered": [], "instrumented": []} for t in b}}))
            self.assertEqual(cm.main(["join", "--union", str(p / "u.txt"), "--parts", str(parts), "--out", str(p / "m.json")]), 0)

    def test_join_fails_on_a_missing_part(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "u.txt").write_text("a\nb\n", encoding="utf-8")
            (p / "parts").mkdir()
            (p / "parts" / "0.json").write_text(json.dumps({"targets": {"a": {"covered": [], "instrumented": []}}}))
            self.assertEqual(cm.main(["join", "--union", str(p / "u.txt"), "--parts", str(p / "parts"), "--out", str(p / "m.json")]), 1)

    def test_run_fails_on_a_bucket_out_of_range_or_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "b.json").write_text(json.dumps({"buckets": [["a"], []]}))
            self.assertEqual(cm.main(["run", "--buckets", str(p / "b.json"), "--bucket", "5", "--out", str(p / "o.json")]), 1)
            self.assertEqual(cm.main(["run", "--buckets", str(p / "b.json"), "--bucket", "1", "--out", str(p / "o.json")]), 1)

    def test_run_fails_when_a_target_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "b.json").write_text(json.dumps({"buckets": [["a"]]}))
            with mock.patch.object(cm.subprocess, "run", side_effect=fake_run("a")):
                self.assertEqual(cm.main(["run", "--buckets", str(p / "b.json"), "--bucket", "0", "--root", ROOT,
                                          "--out", str(p / "o.json")]), 1)
            self.assertFalse((p / "o.json").exists())

    def test_buckets_fails_on_an_empty_union(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "u.txt").write_text("", encoding="utf-8")
            self.assertEqual(cm.main(["buckets", "--union", str(p / "u.txt"), "--k", "2", "--out", str(p / "b.json")]), 1)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    n = result.testsRun
    if n < 20:
        print(f"error: only {n} test(s) ran (expected at least 20)", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if result.wasSuccessful() else 1)
