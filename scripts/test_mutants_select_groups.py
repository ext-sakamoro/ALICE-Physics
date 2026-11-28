#!/usr/bin/env python3
"""Tests for scripts/mutants_select_groups.py (one rule per case)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_select_groups as sg  # noqa: E402

UNION = ["analytic_cloth", "analytic_joint", "audit_solver"]
MAP = {"targets": {
    "analytic_cloth": {"covered": ["src/cloth.rs", "src/math.rs"], "instrumented": ["src/cloth.rs", "src/math.rs", "src/consts.rs"]},
    "analytic_joint": {"covered": ["src/joint.rs", "src/math.rs"], "instrumented": ["src/joint.rs", "src/math.rs"]},
    "audit_solver": {"covered": ["src/solver.rs", "src/math.rs"], "instrumented": ["src/solver.rs", "src/math.rs", "src/unused.rs"]},
    "not_in_union": {"covered": ["src/sensors.rs"], "instrumented": ["src/sensors.rs"]},
}}


def by_file(gs: list[dict]) -> dict[str, dict]:
    return {f: g for g in gs for f in g["files"]}


class Groups(unittest.TestCase):
    def test_a_file_runs_only_the_union_targets_that_execute_it(self):
        g = by_file(sg.groups(["src/cloth.rs"], UNION, MAP))["src/cloth.rs"]
        self.assertEqual(g["tests"], ["analytic_cloth"])
        self.assertIsNone(g["fallback"])

    def test_a_file_every_target_executes_runs_all_of_them(self):
        g = by_file(sg.groups(["src/math.rs"], UNION, MAP))["src/math.rs"]
        self.assertEqual(g["tests"], sorted(UNION))
        self.assertIsNone(g["fallback"])

    def test_files_with_the_same_covering_set_share_one_group(self):
        cmap = json.loads(json.dumps(MAP))
        cmap["targets"]["analytic_joint"]["covered"].append("src/force.rs")
        gs = sg.groups(["src/joint.rs", "src/force.rs", "src/cloth.rs"], UNION, cmap)
        self.assertEqual(len(gs), 2)
        self.assertEqual(by_file(gs)["src/force.rs"]["files"], ["src/force.rs", "src/joint.rs"])

    def test_a_target_outside_the_union_is_never_selected(self):
        # sensors.rs is executed only by a target the plan did not select: the union stands, not that target
        g = by_file(sg.groups(["src/sensors.rs"], UNION, MAP))["src/sensors.rs"]
        self.assertEqual(g["tests"], sorted(UNION))
        self.assertEqual(g["fallback"], sg.NOT_COVERED)

    def test_an_instrumented_file_no_union_target_executes_falls_back_to_the_union(self):
        g = by_file(sg.groups(["src/unused.rs"], UNION, MAP))["src/unused.rs"]
        self.assertEqual((g["tests"], g["fallback"]), (sorted(UNION), sg.NOT_COVERED))

    def test_a_file_with_no_instrumented_line_falls_back_as_compile_time_only(self):
        g = by_file(sg.groups(["src/const_only.rs"], UNION, MAP))["src/const_only.rs"]
        self.assertEqual((g["tests"], g["fallback"]), (sorted(UNION), sg.COMPILE_TIME))

    def test_a_non_src_file_falls_back(self):
        for f in ("build.rs", "Cargo.toml", "src/data.json"):
            g = by_file(sg.groups([f], UNION, MAP))[f]
            self.assertEqual((g["tests"], g["fallback"]), (sorted(UNION), sg.NOT_SRC), f)

    def test_a_selected_group_never_holds_a_test_that_does_not_execute_its_files(self):
        gs = sg.groups(["src/cloth.rs", "src/joint.rs", "src/solver.rs", "src/math.rs"], UNION, MAP)
        for g in gs:
            for f in g["files"]:
                for t in g["tests"]:
                    self.assertIn(f, MAP["targets"][t]["covered"], (f, t))

    def test_every_input_file_is_in_exactly_one_group(self):
        files = ["src/cloth.rs", "src/math.rs", "build.rs", "src/const_only.rs", "src/unused.rs", "src/cloth.rs"]
        gs = sg.groups(files, UNION, MAP)
        flat = [f for g in gs for f in g["files"]]
        self.assertEqual(sorted(flat), sorted(set(files)))

    def test_a_union_target_missing_from_the_map_is_an_error_not_a_fallback(self):
        cmap = {"targets": {k: v for k, v in MAP["targets"].items() if k != "audit_solver"}}
        with self.assertRaises(sg.MapError):
            sg.groups(["src/cloth.rs"], UNION, cmap)

    def test_an_empty_map_is_an_error(self):
        for cmap in ({}, {"targets": {}}):
            for union in (UNION, []):
                with self.assertRaises(sg.MapError):
                    sg.groups(["src/cloth.rs"], union, cmap)


class Buckets(unittest.TestCase):
    def test_every_target_is_in_exactly_one_bucket(self):
        union = [f"t{i}" for i in range(23)]
        bs = sg.buckets(union, {t: float(i) for i, t in enumerate(union)}, 4)
        flat = [t for b in bs for t in b]
        self.assertEqual(sorted(flat), sorted(union))
        self.assertEqual(len(bs), 4)

    def test_a_slow_target_gets_its_own_bucket(self):
        times = {"slow": 1800.0, "a": 2.0, "b": 3.0, "c": 4.0, "d": 1.0}
        bs = sg.buckets(list(times), times, 2)
        self.assertIn(["slow"], bs)

    def test_buckets_are_balanced_by_time(self):
        times = {f"t{i}": float(10 + i) for i in range(16)}
        bs = sg.buckets(list(times), times, 4)
        loads = [sum(times[t] for t in b) for b in bs]
        self.assertLessEqual(max(loads) - min(loads), max(times.values()))

    def test_an_unmeasured_target_is_still_placed(self):
        bs = sg.buckets(["new", "old"], {"old": 5.0}, 2)
        self.assertEqual(sorted(t for b in bs for t in b), ["new", "old"])

    def test_k_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            sg.buckets(["a"], {}, 0)


class Merge(unittest.TestCase):
    def test_parts_join_into_one_map(self):
        a = {"targets": {"analytic_cloth": MAP["targets"]["analytic_cloth"]}}
        b = {"targets": {k: MAP["targets"][k] for k in ("analytic_joint", "audit_solver")}}
        self.assertEqual(sorted(sg.merge([a, b], UNION)["targets"]), sorted(UNION))

    def test_a_lost_part_is_an_error(self):
        a = {"targets": {"analytic_cloth": MAP["targets"]["analytic_cloth"]}}
        with self.assertRaises(sg.MapError):
            sg.merge([a], UNION)

    def test_a_target_in_two_parts_is_an_error(self):
        a = {"targets": {"analytic_cloth": MAP["targets"]["analytic_cloth"]}}
        with self.assertRaises(sg.MapError):
            sg.merge([MAP, a], UNION)

    def test_no_part_is_an_error(self):
        with self.assertRaises(sg.MapError):
            sg.merge([], [])


class Main(unittest.TestCase):
    def run_main(self, files: list[str], union: list[str], cmap: dict) -> int:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "m.json").write_text(json.dumps(cmap), encoding="utf-8")
            (p / "u.txt").write_text("\n".join(union), encoding="utf-8")
            (p / "f.txt").write_text("\n".join(files), encoding="utf-8")
            return sg.main(["--map", str(p / "m.json"), "--union", str(p / "u.txt"), "--files", str(p / "f.txt")])

    def test_main_fails_on_an_empty_union_or_no_file(self):
        self.assertEqual(self.run_main(["src/cloth.rs"], [], MAP), 1)
        self.assertEqual(self.run_main([], UNION, MAP), 1)

    def test_main_fails_on_a_partial_map(self):
        self.assertEqual(self.run_main(["src/cloth.rs"], UNION + ["lost_bucket_target"], MAP), 1)

    def test_main_succeeds_on_a_complete_map(self):
        self.assertEqual(self.run_main(["src/cloth.rs"], UNION, MAP), 0)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    n = result.testsRun
    if n < 23:
        print(f"error: only {n} test(s) ran (expected at least 23)", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if result.wasSuccessful() else 1)
