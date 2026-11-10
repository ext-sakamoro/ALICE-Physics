#!/usr/bin/env python3
"""Tests of scripts/mutants_pass2.py: which integration targets a missed mutant is run
with, the merge of the second pass into the first pass's output, and the rule for test
targets that changed after the coverage map."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutants_pass2 as p2  # noqa: E402

# a map of three targets over one file: line 10 is run by `a`, line 20 by `b` and `c`;
# `f` spans 10..20 (a, b, c), `g` spans 30..31 and no target runs it
COV = {
    "version": p2.MAP_VERSION, "commit": "c0", "features": "",
    "targets": ["a", "b", "c"],
    "lines": {"src/x.rs": {"10": [0], "20": [1, 2]}},
    "fns": {"src/x.rs": {"f": [0, 1, 2], "g": []}},
}
M10 = "src/x.rs:10:5: replace + with - in f"
M20 = "src/x.rs:20:9: replace < with <= in f"
MG = "src/x.rs:30:1: replace g -> bool with true in g"
MNEW = "src/x.rs:40:1: replace h -> bool with true in h"


class ParseTest(unittest.TestCase):
    def test_a_mutant_name_gives_its_file_line_and_function(self):
        self.assertEqual(p2.parse_mutant(M20), ("src/x.rs", 20, "f"))
        self.assertEqual(p2.parse_mutant("src/s.rs:7:3: replace <impl Support for S<T>>::support -> V "
                                         "with Default::default()"),
                         ("src/s.rs", 7, None))
        self.assertEqual(p2.parse_mutant("src/s.rs:7:3: delete - in <impl Support for S<T>>::support")[2],
                         "<impl Support for S<T>>::support")

    def test_a_line_that_is_not_a_mutant_name_is_refused(self):
        with self.assertRaises(ValueError):
            p2.parse_mutant("src/x.rs: replace + with - in f")


class SelectTest(unittest.TestCase):
    def test_at_the_maps_commit_only_the_targets_of_the_line_run(self):
        self.assertEqual(p2.select(COV, M10, True, set()), ["a"])
        self.assertEqual(p2.select(COV, M20, True, set()), ["b", "c"])
        self.assertEqual(p2.select(COV, MG, True, set()), [])

    def test_at_another_commit_the_targets_of_the_function_run(self):
        self.assertEqual(p2.select(COV, M10, False, set()), ["a", "b", "c"])
        self.assertEqual(p2.select(COV, MG, False, set()), [])

    def test_a_function_the_map_does_not_know_takes_every_target(self):
        self.assertEqual(p2.select(COV, MNEW, False, set()), ["a", "b", "c"])

    def test_the_function_key_is_a_superset_of_the_line_key(self):
        for m in (M10, M20, MG):
            self.assertLessEqual(set(p2.select(COV, m, True, set())), set(p2.select(COV, m, False, set())))

    def test_a_changed_target_is_run_even_when_the_map_does_not_list_it(self):
        self.assertEqual(p2.select(COV, MG, False, {"added_later"}), ["added_later"])
        self.assertEqual(p2.select(COV, M10, True, {"c"}), ["a", "c"])


class ChangedTargetsTest(unittest.TestCase):
    def test_a_target_file_names_its_target(self):
        self.assertEqual(p2.target_of_test_path("tests/analytic_x.rs"), "analytic_x")
        self.assertEqual(p2.target_of_test_path("tests/suite/main.rs"), "suite")
        self.assertIsNone(p2.target_of_test_path("tests/common/mod.rs"))
        self.assertIsNone(p2.target_of_test_path("tests/data/golden.bin"))

    def test_a_changed_shared_file_under_tests_takes_every_target(self):
        self.assertEqual(p2.changed_targets(["tests/common/mod.rs"], ["a", "b"]), {"a", "b"})

    def test_files_outside_tests_change_no_target(self):
        self.assertEqual(p2.changed_targets(["src/x.rs", "docs/a.md"], ["a"]), set())
        self.assertEqual(p2.changed_targets(["tests/a.rs", "tests/new_t.rs"], ["a"]), {"a", "new_t"})


def sh(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


class TestAddedAfterTheMapTest(unittest.TestCase):
    """A test target added after the map's commit is not in the map; a mutant that only
    it kills must still be run with it (else the in-diff gate reports a false miss)."""

    def test_a_test_added_after_the_map_is_selected_for_the_mutant_it_catches(self):
        # git writes its objects read-only, which Windows refuses to delete
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            repo = Path(tmp)
            sh(repo, "init", "-q")
            (repo / "src").mkdir()
            (repo / "tests").mkdir()
            (repo / "src/x.rs").write_text("fn g() -> bool { false }\n")
            (repo / "tests/a.rs").write_text("#[test] fn t() {}\n")
            sh(repo, "add", "-A")
            sh(repo, "commit", "-q", "-m", "map commit")
            map_commit = sh(repo, "rev-parse", "HEAD")
            (repo / "tests/new_t.rs").write_text("#[test] fn g_is_false() { assert!(!g()); }\n")
            sh(repo, "add", "-A")
            sh(repo, "commit", "-q", "-m", "a test that kills the g mutant")
            head = sh(repo, "rev-parse", "HEAD")
            changed = p2.changed_targets(p2.changed_since(map_commit, head, repo), COV["targets"])
            self.assertEqual(changed, {"new_t"})
            # `g` is covered by no target in the map: without the rule it would run with none
            self.assertEqual(p2.select(COV, MG, False, set()), [])
            self.assertEqual(p2.select(COV, MG, False, changed), ["new_t"])
            self.assertEqual(p2.changed_since(head, head, repo), [])


class RunMergeTest(unittest.TestCase):
    """`run` replaces the first pass's missed list with the second pass's outcomes."""

    def make_pass1(self, root: Path) -> Path:
        d = root / "pass1" / "mutants.out"
        d.mkdir(parents=True)
        (d / "caught.txt").write_text("src/x.rs:1:1: replace a with b in f\n")
        (d / "missed.txt").write_text(f"{M10}\n{M20}\n{MG}\n")
        (d / "unviable.txt").write_text("")
        (d / "mutants.json").write_text("[]")
        (d / "outcomes.json").write_text(json.dumps({"end_time": "t", "outcomes": []}))
        return d.parent

    def run_with(self, root: Path, outcomes: dict, same: bool = True) -> tuple[int, Path]:
        cov = dict(COV, commit="HEADSHA" if same else "OLD")
        (root / "map.json").write_text(json.dumps(cov))
        calls = []

        def fake_run_one(name, targets, features, out, timeout):
            calls.append((name, targets))
            return outcomes.get(name)

        with mock.patch.object(p2, "git", return_value="HEADSHA\n"), \
             mock.patch.object(p2, "changed_since", return_value=[]), \
             mock.patch.object(p2, "run_one", side_effect=fake_run_one):
            rc = p2.main(["run", "--map", str(root / "map.json"), "--pass1", str(self.make_pass1(root)),
                          "--out", str(root / "out")])
        self.calls = calls
        return rc, root / "out"

    def test_caught_in_pass_two_leaves_the_missed_list_and_joins_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out = self.run_with(Path(tmp), {M10: "caught", M20: "missed"})
            self.assertEqual(rc, 0)
            self.assertEqual(p2.read_list(out / "missed.txt"), [M20, MG])
            self.assertEqual(p2.read_list(out / "caught.txt"), ["src/x.rs:1:1: replace a with b in f", M10])
            self.assertTrue((out / "mutants.json").is_file() and (out / "outcomes.json").is_file())
            # MG has no covering target at the map's commit: not run, still missed
            self.assertEqual([c[0] for c in self.calls], [M10, M20])
            self.assertEqual(json.loads((out / "pass2.json").read_text())["pass1_missed"], 3)

    def test_a_mutant_without_a_pass_two_outcome_fails_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, _ = self.run_with(Path(tmp), {M10: "caught"})
            self.assertEqual(rc, 1)

    def test_a_map_for_other_features_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "map.json").write_text(json.dumps(dict(COV, features="parallel")))
            with self.assertRaises(SystemExit):
                p2.main(["run", "--map", str(root / "map.json"), "--pass1", str(self.make_pass1(root)),
                         "--out", str(root / "out")])


class NoMapTest(unittest.TestCase):
    def test_without_a_map_every_mutant_runs_with_every_target(self):
        cov = p2.no_map("")
        self.assertIn("analytic_hypot3_max_tie", cov["targets"])
        self.assertEqual(p2.select(cov, M10, False, set()), cov["targets"])


class CoveredLinesTest(unittest.TestCase):
    def test_only_executed_lines_of_src_files_are_read(self):
        lcov = "SF:/w/repo/src/x.rs\nDA:3,0\nDA:4,2\nend_of_record\nSF:/w/dep/lib.rs\nDA:1,5\n"
        self.assertEqual(p2.covered_lines(lcov), {"src/x.rs": {4}})


if __name__ == "__main__":
    unittest.main()
