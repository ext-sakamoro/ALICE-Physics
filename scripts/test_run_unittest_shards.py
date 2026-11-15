#!/usr/bin/env python3
"""Tests for scripts/run_unittest_shards.py: the partition covers the list exactly, and the
parent fails a sharded run that is not one plain run of the discovered list."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_unittest_shards as rs  # noqa: E402

RUNNER = Path(__file__).resolve().parent / "run_unittest_shards.py"


def suite_file(d: Path, body: str, name: str = "test_fake.py") -> Path:
    p = d / name
    p.write_text("import unittest\n\n" + textwrap.dedent(body), encoding="utf-8")
    return p


def run(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RUNNER), str(path), *args], capture_output=True, text=True, timeout=120)


SEVEN = """
class A(unittest.TestCase):
    def test_1(self): pass
    def test_2(self): pass
    def test_3(self): pass
    @unittest.skip("on purpose")
    def test_4(self): pass

class B(unittest.TestCase):
    def test_5(self): pass
    def test_6(self): pass
    def test_7(self): pass
"""


class Partition(unittest.TestCase):
    def test_every_id_is_in_exactly_one_part(self):
        ids = [f"t{i:03d}" for i in range(269)]
        for n in range(1, 12):
            parts = [rs.part(ids, k, n) for k in range(n)]
            self.assertEqual(sorted(x for p in parts for x in p), ids, n)
            self.assertEqual(sum(len(p) for p in parts), len(ids), n)


class Verdict(unittest.TestCase):
    ids = ["a", "b", "c", "d", "e"]

    def report(self, k, n, executed=None, **kw):
        given = rs.part(self.ids, k, n)
        r = {"loaded": list(self.ids), "given": given, "executed": given if executed is None else executed,
             "failures": [], "errors": [], "unexpected_successes": [], "skipped": 0, "ran": len(given)}
        r.update(kw)
        return r

    def good(self, n=2):
        return [self.report(k, n) for k in range(n)]

    def test_a_complete_run_passes(self):
        self.assertEqual(rs.verdict(self.ids, self.good(), [0, 0]), [])

    def test_a_test_in_no_shard_fails(self):
        reps = self.good()
        reps[1]["executed"] = reps[1]["executed"][:-1]
        self.assertTrue(any("ran in no shard" in w for w in rs.verdict(self.ids, reps, [0, 0])))

    def test_a_test_in_two_shards_fails(self):
        reps = self.good()
        reps[1]["executed"] = reps[1]["executed"] + ["a"]
        self.assertTrue(any("more than one shard" in w for w in rs.verdict(self.ids, reps, [0, 0])))

    def test_an_undiscovered_test_fails(self):
        reps = self.good()
        reps[0]["executed"] = reps[0]["executed"] + ["zzz"]
        self.assertTrue(any("not discovered" in w for w in rs.verdict(self.ids, reps, [0, 0])))

    def test_a_shard_that_executed_nothing_fails(self):
        reps = self.good()
        reps[1]["executed"] = []
        self.assertTrue(any("executed 0 tests" in w for w in rs.verdict(self.ids, reps, [0, 0])))

    def test_a_shard_that_loaded_another_list_fails(self):
        reps = self.good()
        reps[0]["loaded"] = self.ids[:-1]
        self.assertTrue(any("loaded another test list" in w for w in rs.verdict(self.ids, reps, [0, 0])))

    def test_a_shard_given_another_part_fails(self):
        reps = self.good()
        reps[0]["given"] = reps[0]["given"][:-1]
        self.assertTrue(any("not given its part" in w for w in rs.verdict(self.ids, reps, [0, 0])))

    def test_a_missing_report_fails(self):
        self.assertTrue(any("wrote no report" in w for w in rs.verdict(self.ids, [self.good()[0], None], [0, 1])))

    def test_a_failure_or_a_nonzero_exit_fails(self):
        reps = self.good()
        reps[0]["failures"] = ["a"]
        self.assertTrue(any("failures" in w for w in rs.verdict(self.ids, reps, [1, 0])))
        self.assertTrue(any("exited 3" in w for w in rs.verdict(self.ids, self.good(), [0, 3])))

    def test_nothing_discovered_fails(self):
        self.assertEqual(rs.verdict([], [], []), ["no test was discovered"])


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="shards-test-"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.d, True)

    def test_seven_tests_in_three_shards(self):
        r = run(suite_file(self.d, SEVEN), "--shards", "3", "--shuffle", "7")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("discovered 7, executed 7 in 3 shards (3, 2, 2), skipped 1", r.stdout)

    def test_more_shards_than_tests_uses_one_per_test(self):
        r = run(suite_file(self.d, SEVEN), "--shards", "20")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("executed 7 in 7 shards", r.stdout)

    def test_a_failing_test_fails_the_run(self):
        r = run(suite_file(self.d, SEVEN + "\nclass C(unittest.TestCase):\n    def test_8(self): self.fail('x')\n"),
                "--shards", "2")
        self.assertEqual(r.returncode, 1)
        self.assertIn("failures", r.stderr)

    def test_a_test_that_leans_on_another_tests_state_shows_when_split(self):
        # B.test_y passes only if A.test_x ran first in the same process: one plain run passes, the shards do not
        body = """
        STATE = []
        class A(unittest.TestCase):
            def test_x(self): STATE.append(1)
        class B(unittest.TestCase):
            def test_y(self): self.assertEqual(STATE, [1])
        """
        p = suite_file(self.d, body)
        self.assertEqual(subprocess.run([sys.executable, str(p)], capture_output=True).returncode, 0)
        r = run(p, "--shards", "2")
        self.assertEqual(r.returncode, 1, r.stdout)

    def test_a_file_with_no_tests_fails(self):
        r = run(suite_file(self.d, "X = 1\n"), "--shards", "2")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no test was discovered", r.stderr)

    def test_an_import_error_fails(self):
        r = run(suite_file(self.d, "import no_such_module_here\n"), "--shards", "2")
        self.assertNotEqual(r.returncode, 0)

    def test_zero_shards_is_refused(self):
        self.assertEqual(run(suite_file(self.d, SEVEN), "--shards", "0").returncode, 2)


if __name__ == "__main__":
    unittest.main()
