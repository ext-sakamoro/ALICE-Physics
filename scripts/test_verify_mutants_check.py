#!/usr/bin/env python3
"""Oracle for scripts/verify_mutants_check.py: a cargo-mutants exit that is not a result
(a bad --features exits 1 before any mutant) fails the shard, a single shard of a
non-empty plan with no mutant fails, an axis that tested nothing fails the total even
when the other axis tested some (generated mutants that were not tested do not count),
and a total that read no count fails. Each case fails through an assertion, not through an
exception of the checker (an exception is caught and turned into a failed assertion).

Not covered here: a run whose planned diff is empty is stopped before the checker, in
verify.yml (`src/ changed but the planned diff is empty`), so the checker's handling of an
empty plan is not what decides it."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_mutants_check as vm  # noqa: E402


def call(case: unittest.TestCase, fn, *args):
    """the checker's result, an exception turned into a failed assertion"""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        case.fail(f"{fn.__name__} raised {e!r} instead of returning its findings")


def run_dir(d: Path, generated: int, tested: int, nested: bool = False) -> Path:
    out = d / "mutants.out"
    root = out / "mutants.out" if nested else out
    root.mkdir(parents=True)
    (root / "mutants.json").write_text(json.dumps([{}] * generated), encoding="utf-8")
    outcomes = [{"scenario": "Baseline"}] + [{"scenario": {"Mutant": {}}}] * tested
    (root / "outcomes.json").write_text(json.dumps({"outcomes": outcomes}), encoding="utf-8")
    return out


class Shard(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.plan = self.d / "planned.diff"
        self.plan.write_text("+ a change\n", encoding="utf-8")

    def shard(self, rc: int, out: Path, shards: int = 1) -> list[str]:
        return call(self, vm.shard, rc, out, shards, self.plan, "default", 0, self.d / "counts" / "0-default")

    def test_result_codes_pass(self):
        out = run_dir(self.d, 3, 3)
        for rc in (0, 2, 3):
            self.assertEqual(self.shard(rc, out), [], rc)

    def test_a_bad_features_exit_fails_before_any_mutant(self):
        # cargo-mutants with an unknown --features exits 1 and writes nothing
        errors = self.shard(1, self.d / "mutants.out")
        self.assertTrue(any("exited 1" in e for e in errors), errors)
        self.assertTrue(any("no mutant was generated" in e for e in errors), errors)
        self.assertIn("exited 4", " ".join(self.shard(4, run_dir(self.d, 2, 0))))

    def test_one_shard_of_a_non_empty_plan_with_no_mutant_fails(self):
        self.assertTrue(self.shard(0, run_dir(self.d, 0, 0)))

    def test_an_empty_shard_of_several_is_left_to_the_total(self):
        self.assertEqual(self.shard(0, run_dir(self.d, 0, 0), shards=4), [])

    def test_the_count_is_written_from_either_layout(self):
        self.shard(0, run_dir(self.d, 5, 4, nested=True))
        count = self.d / "counts" / "0-default"
        self.assertTrue(count.is_file(), "the shard wrote no count")
        self.assertEqual(count.read_text(encoding="utf-8"), "default 0 5 4\n")

    def test_a_failing_shard_still_writes_its_count(self):
        self.shard(1, self.d / "mutants.out")
        count = self.d / "counts" / "0-default"
        self.assertTrue(count.is_file(), "a failed shard wrote no count")
        self.assertEqual(count.read_text(encoding="utf-8"), "default 0 0 0\n")


class Total(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def write(self, name: str, text: str) -> None:
        (self.d / name).write_text(text, encoding="utf-8")

    def test_both_axes_tested_pass(self):
        self.write("0-default", "default 0 4 4\n")
        self.write("0-features", "features 0 6 6\n")
        self.assertEqual(call(self, vm.total, self.d, ["default", "features"]), [])

    def test_one_empty_axis_fails_although_the_other_tested_some(self):
        self.write("0-default", "default 0 4 4\n")
        self.write("0-features", "features 0 0 0\n")
        self.assertEqual(call(self, vm.total, self.d, ["default", "features"]), ["axis features tested no mutant"])

    def test_generated_but_untested_mutants_do_not_count(self):
        # (the rule is about mutants tested: a run that planned mutants and tested none verified nothing)
        self.write("0-default", "default 0 4 0\n")
        self.write("0-features", "features 0 6 6\n")
        self.assertEqual(call(self, vm.total, self.d, ["default", "features"]), ["axis default tested no mutant"])

    def test_a_missing_axis_fails(self):
        self.write("0-default", "default 0 4 4\n")
        self.assertEqual(call(self, vm.total, self.d, ["default", "features"]), ["axis features tested no mutant"])

    def test_no_count_fails(self):
        self.assertTrue(call(self, vm.total, self.d, ["default", "features"]))
        self.assertTrue(call(self, vm.total, self.d / "absent", ["default"]))

    def test_a_malformed_or_unknown_count_fails(self):
        self.write("0-default", "default 0 4 4\n")
        self.write("0-features", "features 0 x\n")
        self.write("0-other", "other 0 1 1\n")
        errors = call(self, vm.total, self.d, ["default", "features"])
        self.assertEqual(len(errors), 3, errors)

    def test_an_unknown_axis_is_reported_not_a_crash(self):
        import subprocess

        self.write("0-default", "default 0 4 4\n")
        self.write("0-other", "other 0 1 1\n")
        p = subprocess.run([sys.executable, str(Path(vm.__file__)), "total", "--dir", str(self.d), "--axes", "default"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn("axis other is not one of default", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_the_cli_exit_code(self):
        self.write("0-default", "default 0 4 4\n")
        self.assertEqual(vm.main(["total", "--dir", str(self.d), "--axes", "default"]), 0)
        self.assertEqual(vm.main(["total", "--dir", str(self.d), "--axes", "default,features"]), 1)


if __name__ == "__main__":
    unittest.main()
