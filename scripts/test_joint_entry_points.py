#!/usr/bin/env python3
"""Tests for scripts/joint_entry_points.py (one rule per case)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joint_entry_points as je  # noqa: E402

SOLVER = """\
pub struct PhysicsWorld { joints: Vec<u8> }
impl PhysicsWorld {
    pub fn step(&mut self) {
        let _ = self.try_step();
    }
    pub fn try_step(&mut self) -> Result<(), ()> {
        self.check_step()?;
        self.run();
        Ok(())
    }
    fn check_step(&self) -> Result<(), ()> {
        self.validate_joints()
    }
    fn validate_joints(&self) -> Result<(), ()> {
        Ok(())
    }
    fn run(&mut self) {
        self.inner();
    }
    fn inner(&mut self) {
        solve_joints(&self.joints);
    }
    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }
    pub fn unrelated(&self, seed: [u8; 32]) -> u8 {
        // self.joints in a comment does not count
        seed[0]
    }
}
"""

LIST = """\
step\tvalidated-at-entry\tcalls try_step
try_step\tvalidated-at-entry\tcheck_step first
joint_count\tn/a\treads the length
"""


class Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src" / "solver").mkdir(parents=True)
        self.write(SOLVER)
        self.list = self.root / "list.txt"
        self.list.write_text(LIST, encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, text: str) -> None:
        (self.root / "src" / "solver.rs").write_text(text, encoding="utf-8")

    def check(self) -> tuple[list[str], int]:
        return je.check(self.root, self.list)


class Rules(Case):
    def test_the_classified_entry_points_pass(self) -> None:
        self.assertEqual(self.check(), ([], 3))

    def test_a_new_pub_fn_reaching_the_joints_through_private_calls_fails(self) -> None:
        self.write(SOLVER.replace("    pub fn joint_count", "    pub fn sneaky(&mut self) {\n        self.run();\n    }\n    pub fn joint_count"))
        self.assertTrue(any("`sneaky` reaches the joints and is not classified" in e for e in self.check()[0]))

    def test_validated_at_entry_without_a_validator_call_fails(self) -> None:
        self.write(SOLVER.replace("        self.check_step()?;\n", ""))
        self.assertTrue(any("`try_step` is listed as validated-at-entry but calls no validator" in e
                            for e in self.check()[0]))

    def test_a_listed_method_that_is_gone_fails(self) -> None:
        self.list.write_text(LIST + "removed\tn/a\tgone\n", encoding="utf-8")
        self.assertTrue(any("`removed` is listed but no such method exists" in e for e in self.check()[0]))

    def test_a_listed_method_that_no_longer_reaches_the_joints_fails(self) -> None:
        self.list.write_text(LIST + "unrelated\tn/a\tnothing\n", encoding="utf-8")
        self.assertTrue(any("`unrelated` is listed but is not a pub fn reaching the joints" in e
                            for e in self.check()[0]))

    def test_an_entry_without_a_reason_fails(self) -> None:
        self.list.write_text(LIST.replace("reads the length", ""), encoding="utf-8")
        self.assertTrue(any("`joint_count` has no reason" in e for e in self.check()[0]))

    def test_no_method_found_fails(self) -> None:
        self.write("pub struct PhysicsWorld;\n")
        self.list.write_text("", encoding="utf-8")
        self.assertTrue(any("checked nothing" in e for e in self.check()[0]))

    def test_a_comment_mentioning_the_joints_does_not_reach_them(self) -> None:
        found = je.methods(je.mask(SOLVER))
        self.assertNotIn("unrelated", je.reaching(found))

    def test_a_signature_with_a_semicolon_in_brackets_is_a_method(self) -> None:
        self.assertIn("unrelated", je.methods(je.mask(SOLVER)))

    def test_a_declaration_without_a_body_does_not_take_the_next_body(self) -> None:
        text = "fn f(x: [u8; 2]);\nfn g() {}"
        self.assertEqual(je.body_start(text, 0), -1)
        self.assertEqual(je.body_start(text, text.index("fn g")), text.index("{"))

    def test_methods_in_solver_submodules_are_read(self) -> None:
        (self.root / "src" / "solver" / "extra.rs").write_text(
            "impl PhysicsWorld {\n    pub fn other(&mut self) {\n        self.inner();\n    }\n}\n",
            encoding="utf-8")
        self.assertTrue(any("`other` reaches the joints" in e for e in self.check()[0]))


class RealRepository(unittest.TestCase):
    def test_every_pub_fn_reaching_the_joints_is_classified(self) -> None:
        errors, compared = je.check()
        self.assertEqual(errors, [])
        self.assertGreater(compared, 10)


if __name__ == "__main__":
    unittest.main()
