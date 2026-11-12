"""Oracles of scripts/contact_body_access.py.

run: python3 scripts/test_contact_body_access.py
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cba", HERE / "contact_body_access.py")
cba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cba)

GOOD = "let a = self.contact_body(c.body_a);\nself.contact_set_velocity(c.body_b, v);\n"


class Access(unittest.TestCase):
    def run_on(self, solver: str):
        d = Path(tempfile.mkdtemp())
        (d / "src").mkdir()
        (d / "src/solver.rs").write_text(solver)
        return cba.check(d)

    def test_accessor_only_passes(self):
        self.assertEqual(self.run_on(GOOD), ([], 2))

    def test_a_direct_index_fails(self):
        problems, _ = self.run_on(GOOD + "let b = self.bodies[constraint.body_b];\n")
        self.assertEqual(len(problems), 1, problems)

    def test_a_get_by_constraint_index_fails(self):
        problems, _ = self.run_on(GOOD + "let w = self.bodies.get(c.body_a)?.inv_mass;\n")
        self.assertEqual(len(problems), 1, problems)

    def test_a_comment_is_not_code(self):
        problems, _ = self.run_on(GOOD + "// self.bodies[c.body_a] is never read directly\n")
        self.assertEqual(problems, [])

    def test_no_accessor_compares_nothing(self):
        problems, uses = self.run_on("fn x() {}\n")
        self.assertEqual((problems, uses), ([], 0))


if __name__ == "__main__":
    unittest.main()
