"""Oracles of scripts/contact_law_matrix.py.

run: python3 scripts/test_contact_law_matrix.py
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clm", HERE / "contact_law_matrix.py")
clm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clm)

ENUM = """pub enum StaticCollider {
    /// a plane
    Plane(PlaneCollider),
    HeightField(HeightField),
    TriMesh(TriMesh),
}
"""


def tests_text(floors: str, skip: str = "", ignore: str = "") -> str:
    out = ["fn plane() -> Floor { Floor::Static(StaticCollider::Plane(p())) }\n",
           f"fn floors() -> Vec<(&'static str, Floor)> {{ vec![{floors}] }}\n"]
    for test in clm.LAWS.values():
        if test == skip:
            continue
        attr = '#[test]\n#[ignore = "x"]\n' if test == ignore else "#[test]\n"
        out.append(f"{attr}fn {test}() {{ for f in floors() {{ let _ = f; }} }}\n")
    return "".join(out)


FULL = ('("plane", plane()), ("mesh", Floor::Static(StaticCollider::TriMesh(m()))), '
        '("hf", Floor::Static(StaticCollider::HeightField(h()))), ("body", Floor::Body), ("sdf", Floor::Sdf)')


class Matrix(unittest.TestCase):
    def run_on(self, enum: str, tests: str):
        d = Path(tempfile.mkdtemp())
        (d / "src").mkdir()
        (d / "tests").mkdir()
        (d / clm.ENUM).write_text(enum)
        (d / clm.TESTS).write_text(tests)
        return clm.check(d)

    def test_every_pairing_and_law_present_passes(self):
        problems, compared = self.run_on(ENUM, tests_text(FULL))
        self.assertEqual((problems, compared), ([], 3 + 2 + 4))

    def test_a_new_variant_without_a_floor_fails(self):
        enum = ENUM.replace("TriMesh(TriMesh),", "TriMesh(TriMesh),\n    Capsule(CapsuleCollider),")
        problems, _ = self.run_on(enum, tests_text(FULL))
        self.assertTrue(any("StaticCollider::Capsule" in p for p in problems), problems)

    def test_a_missing_body_or_sdf_pairing_fails(self):
        problems, _ = self.run_on(ENUM, tests_text(FULL.replace(', ("sdf", Floor::Sdf)', "")))
        self.assertTrue(any("sdf" in p for p in problems), problems)

    def test_a_missing_or_ignored_law_fails(self):
        problems, _ = self.run_on(ENUM, tests_text(FULL, skip=clm.LAWS["torque"]))
        self.assertTrue(any("law torque" in p for p in problems), problems)
        problems, _ = self.run_on(ENUM, tests_text(FULL, ignore=clm.LAWS["rolling"]))
        self.assertTrue(any("law rolling" in p for p in problems), problems)

    def test_a_known_defect_pin_counts_as_present(self):
        text = tests_text(FULL).replace(
            f"#[test]\nfn {clm.LAWS['torque']}",
            f'#[test]\n#[ignore = "known defect: tips late"]\nfn {clm.LAWS["torque"]}')
        problems, _ = self.run_on(ENUM, text)
        self.assertEqual(problems, [])

    def test_an_unreadable_enum_compares_nothing(self):
        problems, compared = self.run_on("pub struct Nothing;\n", tests_text(FULL))
        self.assertEqual(compared, 0)
        self.assertTrue(problems)


if __name__ == "__main__":
    unittest.main()
