#!/usr/bin/env python3
"""Tests for scripts/g2_rewrite_check.py: each rewrite shape is flagged until it is reviewed,
a reviewed Fix128 site passes, a stale entry and an empty tree fail, and test modules and
comparisons that are not a max or min (and integer clamps) are not counted."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "g2_rewrite_check.py"

EXPR_F32 = """pub fn top(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}
"""
EXPR_SWAPPED = """pub fn low(a: f32, b: f32) -> f32 {
    if a > b { b } else { a }
}
"""
ASSIGN_SAME_LINE = """pub fn floor(mut x: f32, lo: f32) -> f32 {
    if x < lo { x = lo; }
    x
}
"""
ASSIGN_NEXT_LINE = """pub fn cap(v: &mut [f32], hi: f32) {
    for x in v {
        if *x > hi {
            *x = hi;
        }
    }
}
"""
FIX128 = """use crate::math::Fix128;
pub fn top(a: Fix128, b: Fix128) -> Fix128 {
    if a > b { a } else { b }
}
"""
NOT_A_REWRITE = """pub fn pick(a: f32, b: f32, c: f32) -> f32 {
    if a > b { c } else { b }
}
pub fn bump(mut x: f32, lo: f32) -> f32 {
    if x < lo { x = lo + 1.0; }
    x
}
"""
IN_TEST_MODULE = """pub fn id(x: f32) -> f32 { x }

#[cfg(test)]
mod tests {
    fn top(a: f32, b: f32) -> f32 {
        if a > b { a } else { b }
    }
}
"""
FLOAT_CLAMP = """pub fn unit(p: f64) -> f64 {
    p.clamp(0.0, 1.0)
}
"""
INT_CLAMP = """pub fn cell(c: i64, n: i64) -> usize {
    c.clamp(0, n - 1) as usize
}
"""
HEADER = "# file\tfn\tshape\toperands\tstatus\treason\n"


class G2RewriteCheck(unittest.TestCase):
    def run_check(self, files, table_rows, make_src=True):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            if make_src:
                (root / "src").mkdir()
            for name, text in files.items():
                (root / "src" / name).write_text(text, encoding="utf-8")
            table = root / "sites.tsv"
            table.write_text(HEADER + "".join("\t".join(r) + "\n" for r in table_rows), encoding="utf-8")
            p = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--table", str(table)],
                               capture_output=True, text=True)
        return p.returncode, p.stdout

    def test_each_shape_is_flagged_until_reviewed(self):
        for name, text, key in [
            ("e.rs", EXPR_F32, ("src/e.rs", "top", "expr", "a > b")),
            ("s.rs", EXPR_SWAPPED, ("src/s.rs", "low", "expr", "a > b")),
            ("a.rs", ASSIGN_SAME_LINE, ("src/a.rs", "floor", "assign", "x < lo")),
            ("n.rs", ASSIGN_NEXT_LINE, ("src/n.rs", "cap", "assign", "*x > hi")),
            ("c.rs", FLOAT_CLAMP, ("src/c.rs", "unit", "clamp", "0.0, 1.0")),
        ]:
            with self.subTest(name=name):
                rc, out = self.run_check({name: text}, [])
                self.assertEqual(rc, 1, out)
                self.assertIn("not reviewed: " + "\t".join(key), out)
                rc, out = self.run_check({name: text}, [key + ("pending", "a float site still to route")])
                self.assertEqual(rc, 0, out)
                self.assertIn("1 pending", out)

    def test_a_reviewed_fix128_site_passes_and_an_unreviewed_one_fails(self):
        key = ("src/f.rs", "top", "expr", "a > b")
        rc, out = self.run_check({"f.rs": FIX128}, [key + ("fixed-point", "operands are Fix128 integers")])
        self.assertEqual(rc, 0, out)
        self.assertIn("1 fixed-point", out)
        rc, out = self.run_check({"f.rs": FIX128}, [])
        self.assertEqual(rc, 1, out)

    def test_a_routed_crate_method_is_accepted_only_when_listed(self):
        key = ("src/c.rs", "unit", "clamp", "0.0, 1.0")
        rc, out = self.run_check({"c.rs": FLOAT_CLAMP}, [key + ("routed", "a crate method that calls the Law clamp")])
        self.assertEqual(rc, 0, out)
        self.assertIn("1 routed", out)

    def test_a_stale_entry_fails(self):
        key = ("src/gone.rs", "top", "expr", "a > b")
        rc, out = self.run_check({"x.rs": "pub fn x() {}\n"}, [key + ("pending", "a float site still to route")])
        self.assertEqual(rc, 1, out)
        self.assertIn("listed but not found", out)

    def test_no_source_file_fails(self):
        rc, out = self.run_check({}, [], make_src=False)
        self.assertEqual(rc, 1, out)
        self.assertIn("0 scanned", out)
        rc, out = self.run_check({}, [])
        self.assertEqual(rc, 1, out)

    def test_test_modules_and_other_comparisons_are_not_counted(self):
        rc, out = self.run_check({"t.rs": IN_TEST_MODULE, "o.rs": NOT_A_REWRITE, "i.rs": INT_CLAMP}, [])
        self.assertEqual(rc, 0, out)
        self.assertIn("0 rewrites", out)

    def test_a_bad_table_row_is_refused(self):
        key = ("src/e.rs", "top", "expr", "a > b")
        for row in [key + ("allowed", "an unknown status here"), key + ("pending", "short")]:
            with self.subTest(row=row):
                rc, out = self.run_check({"e.rs": EXPR_F32}, [row])
                self.assertNotEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
