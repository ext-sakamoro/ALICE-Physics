#!/usr/bin/env python3
"""Tests for scripts/fix_conversion_sites.py and the `--float-to-fix` mode of the syntax-tree
scan (scripts/g2-rewrite-scan): a conversion is flagged until it has a reviewed entry, in a
call, as a function value, inside a macro argument and in a `macro_rules!` body; the enclosing
function carries its `impl` type; constant arguments are told from values; a reviewed tree
passes; a stale entry, a wrong count, an empty tree, a tree with fewer sites than asked and a
bad table row fail; test-only items and other `from_f64` functions are not counted."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "fix_conversion_sites.py"


def build_scan() -> str:
    env = os.environ.get("G2_SCAN_BIN")
    if env:
        return env
    target = HERE / "g2-rewrite-scan" / "target"
    subprocess.run(["cargo", "build", "-q", "--release", "--locked", "--target-dir", str(target),
                    "--manifest-path", str(HERE / "g2-rewrite-scan" / "Cargo.toml")], check=True)
    exe = "g2-rewrite-scan.exe" if os.name == "nt" else "g2-rewrite-scan"
    return str(target / "release" / exe)


SCAN = build_scan()

SOURCE = """use crate::math::{Fix128, Vec3Fix};
pub struct Body;
impl Body {
    pub fn place(x: f32, y: f32, z: f32) -> Vec3Fix {
        Vec3Fix::from_f32(x, y, z)
    }
}
pub fn gravity() -> Fix128 {
    Fix128::from_f32(-9.81)
}
pub fn all(v: &[f64]) -> Vec<Fix128> {
    v.iter().copied().map(Fix128::from_f64).collect()
}
pub fn shown(d: f32) {
    debug_assert!(Fix128::from_f32(d) > Fix128::ZERO);
}
macro_rules! fx { ($v:expr) => { Fix128::from_f64($v) }; }
pub struct Other;
impl Other {
    pub fn from_f64(_: f64) -> Self { Other }
}
pub fn not_a_conversion() -> Other {
    Other::from_f64(1.0)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t() { let _ = Fix128::from_f64(f64::NAN); }
}
"""

ROWS = [
    ("src/lib.rs", "Body::place", "Vec3Fix::from_f32", "call", "x , y , z", "value", "1",
     "pending-input", "a point handed in by the caller"),
    ("src/lib.rs", "gravity", "Fix128::from_f32", "call", "- 9.81", "constant", "1",
     "constant", "a finite literal, known in range"),
    ("src/lib.rs", "all", "Fix128::from_f64", "fn-value", "-", "value", "1",
     "pending-input", "each element of a caller's slice"),
    ("src/lib.rs", "shown", "Fix128::from_f32", "call", "d", "value", "1",
     "pending-computed", "inside a debug assertion's argument"),
    ("src/lib.rs", "macro fx", "-", "macro-body", "-", "value", "1",
     "pending-computed", "the macro's argument, whatever the caller passes"),
]


def table_text(rows) -> str:
    return "# header\n" + "".join("\t".join(r) + "\n" for r in rows)


class FloatToFixCheck(unittest.TestCase):
    def run_check(self, source: str, rows, *extra: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text(source, encoding="utf-8")
            table = root / "table.tsv"
            table.write_text(table_text(rows), encoding="utf-8")
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--root", str(root), "--table", str(table),
                 "--scan-bin", SCAN, *extra],
                capture_output=True, text=True, encoding="utf-8")

    def test_a_reviewed_tree_passes_and_test_items_and_other_types_are_not_counted(self):
        p = self.run_check(SOURCE, ROWS)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("5 sites", p.stdout)
        self.assertIn("1 constant", p.stdout)

    def test_each_site_is_flagged_without_its_entry(self):
        for i, row in enumerate(ROWS):
            with self.subTest(row=row[1]):
                p = self.run_check(SOURCE, ROWS[:i] + ROWS[i + 1:])
                self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
                self.assertIn("not reviewed: " + "\t".join(row[:6]), p.stdout)

    def test_a_stale_entry_fails(self):
        stale = ("src/lib.rs", "gone", "Fix128::from_f64", "call", "x", "value", "1",
                 "pending-input", "a site that was removed")
        p = self.run_check(SOURCE, ROWS + [stale])
        self.assertEqual(p.returncode, 1)
        self.assertIn("listed but not found", p.stdout)

    def test_a_wrong_count_fails(self):
        rows = list(ROWS)
        rows[0] = rows[0][:6] + ("2",) + rows[0][7:]
        p = self.run_check(SOURCE, rows)
        self.assertEqual(p.returncode, 1)
        self.assertIn("count 2 in the table, 1 found", p.stdout)

    def test_a_second_identical_site_needs_the_count(self):
        twice = SOURCE.replace("pub fn gravity() -> Fix128 {\n    Fix128::from_f32(-9.81)",
                               "pub fn gravity() -> Fix128 {\n    Fix128::from_f32(-9.81) + Fix128::from_f32(-9.81)")
        p = self.run_check(twice, ROWS)
        self.assertEqual(p.returncode, 1)
        self.assertIn("count 1 in the table, 2 found", p.stdout)

    def test_constant_status_on_a_value_is_refused(self):
        rows = list(ROWS)
        rows[0] = rows[0][:7] + ("constant", "claimed constant")
        p = self.run_check(SOURCE, rows)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("status constant needs kind constant", p.stdout + p.stderr)

    def test_an_empty_tree_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src").mkdir()
            table = root / "t.tsv"
            table.write_text("# empty\n", encoding="utf-8")
            p = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--table",
                                str(table), "--scan-bin", SCAN],
                               capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(p.returncode, 0)

    def test_fewer_sites_than_asked_fails(self):
        p = self.run_check(SOURCE, ROWS, "--min-sites", "6")
        self.assertEqual(p.returncode, 1)
        self.assertIn("fewer than --min-sites 6", p.stdout)

    def test_a_tree_with_no_site_fails(self):
        p = self.run_check("pub fn f() {}\n", [])
        self.assertEqual(p.returncode, 1)
        self.assertIn("0 sites found", p.stdout)

    def test_a_bad_row_fails(self):
        for bad in [ROWS[0][:8], ROWS[0][:7] + ("unknown", "a status that is not one"),
                    ROWS[0][:6] + ("0",) + ROWS[0][7:], ROWS[0][:8] + ("short",)]:
            with self.subTest(bad=bad):
                p = self.run_check(SOURCE, [bad] + ROWS[1:])
                self.assertNotEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
