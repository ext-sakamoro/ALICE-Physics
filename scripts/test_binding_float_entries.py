#!/usr/bin/env python3
"""Tests for scripts/binding_float_entries.py and the `--binding-floats` mode of the
syntax-tree scan: a C function, a `#[wasm_bindgen]` impl's `pub` method, every method of a
`#[pymethods]` impl and a `#[pyfunction]` are listed with their float parameters (a type alias
of floats included, out-parameters and other parameters not); a table equal to the scan passes;
a missing row, a stale row, a bad row, an empty tree and fewer rows than asked fail; test-only
items and functions that are not entry points are not listed."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "binding_float_entries.py"


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

SOURCE = """type Pair = (f64, f64);
type Pairs = Vec<Pair>;
#[no_mangle]
pub unsafe extern "C" fn c_step(world: *mut World, dt: f64, out: *mut f64, n: u32) -> u8 { 0 }
#[wasm_bindgen]
impl Js {
    pub fn set(&mut self, id: usize, x: f64) {}
    fn hidden(&self, x: f64) {}
}
#[pymethods]
impl Py {
    fn grid(&mut self, data: PyReadonlyArray2<f64>, at: Pairs, name: String) {}
}
#[pyfunction]
fn encode(x: f32, flag: bool) {}
pub fn not_an_entry(x: f64) {}
#[cfg(test)]
mod tests {
    #[no_mangle]
    pub extern "C" fn test_only(x: f64) {}
}
"""

ROWS = [
    ("src/lib.rs", "c_step", "dt", "f64"),
    ("src/lib.rs", "Js::set", "x", "f64"),
    ("src/lib.rs", "Py::grid", "data", "PyReadonlyArray2 < f64 >"),
    ("src/lib.rs", "Py::grid", "at", "Pairs"),
    ("src/lib.rs", "encode", "x", "f32"),
]


class BindingFloatEntries(unittest.TestCase):
    def run_check(self, source: str, rows, *extra: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text(source, encoding="utf-8")
            table = root / "t.tsv"
            table.write_text("# header\n" + "".join("\t".join(r) + "\n" for r in rows),
                             encoding="utf-8")
            return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--table",
                                   str(table), "--scan-bin", SCAN, *extra],
                                  capture_output=True, text=True, encoding="utf-8")

    def test_a_table_equal_to_the_scan_passes(self):
        p = self.run_check(SOURCE, ROWS)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("5 parameters", p.stdout)

    def test_each_parameter_is_flagged_without_its_row(self):
        for i, row in enumerate(ROWS):
            with self.subTest(row=row):
                p = self.run_check(SOURCE, ROWS[:i] + ROWS[i + 1:])
                self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
                self.assertIn("no row: " + "\t".join(row), p.stdout)

    def test_a_stale_row_fails(self):
        p = self.run_check(SOURCE, ROWS + [("src/lib.rs", "gone", "x", "f64")])
        self.assertEqual(p.returncode, 1)
        self.assertIn("row not found", p.stdout)

    def test_a_bad_row_fails(self):
        for bad in [("src/lib.rs", "c_step", "dt"), ("src/lib.rs", "c_step", "", "f64")]:
            with self.subTest(bad=bad):
                p = self.run_check(SOURCE, [bad] + ROWS[1:])
                self.assertNotEqual(p.returncode, 0)

    def test_fewer_rows_than_asked_fail(self):
        p = self.run_check(SOURCE, ROWS, "--min-rows", "6")
        self.assertEqual(p.returncode, 1)
        self.assertIn("fewer than --min-rows 6", p.stdout)

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


if __name__ == "__main__":
    unittest.main()
