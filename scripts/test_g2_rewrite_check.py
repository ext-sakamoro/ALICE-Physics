#!/usr/bin/env python3
"""Tests for scripts/g2_rewrite_check.py and the syntax-tree scan it runs
(scripts/g2-rewrite-scan): each shape is flagged until it has a reviewed entry, in any
formatting (a rustfmt'd multi-line `if`, an expression operand, a negated condition, the
assignment below a comment, inside a closure or a const fn, `self.` operands); a `raw_*`
function is reported; a reviewed Fix128 site passes; a stale entry, an empty tree, a tree
with no site and a bad table row fail; test-only items and comparisons that are not a max or
min are not counted."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "g2_rewrite_check.py"


def build_scan() -> str:
    env = os.environ.get("G2_SCAN_BIN")
    if env:
        return env
    # an explicit target directory: a CARGO_TARGET_DIR of the caller (one per checkout)
    # would put the binary somewhere else than the path returned below
    target = HERE / "g2-rewrite-scan" / "target"
    subprocess.run(["cargo", "build", "-q", "--release", "--locked", "--target-dir", str(target),
                    "--manifest-path", str(HERE / "g2-rewrite-scan" / "Cargo.toml")], check=True)
    exe = "g2-rewrite-scan.exe" if os.name == "nt" else "g2-rewrite-scan"
    return str(target / "release" / exe)


SCAN = build_scan()

SINGLE_LINE = "pub fn top(a: f32, b: f32) -> f32 {\n    if a > b { a } else { b }\n}\n"
MULTI_LINE = """pub fn top(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}
"""
EXPR_OPERAND = "pub fn top(x: f32, y: f32) -> f32 {\n    if x + 1.0 > y { x + 1.0 } else { y }\n}\n"
NEGATED = "pub fn top(a: f32, b: f32) -> f32 {\n    if !(a <= b) { a } else { b }\n}\n"
SWAPPED = "pub fn low(a: f32, b: f32) -> f32 {\n    if a > b { b } else { a }\n}\n"
ASSIGN_BELOW_COMMENT = """pub fn floor(mut x: f32, lo: f32) -> f32 {
    if x < lo {
        // raise to the floor
        x = lo;
    }
    x
}
"""
ASSIGN_WITH_OTHER_STATEMENTS = """pub fn track(v: &[f32]) -> f32 {
    let mut best = 0.0;
    let mut n = 0;
    for &x in v {
        if x > best {
            n += 1;
            best = x;
        }
    }
    best + n as f32
}
"""
CLOSURE = "pub fn top(v: &[f32]) -> f32 {\n    v.iter().copied().fold(0.0, |a, b| if a > b { a } else { b })\n}\n"
CONST_FN = "pub const fn top(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}\n"
SELF_OPERANDS = """pub struct S { lo: f32, hi: f32 }
impl S {
    pub fn widen(&mut self, v: f32) {
        if v > self.hi {
            self.hi = v;
        }
    }
}
"""
RAW_FN = "pub(crate) fn raw_thing(x: f64) -> f64 {\n    x\n}\n"
EARLY_RETURN = "pub fn low(a: f32, b: f32) -> f32 {\n    if a < b {\n        return a;\n    }\n    b\n}\n"
DEREF = "pub fn low(a: &f64, b: &f64) -> f64 {\n    if a < b { *a } else { *b }\n}\n"
CLONE = "pub fn top(a: f64, b: f64) -> f64 {\n    if a.clone() > b { a } else { b.clone() }\n}\n"
EXTRA_STMT = "pub fn low(a: f32, b: f32) -> f32 {\n    if a < b {\n        let _k = 0;\n        a\n    } else {\n        b\n    }\n}\n"
ASSIGN_ELSE = "pub fn floor(mut x: f32, y: f32) -> f32 {\n    if x >= y {\n    } else {\n        x = y;\n    }\n    x\n}\n"
ELSE_IF = "pub fn low(a: f32, b: f32, c: bool) -> f32 {\n    if a < b { a } else if c { b } else { b }\n}\n"
BOUND = "pub fn low(a: f32, b: f32) -> f32 {\n    let c = a < b;\n    if c { a } else { b }\n}\n"
METHOD = "pub fn low(a: f32, b: f32) -> f32 {\n    if a.lt(&b) { a } else { b }\n}\n"
MATCH = "pub fn low(a: f32, b: f32) -> f32 {\n    match a < b {\n        true => a,\n        false => b,\n    }\n}\n"
MACRO = """macro_rules! m {
    ($a:expr, $b:expr) => {
        if $a < $b { $a } else { $b }
    };
}
pub fn use_it(a: f32, b: f32) -> f32 {
    m!(a, b)
}
"""
MACRO_UNPARSED = """macro_rules! odd {
    ($a:expr) => { $a + };
}
"""
NOT_A_REWRITE = """pub fn pick(a: f32, b: f32, c: f32) -> f32 {
    if a > b { c } else { b }
}
pub fn bump(mut x: f32, lo: f32) -> f32 {
    if x < lo { x = lo + 1.0; }
    x
}
pub fn eq(a: f32, b: f32) -> f32 {
    if a == b { a } else { b }
}
"""
TEST_ONLY = """pub fn id(x: f32) -> f32 { x }

#[cfg(test)]
mod tests {
    fn top(a: f32, b: f32) -> f32 { if a > b { a } else { b } }
}

#[cfg(all(test, feature = "std"))]
mod more_tests {
    fn top(a: f32, b: f32) -> f32 { if a > b { a } else { b } }
}
"""
NOT_TEST = """#[cfg(not(test))]
pub fn top(a: f32, b: f32) -> f32 { if a > b { a } else { b } }
"""
HEADER = "# file\tfn\tshape\toperands\tstatus\treason\n"
ANCHOR = ("src/anchor.rs", "top", "expr", "a > b")
ANCHOR_SRC = SINGLE_LINE


class G2RewriteCheck(unittest.TestCase):
    def run_check(self, files, rows, make_src=True, anchor=True, extra=(), manifest=None):
        """`anchor` adds one reviewed site so a tree is never empty of sites by accident"""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            if make_src:
                (root / "src").mkdir()
                if anchor:
                    (root / "src" / "anchor.rs").write_text(ANCHOR_SRC, encoding="utf-8")
                    rows = list(rows) + [ANCHOR + ("fixed-point", "a reviewed anchor site")]
            for name, text in files.items():
                (root / "src" / name).write_text(text, encoding="utf-8")
            if manifest is not None:
                (root / "alice-law-conformance.toml").write_text(manifest, encoding="utf-8")
            table = root / "sites.tsv"
            table.write_text(HEADER + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
            p = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), "--table", str(table),
                                "--scan-bin", SCAN, *extra], capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    def assert_flagged_then_reviewed(self, name, text, key):
        rc, out = self.run_check({name: text}, [])
        self.assertEqual(rc, 1, out)
        self.assertIn("not reviewed: " + "\t".join(key), out)
        rc, out = self.run_check({name: text}, [key + ("pending", "a float site still to route")])
        self.assertEqual(rc, 0, out)

    def test_each_form_is_flagged_until_reviewed(self):
        cases = [
            ("single.rs", SINGLE_LINE, ("src/single.rs", "top", "expr", "a > b")),
            ("multi.rs", MULTI_LINE, ("src/multi.rs", "top", "expr", "a > b")),
            ("operand.rs", EXPR_OPERAND, ("src/operand.rs", "top", "expr", "x + 1.0 > y")),
            ("negated.rs", NEGATED, ("src/negated.rs", "top", "expr", "a <= b")),
            ("swapped.rs", SWAPPED, ("src/swapped.rs", "low", "expr", "a > b")),
            ("comment.rs", ASSIGN_BELOW_COMMENT, ("src/comment.rs", "floor", "assign", "x < lo")),
            ("other.rs", ASSIGN_WITH_OTHER_STATEMENTS, ("src/other.rs", "track", "assign", "x > best")),
            ("closure.rs", CLOSURE, ("src/closure.rs", "top", "expr", "a > b")),
            ("constfn.rs", CONST_FN, ("src/constfn.rs", "top", "expr", "a > b")),
            ("selfop.rs", SELF_OPERANDS, ("src/selfop.rs", "widen", "assign", "v > self . hi")),
            ("early.rs", EARLY_RETURN, ("src/early.rs", "low", "return", "a < b")),
            ("deref.rs", DEREF, ("src/deref.rs", "low", "expr", "a < b")),
            ("clone.rs", CLONE, ("src/clone.rs", "top", "expr", "a > b")),
            ("extra.rs", EXTRA_STMT, ("src/extra.rs", "low", "expr", "a < b")),
            ("aelse.rs", ASSIGN_ELSE, ("src/aelse.rs", "floor", "assign", "x >= y")),
            ("elseif.rs", ELSE_IF, ("src/elseif.rs", "low", "expr", "a < b")),
            ("bound.rs", BOUND, ("src/bound.rs", "low", "expr", "a < b")),
            ("method.rs", METHOD, ("src/method.rs", "low", "expr", "a < b")),
            ("match.rs", MATCH, ("src/match.rs", "low", "expr", "a < b")),
        ]
        for name, text, key in cases:
            with self.subTest(name=name):
                self.assert_flagged_then_reviewed(name, text, key)

    def test_a_macro_and_a_select_inside_it_are_reported(self):
        rc, out = self.run_check({"mac.rs": MACRO}, [])
        self.assertEqual(rc, 1, out)
        self.assertIn("not reviewed: src/mac.rs\t<module>\tmacro\tm", out)
        self.assertIn("not reviewed: src/mac.rs\tmacro m\texpr\ta < b", out)
        rows = [("src/mac.rs", "<module>", "macro", "m", "macro-scanned", "body parsed and scanned"),
                ("src/mac.rs", "macro m", "expr", "a < b", "pending", "a select inside the macro")]
        rc, out = self.run_check({"mac.rs": MACRO}, rows)
        self.assertEqual(rc, 0, out)

    def test_a_macro_body_that_does_not_parse_is_named(self):
        rc, out = self.run_check({"odd.rs": MACRO_UNPARSED}, [])
        self.assertEqual(rc, 1, out)
        self.assertIn("macro\todd (a body does not parse)", out)

    def test_a_raw_function_is_reported(self):
        self.assert_flagged_then_reviewed("raw.rs", RAW_FN, ("src/raw.rs", "raw_thing", "raw", "raw_thing"))

    def test_a_reviewed_fix128_site_passes(self):
        rc, out = self.run_check({}, [])
        self.assertEqual(rc, 0, out)
        self.assertIn("1 fixed-point", out)

    def test_a_stale_entry_fails(self):
        rc, out = self.run_check({}, [("src/gone.rs", "top", "expr", "a > b", "pending", "a float site still to route")])
        self.assertEqual(rc, 1, out)
        self.assertIn("listed but not found", out)

    def test_an_empty_tree_and_a_tree_without_sites_fail(self):
        rc, out = self.run_check({}, [], make_src=False)
        self.assertEqual(rc, 1, out)
        rc, out = self.run_check({"x.rs": "pub fn x() {}\n"}, [], anchor=False)
        self.assertEqual(rc, 1, out)
        self.assertIn("fewer than --min-sites", out)

    def test_min_sites_zero_needs_a_written_reason(self):
        empty = {"x.rs": "pub fn x() {}\n"}
        # no manifest, or a manifest without the reason: 0 is refused
        rc, out = self.run_check(empty, [], anchor=False, extra=["--min-sites", "0"])
        self.assertEqual(rc, 1, out)
        self.assertIn("no_comparison_sites_reason", out)
        rc, out = self.run_check(empty, [], anchor=False, extra=["--min-sites", "0"],
                                 manifest='[G2]\nstatus = "pass"\n')
        self.assertEqual(rc, 1, out)
        rc, out = self.run_check(empty, [], anchor=False, extra=["--min-sites", "0"],
                                 manifest='[G2]\nno_comparison_sites_reason = "short"\n')
        self.assertEqual(rc, 1, out)
        # with the reason a tree without sites passes
        reason = '[G2]\nno_comparison_sites_reason = "integer arithmetic only, no float type in the library"\n'
        rc, out = self.run_check(empty, [], anchor=False, extra=["--min-sites", "0"], manifest=reason)
        self.assertEqual(rc, 0, out)
        self.assertIn("1 files scanned; 0 sites", out)
        # and it never excuses a scan that read no file
        rc, out = self.run_check({}, [], make_src=False, extra=["--min-sites", "0"], manifest=reason)
        self.assertEqual(rc, 1, out)
        self.assertIn("0 scanned", out)

    def test_a_scan_that_succeeds_on_no_file_fails(self):
        # the scanner refuses an empty tree itself; this is the checker's own guard behind it: a scan
        # that exits 0 and reports 0 files fails even with --min-sites 0 and a written reason
        import importlib.util
        import contextlib
        import io
        spec = importlib.util.spec_from_file_location("g2_rewrite_check", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.scan = lambda root, scan_bin, dirs: ([], "scanned 0 files")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "alice-law-conformance.toml").write_text(
                '[G2]\nno_comparison_sites_reason = "integer arithmetic only, no float type in the library"\n',
                encoding="utf-8")
            table = root / "sites.tsv"
            table.write_text(HEADER, encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = mod.check(root, table, None, ["src"], 0)
        self.assertEqual(rc, 1, out.getvalue())
        self.assertIn("the scan read no file", out.getvalue())

    def test_test_only_items_and_other_comparisons_are_not_counted(self):
        rc, out = self.run_check({"t.rs": TEST_ONLY, "o.rs": NOT_A_REWRITE}, [])
        self.assertEqual(rc, 0, out)
        self.assertIn("1 fixed-point", out)

    def test_a_not_test_item_is_scanned(self):
        self.assert_flagged_then_reviewed("nt.rs", NOT_TEST, ("src/nt.rs", "top", "expr", "a > b"))

    def test_a_bad_table_row_is_refused(self):
        key = ("src/single.rs", "top", "expr", "a > b")
        for row in [key + ("allowed", "an unknown status here"), key + ("pending", "short"),
                    ("src/single.rs", "top", "clamp", "a > b", "pending", "an unknown shape here")]:
            with self.subTest(row=row):
                rc, out = self.run_check({"single.rs": SINGLE_LINE}, [row])
                self.assertNotEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
