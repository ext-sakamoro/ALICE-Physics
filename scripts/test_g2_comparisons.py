#!/usr/bin/env python3
"""Tests for scripts/g2_comparisons.py and the `--comparisons` mode of scripts/g2-rewrite-scan:
every ordering comparison is listed in any form (binary, method, path call, inside a macro
argument, in a closure, with a context), test items are skipped; the operand wrap nests and
crosses lines; the type probe types a real crate (a float, a Fix128-like struct, a reference,
a type parameter) and reports a configuration that typed nothing; the judge flags a float
comparison without a row, a stale row, a count change, a misclassified type, a `predicate`
on a selection, a row on a non-float comparison and an unprobed comparison, and passes a
reviewed tree; a bad table row is refused."""
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g2_comparisons as g  # noqa: E402


def build_scan() -> str:
    env = os.environ.get("G2_SCAN_BIN")
    if env:
        return env
    subprocess.run(["cargo", "build", "-q", "--release", "--locked",
                    "--manifest-path", str(HERE / "g2-rewrite-scan" / "Cargo.toml")], check=True)
    exe = "g2-rewrite-scan.exe" if os.name == "nt" else "g2-rewrite-scan"
    return str(HERE / "g2-rewrite-scan" / "target" / "release" / exe)


SCAN = build_scan()


def scan_text(src: str, *args: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "src").mkdir()
        (Path(d) / "src" / "lib.rs").write_text(src, encoding="utf-8")
        r = subprocess.run([SCAN, *args, "src"], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout


SOURCE = """pub fn f(a: f32, b: f32, v: &[f32]) -> bool {
    let c = a.partial_cmp(&b);
    debug_assert!(a <= 2.0);
    let n = v.iter().filter(|&&x| x > 0.5).count();
    if a < b && PartialOrd::lt(&b, &a) {
        return a.lt(&b);
    }
    while a >= b {}
    c.is_some() && n > 0
}
#[cfg(test)]
mod tests {
    fn g(a: f32, b: f32) -> bool { a < b }
}
"""


class Inventory(unittest.TestCase):
    def test_every_form_is_listed_with_its_context(self):
        rows = [l.split("\t") for l in scan_text(SOURCE, "--comparisons").splitlines()]
        got = {(r[2], r[3], r[6]) for r in rows}
        self.assertIn(("method", "a partial_cmp b", "let binding"), got)
        self.assertIn(("bin", "a <= 2.0", "assertion"), got)
        self.assertIn(("bin", "x > 0.5", "closure body"), got)
        self.assertIn(("bin", "a < b", "boolean operand"), got)
        self.assertIn(("path", "b lt a", "boolean operand"), got)
        self.assertIn(("method", "a lt b", "return value"), got)
        self.assertIn(("bin", "a >= b", "while condition"), got)
        self.assertIn(("bin", "n > 0", "boolean operand"), got)
        self.assertEqual(len(rows), 8, rows)  # the test module is not counted

    def test_the_probe_span_skips_a_literal_left_operand(self):
        rows = [l.split("\t") for l in scan_text("pub fn f(x: f32) -> bool { 0.5 < x }\n", "--comparisons").splitlines()]
        self.assertEqual(rows[0][5], "1:33-1:34")  # the `x`


class Wrap(unittest.TestCase):
    def test_nested_and_multiline_spans(self):
        text = "fn f() {\n    if (a <\n        b) < c {}\n}"
        # `(a <\n b)` is 2:7-3:10 and `a` is 2:8-2:9
        out = g.wrap(text, [(1, (2, 7, 3, 10)), (2, (2, 8, 2, 9))])
        self.assertEqual(out, "fn f() {\n    if crate::__cmp_probe::s1((crate::__cmp_probe::s2(a) <\n        b)) < c {}\n}")

    def test_two_spans_on_one_line(self):
        out = g.wrap("a < b && c < d", [(0, (1, 0, 1, 1)), (1, (1, 9, 1, 10))])
        self.assertEqual(out, "crate::__cmp_probe::s0(a) < b && crate::__cmp_probe::s1(c) < d")


class Types(unittest.TestCase):
    def test_float_types(self):
        for t in ("f32", "f64", "&f32", "&&f64", "&mut f32", "wide::f32x8", "T"):
            self.assertTrue(g.is_float(t), t)
        for t in ("Fix128", "usize", "&u32", "(Fix128, RayTarget)", "Option<Fix128>", "{integer}"):
            self.assertFalse(g.is_float(t), t)


PROBE_CRATE = """pub struct Fix(pub i64);
impl PartialEq for Fix { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
impl PartialOrd for Fix {
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> { self.0.partial_cmp(&o.0) }
}
pub fn a(x: f32, y: f32) -> bool { x < y }
pub fn b(x: &Fix, y: &Fix) -> bool { x < y }
pub fn c<T: PartialOrd>(x: T, y: T) -> bool { x > y }
pub fn d(v: &[f64]) -> usize { v.iter().filter(|&&x| 0.5 <= x).count() }
"""


class Probe(unittest.TestCase):
    def run_probe(self, configs):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "src").mkdir()
            (root / "src" / "lib.rs").write_text(PROBE_CRATE, encoding="utf-8")
            (root / "Cargo.toml").write_text('[package]\nname = "probe-fixture"\nversion = "0.0.0"\nedition = "2021"\n[workspace]\n', encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=d, check=True)
            subprocess.run(["git", "add", "-A"], cwd=d, check=True)
            sites = g.parse_sites(subprocess.run([SCAN, "--comparisons", "src"], cwd=d, capture_output=True,
                                                 text=True, check=True).stdout)
            env = os.environ.copy()
            os.environ["CARGO_TARGET_DIR"] = str(root / "target")
            try:
                types, wrapped, problems = g.probe(root, sites, configs)
            finally:
                os.environ.clear()
                os.environ.update(env)
            return sites, types, wrapped, problems

    def test_the_probe_types_each_operand(self):
        sites, types, wrapped, problems = self.run_probe([[]])
        self.assertEqual(problems, [])
        self.assertEqual(wrapped, 5)  # the four functions and the impl's own partial_cmp
        got = {s["fn"]: types[s["id"]] for s in sites}
        self.assertEqual(got, {"a": {"f32"}, "b": {"&Fix"}, "c": {"T"}, "d": {"f64"}, "partial_cmp": {"i64"}})

    def test_a_configuration_that_types_nothing_is_a_problem(self):
        _, _, _, problems = self.run_probe([["--features", "no-such-feature"]])
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("typed no comparison", problems[0])


def site(i, ops, fn="f", file="src/a.rs"):
    return {"id": i, "file": file, "fn": fn, "kind": "bin", "ops": ops, "line": i + 1, "span": "1:0-1:1", "ctx": ""}


class Judge(unittest.TestCase):
    SITES = [site(0, "x < y"), site(1, "x < y"), site(2, "i < n")]
    TYPES = {0: {"f32"}, 1: {"f32"}, 2: {"usize"}}
    ROWS = {("src/a.rs", "f", "x < y"): (2, "f32", "predicate", "an if condition: picks a branch")}

    def judge(self, sites=None, types=None, rows=None, selections=frozenset()):
        p, _, _ = g.judge(self.SITES if sites is None else sites, self.TYPES if types is None else types,
                          self.ROWS if rows is None else rows, set(selections))
        return p

    def test_a_reviewed_tree_passes(self):
        self.assertEqual(self.judge(), [])

    def test_a_float_comparison_without_a_row(self):
        p = self.judge(rows={})
        self.assertEqual(len(p), 1)
        self.assertIn("without a row", p[0])

    def test_a_stale_row(self):
        rows = dict(self.ROWS)
        rows[("src/a.rs", "f", "gone < x")] = (1, "f32", "predicate", "an if condition: picks a branch")
        self.assertTrue(any("stale" in x for x in self.judge(rows=rows)))

    def test_a_count_change(self):
        self.assertTrue(any("counts 2" in x for x in self.judge(sites=self.SITES[:1] + self.SITES[2:])))

    def test_a_misclassified_type(self):
        rows = {k: (2, "f64", c, r) for k, (_, _, c, r) in self.ROWS.items()}
        self.assertTrue(any("misclassified" in x for x in self.judge(rows=rows)))

    def test_a_predicate_on_a_selection(self):
        p = self.judge(selections={("src/a.rs", "f", "x < y")})
        self.assertTrue(any("finds a selection" in x for x in p), p)

    def test_a_row_on_a_non_float_comparison(self):
        rows = dict(self.ROWS)
        rows[("src/a.rs", "f", "i < n")] = (1, "usize", "predicate", "an if condition: picks a branch")
        self.assertTrue(any("non-float" in x for x in self.judge(rows=rows)))

    def test_an_unprobed_comparison_needs_a_row(self):
        types = {0: {"f32"}, 1: {"f32"}}
        p = self.judge(types=types)
        self.assertTrue(any("i < n" in x and "unprobed" in x for x in p), p)

    def test_no_comparison_found(self):
        self.assertIn("compared nothing", self.judge(sites=[])[0])

    def test_skipping_the_probe_flags_every_float_row(self):
        p = self.judge(types={})
        self.assertTrue(p)


class Table(unittest.TestCase):
    def load(self, row):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "t.tsv"
            t.write_text("# h\n" + row + "\n", encoding="utf-8")
            return g.load_table(t)[1]

    def test_bad_rows_are_refused(self):
        self.assertTrue(self.load("a\tb\tc\t1\tf32\tbranch\tsome reason here"))
        self.assertTrue(self.load("a\tb\tc\t1\tf32\tpredicate\tshort"))
        self.assertTrue(self.load("a\tb\tc\t1\tf32\tpredicate"))
        self.assertFalse(self.load("a\tb\tc\t1\tf32\tpredicate\tan if condition: picks a branch"))


if __name__ == "__main__":
    unittest.main()
