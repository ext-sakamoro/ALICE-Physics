#!/usr/bin/env python3
"""Tests for scripts/mutants_axis_exclude.py: exact line sets, and failure on
anything it cannot read."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_axis_exclude as ax  # noqa: E402

DEFAULT = {"std"}
PARALLEL = {"std", "parallel", "rayon", "gpu-solver-bridge", "simd"}

SRC = """\
pub fn always() -> u32 {
    1
}

#[cfg(feature = "parallel")]
pub fn batched(
    a: u32,
    b: u32,
) -> u32 {
    a + b
}

#[cfg(not(feature = "parallel"))]
fn sequential() -> u32 {
    2
}

fn mixed(x: u32) -> u32 {
    #[cfg(feature = "parallel")]
    let y = x * 2;
    #[cfg(not(feature = "parallel"))]
    let y = x;
    match y {
        #[cfg(feature = "gpu-solver-bridge")]
        0 => 9,
        _ => y,
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl Lanes {
    fn width() -> usize {
        4
    }
}

#[cfg(not(feature = "std"))]
fn no_std_only() {}

#[cfg(feature = "parallel")]
mod inner {
    #[cfg(feature = "simd")]
    fn deep() {}
}
"""


class Regions(unittest.TestCase):
    def test_items_statements_and_arms_are_delimited_exactly(self):
        got = [(a, b) for a, b, _ in ax.regions(SRC)]
        self.assertEqual(got, [(6, 11), (14, 16), (20, 20), (22, 22), (25, 25),
                               (31, 35), (38, 38), (41, 44), (43, 43)])

    def test_default_axis_leaves_the_feature_lines_to_the_other_axis(self):
        moved = ax.moved_lines(SRC, DEFAULT, [PARALLEL])
        expected = [6, 7, 8, 9, 10, 11, 20, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44]
        self.assertEqual(moved, expected)

    def test_parallel_axis_leaves_the_not_parallel_lines_to_the_default_axis(self):
        self.assertEqual(ax.moved_lines(SRC, PARALLEL, [DEFAULT]), [14, 15, 16, 22])

    def test_lines_no_axis_compiles_are_not_moved(self):
        # `not(feature = "std")` is compiled by neither axis
        for moved in (ax.moved_lines(SRC, DEFAULT, [PARALLEL]),
                      ax.moved_lines(SRC, PARALLEL, [DEFAULT])):
            self.assertNotIn(38, moved)

    def test_nested_regions_need_every_enclosing_predicate(self):
        # `deep` needs parallel (outer) and simd (inner)
        only_parallel = {"std", "parallel"}
        self.assertNotIn(43, ax.moved_lines(SRC, DEFAULT, [only_parallel]))
        self.assertIn(42, ax.moved_lines(SRC, DEFAULT, [only_parallel]))


class WhereClause(unittest.TestCase):
    SRC = """\
#[cfg(feature = "parallel")]
pub(crate) fn dispatch<T, F>(items: &[T], f: F) -> Vec<T>
where
    F: Fn(&T) -> T + Send,
    T: Send,
{
    items.iter().map(f).collect()
}

fn after() {}
"""

    def test_a_where_clause_does_not_end_the_item(self):
        # the commas of a where clause come before the body opens
        self.assertEqual([(a, b) for a, b, _ in ax.regions(self.SRC)], [(2, 8)])
        self.assertEqual(ax.moved_lines(self.SRC, DEFAULT, [PARALLEL]), list(range(2, 9)))


def marked(src: str, tag: str) -> int:
    """The 1-based line carrying the marker comment `// <tag>`."""
    for n, line in enumerate(src.split("\n"), 1):
        if line.rstrip().endswith(f"// {tag}"):
            return n
    raise AssertionError(tag)


def span(src: str, a: str, b: str) -> list[int]:
    return list(range(marked(src, a), marked(src, b) + 1))


class Lexing(unittest.TestCase):
    """Braces in strings, chars, raw strings and comments do not delimit; `else`
    keeps an `if` going; attributes may share a line or span several."""

    SRC = """\
#[cfg(feature = "parallel")]
fn tricky() -> &'static str { // t0
    let a = "}"; let b = '}'; let c = r#"}"#; /* } */ // {
    let _ = '\\u{7D}';
    if a.len() > 0 { a } else { b } // t1
} // t2

#[cfg(feature = "parallel")]
if ready { go() } // e0
else { wait() } // e1

impl Holder {
    #[cfg(feature = "parallel")]
    fn braces(&self) -> &str { "{" } // h0
    fn sibling(&self) {} // h1
}

#[cfg(feature = "parallel")] fn same_line() {} // s0

#[cfg(all(
    feature = "simd",
    target_arch = "x86_64"
))]
#[inline] // m-attr
fn multi() {} // m0

#[cfg_attr(feature = "parallel", inline)]
fn attr_only() {} // a0

#[cfg(feature = "std")]
fn everywhere() {} // b0

#[cfg(feature = "parallel")]
mod outer { // n0
    #[cfg(feature = "std")]
    fn inner() {} // n1
} // n2

#[cfg(any(feature = "parallel", feature = "nope"))]
fn either() {} // y0

fn lifetimes<'a>(x: &'a str) -> &'a str { x } // l0
"""

    def moved(self, axis=DEFAULT, other=PARALLEL):
        return ax.moved_lines(self.SRC, axis, [other])

    def test_braces_in_literals_and_comments_do_not_delimit(self):
        self.assertEqual([n for n in self.moved() if marked(self.SRC, "t0") <= n <= marked(self.SRC, "t2")],
                         span(self.SRC, "t0", "t2"))

    def test_an_else_branch_stays_in_the_region(self):
        self.assertIn(marked(self.SRC, "e1"), self.moved())

    def test_a_brace_in_a_string_does_not_swallow_the_sibling(self):
        self.assertIn(marked(self.SRC, "h0"), self.moved())
        self.assertNotIn(marked(self.SRC, "h1"), self.moved())

    def test_an_attribute_on_the_item_line(self):
        self.assertIn(marked(self.SRC, "s0"), self.moved())

    def test_a_multi_line_attribute_and_attributes_in_between(self):
        moved = self.moved()
        self.assertIn(marked(self.SRC, "m0"), moved)
        self.assertNotIn(marked(self.SRC, "m-attr"), moved)

    def test_cfg_attr_removes_no_code(self):
        self.assertNotIn(marked(self.SRC, "a0"), self.moved())

    def test_a_region_every_axis_compiles_is_not_moved(self):
        self.assertNotIn(marked(self.SRC, "b0"), self.moved())
        self.assertNotIn(marked(self.SRC, "b0"), self.moved(PARALLEL, DEFAULT))

    def test_an_inner_region_needs_the_outer_one_too(self):
        # `inner` is std (default compiles) inside parallel (default does not)
        self.assertIn(marked(self.SRC, "n1"), self.moved())

    def test_any_needs_one_feature(self):
        self.assertIn(marked(self.SRC, "y0"), self.moved())

    def test_a_feature_name_matches_whole(self):
        self.assertFalse(ax.evaluate('feature = "parallel"', {"parallel_extra", "std"}))

    def test_lifetimes_are_not_char_literals(self):
        self.assertNotIn(marked(self.SRC, "l0"), self.moved())

    def test_an_inner_attribute_covers_the_rest_of_the_file(self):
        src = '#![cfg(feature = "parallel")]\nfn a() {}\nfn b() {}\n'
        self.assertEqual(ax.moved_lines(src, DEFAULT, [PARALLEL]), [2, 3])

    def test_an_unclosed_attribute_is_an_error(self):
        with self.assertRaises(ax.Unknown):
            ax.regions('#[cfg(all(feature = "parallel")]\nfn a() {}\n')


class Evaluate(unittest.TestCase):
    def test_predicates(self):
        f = {"std", "simd"}
        self.assertTrue(ax.evaluate('feature = "std"', f))
        self.assertFalse(ax.evaluate('feature = "parallel"', f))
        self.assertTrue(ax.evaluate('all(feature = "simd", target_arch = "x86_64")', f))
        self.assertFalse(ax.evaluate('all(feature = "simd", target_arch = "aarch64")', f))
        self.assertFalse(ax.evaluate('target_feature = "avx2"', f))
        self.assertTrue(ax.evaluate('not(target_feature = "avx2")', f))
        self.assertTrue(ax.evaluate('any(feature = "x", test)', f))
        self.assertFalse(ax.evaluate("loom", f))
        self.assertTrue(ax.evaluate("debug_assertions", f))

    def test_an_unknown_predicate_is_an_error(self):
        with self.assertRaises(ax.Unknown):
            ax.evaluate('panic = "abort"', {"std"})
        with self.assertRaises(ax.Unknown):
            ax.evaluate("miri", {"std"})

    def test_features_expand_through_the_feature_table(self):
        table = {"default": ["std"], "std": ["dep/std"], "parallel": ["rayon"],
                 "rayon": [], "gpu-solver-bridge": ["std"], "simd": []}
        self.assertEqual(ax.expand_features("", table), {"std"})
        self.assertEqual(ax.expand_features("gpu-solver-bridge", table), {"gpu-solver-bridge", "std"})
        self.assertEqual(ax.expand_features("parallel", table), {"parallel", "rayon"})


class Failures(unittest.TestCase):
    def test_a_cfg_with_no_item_is_an_error(self):
        with self.assertRaises(ax.Unknown):
            ax.regions('fn a() {}\n#[cfg(feature = "parallel")]\n')

    def test_an_item_that_never_ends_is_an_error(self):
        with self.assertRaises(ax.Unknown):
            ax.regions('#[cfg(feature = "parallel")]\nfn a() {\n    1\n')

    def test_files_without_any_cfg_fail(self):
        root = Path(__file__).resolve().parent.parent
        tmp = root / "target" / "axis-exclude-test"
        tmp.mkdir(parents=True, exist_ok=True)
        f = tmp / "plain.rs"
        f.write_text("fn a() {}\n", encoding="utf-8")
        rel = str(f.relative_to(root))
        self.assertEqual(ax.main(["--axis", "", "--other", "parallel", "--file", rel]), 1)
        self.assertEqual(ax.main(["--axis", "", "--other", "parallel", "--file", rel,
                                  "--allow-no-cfg"]), 0)

    def test_no_other_axis_fails(self):
        self.assertEqual(ax.main(["--axis", "", "--file", "src/solver.rs"]), 1)


class Patterns(unittest.TestCase):
    def test_one_anchored_pattern_per_file(self):
        p = ax.patterns({"src/solver.rs": [4, 12], "src/math.rs": []})
        self.assertEqual(p, ["^src/solver\\.rs:(4|12):"])
        import re
        self.assertTrue(re.search(p[0], "src/solver.rs:12:5: replace + with - in f"))
        self.assertFalse(re.search(p[0], "src/solver.rs:124:5: replace + with - in f"))
        self.assertFalse(re.search(p[0], "src/solver.rs:1:12: replace + with - in f"))


if __name__ == "__main__":
    unittest.main()
