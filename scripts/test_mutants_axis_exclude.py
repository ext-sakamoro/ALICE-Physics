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
