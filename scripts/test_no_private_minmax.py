#!/usr/bin/env python3
"""Tests for scripts/no_private_minmax.py: the repository passes, a local float
max / min copy in any form fails, same-named functions over other types pass,
a copy inside a comment or a string is not counted, and a tree without the
re-export or without files fails."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_private_minmax as g  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REEXPORT = "pub mod det_math {\n    pub(crate) use alice_det_math::{max, max64, min, min64};\n}\n"


def tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="no-private-minmax-"))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def problems(extra: str) -> list[str]:
    return g.check(tree({"src/lib.rs": REEXPORT, "src/x.rs": extra}))[0]


class NoPrivateMinmax(unittest.TestCase):
    def test_the_repository_passes(self):
        found, files = g.check(ROOT)
        self.assertEqual(found, [])
        self.assertGreater(files, 0)

    def test_a_planted_copy_fails_in_every_form(self):
        copies = [
            "pub(crate) fn max(a: f32, b: f32) -> f32 { if a > b { a } else { b } }",
            "fn min(x: f32, y: f32) -> f32 { x }",
            "pub fn max64(a: f64, b: f64) -> f64 { a }",
            "fn min64 ( mut a : f64 , b : f64 ) -> f64 { a }",
            "fn max<T>(a: f32, b: f32) -> f32 { a }",
            "fn max(a: &f32, b: &f32) -> f32 { *a }",
            "fn min64(a: &mut f64, b: f64) -> f64 { *a }",
            "pub fn max<T: Into<f32>>(a: f32, b: f32) -> f32 { a }",
            "fn r#min(a: f32, b: f32) -> f32 { a }",
            "fn max64<A: Into<Vec<f64>>, B>(a: f64, b: f64) -> f64 { a }",
            "fn min(a: core::primitive::f32, b: f32) -> f32 { a }",
            "type Ff = f32;\nfn min(a: Ff, b: Ff) -> Ff { a }",
            "fn max<F: Fn(f32) -> f32>(a: f32, b: f32) -> f32 { a }",
            "fn max(#[allow(unused)] a: f32, b: f32) -> f32 { b }",
            "impl Num { fn max(a: f32, b: f32) -> f32 { a } }",
            "fn max64<T>(a: T, b: T) -> T where T: PartialOrd { a }",
            "mod m {\n    #[inline]\n    pub(crate) fn min(\n        a: f32,\n        b: f32,\n    ) -> f32 { a }\n}",
        ]
        for src in copies:
            with self.subTest(src=src):
                self.assertEqual(len(problems(src)), 1, problems(src))

    def test_other_laws_of_the_same_name_pass(self):
        for src in [
            "impl S { fn max(self, o: Self) -> Self { self } }",
            "impl S { fn min(&mut self, o: f32) -> f32 { o } }",
            "impl<'a> S<'a> { fn max64(&'a self) -> f64 { 0.0 } }",
            "impl S { fn min64(#[allow(unused)] &self) -> f64 { 0.0 } }",
            "fn min(a: Fix128, b: Fix128) -> Fix128 { a }",
            "impl R { pub const fn max(&self) -> f64 { self.0 } }",
            "fn maximum(a: f32, b: f32) -> f32 { a }",
            "fn max_f32(a: f32, b: f32) -> f32 { a }",
        ]:
            with self.subTest(src=src):
                self.assertEqual(problems(src), [])

    def test_a_copy_in_a_comment_or_a_string_is_not_counted(self):
        for src in [
            "// fn max(a: f32, b: f32) -> f32 { a }",
            "/* fn min(a: f32, b: f32) -> f32 { a } */",
            'const S: &str = "fn max64(a: f64, b: f64)";',
            'const R: &str = r#"fn min64(a: f64, b: f64)"#;',
        ]:
            with self.subTest(src=src):
                self.assertEqual(problems(src), [])

    def test_the_reexport_is_read_as_a_set(self):
        lib = "pub mod det_math {\n    pub(crate) use alice_det_math::{min64, max, min, max64};\n}\n"
        self.assertEqual(g.check(tree({"src/lib.rs": lib}))[0], [])
        partial = "pub mod det_math {\n    pub(crate) use alice_det_math::{max, min, max64};\n}\n"
        self.assertEqual(len(g.check(tree({"src/lib.rs": partial}))[0]), 1)

    def test_an_unreadable_parameter_list_fails(self):
        self.assertEqual(len(problems("fn max")), 1)

    def test_a_missing_reexport_fails(self):
        found, _ = g.check(tree({"src/lib.rs": "pub mod det_math {}\n"}))
        self.assertEqual(len(found), 1)
        self.assertIn("re-export", found[0])

    def test_no_file_compares_nothing(self):
        root = tree({"README.md": "x"})
        (root / "src").mkdir()
        found, files = g.check(root)
        self.assertEqual(files, 0)


if __name__ == "__main__":
    unittest.main()
