#!/usr/bin/env python3
"""Tests for scripts/no_private_minmax.py: the repository passes, every new
function named max / min / max64 / min64 fails whatever its form, the
reviewed list is matched exactly (count drift, stale entry, missing reason),
a copy inside a comment or a string is not counted, and a tree without the
re-export or without files fails."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_private_minmax as g  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REEXPORT = "pub mod det_math {\n    pub(crate) use alice_det_math::{max, max64, min, min64};\n}\n"
ALLOW = {("src/ok.rs", "min"): (1, "a reviewed other law")}


def tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="no-private-minmax-"))
    # removed after this module's tests (the suite fails on anything left, see main)
    unittest.addModuleCleanup(shutil.rmtree, root, True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def problems(extra: str, allowed=None) -> list[str]:
    files = {"src/lib.rs": REEXPORT, "src/x.rs": extra, "src/ok.rs": "fn min(a: Fix128, b: Fix128) -> Fix128 { a }"}
    return g.check(tree(files), ALLOW if allowed is None else allowed)[0]


class NoPrivateMinmax(unittest.TestCase):
    def test_the_repository_passes(self):
        found, files = g.check(ROOT)
        self.assertEqual(found, [])
        self.assertGreater(files, 0)

    def test_every_new_function_of_these_names_fails(self):
        forms = [
            "pub(crate) fn max(a: f32, b: f32) -> f32 { if a > b { a } else { b } }",
            "fn min(x: f32, y: f32) -> f32 { x }",
            "pub fn max64(a: f64, b: f64) -> f64 { a }",
            "fn min64 ( mut a : f64 , b : f64 ) -> f64 { a }",
            "fn max(a: &f32, b: &f32) -> f32 { *a }",
            "pub fn max<T: Into<f32>>(a: f32, b: f32) -> f32 { a }",
            "fn r#min(a: f32, b: f32) -> f32 { a }",
            "fn max64<A: Into<Vec<f64>>, B>(a: f64, b: f64) -> f64 { a }",
            "fn min(a: core::primitive::f32, b: f32) -> f32 { a }",
            "type Ff = f32;\nfn min(a: Ff, b: Ff) -> Ff { a }",
            "fn max<F: Fn(f32) -> f32>(a: f32, b: f32) -> f32 { a }",
            "fn max(#[allow(unused)] a: f32, b: f32) -> f32 { b }",
            "impl Num { fn max(a: f32, b: f32) -> f32 { a } }",
            "fn max64<T>(a: T, b: T) -> T where T: PartialOrd { a }",
            "trait Mx { fn max(self, b: Self) -> Self; }\nimpl Mx for f32 { fn max(self, b: f32) -> f32 { b } }",
            "fn max(a: f32, b: f32, _p: core::marker::PhantomData<Fix128>) -> f32 { a }",
            "impl S { const fn min(&self) -> f64 { 0.0 } }",
            "mod m {\n    #[inline]\n    pub(crate)\n    fn\n    min64(\n        a: f64,\n        b: f64,\n    ) -> f64 { a }\n}",
        ]
        for src in forms:
            with self.subTest(src=src):
                self.assertTrue(problems(src), "a new function of these names passed")

    def test_other_names_pass(self):
        for src in [
            "fn maximum(a: f32, b: f32) -> f32 { a }",
            "fn max_f32(a: f32, b: f32) -> f32 { a }",
            "fn minmax(a: f32) -> f32 { a }",
            "let max = 3;",
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

    def test_a_second_function_in_an_allowed_file_fails(self):
        files = {
            "src/lib.rs": REEXPORT,
            "src/ok.rs": "fn min(a: Fix128, b: Fix128) -> Fix128 { a }\nfn min(a: f32, b: f32) -> f32 { a }",
        }
        found, _ = g.check(tree(files), ALLOW)
        self.assertEqual(len(found), 1)
        self.assertIn("2 found, ALLOWED says 1", found[0])

    def test_another_name_in_an_allowed_file_fails(self):
        # the entry is for `min` in that file; a `max` beside it is not covered
        files = {
            "src/lib.rs": REEXPORT,
            "src/ok.rs": "fn min(a: Fix128, b: Fix128) -> Fix128 { a }\nfn max(a: f32, b: f32) -> f32 { a }",
        }
        found, _ = g.check(tree(files), ALLOW)
        self.assertEqual(len(found), 1)
        self.assertIn("`fn max`", found[0])

    def test_a_stale_entry_fails(self):
        files = {"src/lib.rs": REEXPORT, "src/ok.rs": "fn other() {}"}
        found, _ = g.check(tree(files), ALLOW)
        self.assertEqual(len(found), 1)
        self.assertIn("matches nothing", found[0])

    def test_an_entry_without_a_reason_fails(self):
        found = problems("", {("src/ok.rs", "min"): (1, "  ")})
        self.assertEqual(len(found), 1)
        self.assertIn("no reason", found[0])

    def test_the_reexport_is_read_as_a_set(self):
        lib = "pub mod det_math {\n    pub(crate) use alice_det_math::{min64, max, min, max64};\n}\n"
        self.assertEqual(g.check(tree({"src/lib.rs": lib}), {})[0], [])
        partial = "pub mod det_math {\n    pub(crate) use alice_det_math::{max, min, max64};\n}\n"
        self.assertEqual(len(g.check(tree({"src/lib.rs": partial}), {})[0]), 1)

    def test_a_missing_reexport_fails(self):
        found, _ = g.check(tree({"src/lib.rs": "pub mod det_math {}\n"}), {})
        self.assertEqual(len(found), 1)
        self.assertIn("re-export", found[0])

    def test_no_file_compares_nothing(self):
        root = tree({"README.md": "x"})
        (root / "src").mkdir()
        _, files = g.check(root, {})
        self.assertEqual(files, 0)


def main() -> int:
    """the suite inside a temporary directory of its own (also the $TMPDIR of what it runs): a test
    that leaves a file or directory behind fails the run, naming it"""
    scratch = Path(tempfile.mkdtemp(prefix="test_no_private_minmax-tests-"))
    old_tempdir, old_env = tempfile.tempdir, os.environ.get("TMPDIR")
    tempfile.tempdir = str(scratch)
    os.environ["TMPDIR"] = str(scratch)
    try:
        ok = unittest.main(exit=False).result.wasSuccessful()
        left = sorted(p.name for p in scratch.iterdir())
    finally:
        tempfile.tempdir = old_tempdir
        if old_env is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_env
        shutil.rmtree(scratch, ignore_errors=True)
    if left:
        print(f"test_no_private_minmax: the tests left {len(left)} entries in the temporary directory: {', '.join(left[:10])}",
              file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
