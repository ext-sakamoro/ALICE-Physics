#!/usr/bin/env python3
"""Tests for scripts/allow_reason_guard.py: reasons are required and judged,
production allows are counted, a wide allow over a clock read fails, the
lexer reads comments, strings, char literals and lifetimes, and no file
compares nothing."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allow_reason_guard as g  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOOD = 'reason = "analytic reference: f64 closed form of the damped oscillator"'


def tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="allow-reason-guard-"))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def problems(files: dict[str, str], production=None) -> list[str]:
    return g.check(tree(files), {} if production is None else production)[0]


class Reasons(unittest.TestCase):
    def test_the_repository_passes(self):
        found, files, allows = g.check(ROOT)
        self.assertEqual(found, [])
        self.assertGreater(files, 0)
        self.assertGreater(allows, 0)

    def test_a_good_reason_passes(self):
        src = f"#![allow(clippy::disallowed_methods, {GOOD})]\nfn f() {{ 1.0f64.sin(); }}\n"
        self.assertEqual(problems({"tests/a.rs": src}), [])

    def test_bad_reasons_fail(self):
        for attr in [
            "#![allow(clippy::disallowed_methods)]",
            '#![allow(clippy::disallowed_methods, reason = "x")]',
            '#![allow(clippy::disallowed_methods, reason = "")]',
            '#![allow(clippy::disallowed_methods, reason = "the oracle uses f64 libm here")]',
            '#![allow(clippy::disallowed_methods, reason = "analytic reference: todo")]',
            '#![allow(clippy::disallowed_methods, reason = "analytic reference: oracle")]',
            '#![allow(clippy::disallowed_methods, reason = "timing harness: n/a")]',
            '#![allow(clippy::disallowed_methods, reason = "test input: crowd size")]',
            '#[allow(\n    clippy::disallowed_methods,\n    reason = "short"\n)]',
        ]:
            with self.subTest(attr=attr):
                self.assertEqual(len(problems({"tests/a.rs": attr + "\nfn f() {}\n"})), 1)

    def test_other_lints_are_not_judged(self):
        self.assertEqual(problems({"tests/a.rs": "#![allow(clippy::disallowed_types)]\nfn f() {}\n"}), [])

    def test_a_reason_in_a_comment_does_not_count(self):
        src = f"#![allow(clippy::disallowed_methods)] // {GOOD}\nfn f() {{}}\n"
        self.assertEqual(len(problems({"tests/a.rs": src})), 1)


class Clock(unittest.TestCase):
    def test_a_file_allow_over_a_clock_read_fails(self):
        src = f"#![allow(clippy::disallowed_methods, {GOOD})]\nfn f() {{ let t = std::time::Instant::now(); }}\n"
        found = problems({"tests/a.rs": src})
        self.assertEqual(len(found), 1)
        self.assertIn("clock read", found[0])

    def test_a_fn_or_module_allow_over_a_clock_read_fails(self):
        for item in ["fn f()", "mod m", "impl S"]:
            src = f"#[allow(clippy::disallowed_methods, {GOOD})]\n{item} {{ let d = t.elapsed(); }}\n"
            with self.subTest(item=item):
                self.assertEqual(len(problems({"tests/a.rs": src})), 1)

    def test_a_statement_allow_on_the_clock_read_passes(self):
        reason = 'reason = "timing harness: prints the wall time of the solve"'
        src = f"fn f() {{\n    #[allow(clippy::disallowed_methods, {reason})]\n    let t = std::time::Instant::now();\n}}\n"
        self.assertEqual(problems({"examples/a.rs": src}), [])

    def test_a_clock_word_in_a_string_or_comment_is_not_a_read(self):
        src = (f"#![allow(clippy::disallowed_methods, {GOOD})]\n"
               '// Instant::now() is not called here\nconst S: &str = "x.elapsed()";\nfn f() {}\n')
        self.assertEqual(problems({"tests/a.rs": src}), [])


class Production(unittest.TestCase):
    def test_an_allow_outside_cfg_test_in_src_fails(self):
        src = f"#[allow(clippy::disallowed_methods, {GOOD})]\nfn f() {{}}\n"
        found = problems({"src/a.rs": src})
        self.assertEqual(len(found), 1)
        self.assertIn("production", found[0])

    def test_an_allow_inside_a_cfg_test_module_passes(self):
        src = f'#[cfg(test)]\n// a "quote" in a comment\nmod tests {{\n    #[allow(clippy::disallowed_methods, {GOOD})]\n    fn f() {{}}\n}}\n'
        self.assertEqual(problems({"src/a.rs": src}), [])

    def test_the_production_list_is_counted(self):
        src = f"#[allow(clippy::disallowed_methods, {GOOD})]\nfn f() {{}}\n"
        self.assertEqual(problems({"src/a.rs": src}, {"src/a.rs": (1, "listed")}), [])
        self.assertEqual(len(problems({"src/a.rs": src}, {"src/a.rs": (2, "listed")})), 1)
        self.assertEqual(len(problems({"src/a.rs": "fn f() {}\n"}, {"src/a.rs": (1, "listed")})), 1)
        self.assertEqual(len(problems({"src/a.rs": src}, {"src/a.rs": (1, " ")})), 1)


class Lexer(unittest.TestCase):
    def test_comments_strings_chars_and_lifetimes(self):
        src = ('// a "quote\nlet s = "keep // not a comment"; let c = \'"\'; fn f<\'a>(x: &\'a str) {}\n'
               '/* nested /* block */ "x" */ let r = r#"raw "x" str"#;\n#[cfg(test)]\nmod tests {}\n')
        code = g.blank_all(src)
        self.assertIn("#[cfg(test)]", code)
        self.assertIn("fn f<'a>(x: &'a str)", code)
        self.assertNotIn("keep", code)
        self.assertNotIn("raw", code)
        kept = g.blank_comments(src)
        self.assertIn('"keep // not a comment"', kept)
        self.assertNotIn("quote", kept)


class Nothing(unittest.TestCase):
    def test_no_file_compares_nothing(self):
        root = tree({"README.md": "x"})
        self.assertEqual(g.check(root, {})[1], 0)


if __name__ == "__main__":
    unittest.main()
