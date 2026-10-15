#!/usr/bin/env python3
"""Tests for scripts/allow_reason_guard.py: the manifest keeps the reason lint
at deny and sets no group, reasons are judged, lint groups are refused,
production allows of the ban are counted, a wide allow over a clock read
fails, the canary judgement needs every form flagged, and nothing compared
fails."""
import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allow_reason_guard as g  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GOOD = 'reason = "analytic reference: f64 closed form of the damped oscillator"'
PLAIN = 'reason = "the helper is only read by the std build of the module"'
DENY = '[lints.clippy]\nallow_attributes_without_reason = "deny"\n'


def tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="allow-reason-guard-"))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def problems(files: dict[str, str], production=None) -> list[str]:
    files = {"Cargo.toml": DENY, **files}
    return g.check(tree(files), {} if production is None else production)[0]


class Repository(unittest.TestCase):
    def test_the_repository_passes(self):
        found, attrs = g.check(ROOT)
        self.assertEqual(found, [])
        self.assertGreater(attrs, 0)


class Manifest(unittest.TestCase):
    def test_deny_passes(self):
        self.assertEqual(g.check_manifest(tomllib.loads(DENY)), [])

    def test_a_lower_or_missing_level_fails(self):
        for text in ['[lints.clippy]\nallow_attributes_without_reason = "warn"\n',
                     '[lints.clippy]\nallow_attributes_without_reason = { level = "allow" }\n',
                     '[lints.rust]\nunsafe_code = "warn"\n', ""]:
            with self.subTest(text=text):
                self.assertEqual(len(g.check_manifest(tomllib.loads(text))), 1)

    def test_a_group_in_the_manifest_fails(self):
        for extra in ['all = "allow"', 'style = "warn"', 'pedantic = { level = "warn", priority = -1 }']:
            with self.subTest(extra=extra):
                found = g.check_manifest(tomllib.loads(DENY + extra + "\n"))
                self.assertEqual(len(found), 1, found)
                self.assertIn("group", found[0])
        found = g.check_manifest(tomllib.loads(DENY + '[lints.rust]\nwarnings = "allow"\n'))
        self.assertEqual(len(found), 1, found)

    def test_the_lints_tables_are_copied_verbatim(self):
        text = ('[package]\nname = "x"\n\n[lints.rust]\nunexpected_cfgs = { level = "warn", '
                'check-cfg = ["cfg(loom)"] }\n\n' + DENY + '\n[dependencies]\nfoo = "1"\n')
        out = g.lints_tables(text)
        self.assertIn('check-cfg = ["cfg(loom)"]', out)
        self.assertIn('allow_attributes_without_reason = "deny"', out)
        self.assertNotIn("foo", out)
        self.assertNotIn("[package]", out)
        tomllib.loads(out)


class Reasons(unittest.TestCase):
    def test_good_reasons_pass(self):
        for attr in [f"#![allow(clippy::disallowed_methods, {GOOD})]", f"#[allow(dead_code, {PLAIN})]",
                     f"#[cfg_attr(not(feature = \"std\"), allow(dead_code, {PLAIN}))]",
                     f"#[expect(unused_mut, {PLAIN})]"]:
            with self.subTest(attr=attr):
                self.assertEqual(problems({"tests/a.rs": attr + "\nfn f() {}\n"}), [])

    def test_bad_reasons_fail(self):
        for attr in [
            "#[cfg_attr(not(feature = \"std\"), allow(dead_code))]",  # off here, so clippy never saw it
            '#[allow(dead_code, reason = "short")]',
            '#[allow(dead_code, reason = "dead_code")]',
            '#[allow(clippy::too_many_arguments, reason = "too_many_arguments")]',
            # long enough, so only the lint-name rule can refuse it
            '#[allow(clippy::needless_pass_by_ref_mut, reason = "clippy::needless_pass_by_ref_mut")]',
            '#[allow(dead_code, reason = "                          todo")]',
            '#![allow(clippy::disallowed_methods, reason = "the oracle uses f64 libm here")]',
            '#![allow(clippy::disallowed_methods, reason = "analytic reference: todo")]',
            # long enough, so only the kind rule can refuse it
            '#![allow(clippy::disallowed_methods, reason = "the f64 libm values here are the reference")]',
            # long enough with the kind, so only the placeholder rule can refuse it
            '#![allow(clippy::disallowed_methods, reason = "analytic reference:            oracle")]',
            '#[allow(\n    clippy::disallowed_methods,\n    reason = "short"\n)]',
        ]:
            with self.subTest(attr=attr):
                self.assertEqual(len(problems({"tests/a.rs": attr + "\nfn f() {}\n"})), 1)

    def test_a_reason_in_a_comment_does_not_count(self):
        src = f"#[allow(dead_code)] // {PLAIN}\nfn f() {{}}\n"
        self.assertEqual(len(problems({"tests/a.rs": src})), 1)

    def test_a_reason_line_in_a_string_is_not_an_attribute(self):
        src = 'const S: &str = "#[allow(dead_code)]";\nfn f() {}\n'
        self.assertEqual(problems({"tests/a.rs": src}), [])


class Groups(unittest.TestCase):
    def test_a_lint_group_is_refused_whatever_the_reason(self):
        for attr in [f"#[allow(clippy::style, {PLAIN})]", f"#[allow(clippy::all, {PLAIN})]",
                     f"#![allow(warnings, {PLAIN})]", f"#[expect(clippy::pedantic, {PLAIN})]",
                     f"#[cfg_attr(test, allow(clippy::all, {PLAIN}))]", f"#[allow(clippy :: style, {PLAIN})]"]:
            with self.subTest(attr=attr):
                found = problems({"tests/a.rs": attr + "\nfn f() {}\n"})
                self.assertEqual(len(found), 1, found)
                self.assertIn("also silences", found[0])


class Production(unittest.TestCase):
    def test_an_allow_of_the_ban_outside_cfg_test_in_src_fails(self):
        src = f"#[allow(clippy::disallowed_methods, {GOOD})]\nfn f() {{}}\n"
        found = problems({"src/a.rs": src})
        self.assertEqual(len(found), 1)
        self.assertIn("production", found[0])

    def test_other_lints_in_src_are_not_production_allows(self):
        self.assertEqual(problems({"src/a.rs": f"#[allow(dead_code, {PLAIN})]\nfn f() {{}}\n"}), [])

    def test_an_allow_inside_a_cfg_test_module_passes(self):
        src = f'#[cfg(test)]\n// a "quote" in a comment\nmod tests {{\n    #[allow(clippy::disallowed_methods, {GOOD})]\n    fn f() {{}}\n}}\n'
        self.assertEqual(problems({"src/a.rs": src}), [])

    def test_the_production_list_is_counted(self):
        src = f"#[allow(clippy::disallowed_methods, {GOOD})]\nfn f() {{}}\n"
        self.assertEqual(problems({"src/a.rs": src}, {"src/a.rs": (1, "listed")}), [])
        self.assertEqual(len(problems({"src/a.rs": src}, {"src/a.rs": (2, "listed")})), 1)
        self.assertEqual(len(problems({"src/a.rs": "fn f() {}\n"}, {"src/a.rs": (1, "listed")})), 1)
        self.assertEqual(len(problems({"src/a.rs": src}, {"src/a.rs": (1, " ")})), 1)


def clock(src: str, line: int) -> list[str]:
    root = tree({"tests/a.rs": src})
    return g.clock_problems(g.attributes(root), [("tests/a.rs", line, "std::time::Instant::now")])


class Clock(unittest.TestCase):
    def test_a_file_allow_over_a_clock_read_fails(self):
        src = f"#![allow(clippy::disallowed_methods, {GOOD})]\nfn f() {{\n    let t = Clk::now();\n}}\n"
        found = clock(src, 3)
        self.assertEqual(len(found), 1)
        self.assertIn("clock read", found[0])

    def test_a_fn_or_module_or_impl_allow_over_a_clock_read_fails(self):
        for item in ["fn f()", "mod m", "impl S"]:
            src = f"#[allow(clippy::disallowed_methods, {GOOD})]\n{item} {{\n    let d = t.elapsed();\n}}\n"
            with self.subTest(item=item):
                self.assertEqual(len(clock(src, 3)), 1)

    def test_an_inner_allow_in_a_function_body_over_a_clock_read_fails(self):
        src = f"fn f() {{\n    #![allow(clippy::disallowed_methods, {GOOD})]\n    let t = Clk::now();\n}}\n"
        self.assertEqual(len(clock(src, 3)), 1)

    def test_a_statement_allow_on_the_clock_read_passes(self):
        reason = 'reason = "timing harness: prints the wall time of the solve"'
        src = f"fn f() {{\n    #[allow(clippy::disallowed_methods, {reason})]\n    let t = std::time::Instant::now();\n}}\n"
        self.assertEqual(clock(src, 3), [])

    def test_a_wide_allow_of_another_lint_or_another_item_passes(self):
        src = (f"#[allow(dead_code, {PLAIN})]\nfn f() {{\n    let t = now();\n}}\n"
               f"#[allow(clippy::disallowed_methods, {GOOD})]\nfn g() {{}}\n")
        self.assertEqual(clock(src, 3), [])

    def test_clock_reads_are_read_from_clippy_messages(self):
        def msg(code, text, file, line):
            return {"reason": "compiler-message", "message": {
                "code": {"code": code}, "message": text,
                "spans": [{"is_primary": True, "file_name": file, "line_start": line}]}}
        hits = g.clock_hits([
            msg("clippy::disallowed_methods", "use of a disallowed method `std::time::Instant::now`", "tests/a.rs", 3),
            msg("clippy::disallowed_methods", "use of a disallowed method `std::time::Instant::elapsed`", "tests/a.rs", 9),
            msg("clippy::disallowed_methods", "use of a disallowed method `f64::sin`", "tests/a.rs", 4),
            msg("dead_code", "function `std::time::Instant::now` is never used", "tests/a.rs", 5),
            msg("clippy::disallowed_methods", "use of a disallowed method `std::time::SystemTime::now`",
                "/rustc/library/std/src/x.rs", 6),
            {"reason": "build-finished", "success": True},
        ])
        self.assertEqual(hits, [("tests/a.rs", 3, "std::time::Instant::now"),
                                ("tests/a.rs", 9, "std::time::Instant::elapsed")])


class Canary(unittest.TestCase):
    def flagged(self, lines):
        return [{"reason": "compiler-message", "message": {
            "code": {"code": "clippy::allow_attributes_without_reason"},
            "spans": [{"is_primary": True, "line_start": n}]}} for n in lines]

    def lines(self, forms):
        src = g.canary_source().split("\n")
        return [src.index(f) + 1 for f in forms]

    def test_every_form_flagged_and_no_clean_one_passes(self):
        self.assertEqual(g.judge_canary(101, self.flagged(self.lines(g.CANARY_FORMS)), g.canary_source()), [])

    def test_a_form_without_a_diagnostic_fails(self):
        for k in range(len(g.CANARY_FORMS)):
            want = self.lines(g.CANARY_FORMS)
            del want[k]
            with self.subTest(form=g.CANARY_FORMS[k]):
                self.assertEqual(len(g.judge_canary(101, self.flagged(want), g.canary_source())), 1)

    def test_a_flagged_clean_line_fails(self):
        msgs = self.flagged(self.lines(g.CANARY_FORMS) + self.lines(g.CANARY_CLEAN[:1]))
        self.assertEqual(len(g.judge_canary(101, msgs, g.canary_source())), 1)

    def test_exit_zero_or_no_diagnostic_fails(self):
        self.assertIn("exited 0", g.judge_canary(0, self.flagged(self.lines(g.CANARY_FORMS)),
                                                  g.canary_source())[0])
        found = g.judge_canary(101, [], g.canary_source())
        self.assertTrue(any("compared nothing" in p for p in found), found)


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
    def test_no_attribute_compares_nothing(self):
        root = tree({"Cargo.toml": DENY, "README.md": "x"})
        self.assertEqual(g.check(root, {})[1], 0)


if __name__ == "__main__":
    unittest.main()
