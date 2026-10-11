#!/usr/bin/env python3
"""Tests for scripts/mutants_skip_check.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_skip_check as sk  # noqa: E402

TEST_MOD = """\
pub fn production() {}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[cfg(feature = "gpu-solver-bridge")]
    #[mutants::skip]
    fn helper() -> u32 {
        let s = "#[mutants::skip] in a string";
        1
    }
}
"""

PRODUCTION = """\
#[mutants::skip]
pub fn production() -> u32 {
    1
}
"""

COMMENT_ONLY = """\
// #[mutants::skip] in a comment is not a skip
/* #[mutants::skip] */
pub fn production() {}
"""

AFTER_TEST_MOD = """\
#[cfg(test)]
mod tests {
    fn t() {}
}

#[mutants::skip]
pub fn after() {}
"""

CFG_ATTR = """\
#[cfg(test)]
mod tests {
    #[cfg_attr(test, mutants::skip)]
    fn helper() {}
}
"""

TEST_ONLY_USE_THEN_ITEM = """\
#[cfg(test)]
use std::fmt;

#[mutants::skip]
pub fn production() {}
"""

INNER_AT_FILE_TOP = """\
#![mutants::skip]
pub fn production() {}
"""

INNER_CFG_ATTR = """\
#![cfg_attr(test, mutants::skip)]
pub fn production() {}
"""

INNER_IN_TEST_FN = """\
#[cfg(test)]
mod tests {
    fn helper() {
        #![mutants::skip]
    }
}
"""

LEADING_COLONS_ON_PRODUCTION = """\
#[::mutants::skip]
pub fn production() {}
"""

LEADING_COLONS_IN_TEST = """\
#[cfg(test)]
mod tests {
    #[::mutants::skip]
    fn helper() {}
}
"""

SPACED_AND_NESTED = """\
#[cfg(test)]
mod tests {
    # [ mutants :: skip ]
    fn spaced() {}
    #[cfg_attr(any(test, feature = "x"), allow(dead_code), mutants::skip)]
    fn listed_second() {}
    #[cfg_attr(test, cfg_attr(feature = "x", mutants::skip))]
    fn nested() {}
}
"""

USE_THEN_BARE = """\
use mutants::skip;

#[cfg(test)]
mod tests {
    #[skip]
    fn helper() {}
}
"""

REASON = "only built with gpu-solver-bridge; its struct literal is not observable"


class Check(unittest.TestCase):
    def run_check(self, files: dict[str, str], ledger: str) -> list[str]:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            if "src/lib.rs" not in files:
                files = dict(files)
                files["src/lib.rs"] = "".join(
                    f"pub mod {Path(r).stem};\n" for r in files if r.startswith("src/") and r.count("/") == 1
                )
            for rel, text in files.items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            (root / "src").mkdir(exist_ok=True)
            ledger_path = root / "ledger.txt"
            ledger_path.write_text(ledger, encoding="utf-8")
            return sk.check(root, ledger_path)

    def test_a_listed_skip_in_a_test_module_passes(self):
        self.assertEqual(self.run_check({"src/a.rs": TEST_MOD}, f"src/a.rs::helper\t{REASON}\n"), [])

    def test_cfg_attr_form_is_a_skip(self):
        self.assertEqual(self.run_check({"src/a.rs": CFG_ATTR}, f"src/a.rs::helper\t{REASON}\n"), [])

    def test_a_skip_on_production_code_fails(self):
        errors = self.run_check({"src/a.rs": PRODUCTION}, f"src/a.rs::production\t{REASON}\n")
        self.assertTrue(any("non-test build" in e for e in errors), errors)

    def test_a_skip_after_a_test_module_is_production(self):
        errors = self.run_check({"src/a.rs": AFTER_TEST_MOD}, f"src/a.rs::after\t{REASON}\n")
        self.assertTrue(any("non-test build" in e for e in errors), errors)

    def test_a_test_only_use_does_not_carry_over_to_the_next_item(self):
        errors = self.run_check({"src/a.rs": TEST_ONLY_USE_THEN_ITEM}, f"src/a.rs::production\t{REASON}\n")
        self.assertTrue(any("non-test build" in e for e in errors), errors)

    def test_an_unlisted_skip_fails(self):
        errors = self.run_check({"src/a.rs": TEST_MOD}, "src/b.rs::other\t" + REASON + "\n")
        self.assertTrue(any("not in" in e for e in errors), errors)
        self.assertTrue(any("names no" in e for e in errors), errors)

    def test_a_short_reason_fails(self):
        errors = self.run_check({"src/a.rs": TEST_MOD}, "src/a.rs::helper\tshort\n")
        self.assertTrue(any("reason shorter" in e for e in errors), errors)

    def test_comments_and_strings_are_not_skips(self):
        errors = self.run_check({"src/a.rs": COMMENT_ONLY}, "")
        self.assertEqual(errors, ["no #[mutants::skip] and no ledger line: compared nothing"])

    def test_nothing_to_compare_fails(self):
        self.assertTrue(self.run_check({"src/a.rs": "pub fn f() {}\n"}, ""))

    def test_needs_test(self):
        self.assertTrue(sk.needs_test("test"))
        self.assertTrue(sk.needs_test('all(test, feature = "std")'))
        self.assertTrue(sk.needs_test("any(test, all(test, loom))"))
        self.assertFalse(sk.needs_test('feature = "std"'))
        self.assertFalse(sk.needs_test('any(test, feature = "std")'))
        self.assertFalse(sk.needs_test("not(test)"))


class Shapes(unittest.TestCase):
    run_check = Check.run_check

    def test_an_inner_skip_at_the_top_of_a_file_fails(self):
        errors = self.run_check({"src/a.rs": INNER_AT_FILE_TOP}, "")
        self.assertTrue(any("inner #![...]" in e for e in errors), errors)

    def test_an_inner_cfg_attr_skip_fails(self):
        errors = self.run_check({"src/a.rs": INNER_CFG_ATTR}, "")
        self.assertTrue(any("inner #![...]" in e for e in errors), errors)

    def test_an_inner_skip_inside_a_test_function_still_fails(self):
        errors = self.run_check({"src/a.rs": INNER_IN_TEST_FN}, "")
        self.assertTrue(any("inner #![...]" in e for e in errors), errors)

    def test_a_leading_double_colon_on_production_code_fails(self):
        errors = self.run_check({"src/a.rs": LEADING_COLONS_ON_PRODUCTION}, f"src/a.rs::production\t{REASON}\n")
        self.assertTrue(any("non-test build" in e for e in errors), errors)

    def test_a_leading_double_colon_in_a_test_module_is_a_listed_skip(self):
        self.assertEqual(self.run_check({"src/a.rs": LEADING_COLONS_IN_TEST}, f"src/a.rs::helper\t{REASON}\n"), [])

    def test_spaced_and_nested_forms_are_skips_that_need_the_ledger(self):
        ledger = "".join(f"src/a.rs::{n}\t{REASON}\n" for n in ("spaced", "listed_second", "nested"))
        self.assertEqual(self.run_check({"src/a.rs": SPACED_AND_NESTED}, ledger), [])
        errors = self.run_check({"src/a.rs": SPACED_AND_NESTED}, f"src/a.rs::spaced\t{REASON}\n")
        self.assertEqual(sorted(e.split(":")[2] for e in errors if "not in" in e), ["listed_second", "nested"])

    def test_a_form_the_reader_does_not_parse_fails(self):
        errors = self.run_check({"src/a.rs": USE_THEN_BARE}, "")
        self.assertTrue(any("token run(s)" in e for e in errors), errors)

    def test_skip_shapes_the_parser_reports(self):
        cases = {
            "#[cfg(test)]\nmod t {\n    #[mutants::skip]\n    fn a() {}\n}\n": ["a"],
            "#[cfg(test)]\nmod t {\n    #[cfg_attr(not(test), mutants::skip)]\n    fn a() {}\n}\n": ["a"],
            "#[cfg(test)]\nmod t {\n    #[cfg_attr(test, allow(x), ::mutants::skip)]\n    fn a() {}\n}\n": ["a"],
            "#[cfg(test)]\nmod t {\n    #[mutants::skip_all]\n    fn a() {}\n}\n": [],
            "#[cfg(test)]\nmod t {\n    #[allow(dead_code)]\n    fn a() {}\n}\n": [],
        }
        for src, names in cases.items():
            ledger = "".join(f"src/a.rs::{n}\t{REASON}\n" for n in names)
            errors = self.run_check({"src/a.rs": src}, ledger)
            want = [] if names else ["no #[mutants::skip] and no ledger line: compared nothing"]
            self.assertEqual(errors, want, src)

    def test_a_skip_in_the_predicate_position_is_a_token_the_parser_does_not_report(self):
        src = "#[cfg(test)]\nmod t {\n    #[cfg_attr(mutants::skip, allow(x))]\n    fn a() {}\n}\n"
        errors = self.run_check({"src/a.rs": src}, "")
        self.assertTrue(any("token run(s)" in e for e in errors), errors)

PRODUCTION_FN = "#[mutants::skip]\npub fn f() {}\n"
TEST_FN = "#[mutants::skip]\npub fn f() {}\n"


class ModuleTree(unittest.TestCase):
    run_check = Check.run_check

    def test_a_skip_in_a_file_outside_src_reached_by_path_is_checked(self):
        # the repro: `#[path]` out of src/, a skip on code a non-test build compiles
        files = {"src/lib.rs": '#[path = "../zz_out/zz_mod.rs"]\npub mod zz_mod;\n', "zz_out/zz_mod.rs": PRODUCTION_FN}
        errors = self.run_check(files, f"zz_out/zz_mod.rs::f\t{REASON}\n")
        self.assertTrue(any("non-test build" in e for e in errors), errors)
        errors = self.run_check(files, "")
        self.assertTrue(any("not in" in e for e in errors), errors)

    def test_a_file_declared_from_a_test_only_line_is_test_only(self):
        files = {"src/lib.rs": '#[cfg(test)]\n#[path = "../tests/lib/x.rs"]\nmod x;\n', "tests/lib/x.rs": TEST_FN}
        self.assertEqual(self.run_check(files, f"tests/lib/x.rs::f\t{REASON}\n"), [])

    def test_a_nested_path_is_followed(self):
        files = {
            "src/lib.rs": '#[path = "../zz_out/outer.rs"]\npub mod outer;\n',
            "zz_out/outer.rs": '#[path = "deeper/inner.rs"]\npub mod inner;\n',
            "zz_out/deeper/inner.rs": PRODUCTION_FN,
        }
        errors = self.run_check(files, f"zz_out/deeper/inner.rs::f\t{REASON}\n")
        self.assertTrue(any("zz_out/deeper/inner.rs::f" in e and "non-test build" in e for e in errors), errors)

    def test_a_plain_mod_in_a_reached_file_and_under_an_inline_mod_is_followed(self):
        files = {
            "src/lib.rs": "pub mod a {\n    pub mod b;\n}\npub mod c;\n",
            "src/a/b.rs": PRODUCTION_FN,
            "src/c.rs": "pub mod d;\n",
            "src/c/d/mod.rs": PRODUCTION_FN,
        }
        errors = self.run_check(files, f"src/a/b.rs::f\t{REASON}\nsrc/c/d/mod.rs::f\t{REASON}\n")
        self.assertEqual(sorted(e.split(" ")[0] for e in errors if "non-test build" in e),
                         ["src/a/b.rs::f", "src/c/d/mod.rs::f"])

    def test_include_is_followed(self):
        files = {"src/lib.rs": 'include!("../zz_out/inc.rs");\n', "zz_out/inc.rs": PRODUCTION_FN}
        errors = self.run_check(files, f"zz_out/inc.rs::f\t{REASON}\n")
        self.assertTrue(any("non-test build" in e for e in errors), errors)

    def test_a_missing_target_fails(self):
        for lib in ('#[path = "../nowhere.rs"]\nmod gone;\n', "mod gone;\n", 'include!("nowhere.rs");\n'):
            errors = self.run_check({"src/lib.rs": lib}, "")
            self.assertTrue(any("names no file" in e for e in errors), (lib, errors))

    def test_a_path_not_on_a_mod_declaration_fails(self):
        errors = self.run_check({"src/lib.rs": '#[path = "x.rs"]\nmod x { }\n'}, "")
        self.assertTrue(any("on the inline `mod x" in e for e in errors), errors)

    def test_a_missing_crate_root_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "ledger.txt").write_text("", encoding="utf-8")
            errors = sk.check(root, root / "ledger.txt")
        self.assertTrue(any("src/lib.rs is missing" in e for e in errors), errors)

    def test_a_file_reached_from_test_and_production_code_is_production(self):
        # the test-only declaration is walked first; the production one must still win
        lib = "pub mod a;\n#[cfg(test)]\n#[path = \"a.rs\"]\nmod a_again;\n"
        errors = self.run_check({"src/lib.rs": lib, "src/a.rs": PRODUCTION_FN}, f"src/a.rs::f\t{REASON}\n")
        self.assertTrue(any("non-test build" in e for e in errors), errors)
        lib = "#[cfg(test)]\n#[path = \"a.rs\"]\nmod a_again;\npub mod a;\n"
        errors = self.run_check({"src/lib.rs": lib, "src/a.rs": PRODUCTION_FN}, f"src/a.rs::f\t{REASON}\n")
        self.assertTrue(any("non-test build" in e for e in errors), errors)

    def test_a_file_not_in_the_module_tree_is_not_read(self):
        files = {"src/lib.rs": "", "src/orphan.rs": PRODUCTION_FN}
        errors = self.run_check(files, "")
        self.assertEqual(errors, ["no #[mutants::skip] and no ledger line: compared nothing"])

LEGIT = "#[cfg(test)]\nmod t {\n    #[mutants::skip]\n    fn legit() {}\n}\n"
LEGIT_LEDGER = f"src/lib.rs::legit\t{REASON}\n"


class Lexing(unittest.TestCase):
    """Literals and comments that a hand-written scanner gets wrong: a
    production skip after each must still be seen (and refused)."""

    run_check = Check.run_check

    def assert_production_skip_seen(self, between: str) -> None:
        lib = LEGIT + between + "\n#[mutants::skip]\nfn prod() {}\n"
        errors = self.run_check({"src/lib.rs": lib}, LEGIT_LEDGER + f"src/lib.rs::prod\t{REASON}\n")
        self.assertTrue(any("src/lib.rs::prod" in e and "non-test build" in e for e in errors), (between, errors))

    def test_raw_strings(self):
        for lit in ('r#"a"{"#', 'r#"say "hi""#', 'r"C:\\"', 'r##"a"#b"##'):
            self.assert_production_skip_seen(f"const S: &str = {lit};")

    def test_byte_and_c_strings(self):
        for lit in ('br#"q"x"#', 'b"x\\"y"', "b'\"'"):
            self.assert_production_skip_seen(f"const B: &[u8] = &[0]; const C: &[u8] = {lit} as _;"
                                             if lit.startswith("b'") else f"const B: &[u8] = {lit};")
        self.assert_production_skip_seen('const C: &core::ffi::CStr = cr"z\\";')

    def test_char_literals_and_lifetimes(self):
        self.assert_production_skip_seen("const Q: char = '\"';")
        self.assert_production_skip_seen("fn f<'a>(x: &'a str) -> &'a str { x }")

    def test_nested_block_comments_with_quotes(self):
        self.assert_production_skip_seen('/* outer " /* inner " */ still " comment */')

    def test_a_skip_inside_a_string_or_comment_is_not_a_skip(self):
        lib = LEGIT + 'const S: &str = r#"#[mutants::skip]"#;\n// #[mutants::skip]\nfn prod() {}\n'
        self.assertEqual(self.run_check({"src/lib.rs": lib}, LEGIT_LEDGER), [])

    def test_a_file_that_does_not_parse_fails(self):
        for bad in ('const S: &str = "unterminated;\n', "fn f( {\n", 'const S: &str = r#"x"##;\n'):
            errors = self.run_check({"src/lib.rs": LEGIT + bad}, LEGIT_LEDGER)
            self.assertTrue(any("the parser refused the file" in e for e in errors), (bad, errors))

    def test_a_skip_inside_a_macro_body_fails(self):
        lib = LEGIT + "macro_rules! m {\n    () => { #[mutants::skip] fn made() {} };\n}\n"
        errors = self.run_check({"src/lib.rs": lib}, LEGIT_LEDGER)
        self.assertTrue(any("token run(s)" in e for e in errors), errors)

    def test_a_one_line_test_module_does_not_make_the_next_item_test_only(self):
        files = {"src/lib.rs": LEGIT + "#[cfg(test)]\nmod u;\n#[mutants::skip]\nfn prod() {}\n", "src/u.rs": ""}
        errors = self.run_check(files, LEGIT_LEDGER + f"src/lib.rs::prod\t{REASON}\n")
        self.assertTrue(any("src/lib.rs::prod" in e and "non-test build" in e for e in errors), errors)

    def test_a_semicolon_in_a_test_fn_type_does_not_end_its_gate_early(self):
        lib = "#[cfg(test)]\nfn helper() -> [u8; 3] {\n    [0; 3]\n}\n#[cfg(test)]\n#[mutants::skip]\nfn h2() -> [u8; 3] { [0; 3] }\n"
        self.assertEqual(self.run_check({"src/lib.rs": lib}, f"src/lib.rs::h2\t{REASON}\n"), [])


class Inheritance(unittest.TestCase):
    run_check = Check.run_check

    def test_test_only_is_inherited_down_a_chain_of_files(self):
        files = {
            "src/lib.rs": "#[cfg(test)]\nmod a;\n",
            "src/a.rs": "mod b;\n",
            "src/a/b.rs": "pub mod c;\n",
            "src/a/b/c.rs": TEST_FN,
        }
        self.assertEqual(self.run_check(files, f"src/a/b/c.rs::f\t{REASON}\n"), [])
        files["src/lib.rs"] = "mod a;\n"
        errors = self.run_check(files, f"src/a/b/c.rs::f\t{REASON}\n")
        self.assertTrue(any("non-test build" in e for e in errors), errors)

    def test_a_path_inside_an_inline_module_is_relative_to_its_module_directory(self):
        # in lib.rs (a mod-rs file) `mod outer { #[path = "p.rs"] mod x; }` reads src/outer/p.rs
        files = {"src/lib.rs": 'mod outer {\n    #[path = "p.rs"]\n    mod x;\n}\n', "src/outer/p.rs": PRODUCTION_FN,
                 "src/p.rs": ""}
        errors = self.run_check(files, f"src/outer/p.rs::f\t{REASON}\n")
        self.assertTrue(any("src/outer/p.rs::f" in e and "non-test build" in e for e in errors), errors)

if __name__ == "__main__":
    unittest.main()
