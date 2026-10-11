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

REASON = "only built with gpu-solver-bridge; its struct literal is not observable"


class Check(unittest.TestCase):
    def run_check(self, files: dict[str, str], ledger: str) -> list[str]:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
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


if __name__ == "__main__":
    unittest.main()
