#!/usr/bin/env python3
"""Tests for scripts/coverage_refs_to_symbols.py: each kind of line maps to the
item it belongs to, and a converted table passes scripts/coverage_check.py."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coverage_refs_to_symbols as conv  # noqa: E402

SRC = """//! Module documentation: only the straight case is handled.
//!
use std::fmt;

/// Bends nothing.
/// Its limitation is stated here.
#[inline]
pub fn straight(x: u8) -> u8 {
    let y = x + 1;
    y
}

/// Fixed-point product; it wraps instead of saturating.
impl std::ops::Mul for Q {
    type Output = Q;
    fn mul(self, r: Q) -> Q { r }
}

pub struct Q;
"""
#  line numbers:            1 module doc, 5-7 docs/attr of `straight`, 9 code in
#  `straight`, 13 doc of an unnamed impl, 16 `fn mul`


def table(evidence: str, limitation: str = "") -> str:
    return ("[[item]]\nid = \"COV-TST-001\"\naxis = \"a\"\nitem = \"x\"\nsource = \"s\"\n"
            "status = \"missing\"\n"
            f"evidence = \"{evidence}\"\noracle_candidate = \"-\"\nlimitation = \"{limitation}\"\n")


class Convert(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src/m.rs").write_text(SRC, encoding="utf-8")
        (self.root / "docs/coverage").mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def run_on(self, evidence: str, limitation: str = "") -> tuple[str, list[str]]:
        t = self.root / "docs/coverage/tst.toml"
        t.write_text(table(evidence, limitation), encoding="utf-8")
        _, left = conv.convert(self.root, write=True)
        return t.read_text(encoding="utf-8"), left

    def test_a_doc_comment_or_attribute_line_names_the_item_below(self):
        for n in (5, 6, 7):
            text, _ = self.run_on(f"src/m.rs:{n}")
            self.assertIn('evidence = "src/m.rs::straight"', text, n)

    def test_a_code_line_names_the_enclosing_item(self):
        text, _ = self.run_on("src/m.rs:9 and src/m.rs:16")
        self.assertIn('evidence = "src/m.rs::straight and src/m.rs::mul"', text)

    def test_the_module_documentation_becomes_the_bare_file(self):
        text, _ = self.run_on("src/m.rs:1")
        self.assertIn('evidence = "src/m.rs"', text)

    def test_module_documentation_directly_above_an_item_stays_the_file(self):
        # `//!` documents the module even when the next line is a definition
        (self.root / "src/m.rs").write_text("//! The module.\n/// The fn.\npub fn f() {}\n", encoding="utf-8")
        text, _ = self.run_on("src/m.rs:1")
        self.assertIn('evidence = "src/m.rs"', text)

    def test_a_limitation_keeps_its_quote_and_gets_the_item(self):
        text, left = self.run_on("x", "src/m.rs:6 'Its limitation is stated here.'")
        self.assertIn("limitation = \"src/m.rs::straight 'Its limitation is stated here.'\"", text)
        self.assertEqual(left, [])

    def test_a_quote_outside_any_named_item_is_widened_to_the_file_and_reported(self):
        text, left = self.run_on("x", "src/m.rs:13 'it wraps instead of saturating'")
        self.assertIn("limitation = \"src/m.rs 'it wraps instead of saturating'\"", text)
        self.assertTrue(any("widened to the whole file" in m for m in left), left)

    def test_a_reference_past_the_end_is_left_and_reported(self):
        text, left = self.run_on("src/m.rs:999")
        self.assertIn('evidence = "src/m.rs:999"', text)
        self.assertTrue(any("still holds a line reference" in m for m in left), left)

    def test_check_mode_writes_nothing(self):
        t = self.root / "docs/coverage/tst.toml"
        t.write_text(table("src/m.rs:9"), encoding="utf-8")
        self.assertEqual(conv.main(["--root", str(self.root), "--check"]), 1)
        self.assertIn('"src/m.rs:9"', t.read_text(encoding="utf-8"))

    def test_no_table_is_reported(self):
        (self.root / "docs/coverage").rmdir()
        changed, left = conv.convert(self.root, write=False)
        self.assertEqual(changed, 0)
        self.assertTrue(left)


if __name__ == "__main__":
    unittest.main()
