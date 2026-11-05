#!/usr/bin/env python3
"""scripts/wasm_literal_check.py against made-up sources and wasm files (the constant here is invented)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wasm_literal_check as wlc  # noqa: E402

SRC = 'pub const MADE_UP_DOMAIN: &[u8] = b"zorblax/made-up/v1";\n'
CONST = [("src/x.rs", "MADE_UP_DOMAIN")]


class Check(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "x.rs").write_text(SRC, encoding="utf-8")
        self.wasm = self.root / "pkg"
        self.wasm.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self):
        return wlc.check(self.wasm, self.root, CONST)

    def test_a_build_without_the_text_passes_and_says_what_it_read(self):
        (self.wasm / "a.wasm").write_bytes(b"\0asm\x01\0\0\0 masked bytes only")
        code, lines = self.run_check()
        self.assertEqual(code, 0, lines)
        self.assertIn("1 file(s), 1 constant(s), 0 found", lines[-1])

    def test_the_text_in_any_wasm_file_fails_and_names_it(self):
        (self.wasm / "a.wasm").write_bytes(b"\0asm clean")
        (self.wasm / "sub").mkdir()
        (self.wasm / "sub" / "b.wasm").write_bytes(b"\0asm ..zorblax/made-up/v1..")
        code, lines = self.run_check()
        self.assertEqual(code, 1, lines)
        self.assertTrue(any("b.wasm: holds the text of MADE_UP_DOMAIN" in l for l in lines), lines)

    def test_no_wasm_file_is_nothing_checked(self):
        code, lines = self.run_check()
        self.assertEqual(code, 2, lines)

    def test_a_constant_missing_from_its_source_is_nothing_checked(self):
        (self.root / "src" / "x.rs").write_text("// renamed away\n", encoding="utf-8")
        (self.wasm / "a.wasm").write_bytes(b"\0asm")
        self.assertEqual(self.run_check()[0], 2)

    def test_the_repository_list_names_constants_its_sources_hold(self):
        for rel, name in wlc.MASKED_CONSTANTS:
            self.assertIsNotNone(wlc.constant_bytes((wlc.REPO / rel).read_text(encoding="utf-8"), name), name)


if __name__ == "__main__":
    unittest.main()
