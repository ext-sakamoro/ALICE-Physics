#!/usr/bin/env python3
"""Tests of scripts/api_os_gate_check.py (python3 scripts/test_api_os_gate_check.py)."""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import api_os_gate_check as chk  # noqa: E402


def run(files: dict[str, str]) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            rc = chk.main(["--root", str(root)])
        return rc, err.getvalue() + out.getvalue()


class ApiOsGateCheck(unittest.TestCase):
    def test_no_file_is_a_failure(self) -> None:
        rc, msg = run({"README.md": "x\n"})
        self.assertEqual(rc, 2, msg)

    def test_plain_crate_passes_and_counts(self) -> None:
        rc, msg = run({"src/lib.rs": "pub fn f() {}\n", "crates/a/src/lib.rs": "pub use x::*;\n"})
        self.assertEqual(rc, 0, msg)
        self.assertIn("rust files 2", msg)

    def test_os_gate_on_a_public_item_fails(self) -> None:
        for attr in ('#[cfg(target_os = "macos")]', "#[cfg(unix)]", '#[cfg(all(feature = "std", windows))]',
                     '#[cfg_attr(target_family = "wasm", doc = "x")]', '#[cfg(not(target_env = "msvc"))]'):
            with self.subTest(attr=attr):
                rc, msg = run({"src/lib.rs": f"{attr}\n/// doc\n#[inline]\npub fn f() {{}}\n"})
                self.assertEqual(rc, 1, msg)
                self.assertIn("src/lib.rs:1", msg)

    def test_os_gate_in_a_member_crate_fails(self) -> None:
        rc, msg = run({"src/lib.rs": "pub fn g() {}\n", "crates/a/src/lib.rs": "#[cfg(windows)]\npub mod m {}\n"})
        self.assertEqual(rc, 1, msg)
        self.assertIn("crates/a/src/lib.rs:1", msg)

    def test_os_gate_on_a_private_or_restricted_item_passes(self) -> None:
        text = ("#[cfg(unix)]\n#[test]\nfn t() {}\n"
                "#[cfg(target_os = \"linux\")]\npub(crate) fn h() {}\n"
                "fn body() { if cfg!(target_os = \"linux\") {} }\n")
        rc, msg = run({"src/lib.rs": text})
        self.assertEqual(rc, 0, msg)

    def test_a_non_os_gate_on_a_public_item_passes(self) -> None:
        rc, msg = run({"src/lib.rs": '#[cfg(feature = "std")]\npub fn f() {}\n#[cfg(target_arch = "x86_64")]\npub fn s() {}\n'})
        self.assertEqual(rc, 0, msg)

    def test_file_level_os_gate_with_public_items_fails(self) -> None:
        rc, msg = run({"src/os.rs": "#![cfg(unix)]\npub fn f() {}\n"})
        self.assertEqual(rc, 1, msg)
        rc, msg = run({"src/os.rs": "#![cfg(unix)]\nfn f() {}\n"})
        self.assertEqual(rc, 0, msg)

    def test_a_multi_line_os_gate_on_a_public_item_fails(self) -> None:
        # rustfmt writes a long cfg over several lines; the OS word is not on the `#[cfg(` line
        text = '#[cfg(any(\n    feature = "x",\n    target_os = "macos",\n))]\npub fn only_mac() {}\n'
        rc, msg = run({"src/lib.rs": text})
        self.assertEqual(rc, 1, msg)
        self.assertIn("src/lib.rs:1", msg)

    def test_an_os_gate_and_its_public_item_on_one_line_fails(self) -> None:
        rc, msg = run({"src/lib.rs": "#[cfg(unix)] pub fn f() {}\n"})
        self.assertEqual(rc, 1, msg)
        rc, msg = run({"src/lib.rs": "#[cfg(unix)] #[inline] pub fn f() {}\n"})
        self.assertEqual(rc, 1, msg)
        rc, msg = run({"src/lib.rs": "#[cfg(unix)] fn f() {}\n"})
        self.assertEqual(rc, 0, msg)

    def test_an_os_gate_then_a_multi_line_attribute_then_a_public_item_fails(self) -> None:
        text = '#[cfg(windows)]\n#[doc = concat!(\n    "a",\n    "b"\n)]\npub fn f() {}\n'
        rc, msg = run({"src/lib.rs": text})
        self.assertEqual(rc, 1, msg)

    def test_a_multi_line_non_os_gate_on_a_public_item_passes(self) -> None:
        text = '#[cfg(all(\n    feature = "std",\n    target_arch = "x86_64",\n))]\npub fn s() {}\n'
        rc, msg = run({"src/lib.rs": text})
        self.assertEqual(rc, 0, msg)

    def test_gates_in_comments_and_strings_are_not_attributes(self) -> None:
        text = ('/// ```\n/// #[cfg(unix)]\n/// pub fn doc() {}\n/// ```\n'
                '/*\n#[cfg(windows)]\npub fn commented() {}\n*/\npub fn a() {}\n'
                'pub const S: &str = "#[cfg(unix)]";\npub fn b() {}\n')
        rc, msg = run({"src/lib.rs": text})
        self.assertEqual(rc, 0, msg)

    def test_a_gate_belongs_to_its_own_item_only(self) -> None:
        # the gated item is private; the public item after it is not gated
        rc, msg = run({"src/lib.rs": "#[cfg(unix)]\nfn private() {}\npub fn public() {}\n"})
        self.assertEqual(rc, 0, msg)

    def test_the_item_itself_must_be_pub_not_a_field_inside_it(self) -> None:
        rc, msg = run({"src/lib.rs": "#[cfg(unix)]\npub(crate) struct S { pub x: u8 }\n"})
        self.assertEqual(rc, 0, msg)

    def test_this_repository_passes(self) -> None:
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            rc = chk.main([])
        self.assertEqual(rc, 0, err.getvalue())


if __name__ == "__main__":
    unittest.main()
