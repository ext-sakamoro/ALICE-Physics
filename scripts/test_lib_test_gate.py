"""Oracles of scripts/lib_test_gate.py on temporary git repositories.

run: python3 scripts/test_lib_test_gate.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("lib_test_gate", os.path.join(HERE, "lib_test_gate.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def git(root, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root,
                   check=True, capture_output=True)


def repo(base: dict[str, str], change: dict[str, str]) -> str:
    """A repository whose `base` commit holds `base` and whose HEAD adds `change`."""
    root = tempfile.mkdtemp()
    git(root, "init", "-q", "-b", "main")
    for files in (base, change):
        for rel, text in files.items():
            os.makedirs(os.path.dirname(os.path.join(root, rel)) or root, exist_ok=True)
            with open(os.path.join(root, rel), "w", encoding="utf-8") as f:
                f.write(text)
        git(root, "add", "-A")
        git(root, "commit", "-q", "--allow-empty", "-m", "c")
        if files is base:
            git(root, "tag", "base")
    return root


BASE = {"src/lib.rs": "pub mod a;\n", "src/a.rs": "pub fn a() {}\n#[cfg(test)]\nmod tests { fn t() { super::a(); } }\n"}


class Gate(unittest.TestCase):
    def check(self, change, base=BASE):
        return gate.check(repo(base, change), "base")

    def test_a_new_file_without_lib_tests_is_an_error(self):
        errors, _, counts = self.check({"src/law_id.rs": "pub fn law_id() -> u8 { 1 }\n"})
        self.assertEqual(counts["new_files"], 1)
        self.assertTrue(any("src/law_id.rs" in e and "without lib tests" in e for e in errors), errors)

    def test_a_new_file_with_a_test_module_passes(self):
        text = "pub fn f() {}\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() { super::f(); }\n}\n"
        errors, _, counts = self.check({"src/b.rs": text})
        self.assertEqual((errors, counts["new_files"]), ([], 1))

    def test_cfg_all_test_counts_as_a_test_module(self):
        text = 'pub fn f() {}\n#[cfg(all(test, feature = "std"))]\nmod tests {}\n'
        self.assertEqual(self.check({"src/b.rs": text})[0], [])

    def test_a_path_include_of_a_tests_oracle_passes(self):
        text = 'pub fn f() {}\n#[cfg(feature = "std")]\n#[path = "../tests/b_oracle.rs"]\nmod oracle;\n'
        self.assertEqual(self.check({"src/b.rs": text})[0], [])

    def test_an_exempt_file_needs_a_reason(self):
        change = {"src/m/mod.rs": "pub use self::x::*;\n",
                  "scripts/lib-test-exempt.txt": "src/m/mod.rs\n"}
        errors, _, _ = self.check(change)
        self.assertTrue(any("has no reason" in e for e in errors), errors)
        change["scripts/lib-test-exempt.txt"] = "# comment\nsrc/m/mod.rs re-exports only\n"
        self.assertEqual(self.check(change)[0], [])

    def test_no_new_file_is_reported_as_skipped(self):
        errors, warnings, counts = self.check({"README.md": "x\n"})
        self.assertEqual((errors, warnings, counts["new_files"]), ([], [], 0))

    def test_a_new_pub_fn_not_named_by_the_tests_is_a_warning(self):
        text = BASE["src/a.rs"].replace("pub fn a() {}\n", "pub fn a() {}\npub fn b() {}\npub(crate) fn c() {}\n")
        text = text.replace("super::a();", "super::a(); super::c();")
        errors, warnings, counts = self.check({"src/a.rs": text})
        self.assertEqual(errors, [])
        self.assertEqual(counts["new_fns"], 2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("`b`", warnings[0])

    def test_main_exits_1_on_an_error(self):
        root = repo(BASE, {"src/z.rs": "pub fn z() {}\n"})
        old = os.getcwd()
        try:
            os.chdir(root)
            errors, _, _ = gate.check(root, "base")
        finally:
            os.chdir(old)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
