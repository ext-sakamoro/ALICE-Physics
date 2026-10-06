#!/usr/bin/env python3
"""Oracle for scripts/run_examples.py.

The checker must not pass vacuously: (1) every binary exiting 0 is green
(2) a binary that panics (exit 101) is red (3) a binary over the timeout is
red (4) zero binaries is red (5) a binary missing after the build is red
(6) discovery keeps only the root package's example targets and the feature
union drops the default `std`. The fixtures are small executables written
here; cargo is not called.

run: python3 scripts/test_run_examples.py
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_examples as r  # noqa: E402


def exe(body: str) -> Path:
    """A python executable fixture (POSIX shebang, or `python file` elsewhere)."""
    d = Path(tempfile.mkdtemp(prefix="examples-fixture-"))
    path = d / "fixture.py"
    path.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@unittest.skipIf(os.name == "nt", "fixtures are shebang executables")
class RunAll(unittest.TestCase):
    def test_all_ok_is_green(self):
        rows, problems = r.run_all({"a": exe("print('a')"), "b": exe("pass")}, 10)
        self.assertEqual(problems, [])
        self.assertEqual([row[1] for row in rows], ["ok", "ok"])

    def test_panic_exit_101_is_red(self):
        _, problems = r.run_all(
            {"ok": exe("pass"), "bad": exe("import sys; print('assert'); sys.exit(101)")}, 10
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("bad: exit 101", problems[0])
        self.assertIn("assert", problems[0])

    def test_timeout_is_red(self):
        _, problems = r.run_all({"slow": exe("import time; time.sleep(30)")}, 0.5)
        self.assertEqual(len(problems), 1)
        self.assertIn("slow: timeout", problems[0])

    def test_zero_examples_is_red(self):
        rows, problems = r.run_all({}, 10)
        self.assertEqual(rows, [])
        self.assertTrue(problems and "0 executed" in problems[0])

    def test_missing_binary_is_red(self):
        missing = Path(tempfile.mkdtemp()) / "nope"
        _, problems = r.run_all({"gone": missing}, 10)
        self.assertTrue(any("gone: binary" in p for p in problems))
        self.assertTrue(any("0 examples executed" in p for p in problems))


class Discover(unittest.TestCase):
    def test_root_package_examples_only_and_feature_union(self):
        meta = {
            "packages": [
                {
                    "manifest_path": str(r.ROOT / "Cargo.toml"),
                    "targets": [
                        {"name": "lib", "kind": ["lib"]},
                        {"name": "b_ex", "kind": ["example"], "required-features": ["std", "replay"]},
                        {"name": "a_ex", "kind": ["example"]},
                        {"name": "c_ex", "kind": ["example"], "required-features": ["neural"]},
                    ],
                },
                {
                    "manifest_path": str(r.ROOT / "fuzz" / "Cargo.toml"),
                    "targets": [{"name": "other", "kind": ["example"]}],
                },
            ]
        }
        ex = r.discover(meta)
        self.assertEqual([n for n, _ in ex], ["a_ex", "b_ex", "c_ex"])
        self.assertEqual(r.feature_union(ex), ["neural", "replay"])

    def test_no_examples_discovered(self):
        self.assertEqual(r.discover({"packages": []}), [])


if __name__ == "__main__":
    unittest.main()
