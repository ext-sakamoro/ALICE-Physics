"""Oracles of scripts/package_contents_check.py.

run: python3 scripts/test_package_contents_check.py
"""
from __future__ import annotations

import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "package_contents_check", os.path.join(HERE, "package_contents_check.py")
)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

GOOD = [
    "Cargo.toml",
    "Cargo.toml.orig",
    "Cargo.lock",
    "src/lib.rs",
    "tests/fixtures/world_snapshot_v1_stacked.bin",
    "README.md",
    "LICENSE-AGPL",
    "LICENSE-COMMERCIAL.md",
]


class PackageContents(unittest.TestCase):
    def test_good_list_passes(self) -> None:
        self.assertEqual(pc.check(GOOD), [])

    def test_empty_list_fails(self) -> None:
        self.assertEqual(len(pc.check([])), 1)
        self.assertEqual(len(pc.check(["", "  "])), 1)

    def test_each_forbidden_dir_fails(self) -> None:
        for d in pc.FORBIDDEN_DIRS:
            errs = pc.check(GOOD + [f"{d}/x.txt"])
            self.assertEqual(errs, [f"shipped but not allowed: {d}/x.txt"], d)

    def test_toolchain_pin_fails(self) -> None:
        for f in pc.FORBIDDEN_FILES:
            self.assertEqual(pc.check(GOOD + [f]), [f"shipped but not allowed: {f}"])

    def test_top_level_file_named_like_dir_passes(self) -> None:
        # a top-level file called "docs" is not the docs/ directory
        self.assertEqual(pc.check(GOOD + ["docs"]), [])

    def test_missing_required_fails(self) -> None:
        for req in pc.REQUIRED:
            errs = pc.check([p for p in GOOD if p != req])
            self.assertEqual(errs, [f"required file missing from the package: {req}"], req)


if __name__ == "__main__":
    unittest.main()
