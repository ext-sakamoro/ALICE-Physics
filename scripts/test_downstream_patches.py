#!/usr/bin/env python3
"""Tests for scripts/downstream_patches.py: one patch per source and package of the downstream lock.

A crates.io package gets `patch.crates-io`, a git package `patch."<url>"` (the URL without `git+`,
the query and the fragment); a path package and other names get none; a registry other than crates.io,
an unknown source, no lock, a lock that does not parse and a lock with none of the packages fail.
The resolve guard on both kinds against cargo is in scripts/test_downstream_resolve.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import downstream_patches as dp  # noqa: E402

GIT = "git+https://github.com/ext-sakamoro/ALICE-Physics?rev=4a05357053e6e1e362e5d6ef5c9031d1b0e8ad0d#4a05357053e6e1e362e5d6ef5c9031d1b0e8ad0d"
CRATES = "registry+https://github.com/rust-lang/crates.io-index"
SPARSE = "sparse+https://index.crates.io/"
PKGS = {"alice-physics": "/f", "alice-physics-core": "/c"}


def lock(*entries: tuple[str, str | None]) -> str:
    out = "version = 4\n"
    for name, source in entries:
        out += f'\n[[package]]\nname = "{name}"\nversion = "4.0.0"\n'
        if source is not None:
            out += f'source = "{source}"\n'
    return out


class Patches(unittest.TestCase):
    def test_a_git_source_is_patched_by_its_url_for_both_packages(self):
        self.assertEqual(dp.patches(lock(("alice-physics", GIT), ("alice-physics-core", GIT)), PKGS), [
            'patch."https://github.com/ext-sakamoro/ALICE-Physics".alice-physics.path="/f"',
            'patch."https://github.com/ext-sakamoro/ALICE-Physics".alice-physics-core.path="/c"',
        ])

    def test_crates_io_is_patched_as_crates_io_in_both_index_forms(self):
        for source in (CRATES, SPARSE):
            with self.subTest(source=source):
                self.assertEqual(dp.patches(lock(("alice-physics", source)), PKGS),
                                 ['patch.crates-io.alice-physics.path="/f"'])

    def test_both_kinds_in_one_lock_give_one_line_each(self):
        got = dp.patches(lock(("alice-physics", CRATES), ("alice-physics", GIT), ("alice-physics", GIT)), PKGS)
        self.assertEqual(got, ['patch.crates-io.alice-physics.path="/f"',
                               'patch."https://github.com/ext-sakamoro/ALICE-Physics".alice-physics.path="/f"'])

    def test_path_packages_and_other_names_are_not_patched(self):
        self.assertEqual(dp.patches(lock(("alice-physics", None), ("serde", CRATES)), PKGS), [])

    def test_another_registry_and_an_unknown_source_fail(self):
        for source in ("registry+https://example.invalid/index", "local+/x"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    dp.patches(lock(("alice-physics", source)), PKGS)


class Main(unittest.TestCase):
    def run_main(self, text: str | None, *packages: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            if text is not None:
                with open(os.path.join(d, "Cargo.lock"), "w", encoding="utf-8") as f:
                    f.write(text)
            p = subprocess.run([sys.executable, os.path.join(HERE, "downstream_patches.py"), d, *packages],
                               capture_output=True, text=True, check=False)
        return p.returncode, p.stdout + p.stderr

    def test_it_prints_one_line_per_patch(self):
        rc, out = self.run_main(lock(("alice-physics", GIT)), "--package", "alice-physics=/f")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out.strip().splitlines(),
                         ['patch."https://github.com/ext-sakamoro/ALICE-Physics".alice-physics.path="/f"'])

    def test_compared_nothing_and_unreadable_input_fail(self):
        for text, args, why in [
            (lock(("serde", CRATES)), ["--package", "alice-physics=/f"], "compared nothing"),
            (lock(("alice-physics", GIT)), [], "compared nothing"),
            (None, ["--package", "alice-physics=/f"], "Cargo.lock"),
            ("[[package", ["--package", "alice-physics=/f"], "Cargo.lock"),
            (lock(("alice-physics", GIT)), ["--package", "alice-physics"], "NAME=DIR"),
        ]:
            with self.subTest(why=why, args=args):
                rc, out = self.run_main(text, *args)
                self.assertEqual(rc, 1, out)
                self.assertIn(why, out)


if __name__ == "__main__":
    unittest.main()
