#!/usr/bin/env python3
"""oracle for scripts/semver_baseline_has.py, on a throwaway repository: a root package, a
workspace member, a package added in a later commit, and an unreadable revision."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "semver_baseline_has.py"
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
ID = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-c", "maintenance.auto=false"]


def git(d: Path, *args: str) -> None:
    subprocess.run(["git", *ID, *args], cwd=d, check=True, capture_output=True, env=ENV)


class BaselineHas(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="semver-baseline-"))
        self.addCleanup(shutil.rmtree, self.d, True)
        git(self.d, "init", "-q")
        (self.d / "Cargo.toml").write_text('[package]\nname = "engine"\nversion = "1.0.0"\n')
        git(self.d, "add", "-A")
        git(self.d, "commit", "-qm", "one package")
        (self.d / "Cargo.toml").write_text('[package]\nname = "engine-core"\nversion = "2.0.0"\n\n'
                                           '[workspace]\nmembers = [".", "crates/engine"]\n')
        (self.d / "crates" / "engine").mkdir(parents=True)
        (self.d / "crates" / "engine" / "Cargo.toml").write_text('[package]\nname = "engine"\nversion = "2.0.0"\n')
        git(self.d, "add", "-A")
        git(self.d, "commit", "-qm", "split")

    def run_it(self, rev: str, pkg: str) -> int:
        return subprocess.run([sys.executable, str(SCRIPT), rev, pkg], cwd=self.d, capture_output=True, env=ENV).returncode

    def test_the_root_package_at_the_baseline(self):
        self.assertEqual(self.run_it("HEAD~1", "engine"), 0)

    def test_a_package_introduced_later_is_not_at_the_baseline(self):
        self.assertEqual(self.run_it("HEAD~1", "engine-core"), 1)

    def test_a_workspace_member_and_the_renamed_root(self):
        self.assertEqual(self.run_it("HEAD", "engine"), 0)
        self.assertEqual(self.run_it("HEAD", "engine-core"), 0)

    def test_an_unknown_package(self):
        self.assertEqual(self.run_it("HEAD", "other"), 1)

    def test_an_unreadable_revision_is_an_error_not_absent(self):
        self.assertEqual(self.run_it("no-such-rev", "engine"), 2)


if __name__ == "__main__":
    unittest.main()
