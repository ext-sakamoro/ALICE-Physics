#!/usr/bin/env python3
"""Tests of ci_base.py in a throwaway git repository (shared file: the canonical copy is ALICE-DetMath
scripts/test_ci_base.py, carried byte for byte with scripts/ci_base.py)."""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_base  # noqa: E402


class CiBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = Path(self._tmp.name)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.d)
        self.env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                    "GIT_COMMITTER_EMAIL": "t@t"}
        self.git("init", "-q", "-b", "main")
        self.n = 0

    def git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.d, env=self.env, check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout.strip()

    def commit(self) -> str:
        self.n += 1
        (self.d / "f").write_text(str(self.n), encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", str(self.n))
        return self.git("rev-parse", "HEAD")

    def run_cli(self, *args: str, origin: str | None = None) -> tuple[int, str, str]:
        self.git("update-ref", "refs/remotes/origin/main", origin or "HEAD")
        out, err = io.StringIO(), io.StringIO()
        saved = os.environ.copy()
        os.environ.update(self.env)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = ci_base.main(list(args))
        finally:
            os.environ.clear()
            os.environ.update(saved)
        return rc, out.getvalue(), err.getvalue()

    def test_git_output_is_read_as_utf_8_whatever_the_locale(self):
        # a Windows runner's locale code page is cp1252: git's UTF-8 output (a ref, a message, a path)
        # must not be decoded with it. A child interpreter outside UTF-8 mode with the locale encoding
        # set to cp1252 reads a non-ASCII commit subject through ci_base.git(); the same call without an
        # encoding (the control) must come back different, or the locale was not in effect (it raises on
        # POSIX; on Windows the reader thread dies and the result has no output)
        subject = "ブランチ — naïve"
        (self.d / "f").write_text("x", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", subject)
        child = (
            "import locale, subprocess, sys\n"
            "locale.getencoding = lambda: 'cp1252'\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
            "import ci_base\n"
            "cmd = ['git', 'log', '-1', '--format=%s']\n"
            "try:\n"
            "    out = subprocess.run(cmd, capture_output=True, text=True).stdout\n"
            "except UnicodeDecodeError:\n"
            "    out = None\n"
            "control = '<undecodable>' if out is None else out.strip()\n"
            "sys.stdout.buffer.write((control + '\\n' + ci_base.git(*cmd[1:]).stdout.strip()).encode('utf-8'))\n"
        )
        r = subprocess.run([sys.executable, "-X", "utf8=0", "-B", "-c", child], cwd=self.d, env=self.env,
                           capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
        control, got = r.stdout.decode("utf-8").split("\n")
        self.assertNotEqual(control, subject, "the cp1252 locale did not take effect (vacuous)")
        self.assertEqual(got, subject)

    def test_a_push_to_main_of_two_commits_uses_main_before_the_push(self):
        before = self.commit()
        self.commit()
        self.commit()
        rc, out, err = self.run_cli("--event", "push", "--ref", "refs/heads/main", "--before", before)
        self.assertEqual((rc, out), (0, before + "\n"), err)
        self.assertIn(f"ci_base: base {before} (rule before, push refs/heads/main)", err)

    def test_a_before_this_clone_lacks_fails(self):
        self.commit()
        rc, out, err = self.run_cli("--event", "push", "--ref", "refs/heads/main", "--before", "ab" * 20)
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("is not in this clone", err)

    def test_a_before_that_is_not_an_ancestor_fails(self):
        self.commit()
        side = self.git("commit-tree", "-m", "side", self.git("rev-parse", "HEAD^{tree}"))
        self.commit()
        rc, out, err = self.run_cli("--event", "push", "--ref", "refs/heads/main", "--before", side)
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("is not an ancestor of HEAD", err)

    def test_an_all_zero_empty_or_short_before_on_main_fails(self):
        self.commit()
        for before, why in (("0" * 40, "all-zero before"), ("", "as a full SHA"), ("abc", "as a full SHA")):
            rc, out, err = self.run_cli("--event", "push", "--ref", "refs/heads/main", "--before", before)
            self.assertEqual((rc, out), (2, ""), before)
            self.assertIn(why, err)

    def test_a_short_prefix_of_a_real_ancestor_on_main_fails(self):
        # the prefix names an existing ancestor, so only the full-SHA rule can refuse it (cat-file and
        # merge-base --is-ancestor both accept a unique prefix)
        before = self.commit()
        self.commit()
        short = before[:12]
        self.assertEqual(self.git("rev-parse", short), before)
        rc, out, err = self.run_cli("--event", "push", "--ref", "refs/heads/main", "--before", short)
        self.assertEqual((rc, out), (2, ""), err)
        self.assertIn(f"needs github.event.before as a full SHA, got {short!r}", err)

    def test_off_main_the_base_is_the_merge_base(self):
        base = self.commit()
        self.commit()
        rc, out, err = self.run_cli("--event", "push", "--ref", "refs/heads/ci/x", "--before", "0" * 40,
                                    origin=base)
        self.assertEqual((rc, out), (0, base + "\n"), err)
        self.assertIn(f"base {base} (rule merge-base, push refs/heads/ci/x)", err)

    def test_on_origin_main_off_main_the_base_is_the_parent(self):
        parent = self.commit()
        self.commit()
        rc, out, err = self.run_cli()
        self.assertEqual((rc, out), (0, parent + "\n"), err)
        self.assertIn(f"base {parent} (rule HEAD~1, local)", err)

    def test_a_root_commit_on_origin_main_has_no_base(self):
        self.commit()
        rc, out, err = self.run_cli()
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("has no parent", err)


if __name__ == "__main__":
    unittest.main()
