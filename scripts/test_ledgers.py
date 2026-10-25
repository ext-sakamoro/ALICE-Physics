#!/usr/bin/env python3
"""Tests for scripts/ledgers.py: `--check` lists every stale file and every
failing check-only script in one run (not the first one only), restores the
tree, treats a missing tool as an error unless allowed, and fails when it
compares nothing. The real generators are replaced by small scripts in a
temporary tree."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ledgers as g  # noqa: E402

PY = sys.executable


def tree() -> Path:
    root = Path(tempfile.mkdtemp(prefix="ledgers-"))
    # removed after this module's tests (the suite fails on anything left, see main)
    unittest.addModuleCleanup(shutil.rmtree, root, True)
    (root / "docs").mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (root / "docs" / name).write_text("old\n", encoding="utf-8")
    (root / "w.py").write_text(
        "import sys, pathlib\npathlib.Path(sys.argv[1]).write_text(sys.argv[2] + '\\n')\n", encoding="utf-8")
    (root / "fail.py").write_text("import sys\nprint('table out of date', file=sys.stderr)\nsys.exit(1)\n",
                                  encoding="utf-8")
    (root / "ok.py").write_text("", encoding="utf-8")
    return root


def patched(root, writers, check_only=(), index=True, api=True):
    return [
        mock.patch.object(g, "GENERATED", ("docs/a.md", "docs/b.md", "docs/c.md")),
        mock.patch.object(g, "SEMI_GENERATED", ()),
        mock.patch.object(g, "WRITERS", writers),
        mock.patch.object(g, "OUTPUTS", {"a": ("docs/a.md",), "b": ("docs/b.md",), "c": ("docs/c.md",),
                                         "broken": (), "public API": ()}),
        mock.patch.object(g, "CHECK_ONLY", check_only),
        mock.patch.object(g, "INDEX", [PY, str(root / "ok.py")]),
        mock.patch.object(g, "have_index_tool", lambda: index),
        mock.patch.object(g, "have_public_api_tool", lambda r: api),
        mock.patch.object(g, "write_public_api", lambda r: None),
    ]


def run_check(root, writers, check_only=(), index=True, api=True, allow=False):
    ps = patched(root, writers, check_only, index, api)
    for p in ps:
        p.start()
    try:
        return g.check(root, allow)
    finally:
        for p in ps:
            p.stop()


class Check(unittest.TestCase):
    def test_every_stale_file_is_listed_and_the_tree_is_restored(self):
        root = tree()
        writers = (
            ("a", [PY, str(root / "w.py"), str(root / "docs/a.md"), "new"], False),
            ("b", [PY, str(root / "w.py"), str(root / "docs/b.md"), "old"], False),
            ("c", [PY, str(root / "w.py"), str(root / "docs/c.md"), "newer"], True),
        )
        problems, skipped, compared = run_check(root, writers)
        stale = sorted(p.split(" ")[0] for p in problems)
        self.assertEqual(stale, ["docs/a.md", "docs/c.md"])
        self.assertEqual(skipped, [])
        self.assertEqual(compared, 3)
        for name in ("a.md", "b.md", "c.md"):
            self.assertEqual((root / "docs" / name).read_text(), "old\n", name)

    def test_current_files_pass(self):
        root = tree()
        writers = (("a", [PY, str(root / "w.py"), str(root / "docs/a.md"), "old"], False),)
        problems, _, _ = run_check(root, writers)
        self.assertEqual(problems, [])

    def test_every_failing_check_only_script_is_listed(self):
        root = tree()
        check_only = (("one", [PY, str(root / "fail.py")]), ("two", [PY, str(root / "ok.py")]),
                      ("three", [PY, str(root / "fail.py")]))
        problems, _, compared = run_check(root, (), check_only)
        self.assertEqual(len(problems), 2)
        self.assertIn("table out of date", problems[0])
        # no writer ran, so only the three check-only scripts were compared
        self.assertEqual(compared, 3)

    def test_a_missing_tool_fails_unless_allowed(self):
        root = tree()
        writers = (("c", [PY, str(root / "w.py"), str(root / "docs/c.md"), "new"], True),)
        problems, skipped, _ = run_check(root, writers, index=False, api=False)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("rust-analyzer" in p for p in problems))
        problems, skipped, _ = run_check(root, writers, index=False, api=False, allow=True)
        self.assertEqual(problems, [])
        self.assertEqual(len(skipped), 3)

    def test_a_failing_writer_is_reported_and_the_tree_restored(self):
        root = tree()
        writers = (("broken", [PY, str(root / "fail.py")], False),
                   ("a", [PY, str(root / "w.py"), str(root / "docs/a.md"), "new"], False))
        problems, _, _ = run_check(root, writers)
        self.assertEqual(len(problems), 2)
        self.assertEqual((root / "docs/a.md").read_text(), "old\n")

    def test_a_missing_ledger_is_reported_and_not_left_behind(self):
        root = tree()
        (root / "docs/b.md").unlink()
        writers = (("b", [PY, str(root / "w.py"), str(root / "docs/b.md"), "x"], False),)
        problems, _, compared = run_check(root, writers)
        self.assertEqual(len(problems), 1)
        self.assertIn("docs/b.md is missing", problems[0])
        self.assertFalse((root / "docs/b.md").exists())
        self.assertEqual(compared, 0)

    def test_files_of_skipped_writers_are_not_counted(self):
        root = tree()
        writers = (("a", [PY, str(root / "w.py"), str(root / "docs/a.md"), "old"], False),
                   ("c", [PY, str(root / "w.py"), str(root / "docs/c.md"), "new"], True))
        problems, skipped, compared = run_check(root, writers, index=False, api=False, allow=True)
        # only a ran: c's writer needs the index, which is missing
        self.assertEqual(problems, [])
        self.assertEqual(compared, 1)

    def test_the_outputs_table_matches_the_lists(self):
        files = {f for v in g.OUTPUTS.values() for f in v}
        self.assertEqual(files, set(g.GENERATED) | set(g.SEMI_GENERATED))
        self.assertEqual(set(g.OUTPUTS) - {"public API"}, {n for n, _, _ in g.WRITERS})

    def test_nothing_compared_fails(self):
        root = tree()  # the index and public API steps succeed; only the count is zero
        ps = patched(root, ())
        for p in ps:
            p.start()
        try:
            with mock.patch.object(g, "ROOT", root), mock.patch.object(g, "GENERATED", ()), \
                    mock.patch.object(sys, "argv", ["ledgers.py", "--check"]):
                self.assertEqual(g.main(), 1)
        finally:
            for p in ps:
                p.stop()

    def test_land_uses_the_same_list(self):
        import land
        self.assertIs(land.GENERATED, g.GENERATED)
        self.assertIs(land.SEMI_GENERATED, g.SEMI_GENERATED)
        self.assertEqual(land.NIGHTLY, g.NIGHTLY)


def main() -> int:
    """the suite inside a temporary directory of its own (also the $TMPDIR of what it runs): a test
    that leaves a file or directory behind fails the run, naming it"""
    scratch = Path(tempfile.mkdtemp(prefix="test_ledgers-tests-"))
    old_tempdir, old_env = tempfile.tempdir, os.environ.get("TMPDIR")
    tempfile.tempdir = str(scratch)
    os.environ["TMPDIR"] = str(scratch)
    try:
        ok = unittest.main(exit=False).result.wasSuccessful()
        left = sorted(p.name for p in scratch.iterdir())
    finally:
        tempfile.tempdir = old_tempdir
        if old_env is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_env
        shutil.rmtree(scratch, ignore_errors=True)
    if left:
        print(f"test_ledgers: the tests left {len(left)} entries in the temporary directory: {', '.join(left[:10])}",
              file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
