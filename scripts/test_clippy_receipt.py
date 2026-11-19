#!/usr/bin/env python3
"""Tests of scripts/clippy_receipt.py: the lines it reads from a workflow, what its key
moves with, and each way `check` refuses."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clippy_receipt as c  # noqa: E402

WORKFLOW = """\
jobs:
  clippy:
    steps:
      - uses: actions/checkout@v5
      - name: plain
        run: cargo clippy --all-targets -- -D warnings
      # run: cargo clippy --in-a-comment
      - name: folded
        run: >-
          cargo clippy
          --features "a,b"
          --all-targets -- -D warnings
      - name: other
        run: cargo test
      - name: literal with env
        env:
          CLIPPY_CONF_DIR: .github/clippy-lib
        run: |
          cargo clippy --lib --features "wasm" -- -D clippy::disallowed_methods
          cargo clippy --lib --no-default-features -- -D clippy::disallowed_methods
"""


def repo(tmp: str, workflow: str = WORKFLOW) -> Path:
    root = Path(tmp) / "repo"
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/ci.yml").write_text(workflow, encoding="utf-8")
    (root / "src").mkdir()
    (root / "src/lib.rs").write_text("pub fn one() -> u32 { 1 }\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs/ledger.md").write_text("ledger 1\n", encoding="utf-8")
    (root / "README.md").write_text("readme 1\n", encoding="utf-8")
    sh(root, "init", "-q")
    commit(root, "first")
    return root


def sh(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args], cwd=root, check=True,
                   capture_output=True)


def commit(root: Path, message: str) -> None:
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", message)


class Lines(unittest.TestCase):
    def test_the_three_forms_of_run_are_read_in_order_with_the_step_env(self):
        found = c.commands(WORKFLOW, "ci.yml")
        self.assertEqual([f["cmd"] for f in found], [
            "cargo clippy --all-targets -- -D warnings",
            'cargo clippy --features "a,b" --all-targets -- -D warnings',
            'cargo clippy --lib --features "wasm" -- -D clippy::disallowed_methods',
            "cargo clippy --lib --no-default-features -- -D clippy::disallowed_methods",
        ])
        self.assertEqual([f["env"] for f in found], [{}, {}, {"CLIPPY_CONF_DIR": ".github/clippy-lib"},
                                                     {"CLIPPY_CONF_DIR": ".github/clippy-lib"}])

    def test_no_line_read_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = repo(tmp, "jobs:\n  t:\n    steps:\n      - run: cargo test\n")
            with self.assertRaisesRegex(c.ReceiptError, "no `cargo clippy` line read from 1 workflow"):
                c.workflow_lines(root)

    def test_a_clippy_that_is_not_one_readable_line_is_an_error(self):
        for run in ("cd x && cargo clippy", "cargo clippy --features ${{ matrix.f }}", "cargo clippy \\"):
            with self.assertRaises(c.ReceiptError, msg=run):
                c.commands(f"steps:\n  - name: n\n    run: {run}\n", "w.yml")


class Receipt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = repo(self.tmp.name)
        self.old = os.environ.get("CLIPPY_RECEIPT_DIR")
        os.environ["CLIPPY_RECEIPT_DIR"] = str(Path(self.tmp.name) / "receipts")

    def tearDown(self):
        if self.old is None:
            del os.environ["CLIPPY_RECEIPT_DIR"]
        else:
            os.environ["CLIPPY_RECEIPT_DIR"] = self.old
        self.tmp.cleanup()

    def run_all(self, red: str | None = None) -> list:
        seen = []

        def runner(cmd, env):
            seen.append((cmd, env))
            return 101 if red and red in cmd else 0
        c.run(self.root, runner)
        return seen

    def test_without_a_receipt_the_check_refuses_and_with_a_green_one_it_passes(self):
        with self.assertRaisesRegex(c.ReceiptError, "no clippy receipt for the code of HEAD"):
            c.check(self.root)
        seen = self.run_all()
        self.assertEqual(len(seen), 4, "every line is run")
        self.assertEqual(seen[3][1], {"CLIPPY_CONF_DIR": ".github/clippy-lib"})
        self.assertIn("4 of 4 lines green", c.check(self.root))

    def test_a_red_line_does_not_stop_the_next_and_the_check_refuses(self):
        seen = self.run_all(red="--all-targets -- -D warnings")
        self.assertEqual(len(seen), 4, "the lines after the red ones are still run")
        with self.assertRaisesRegex(c.ReceiptError, "2 of 4 clippy line"):
            c.check(self.root)

    def test_a_change_of_code_asks_for_a_new_run_and_a_change_of_a_ledger_does_not(self):
        self.run_all()
        (self.root / "docs/ledger.md").write_text("ledger 2\n", encoding="utf-8")
        (self.root / "README.md").write_text("readme 2\n", encoding="utf-8")
        commit(self.root, "ledgers")
        self.assertIn("4 of 4", c.check(self.root))
        (self.root / "src/lib.rs").write_text("pub fn one() -> u32 { 2 }\n", encoding="utf-8")
        commit(self.root, "code")
        with self.assertRaisesRegex(c.ReceiptError, "no clippy receipt"):
            c.check(self.root)

    def test_a_change_of_a_clippy_line_asks_for_a_new_run(self):
        self.run_all()
        wf = self.root / ".github/workflows/ci.yml"
        wf.write_text(wf.read_text(encoding="utf-8").replace('"a,b"', '"a,b,c"'), encoding="utf-8")
        commit(self.root, "one more feature linted")
        with self.assertRaisesRegex(c.ReceiptError, "no clippy receipt"):
            c.check(self.root)

    def test_a_receipt_of_fewer_lines_or_of_none_is_refused(self):
        self.run_all()
        path = c.receipt_path(c.code_key(self.root))
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**data, "lines": data["lines"][:3], "compared": 3}), encoding="utf-8")
        with self.assertRaisesRegex(c.ReceiptError, "not of these 4 lines"):
            c.check(self.root)
        path.write_text(json.dumps({**data, "lines": [], "compared": 0}), encoding="utf-8")
        with self.assertRaises(c.ReceiptError):
            c.check(self.root)

    def test_a_checkout_that_differs_from_head_gets_no_receipt(self):
        (self.root / "src/lib.rs").write_text("pub fn one() -> u32 { 3 }\n", encoding="utf-8")
        with self.assertRaisesRegex(c.ReceiptError, "differ from HEAD"):
            self.run_all()
        with self.assertRaisesRegex(c.ReceiptError, "no clippy receipt"):
            c.check(self.root)


class ThisRepository(unittest.TestCase):
    def test_the_workflows_here_give_lines(self):
        lines = c.workflow_lines(c.ROOT)
        self.assertGreaterEqual(len(lines), 7)
        self.assertTrue(all(line["cmd"].startswith("cargo clippy") for line in lines))


if __name__ == "__main__":
    unittest.main()
