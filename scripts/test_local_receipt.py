#!/usr/bin/env python3
"""Tests of scripts/local_receipt.py: the steps it reads from a workflow, how the rules sort
them, what its key moves with, and each way `check` refuses."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_receipt as c  # noqa: E402

WORKFLOW = """\
env:
  CARGO_TERM_COLOR: always

jobs:
  lint:
    env:
      RUSTDOCFLAGS: -Dwarnings
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
      - name: a test leg
        run: cargo test
      - name: literal with env
        env:
          CLIPPY_CONF_DIR: .github/clippy-lib
        working-directory: web
        run: |
          cargo clippy --lib --features "wasm"
          if true; then
            cargo clippy --lib --no-default-features
          fi
  other:
    steps:
      - run: python3 scripts/decide.py
        env:
          EVENT: ${{ github.event_name }}
      - name: doc
        run: cargo doc --lib
"""

RULES = """\
# class<TAB>regex<TAB>reason
ci\t\\$\\{\\{\treads the event: only the CI has it
ci\t^cargo test\\b\ta test leg: the CI's evidence
local\t^cargo (clippy|doc)\\b\ta lint or a doc build: runs here
"""


def repo(tmp: str, workflow: str = WORKFLOW, rules: str = RULES) -> Path:
    root = Path(tmp) / "repo"
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/ci.yml").write_text(workflow, encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts/local-receipt-rules.tsv").write_text(rules, encoding="utf-8")
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


class Steps(unittest.TestCase):
    def test_every_run_is_read_with_its_command_env_and_directory(self):
        found = c.steps(WORKFLOW, "ci.yml")
        self.assertEqual([f["cmd"] for f in found], [
            "cargo clippy --all-targets -- -D warnings",
            'cargo clippy --features "a,b" --all-targets -- -D warnings',
            "cargo test",
            'cargo clippy --lib --features "wasm"\nif true; then\n  cargo clippy --lib --no-default-features\nfi',
            "python3 scripts/decide.py",
            "cargo doc --lib",
        ])
        lint = {"CARGO_TERM_COLOR": "always", "RUSTDOCFLAGS": "-Dwarnings"}
        self.assertEqual(found[0]["env"], lint, "the workflow's env and the job's")
        self.assertEqual(found[3]["env"], {**lint, "CLIPPY_CONF_DIR": ".github/clippy-lib"}, "and the step's")
        self.assertEqual(found[3]["cwd"], "web")
        self.assertEqual(found[4]["env"], {"CARGO_TERM_COLOR": "always", "EVENT": "${{ github.event_name }}"},
                         "the other job's env is not this job's; env after `run:` is read")
        self.assertEqual(found[5]["env"], {"CARGO_TERM_COLOR": "always"})

    def test_no_step_read_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = repo(tmp, "name: nothing\n")
            with self.assertRaisesRegex(c.ReceiptError, "no `run:` step read from 1 workflow"):
                c.workflow_steps(root)


class Sorting(unittest.TestCase):
    def sort(self, rules: str = RULES, workflow: str = WORKFLOW) -> list:
        return c.sort_steps(c.steps(workflow, "ci.yml"), c.read_rules(rules))

    def test_the_first_matching_rule_decides_and_the_env_is_part_of_what_a_rule_reads(self):
        self.assertEqual([s["class"] for s in self.sort()], ["local", "local", "ci", "local", "ci", "local"])

    def test_a_step_without_a_rule_is_an_error(self):
        with self.assertRaisesRegex(c.ReceiptError, r"ci\.yml:\d+: no rule for this step.*cargo fmt"):
            self.sort(workflow=WORKFLOW + "      - run: cargo fmt --check\n")

    def test_a_rule_that_matches_no_step_is_an_error(self):
        with self.assertRaisesRegex(c.ReceiptError, r"rules\.tsv:5: the rule matches no step"):
            self.sort(RULES + "ci\t^cargo bench\\b\ta bench leg: nothing has it now\n")

    def test_a_local_rule_may_not_take_a_step_that_reads_the_runner(self):
        greedy = "local\t^python3 scripts/\\w+\\.py\ta checker: runs here in seconds\n" + RULES
        with self.assertRaisesRegex(c.ReceiptError, "matches a step that reads the runner"):
            self.sort(greedy.replace("ci\t\\$\\{\\{\treads the event: only the CI has it\n", "")
                      + "ci\t^never\tno step: this row is stale but the error above comes too\n")
        pwsh = WORKFLOW + "      - run: cargo clippy --win\n        shell: pwsh\n"
        with self.assertRaisesRegex(c.ReceiptError, "'pwsh'"):
            self.sort(workflow=pwsh)

    def test_a_rule_needs_a_class_a_regex_and_a_reason(self):
        for bad in ("maybe\t^x\ta reason long enough\n", "ci\t^x\tshort\n", "ci\t^(x\ta reason long enough\n",
                    "# only a comment\n"):
            with self.assertRaises(c.ReceiptError, msg=bad):
                c.read_rules(bad)


class Receipt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = repo(self.tmp.name)
        self.old = os.environ.get("LOCAL_RECEIPT_DIR")
        os.environ["LOCAL_RECEIPT_DIR"] = str(Path(self.tmp.name) / "receipts")

    def tearDown(self):
        if self.old is None:
            del os.environ["LOCAL_RECEIPT_DIR"]
        else:
            os.environ["LOCAL_RECEIPT_DIR"] = self.old
        self.tmp.cleanup()

    def run_all(self, red: str | None = None, only: str | None = None) -> list:
        seen = []

        def runner(n, unit):
            seen.append(unit)
            return 101 if red and red in unit["cmd"] else 0
        c.run(self.root, runner, only=only)
        return seen

    def test_without_a_receipt_the_check_refuses_and_with_a_green_one_it_passes(self):
        with self.assertRaisesRegex(c.ReceiptError, "no receipt for the code of HEAD"):
            c.check(self.root)
        seen = self.run_all()
        self.assertEqual(len(seen), 4, "every local step is run, the CI's are not")
        self.assertEqual(seen[2]["cwd"], "web")
        self.assertEqual(seen[2]["env"]["CLIPPY_CONF_DIR"], ".github/clippy-lib")
        self.assertIn("4 of 4 steps green", c.check(self.root))

    def test_the_shell_runs_a_step_in_its_directory_with_its_env_and_stops_it_at_a_red_line(self):
        (self.root / "web").mkdir()
        logs = Path(self.tmp.name) / "logs"
        logs.mkdir()
        runner = c.shell(self.root, logs)
        unit = {"cmd": 'echo "$PWD $FLAG"\nfalse\necho after', "env": {"FLAG": "set"}, "cwd": "web"}
        self.assertNotEqual(runner(1, unit), 0, "a red line inside a step is the step's result")
        out = (logs / "001.log").read_text(encoding="utf-8")
        self.assertRegex(out, r"/web set\n$")

    def test_a_red_step_does_not_stop_the_next_and_the_check_refuses(self):
        seen = self.run_all(red="--all-targets -- -D warnings")
        self.assertEqual(len(seen), 4, "the steps after the red ones are still run")
        with self.assertRaisesRegex(c.ReceiptError, "2 of 4 local step"):
            c.check(self.root)

    def test_a_change_of_code_asks_for_a_new_run_and_a_change_of_a_ledger_does_not(self):
        self.run_all()
        (self.root / "docs/ledger.md").write_text("ledger 2\n", encoding="utf-8")
        (self.root / "README.md").write_text("readme 2\n", encoding="utf-8")
        commit(self.root, "ledgers")
        self.assertIn("4 of 4", c.check(self.root))
        (self.root / "src/lib.rs").write_text("pub fn one() -> u32 { 2 }\n", encoding="utf-8")
        commit(self.root, "code")
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):
            c.check(self.root)

    def test_a_step_added_to_a_workflow_asks_for_a_new_run(self):
        self.run_all()
        wf = self.root / ".github/workflows/ci.yml"
        wf.write_text(wf.read_text(encoding="utf-8") + "      - run: cargo doc --lib --features x\n", encoding="utf-8")
        commit(self.root, "one more doc build")
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):
            c.check(self.root)
        self.assertEqual(len(self.run_all()), 5)

    def test_a_receipt_of_fewer_steps_or_of_none_is_refused(self):
        self.run_all()
        path = c.receipt_path(c.code_key(self.root))
        data = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**data, "steps": data["steps"][:3], "compared": 3}), encoding="utf-8")
        with self.assertRaisesRegex(c.ReceiptError, "not of these 4 steps"):
            c.check(self.root)
        path.write_text(json.dumps({**data, "steps": [], "compared": 0}), encoding="utf-8")
        with self.assertRaises(c.ReceiptError):
            c.check(self.root)

    def test_rules_that_leave_nothing_to_run_here_are_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = repo(tmp, rules="ci\t^(cargo|python3) \tevery step is the CI's: nothing runs here\n")
            with self.assertRaisesRegex(c.ReceiptError, "no local step"):
                c.local_steps(root)
            with self.assertRaisesRegex(c.ReceiptError, "no local step"):
                c.check(root)

    def test_a_partial_run_writes_no_receipt(self):
        self.assertEqual(len(self.run_all(only="cargo doc")), 1)
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):
            c.check(self.root)

    def test_a_checkout_that_differs_from_head_gets_no_receipt(self):
        (self.root / "src/lib.rs").write_text("pub fn one() -> u32 { 3 }\n", encoding="utf-8")
        with self.assertRaisesRegex(c.ReceiptError, "differ from HEAD"):
            self.run_all()
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):
            c.check(self.root)


class ThisRepository(unittest.TestCase):
    def test_every_step_of_the_workflows_here_has_a_rule_and_every_rule_a_step(self):
        units = c.local_steps(c.ROOT)
        cmds = [u["cmd"] for u in units]
        self.assertGreaterEqual(len(units), 100)
        for wanted in ("cargo clippy --all-targets -- -D warnings", "cargo doc --lib --no-deps",
                       "python3 scripts/g2_comparisons.py", "cargo fmt --all -- --check",
                       "python scripts/run_examples.py"):
            self.assertIn(wanted, cmds)
        doc = next(u for u in units if u["cmd"] == "cargo doc --lib --no-deps")
        self.assertEqual(doc["env"].get("RUSTDOCFLAGS"), "-Dwarnings", "the job's env reaches the step")
        self.assertFalse([cmd for cmd in cmds if cmd.startswith("cargo test")], "no test leg is run here")


if __name__ == "__main__":
    unittest.main()
