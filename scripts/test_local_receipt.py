#!/usr/bin/env python3
"""Tests of scripts/local_receipt.py: the steps it reads from a workflow, how the rules sort
them, what its key moves with, and each way `check` refuses."""
from __future__ import annotations

import json
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
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

SLOW_WORKFLOW = WORKFLOW + "      - run: cargo llvm-cov --lib\n      - run: python3 scripts/ratchet.py\n"
SLOW_RULES = "slow\t^cargo llvm-cov\\b|^python3 scripts/ratchet\\.py$\tthe lib coverage and its ratchet: half an hour here\n"

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
    (root / "src/lib.rs").write_text("pub fn one() -> u32 { 1 }\n#[cfg(test)]\n#[path = \"../tests/in_lib.rs\"]\nmod in_lib;\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs/oracle-status.md").write_text("ledger 1\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests/in_lib.rs").write_text('// included\n#[path = "support/deeper.rs"]\nmod deeper;\n', encoding="utf-8")
    (root / "tests/support").mkdir()
    (root / "tests/support/deeper.rs").write_text("// included by an included file\n", encoding="utf-8")
    (root / "tests/support/apart.rs").write_text("// of the test binaries only\n", encoding="utf-8")
    (root / "tests/apart.rs").write_text("// its own binary\n", encoding="utf-8")
    (root / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    (root / "README.md").write_text("readme 1\n", encoding="utf-8")
    sh(root, "init", "-q")
    commit(root, "first")
    sh(root, "branch", "base")
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
        slow = "slow\t^python3 scripts/\\w+\\.py\ta slow checker: runs here in an hour\n" + RULES
        with self.assertRaisesRegex(c.ReceiptError, "a `slow` rule .* matches a step that reads the runner"):
            self.sort(slow.replace("ci\t\\$\\{\\{\treads the event: only the CI has it\n", ""))
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
        self.base, c.BASE = c.BASE, "base"

    def tearDown(self):
        c.BASE = self.base
        if self.old is None:
            os.environ.pop("LOCAL_RECEIPT_DIR", None)
        else:
            os.environ["LOCAL_RECEIPT_DIR"] = self.old
        self.tmp.cleanup()

    def run_all(self, red: str | None = None, only: str | None = None, cls: str = "local") -> list:
        seen = []

        def runner(n, unit):
            seen.append(unit)
            return 101 if red and red in unit["cmd"] else 0
        c.run(self.root, runner, only=only, cls=cls)
        return seen

    def slow_repo(self, change: str) -> None:
        """the fixture with two slow steps, and one commit after `base` that changes `change`"""
        self.root = repo(str(Path(self.tmp.name) / change.replace("/", "-")), SLOW_WORKFLOW, SLOW_RULES + RULES)
        with open(self.root / change, "a", encoding="utf-8") as f:
            f.write("# changed\n" if change.endswith((".yml", ".toml", ".txt")) else "// changed\n")
        commit(self.root, "the series")

    def at_home(self):
        """the default directory, under a HOME of this test (no LOCAL_RECEIPT_DIR)"""
        home = Path(self.tmp.name) / "home"
        home.mkdir()
        del os.environ["LOCAL_RECEIPT_DIR"]   # (tearDown puts back what setUp found)
        self.home_is(home)
        return home

    def home_is(self, home: Path) -> None:
        at = mock.patch.object(Path, "home", return_value=home)
        at.start()
        self.addCleanup(at.stop)

    def test_the_default_place_is_a_state_directory_and_no_part_of_it_is_a_cache(self):
        home = self.at_home()
        d = c.receipt_dir()
        self.assertEqual(d, home / ".local" / "state" / "alice" / "local-receipts")
        self.assertFalse(any("cache" in part.lower() for part in d.relative_to(home).parts), d)
        self.assertNotEqual(d, c.old_receipt_dir())
        self.assertIn(".cache", c.old_receipt_dir().parts)
        self.run_all()
        self.assertEqual([p.parent for p in d.glob("*.json")], [d], "run writes the receipt in the default place")
        self.assertFalse(c.old_receipt_dir().exists(), "and nothing in the old one")
        self.assertIn("steps green", c.check(self.root))

    def test_a_receipt_in_the_old_place_is_read_once_with_a_warning_and_then_from_the_new_place(self):
        home = self.at_home()
        self.run_all()
        new = next(c.receipt_dir().glob("*.json"))
        old = c.old_receipt_dir() / new.name
        old.parent.mkdir(parents=True)
        new.rename(old)
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):   # (another name in the old place is not this one)
            (old.parent / "other.json").write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
            old.rename(old.parent / "kept.json")
            c.check(self.root)
        (old.parent / "kept.json").rename(old)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            line = c.check(self.root)
        self.assertIn("steps green", line)
        self.assertIn(f"warning: {new.name} was read from the old place {old.parent}", err.getvalue())
        self.assertTrue(old.is_file() and new.is_file(), "copied, the old file is left")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(c.check(self.root), line)
        self.assertEqual(err.getvalue(), "", "once: the second check reads the new place")
        # a red receipt in the old place is as red after the copy
        new.unlink()
        data = json.loads(old.read_text(encoding="utf-8"))
        data["steps"][0]["rc"] = 1
        old.write_text(json.dumps(data), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(c.ReceiptError, "not green"):
            c.check(self.root)

    def test_a_directory_given_by_the_environment_never_reads_the_old_place(self):
        home = Path(self.tmp.name) / "home"
        home.mkdir()
        self.run_all()
        new = next(c.receipt_dir().glob("*.json"))
        self.home_is(home)
        (c.old_receipt_dir()).mkdir(parents=True)
        new.rename(c.old_receipt_dir() / new.name)
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):
            c.check(self.root)

    def test_a_series_that_changes_src_needs_the_slow_receipt_and_another_does_not(self):
        self.slow_repo("scripts/tool.py")
        self.assertEqual(len(self.run_all()), 4, "the slow steps are not among the local ones")
        self.assertIn("slow receipt not asked", c.check(self.root))
        self.slow_repo("src/lib.rs")
        # neither run yet: the one refusal names both, so both can be queued
        with self.assertRaises(c.ReceiptError) as both:
            c.check(self.root)
        said = str(both.exception)
        self.assertRegex(said, r"^2 receipts are missing or not green:\n  - no receipt for the code of HEAD .*local_receipt\.py run` "
                               r".*\n  - no slow receipt .*run --slow` ")
        self.run_all(cls="slow")
        with self.assertRaisesRegex(c.ReceiptError, r"^no receipt for the code of HEAD"):   # (the slow one is held)
            c.check(self.root)
        next(Path(os.environ["LOCAL_RECEIPT_DIR"]).glob("*.slow.json")).unlink()
        self.run_all()
        with self.assertRaisesRegex(c.ReceiptError, r"^no slow receipt.*run --slow.*2 slow steps"):
            c.check(self.root)
        seen = self.run_all(cls="slow")
        self.assertEqual([u["cmd"] for u in seen], ["cargo llvm-cov --lib", "python3 scripts/ratchet.py"],
                         "in the workflow's order: the ratchet reads what the step before wrote")
        self.assertRegex(c.check(self.root), r"slow receipt: 2 of 2 steps green.*1 input\(s\).*src/lib\.rs")

    def test_every_input_of_the_lib_coverage_asks_for_the_slow_receipt(self):
        for change, asked in (("tests/in_lib.rs", True), ("tests/apart.rs", False), ("tests/support/deeper.rs", True),
                              ("tests/support/apart.rs", False), ("Cargo.toml", True),
                              ("build.rs", True), ("scripts/line-coverage-baseline.txt", True),
                              (".github/workflows/ci.yml", True), ("README.md", False)):
            self.slow_repo(change)
            self.assertEqual(c.needs_slow(self.root) == [change], asked, change)

    def test_an_include_that_names_no_file_is_an_error(self):
        self.slow_repo("src/lib.rs")
        (self.root / "src/lib.rs").write_text('#[path = "../tests/gone.rs"]\nmod gone;\n', encoding="utf-8")
        commit(self.root, "an include of nothing")
        with self.assertRaisesRegex(c.ReceiptError, r"src/lib\.rs: .*names tests/gone\.rs"):
            c.needs_slow(self.root)

    def test_a_red_slow_step_and_a_slow_receipt_alone_are_refused(self):
        self.slow_repo("src/lib.rs")
        self.run_all(cls="slow")
        with self.assertRaisesRegex(c.ReceiptError, "no receipt for the code of HEAD"):
            c.check(self.root)
        self.run_all()
        self.run_all(red="ratchet", cls="slow")
        with self.assertRaisesRegex(c.ReceiptError, "1 of 2 slow step"):
            c.check(self.root)

    def test_a_base_that_cannot_be_read_and_src_changed_without_a_slow_step_are_errors(self):
        (self.root / "src/lib.rs").write_text("pub fn one() -> u32 { 4 }\n", encoding="utf-8")
        commit(self.root, "code")
        self.run_all()
        with self.assertRaisesRegex(c.ReceiptError, "no slow step"):
            c.check(self.root)
        with self.assertRaisesRegex(c.ReceiptError, "merge-base nowhere/main"):
            c.check(self.root, "nowhere/main")

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
        (self.root / "docs/oracle-status.md").write_text("ledger 2\n", encoding="utf-8")
        commit(self.root, "a ledger the landing writes")
        self.assertIn("4 of 4", c.check(self.root))
        (self.root / "README.md").write_text("readme 2\n", encoding="utf-8")
        commit(self.root, "a document the checkers read")
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):
            c.check(self.root)
        self.run_all()
        (self.root / "src/lib.rs").write_text("pub fn one() -> u32 { 2 }\n", encoding="utf-8")
        commit(self.root, "code")
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):
            c.check(self.root)

    MODULES = ("# Modules\n\n| Module | Role | Integration |\n|--------|------|-------------|\n"
               "| `alpha` | first | {a} |\n| `beta` | second | {b} |\n\ntext after the table\n")

    def modules(self, **cells: str) -> None:
        (self.root / "docs/MODULES.md").write_text(self.MODULES.format(**cells), encoding="utf-8")

    def test_the_generated_column_of_the_module_table_does_not_ask_for_a_new_run_and_its_other_cells_do(self):
        # the landing rewrites the Integration column of docs/MODULES.md from the tree (as it
        # rewrites the ledgers); the rest of the file is hand-written and read by the checkers
        self.modules(a="L3", b="L0")
        commit(self.root, "the module table")
        self.run_all()
        self.modules(a="L3", b="L2")
        commit(self.root, "the column the landing regenerated")
        self.assertIn("4 of 4", c.check(self.root))
        (self.root / "docs/MODULES.md").write_text(
            self.MODULES.format(a="L3", b="L2").replace("| second |", "| second, reworded |"), encoding="utf-8")
        commit(self.root, "a hand-written cell")
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):
            c.check(self.root)
        self.run_all()
        (self.root / "docs/MODULES.md").write_text(
            self.MODULES.format(a="L3", b="L2").replace("text after the table", "L1 outside any table"), encoding="utf-8")
        commit(self.root, "text outside the table")
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
        sh(self.root, "checkout", "-q", "--", "src/lib.rs")
        (self.root / "src/new_module.rs").write_text("pub fn two() -> u32 { 2 }\n", encoding="utf-8")
        with self.assertRaisesRegex(c.ReceiptError, r"not in it \(src/new_module\.rs"):
            self.run_all()
        (self.root / "src/new_module.rs").unlink()
        with self.assertRaisesRegex(c.ReceiptError, "no receipt"):
            c.check(self.root)
        (self.root / "docs/oracle-status.md").write_text("regenerated\n", encoding="utf-8")
        self.assertEqual(len(self.run_all()), 4, "a ledger the landing writes may differ")


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
        import land
        self.assertEqual(c.LANDING_WRITES, land.LOCAL_LEDGERS | {"docs/PUBLIC_API_SNAPSHOT.txt"},
                         "the receipt outlives what a landing writes, and nothing else")
        includes, holders = c.slow_inputs(c.ROOT)
        self.assertGreaterEqual(len(includes), 10, "the tests/ files the lib includes are read from src/")
        self.assertIn("tests/support/law_rows.rs", includes, "an include of an included file")
        self.assertEqual(holders, {".github/workflows/security-audit.yml"})
        slow = [u["cmd"] for u in c.local_steps(c.ROOT, "slow")]
        self.assertEqual(len(slow), 2, "the push's lib coverage and its ratchet, in this order")
        self.assertIn("cargo llvm-cov --lib --summary-only", slow[0])
        self.assertEqual(slow[1], "python3 scripts/line_coverage_ratchet.py --summary coverage-summary.txt")


if __name__ == "__main__":
    unittest.main()
