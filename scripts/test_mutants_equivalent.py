#!/usr/bin/env python3
"""Tests for scripts/mutants_equivalent.py and its use in scripts/mutants_ratchet.py
(one rule per case)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_equivalent as me  # noqa: E402
import mutants_ratchet as mr  # noqa: E402

SRC = "fn f() {\n    let mut s = 0;\n    s = s + x;\n    s\n}\n"
LINE = "    s = s + x;"
MUT = "replace + with - in f"
MISSED = f"src/a.rs:3:11: {MUT}"
REASON = "s is zero before this line and only its absolute value is used"


def entry(pos: str = "src/a.rs:3:11", digest: str | None = None, mutation: str = MUT,
          reason: str = REASON) -> str:
    return f"{pos}\t{digest or me.line_digest(LINE)}\t{mutation}\t{reason}\n"


class Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        self.write_src(SRC)
        self.list = self.root / "equivalent.txt"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_src(self, text: str) -> None:
        (self.root / "src" / "a.rs").write_text(text, encoding="utf-8")

    def write_list(self, *lines: str) -> None:
        self.list.write_text("# comment\n" + "".join(lines), encoding="utf-8")

    def check(self) -> list[str]:
        return me.check(self.list, self.root)

    def filtered(self, *missed: str) -> tuple[list[str], list[str]]:
        return me.split_missed(list(missed), me.valid_keys(self.list, self.root))


class CheckTest(Case):
    def test_a_valid_entry_passes(self) -> None:
        self.write_list(entry())
        self.assertEqual(self.check(), [])

    def test_an_empty_list_fails(self) -> None:
        self.write_list()
        self.assertTrue(any("checked nothing" in e for e in self.check()))

    def test_a_missing_list_fails(self) -> None:
        self.assertTrue(any("does not exist" in e for e in self.check()))

    def test_an_entry_without_a_reason_fails(self) -> None:
        self.write_list(entry(reason=""))
        self.assertTrue(any("empty reason" in e for e in self.check()))

    def test_a_wrong_field_count_fails(self) -> None:
        self.write_list(f"src/a.rs:3:11\t{me.line_digest(LINE)}\t{MUT}\n")
        self.assertTrue(any("4 or 5 tab-separated" in e for e in self.check()))

    def test_a_duplicate_entry_fails(self) -> None:
        self.write_list(entry(), entry())
        self.assertTrue(any("duplicate" in e for e in self.check()))

    def test_an_edited_line_makes_the_entry_stale(self) -> None:
        self.write_list(entry())
        self.write_src(SRC.replace("s + x", "s + y"))
        self.assertTrue(any("changed" in e for e in self.check()))

    def test_a_column_that_is_not_the_operator_fails(self) -> None:
        self.write_list(entry(pos="src/a.rs:3:9"))
        self.assertTrue(any("does not start with '+'" in e for e in self.check()))

    def test_a_line_past_the_end_fails(self) -> None:
        self.write_list(entry(pos="src/a.rs:30:11"))
        self.assertTrue(any("has 5 lines" in e for e in self.check()))


class PinningTestsTest(Case):
    TESTS = """\
#[test]
fn plain() {}

#[test]
#[should_panic(expected = "x")]
fn panics() {}
"""

    def setUp(self) -> None:
        super().setUp()
        (self.root / "tests").mkdir()
        (self.root / "tests" / "pin.rs").write_text(self.TESTS, encoding="utf-8")

    def test_named_tests_that_exist_pass(self) -> None:
        self.write_list(entry().rstrip("\n") + "\ttests/pin.rs::plain, tests/pin.rs::panics\n")
        self.assertEqual(self.check(), [])

    def test_a_named_test_that_is_gone_makes_the_entry_stale(self) -> None:
        self.write_list(entry().rstrip("\n") + "\ttests/pin.rs::plain,tests/pin.rs::renamed\n")
        self.assertTrue(any("is gone: tests/pin.rs::renamed" in e for e in self.check()))

    def test_a_function_without_the_test_attribute_does_not_count(self) -> None:
        (self.root / "tests" / "pin.rs").write_text("fn plain() {}\n", encoding="utf-8")
        self.write_list(entry().rstrip("\n") + "\ttests/pin.rs::plain\n")
        self.assertTrue(any("is gone" in e for e in self.check()))

    def test_a_named_test_file_that_is_gone_makes_the_entry_stale(self) -> None:
        self.write_list(entry().rstrip("\n") + "\ttests/missing.rs::plain\n")
        self.assertTrue(any("is gone" in e for e in self.check()))

    def test_a_malformed_test_reference_fails(self) -> None:
        self.write_list(entry().rstrip("\n") + "\tplain\n")
        self.assertTrue(any("tests/<file>.rs::<fn>" in e for e in self.check()))

    def test_a_stale_pin_does_not_filter(self) -> None:
        self.write_list(entry().rstrip("\n") + "\ttests/pin.rs::renamed\n")
        self.assertEqual(self.filtered(MISSED), ([MISSED], []))


class FilterTest(Case):
    def test_a_listed_mutant_is_equivalent(self) -> None:
        self.write_list(entry())
        self.assertEqual(self.filtered(MISSED), ([], [MISSED]))

    def test_a_coloured_line_is_matched(self) -> None:
        self.write_list(entry())
        coloured = "src/a.rs:3:11: replace \x1b[33m+\x1b[0m with \x1b[38;5;11m-\x1b[0m in f"
        self.assertEqual(self.filtered(coloured), ([], [coloured]))

    def test_another_mutation_at_the_same_position_is_missed(self) -> None:
        self.write_list(entry())
        other = "src/a.rs:3:11: replace + with * in f"
        self.assertEqual(self.filtered(other), ([other], []))

    def test_a_moved_mutant_is_missed(self) -> None:
        # a line inserted above moves the mutant to line 4: the entry keyed by
        # line 3 neither matches it nor stays valid
        self.write_list(entry())
        self.write_src("// moved\n" + SRC)
        moved = f"src/a.rs:4:11: {MUT}"
        self.assertEqual(self.filtered(moved, MISSED), ([moved, MISSED], []))

    def test_a_stale_entry_does_not_filter(self) -> None:
        self.write_list(entry())
        self.write_src(SRC.replace("s + x", "s + y"))
        self.assertEqual(self.filtered(MISSED), ([MISSED], []))


class RepositoryListTest(unittest.TestCase):
    def test_the_repository_list_is_current_and_not_empty(self) -> None:
        entries, errors = me.load()
        self.assertEqual(me.check(), [])
        self.assertEqual(errors, [])
        self.assertGreater(len(entries), 0)


class RatchetTest(Case):
    def run_dir(self, missed: list[str]) -> Path:
        d = self.root / "out-default" / "mutants.out"
        d.mkdir(parents=True)
        (d / "missed.txt").write_text("".join(f"{m}\n" for m in missed), encoding="utf-8")
        (d / "caught.txt").write_text("src/a.rs:4:5: replace f with ()\n", encoding="utf-8")
        n = len(missed) + 1
        (d / "mutants.json").write_text(json.dumps([{}] * n), encoding="utf-8")
        (d / "outcomes.json").write_text(json.dumps({
            "outcomes": [{"scenario": "Baseline"}] + [{"scenario": {"Mutant": {}}}] * n,
            "end_time": "2026-10-11T00:00:00Z",
        }), encoding="utf-8")
        return self.root / "out-default"

    def ratchet(self, missed: list[str]) -> int:
        base = self.root / "baseline.txt"
        base.write_text("", encoding="utf-8")
        return mr.main(["--out", str(self.run_dir(missed)), "--baseline", str(base),
                        "--equivalent", str(self.list), "--source-root", str(self.root)])

    def test_an_equivalent_mutant_is_not_newly_missed(self) -> None:
        self.write_list(entry())
        self.assertEqual(self.ratchet([MISSED]), 0)

    def test_a_stale_entry_reads_newly_missed(self) -> None:
        self.write_list(entry())
        self.write_src(SRC.replace("s + x", "s + y"))
        self.assertEqual(self.ratchet([MISSED]), 1)

    def test_an_equivalent_mutant_is_not_written_to_the_baseline(self) -> None:
        self.write_list(entry())
        base = self.root / "baseline.txt"
        mr.main(["--out", str(self.run_dir([MISSED])), "--baseline", str(base),
                 "--equivalent", str(self.list), "--source-root", str(self.root), "--write"])
        self.assertNotIn("replace + with -", base.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
