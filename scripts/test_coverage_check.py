#!/usr/bin/env python3
"""Tests for scripts/coverage_check.py: every failure path on a fixture tree, and
the real repository green."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coverage_check as cc  # noqa: E402

SRC = """//! A module.
//!
//! - Only the straight case is handled; the curved case
//!   is not supported.
// LIMITATION(COV-TST-002): Only the straight case is handled; the curved case is not supported.
pub fn straight() {}
"""

TESTS = """#[test]
fn runs() {}

#[test]
#[ignore = "runtime: about 40 s in release"]
fn slow() {}

#[test]
#[ignore = "known defect: AUD-A-S1W1-001: returns Ok for a body nothing holds"]
fn defect() {}
"""


def item(iid: str, status: str, evidence: str = "", limitation: str = "", drop: tuple[str, ...] = ()) -> str:
    fields = {
        "id": iid, "axis": "element", "item": "x", "source": "Book ch.1", "status": status,
        "evidence": evidence, "oracle_candidate": "-", "limitation": limitation,
    }
    lines = ["[[item]]"] + [f'{k} = "{v}"' for k, v in fields.items() if k not in drop]
    return "\n".join(lines) + "\n"


LIM = "src/m.rs::straight 'Only the straight case is handled; the curved case is not supported.'"
GOOD = (item("COV-TST-001", "implemented+oracle", "src/m.rs::straight / tests/t.rs::runs")
        + item("COV-TST-002", "partial", "src/m.rs", LIM)
        + item("COV-TST-003", "missing", "nothing"))


def tree(root: Path, table: str | None = GOOD, src: str | None = SRC, tests: str = TESTS,
         newline: str = "\n") -> Path:
    def write(rel: str, text: str) -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.replace("\n", newline).encode("utf-8"))
    (root / "docs" / "coverage").mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(exist_ok=True)
    if table is not None:
        write("docs/coverage/tst.toml", "# table\n\n" + table)
    if src is not None:
        write("src/m.rs", src)
    write("tests/t.rs", tests)
    return root


def run(root: Path, exempt: dict[str, str] | None = None, write_status: bool = True) -> tuple[int, str, str]:
    """`write_status` regenerates docs/coverage/status.md first, so the cases that
    are not about that file do not need to keep it in step."""
    out, err = io.StringIO(), io.StringIO()
    saved = dict(cc.MARKER_EXEMPT)
    try:
        cc.MARKER_EXEMPT.clear()
        cc.MARKER_EXEMPT.update(exempt or {})
        with redirect_stdout(out), redirect_stderr(err):
            code = cc.main(["--root", str(root), *(["--write-status"] if write_status else [])])
    finally:
        cc.MARKER_EXEMPT.clear()
        cc.MARKER_EXEMPT.update(saved)
    return code, out.getvalue(), err.getvalue()


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def assertRed(self, needle: str, **kw):
        exempt = kw.pop("exempt", None)
        code, _, err = run(tree(self.root, **kw), exempt)
        self.assertEqual(code, 1, err)
        self.assertIn(needle, err)

    def assertGreen(self, **kw):
        exempt = kw.pop("exempt", None)
        code, out, err = run(tree(self.root, **kw), exempt)
        self.assertEqual(code, 0, err)
        return out


class IdFormat(unittest.TestCase):
    def test_the_pattern_is_fixed(self):
        self.assertEqual(cc.ID_RE.pattern, r"COV-[A-Z]+-\d{3}")

    def test_accepted_and_rejected_ids(self):
        for ok in ("COV-FEM-001", "COV-CFD-120"):
            self.assertTrue(cc.ID_RE.fullmatch(ok), ok)
        for bad in ("COV-FEM-01", "COV-FEM-0001", "COV-fem-001", "EL-01", "AUD-FEM-001", "COV-FEM-001 "):
            self.assertIsNone(cc.ID_RE.fullmatch(bad), bad)

    def test_the_five_statuses(self):
        self.assertEqual(cc.STATUSES, ("implemented+oracle", "implemented-no-oracle", "partial",
                                       "missing", "out-of-scope"))


class Green(Fixture):
    def test_fixture_is_green_and_counts_by_status(self):
        out = self.assertGreen()
        self.assertIn("3 items: implemented+oracle 1 / implemented-no-oracle 0 / partial 1 / missing 1", out)
        self.assertIn("LIMITATION comments: 1 in src/ for 1 items; exempt: 0", out)

    def test_crlf_files_are_read_the_same(self):
        self.assertGreen(newline="\r\n")

    def test_known_defect_next_to_a_running_oracle_is_allowed(self):
        self.assertGreen(table=GOOD + item("COV-TST-004", "implemented+oracle",
                                           "tests/t.rs::runs / tests/t.rs::defect"))


class Items(Fixture):
    def test_duplicate_id(self):
        self.assertRed("id COV-TST-003 is also used", table=GOOD + item("COV-TST-003", "missing"))

    def test_malformed_id(self):
        self.assertRed("id does not match", table=GOOD + item("COV-TST-04", "missing"))

    def test_unknown_status(self):
        self.assertRed("status 'done' is not one of", table=GOOD + item("COV-TST-004", "done"))

    def test_empty_status(self):
        self.assertRed("status '' is not one of", table=GOOD + item("COV-TST-004", ""))

    def test_missing_key(self):
        self.assertRed("missing key `source`", table=GOOD + item("COV-TST-004", "missing", drop=("source",)))

    def test_evidence_file_that_does_not_exist(self):
        self.assertRed("src/gone.rs, which does not exist", table=GOOD + item("COV-TST-004", "missing", "src/gone.rs::x"))

    def test_a_line_reference_in_evidence_is_refused(self):
        self.assertRed("evidence src/m.rs:6 is a line reference; use `file::symbol`",
                       table=GOOD + item("COV-TST-004", "missing", "src/m.rs:6"))

    def test_evidence_test_that_does_not_exist(self):
        self.assertRed("tests/t.rs::nope, which is not a #[test] fn",
                       table=GOOD + item("COV-TST-004", "missing", "tests/t.rs::nope"))


class Oracle(Fixture):
    def test_an_ignored_oracle_does_not_count(self):
        self.assertRed("tests/t.rs::slow is #[ignore]d",
                       table=GOOD + item("COV-TST-004", "implemented+oracle", "tests/t.rs::runs / tests/t.rs::slow"))

    def test_only_ignored_oracles(self):
        self.assertRed("names no test that runs",
                       table=GOOD + item("COV-TST-004", "implemented+oracle", "tests/t.rs::slow"))

    def test_only_a_known_defect_is_not_an_oracle(self):
        self.assertRed("names no test that runs",
                       table=GOOD + item("COV-TST-004", "implemented+oracle", "tests/t.rs::defect"))

    def test_no_test_at_all(self):
        self.assertRed("names no test that runs", table=GOOD + item("COV-TST-004", "implemented+oracle", "src/m.rs::straight"))


class Limitation(Fixture):
    def test_partial_without_limitation(self):
        self.assertRed("partial needs a verbatim limitation",
                       table=GOOD.replace(f'limitation = "{LIM}"', 'limitation = ""'))

    def test_quote_not_in_the_file(self):
        self.assertRed("limitation quote is not in src/m.rs",
                       table=GOOD.replace("curved case is not", "curved case is"))

    def test_limitation_without_location(self):
        self.assertRed("limitation is not `<file>::<symbol> '<quote>'`",
                       table=GOOD.replace(LIM, "'Only the straight case is handled'"))

    def test_a_line_reference_in_a_limitation_is_refused(self):
        self.assertRed("limitation uses a line reference",
                       table=GOOD.replace(LIM, LIM.replace("src/m.rs::straight", "src/m.rs:3")))

    def test_the_quote_must_be_inside_the_named_item(self):
        src = SRC + "\n/// Another item.\npub fn other() {}\n"
        self.assertRed("limitation quote is not in src/m.rs::other",
                       table=GOOD.replace(LIM, LIM.replace("::straight", "::other")), src=src)

    def test_a_symbol_the_file_does_not_define_in_a_limitation_is_red(self):
        self.assertRed("src/m.rs::bent, which that file does not define",
                       table=GOOD.replace(LIM, LIM.replace("::straight", "::bent")))

    def test_a_bare_file_matches_the_module_documentation(self):
        self.assertGreen(table=GOOD.replace(LIM, LIM.replace("src/m.rs::straight", "src/m.rs")))

    def test_inserting_lines_above_keeps_a_symbol_reference_valid(self):
        # the point of symbol references: the same table stays green after the source grows
        self.assertGreen(src="\n" * 40 + SRC.replace("//! A module.\n", "//! A module.\n" + "//!\n" * 30))


class Markers(Fixture):
    def test_partial_without_a_comment(self):
        self.assertRed("COV-TST-002: partial item without a LIMITATION(COV-TST-002) comment",
                       src=SRC.replace("// LIMITATION(COV-TST-002)", "// note"))

    def test_comment_naming_no_item(self):
        self.assertRed("LIMITATION(COV-TST-009) names no item", src=SRC + "// LIMITATION(COV-TST-009): x\n")

    def test_comment_naming_an_item_that_is_not_partial(self):
        self.assertRed("whose status is missing, not partial", src=SRC + "// LIMITATION(COV-TST-003): x\n")

    def test_comment_with_a_malformed_id(self):
        self.assertRed("LIMITATION(EL-16) does not name an id", src=SRC + "// LIMITATION(EL-16): x\n")

    def test_exemption_replaces_the_comment(self):
        out = self.assertGreen(src=SRC.replace("// LIMITATION(COV-TST-002)", "// note"),
                               exempt={"COV-TST-002": "the file is being rewritten"})
        self.assertIn("exempt: 1", out)
        self.assertIn("COV-TST-002: the file is being rewritten", out)

    def test_stale_exemption(self):
        self.assertRed("exempt from the LIMITATION comment but has one", exempt={"COV-TST-002": "stale"})

    def test_exemption_for_an_item_that_is_not_partial(self):
        self.assertRed("COV-TST-003: exemption names no partial item", exempt={"COV-TST-003": "wrong"})


class NothingCompared(Fixture):
    def test_no_table(self):
        self.assertRed("no table", table=None)

    def test_table_without_items(self):
        self.assertRed("no [[item]]", table="")

    def test_no_source_file(self):
        self.assertRed("no source file under src/", src=None,
                       table=item("COV-TST-003", "missing", "nothing"))


class Locale(Fixture):
    """The script names its encodings; a non-UTF-8 locale (the default on Windows)
    must not change what it reads or crash its output."""

    def test_non_utf8_locale(self):
        tree(self.root, table=GOOD + item("COV-TST-004", "missing", "非 ASCII の根拠 σ_y"), newline="\r\n")
        env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONIOENCODING="", PYTHONCOERCECLOCALE="0")
        env.pop("PYTHONUTF8", None)
        cmd = [sys.executable, "-X", "utf8=0", str(HERE / "coverage_check.py"), "--root", str(self.root)]
        # the status doc written under this locale must read back the same under it
        for extra in (["--write-status"], []):
            r = subprocess.run(cmd + extra, capture_output=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
            self.assertIn(b"coverage: ok", r.stdout)


COVERED = TESTS + """
// covers: COV-TST-003
#[test]
fn closes_three() {}
"""


class Covers(Fixture):
    """test -> table: a test that runs and covers an item forces the table to say so."""

    def test_a_running_test_covering_a_missing_item_is_red(self):
        self.assertRed("tests/t.rs::closes_three runs and covers COV-TST-003, whose status is still missing",
                       tests=COVERED)

    def test_updating_the_table_turns_it_green(self):
        table = GOOD.replace(item("COV-TST-003", "missing", "nothing"),
                             item("COV-TST-003", "implemented+oracle", "tests/t.rs::closes_three"))
        out = self.assertGreen(tests=COVERED, table=table)
        self.assertIn("covers links: 1;", out)

    def test_implemented_no_oracle_is_not_enough_either(self):
        table = GOOD.replace(item("COV-TST-003", "missing", "nothing"),
                             item("COV-TST-003", "implemented-no-oracle", "src/m.rs::straight"))
        self.assertRed("whose status is still implemented-no-oracle", tests=COVERED, table=table)

    def test_an_ignored_covering_test_does_not_force_the_update(self):
        ignored = TESTS + '\n// covers: COV-TST-003\n#[test]\n#[ignore = "src gap: COV-TST-003"]\nfn ahead() {}\n'
        self.assertGreen(tests=ignored)

    def test_attributes_and_doc_comments_may_sit_between(self):
        t = TESTS + "\n// covers: COV-TST-003\n/// the closed form\n#[test]\n#[cfg(not(miri))]\nfn closes_three() {}\n"
        self.assertRed("closes_three runs and covers COV-TST-003", tests=t)

    def test_several_ids_on_one_line(self):
        t = TESTS + "\n// covers: COV-TST-001, COV-TST-003\n#[test]\nfn both() {}\n"
        self.assertRed("both runs and covers COV-TST-003", tests=t)

    def test_an_unknown_id_is_red(self):
        self.assertRed("covers COV-TST-099, which is no item", tests=TESTS + "\n// covers: COV-TST-099\n#[test]\nfn x() {}\n")

    def test_covers_on_a_fn_that_is_not_a_test_is_red(self):
        self.assertRed("which is not a #[test] fn", tests=TESTS + "\n// covers: COV-TST-001\nfn helper() {}\n")

    def test_covers_not_followed_by_a_fn_is_red(self):
        self.assertRed("is not followed by a fn", tests=TESTS + "\n// covers: COV-TST-001\nconst X: u8 = 1;\n")

    def test_a_malformed_list_is_red(self):
        self.assertRed("comma-separated", tests=TESTS + "\n// covers: COV-TST-001 and more\n#[test]\nfn x() {}\n")

    def test_prose_that_says_covers_is_not_a_link(self):
        self.assertGreen(tests="//! The unit tests already\n//! covers: the empty case\n" + TESTS)

    def test_covers_in_a_src_unit_test(self):
        src = SRC + "#[cfg(test)]\nmod tests {\n    // covers: COV-TST-003\n    #[test]\n    fn unit() {}\n}\n"
        self.assertRed("src/m.rs::unit runs and covers COV-TST-003", src=src)


class SrcGap(Fixture):
    AHEAD = TESTS + '\n#[test]\n#[ignore = "src gap: COV-TST-001: not implemented yet"]\nfn ahead() {}\n'

    def test_an_ignore_left_on_an_implemented_item_is_red(self):
        self.assertRed("still ignored as `src gap: COV-TST-001`, but COV-TST-001 is implemented+oracle",
                       tests=self.AHEAD)

    def test_an_ignore_on_a_missing_item_is_the_oracle_ahead(self):
        out = self.assertGreen(tests=self.AHEAD.replace("COV-TST-001", "COV-TST-003"))
        self.assertIn("src gap oracles: 1", out)

    def test_an_ignore_naming_no_item_is_red(self):
        self.assertRed("which is no item", tests=self.AHEAD.replace("COV-TST-001", "COV-TST-077"))


class SymbolEvidence(Fixture):
    def test_a_test_helper_in_tests_is_evidence_but_not_an_oracle(self):
        tests = TESTS + "\nfn helper() {}\n"
        self.assertGreen(tests=tests, table=GOOD + item("COV-TST-004", "missing", "tests/t.rs::helper"))
        self.assertRed("names no test that runs", tests=tests,
                       table=GOOD + item("COV-TST-004", "implemented+oracle", "tests/t.rs::helper"))

    def test_a_source_symbol_is_evidence(self):
        table = GOOD
        self.assertGreen(table=table)

    def test_a_symbol_the_file_does_not_define_is_red(self):
        table = GOOD.replace('"src/m.rs::straight / tests/t.rs::runs"', '"src/m.rs::curved / tests/t.rs::runs"')
        self.assertRed("src/m.rs::curved, which is not a #[test] fn there", table=table)

    def test_tests_still_need_a_test_fn(self):
        table = GOOD.replace("tests/t.rs::runs", "tests/t.rs::nothing")
        self.assertRed("tests/t.rs::nothing, which is not a #[test] fn there", table=table)


class StatusDoc(Fixture):
    def test_a_missing_status_doc_is_red(self):
        code, _, err = run(tree(self.root), write_status=False)
        self.assertEqual(code, 1)
        self.assertIn("docs/coverage/status.md is missing", err)

    def test_write_then_check_round_trips_and_a_table_edit_makes_it_stale(self):
        tree(self.root)
        self.assertEqual(run(self.root)[0], 0)
        self.assertEqual(run(self.root, write_status=False)[0], 0)
        text = (self.root / "docs/coverage/status.md").read_text(encoding="utf-8")
        self.assertIn("| `docs/coverage/tst.toml` | 3 | 1 | 0 | 1 | 1 | 0 |", text)
        self.assertIn("| element | 3 | 1 | 0 | 1 | 1 | 0 |", text)
        t = self.root / "docs/coverage/tst.toml"
        t.write_text(t.read_text(encoding="utf-8").replace('"missing"', '"out-of-scope"', 1), encoding="utf-8")
        code, _, err = run(self.root, write_status=False)
        self.assertEqual(code, 1)
        self.assertIn("docs/coverage/status.md is stale", err)

    def test_every_table_gets_a_row_and_a_total(self):
        tree(self.root)
        (self.root / "docs/coverage/two.toml").write_text(item("COV-TWO-001", "missing", "nothing"), encoding="utf-8")
        self.assertEqual(run(self.root)[0], 0)
        text = (self.root / "docs/coverage/status.md").read_text(encoding="utf-8")
        self.assertIn("| `docs/coverage/two.toml` | 1 | 0 | 0 | 0 | 1 | 0 |", text)
        self.assertIn("| **total** | 4 | 1 | 0 | 1 | 2 | 0 |", text)
        self.assertIn("## `docs/coverage/two.toml`", text)


class RealRepo(unittest.TestCase):
    def test_the_repository_is_green(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cc.main(["--root", str(HERE.parent)])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("docs/coverage/fem.toml:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
