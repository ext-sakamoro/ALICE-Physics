#!/usr/bin/env python3
"""scripts/run_feature_gated_tests.py の oracle.

検査器の空振り (比較件数 0 で green) を最も危ない形として固定する:
(1) 全 target が 1 本以上走れば green (2) gate が開かず 0 本の target が 1 つでも
あれば red (3) 結果行が無い target は red (4) 選択 0 件 / 実行合計 0 は red
(5) 一覧の feature が 1 file も選ばなければ red (6) 失敗があれば red
(7) file 単位と item 単位の gate の両方を選び、無関係な feature は選ばない
期待値は fixture の構造から決まり、cargo は呼ばない

run: python3 scripts/test_run_feature_gated_tests.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_feature_gated_tests as g  # noqa: E402

FEATS = ("replay", "gpu-solver-bridge")


def tests_dir(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="gated-fixture-"))
    for name, body in files.items():
        (root / name).write_text(body, encoding="utf-8")
    return root


def output(blocks: list[tuple[str, int, int, int]]) -> str:
    lines = []
    for name, p, f, i in blocks:
        lines.append(f"     Running tests/{name}.rs (target/debug/deps/{name}-0123abcd)")
        lines.append("")
        lines.append(f"running {p + f + i} tests")
        lines.append(
            f"test result: {'ok' if f == 0 else 'FAILED'}. {p} passed; {f} failed; "
            f"{i} ignored; 0 measured; 0 filtered out; finished in 0.00s"
        )
    return "\n".join(lines)


class Select(unittest.TestCase):
    def test_file_and_item_gates_are_selected_and_others_are_not(self):
        d = tests_dir({
            "a.rs": '#![cfg(all(feature = "std", feature = "replay"))]\n',
            "b.rs": 'mod m {\n#[cfg(feature = "gpu-solver-bridge")]\nfn x() {}\n}\n',
            "c.rs": '#![cfg(feature = "std")]\n',
            "d.rs": '#[cfg(feature = "parallel")]\nfn y() {}\n',
        })
        self.assertEqual(
            g.select(d, FEATS), {"a": {"replay"}, "b": {"gpu-solver-bridge"}}
        )


class Parse(unittest.TestCase):
    def test_counts_per_target_including_windows_paths(self):
        out = output([("a", 3, 0, 1)]) + "\n" + (
            "     Running tests\\b.rs (target\\debug\\deps\\b-ff.exe)\n"
            "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
        )
        self.assertEqual(g.parse(out), {"a": (3, 0, 1), "b": (2, 0, 0)})


class CiOutput(unittest.TestCase):
    """The forms CI produces: CARGO_TERM_COLOR=always wraps "Running" in ANSI
    codes, Windows uses backslash paths and CRLF, and its console is cp1252."""

    ESC = "\x1b"

    def coloured(self, path: str, eol: str = "\n") -> str:
        e = self.ESC
        return (f"{e}[1m{e}[92m     Running{e}[0m {path} (target/debug/deps/x-0123){eol}"
                f"{eol}running 3 tests{eol}"
                f"test result: {e}[32mok{e}[0m. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out{eol}")

    def test_libtest_sgr0_after_the_result_word_parses(self):
        # libtest with `--color always` ends the coloured word with ESC ( B ESC [ m
        out = self.coloured("tests/a.rs").replace(f"{self.ESC}[0m. 3 passed", f"{self.ESC}(B{self.ESC}[m. 3 passed")
        self.assertIn(f"{self.ESC}(B", out)
        self.assertEqual(g.parse(out), {"a": (3, 0, 0)})

    def test_coloured_running_lines_parse(self):
        out = self.coloured("tests/a.rs") + self.coloured("tests\\b.rs", "\r\n")
        self.assertEqual(g.parse(out), {"a": (3, 0, 0), "b": (3, 0, 0)})

    def test_unparseable_output_is_still_red_through_the_zero_guard(self):
        # a future cargo format the regex misses must not read as green
        counts = g.parse("     Running-ish something else\ntest outcome: fine\n")
        problems = g.check({"a": {"replay"}, "b": {"gpu-solver-bridge"}}, counts, FEATS)
        self.assertTrue(any("0 tests in total" in p for p in problems), problems)

    def test_emit_survives_a_cp1252_stream(self):
        import io
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
        g.emit("ok \u2713 \u03b5 \u2192 \u5b8c\u4e86\n", stream)
        stream.flush()
        self.assertIn("\u2713".encode("utf-8"), raw.getvalue())

    def test_emit_on_a_stream_without_a_buffer_replaces_unencodable_text(self):
        import io

        class Narrow(io.StringIO):
            encoding = "cp1252"

        s = Narrow()
        g.emit("\u5b8c ok\n", s)
        self.assertEqual(s.getvalue(), "? ok\n")


class Check(unittest.TestCase):
    picked = {"a": {"replay"}, "b": {"gpu-solver-bridge"}}

    def test_green_when_every_target_executes(self):
        counts = g.parse(output([("a", 3, 0, 1), ("b", 1, 0, 0)]))
        self.assertEqual(g.check(self.picked, counts, FEATS), [])

    def test_red_when_one_target_executes_zero(self):
        counts = g.parse(output([("a", 3, 0, 0), ("b", 0, 0, 2)]))
        problems = g.check(self.picked, counts, FEATS)
        self.assertTrue(any("b: executed 0" in p for p in problems), problems)

    def test_red_when_a_target_has_no_result_line(self):
        counts = g.parse(output([("a", 3, 0, 0)]))
        problems = g.check(self.picked, counts, FEATS)
        self.assertTrue(any("b: no test result" in p for p in problems), problems)

    def test_red_when_nothing_is_selected(self):
        problems = g.check({}, {}, FEATS)
        self.assertTrue(any("no test file" in p for p in problems), problems)

    def test_red_when_the_total_is_zero(self):
        counts = g.parse(output([("a", 0, 0, 1), ("b", 0, 0, 1)]))
        problems = g.check(self.picked, counts, FEATS)
        self.assertTrue(any("0 tests in total" in p for p in problems), problems)

    def test_red_when_a_listed_feature_selects_no_file(self):
        counts = g.parse(output([("a", 1, 0, 0)]))
        problems = g.check({"a": {"replay"}}, counts, FEATS)
        self.assertTrue(
            any("'gpu-solver-bridge' selects no test file" in p for p in problems),
            problems,
        )

    def test_red_when_a_test_fails(self):
        counts = g.parse(output([("a", 2, 1, 0), ("b", 1, 0, 0)]))
        problems = g.check(self.picked, counts, FEATS)
        self.assertTrue(any("a: 1 failed" in p for p in problems), problems)



class RepositoryList(unittest.TestCase):
    def test_the_simd_gated_integration_tests_are_selected(self):
        # the x86_64 + simd module of analytic_math_wiring (add_simd / cross_simd
        # / dot_batch_4) and the simd items of audit_math ran in no lane: the
        # SIMD step of ci.yml runs --lib only
        picked = g.select(g.ROOT / "tests")
        for target in ("analytic_math_wiring", "audit_math"):
            self.assertIn("simd", picked.get(target, set()), (target, picked.get(target)))


if __name__ == "__main__":
    unittest.main()
