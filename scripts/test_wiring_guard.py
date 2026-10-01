#!/usr/bin/env python3
"""scripts/wiring_guard.py の oracle.

検査器は「比較件数 0 でも green になる」型の空振りが最も危ないので、
(1) 配線済みで通る (2) 未配線で落ちる (3) 落ちるべき 5 つの形 (test のみ参照 /
tests/ のみ参照 / doc 言及のみ / use 行のみ / 自身の定義のみ) で落ちる
(4) マーカーと baseline の欠落・古さで落ちる (5) 検査対象 0 件で落ちる
を fixture crate で固定する。期待値は fixture の構造から決まり、検査器を呼んで作らない。

run: python3 scripts/test_wiring_guard.py
"""
from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiring_guard as wg  # noqa: E402


def crate(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="wiring-fixture-"))
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body), encoding="utf-8")
    return root


def unwired(vs):
    return {v.key.split("::")[-1] for v in vs if v.kind == "unwired"}


def kinds(vs):
    return {v.kind for v in vs}


LIB = "pub mod a;\npub mod b;\n"


class UnwiredItems(unittest.TestCase):
    def test_an_item_called_from_another_production_module_is_wired(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn helper() {}\n",
            "src/b.rs": "pub fn entry() { crate::a::helper(); }\n",
        })
        self.assertNotIn("helper", unwired(wg.check(r)))

    def test_an_item_nobody_calls_is_unwired(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn lonely() {}\n", "src/b.rs": "pub fn x() {}\n"})
        self.assertIn("lonely", unwired(wg.check(r)))

    def test_pub_crate_items_are_checked_too(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub(crate) fn lonely_c() {}\n", "src/b.rs": "pub fn x() {}\n"})
        self.assertIn("lonely_c", unwired(wg.check(r)))

    def test_struct_enum_const_trait_are_checked(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub struct Sx;\npub enum Ex { A }\npub const CX: u32 = 1;\npub trait Tx {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertTrue({"Sx", "Ex", "CX", "Tx"} <= unwired(wg.check(r)))

    def test_referenced_only_inside_cfg_test_does_not_count(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn only_tested() {}\n",
            "src/b.rs": """
                pub fn x() {}
                #[cfg(test)]
                mod tests {
                    #[test]
                    fn t() { crate::a::only_tested(); }
                }
            """,
        })
        self.assertIn("only_tested", unwired(wg.check(r)))

    def test_referenced_only_from_the_tests_directory_does_not_count(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn only_integration() {}\n",
            "src/b.rs": "pub fn x() {}\n",
            "tests/t.rs": "fn t() { mycrate::a::only_integration(); }\n",
        })
        self.assertIn("only_integration", unwired(wg.check(r)))

    def test_a_mention_in_a_doc_comment_or_string_does_not_count(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn ghosted() {}\n",
            "src/b.rs": '/// see [`ghosted`] for details\npub fn x() { let _ = "ghosted"; }\n',
        })
        self.assertIn("ghosted", unwired(wg.check(r)))

    def test_a_use_line_alone_does_not_count(self):
        r = crate({
            "src/lib.rs": LIB + "pub use a::reexported;\n",
            "src/a.rs": "pub fn reexported() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertIn("reexported", unwired(wg.check(r)))

    def test_a_call_from_examples_or_benches_counts(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "pub fn demo_entry() {}\n",
            "src/b.rs": "pub fn x() {}\n",
            "examples/e.rs": "fn main() { mycrate::a::demo_entry(); }\n",
        })
        self.assertNotIn("demo_entry", unwired(wg.check(r)))

    def test_private_items_are_not_checked(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "fn private_lonely() {}\n", "src/b.rs": "pub fn x() {}\n"})
        self.assertNotIn("private_lonely", unwired(wg.check(r)))


class Markers(unittest.TestCase):
    def test_allow_unwired_with_a_reason_silences_the_item(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-UNWIRED: public entry point reached from downstream crates\npub fn api() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertNotIn("api", unwired(wg.check(r)))

    def test_allow_unwired_with_a_short_reason_is_rejected(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-UNWIRED: later\npub fn api() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertIn("bad_marker", kinds(wg.check(r)))

    def test_a_stale_allow_unwired_marker_is_rejected(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-UNWIRED: public entry point reached from downstream crates\npub fn api() {}\n",
            "src/b.rs": "pub fn x() { crate::a::api(); }\n",
        })
        self.assertIn("stale_marker", kinds(wg.check(r)))

    def test_allow_dead_code_without_a_marker_is_rejected(self):
        r = crate({"src/lib.rs": LIB + "#![allow(dead_code)]\n", "src/a.rs": "", "src/b.rs": "pub fn x() {}\n"})
        self.assertIn("dead_code", kinds(wg.check(r)))

    def test_allow_dead_code_with_a_marker_passes(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-DEAD: reserved variants awaiting solver integration\n#![allow(dead_code)]\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertNotIn("dead_code", kinds(wg.check(r)))

    def test_allow_dead_code_marker_with_a_short_reason_is_rejected(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-DEAD: todo\n#[allow(dead_code)]\nfn f() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertIn("bad_marker", kinds(wg.check(r)))

    def test_a_stale_allow_dead_marker_is_rejected(self):
        r = crate({
            "src/lib.rs": LIB,
            "src/a.rs": "// ALLOW-DEAD: reserved variants awaiting solver integration\nfn f() {}\n",
            "src/b.rs": "pub fn x() {}\n",
        })
        self.assertIn("stale_marker", kinds(wg.check(r)))


class Baseline(unittest.TestCase):
    FILES = {"src/lib.rs": LIB, "src/a.rs": "pub fn legacy() {}\n", "src/b.rs": "pub fn x() {}\n"}

    def test_an_item_in_the_baseline_is_tolerated(self):
        r = crate(self.FILES)
        self.assertNotIn("legacy", unwired(wg.check(r, "unwired src/a.rs::legacy\n")))

    def test_an_item_not_in_the_baseline_fails(self):
        r = crate(self.FILES)
        self.assertIn("legacy", unwired(wg.check(r, "")))

    def test_a_stale_baseline_entry_fails_so_the_ratchet_tightens(self):
        r = crate({"src/lib.rs": LIB, "src/a.rs": "pub fn legacy() {}\n", "src/b.rs": "pub fn x() { crate::a::legacy(); }\n"})
        self.assertIn("stale_baseline", kinds(wg.check(r, "unwired src/a.rs::legacy\n")))

    def test_dead_code_count_may_not_grow_beyond_the_baseline(self):
        body = "#![allow(dead_code)]\n#[allow(dead_code)]\nfn f() {}\n"
        r = crate({"src/lib.rs": LIB, "src/a.rs": body, "src/b.rs": "pub fn x() {}\n"})
        self.assertIn("dead_code", kinds(wg.check(r, "dead_code src/a.rs 1\n")))
        self.assertNotIn("dead_code", kinds(wg.check(r, "dead_code src/a.rs 2\n")))


class EmptyScan(unittest.TestCase):
    def test_scanning_zero_items_is_a_failure_not_a_green(self):
        r = crate({"src/lib.rs": "fn private_only() {}\n"})
        self.assertIn("empty_scan", kinds(wg.check(r)))

    def test_a_missing_src_directory_is_a_failure(self):
        r = crate({"README.md": "x\n"})
        self.assertIn("empty_scan", kinds(wg.check(r)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
