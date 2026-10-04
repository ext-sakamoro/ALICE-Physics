#!/usr/bin/env python3
"""Tests for scripts/audit_refs.py (hand-built SCIP, no rust-analyzer needed)."""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_refs as ar  # noqa: E402
from test_scip_reach import Doc, build  # noqa: E402

SRC = """pub struct DebugDrawData;
impl DebugDrawData {
    pub fn aabb(&self) {}
}
macro_rules! impl_sketch { ($name:ident, $n:expr) => { impl $name { pub fn insert_hash(&mut self) {} } } }
impl_sketch!(HyperLogLog14, 14);
pub type HyperLogLog = HyperLogLog14;
pub fn arrow() {}"""


def src_doc() -> Doc:
    return (Doc("src/debug_render.rs", SRC)
            .define("debug_render/DebugDrawData#", 0, "DebugDrawData")
            .define("debug_render/impl#[DebugDrawData]aabb().", 2, "aabb")
            .define("debug_render/arrow().", 7, "arrow"))


def test_file(*reasons: str) -> Doc:
    lines = []
    for i, r in enumerate(reasons):
        lines += ["#[test]", f'#[ignore = "known defect: {r}"]', f"fn t{i}() {{}}", ""]
    return Doc("tests/audit_x.rs", "\n".join(lines))


def run(docs, *args) -> tuple[int, str, str]:
    d = build(docs)
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = ar.main(["--root", str(d), *args])
    return code, out.getvalue(), err.getvalue()


class Mentions(unittest.TestCase):
    def test_paths_calls_and_ticked_identifiers(self):
        m = ar.mentions("known defect: AUD-A-S1W1-001: DebugDrawData::aabb and arrow() and `step_with_options`")
        self.assertEqual(m, ["step_with_options", "DebugDrawData::aabb", "arrow"])

    def test_std_items_are_not_crate_mentions(self):
        self.assertEqual(ar.mentions("AUD-A-S1W1-001: uses f32::clamp and i64::try_from and Option::map"), [])

    def test_test_file_reference_is_not_an_item(self):
        r = "AUD-A-S1W1-001: see tests/analytic_x.rs::beyond_range and audit_y.rs::other"
        self.assertEqual(ar.mentions(r), [])

    def test_items_of_an_external_root_cause_are_not_crate_mentions(self):
        r = "AUD-A-S5W1-001: query_energy() respaces; root: external alice-db 0.2.0-beta.3 (Segment::query_range)"
        self.assertEqual(ar.mentions(r), ["query_energy"])

    def test_single_lowercase_word_in_backticks_is_prose(self):
        self.assertEqual(ar.mentions("AUD-A-S1W1-001: returns `true` when `empty`"), [])


class Resolve(unittest.TestCase):
    def test_all_mentions_resolve(self):
        t = test_file("AUD-A-S4W3-008: DebugDrawData::aabb exists; arrow() drops the head")
        code, out, _ = run([src_doc(), t], "--check")
        self.assertEqual(code, 0, out)
        self.assertIn("mentions 2, unresolved 0", out)

    def test_a_name_defined_nowhere_fails_the_check(self):
        t = test_file("AUD-A-S4W3-009: DebugDrawData::wireframe does not exist")
        code, out, err = run([src_doc(), t], "--check")
        self.assertEqual(code, 1)
        self.assertIn("`DebugDrawData::wireframe`, defined nowhere", err)

    def test_method_generated_by_a_macro_resolves_through_the_alias(self):
        # SCIP has no definition inside the macro body: the text fallback finds `fn insert_hash`
        t = test_file("AUD-A-S3W1-010: HyperLogLog::insert_hash wraps")
        code, out, err = run([src_doc(), t], "--check")
        self.assertEqual(code, 0, err)

    def test_lookup_by_id_prints_definition_locations(self):
        t = test_file("AUD-A-S4W3-008: DebugDrawData::aabb exists")
        code, out, _ = run([src_doc(), t], "AUD-A-S4W3-008")
        self.assertEqual(code, 0)
        self.assertIn("DebugDrawData::aabb", out)
        self.assertIn("src/debug_render.rs:3", out)

    def test_lookup_by_test_name_for_a_reason_without_id(self):
        t = test_file("tracked as unwired flags (wireframe exists as DebugDrawData::aabb)")
        code, out, _ = run([src_doc(), t], "t0")
        self.assertEqual(code, 0)
        self.assertIn("src/debug_render.rs:3", out)

    def test_unknown_id_fails(self):
        code, _, err = run([src_doc(), test_file("AUD-A-S1W1-001: arrow() x")], "AUD-Z-S9W9-999")
        self.assertEqual(code, 1)

    def test_compared_nothing_fails(self):
        code, _, err = run([src_doc(), test_file("AUD-A-S1W1-001: no code named here")], "--check")
        self.assertEqual(code, 1)
        self.assertIn("compared nothing", err)

    def test_missing_index_fails(self):
        import tempfile
        d = tempfile.mkdtemp()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(ar.main(["--root", d, "--check"]), 1)


if __name__ == "__main__":
    unittest.main()
