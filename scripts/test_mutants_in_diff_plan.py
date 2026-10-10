#!/usr/bin/env python3
"""Tests for scripts/mutants_in_diff_plan.py (one rule per case)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mutants_in_diff_plan as mp  # noqa: E402

LIB = """\
pub mod math;
#[cfg(feature = "replay")]
pub mod replay;
/// doc between the attribute and the module
#[cfg(feature = "neural")]
/// more doc
pub mod neural;
#[cfg(all(feature = "wasm", not(feature = "std")))]
pub mod wasm_only;
#[cfg(any(feature = "ffi", feature = "python"))]
pub mod bindings_common;
#[cfg(feature = "python")]
pub mod python;
#[cfg(not(feature = "std"))]
pub mod no_std_alloc;
"""
CARGO = """\
[package]
name = "x"

[[test]]
name = "audit_replay"
required-features = ["std", "replay"]

[[test]]
name = "audit_math"
required-features = ["analytics"]

[[test]]
name = "python_replay_binding"
required-features = ["python"]
"""


def diff_of(*paths: str) -> str:
    return "".join(f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-a\n+b\n" for p in paths)


class Plan(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src/lib.rs").write_text(LIB, encoding="utf-8")
        (self.root / "Cargo.toml").write_text(CARGO, encoding="utf-8")
        for t in ["audit_replay", "analytic_replay_in_memory", "audit_replayer_x", "audit_math",
                  "python_replay_binding", "audit_neural_net"]:
            (self.root / f"tests/{t}.rs").write_text("", encoding="utf-8")

    def plan(self, *paths, base=()):
        return mp.plan(diff_of(*paths), list(base), self.root)

    def test_a_gated_module_turns_its_feature_on(self):
        self.assertEqual(self.plan("src/replay.rs")["features"], ["replay"])

    def test_an_ungated_module_adds_no_feature_and_keeps_the_matrix_ones(self):
        p = self.plan("src/math.rs", base=["parallel"])
        # the module itself adds nothing; its test brings its required feature
        self.assertEqual(p["features"], ["analytics", "parallel"])
        self.assertEqual(p["tests"], ["audit_math"])

    def test_doc_comments_between_the_cfg_and_the_mod_line_are_skipped(self):
        self.assertIn("neural", self.plan("src/neural.rs")["features"])

    def test_tests_named_after_the_module_run_but_not_longer_names(self):
        # audit_replayer_x is a different word; the python binding test needs a
        # host-unbuildable feature and is left out
        self.assertEqual(self.plan("src/replay.rs")["tests"], ["analytic_replay_in_memory", "audit_replay"])

    def test_required_features_of_a_selected_test_are_added(self):
        self.assertEqual(self.plan("src/replay/store.rs")["features"], ["replay"])

    def test_a_module_that_needs_not_std_is_skipped_and_removed_from_the_diff(self):
        p = self.plan("src/wasm_only.rs", "src/math.rs")
        self.assertEqual(p["skipped"], ["src/wasm_only.rs"])
        self.assertNotIn("wasm_only", p["diff"])
        self.assertIn("src/math.rs", p["diff"])

    def test_a_not_std_module_is_skipped(self):
        self.assertEqual(self.plan("src/no_std_alloc.rs")["skipped"], ["src/no_std_alloc.rs"])

    def test_a_host_unbuildable_feature_skips_the_module(self):
        self.assertEqual(self.plan("src/python.rs")["skipped"], ["src/python.rs"])

    def test_any_takes_its_first_feature(self):
        self.assertEqual(self.plan("src/bindings_common.rs")["features"], ["ffi"])

    def test_files_outside_src_and_lib_rs_pass_through_unchanged(self):
        p = self.plan("src/lib.rs", "Cargo.toml")
        self.assertEqual((p["features"], p["tests"], p["skipped"]), ([], [], []))
        self.assertIn("Cargo.toml", p["diff"])

    # src/geom.rs is not gated in lib.rs; one function and one impl inside it are
    MATH = """\
pub fn plain() -> u8 {
    1
}

#[cfg(feature = "gpu-solver-bridge")]
/// doc between the attribute and the item
pub fn bridged(x: u8) -> u8 {
    let s = "a { brace in a string";
    x + 1 // } a brace in a comment
}

#[cfg(feature = "simd")]
impl Thing {
    fn inner(&self) -> u8 {
        if true {
            2
        } else {
            3
        }
    }
}
"""

    def changed(self, line: int) -> str:
        return f"diff --git a/src/geom.rs b/src/geom.rs\n--- a/src/geom.rs\n+++ b/src/geom.rs\n@@ -{line} +{line} @@\n-old\n+new\n"

    def plan_math(self, line: int):
        (self.root / "src/geom.rs").write_text(self.MATH, encoding="utf-8")
        return mp.plan(self.changed(line), [], self.root)

    def test_a_changed_line_in_an_item_gated_fn_turns_its_feature_on(self):
        # line 9: `x + 1` inside `bridged`
        self.assertEqual(self.plan_math(9)["features"], ["gpu-solver-bridge"])

    def test_a_changed_line_outside_the_gated_items_adds_no_feature(self):
        self.assertEqual(self.plan_math(2)["features"], [])
        # the blank line after the gated fn closes, before the impl's attribute
        self.assertEqual(self.plan_math(11)["features"], [])

    def test_a_changed_line_deep_in_a_gated_impl_turns_its_feature_on(self):
        # line 18: `3` inside the else of `inner`, two blocks below the impl
        self.assertEqual(self.plan_math(18)["features"], ["simd"])

    def test_changed_lines_are_counted_in_the_new_file(self):
        d = "@@ -1,3 +5,4 @@\n ctx\n-gone\n+new1\n+new2\n ctx\n@@ -20 +30 @@\n-x\n+y\n"
        self.assertEqual(mp.changed_lines(d), [6, 7, 30])

    def test_the_real_lib_rs_gates_replay_behind_its_feature(self):
        gates = mp.module_gates((HERE.parent / "src/lib.rs").read_text(encoding="utf-8"))
        self.assertEqual(mp.features_for(gates["replay"]), (["replay"], True))
        self.assertGreater(len(gates), 5, "the real lib.rs has gated modules")


if __name__ == "__main__":
    unittest.main()
