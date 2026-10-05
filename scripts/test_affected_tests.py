#!/usr/bin/env python3
"""scripts/affected_tests.py の oracle (git / cargo を呼ばない純関数の検査).

選択器は「選び漏らして green」と「選ばれず 0 本で green」が最も危ない 期待値は fixture の構造
(どの file がどの module を参照しているか) から決まり、選択器を呼んで作らない.

run: python3 scripts/test_affected_tests.py
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("affected_tests", HERE / "affected_tests.py")
at = importlib.util.module_from_spec(spec)
spec.loader.exec_module(at)

TESTS = {
    "audit_pressure": "use alice_physics::pressure::PressureModifier;",
    "audit_solver": "use alice_physics::solver::PhysicsWorld;",
    "analytic_world_api": "use alice_physics::{solver::PhysicsWorld, pressure::PressureConfig};",
    "unrelated": "fn helper() {}  // pressure but no crate import",
    "gated": "use alice_physics::pressure::X;",
    "determinism_golden": "use alice_physics::solver::PhysicsWorld;",
}
REQ = {"gated": {"replay"}}


def sel(changed, features=("std",)):
    return at.select(changed, TESTS, REQ, set(features))


class ModuleOf(unittest.TestCase):
    def test_a_flat_file_and_a_nested_file_map_to_the_same_module(self):
        self.assertEqual(at.module_of("src/pressure.rs"), "pressure")
        self.assertEqual(at.module_of("src/pressure/field.rs"), "pressure")

    def test_lib_rs_cannot_be_narrowed(self):
        self.assertIsNone(at.module_of("src/lib.rs"))

    def test_non_src_files_are_not_modules(self):
        self.assertIsNone(at.module_of("tests/audit_pressure.rs"))
        self.assertIsNone(at.module_of("docs/x.md"))


class Select(unittest.TestCase):
    def test_a_test_naming_the_changed_module_is_selected(self):
        r = sel(["src/pressure.rs"])
        self.assertIn("audit_pressure", r["targets"])
        self.assertIn("analytic_world_api", r["targets"])

    def test_a_test_not_naming_the_module_is_not_selected(self):
        r = sel(["src/pressure.rs"])
        self.assertNotIn("audit_solver", r["targets"])

    def test_a_word_match_without_the_crate_import_is_not_selected(self):
        # `alice_physics` が無い file は module 名を含んでいても対象外
        self.assertNotIn("unrelated", sel(["src/pressure.rs"])["targets"])

    def test_a_changed_test_file_selects_itself(self):
        r = sel(["tests/audit_solver.rs"])
        self.assertIn("audit_solver", r["targets"])
        self.assertFalse(r["src_changed"])

    def test_required_features_outside_the_run_are_skipped_not_selected(self):
        r = sel(["src/pressure.rs"], features=("std",))
        self.assertNotIn("gated", r["targets"])
        self.assertIn("gated", r["skipped"])

    def test_required_features_inside_the_run_are_selected(self):
        r = sel(["src/pressure.rs"], features=("std", "replay"))
        self.assertIn("gated", r["targets"])

    def test_lib_rs_falls_back_to_everything(self):
        r = sel(["src/lib.rs", "src/pressure.rs"])
        self.assertTrue(r["all"])

    def test_a_docs_only_change_selects_no_module_and_is_not_src_changed(self):
        r = sel(["README.md", "docs/oracle-status.md"])
        self.assertEqual(r["modules"], [])
        self.assertFalse(r["src_changed"])
        self.assertEqual(r["targets"], [])

    def test_a_module_nobody_tests_is_src_changed_with_no_targets(self):
        # main() はこれを exit 3 にする (空振りで green にしない)
        r = sel(["src/nobody_tests_this.rs"])
        self.assertTrue(r["src_changed"])
        self.assertEqual(r["targets"], [])

    def test_several_modules_union_their_tests(self):
        r = sel(["src/pressure.rs", "src/solver.rs"])
        self.assertEqual(
            set(r["targets"]),
            {"audit_pressure", "audit_solver", "analytic_world_api", "determinism_golden"},
        )


class LibRs(unittest.TestCase):
    ADD = (
        "diff --git a/src/lib.rs b/src/lib.rs\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1,2 +1,6 @@\n"
        " pub mod solver;\n"
        '+#[cfg(feature = "std")]\n+pub mod vehicle_dynamics;\n+pub mod sensors;\n+pub use sensors::Lidar;\n+// docs\n'
    )

    def test_only_added_module_declarations_return_the_module_names(self):
        self.assertEqual(at.lib_rs_added_modules(self.ADD), ["sensors", "vehicle_dynamics"])

    def test_a_removed_line_cannot_be_narrowed(self):
        self.assertIsNone(at.lib_rs_added_modules(self.ADD + "-pub mod old;\n"))

    def test_a_reexport_list_edit_is_narrowable(self):
        d = self.ADD + "-    SolverBackend,\n+    SolverBackend, WorldSnapshotError,\n"
        self.assertEqual(at.lib_rs_added_modules(d), ["sensors", "vehicle_dynamics"])

    def test_a_removed_module_declaration_is_not_a_reexport_edit(self):
        self.assertIsNone(at.lib_rs_added_modules(self.ADD + "-pub mod old;\n"))

    def test_an_added_expression_cannot_be_narrowed(self):
        self.assertIsNone(at.lib_rs_added_modules(self.ADD + "+pub const X: u32 = 1;\n"))

    def test_a_narrowed_lib_rs_selects_the_added_modules_tests_instead_of_everything(self):
        r = at.select(["src/lib.rs", "src/pressure.rs"], TESTS, REQ, {"std"}, ["pressure"])
        self.assertFalse(r["all"])
        self.assertIn("audit_pressure", r["targets"])

    def test_a_new_module_alone_selects_tests_that_name_it(self):
        r = at.select(["src/lib.rs"], TESTS, REQ, {"std"}, ["pressure"])
        self.assertFalse(r["all"])
        self.assertEqual(r["modules"], ["pressure"])
        self.assertIn("audit_pressure", r["targets"])

    def test_without_the_lib_rs_diff_it_still_falls_back_to_everything(self):
        self.assertTrue(at.select(["src/lib.rs"], TESTS, REQ, {"std"}, None)["all"])


class Counting(unittest.TestCase):
    def test_passed_counts_are_summed_over_binaries(self):
        out = (
            "test result: ok. 12 passed; 0 failed; 1 ignored\n"
            "test result: ok. 30 passed; 0 failed; 0 ignored\n"
        )
        self.assertEqual(at.passed_count(out), 42)

    def test_zero_executed_tests_counts_as_zero(self):
        out = "test result: ok. 0 passed; 0 failed; 5 ignored\n"
        self.assertEqual(at.passed_count(out), 0)

    def test_no_result_line_counts_as_zero(self):
        self.assertEqual(at.passed_count("error: could not compile\n"), 0)


class ChangedFiles(unittest.TestCase):
    """changed_files は git を呼ぶので、使い捨ての repo で見る."""

    def test_an_untracked_new_file_is_reported_as_changed(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = lambda *a: subprocess.run(a, cwd=root, check=True, capture_output=True)
            run("git", "init", "-q", "-b", "main")
            (root / "tests").mkdir()
            (root / "tests" / "old.rs").write_text("// old\n")
            run("git", "add", ".")
            run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
            (root / "tests" / "brand_new.rs").write_text("// new\n")  # git add していない
            saved = at.ROOT
            at.ROOT = root
            try:
                self.assertIn("tests/brand_new.rs", at.changed_files("main"))
            finally:
                at.ROOT = saved


class CargoToml(unittest.TestCase):
    def test_required_features_are_read_per_test_target(self):
        toml = (
            '[package]\nname = "x"\nversion = "0.1.0"\n'
            '[[test]]\nname = "a"\nrequired-features = ["replay", "std"]\n'
            '[[test]]\nname = "b"\n'
        )
        self.assertEqual(at.required_features(toml), {"a": {"replay", "std"}, "b": set()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
