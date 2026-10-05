#!/usr/bin/env python3
"""Tests for scripts/integration_levels.py.

A synthetic crate (sources plus a hand-built SCIP index from test_scip_reach's
encoder) with one module per level, C ABI consumers written as plain files, and
the README / MODULES.md / ledger documents. One rule per case.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import integration_levels as il  # noqa: E402
from test_scip_reach import Doc, build  # noqa: E402

LIB = "pub mod solver;\npub mod alpha;\npub mod beta;\npub mod gamma;\npub mod delta;\npub mod eps;\npub mod mixed;\npub mod ffi;"
SOLVER = """pub struct PhysicsWorld;
impl PhysicsWorld {
    pub fn step(&self) { crate::alpha::a_step(); crate::mixed::m1(); }
    pub fn query(&self) { crate::beta::b_world(); }
}"""
FFI = 'pub extern "C" fn alice_physics_api() { crate::gamma::g_bind(); }\npub extern "C" fn alice_physics_other() {}'
EXAMPLE = "fn main() { delta::d_example(); mixed::m2(); mixed::m3(); }"


def crate_docs() -> list[Doc]:
    return [
        Doc("src/lib.rs", LIB),
        (Doc("src/solver.rs", SOLVER)
         .define("solver/PhysicsWorld#", 0, "PhysicsWorld")
         .define("solver/impl#[PhysicsWorld]step().", 2, "step", end_line=2)
         .ref("alpha/a_step().", 2, "a_step")
         .ref("mixed/m1().", 2, "m1")
         .define("solver/impl#[PhysicsWorld]query().", 3, "query", end_line=3)
         .ref("beta/b_world().", 3, "b_world")),
        Doc("src/alpha.rs", "pub fn a_step() {}").define("alpha/a_step().", 0, "a_step"),
        Doc("src/beta.rs", "pub fn b_world() {}").define("beta/b_world().", 0, "b_world"),
        Doc("src/gamma.rs", "pub fn g_bind() {}").define("gamma/g_bind().", 0, "g_bind"),
        Doc("src/delta.rs", "pub fn d_example() {}").define("delta/d_example().", 0, "d_example"),
        Doc("src/eps.rs", "pub fn e_unused() {}").define("eps/e_unused().", 0, "e_unused"),
        (Doc("src/mixed.rs", "pub fn m1() {}\npub fn m2() {}\npub fn m3() {}")
         .define("mixed/m1().", 0, "m1").define("mixed/m2().", 1, "m2").define("mixed/m3().", 2, "m3")),
        (Doc("src/ffi.rs", FFI)
         .define("ffi/alice_physics_api().", 0, "alice_physics_api", end_line=0)
         .ref("gamma/g_bind().", 0, "g_bind")
         .define("ffi/alice_physics_other().", 1, "alice_physics_other", end_line=1)),
        (Doc("examples/demo.rs", EXAMPLE)
         .ref("delta/d_example().", 0, "d_example")
         .ref("mixed/m2().", 0, "m2").ref("mixed/m3().", 0, "m3")),
    ]


CONSUMERS = {
    "include/alice_physics.h": "void alice_physics_api(void);\nvoid alice_physics_other(void);\n",
    "bindings/AlicePhysics.h": "void alice_physics_api(void);\nvoid alice_physics_other(void);\n",
    "bindings/AlicePhysics.cs": "extern void alice_physics_api();\nextern void alice_physics_other();\n",
    "unreal-plugin/Source/P/Private/C.cpp": "void F() { alice_physics_api(); }\n",
}
GAPS = "unreal alice_physics_other not needed by the component in this fixture\n"


def make_root(gaps: str | None = GAPS) -> Path:
    root = build(crate_docs())
    for rel, text in CONSUMERS.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    if gaps is not None:
        (root / "scripts").mkdir(exist_ok=True)
        (root / il.GAPS_FILE).write_text(gaps, encoding="utf-8")
    return root


def scip(root: Path) -> list[Path]:
    s = root / "target" / "scip"
    return [s / "native.scip", s / "wasm.scip", s / "fuzz.scip"]


def summary(counts: dict[str, int]) -> str:
    rows = "\n".join(f"| {k}: x | {v} |" for k, v in counts.items())
    return f"# R\n\n{il.SUMMARY_MARK}\n| How | Modules |\n|---|---:|\n{rows}\n\nafter\n"


def modules_doc(levels: "il.Levels", override: dict[str, str] | None = None) -> str:
    rows = []
    for m in sorted(levels.module_level):
        label = (override or {}).get(m, levels.label(m))
        rows.append(f"| `{m}` | world ray queries | | | {label} |")
    return "| Module | Summary | Feature | Example | Integration |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n"


def write_docs(root: Path, levels: "il.Levels") -> None:
    (root / "README.md").write_text(summary(levels.counts()), encoding="utf-8")
    (root / "README_JP.md").write_text(summary(levels.counts()), encoding="utf-8")
    (root / "docs").mkdir(exist_ok=True)
    (root / il.MODULES_DOC).write_text(modules_doc(levels), encoding="utf-8")


def run_main(root: Path, *args: str) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = il.main(["--root", str(root), *args])
    return code, out.getvalue() + err.getvalue()


class ModuleLevels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = make_root()
        cls.lv = il.Levels(cls.root, scip(cls.root))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_each_module_gets_the_level_its_callers_give(self):
        self.assertEqual(self.lv.module_level["alpha"], "step")
        self.assertEqual(self.lv.module_level["beta"], "world API")
        self.assertEqual(self.lv.module_level["gamma"], "binding")
        self.assertEqual(self.lv.module_level["eps"], "unused")
        self.assertEqual(self.lv.module_level["ffi"], "binding")

    def test_an_example_caller_is_not_integration(self):
        self.assertEqual(self.lv.module_level["delta"], "standalone")

    def test_the_majority_labels_and_higher_items_are_noted(self):
        # m1 runs in step, m2 / m3 only from the example
        self.assertEqual(self.lv.label("mixed"), "standalone (step 1 of 3 items)")

    def test_step_entries_are_the_world_methods_named_step(self):
        self.assertEqual(self.lv.step_entries, ["step"])

    def test_counts_cover_the_public_modules(self):
        c = self.lv.counts()
        self.assertEqual(sum(c.values()), len(self.lv.module_level))
        self.assertEqual(c["unused"], 1)


class CAbiCoverage(unittest.TestCase):
    def setUp(self):
        self.root = make_root()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def problems(self) -> list[str]:
        return il.CAbi(self.root).problems()

    def test_full_coverage_with_a_listed_gap_has_no_problem(self):
        self.assertEqual(self.problems(), [])

    def test_a_consumer_missing_a_function_fails(self):
        (self.root / "bindings/AlicePhysics.cs").write_text("extern void alice_physics_api();\n", encoding="utf-8")
        self.assertTrue(any("Unity C# does not declare or call `alice_physics_other`" in p for p in self.problems()))

    def test_a_commented_out_declaration_does_not_count(self):
        (self.root / "include/alice_physics.h").write_text(
            "void alice_physics_api(void);\n/* void alice_physics_other(void); */\n", encoding="utf-8")
        self.assertTrue(any("C header does not declare" in p for p in self.problems()))

    def test_an_unlisted_gap_fails(self):
        (self.root / il.GAPS_FILE).write_text("", encoding="utf-8")
        self.assertTrue(any("Unreal Engine plugin does not declare or call `alice_physics_other`" in p
                            for p in self.problems()))

    def test_a_closed_gap_that_is_still_listed_fails(self):
        p = self.root / "unreal-plugin/Source/P/Private/C.cpp"
        p.write_text("void F() { alice_physics_api(); alice_physics_other(); }\n", encoding="utf-8")
        self.assertTrue(any("now uses `alice_physics_other`" in x for x in self.problems()))

    def test_a_gap_for_a_function_that_does_not_exist_fails(self):
        (self.root / il.GAPS_FILE).write_text(GAPS + "unity alice_physics_gone listed for a removed function\n",
                                             encoding="utf-8")
        self.assertTrue(any("`unity alice_physics_gone` names a function" in p for p in self.problems()))

    def test_a_gap_without_a_reason_or_with_an_unknown_consumer_fails(self):
        (self.root / il.GAPS_FILE).write_text(GAPS + "unreal alice_physics_api short\nsteam alice_physics_api for a consumer that is not checked\n",
                                             encoding="utf-8")
        ps = self.problems()
        self.assertTrue(any("expected `<consumer> <function> <reason" in p for p in ps))
        self.assertTrue(any("unknown consumer `steam`" in p for p in ps))

    def test_a_declaration_of_a_function_that_is_not_exported_fails(self):
        (self.root / "bindings/AlicePhysics.h").write_text(
            CONSUMERS["bindings/AlicePhysics.h"] + "void alice_physics_removed(void);\n", encoding="utf-8")
        self.assertTrue(any("`alice_physics_removed`, which src/ffi.rs does not export" in p for p in self.problems()))

    def test_a_consumer_with_no_files_compares_nothing_and_fails(self):
        (self.root / "bindings/AlicePhysics.cs").unlink()
        self.assertTrue(any("Unity C#: no files found" in p for p in self.problems()))

    def test_no_exported_function_compares_nothing_and_fails(self):
        (self.root / "src/ffi.rs").write_text("// nothing exported\n", encoding="utf-8")
        self.assertTrue(any("no extern \"C\" functions" in p for p in self.problems()))


class Documents(unittest.TestCase):
    def setUp(self):
        self.root = make_root()
        self.lv = il.Levels(self.root, scip(self.root))
        write_docs(self.root, self.lv)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_matching_documents_have_no_problem(self):
        self.assertEqual(il.doc_problems(self.root, self.lv), [])

    def test_a_wrong_summary_count_fails(self):
        c = dict(self.lv.counts())
        c["standalone"] += 1
        (self.root / "README_JP.md").write_text(summary(c), encoding="utf-8")
        self.assertTrue(any("README_JP.md: summary table" in p for p in il.doc_problems(self.root, self.lv)))

    def test_a_missing_summary_marker_fails(self):
        (self.root / "README.md").write_text("# R\n", encoding="utf-8")
        self.assertTrue(any("README.md: no" in p for p in il.doc_problems(self.root, self.lv)))

    def test_a_wrong_module_label_fails(self):
        (self.root / il.MODULES_DOC).write_text(modules_doc(self.lv, {"delta": "binding"}), encoding="utf-8")
        self.assertTrue(any("`delta` is listed as `binding`, measured `standalone`" in p
                            for p in il.doc_problems(self.root, self.lv)))

    def test_the_label_is_read_from_the_integration_column_only(self):
        # every row's Summary cell starts with "world"; it must not be taken for the label
        self.assertEqual(il.doc_problems(self.root, self.lv), [])

    def test_no_integration_column_compares_nothing_and_fails(self):
        (self.root / il.MODULES_DOC).write_text("| Module | Summary |\n|---|---|\n| `delta` | x |\n", encoding="utf-8")
        self.assertTrue(any("no Integration column" in p for p in il.doc_problems(self.root, self.lv)))


class Main(unittest.TestCase):
    def setUp(self):
        self.root = make_root()
        lv = il.Levels(self.root, scip(self.root))
        write_docs(self.root, lv)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_write_then_check_passes(self):
        code, out = run_main(self.root, "--write")
        self.assertEqual(code, 0, out)
        code, out = run_main(self.root, "--check")
        self.assertEqual(code, 0, out)

    def test_a_stale_ledger_fails_the_check(self):
        run_main(self.root, "--write")
        (self.root / il.LEDGER).write_text("old\n", encoding="utf-8")
        code, out = run_main(self.root, "--check")
        self.assertEqual(code, 1)
        self.assertIn("is stale", out)

    def test_the_ledger_lists_the_gap_reason(self):
        run_main(self.root, "--write")
        text = (self.root / il.LEDGER).read_text(encoding="utf-8")
        self.assertIn("not needed by the component in this fixture", text)

    def test_a_missing_index_fails(self):
        shutil.rmtree(self.root / "target" / "scip")
        code, out = run_main(self.root, "--check")
        self.assertEqual(code, 1)
        self.assertIn("SCIP index missing", out)

    def test_no_index_mode_checks_the_c_abi_only(self):
        shutil.rmtree(self.root / "target" / "scip")
        self.assertEqual(run_main(self.root, "--no-index", "--check")[0], 0)
        (self.root / il.GAPS_FILE).write_text("", encoding="utf-8")
        self.assertEqual(run_main(self.root, "--no-index", "--check")[0], 1)


if __name__ == "__main__":
    unittest.main()
