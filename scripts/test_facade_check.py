#!/usr/bin/env python3
"""oracle for scripts/facade_check.py: each finding is produced by the input that should
produce it, and the clean inputs produce none."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import facade_check as fc  # noqa: E402

CORE = {"features": {"default": ["std"], "std": [], "simd": [], "ffi": ["std"]}}


def facade(**over):
    f = {"default": ["std"], "std": ["alice_physics_core/std"], "simd": ["alice_physics_core/simd"],
         "ffi": ["alice_physics_core/ffi"]}
    f.update(over)
    return {"features": {k: v for k, v in f.items() if v is not None}}


class Features(unittest.TestCase):
    def test_forwarding_every_feature_under_its_name_is_clean(self):
        self.assertEqual(fc.feature_problems(CORE, facade()), ([], 4))

    def test_a_core_feature_the_facade_lacks(self):
        p, _ = fc.feature_problems(CORE, facade(ffi=None))
        self.assertEqual(p, ["feature 'ffi' of alice-physics-core is not forwarded by the facade"])

    def test_a_facade_feature_the_core_lacks(self):
        p, n = fc.feature_problems(CORE, facade(extra=["alice_physics_core/extra"]))
        self.assertEqual(p, ["the facade has feature 'extra', which alice-physics-core does not"])
        self.assertEqual(n, 5)

    def test_forwarding_to_another_name(self):
        p, _ = fc.feature_problems(CORE, facade(simd=["alice_physics_core/std"]))
        self.assertEqual(len(p), 1)
        self.assertIn("'simd'", p[0])

    def test_forwarding_more_than_the_feature(self):
        p, _ = fc.feature_problems(CORE, facade(ffi=["alice_physics_core/ffi", "alice_physics_core/std"]))
        self.assertEqual(len(p), 1)

    def test_a_different_default(self):
        p, _ = fc.feature_problems(CORE, facade(default=["std", "simd"]))
        self.assertEqual(len(p), 1)
        self.assertIn("'default'", p[0])

    def test_no_features_compares_nothing(self):
        self.assertEqual(fc.feature_problems({}, {}), ([], 0))


class Dependency(unittest.TestCase):
    CORE = {"package": {"version": "4.0.0"}}

    def test_the_exact_engine_version_is_clean(self):
        f = {"dependencies": {"alice_physics_core": {"package": "alice-physics-core", "version": "=4.0.0"}}}
        self.assertEqual(fc.dependency_problems(self.CORE, f), [])

    def test_a_caret_or_another_version_or_none(self):
        for dep in ({"version": "4.0.0"}, {"version": "=3.1.0"}, {}):
            self.assertEqual(len(fc.dependency_problems(self.CORE, {"dependencies": {"alice_physics_core": dep}})), 1)
        self.assertEqual(len(fc.dependency_problems(self.CORE, {})), 1)


class Lib(unittest.TestCase):
    def test_the_glob_with_docs_and_attributes_is_clean(self):
        self.assertEqual(fc.lib_problems("//! doc; with a semicolon\n#![no_std]\n\npub use alice_physics_core::*;\n"), [])

    def test_a_narrowed_re_export(self):
        self.assertEqual(len(fc.lib_problems("pub use alice_physics_core::{PhysicsWorld};\n")), 1)

    def test_an_item_of_its_own(self):
        self.assertEqual(len(fc.lib_problems("pub use alice_physics_core::*;\npub const X: u8 = 1;\n")), 1)

    def test_an_empty_lib(self):
        self.assertEqual(len(fc.lib_problems("#![no_std]\n")), 1)


PIN = 'const PINNED_ID_HEX: &str = "' + "ab" * 32 + '";\n'


class Pin(unittest.TestCase):
    def test_the_same_pin_is_clean(self):
        self.assertEqual(fc.pin_problems("x\n" + PIN, PIN + "y\n"), ([], 1))

    def test_a_different_pin(self):
        p, n = fc.pin_problems(PIN, PIN.replace("ab", "cd", 1))
        self.assertEqual(n, 1)
        self.assertEqual(len(p), 1)

    def test_a_missing_pin_compares_nothing(self):
        p, n = fc.pin_problems("no pin\n", PIN)
        self.assertEqual(n, 0)
        self.assertEqual(len(p), 1)

    def test_two_pins_compare_nothing(self):
        self.assertEqual(fc.pin_problems(PIN + PIN, PIN)[1], 0)


class Bindings(unittest.TestCase):
    UNITY = '#if IOS\n        private const string DLL = "__Internal";\n#else\n        private const string DLL = "alice_physics";\n#endif\n'
    UNREAL = ['string a = Path.Combine(p, "alice_physics.dll");\nstring b = "libalice_physics.dylib";\nstring c = "libalice_physics.so";\n']

    def test_the_facade_library_everywhere_is_clean(self):
        self.assertEqual(fc.binding_problems("alice_physics", self.UNITY, self.UNREAL), ([], 4))

    def test_a_binding_loading_the_engine_library(self):
        p, n = fc.binding_problems("alice_physics", self.UNITY.replace('"alice_physics"', '"alice_physics_core"'), self.UNREAL)
        self.assertEqual(len(p), 1)
        self.assertIn("alice_physics_core", p[0])
        p, _ = fc.binding_problems("alice_physics", self.UNITY, [self.UNREAL[0].replace("libalice_physics.so", "libalice_physics_core.so")])
        self.assertEqual(len(p), 1)

    def test_no_binding_names_compares_nothing(self):
        self.assertEqual(fc.binding_problems("alice_physics", "", []), ([], 0))


SNAPSHOT = """pub mod alice_physics_core
pub mod alice_physics_core::thermal
pub mod alice_physics_core::thermal::inner
pub struct alice_physics_core::thermal::Grid
pub fn alice_physics_core::thermal::Grid::new(n: usize) -> Self
pub alice_physics_core::thermal::Grid::cells: usize
impl core::clone::Clone for alice_physics_core::thermal::Grid
pub fn alice_physics_core::thermal::step(g: &mut alice_physics_core::thermal::Grid)
pub const alice_physics_core::PHYSICS_SEMANTICS_ID: [u8; 32]
pub unsafe extern "C" fn alice_physics_core::ffi_world_new() -> *mut u8
pub trait alice_physics_core::thermal::inner::Law
pub fn alice_physics_core::thermal::inner::Law::eval(&self) -> f64
pub enum alice_physics_core::Shape
pub use alice_physics_core::det_math::sqrt
pub mod alice_physics_core::det_math
pub alice_physics_core::Shape::Sphere
"""


class Importable(unittest.TestCase):
    def test_modules_and_module_level_items_only(self):
        self.assertEqual(fc.importable(SNAPSHOT), [
            "PHYSICS_SEMANTICS_ID", "Shape", "det_math", "det_math::sqrt", "ffi_world_new", "thermal", "thermal::Grid",
            "thermal::inner", "thermal::inner::Law", "thermal::step"])

    def test_the_crate_root_alone_is_not_a_path(self):
        self.assertEqual(fc.importable("pub mod alice_physics_core\n"), [])

    def test_render_counts_and_asserts_the_identifier(self):
        out = fc.render(["thermal", "thermal::step"])
        self.assertIn("use alice_physics::thermal as _;\nuse alice_physics::thermal::step as _;\n", out)
        self.assertIn("// 2 paths", out)
        self.assertIn("alice_physics::PHYSICS_SEMANTICS_ID, alice_physics_core::PHYSICS_SEMANTICS_ID", out)


if __name__ == "__main__":
    unittest.main()
