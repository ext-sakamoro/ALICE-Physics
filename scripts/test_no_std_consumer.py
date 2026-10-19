#!/usr/bin/env python3
"""Tests for no_std_consumer.py. Usage: python3 scripts/test_no_std_consumer.py"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_std_consumer as nc  # noqa: E402

LIB_ID = "path+file:///w/ALICE-Physics/crates/alice-physics#alice-physics@4.0.0"
CORE_ID = "path+file:///w/ALICE-Physics#alice-physics-core@4.0.0"
CONSUMER_ID = "path+file:///w/ALICE-Physics/consumers/no_std#alice-physics-no-std-consumer@0.0.0"


def art(pid: str, kinds: list[str], files: list[str]) -> dict:
    return {"reason": "compiler-artifact", "package_id": pid, "target": {"crate_types": kinds}, "filenames": files}


CORE = art(CORE_ID, ["rlib"], ["/t/libalice_physics_core-1.rlib", "/t/libalice_physics_core-1.rmeta"])
GOOD = [
    art(LIB_ID, ["rlib"], ["/t/libalice_physics-1.rlib", "/t/libalice_physics-1.rmeta"]),
    art(CONSUMER_ID, ["lib"], ["/t/libconsumer.rlib"]),
    CORE,
]


class Judge(unittest.TestCase):
    def test_rlib_only_passes(self):
        self.assertEqual(nc.judge(GOOD, 0), ([], 1))

    def test_a_cdylib_in_the_crate_types_fails(self):
        msgs = [art(LIB_ID, ["rlib", "cdylib"], ["/t/libalice_physics.rlib", "/t/libalice_physics.dylib"]), GOOD[1], CORE]
        errors, n = nc.judge(msgs, 0)
        self.assertEqual(n, 1)
        self.assertTrue(any("cdylib" in e for e in errors), errors)

    def test_a_failed_build_fails(self):
        errors, n = nc.judge([], 101)
        self.assertEqual(n, 0)
        self.assertTrue(any("exit 101" in e for e in errors), errors)

    def test_no_rlib_file_fails(self):
        msgs = [art(LIB_ID, ["rlib"], ["/t/libalice_physics.rmeta"]), GOOD[1], CORE]
        self.assertTrue(nc.judge(msgs, 0)[0])

    def test_an_engine_built_as_a_cdylib_fails(self):
        msgs = [GOOD[0], GOOD[1], art(CORE_ID, ["rlib", "staticlib"], ["/t/libalice_physics_core.rlib", "/t/libalice_physics_core.a"])]
        errors, _ = nc.judge(msgs, 0)
        self.assertTrue(any("alice-physics-core was built as" in e for e in errors), errors)

    def test_an_unread_engine_compares_nothing(self):
        errors, n = nc.judge(GOOD[:2], 0)
        self.assertEqual(n, 0)
        self.assertEqual(nc.exit_code(errors, n, 0), 2)

    def test_consumer_not_built_fails(self):
        self.assertTrue(nc.judge([GOOD[0], CORE], 0)[0])

    def test_a_directory_named_as_the_package_is_read(self):
        # cargo writes `…/crates/alice-physics#4.0.0` when the directory is the package's name
        lib = art("path+file:///w/ALICE-Physics/crates/alice-physics#4.0.0", ["rlib"], ["/t/libalice_physics-1.rlib"])
        self.assertEqual(nc.judge([lib, GOOD[1], CORE], 0), ([], 1))
        self.assertEqual(nc.package_name("path+file:///w/x/alice-physics/#4.0.0"), "alice-physics")
        self.assertEqual(nc.package_name("path+file:///w/ALICE-Physics#alice-physics-core@4.0.0"), "alice-physics-core")

    def test_old_package_id_form_is_read(self):
        old = art("alice-physics 3.0.1 (path+file:///w/ALICE-Physics)", ["rlib"], ["/t/x.rlib"])
        self.assertEqual(len(nc.artifacts([old], "alice-physics")), 1)

    def test_a_successful_build_with_no_artifact_compares_nothing(self):
        # a fresh build that cargo reported as up to date with no alice-physics message:
        # rc 0, nothing compared -> 2, not a pass
        errors, n = nc.judge([], 0)
        self.assertEqual(nc.exit_code(errors, n, 0), 2)

    def test_exit_codes(self):
        self.assertEqual(nc.exit_code([], 1, 0), 0)
        self.assertEqual(nc.exit_code(["x"], 1, 0), 1)
        self.assertEqual(nc.exit_code(["x"], 0, 101), 1)

    def test_other_packages_are_not_counted(self):
        other = art("registry+https://x#alice-physics-ffi@1.0.0", ["rlib"], ["/t/y.rlib"])
        self.assertEqual(nc.artifacts([other], "alice-physics"), [])


if __name__ == "__main__":
    unittest.main()
