#!/usr/bin/env python3
"""Tests of scripts/lazy_link_check.py without a build: the name section reader on a module made
here, the path match on mangled names, and the verdict (absent from bare, present in full, a
watched name the full program lacks is a failure, nothing compared is reported)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lazy_link_check as llc  # noqa: E402


def uleb(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def module(names: list[str]) -> bytes:
    """a wasm module holding only a name section with these function names"""
    entries = b"".join(uleb(i) + uleb(len(n.encode())) + n.encode() for i, n in enumerate(names))
    sub = bytes([1]) + uleb(len(uleb(len(names)) + entries)) + uleb(len(names)) + entries
    body = uleb(4) + b"name" + sub
    return b"\0asm\x01\0\0\0" + bytes([0]) + uleb(len(body)) + body


CAST = "_RNvMsd_NtCs8NptI9FjOkW_18alice_physics_core17world_shape_queryNtNtB7_6solver12PhysicsWorld11cast_sphere"
STEP = "_RNvMs2_NtCs8NptI9FjOkW_18alice_physics_core5clothNtB5_5Cloth10step_awake"


class Reader(unittest.TestCase):
    def test_reads_the_names_back(self):
        self.assertEqual(llc.function_names(module(["a", CAST, "ünï"])), ["a", CAST, "ünï"])

    def test_refuses_what_is_not_wasm(self):
        with self.assertRaises(ValueError):
            llc.function_names(b"not wasm at all")


class Match(unittest.TestCase):
    def test_matches_the_path_in_order(self):
        self.assertTrue(llc.links([CAST], ["world_shape_query", "cast_sphere"]))
        self.assertTrue(llc.links([STEP], ["cloth", "step_awake"]))

    def test_does_not_match_out_of_order_or_a_longer_identifier(self):
        self.assertFalse(llc.links([CAST], ["cast_sphere", "world_shape_query"]))
        # `11cast_sphere` is not `12cast_spheres`, and `5cloth` is not inside `10step_cloth`
        self.assertFalse(llc.links(["_RNv12cast_spheres"], ["cast_sphere"]))
        self.assertFalse(llc.links(["_RNv10step_cloths"], ["cloth", "step_awake"]))


class Verdict(unittest.TestCase):
    def setUp(self):
        self.saved = llc.WATCHED
        llc.WATCHED = [("cast", ["world_shape_query", "cast_sphere"]), ("step", ["cloth", "step_awake"])]

    def tearDown(self):
        llc.WATCHED = self.saved

    def test_absent_from_bare_and_present_in_full_passes(self):
        self.assertEqual(llc.check({"bare": ["x"], "full": [CAST, STEP]}), (2, []))

    def test_linked_into_bare_fails(self):
        compared, problems = llc.check({"bare": [STEP], "full": [CAST, STEP]})
        self.assertEqual(compared, 2)
        self.assertEqual(len(problems), 1)
        self.assertIn("never asked for it", problems[0])

    def test_a_name_the_full_program_lacks_fails_and_is_not_compared(self):
        compared, problems = llc.check({"bare": [], "full": [CAST]})
        self.assertEqual(compared, 1)
        self.assertEqual(len(problems), 1)
        self.assertIn("compare nothing", problems[0])

    def test_nothing_compared(self):
        self.assertEqual(llc.check({"bare": [], "full": []})[0], 0)


if __name__ == "__main__":
    unittest.main()
