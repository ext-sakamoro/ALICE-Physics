#!/usr/bin/env python3
"""Oracles of scripts/public_api_arch_check.py: an architecture-only `pub` item of src/ must be in its
architecture's snapshot, the two snapshots must differ on the x86_64 ones, a missing or empty snapshot
fails, and reading no x86_64-only item fails (compared nothing).

run: python3 scripts/test_public_api_arch_check.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_api_arch_check as c  # noqa: E402

SRC = '''pub struct Fix128;
impl Fix128 {
    /// Adds.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    pub unsafe fn add_simd(self, rhs: Self) -> Self { self }

    #[cfg(all(
        feature = "simd",
        target_arch = "x86_64",
        not(feature = "scalar")
    ))]
    pub fn cross_simd(self) -> Self { self }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn scalar_only(self) -> Self { self }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    pub fn both(self) -> Self { self }

    #[cfg(target_arch = "x86_64")]
    fn private_simd(self) -> Self { self }
}
'''
AARCH64 = "pub struct m::Fix128\npub fn m::Fix128::scalar_only(self) -> Self\npub fn m::Fix128::both(self) -> Self\n"
X86 = ("pub struct m::Fix128\npub unsafe fn m::Fix128::add_simd(self, Self) -> Self\n"
       "pub fn m::Fix128::cross_simd(self) -> Self\npub fn m::Fix128::both(self) -> Self\n")
PATHS = dict((t.split("-")[0], p) for t, p in c.PUBLIC_API_SNAPSHOTS)


def tree(src: str = SRC, aarch64: str | None = AARCH64, x86: str | None = X86) -> Path:
    root = Path(tempfile.mkdtemp())
    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / "src" / "math.rs").write_text(src, encoding="utf-8")
    for arch, text in (("aarch64", aarch64), ("x86_64", x86)):
        if text is not None:
            (root / PATHS[arch]).write_text(text, encoding="utf-8")
    return root


class ArchCheck(unittest.TestCase):
    def test_the_architecture_only_items_are_read_from_the_source(self):
        items = c.arch_items(tree() / "src")
        self.assertEqual([(a, n) for a, _, n, _ in items], [("x86_64", "add_simd"), ("x86_64", "cross_simd")],
                         "not(...), two architectures and private items are not architecture-only pub items")

    def test_matching_snapshots_pass(self):
        errors, counts = c.check(tree())
        self.assertEqual(errors, [])
        self.assertEqual(counts, {"aarch64": 0, "x86_64": 2})

    def test_a_missing_or_empty_snapshot_fails(self):
        for kw in ({"x86": None}, {"x86": " \n"}, {"aarch64": None}):
            errors, _ = c.check(tree(**kw))
            self.assertTrue(any("missing or empty" in e for e in errors), (kw, errors))

    def test_an_item_missing_from_its_snapshot_fails(self):
        errors, _ = c.check(tree(x86=X86.replace("pub fn m::Fix128::cross_simd(self) -> Self\n", "")))
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("cross_simd", errors[0])

    def test_a_longer_name_does_not_stand_for_the_item(self):
        errors, _ = c.check(tree(x86=X86.replace("::cross_simd(self)", "::cross_simd_wide(self)")))
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("cross_simd", errors[0])

    def test_a_snapshot_regenerated_for_the_other_target_fails(self):
        errors, _ = c.check(tree(x86=AARCH64))
        self.assertEqual(len(errors), 2, errors)

    def test_two_snapshots_that_do_not_differ_fail(self):
        errors, _ = c.check(tree(aarch64=X86))
        self.assertTrue(any("do not differ" in e for e in errors), errors)

    def test_no_x86_64_item_in_the_source_compared_nothing(self):
        errors, counts = c.check(tree(src="pub fn plain() {}\n"))
        self.assertEqual(counts["x86_64"], 0)
        self.assertTrue(any("compared nothing" in e for e in errors), errors)

    def test_without_not_removes_balanced_groups_only(self):
        self.assertEqual(c.without_not('all(a, not(any(b, c(d))), e)'), "all(a, , e)")
        self.assertEqual(c.without_not("cannot(x)"), "cannot(x)", "a word ending in `not` is not a not(...)")

    def test_this_repository(self):
        errors, counts = c.check()
        self.assertEqual(errors, [])
        self.assertGreaterEqual(counts["x86_64"], 4)


if __name__ == "__main__":
    unittest.main()
