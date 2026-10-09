"""Oracles of scripts/preflight_ci_parity.py.

run: python3 scripts/test_preflight_ci_parity.py
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("parity", os.path.join(HERE, "preflight_ci_parity.py"))
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def tree(ci: str, pf: str) -> str:
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, ".github", "workflows"))
    os.makedirs(os.path.join(root, "scripts"))
    open(os.path.join(root, ".github", "workflows", "ci.yml"), "w").write(ci)
    open(os.path.join(root, "scripts", "preflight.sh"), "w").write(pf)
    return root


CI = """jobs:
  t:
    steps:
      - name: a
        run: cargo test
      - name: b
        run: cargo test --features "parallel"
      - name: c
        run: 'cargo test --lib --features "ffi" "ffi::"'
      - name: d
        run: cargo test --lib --features "std,simd"
"""
PF = """NATIVE='std,simd'
cargo test --no-fail-fast
cargo test --no-fail-fast --features "parallel"
cargo test --lib --features "ffi" "ffi::"
cargo test --lib --features "$NATIVE"
"""


class Parity(unittest.TestCase):
    def test_matching_commands_pass_after_normalising(self):
        errors, n = parity.check(tree(CI, PF))
        self.assertEqual((errors, n), ([], 4))

    def test_a_ci_command_missing_from_preflight_fails(self):
        pf = PF.replace('cargo test --no-fail-fast --features "parallel"\n', "")
        errors, _ = parity.check(tree(CI, pf))
        self.assertEqual(len(errors), 1)
        self.assertIn('--features "parallel"', errors[0])

    def test_a_different_feature_set_is_not_a_match(self):
        pf = PF.replace("NATIVE='std,simd'", "NATIVE='std,simd,parallel'")
        errors, _ = parity.check(tree(CI, pf))
        self.assertTrue(any("std,simd" in e for e in errors), errors)

    def test_comments_are_not_commands(self):
        ci = CI + "      # run: cargo test --features nothing\n"
        self.assertEqual(parity.check(tree(ci, PF))[0], [])

    def test_no_cargo_test_in_ci_fails(self):
        errors, n = parity.check(tree("jobs: {}\n", PF))
        self.assertEqual(n, 0)
        self.assertTrue(any("compared nothing" in e for e in errors), errors)

    def test_this_repository(self):
        errors, n = parity.check()
        self.assertEqual(errors, [])
        self.assertGreater(n, 5)


if __name__ == "__main__":
    unittest.main()
