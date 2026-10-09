"""Oracles of scripts/fuzz_targets.py.

run: python3 scripts/test_fuzz_targets.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fuzz_targets", os.path.join(HERE, "fuzz_targets.py"))
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


def tree(names: list[str], missing: tuple[str, ...] = ()) -> str:
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "fuzz", "fuzz_targets"))
    cargo = '[package]\nname = "f"\nversion = "0.0.0"\n\n[dependencies]\nx = "1"\n'
    for n in names:
        cargo += f'\n[[bin]]\nname = "{n}"\npath = "fuzz_targets/{n}.rs"\ntest = false\n'
        if n not in missing:
            open(os.path.join(root, "fuzz", "fuzz_targets", f"{n}.rs"), "w").close()
    open(os.path.join(root, "fuzz", "Cargo.toml"), "w").write(cargo)
    return root


class FuzzTargets(unittest.TestCase):
    def test_every_bin_is_found(self):
        root = tree(ft.CORE + ["fuzz_extra"])
        self.assertEqual([n for n, _ in ft.targets(root)], ft.CORE + ["fuzz_extra"])
        self.assertEqual(ft.check(root), [])

    def test_a_new_bin_is_run_nightly_but_not_on_push(self):
        root = tree(ft.CORE + ["fuzz_tenth"])
        self.assertIn("fuzz_tenth", ft.matrix(root, "schedule"))
        self.assertIn("fuzz_tenth", ft.matrix(root, "workflow_dispatch"))
        self.assertEqual(ft.matrix(root, "push"), ft.CORE)
        self.assertEqual(ft.matrix(root, "pull_request"), ft.CORE)

    def test_no_bin_is_an_error(self):
        self.assertTrue(any("no [[bin]]" in e for e in ft.check(tree([]))))

    def test_a_missing_target_file_is_an_error(self):
        errors = ft.check(tree(ft.CORE + ["fuzz_gone"], missing=("fuzz_gone",)))
        self.assertTrue(any("fuzz_gone" in e and "does not exist" in e for e in errors), errors)

    def test_a_renamed_core_target_is_an_error(self):
        names = [n for n in ft.CORE if n != "fuzz_step"] + ["fuzz_step2"]
        self.assertTrue(any("core fuzz target fuzz_step " in e for e in ft.check(tree(names))))

    def test_this_repository(self):
        root = os.path.dirname(HERE)
        self.assertEqual(ft.check(root), [])
        self.assertEqual(len(ft.matrix(root, "schedule")), len(ft.targets(root)))
        self.assertGreaterEqual(len(ft.targets(root)), 9)
        self.assertEqual(json.loads(json.dumps(ft.matrix(root, "push"))), ft.CORE)


if __name__ == "__main__":
    unittest.main()
