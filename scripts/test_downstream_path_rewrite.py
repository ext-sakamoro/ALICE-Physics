#!/usr/bin/env python3
"""oracle for scripts/downstream_path_rewrite.py on throwaway trees: a provider (engine at the
root, facade in crates/alice-physics) next to downstream repositories."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import downstream_path_rewrite as pr  # noqa: E402


def write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class PathRewrite(unittest.TestCase):
    def setUp(self):
        self.top = Path(tempfile.mkdtemp(prefix="path-rewrite-"))
        self.addCleanup(shutil.rmtree, self.top, True)
        self.provider = self.top / "ALICE-Physics"
        write(self.provider / "Cargo.toml", '[package]\nname = "alice-physics-core"\nversion = "4.0.0"\n')
        write(self.provider / "crates/alice-physics/Cargo.toml", '[package]\nname = "alice-physics"\nversion = "4.0.0"\n')

    def tree(self, name: str, manifests: dict[str, str]) -> Path:
        d = self.top / name
        for rel, text in manifests.items():
            write(d / rel, text)
        return d

    def run_it(self, *trees: Path) -> list[str]:
        return [ln for t in trees for ln in pr.rewrite_tree(str(t), str(self.provider), "alice-physics")]

    def test_a_root_path_dependency_moves_to_the_facade(self):
        d = self.tree("ALICE-TRT", {"Cargo.toml": '[package]\nname = "trt"\nversion = "0.1.0"\n\n[dependencies]\n'
                                    'alice-physics = { path = "../ALICE-Physics", version = "4", optional = true }\n'})
        report = self.run_it(d)
        self.assertEqual(sum(1 for r in report if r.startswith("+")), 1)
        self.assertIn('path = "../ALICE-Physics/crates/alice-physics"', (d / "Cargo.toml").read_text())
        self.assertIn('optional = true', (d / "Cargo.toml").read_text())

    def test_a_member_manifest_two_levels_down_and_a_renamed_key(self):
        d = self.tree("ALICE-LOL", {
            "Cargo.toml": '[workspace]\nmembers = ["crates/a"]\n',
            "crates/a/Cargo.toml": '[package]\nname = "a"\nversion = "0.1.0"\n\n[dependencies.phys]\n'
                                   'package = "alice-physics"\npath = "../../../ALICE-Physics"\n'})
        self.assertEqual(sum(1 for r in self.run_it(d) if r.startswith("+")), 1)
        self.assertIn('path = "../../../ALICE-Physics/crates/alice-physics"', (d / "crates/a/Cargo.toml").read_text())

    def test_already_on_the_facade_and_other_packages_are_left_alone(self):
        text = ('[package]\nname = "v"\nversion = "0.1.0"\n\n[dependencies]\n'
                'alice-physics = { path = "../ALICE-Physics/crates/alice-physics" }\n'
                'alice-physics-core = { path = "../ALICE-Physics" }\n'
                'other = { path = "../ALICE-Physics" }\n')
        d = self.tree("ALICE-View", {"Cargo.toml": text})
        self.assertEqual(self.run_it(d), [])
        self.assertEqual((d / "Cargo.toml").read_text(), text)

    def test_a_dependency_on_another_directory_is_left_alone(self):
        write(self.top / "Elsewhere/Cargo.toml", '[package]\nname = "alice-physics"\nversion = "1.0.0"\n')
        text = '[package]\nname = "x"\nversion = "0.1.0"\n\n[dependencies]\nalice-physics = { path = "../Elsewhere" }\n'
        d = self.tree("X", {"Cargo.toml": text})
        self.assertEqual(self.run_it(d), [])

    def test_target_and_dev_tables(self):
        d = self.tree("Y", {"Cargo.toml": '[package]\nname = "y"\nversion = "0.1.0"\n\n'
                            '[dev-dependencies]\nalice-physics = { path = "../ALICE-Physics" }\n\n'
                            '[target.\'cfg(unix)\'.dependencies]\nalice-physics = { path = "../ALICE-Physics" }\n'})
        self.assertEqual(sum(1 for r in self.run_it(d) if r.startswith("+")), 2)

    def test_the_same_path_string_on_another_package_stops(self):
        # the textual edit would also move `other`, which is not the facade's package: the
        # parse check refuses instead of leaving a manifest the downstream would not write
        d = self.tree("Z", {"Cargo.toml": '[package]\nname = "z"\nversion = "0.1.0"\n\n'
                            '[dependencies.alice-physics]\npath = "../ALICE-Physics"\n\n'
                            '[dependencies.other]\npath = "../ALICE-Physics"\n'})
        with self.assertRaisesRegex(ValueError, "could not be rewritten"):
            self.run_it(d)

    def test_the_facade_package_name_is_read_from_its_manifest(self):
        self.assertEqual(pr.facade_package(str(self.provider)), "alice-physics")


if __name__ == "__main__":
    unittest.main()
