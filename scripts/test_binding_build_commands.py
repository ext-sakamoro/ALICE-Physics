#!/usr/bin/env python3
"""Tests for binding_build_commands.py. Usage: python3 scripts/test_binding_build_commands.py"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import binding_build_commands as bbc  # noqa: E402

BLOCK = """<!-- binding-build-commands: x -->
```sh
# C
cargo rustc -p alice-physics --lib --release --features ffi --crate-type cdylib
cargo rustc -p alice-physics --lib --release --features ffi --crate-type staticlib
bash scripts/build_wasm_pkg.sh web web/pkg
maturin build --release -m crates/alice-physics/Cargo.toml --features python
```
<!-- /binding-build-commands -->
"""
WORKFLOW = "      - run: maturin build -m crates/alice-physics/Cargo.toml --features python --out dist/\n"
RELEASE = ("      - run: |\n          cargo rustc -p alice-physics --lib --release --features \"ffi\" --target t \\\n"
           "            --crate-type cdylib\n      - run: maturin build --release -m crates/alice-physics/Cargo.toml --features python\n")
WASM_SH = "cargo rustc -p alice-physics --lib --target wasm32-unknown-unknown --features wasm --crate-type cdylib\n"


def tree(d: Path, en: str = BLOCK, ja: str | None = None, build_wasm: str = "cd .. && bash scripts/build_wasm_pkg.sh web web/pkg",
         workflow: str = WORKFLOW, release: str = RELEASE, wasm_sh: str = WASM_SH) -> Path:
    (d / "web").mkdir()
    (d / ".github" / "workflows").mkdir(parents=True)
    (d / "README.md").write_text("# x\n" + en, encoding="utf-8")
    (d / "README_JP.md").write_text("# x\n" + (en.replace("# C", "# C の説明") if ja is None else ja), encoding="utf-8")
    (d / "web" / "package.json").write_text(json.dumps({"scripts": {"build:wasm": build_wasm}}), encoding="utf-8")
    (d / ".github" / "workflows" / "python-binding.yml").write_text(workflow, encoding="utf-8")
    (d / ".github" / "workflows" / "release.yml").write_text(release, encoding="utf-8")
    (d / "scripts").mkdir()
    (d / "scripts" / "build_wasm_pkg.sh").write_text(wasm_sh, encoding="utf-8")
    return d


class Check(unittest.TestCase):
    def run_check(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            return bbc.check(tree(Path(d), **kw))

    def test_consistent_tree_passes(self):
        self.assertEqual(self.run_check(), ([], 4))

    def test_readmes_with_different_commands_fail(self):
        errors, _ = self.run_check(ja=BLOCK.replace("maturin build --release -m", "maturin build -m"))
        self.assertTrue(any("different" in e for e in errors), errors)

    def test_missing_staticlib_fails(self):
        en = BLOCK.replace("cargo rustc -p alice-physics --lib --release --features ffi --crate-type staticlib\n", "")
        errors, _ = self.run_check(en=en, ja=en)
        self.assertTrue(any("staticlib" in e for e in errors), errors)

    def test_package_json_drift_fails(self):
        errors, _ = self.run_check(build_wasm="cd .. && wasm-pack build --target web --out-dir web/pkg -- --features wasm")
        self.assertTrue(any("build:wasm" in e for e in errors), errors)

    def test_workflow_without_the_python_feature_fails(self):
        errors, _ = self.run_check(workflow="      - run: maturin build --out dist/\n")
        self.assertTrue(any("python-binding.yml" in e for e in errors), errors)

    def test_a_c_command_of_the_engine_crate_fails(self):
        en = BLOCK.replace("cargo rustc -p alice-physics --lib --release --features ffi --crate-type cdylib",
                           "cargo rustc --lib --release --features ffi --crate-type cdylib")
        errors, _ = self.run_check(en=en, ja=en)
        self.assertTrue(any("C ABI command does not build the facade" in e for e in errors), errors)

    def test_another_package_named_alike_does_not_count(self):
        en = BLOCK.replace("-p alice-physics --lib --release --features ffi --crate-type cdylib",
                           "-p alice-physics-core --lib --release --features ffi --crate-type cdylib")
        errors, _ = self.run_check(en=en, ja=en)
        self.assertTrue(any("C ABI command does not build the facade" in e for e in errors), errors)

    def test_a_python_command_of_the_engine_crate_fails(self):
        en = BLOCK.replace(" -m crates/alice-physics/Cargo.toml", "")
        errors, _ = self.run_check(en=en, ja=en)
        self.assertTrue(any("Python command does not build the facade" in e for e in errors), errors)
        errors, _ = self.run_check(workflow="      - run: maturin build --features python --out dist/\n")
        self.assertTrue(any("python-binding.yml: `maturin build" in e for e in errors), errors)

    def test_a_release_build_of_the_engine_crate_fails(self):
        errors, _ = self.run_check(release=RELEASE.replace("-p alice-physics ", ""))
        self.assertTrue(any("release.yml: `cargo rustc" in e for e in errors), errors)
        errors, _ = self.run_check(release=RELEASE.replace(" -m crates/alice-physics/Cargo.toml", ""))
        self.assertTrue(any("release.yml: `maturin build" in e for e in errors), errors)

    def test_a_release_workflow_without_cargo_rustc_fails(self):
        errors, _ = self.run_check(release="      - run: maturin build --release -m crates/alice-physics/Cargo.toml --features python\n")
        self.assertTrue(any("compared nothing" in e for e in errors), errors)

    def test_a_wasm_script_of_the_engine_crate_fails(self):
        errors, _ = self.run_check(wasm_sh=WASM_SH.replace("-p alice-physics ", ""))
        self.assertTrue(any("build_wasm_pkg.sh" in e for e in errors), errors)

    def test_no_block_reads_zero(self):
        self.assertEqual(self.run_check(en="no block\n", ja="no block\n")[1], 0)

    def test_the_real_tree_passes(self):
        errors, n = bbc.check(bbc.ROOT)
        self.assertEqual(errors, [])
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
