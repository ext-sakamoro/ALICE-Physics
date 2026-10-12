"""Oracles of scripts/package_contents_check.py.

run: python3 scripts/test_package_contents_check.py
"""
from __future__ import annotations

import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "package_contents_check", os.path.join(HERE, "package_contents_check.py")
)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

GOOD = [
    "Cargo.toml",
    "Cargo.toml.orig",
    "Cargo.lock",
    "src/lib.rs",
    "src/bvh.rs",
    "tests/lib/bvh_oracles.rs",
    "tests/snapshot.rs",
    "tests/fixtures/world_snapshot_v1_stacked.bin",
    "tests/fixtures/world_snapshot_v2_stacked.bin",
    "README.md",
    "LICENSE-AGPL",
    "LICENSE-COMMERCIAL.md",
]
TARGETS = ["src/lib.rs", "tests/snapshot.rs"]
SOURCES = {
    "src/lib.rs": "pub mod bvh;\n",
    "src/bvh.rs": '#[cfg(test)]\n#[path = "../tests/lib/bvh_oracles.rs"]\nmod oracles;\n',
    "tests/lib/bvh_oracles.rs": "",
    "tests/snapshot.rs": 'const V1: &[u8] = include_bytes!("fixtures/world_snapshot_v1_stacked.bin");\n',
}
TRACKED = [p for p in GOOD if not p.startswith(("Cargo.toml.orig", "Cargo.lock"))]

# written out, not read from the module: deleting an entry from the constant
# must make the test fail
EXPECTED_FORBIDDEN_DIRS = [
    ".github",
    ".cargo",
    "scripts",
    "docs",
    "fuzz",
    "web",
    "unreal-plugin",
    "python",
    "bindings",
    "target",
]
EXPECTED_FORBIDDEN_FILES = ["rust-toolchain.toml", "rust-toolchain"]
EXPECTED_REQUIRED = ["Cargo.toml", "src/lib.rs", "README.md", "LICENSE-AGPL", "LICENSE-COMMERCIAL.md"]


def run(paths=GOOD, targets=TARGETS, sources=SOURCES, tracked=TRACKED):
    return pc.check(list(paths), list(targets), dict(sources), list(tracked))


class PackageContents(unittest.TestCase):
    def test_good_list_passes_and_counts(self) -> None:
        errs, counts = run()
        self.assertEqual(errs, [])
        self.assertEqual(
            counts, {"files": len(GOOD), "targets": 2, "sources": 4, "embeds": 2, "fixtures": 0}
        )

    def test_empty_list_fails(self) -> None:
        self.assertEqual(len(run(paths=[])[0]), 1)
        self.assertEqual(len(run(paths=["", "  "])[0]), 1)

    def test_no_target_compared_fails(self) -> None:
        self.assertEqual(run(targets=[])[0], ["no Cargo target was compared"])

    def test_no_source_scanned_fails(self) -> None:
        self.assertEqual(run(sources={})[0], ["no packaged Rust source was scanned"])

    def test_each_forbidden_dir_fails(self) -> None:
        for d in EXPECTED_FORBIDDEN_DIRS:
            errs, _ = run(paths=GOOD + [f"{d}/x.txt"])
            self.assertEqual(errs, [f"shipped but not allowed: {d}/x.txt"], d)

    def test_toolchain_pin_fails(self) -> None:
        for f in EXPECTED_FORBIDDEN_FILES:
            self.assertEqual(run(paths=GOOD + [f])[0], [f"shipped but not allowed: {f}"])

    def test_top_level_file_named_like_dir_passes(self) -> None:
        # a top-level file called "docs" is not the docs/ directory
        self.assertEqual(run(paths=GOOD + ["docs"])[0], [])

    def test_missing_required_fails(self) -> None:
        for req in EXPECTED_REQUIRED:
            errs, _ = run(paths=[p for p in GOOD if p != req], targets=[t for t in TARGETS if t != req])
            self.assertEqual(errs, [f"required file missing from the package: {req}"], req)

    def test_missing_target_fails(self) -> None:
        errs, _ = run(
            paths=[p for p in GOOD if p != "tests/snapshot.rs"],
            sources={k: v for k, v in SOURCES.items() if k != "tests/snapshot.rs"},
        )
        self.assertEqual(errs, ["Cargo target missing from the package: tests/snapshot.rs"])

    def test_path_attribute_target_missing_fails(self) -> None:
        # `/tests/**/*.rs` narrowed to `/tests/*.rs` drops tests/lib/bvh_oracles.rs
        errs, _ = run(
            paths=[p for p in GOOD if p != "tests/lib/bvh_oracles.rs"],
            sources={k: v for k, v in SOURCES.items() if k != "tests/lib/bvh_oracles.rs"},
        )
        self.assertEqual(errs, ["src/bvh.rs embeds tests/lib/bvh_oracles.rs, which is not in the package"])

    def test_included_bytes_missing_fails(self) -> None:
        # `/tests/fixtures/*.bin` removed from the include list
        errs, _ = run(paths=[p for p in GOOD if not p.endswith(".bin")])
        self.assertEqual(
            errs,
            ["tests/snapshot.rs embeds tests/fixtures/world_snapshot_v1_stacked.bin, which is not in the package"],
        )

    def test_each_embed_form_is_resolved(self) -> None:
        for src in (
            'include!("gen.rs");',
            'include_str!( "gen.rs" );',
            'include_bytes!("./gen.rs");',
            '#[path = "gen.rs"]\nmod gen;',
        ):
            errs, counts = run(sources={**SOURCES, "src/lib.rs": src})
            self.assertEqual(errs, ["src/lib.rs embeds src/gen.rs, which is not in the package"], src)
            self.assertEqual(counts["embeds"], 3, src)

    def test_embed_outside_package_fails(self) -> None:
        errs, _ = run(sources={**SOURCES, "src/lib.rs": 'include_str!("../../other/x.txt");'})
        self.assertEqual(errs, ["src/lib.rs embeds ../other/x.txt, which is not in the package"])

    def test_fixture_literal_file_and_directory(self) -> None:
        src = 'let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/world_snapshot_v2_stacked.bin");'
        errs, counts = run(sources={**SOURCES, "tests/snapshot.rs": src})
        self.assertEqual((errs, counts["fixtures"]), ([], 1))
        missing = [p for p in GOOD if not p.endswith("v2_stacked.bin")]
        errs, _ = run(paths=missing, sources={**SOURCES, "tests/snapshot.rs": src})
        self.assertEqual(errs, ["tests/snapshot.rs reads tests/fixtures/world_snapshot_v2_stacked.bin, "
                                "which is not in the package"])
        # a directory literal needs every tracked file under it
        dir_src = 'let d = "tests/fixtures";'
        errs, counts = run(paths=missing, sources={**SOURCES, "tests/snapshot.rs": dir_src})
        self.assertEqual(counts["fixtures"], 2)
        self.assertEqual(errs, ["tests/snapshot.rs reads tests/fixtures/world_snapshot_v2_stacked.bin, "
                                "which is not in the package"])

    def test_constants_match_the_written_lists(self) -> None:
        self.assertEqual(list(pc.FORBIDDEN_DIRS), EXPECTED_FORBIDDEN_DIRS)
        self.assertEqual(list(pc.FORBIDDEN_FILES), EXPECTED_FORBIDDEN_FILES)
        self.assertEqual(list(pc.REQUIRED), EXPECTED_REQUIRED)


if __name__ == "__main__":
    unittest.main()
