#!/usr/bin/env python3
"""Oracle for scripts/preflight_tools_check.py: the programs, toolchains and targets a preflight calls are read from
its text (assignments, paths, comments, builtins and its own functions are not programs; a --target in a python
script it runs counts), each missing one is named, and a preflight that names no program fails."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight_tools_check as pt  # noqa: E402

PREFLIGHT = """#!/usr/bin/env bash
set -euo pipefail
# cargo-fake is only mentioned in a comment: if command -v cargo-fake; cargo build --target armv7-fake-none
step() { echo "== $1"; }
fast=1
step "lint"
actionlint .github/workflows/*.yml
cargo +1.85 test --lib
cargo +nightly-2026-09-26 public-api --version
if command -v cargo-deny >/dev/null; then cargo deny check; fi
(cd web && npm ci)
v=$(rustc --version)
python3 scripts/probe.py
scripts/run.sh
rustup target add wasm32-wasip1
cargo build --target wasm32-wasip1
"""
PROBE = 'ARGS = ["--target", "x86_64-unknown-linux-gnu", "--features", "std"]\n'


def repo(preflight: str = PREFLIGHT, probe: str = PROBE) -> Path:
    d = Path(tempfile.mkdtemp())
    (d / "scripts").mkdir()
    (d / "scripts" / "preflight.sh").write_text(preflight, encoding="utf-8")
    (d / "scripts" / "probe.py").write_text(probe, encoding="utf-8")
    return d


def everything(*_a) -> str:
    return "x"


class ToolsCheck(unittest.TestCase):
    def test_the_needs_are_read_from_the_preflight(self):
        tools, chains, targets, api = pt.needs(repo())
        self.assertEqual(tools, {"actionlint", "cargo", "cargo-deny", "npm", "rustc", "python3", "rustup"})
        self.assertEqual(chains, {"1.85", "nightly-2026-09-26"})
        self.assertEqual(targets, {"x86_64-unknown-linux-gnu"}, "a --target in a script the preflight runs counts")
        self.assertEqual(api, "nightly-2026-09-26", "the toolchain of the public-api command")
        self.assertNotIn("cargo-fake", tools, "a comment names no program")
        self.assertNotIn("fast", tools, "an assignment is not a program")
        self.assertNotIn("step", tools, "a function the preflight defines is not a program")
        self.assertNotIn("scripts", tools, "a path is not a program")

    def test_a_test_scripts_fixture_targets_are_not_needed(self):
        d = repo(preflight=PREFLIGHT + "python3 scripts/test_probe.py\n")
        (d / "scripts" / "test_probe.py").write_text('CASE = ["--target", "armv7-fake-none"]\n', encoding="utf-8")
        self.assertEqual(pt.needs(d)[2], {"x86_64-unknown-linux-gnu"})

    def test_a_target_the_preflight_installs_itself_is_not_needed(self):
        self.assertNotIn("wasm32-wasip1", pt.needs(repo())[2])

    def test_each_missing_one_is_named(self):
        gone, _ = pt.missing(repo(), which=lambda t: None if t in ("npm", "cargo-deny") else "/bin/" + t,
                             toolchains=lambda: "1.85-aarch64-apple-darwin\n", targets=lambda: ["wasm32-wasip1"],
                             public_api=lambda tc: tc != "nightly-2026-09-26")
        self.assertEqual(gone, ["cargo-deny", "npm", "toolchain nightly-2026-09-26", "target x86_64-unknown-linux-gnu",
                                "cargo +nightly-2026-09-26 public-api"])

    def test_all_present_passes(self):
        gone, _ = pt.missing(repo(), which=everything, toolchains=lambda: "1.85-x\nnightly-2026-09-26-x\n",
                             targets=lambda: ["x86_64-unknown-linux-gnu"], public_api=lambda tc: True)
        self.assertEqual(gone, [])

    def test_only_targets_checks_the_targets_alone(self):
        gone, _ = pt.missing(repo(), only="targets", which=lambda t: None, toolchains=lambda: "",
                             targets=lambda: [], public_api=lambda tc: False)
        self.assertEqual(gone, ["target x86_64-unknown-linux-gnu"])

    def test_a_toolchain_name_must_match_whole(self):
        gone, _ = pt.missing(repo(), only="toolchains", toolchains=lambda: "1.850-x\nnightly-2026-09-26-x\nnightly-x\n")
        self.assertEqual(gone, ["toolchain 1.85"])

    def test_an_unpinned_nightly_does_not_stand_for_a_pinned_one(self):
        gone, _ = pt.missing(repo(), only="toolchains", toolchains=lambda: "1.85-x\nnightly-aarch64-apple-darwin\n")
        self.assertEqual(gone, ["toolchain nightly-2026-09-26"])

    def test_a_preflight_that_names_no_program_fails(self):
        d = repo(preflight="#!/usr/bin/env bash\n# nothing\n", probe="")
        self.assertEqual(pt.main(["--root", str(d), "--only", "targets"]), 1)


if __name__ == "__main__":
    unittest.main()
