#!/usr/bin/env python3
"""Tests for scripts/downstream_major_rewrite.py (no network: the index entry is
given as data or as a file). The last class builds a small provider and consumer
with cargo (offline, no dependencies) to show that the rewrite only edits the
requirement: a consumer that fits the new major compiles, one that calls an API
the new major changed does not.

run: python3 scripts/test_downstream_major_rewrite.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import downstream_major_rewrite as mr  # noqa: E402
import downstream_select_tag as st  # noqa: E402

C = "alice-physics"


def index(*versions: str, yanked: tuple[str, ...] = ()) -> str:
    return "\n".join(json.dumps({"name": C, "vers": v, "yanked": v in yanked}) for v in versions) + "\n"


class Index(unittest.TestCase):
    def test_newest_release(self) -> None:
        self.assertEqual(mr.newest_published(index("1.4.0", "2.2.0", "2.0.0")), "2.2.0")

    def test_yanked_and_prerelease_skipped(self) -> None:
        self.assertEqual(mr.newest_published(index("2.1.0", "2.2.0", "3.0.0-beta.1", yanked=("2.2.0",))), "2.1.0")

    def test_empty_index_fails(self) -> None:
        with self.assertRaises(ValueError):
            mr.newest_published("")
        with self.assertRaises(ValueError):
            mr.newest_published(index("3.0.0-beta.1"))

    def test_index_path(self) -> None:
        self.assertEqual(mr.index_path("alice-physics"), "al/ic/alice-physics")
        self.assertEqual(mr.index_path("abc"), "3/a/abc")
        self.assertEqual(mr.index_path("ab"), "2/ab")


class Bump(unittest.TestCase):
    def test_major(self) -> None:
        self.assertTrue(mr.is_major_bump("2.2.0", "3.0.0"))
        self.assertTrue(mr.is_major_bump("0.4.1", "0.5.0"))
        self.assertTrue(mr.is_major_bump("2.2.0", "3.0.0-beta.1"))

    def test_not_major(self) -> None:
        for pub, v in [("2.2.0", "2.3.0"), ("2.2.0", "2.2.1"), ("2.2.0", "2.2.0"), ("3.0.0", "2.9.0"),
                       ("0.4.1", "0.4.2")]:
            self.assertFalse(mr.is_major_bump(pub, v), (pub, v))


MANIFEST = '''[package]
name = "consumer"
version = "1.0.0"

[dependencies]
alice-zip = { version = "2.2", default-features = false }
alice-physics = { version = "2.2", optional = true }
other = "2.2"

[target.'cfg(unix)'.dependencies]
"alice-physics" = "2.0"

[dev-dependencies.alice-physics]
path = "../../ALICE-Physics"
version = "2.1"
features = ["std"]

[build-dependencies]
phys = { package = "alice-physics", path = "../ALICE-Physics", version = "^2" }

[workspace.dependencies]
alice-physics = "2"
'''


def reqs(text: str) -> list[tuple[str, str | None]]:
    return st.dependency_requirements(text, C)


class Rewrite(unittest.TestCase):
    def test_every_form_rewritten_and_nothing_else(self) -> None:
        new, diff = mr.rewrite_text(MANIFEST, C, "2.2.0", "3.0.0")
        self.assertEqual({r for _, r in reqs(new)}, {"3.0.0"})
        self.assertEqual(len(reqs(new)), len(reqs(MANIFEST)))
        self.assertEqual(len(diff), 2 * 5)
        # other dependencies with the same version string are untouched
        self.assertIn('alice-zip = { version = "2.2", default-features = false }', new)
        self.assertIn('other = "2.2"', new)
        self.assertEqual(new.count("\n"), MANIFEST.count("\n"))
        # every requirement now accepts the checkout's version
        self.assertTrue(all(st.satisfies("3.0.0", r) for _, r in reqs(new)))

    def test_renamed_table_form(self) -> None:
        text = ('[package]\nname = "c"\nversion = "1.0.0"\n\n[dependencies.phys]\n'
                'package = "alice-physics"\nversion = "2.2"\n')
        new, diff = mr.rewrite_text(text, C, "2.2.0", "3.0.0")
        self.assertEqual(reqs(new), [("[dependencies] phys", "3.0.0")])
        self.assertEqual(diff, ['-version = "2.2"', '+version = "3.0.0"'])

    def test_stale_requirement_left_and_still_rejected(self) -> None:
        # requires 1.x while 2.2.0 is published: not this bump's consumer, left for the check to fail
        text = '[package]\nname = "c"\nversion = "1.0.0"\n\n[dependencies]\nalice-physics = "1.1"\n'
        new, diff = mr.rewrite_text(text, C, "2.2.0", "3.0.0")
        self.assertEqual((new, diff), (text, []))
        ok, _ = st.check([("c", [("Cargo.toml", new)])], {C: "3.0.0"})
        self.assertFalse(ok)

    def test_versionless_path_dependency_left(self) -> None:
        text = ('[package]\nname = "c"\nversion = "1.0.0"\n\n[dependencies]\n'
                'alice-physics = { path = "../ALICE-Physics", optional = true }\n')
        self.assertEqual(mr.rewrite_text(text, C, "2.2.0", "3.0.0"), (text, []))

    def test_already_accepting_left(self) -> None:
        text = '[package]\nname = "c"\nversion = "1.0.0"\n\n[dependencies]\nalice-physics = "3"\n'
        self.assertEqual(mr.rewrite_text(text, C, "2.2.0", "3.0.0"), (text, []))

    def test_unparseable_edit_fails(self) -> None:
        # a form the line edit does not handle (a multi-line basic string): the
        # re-parse check refuses instead of leaving the requirement unchanged
        text = ('[package]\nname = "c"\nversion = "1.0.0"\n\n[dependencies]\n'
                'alice-physics = { optional = true, version = """2.2""" }\n')
        with self.assertRaises(ValueError):
            mr.rewrite_text(text, C, "2.2.0", "3.0.0")


def write_tree(root: str, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)


class Main(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()
        write_tree(self.dir, {
            "phys/Cargo.toml": '[package]\nname = "alice-physics"\nversion = "3.0.0"\n',
            "down/Cargo.toml": MANIFEST,
            "down/sub/Cargo.toml": '[package]\nname = "s"\nversion = "0.1.0"\n\n[dependencies]\n'
                                   'alice-physics.workspace = true\n',
        })
        self.idx = os.path.join(self.dir, "index")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_main(self, *versions: str) -> int:
        with open(self.idx, "w", encoding="utf-8") as f:
            f.write(index(*versions))
        return mr.main(["--provider", os.path.join(self.dir, "phys"), "--tree", os.path.join(self.dir, "down"),
                        "--index-file", self.idx])

    def read(self, rel: str) -> str:
        with open(os.path.join(self.dir, rel), encoding="utf-8") as f:
            return f.read()

    def test_major_bump_rewrites_on_disk(self) -> None:
        self.assertEqual(self.run_main("2.1.0", "2.2.0"), 0)
        self.assertEqual({r for _, r in reqs(self.read("down/Cargo.toml"))}, {"3.0.0"})
        ok, _ = st.check([("down", st.manifests_on_disk(os.path.join(self.dir, "down")))], {C: "3.0.0"})
        self.assertTrue(ok)

    def test_not_a_bump_leaves_files_byte_identical(self) -> None:
        # 3.0.0 already published: the checkout is no bump and the 2.x requirements stay (the check then fails)
        self.assertEqual(self.run_main("2.2.0", "3.0.0"), 0)
        self.assertEqual(self.read("down/Cargo.toml"), MANIFEST)
        ok, _ = st.check([("down", st.manifests_on_disk(os.path.join(self.dir, "down")))], {C: "3.0.0"})
        self.assertFalse(ok)

    def test_minor_commit_leaves_files(self) -> None:
        write_tree(self.dir, {"phys/Cargo.toml": '[package]\nname = "alice-physics"\nversion = "2.3.0"\n'})
        self.assertEqual(self.run_main("2.2.0"), 0)
        self.assertEqual(self.read("down/Cargo.toml"), MANIFEST)

    def test_bump_with_no_consumer_requirement_fails(self) -> None:
        # every requirement is versionless or stale: a bump that rewrote nothing fails
        write_tree(self.dir, {"down/Cargo.toml": '[package]\nname = "c"\nversion = "1.0.0"\n\n[dependencies]\n'
                                                 'alice-physics = { path = "../phys" }\nx = "1"\n',
                              "down/sub/Cargo.toml": '[package]\nname = "s"\nversion = "0.1.0"\n\n[dependencies]\n'
                                                     'alice-physics = "1.1"\n'})
        self.assertEqual(self.run_main("2.2.0"), 1)

    def test_a_range_that_accepts_both_majors_needs_no_edit(self) -> None:
        # ">=2.0, <4" accepts the published 2.2.0 and this 3.0.0: nothing to rewrite, and the bump
        # is still checked against that consumer (it is not "no requirement names it")
        ranged = ('[package]\nname = "c"\nversion = "1.0.0"\n\n[dependencies]\n'
                  'alice-physics = { path = "../../phys", version = ">=2.0, <4", optional = true }\n')
        write_tree(self.dir, {"down/Cargo.toml": ranged,
                              "down/sub/Cargo.toml": '[package]\nname = "s"\nversion = "0.1.0"\n'})
        self.assertEqual(self.run_main("2.1.0", "2.2.0"), 0)
        self.assertEqual(self.read("down/Cargo.toml"), ranged)

    def test_a_range_that_stops_below_the_bump_still_fails(self) -> None:
        # ">=2.0, <3" accepts the published release but not 3.0.0 and is not a caret the rewrite
        # raises: nothing accepts the bump, so it fails
        write_tree(self.dir, {"down/Cargo.toml": '[package]\nname = "c"\nversion = "1.0.0"\n\n[dependencies]\n'
                                                 'alice-physics = { version = ">=2.0, <3" }\n',
                              "down/sub/Cargo.toml": '[package]\nname = "s"\nversion = "0.1.0"\n'})
        rc = self.run_main("2.2.0")
        if rc == 0:   # the rewrite raised it: then it must accept 3.0.0 now
            self.assertTrue(any(st.satisfies("3.0.0", r) for _, r in reqs(self.read("down/Cargo.toml"))))
        else:
            self.assertEqual(rc, 1)

    def test_rerun_over_edited_copies_passes(self) -> None:
        self.assertEqual(self.run_main("2.2.0"), 0)
        edited = self.read("down/Cargo.toml")
        self.assertEqual(self.run_main("2.2.0"), 0)
        self.assertEqual(self.read("down/Cargo.toml"), edited)

    def test_index_without_release_fails(self) -> None:
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(self.read("down/Cargo.toml"), MANIFEST)


class CargoBuild(unittest.TestCase):
    """A provider at 3.0.0 and a consumer written for 2.x (path + version, as
    ALICE-LOL depends on alice-physics). The requirement edit alone lets the
    consumer build; a provider whose public signature changed does not."""

    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, provider_lib: str) -> subprocess.CompletedProcess:
        write_tree(self.dir, {
            "ALICE-Physics/Cargo.toml": '[package]\nname = "alice-physics"\nversion = "3.0.0"\nedition = "2021"\n',
            "ALICE-Physics/src/lib.rs": provider_lib,
            "down/Cargo.toml": '[package]\nname = "down"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n'
                               'alice-physics = { path = "../ALICE-Physics", version = "2.2" }\n\n[workspace]\n',
            "down/src/lib.rs": "pub fn twice(x: i64) -> i64 { alice_physics::step(x) * 2 }\n",
        })
        idx = os.path.join(self.dir, "index")
        with open(idx, "w", encoding="utf-8") as f:
            f.write(index("2.2.0"))
        rc = mr.main(["--provider", os.path.join(self.dir, "ALICE-Physics"), "--tree", os.path.join(self.dir, "down"),
                      "--index-file", idx])
        self.assertEqual(rc, 0)
        env = dict(os.environ, CARGO_TARGET_DIR=os.path.join(self.dir, "target"))
        return subprocess.run(["cargo", "check", "--offline", "-q"], cwd=os.path.join(self.dir, "down"),
                              capture_output=True, text=True, env=env)

    def test_requirement_only_edit_builds(self) -> None:
        r = self.build("pub fn step(x: i64) -> i64 { x + 1 }\n")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_api_break_is_red(self) -> None:
        # the new major changed `step`'s signature: the consumer must fail to compile
        r = self.build("pub fn step(x: i64, dt: i64) -> i64 { x + dt }\n")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("E0061", r.stderr)

    def test_without_rewrite_the_requirement_rejects(self) -> None:
        # control: the same consumer without the edit does not resolve against 3.0.0
        write_tree(self.dir, {
            "ALICE-Physics/Cargo.toml": '[package]\nname = "alice-physics"\nversion = "3.0.0"\nedition = "2021"\n',
            "ALICE-Physics/src/lib.rs": "pub fn step(x: i64) -> i64 { x + 1 }\n",
            "down/Cargo.toml": '[package]\nname = "down"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\n'
                               'alice-physics = { path = "../ALICE-Physics", version = "2.2" }\n\n[workspace]\n',
            "down/src/lib.rs": "pub fn twice(x: i64) -> i64 { alice_physics::step(x) * 2 }\n",
        })
        env = dict(os.environ, CARGO_TARGET_DIR=os.path.join(self.dir, "target"))
        r = subprocess.run(["cargo", "check", "--offline", "-q"], cwd=os.path.join(self.dir, "down"),
                           capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
