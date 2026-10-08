#!/usr/bin/env python3
"""Tests for scripts/downstream_select_tag.py (no network: the tags and their
manifests are given as data)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import downstream_select_tag as st  # noqa: E402

LOL = [
    ("Cargo.toml", '[workspace]\nmembers = ["alice-lol", "robot"]\n'),
    ("alice-lol/Cargo.toml",
     '[package]\nname = "alice-lol"\nversion = "0.4.2"\n\n[dependencies]\n'
     'alice-zip = { path = "../../ALICE-Zip", version = "0.6", default-features = false }\n'),
    ("robot/Cargo.toml",
     '[package]\nname = "robot"\nversion = "0.1.0"\n\n[dependencies]\n'
     'alice-kinematics = { path = "../../ALICE-Kinematics", version = "0.1", optional = true }\n'),
]
SDF = [("Cargo.toml", '[package]\nname = "alice-sdf"\nversion = "5.0.0"\n')]
PROVIDED = st.provided_packages([LOL, SDF])

ZIP_TAGS = ["refs/tags/v0.5.2", "refs/tags/v0.6.0", "refs/tags/v0.6.1-beta.1", "refs/tags/v0.7.0"]
KIN_TAGS = ["refs/tags/alice-kinematics-v0.1.0"]


def plain(_tag):
    return [("Cargo.toml", '[package]\nname = "x"\nversion = "0.0.0"\n')]


def select(crate, refs, manifests=plain, workspace=LOL):
    return st.select(crate, workspace, st.release_tags(refs, crate), PROVIDED, manifests)


class Requirements(unittest.TestCase):
    def test_caret_and_tilde_and_ranges(self):
        cases = [
            ("0.6", (0, 6, 0), True), ("0.6", (0, 6, 9), True), ("0.6", (0, 7, 0), False),
            ("1.3.0", (1, 5, 0), True), ("1.3.0", (2, 0, 0), False), ("1.3.0", (1, 2, 9), False),
            ("0.0.3", (0, 0, 3), True), ("0.0.3", (0, 0, 4), False),
            ("~1.2", (1, 2, 7), True), ("~1.2", (1, 3, 0), False), ("~1", (1, 9, 0), True),
            ("=0.6.0", (0, 6, 0), True), ("=0.6.0", (0, 6, 1), False),
            (">=0.6.1, <0.8", (0, 7, 5), True), (">=0.6.1, <0.8", (0, 6, 0), False),
            (">0.6", (0, 7, 0), True), (">0.6", (0, 6, 5), False), ("<=1.2.3", (1, 2, 3), True),
            ("0.6.*", (0, 6, 4), True), ("0.6.*", (0, 7, 0), False),
        ]
        for req, v, want in cases:
            self.assertEqual(st.satisfies(v, req), want, (req, v))

    def test_pre_release_tags_are_skipped(self):
        tags = st.release_tags(ZIP_TAGS + ["refs/tags/v1.0.0-rc.1"], "alice-zip")
        self.assertEqual(sorted(tags.values()), ["v0.5.2", "v0.6.0", "v0.7.0"])

    def test_crate_prefixed_tags_are_read(self):
        self.assertEqual(list(st.release_tags(KIN_TAGS, "alice-kinematics").values()), ["alice-kinematics-v0.1.0"])

    def test_provided_versions(self):
        self.assertEqual(PROVIDED, {"alice-lol": "0.4.2", "robot": "0.1.0", "alice-sdf": "5.0.0"})

    def test_workspace_inherited_version(self):
        ws = [("Cargo.toml", '[workspace.package]\nversion = "2.1.0"\n'),
              ("a/Cargo.toml", '[package]\nname = "a"\nversion.workspace = true\n')]
        self.assertEqual(st.provided_packages([ws]), {"a": "2.1.0"})


class Select(unittest.TestCase):
    def test_newest_tag_satisfying_the_workspace(self):
        tag, lines = select("alice-zip", ZIP_TAGS)
        self.assertEqual(tag, "v0.6.0", lines)  # not v0.7.0 (outside ^0.6), not the beta

    def test_newest_of_several_matching_tags(self):
        tag, lines = select("alice-zip", ["refs/tags/v0.6.0", "refs/tags/v0.6.3", "refs/tags/v0.6.1"])
        self.assertEqual(tag, "v0.6.3", lines)

    def test_every_requirement_must_hold(self):
        ws = LOL + [("b/Cargo.toml", '[dependencies]\nalice-zip = ">=0.6.1"\n')]
        tag, lines = select("alice-zip", ZIP_TAGS, workspace=ws)
        self.assertIsNone(tag)
        text = "\n".join(lines)
        self.assertIn("alice-lol/Cargo.toml [dependencies] alice-zip: 0.6", text)
        self.assertIn("b/Cargo.toml [dependencies] alice-zip: >=0.6.1", text)
        self.assertIn("release tags: v0.5.2, v0.6.0, v0.7.0", text)

    def test_reverse_requirement_rejects_the_tag(self):
        # the tag requires alice-lol 0.3 while 0.4.2 is placed next to it
        def manifests(_tag):
            return [("Cargo.toml", '[package]\nname = "alice-kinematics"\nversion = "0.1.0"\n\n[dependencies]\n'
                                   'alice-lol = { path = "../ALICE-LOL/alice-lol", version = "0.3", optional = true }\n')]
        tag, lines = select("alice-kinematics", KIN_TAGS, manifests)
        self.assertIsNone(tag)
        text = "\n".join(lines)
        self.assertIn("alice-kinematics-v0.1.0: Cargo.toml [dependencies] alice-lol requires alice-lol 0.3, placed 0.4.2", text)
        self.assertIn("no release tag of `alice-kinematics` fits", text)

    def test_reverse_check_falls_back_to_an_older_tag(self):
        def manifests(tag):
            req = "4" if tag == "v0.6.1" else "5"
            return [("sub/Cargo.toml", f'[dependencies]\nalice-sdf = {{ path = "../../ALICE-SDF", version = "{req}" }}\n')]
        tag, lines = select("alice-zip", ["refs/tags/v0.6.0", "refs/tags/v0.6.1"], manifests)
        self.assertEqual(tag, "v0.6.0", lines)

    def test_reverse_requirement_accepting_the_placed_version_passes(self):
        def manifests(_tag):
            return [("Cargo.toml", '[dependencies]\nalice-lol = { path = "x", version = "0.4" }\n')]
        tag, _ = select("alice-kinematics", KIN_TAGS, manifests)
        self.assertEqual(tag, "alice-kinematics-v0.1.0")

    def test_registry_dependency_of_a_tag_is_not_checked_against_the_placed_version(self):
        def manifests(_tag):
            return [("Cargo.toml", '[dependencies]\nalice-lol = "0.3"\n')]
        self.assertEqual(select("alice-kinematics", KIN_TAGS, manifests)[0], "alice-kinematics-v0.1.0")

    def test_unrelated_dependencies_are_ignored(self):
        def manifests(_tag):
            return [("Cargo.toml", '[dependencies]\nserde = "0.1"\n')]
        self.assertEqual(select("alice-zip", ZIP_TAGS, manifests)[0], "v0.6.0")

    def test_no_requirement_in_the_workspace_fails(self):
        tag, lines = select("alice-llm", ["refs/tags/v1.5.0"])
        self.assertIsNone(tag)
        self.assertIn("no Cargo.toml of the workspace requires `alice-llm`", lines[0])

    def test_dependency_without_version_fails(self):
        ws = [("a/Cargo.toml", '[dependencies]\nalice-zip = { path = "../../ALICE-Zip" }\n')]
        tag, lines = select("alice-zip", ZIP_TAGS, workspace=ws)
        self.assertIsNone(tag)
        self.assertIn("has no version requirement", lines[0])

    def test_no_tags_fails(self):
        tag, lines = select("alice-zip", [])
        self.assertIsNone(tag)
        self.assertIn("release tags: (none)", "\n".join(lines))


class ManifestForms(unittest.TestCase):
    """Dependencies written in every form Cargo accepts are read; `[features]`
    entries are not requirements."""

    def reqs(self, text, crate="alice-zip"):
        return [r for _, r in st.dependency_requirements(text, crate)]

    def test_table_form_with_version_on_a_later_line(self):
        text = '[dependencies.alice-zip]\npath = "../../ALICE-Zip"\nversion = "0.5"\ndefault-features = false\n'
        self.assertEqual(self.reqs(text), ["0.5"])

    def test_dev_build_and_target_tables(self):
        text = ('[dev-dependencies.alice-zip]\nversion = "0.6"\n\n'
                '[build-dependencies]\nalice-zip = "0.6.1"\n\n'
                "[target.'cfg(unix)'.dependencies.alice-zip]\nversion = \"0.6.2\"\n\n"
                '[target.wasm32-unknown-unknown.dev-dependencies]\nalice-zip = { version = "0.6.3" }\n')
        self.assertEqual(sorted(self.reqs(text)), ["0.6", "0.6.1", "0.6.2", "0.6.3"])

    def test_renamed_dependency_is_read_under_its_package(self):
        text = '[dependencies]\nzip = { package = "alice-zip", path = "x", version = "0.5" }\nalice-zip-cli = "9"\n'
        self.assertEqual(self.reqs(text), ["0.5"])
        self.assertEqual(self.reqs(text, "zip"), [])

    def test_feature_entries_are_not_requirements(self):
        text = ('[dependencies]\nalice-zip = { path = "x", version = "0.6", optional = true }\n\n'
                '[features]\nalice-zip = ["dep:alice-zip"]\nzip = ["alice-zip/std"]\n')
        self.assertEqual(self.reqs(text), ["0.6"])

    def test_workspace_inherited_dependency_takes_the_root_requirement(self):
        ws = [("Cargo.toml", '[workspace]\nmembers = ["a"]\n\n[workspace.dependencies]\n'
                             'alice-zip = { path = "../ALICE-Zip", version = "0.5" }\n'),
              ("a/Cargo.toml", '[package]\nname = "a"\nversion = "0.1.0"\n\n[dependencies]\nalice-zip.workspace = true\n')]
        tag, lines = select("alice-zip", ZIP_TAGS, workspace=ws)
        self.assertEqual(tag, "v0.5.2", lines)

    def test_table_form_in_the_workspace_selects_by_it(self):
        ws = LOL + [("fuzz/Cargo.toml", '[dependencies.alice-zip]\npath = "../../ALICE-Zip"\nversion = "=0.6.0"\n')]
        tag, lines = select("alice-zip", ZIP_TAGS + ["refs/tags/v0.6.4"], workspace=ws)
        self.assertEqual(tag, "v0.6.0", lines)  # not v0.6.4, which the fuzz manifest excludes

    def test_table_form_in_a_tag_rejects_it(self):
        def manifests(_tag):
            return [("fuzz/Cargo.toml", '[dependencies.alice-lol]\npath = "../../ALICE-LOL/alice-lol"\nversion = "0.3"\n')]
        tag, lines = select("alice-kinematics", KIN_TAGS, manifests)
        self.assertIsNone(tag, lines)
        self.assertIn("fuzz/Cargo.toml [dependencies] alice-lol requires alice-lol 0.3, placed 0.4.2", "\n".join(lines))

    def test_feature_entry_in_a_tag_is_not_a_missing_version(self):
        def manifests(_tag):
            return [("Cargo.toml", '[dependencies]\nalice-lol = { path = "x", version = "0.4", optional = true }\n\n'
                                   '[features]\nalice-lol = ["dep:alice-lol"]\n')]
        self.assertEqual(select("alice-kinematics", KIN_TAGS, manifests)[0], "alice-kinematics-v0.1.0")

    def test_feature_entry_in_the_workspace_is_not_a_missing_version(self):
        ws = LOL + [("x/Cargo.toml", '[features]\nalice-zip = ["dep:alice-zip"]\n')]
        self.assertEqual(select("alice-zip", ZIP_TAGS, workspace=ws)[0], "v0.6.0")

    def test_invalid_manifest_is_an_error(self):
        with self.assertRaises(ValueError):
            st.dependency_requirements("[dependencies\nalice-zip = 1", "alice-zip", "bad/Cargo.toml")


class PreRelease(unittest.TestCase):
    def test_pre_release_needs_a_pre_release_comparator_on_the_same_version(self):
        cases = [
            ("0.3.0-beta.2", "0.3", False), ("0.3.0-beta.2", ">=0.2", False), ("0.3.0-beta.2", "*", False),
            ("0.3.0-beta.2", "0.3.0-beta.1", True), ("0.3.0-beta.2", ">=0.3.0-beta.1", True),
            ("0.3.0-beta.2", "=0.3.0-beta.2", True), ("0.3.0-beta.2", "=0.3.0-beta.3", False),
            ("0.3.0-beta.2", "0.3.0-beta.10", False),  # numeric identifiers compare as numbers
            ("0.3.0-beta.2", ">=0.2.0-beta.1", False),  # a pre-release comparator on another version
            ("1.0.0-rc.1", ">=1.0.0-alpha, <1.0.0", True), ("1.0.0", "1.0.0-rc.1", True),
            ("1.0.0-rc.1", "1.0.0", False),
        ]
        for v, req, want in cases:
            self.assertEqual(st.satisfies(v, req), want, (v, req))

    def test_placed_pre_release_rejects_a_tag_requiring_the_release(self):
        provided = dict(PROVIDED, **{"alice-db": "0.3.0-beta.2"})

        def manifests(tag):
            req = "0.3" if tag == "v0.6.1" else "0.3.0-beta.1"
            return [("Cargo.toml", f'[dependencies]\nalice-db = {{ path = "x", version = "{req}" }}\n')]
        tag, lines = st.select("alice-zip", LOL, st.release_tags(["refs/tags/v0.6.0", "refs/tags/v0.6.1"], "alice-zip"),
                               provided, manifests)
        self.assertEqual(tag, "v0.6.0", lines)


class Providers(unittest.TestCase):
    """The reverse check covers alice-physics of this checkout and the sibling
    repositories placed before the one being selected."""

    def test_tag_requiring_an_older_alice_physics_is_rejected(self):
        physics = [("Cargo.toml", '[package]\nname = "alice-physics"\nversion = "2.0.0"\n')]
        provided = st.provided_packages([LOL, SDF, physics])

        def manifests(tag):
            req = "1.0" if tag == "v0.6.1" else "2"
            return [("Cargo.toml", f'[dependencies]\nalice-physics = {{ path = "../ALICE-Physics", version = "{req}" }}\n')]
        tags = st.release_tags(["refs/tags/v0.6.0", "refs/tags/v0.6.1"], "alice-zip")
        tag, lines = st.select("alice-zip", LOL, tags, provided, manifests)
        self.assertEqual(tag, "v0.6.0", lines)
        tag, lines = st.select("alice-zip", LOL, {k: v for k, v in tags.items() if v == "v0.6.1"}, provided, manifests)
        self.assertIsNone(tag)
        self.assertIn("requires alice-physics 1.0, placed 2.0.0", "\n".join(lines))

    def test_tag_requiring_a_sibling_is_checked_against_the_placed_sibling(self):
        llm = [("Cargo.toml", '[package]\nname = "alice-llm"\nversion = "1.5.0"\n')]
        provided = st.provided_packages([LOL, SDF, llm])

        def manifests(tag):
            req = "1.6" if tag == "v0.6.1" else "1.3"
            return [("Cargo.toml", f'[dependencies]\nalice-llm = {{ path = "../ALICE-LLM", version = "{req}" }}\n')]
        tags = st.release_tags(["refs/tags/v0.6.0", "refs/tags/v0.6.1"], "alice-zip")
        tag, lines = st.select("alice-zip", LOL, tags, provided, manifests)
        self.assertEqual(tag, "v0.6.0", lines)


class Check(unittest.TestCase):
    """--check: the placed downstreams must accept the version of this checkout."""

    PHYSICS = {"alice-physics": "2.0.0"}

    def run_check(self, *trees):
        return st.check([(f"T{i}", m) for i, m in enumerate(trees)], self.PHYSICS)

    def test_old_requirement_fails_with_the_reason(self):
        sdf = [("Cargo.toml", '[package]\nname = "alice-sdf"\nversion = "5.0.0"\n\n'
                              '[dependencies]\nalice-physics = { version = "1.1", optional = true }\n')]
        ok, lines = self.run_check(sdf)
        self.assertFalse(ok)
        self.assertIn("error: T0/Cargo.toml [dependencies] alice-physics requires alice-physics 1.1, placed 2.0.0",
                      lines)

    def test_requirement_of_a_member_in_table_form_fails(self):
        lol = [("Cargo.toml", '[workspace]\nmembers = ["a"]\n'),
               ("a/Cargo.toml", '[dependencies.alice-physics]\npath = "../../ALICE-Physics"\nversion = "1.0"\n')]
        ok, lines = self.run_check(lol)
        self.assertFalse(ok, lines)

    def test_matching_requirements_pass(self):
        sdf = [("Cargo.toml", '[dependencies]\nalice-physics = { version = "2", optional = true }\n')]
        trt = [("Cargo.toml", '[dependencies]\nalice-physics = { path = "../ALICE-Physics", optional = true }\n')]
        ok, lines = self.run_check(sdf, trt)
        self.assertTrue(ok, lines)

    def test_tree_without_a_requirement_fails(self):
        ok, lines = self.run_check([("Cargo.toml", '[dependencies]\nserde = "1"\n\n[features]\nalice-physics = []\n')])
        self.assertFalse(ok)
        self.assertIn("no Cargo.toml requires any of alice-physics 2.0.0", lines[0])


if __name__ == "__main__":
    unittest.main()
