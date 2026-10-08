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
        self.assertIn("alice-lol/Cargo.toml:6: 0.6", text)
        self.assertIn("b/Cargo.toml:2: >=0.6.1", text)
        self.assertIn("release tags: v0.5.2, v0.6.0, v0.7.0", text)

    def test_reverse_requirement_rejects_the_tag(self):
        # the tag requires alice-lol 0.3 while 0.4.2 is placed next to it
        def manifests(_tag):
            return [("Cargo.toml", '[package]\nname = "alice-kinematics"\nversion = "0.1.0"\n\n[dependencies]\n'
                                   'alice-lol = { path = "../ALICE-LOL/alice-lol", version = "0.3", optional = true }\n')]
        tag, lines = select("alice-kinematics", KIN_TAGS, manifests)
        self.assertIsNone(tag)
        text = "\n".join(lines)
        self.assertIn("alice-kinematics-v0.1.0: Cargo.toml:6 requires alice-lol 0.3, placed 0.4.2", text)
        self.assertIn("no release tag of `alice-kinematics` fits", text)

    def test_reverse_check_falls_back_to_an_older_tag(self):
        def manifests(tag):
            req = "4" if tag == "v0.6.1" else "5"
            return [("sub/Cargo.toml", f'[dependencies]\nalice-sdf = "{req}"\n')]
        tag, lines = select("alice-zip", ["refs/tags/v0.6.0", "refs/tags/v0.6.1"], manifests)
        self.assertEqual(tag, "v0.6.0", lines)

    def test_reverse_requirement_accepting_the_placed_version_passes(self):
        def manifests(_tag):
            return [("Cargo.toml", '[dependencies]\nalice-lol = { path = "x", version = "0.4" }\n')]
        tag, _ = select("alice-kinematics", KIN_TAGS, manifests)
        self.assertEqual(tag, "alice-kinematics-v0.1.0")

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


if __name__ == "__main__":
    unittest.main()
