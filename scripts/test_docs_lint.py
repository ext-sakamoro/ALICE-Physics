#!/usr/bin/env python3
"""Tests for scripts/docs_lint.py.

Each case builds a small tree in a temporary directory, breaks exactly one
thing, and asserts that the linter reports it. The first case runs the linter
against this repository.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docs_lint as dl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CARGO = '[package]\nname = "demo"\nversion = "1.5.0"\n\n[features]\n'

CHANGELOG = """# Changelog

## [Unreleased]

### Added

- `world::step_n`: run several steps

### Changed

- **Behavior change:** `RigidBody::new` friction default 0.3 -> 0.5

### Fixed

- `fluid::kernel` normalisation

## [1.4.0] - 2026-09-17

### Added — old style heading kept as history

## [1.3.0] - 2026-09-16
"""

README = "# demo\n\nA physics engine. The memory footprint is small.\n"


def tree(overrides: dict[str, str] | None = None, drop: tuple[str, ...] = ()) -> str:
    files = {
        "Cargo.toml": CARGO,
        "CHANGELOG.md": CHANGELOG,
        "README.md": README,
        "README_JP.md": README,
        "docs/MODULES.md": "# Modules\n",
    }
    files.update(overrides or {})
    d = tempfile.mkdtemp()
    for rel, text in files.items():
        if rel in drop:
            continue
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
    return d


def errors(overrides=None, drop=()):
    return dl.check(tree(overrides, drop))[0]


class RealRepo(unittest.TestCase):
    def test_this_repository_passes(self):
        errs, counts = dl.check(ROOT)
        self.assertEqual(errs, [])
        self.assertGreater(counts["versions"], 5)


class Vocabulary(unittest.TestCase):
    def test_clean_tree_passes(self):
        self.assertEqual(errors(), [])

    def test_agent_process_word(self):
        e = errors({"README.md": README + "Measured by worker 3.\n"})
        self.assertTrue(any("agent process `worker`" in x for x in e), e)

    def test_every_hit_on_a_line_is_reported(self):
        # one hit must not hide a second hit of the same kind on the same line
        e = errors({"README.md": README + "worker 1 and subagent 2 measured it\n"})
        self.assertEqual(len([x for x in e if "agent process" in x]), 2, e)

    def test_internal_tracker_word(self):
        e = errors({"CHANGELOG.md": CHANGELOG.replace("run several steps", "run several steps (Backlog item)")})
        self.assertTrue(any("internal tracker `Backlog`" in x for x in e), e)

    def test_session_name(self):
        e = errors({"README_JP.md": README + "ys-00 が検出\n"})
        self.assertTrue(any("session name" in x for x in e), e)

    def test_private_names_are_matched_by_hash_including_phrases(self):
        # stand-in names: the real ones are only stored as hashes in docs_lint.py
        import hashlib
        fake = {hashlib.sha256(n.encode()).hexdigest() for n in ("zorblax", "quux-lab", "acme rocket works")}
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = fake
            e = errors({"docs/MODULES.md": "consumers: Zorblax, quux-lab and the Acme Rocket Works team\n"})
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        hits = sorted(x.split("(`")[1].rstrip("`)") for x in e if "private or internal name" in x)
        self.assertEqual(hits, ["Acme Rocket Works", "Zorblax", "quux-lab"], e)

    def test_private_names_are_found_in_every_file_and_path_of_the_tree(self):
        # comments, generated ledgers, workflows and scripts are published too
        import hashlib
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = {hashlib.sha256(b"zorblax").hexdigest()}
            e = errors({"src/a.rs": "// see the Zorblax notes\nfn a() {}\n",
                        ".github/workflows/ci.yml": "      # Zorblax rule\n",
                        "docs/oracle-status.md": "For details: [ZORBLAX.md](../ZORBLAX.md)\n",
                        "src/zorblax_glue.rs": "fn b() {}\n"})
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        where = sorted({x.split(":")[0] for x in e if "private or internal name" in x})
        self.assertEqual(where, [".github/workflows/ci.yml", "docs/oracle-status.md", "src/a.rs",
                                 "src/zorblax_glue.rs"], e)
        self.assertTrue(any("in a file path" in x for x in e), e)

    def test_binary_files_are_not_read_as_text(self):
        import hashlib
        d = tree()
        with open(os.path.join(d, "blob.bin"), "wb") as f:
            f.write(b"zorblax\0\x01\x02")
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = {hashlib.sha256(b"zorblax").hexdigest()}
            e, counts = dl.check(d)
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        self.assertEqual(e, [])
        self.assertGreater(counts["tree files"], 0)

    def test_a_japanese_name_glued_to_a_particle_is_still_found(self):
        import hashlib
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = saved | {hashlib.sha256("ゾルバクス".encode()).hexdigest(),
                                              hashlib.sha256(b"zorb").hexdigest()}
            for text in ("ゾルバクスの世界", "第二章ゾルバクス編", "zorbの設定"):
                self.assertTrue(dl.private_names(text), text)
            # a name only as part of a longer katakana word is not that name
            self.assertEqual(dl.private_names("ゾルバクスター"), [])
        finally:
            dl.PRIVATE_NAME_HASHES = saved

    def test_the_hash_list_is_not_empty_and_holds_no_plain_names(self):
        self.assertGreaterEqual(len(dl.PRIVATE_NAME_HASHES), 10)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in dl.PRIVATE_NAME_HASHES))

    def test_private_address_and_device(self):
        # separate inputs: a device name and an address are tested apart, never as a pair
        e = errors({"README.md": README + "the build host is a Jetson board\n"})
        self.assertTrue(any("device `Jetson`" in x for x in e), e)
        e = errors({"README.md": README + "the service listens on 100.127.255.254\n"})
        self.assertTrue(any("private address `100.127.255.254`" in x for x in e), e)

    def test_owned_device_names_are_held_as_hashes(self):
        self.assertGreaterEqual(len(dl.DEVICE_NAME_HASHES), 3)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in dl.DEVICE_NAME_HASHES))
        self.assertLessEqual(dl.DEVICE_NAME_HASHES, dl.PRIVATE_NAME_HASHES)

    def test_a_hashed_device_name_is_found_in_documents_and_the_tree(self):
        import hashlib
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = saved | {hashlib.sha256(b"qx-9000").hexdigest()}
            e = errors({"README.md": README + "measured on the QX-9000\n",
                        "src/a.rs": "// measured on the qx-9000\n"})
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        self.assertTrue(any(x.startswith("README.md:") and "`QX-9000`" in x for x in e), e)
        self.assertTrue(any(x.startswith("src/a.rs:1:") and "`qx-9000`" in x for x in e), e)

    def test_ordinary_english_is_not_flagged(self):
        # "memory footprint" and a netcode "session" are ordinary words
        self.assertEqual(errors({"README.md": README + "A rollback session keeps memory low.\n"}), [])

    def test_identifiers_inside_code_blocks_are_ignored(self):
        self.assertEqual(errors({"README.md": README + "```rust\nlet worker = 1;\n```\n"}), [])

    def test_reported_line_number_matches_the_file(self):
        text = README + "```\nx\ny\n```\nBacklog here\n"
        e = errors({"README.md": text})
        line = text.split("\n").index("Backlog here") + 1
        self.assertTrue(any(x.startswith(f"README.md:{line}:") for x in e), e)

    def test_missing_document(self):
        e = errors(drop=("README_JP.md",))
        self.assertTrue(any("README_JP.md: missing" in x for x in e), e)


class DevelopmentVocabulary(unittest.TestCase):
    """How the work was organised: in the documents and in every other tracked file."""

    def test_an_english_word_next_to_japanese_is_found(self):
        # `\b` sees no boundary between "worker" and "が"; this is the case that slipped through
        e = errors({"README_JP.md": README + "3 件とも workerが見つけた\n"})
        self.assertTrue(any("agent process `worker`" in x for x in e), e)
        e = errors({"README_JP.md": README + "詳細はBacklogに記録\n"})
        self.assertTrue(any("internal tracker `Backlog`" in x for x in e), e)

    def test_worktree_and_multi_agent_words(self):
        for text, hit in (("each change was made in a worktree", "worktree"),
                          ("a multi-agent setup", "multi-agent"),
                          ("マルチエージェントで並列化", "マルチエージェント"),
                          ("エージェントが測定", "エージェント")):
            e = errors({"README.md": README + text + "\n"})
            self.assertTrue(any(f"agent process `{hit}`" in x for x in e), (text, e))

    def test_a_git_worktree_command_is_not_a_description(self):
        wf = "jobs:\n  x:\n    steps:\n      - run: git worktree add /tmp/base origin/main\n"
        self.assertEqual(errors({".github/workflows/x.yml": wf}), [])

    def test_physics_words_that_share_letters_are_not_flagged(self):
        # a numerical integrator and a CCD agent radius are physics, not process
        src = "// the integrator's order\n// zero agent radius still converges\nlet reference_temp_k = 1;\nproject_pressure_multigrid();\n"
        self.assertEqual(errors({"src/a.rs": src}), [])

    def test_vocabulary_is_checked_in_every_tracked_file(self):
        for rel, text, label in (("src/a.rs", "// see the Backlog\n", "internal tracker"),
                                 ("docs/ROADMAP.md", "worker で実施\n", "agent process"),
                                 ("tests/b.rs", "// (ys-00 判断)\n", "session name"),
                                 ("examples/c.rs", "//! user 裁定: 追加しない\n", "instruction source"),
                                 ("src/d.rs", "/// oracle: `sakamoro-00`'s derivation\n", "session name")):
            e = errors({rel: text})
            self.assertTrue(any(x.startswith(f"{rel}:1: {label}") for x in e), (rel, e))

    def test_private_note_names(self):
        for text in ("// see [[feedback_sample_note_name]]",
                     "//! `feedback_another_sample_note` measured",
                     "# canonical CI template: [[reference_alice_sample_note]]",
                     "/// (`project_alice_sample_note`)",
                     "// `memory/feedback_x.md`"):
            e = errors({"src/a.rs": text + "\n"})
            self.assertTrue(any("internal note" in x or "internal tracker" in x for x in e), (text, e))

    def test_internal_rule_and_template_references(self):
        for text, label in (("/// (see skill §1 経路 5)", "agent process"),
                            ("# 罠 #1 に従う", "internal note"),
                            ("# canonical CI template", "internal note")):
            e = errors({"src/a.rs": text + "\n"})
            self.assertTrue(any(label in x for x in e), (text, e))

    def test_internal_rule_names_are_matched_by_hash(self):
        import hashlib
        saved = dl.PRIVATE_NAME_HASHES
        try:
            dl.PRIVATE_NAME_HASHES = saved | {hashlib.sha256(b"zorb-quux-rules").hexdigest()}
            e = errors({"src/a.rs": "/// (`zorb-quux-rules` §11.2)\n"})
        finally:
            dl.PRIVATE_NAME_HASHES = saved
        self.assertTrue(any("private or internal name" in x for x in e), e)

    def test_the_files_that_define_the_vocabulary_are_exempt(self):
        e = errors({"scripts/test_land.py": "msg = 'found by the worker'\n"})
        self.assertEqual(e, [])

    def test_tree_vocabulary_compared_something(self):
        _, counts = dl.check(tree({"src/a.rs": "fn a() {}\n"}))
        self.assertGreater(counts["tree vocabulary"], 0)


class Changelog(unittest.TestCase):
    def test_duplicate_version(self):
        e = errors({"CHANGELOG.md": CHANGELOG + "\n## [1.3.0] - 2026-09-16\n"})
        self.assertTrue(any("appear twice" in x for x in e), e)

    def test_unreleased_must_be_first(self):
        e = errors({"CHANGELOG.md": CHANGELOG.replace("## [Unreleased]", "## [1.5.0] - 2026-10-04\n\n## [Unreleased]")})
        self.assertTrue(any("not [Unreleased]" in x for x in e), e)

    def test_versions_descending(self):
        e = errors({"CHANGELOG.md": CHANGELOG.replace("## [1.3.0]", "## [1.4.1]")})
        self.assertTrue(any("is not newer" in x for x in e), e)

    def test_prerelease_sorts_below_its_release(self):
        cl = CHANGELOG + "\n## [1.0.0] - x\n\n## [1.0.0-preview.2] - x\n\n## [1.0.0-preview.1] - x\n"
        self.assertEqual(errors({"CHANGELOG.md": cl}), [])

    def test_cargo_version_without_section_or_unreleased(self):
        cl = CHANGELOG.replace("## [Unreleased]\n", "")
        cl = cl.replace("### Added\n\n- `world::step_n`", "## [1.4.5] - x\n\n### Added\n\n- `world::step_n`", 1)
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("no section and there is no [Unreleased]" in x for x in e), e)

    def test_cargo_version_older_than_newest_section(self):
        e = errors({"Cargo.toml": CARGO.replace('1.5.0', '1.2.0')})
        self.assertTrue(any("older than the newest section" in x for x in e), e)

    def test_category_twice_in_unreleased(self):
        cl = CHANGELOG.replace("### Fixed\n", "### Added\n\n- extra\n\n### Fixed\n")
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("`### Added` appears 2 times" in x for x in e), e)

    def test_non_keep_a_changelog_category(self):
        cl = CHANGELOG.replace("### Fixed\n", "### Improvements\n\n- x\n\n### Fixed\n")
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("`### Improvements` is not a Keep a Changelog category" in x for x in e), e)

    def test_emoji_marker_in_unreleased(self):
        cl = CHANGELOG.replace("**Behavior change:**", "⚠️")
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("emoji status marker" in x for x in e), e)

    def test_released_sections_keep_their_old_headings(self):
        # "### Added — old style heading" in [1.4.0] is history, not checked
        self.assertEqual(errors(), [])

    def test_unreleased_without_categories_compares_nothing(self):
        cl = CHANGELOG.split("### Added")[0] + "- an entry under no category\n\n## [1.4.0] - 2026-09-17\n"
        e = errors({"CHANGELOG.md": cl})
        self.assertTrue(any("compared nothing" in x and "categories" in x for x in e), e)

    def test_an_empty_unreleased_right_after_a_release_is_accepted(self):
        cl = CHANGELOG.split("### Added")[0] + "## [1.4.0] - 2026-09-17\n"
        e = errors({"CHANGELOG.md": cl})
        self.assertFalse(any("categories" in x for x in e), e)


SDF_SRC = """//! module doc: `pub trait SdfField` is mentioned here and must not count
use crate::math::Fix128;

/// Trait for evaluating a signed distance field.
pub trait SdfField: Send + Sync {
    /// signed distance
    fn distance(&self, x: f32, y: f32, z: f32) -> f32;

    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32);

    /// default: two calls
    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) {
        (self.distance(x, y, z), self.normal(x, y, z))
    }
}

pub trait Bridge {
    fn send(&mut self, p: &[[Fix128; 3]]);
    fn rot(&mut self, _r: &[[Fix128; 4]]) {
        panic!("not implemented");
    }
    fn joints(&mut self, _j: &[crate::joint::Joint]) {}
}
"""

CONTRACT_DOC = """# Contracts

```rust
pub trait SdfField: Send + Sync {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32;
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32);
    // a comment in the document is not a method
    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) { ... }
}
```

```rust
pub trait Bridge {
    fn send(&mut self, p: &[[Fix128; 3]]);
    fn rot(&mut self, _r: &[[Fix128; 4]]) { ... }
    fn joints(&mut self, _j: &[Joint]) { ... }
}
```
"""

# the SdfField block as the document described it before it was aligned with src/
OLD_SDF_DOC = """```rust
pub trait SdfField: Send + Sync {
    fn sample(&self, point: Vec3Fix) -> Fix128;
    fn sample_batch(&self, points: &[Vec3Fix], out: &mut [Fix128]) {
        // default impl provided
    }
    // ...
}
```
"""


def contract_errors(doc=CONTRACT_DOC, src=SDF_SRC):
    d = tree({"src/sdf.rs": src, "docs/ECOSYSTEM_CONTRACTS.md": doc})
    return dl.check_contracts(d)


class Contracts(unittest.TestCase):
    def test_this_repository_passes(self):
        errs, counts = dl.check_contracts(ROOT)
        self.assertEqual(errs, [])
        self.assertGreaterEqual(counts["contract traits"], 2)
        self.assertGreaterEqual(counts["contract items"], 15)

    def test_matching_document_passes(self):
        errs, counts = contract_errors()
        self.assertEqual(errs, [])
        self.assertEqual(counts, {"contract traits": 2, "contract items": 6})

    def test_old_sample_signature_fails(self):
        doc = CONTRACT_DOC.split("```rust\npub trait Bridge")[0].split("```rust")[0] + OLD_SDF_DOC
        errs, _ = contract_errors(doc)
        self.assertTrue(any("`SdfField::sample` is in the document but not in src/sdf.rs" in x for x in errs), errs)
        self.assertTrue(any("`SdfField::sample_batch` is in the document" in x for x in errs), errs)
        self.assertTrue(any("`SdfField::distance` is in src/sdf.rs but not in the document" in x for x in errs), errs)

    def test_old_document_of_this_repository_fails(self):
        # the old SdfField block against this repository's src/
        d = tree({"docs/ECOSYSTEM_CONTRACTS.md": OLD_SDF_DOC})
        shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"))
        errs, _ = dl.check_contracts(d)
        self.assertTrue(any("`SdfField::sample`" in x for x in errs), errs)

    def test_argument_type_change_fails(self):
        errs, _ = contract_errors(CONTRACT_DOC.replace("fn distance(&self, x: f32, y: f32, z: f32) -> f32;",
                                                       "fn distance(&self, x: f64, y: f64, z: f64) -> f64;"))
        self.assertTrue(any("`SdfField::distance` is `fn distance(&self,x:f64" in x for x in errs), errs)

    def test_type_inside_brackets_is_compared(self):
        # `;` inside `[[Fix128; 4]]` must not end the signature early
        errs, _ = contract_errors(CONTRACT_DOC.replace("[[Fix128; 4]]", "[[Fix128; 3]]"))
        self.assertTrue(any("`Bridge::rot`" in x for x in errs), errs)

    def test_default_body_mismatch_fails(self):
        errs, _ = contract_errors(CONTRACT_DOC.replace(
            "-> (f32, (f32, f32, f32)) { ... }", "-> (f32, (f32, f32, f32));"))
        self.assertTrue(any("`SdfField::distance_and_normal` has no default body in the document but has a default body" in x
                            for x in errs), errs)

    def test_bound_change_fails(self):
        errs, _ = contract_errors(src=SDF_SRC.replace("pub trait SdfField: Send + Sync", "pub trait SdfField: Send"))
        self.assertTrue(any("trait header" in x for x in errs), errs)

    def test_new_src_method_must_be_documented(self):
        src = SDF_SRC.replace("    fn normal(", "    fn gradient(&self) -> f32;\n    fn normal(")
        errs, _ = contract_errors(src=src)
        self.assertTrue(any("`SdfField::gradient` is in src/sdf.rs but not in the document" in x for x in errs), errs)

    def test_trait_missing_from_src_fails(self):
        errs, _ = contract_errors(src=SDF_SRC.replace("pub trait Bridge", "pub trait Other"))
        self.assertTrue(any("`pub trait Bridge` is defined 0 times" in x for x in errs), errs)

    def test_document_without_traits_compares_nothing(self):
        errs, _ = contract_errors("# Contracts\n\nno code\n")
        self.assertTrue(any("check `contract traits` compared nothing" in x for x in errs), errs)

    def test_missing_document_fails(self):
        errs, _ = dl.check_contracts(tree({"src/sdf.rs": SDF_SRC}))
        self.assertTrue(any("ECOSYSTEM_CONTRACTS.md: missing" in x for x in errs), errs)


ASSOC_SRC = """use crate::math::Fix128;

pub unsafe trait Backend: Send {
    type Buffer: AsRef<[u8]> + Iterator<Item = u8>;
    type Id;
    const LANES: usize;
    const NAME: &'static str = "cpu";
    unsafe fn map(&mut self, p: *mut u8) -> usize;
    async fn wait(&self);
    const fn width() -> usize { 4 }
    extern "C" fn hook(x: i32) -> i32;
    fn plain(&self) -> Fix128;
}
"""

ASSOC_DOC = """```rust
pub unsafe trait Backend: Send {
    type Buffer: AsRef<[u8]> + Iterator<Item = u8>;
    type Id;
    const LANES: usize;
    const NAME: &'static str = ...;
    unsafe fn map(&mut self, p: *mut u8) -> usize;
    async fn wait(&self);
    const fn width() -> usize { ... }
    extern "C" fn hook(x: i32) -> i32;
    fn plain(&self) -> Fix128;
}
```
"""


def assoc_errors(doc=ASSOC_DOC, src=ASSOC_SRC):
    return dl.check_contracts(tree({"src/backend.rs": src, "docs/ECOSYSTEM_CONTRACTS.md": doc}))


class ContractItems(unittest.TestCase):
    """Associated types and consts, fn qualifiers and `unsafe trait` are compared."""

    def test_matching_document_passes(self):
        errs, counts = assoc_errors()
        self.assertEqual(errs, [])
        self.assertEqual(counts, {"contract traits": 1, "contract items": 9})

    def test_associated_type_missing_from_the_document_fails(self):
        errs, _ = assoc_errors(ASSOC_DOC.replace("    type Id;\n", ""))
        self.assertIn("docs/ECOSYSTEM_CONTRACTS.md: `Backend::Id` (type) is in src/backend.rs but not in the document",
                      errs)

    def test_associated_type_bound_change_fails(self):
        errs, _ = assoc_errors(src=ASSOC_SRC.replace("type Id;", "type Id: Copy;"))
        self.assertTrue(any("`Backend::Id` (type) is `type Id` in the document but `type Id:Copy`" in x for x in errs), errs)
        errs, _ = assoc_errors(src=ASSOC_SRC.replace("Item = u8", "Item = u16"))
        self.assertTrue(any("`Backend::Buffer` (type)" in x for x in errs), errs)

    def test_associated_const_type_and_default_are_compared(self):
        errs, _ = assoc_errors(src=ASSOC_SRC.replace("const LANES: usize;", "const LANES: u32;"))
        self.assertTrue(any("`Backend::LANES` (const) is `const LANES:usize`" in x for x in errs), errs)
        errs, _ = assoc_errors(src=ASSOC_SRC.replace(' = "cpu";', ";"))
        self.assertTrue(any("`Backend::NAME` (const) has a default in the document but has no default" in x
                            for x in errs), errs)
        errs, _ = assoc_errors(ASSOC_DOC.replace("    const LANES: usize;\n", ""))
        self.assertTrue(any("`Backend::LANES` (const) is in src/backend.rs but not in the document" in x
                            for x in errs), errs)

    def test_fn_qualifiers_are_part_of_the_signature(self):
        for qual, sig in (("unsafe fn map", "fn map"), ("async fn wait", "fn wait"),
                          ("const fn width", "fn width"), ('extern "C" fn hook', "fn hook")):
            errs, _ = assoc_errors(ASSOC_DOC.replace(qual, sig))
            name = sig.split()[-1]
            self.assertTrue(any(f"`Backend::{name}` is `{sig}" in x for x in errs), (qual, errs))
        errs, _ = assoc_errors(ASSOC_DOC.replace('extern "C" fn hook', 'extern "system" fn hook'))
        self.assertTrue(any("`Backend::hook`" in x for x in errs), errs)

    def test_unsafe_trait_in_the_document_is_compared(self):
        errs, _ = assoc_errors(ASSOC_DOC.replace("    fn plain(&self) -> Fix128;\n", ""))
        self.assertTrue(any("`Backend::plain` is in src/backend.rs but not in the document" in x for x in errs), errs)

    def test_unsafe_dropped_from_the_trait_header_fails(self):
        errs, _ = assoc_errors(ASSOC_DOC.replace("pub unsafe trait Backend", "pub trait Backend"))
        self.assertTrue(any("trait header `pub trait Backend:Send` differs" in x for x in errs), errs)
        errs, _ = assoc_errors(src=ASSOC_SRC.replace("pub unsafe trait Backend", "pub trait Backend"))
        self.assertTrue(any("trait header" in x for x in errs), errs)


CHAR_SRC = """pub fn quote() -> char { '"' }
pub fn escaped() -> char { '\\'' }
pub fn longest<'a>(x: &'a str) -> &'a str { x }
pub const URL: &str = r"http://example.invalid/x";
/// the "default" body
pub trait Quoted {
    // fn ghost(&self);
    /* fn ghost2(&self); /* nested */ fn ghost3(&self); */
    fn real<'a>(&'a self) -> &'a str;
}
"""

CHAR_DOC = """```rust
pub trait Quoted {
    fn real<'a>(&'a self) -> &'a str;
}
```
"""


class CommentStripping(unittest.TestCase):
    def test_char_literals_lifetimes_raw_strings_and_nested_comments(self):
        errs, counts = dl.check_contracts(tree({"src/q.rs": CHAR_SRC, "docs/ECOSYSTEM_CONTRACTS.md": CHAR_DOC}))
        self.assertEqual(errs, [])
        self.assertEqual(counts["contract items"], 1)

    def test_strip_keeps_literals_and_drops_comments(self):
        out = dl.strip_rust_comments("let a = '\"'; // gone\nlet b = \"// kept\"; /* gone */ let c = 'x';")
        self.assertEqual(out, "let a = '\"'; \nlet b = \"// kept\";  let c = 'x';")


if __name__ == "__main__":
    unittest.main()
