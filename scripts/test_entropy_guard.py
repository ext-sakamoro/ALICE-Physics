#!/usr/bin/env python3
"""Tests for scripts/entropy_guard.py: the repository passes, each entropy or
clock source in production code is reported, test modules are exempt, and the
allowed counts are exact."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entropy_guard as eg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="entropy-guard-"))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


class EntropyGuard(unittest.TestCase):
    def setUp(self):
        self._allowed = dict(eg.ALLOWED)
        eg.ALLOWED.clear()

    def tearDown(self):
        eg.ALLOWED.clear()
        eg.ALLOWED.update(self._allowed)

    def problems(self, files):
        return eg.check(tree(files))[0]

    def test_both_clippy_configs_ban_elapsed(self):
        # `.elapsed()` reads the wall clock exactly like `::now()`; a lexical
        # pattern for it here would be too broad (any value named `elapsed`),
        # so clippy's type-resolved disallowed-methods carries the ban. This
        # canary confirms the entries are present in both configurations (the
        # all-targets one and the library one), not absent by a typo.
        compared = 0
        for rel in ("clippy.toml", ".github/clippy-lib/clippy.toml"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for path in ("std::time::Instant::elapsed", "std::time::SystemTime::elapsed"):
                self.assertIn(f'path = "{path}"', text, f"{rel} must disallow {path}")
                compared += 1
        self.assertEqual(compared, 4)

    def test_the_repository_passes(self):
        eg.ALLOWED.update(self._allowed)
        problems, files, patterns = eg.check(ROOT)
        self.assertEqual(problems, [])
        self.assertGreater(files, 100)
        self.assertEqual(patterns, len(eg.PATTERNS))

    def test_every_source_in_production_code_is_reported(self):
        for call in ["getrandom::getrandom(&mut b)", "rand::rngs::OsRng", "rand::thread_rng()",
                     "rand::random::<u64>()", "SecureRng::from_entropy()", "DpNoise::try_from_entropy(1.0, 1.0)",
                     "std::time::SystemTime::now()", "std::time::Instant::now()"]:
            p = self.problems({"src/a.rs": f"pub fn f() {{ let _ = {call}; }}\n"})
            self.assertEqual(len(p), 1, (call, p))

    def test_a_test_module_is_exempt(self):
        src = "pub fn f() {}\n\n#[cfg(test)]\nmod tests {\n    fn t() {\n        let _ = std::time::Instant::now();\n    }\n}\n"
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_code_after_a_test_module_is_checked_again(self):
        src = "#[cfg(test)]\nmod tests {\n    fn t() {}\n}\n\npub fn f() { let _ = rand::thread_rng(); }\n"
        p = self.problems({"src/a.rs": src})
        self.assertEqual(len(p), 1, p)
        self.assertIn("src/a.rs:6", p[0])

    def test_a_definition_strings_and_comments_are_not_calls(self):
        src = ('pub fn from_entropy() -> u64 { 0 }\n'
               'pub const S: &str = "thread_rng()";\n'
               '// SystemTime::now() is not used here\n')
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_an_allowed_site_needs_its_exact_count(self):
        src = "pub fn f() { let _ = std::time::SystemTime::now(); }\n"
        eg.ALLOWED[("src/a.rs", "SystemTime::now")] = (1, "reason")
        self.assertEqual(self.problems({"src/a.rs": src}), [])
        eg.ALLOWED[("src/a.rs", "SystemTime::now")] = (2, "reason")
        self.assertTrue(any("1 call sites, ALLOWED says 2" in x for x in self.problems({"src/a.rs": src})))

    def test_an_allowed_entry_needs_a_reason(self):
        src = "pub fn f() { let _ = std::time::SystemTime::now(); }\n"
        eg.ALLOWED[("src/a.rs", "SystemTime::now")] = (1, "  ")
        self.assertTrue(any("has no reason" in x for x in self.problems({"src/a.rs": src})))

    def test_a_stale_allowed_entry_fails(self):
        eg.ALLOWED[("src/a.rs", "OsRng")] = (1, "reason")
        p = self.problems({"src/a.rs": "pub fn f() {}\n"})
        self.assertTrue(any("matches nothing" in x for x in p), p)

    def test_an_aliased_clock_is_found(self):
        src = "use std::time::Instant as Clock;\npub fn f() { let _ = Clock::now(); }\n"
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x and "src/a.rs:2" in x for x in p), p)

    def test_an_aliased_map_is_found(self):
        src = "use std::collections::HashMap as Map;\npub fn f() -> Map<u8, u8> { Map::new() }\n"
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any(x.startswith("HashMap") for x in p), p)

    def test_a_call_split_over_lines_is_found(self):
        src = "pub fn f() {\n    let _ = std::time::Instant\n        ::now();\n}\n"
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x and "src/a.rs:2" in x for x in p), p)

    def test_random_state_new_is_found(self):
        src = "pub fn f() { let _ = std::collections::hash_map::RandomState::new(); }\n"
        self.assertTrue(any(x.startswith("RandomState") for x in self.problems({"src/a.rs": src})))

    def test_a_default_hasher_map_is_found(self):
        for src in ["pub fn f() { let _: std::collections::HashMap<u8, u8> = std::collections::HashMap::new(); }\n",
                    "pub fn f() { let _ = std::collections::HashSet::<u8>::new(); }\n"]:
            self.assertEqual(len(self.problems({"src/a.rs": src})), 1, src)

    def test_the_seedless_map_alias_is_not_a_finding(self):
        src = "pub fn f() { let _: DetHashMap<u8, u8> = DetHashMap::default(); }\n"
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_unix_epoch_elapsed_is_found(self):
        src = "pub fn f() { let _ = std::time::UNIX_EPOCH.elapsed(); }\n"
        self.assertTrue(any(x.startswith("UNIX_EPOCH") for x in self.problems({"src/a.rs": src})))

    def test_attributes_between_cfg_test_and_the_module_keep_it_exempt(self):
        src = ("pub fn f() {}\n\n#[cfg(test)]\n// timing only\n#[allow(clippy::disallowed_methods)]\n"
               "mod tests {\n    fn t() { let _ = std::time::Instant::now(); }\n}\n")
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    # The cases below follow ALICE-DB's guard fixes (ALICE-DB 39822ee, b08a967):
    # each is a source the guard missed before, with a production call after
    # or inside a construct the line-based scan misread.

    def test_random_state_default_is_found(self):
        # `RandomState::default()` reaches the same OS-seeded hasher as
        # `::new()`, and clippy cannot ban a trait method by the type alone
        src = "pub fn f() { let _ = std::collections::hash_map::RandomState::default(); }\n"
        self.assertTrue(any(x.startswith("RandomState") for x in self.problems({"src/a.rs": src})))

    def test_a_qualified_self_clock_call_is_found(self):
        for call in ["<std::time::Instant>::now()", "<std::time::Instant as SomeTrait>::now()",
                     "<std::time::SystemTime>::now()"]:
            p = self.problems({"src/a.rs": f"pub fn f() {{ let _ = {call}; }}\n"})
            self.assertEqual(len(p), 1, (call, p))

    def test_a_qualified_self_aliased_clock_is_found(self):
        src = "use std::time::Instant as Clock;\npub fn f() { let _ = <Clock>::now(); }\n"
        self.assertTrue(any("Instant::now" in x for x in self.problems({"src/a.rs": src})))

    def test_uuid_v4_is_found(self):
        for call in ["uuid::Uuid::new_v4()", "<uuid::Uuid>::new_v4()"]:
            p = self.problems({"src/a.rs": f"pub fn f() {{ let _ = {call}; }}\n"})
            self.assertEqual(len(p), 1, (call, p))

    def _after_test_module(self, inner: str):
        """A test module holding `inner`, then a production clock call."""
        return ("#[cfg(test)]\nmod t {\n" + inner + "\n    fn t() {}\n}\n\n"
                "pub fn f() { let _ = std::time::Instant::now(); }\n")

    def test_a_char_literal_brace_does_not_break_the_test_module_skip(self):
        for inner in ["    const C: char = '{';", "    const B: u8 = b'{';", "    const E: char = '\\'';"]:
            p = self.problems({"src/a.rs": self._after_test_module(inner)})
            self.assertTrue(any("Instant::now" in x for x in p), (inner, p))

    def test_a_raw_string_brace_does_not_break_the_test_module_skip(self):
        # one embedded plain quote before the brace: pairing quotes leaves `hi{`
        for inner in ['    const S: &str = r#"say "hi{"#;', '    const S: &[u8] = br"{";']:
            p = self.problems({"src/a.rs": self._after_test_module(inner)})
            self.assertTrue(any("Instant::now" in x for x in p), (inner, p))

    def test_a_block_comment_brace_does_not_break_the_test_module_skip(self):
        for inner in ["    /* { */", "    /* outer /* { */ still a comment */", "    /*\n    {\n    */"]:
            p = self.problems({"src/a.rs": self._after_test_module(inner)})
            self.assertTrue(any("Instant::now" in x for x in p), (inner, p))

    def test_a_lifetime_is_not_a_char_literal(self):
        # `'a` has no closing quote: it must stay code, so the braces count
        src = ("#[cfg(test)]\nmod t {\n    fn t<'a>(x: &'a str) -> &'a str { x }\n}\n\n"
               "pub fn f() { let _ = std::time::Instant::now(); }\n")
        self.assertTrue(any("Instant::now" in x for x in self.problems({"src/a.rs": src})))

    def test_a_block_comment_and_strings_are_not_calls(self):
        src = ('pub fn f() {\n    /* std::time::Instant::now() */\n'
               '    let _ = r#"std::time::Instant::now()"#;\n    let _ = \'"\';\n}\n')
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_a_cfg_all_test_module_is_exempt_and_code_after_it_is_checked(self):
        src = ('#[cfg(all(test, feature = "std"))]\nmod t {\n    fn t() { let _ = std::time::Instant::now(); }\n}\n\n'
               "pub fn f() { let _ = rand::thread_rng(); }\n")
        p = self.problems({"src/a.rs": src})
        self.assertEqual(len(p), 1, p)
        self.assertIn("thread_rng", p[0])
        self.assertIn("src/a.rs:6", p[0])

    def test_main_exits_1_when_it_scans_no_file(self):
        root = tree({"README.md": "x\n"})
        saved = eg.ROOT
        try:
            eg.ROOT = root
            with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
                code = eg.main()
        finally:
            eg.ROOT = saved
        self.assertEqual(code, 1)
        self.assertIn("compared nothing", err.getvalue())

    def test_no_source_file_compares_nothing(self):
        _, files, _ = eg.check(tree({"README.md": "x\n"}))
        self.assertEqual(files, 0)


if __name__ == "__main__":
    unittest.main()
