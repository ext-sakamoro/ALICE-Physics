#!/usr/bin/env python3
"""Tests for scripts/land.py.

Each case builds a throwaway origin (a bare repository) and a clone, makes
commits, and lands them with a Lander whose heavy steps (preflight,
regeneration, checks, CI wait) are recorded instead of run. One rule per case.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import land  # noqa: E402

NAME, EMAIL = "Moroya Sakamoto", "sakamoro@alicelaw.net"
ID = ["-c", f"user.name={NAME}", "-c", f"user.email={EMAIL}"]


def git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", *ID, *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr}")
    return r.stdout.strip()


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def commit(root: Path, files: dict[str, str], msg: str, author: str = f"{NAME} <{EMAIL}>") -> str:
    for rel, text in files.items():
        write(root, rel, text)
    git(root, "add", "-A")
    git(root, "commit", "-q", "--author", author, "-m", msg)
    return git(root, "rev-parse", "HEAD")


class FakeLander(land.Lander):
    """Heavy steps recorded; regeneration writes a ledger derived from src/."""

    def __init__(self, *a, ci_result: str = "success", before_push=None, **k):
        super().__init__(*a, log=lambda *_: None, **k)
        self.calls: list[str] = []
        self.ci_result = ci_result
        self.before_push = before_push

    def run_preflight(self):
        self.preflight_runs += 1
        self.calls.append("preflight")

    def regenerate(self):
        self.calls.append("regenerate")
        src = sorted(p.name for p in (self.root / "src").glob("*.rs"))
        write(self.root, "docs/integration-status.md", "ledger: " + ", ".join(src) + "\n")

    def run_checks(self):
        self.calls.append("checks")
        self.check_changelog()

    def run_compile_check(self):
        self.calls.append("compile")

    def wait_ci(self, ref, sha):
        self.calls.append(f"wait_ci {ref}")
        if self.ci_result != "success":
            raise land.LandError(f"CI on {ref} ended `{self.ci_result}`")

    def fetch(self):
        if self.before_push is not None and "checks" in self.calls:
            hook, self.before_push = self.before_push, None
            hook()
        super().fetch()


class Repo:
    """origin (bare) + `upstream` clone (other sessions) + `work` clone (ours)."""

    def __init__(self, gitattributes: bool = True):
        self.dir = Path(tempfile.mkdtemp())
        self.origin = self.dir / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)], check=True)
        self.upstream = self.dir / "upstream"
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.upstream)], check=True,
                       capture_output=True)
        git(self.upstream, "checkout", "-q", "-b", "main")
        commit(self.upstream, {"src/a.rs": "fn a() {}\n",
                                "scripts/lib-test-exempt.txt": "src/b.rs fixture for the landing steps\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n",
                               "docs/integration-status.md": "ledger: a.rs\n",
                               **({".gitattributes": "CHANGELOG.md merge=union\n"} if gitattributes else {})},
               "init")
        git(self.upstream, "push", "-q", "origin", "main")
        self.work = self.dir / "work"
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.work)], check=True, capture_output=True)

    def lander(self, **k) -> FakeLander:
        return FakeLander(self.work, "T-1", **k)

    def origin_head(self) -> str:
        return git(self.upstream, "ls-remote", "origin", "main").split()[0]

    def push_upstream(self, files: dict[str, str], msg: str) -> None:
        git(self.upstream, "pull", "-q", "--rebase", "origin", "main")
        commit(self.upstream, files, msg)
        git(self.upstream, "push", "-q", "origin", "main")

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class Pure(unittest.TestCase):
    def test_lane_by_path(self):
        self.assertEqual(land.lane_of(["src/a.rs", "tests/x.rs"]), "direct")
        for p in (".github/workflows/ci.yml", "scripts/x.py", "Cargo.toml", "Cargo.lock", "bindings/A.cs"):
            self.assertEqual(land.lane_of(["src/a.rs", p]), "ci", p)
        self.assertEqual(land.lane_of(["docs/Cargo.toml.md"]), "direct")

    def test_docs_only(self):
        self.assertTrue(land.docs_only(["docs/oracle-status.md", "CHANGELOG.md", "README_JP.md"]))
        self.assertFalse(land.docs_only(["docs/x.md", "src/a.rs"]))
        self.assertFalse(land.docs_only([]))

    def test_os_specific_code_goes_to_the_ci_lane(self):
        self.assertEqual(land.lane_of(["src/a.rs"], "#[cfg(windows)]\nfn f() {}"), "ci")
        self.assertEqual(land.lane_of(["src/a.rs"], 'let v = std::env::var("X");'), "ci")
        self.assertEqual(land.lane_of(["src/a.rs"], "fn plain() {}"), "direct")
        for p in ("fuzz/Cargo.toml", "deny.toml", "rust-toolchain.toml", "include/a.h", "unreal-plugin/x.cpp",
                  ".gitattributes", ".cargo/config.toml"):
            self.assertEqual(land.lane_of([p]), "ci", p)

    def test_changelog_duplicates(self):
        ok = "# C\n\n## [Unreleased]\n\n### Added\n- a\n- b\n\n## [1.0.0]\n\n### Added\n- a\n"
        self.assertEqual(land.changelog_problems(ok), [])
        self.assertTrue(any("line appears 2" in p for p in land.changelog_problems(ok.replace("- b", "- a"))))
        dup_head = ok.replace("- b\n", "- b\n\n### Added\n- c\n")
        self.assertTrue(any("heading appears 2" in p for p in land.changelog_problems(dup_head)))
        self.assertTrue(land.changelog_problems("# C\n\n## [1.0.0]\n- a\n"))

    def test_blank_integration_empties_only_that_column(self):
        doc = "| Module | Summary | Integration |\n|---|---|---|\n| `a` | sum | step |\n\ntext | x |\n"
        out = land.blank_integration(doc)
        self.assertIn("| `a` | sum | |", out.replace("  ", " "))
        self.assertIn("text | x |", out)
        self.assertIn("| Module | Summary | Integration |", out)

    def test_message_vocabulary(self):
        self.assertEqual(land.message_problems("fix(tgs): keys follow body identity\n\nbody text"), [])
        self.assertTrue(land.message_problems("fix: found by the worker"))


class Landing(unittest.TestCase):
    def setUp(self):
        self.r = Repo()

    def tearDown(self):
        self.r.close()

    def test_a_new_source_file_without_lib_tests_stops_the_land(self):
        commit(self.r.work, {"src/c.rs": "pub fn c() {}\n",
                             "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- c\n"}, "feat: c")
        lander = self.r.lander()
        with self.assertRaises(land.LandError) as cm:
            lander.land()
        self.assertIn("lib-test gate", str(cm.exception))
        self.assertIn("src/c.rs: new source file without lib tests", str(cm.exception))

    def test_direct_lane_lands_and_regenerates(self):
        commit(self.r.work, {"src/b.rs": "fn b() {}\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander()
        sha = lander.land()
        self.assertEqual(self.r.origin_head(), sha)
        self.assertNotIn("wait_ci ci/T-1", lander.calls)
        self.assertEqual((self.r.work / "docs/integration-status.md").read_text(), "ledger: a.rs, b.rs\n")
        self.assertIn("docs: regenerate the generated ledgers", git(self.r.work, "log", "-1", "--format=%s"))

    def test_ci_lane_waits_for_the_branch_ci(self):
        commit(self.r.work, {"scripts/x.py": "print(1)\n"}, "ci: add x")
        lander = self.r.lander()
        lander.land()
        self.assertIn("wait_ci ci/T-1", lander.calls)

    def test_red_branch_ci_stops_before_main(self):
        before = self.r.origin_head()
        commit(self.r.work, {"scripts/x.py": "print(1)\n"}, "ci: add x")
        with self.assertRaises(land.LandError):
            self.r.lander(ci_result="failure").land()
        self.assertEqual(self.r.origin_head(), before)

    def test_wrong_author_is_refused(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- b\n"}, "feat: b",
               author="Someone <someone@host.local>")
        with self.assertRaisesRegex(land.LandError, "author"):
            self.r.lander().land()

    def test_internal_vocabulary_in_a_message_is_refused(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- b\n"}, "feat: b (found by a worker)")
        with self.assertRaisesRegex(land.LandError, "worker"):
            self.r.lander().land()

    def test_src_change_needs_a_changelog_line(self):
        commit(self.r.work, {"src/b.rs": "x\n"}, "feat: b")
        with self.assertRaisesRegex(land.LandError, "CHANGELOG"):
            self.r.lander().land()
        self.r.lander(no_changelog=True).land()

    def test_nothing_to_land_fails(self):
        with self.assertRaisesRegex(land.LandError, "nothing to land"):
            self.r.lander().land()

    def test_uncommitted_changes_fail(self):
        commit(self.r.work, {"docs/x.md": "x\n"}, "docs: x")
        write(self.r.work, "docs/y.md", "dirty\n")
        with self.assertRaisesRegex(land.LandError, "uncommitted"):
            self.r.lander().land()

    def test_ledger_conflict_takes_main_and_regenerates(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n",
                             "docs/integration-status.md": "ledger: mine\n"}, "feat: b")
        self.r.push_upstream({"docs/integration-status.md": "ledger: bot\n"}, "docs: update integration status")
        sha = self.r.lander().land()
        self.assertEqual(self.r.origin_head(), sha)
        self.assertEqual((self.r.work / "docs/integration-status.md").read_text(), "ledger: a.rs, b.rs\n")

    def test_public_api_snapshot_conflict_is_regenerated(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n",
                             "docs/PUBLIC_API_SNAPSHOT.txt": "pub fn b\n"}, "feat: b")
        self.r.push_upstream({"docs/PUBLIC_API_SNAPSHOT.txt": "pub fn c\n"}, "feat: c api")
        sha = self.r.lander().land()
        self.assertEqual(self.r.origin_head(), sha)

    def test_changelog_lines_from_both_sides_merge(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- mine\n"}, "feat: b")
        self.r.push_upstream({"CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- theirs\n"}, "docs(changelog): theirs")
        self.r.lander().land()
        text = (self.r.work / "CHANGELOG.md").read_text()
        self.assertIn("- mine", text)
        self.assertIn("- theirs", text)

    def test_changelog_conflict_merges_without_gitattributes(self):
        self.r.close()
        self.r = Repo(gitattributes=False)
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- mine\n"},
               "feat: b")
        self.r.push_upstream({"CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- theirs\n"},
                             "docs(changelog): theirs")
        sha = self.r.lander().land()
        self.assertEqual(self.r.origin_head(), sha)
        text = (self.r.work / "CHANGELOG.md").read_text()
        self.assertIn("- mine", text)
        self.assertIn("- theirs", text)
        self.assertNotIn("<<<<<<<", text)

    def test_source_conflict_stops_and_leaves_no_rebase(self):
        commit(self.r.work, {"src/a.rs": "fn a() { mine }\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- a\n"}, "fix: a")
        self.r.push_upstream({"src/a.rs": "fn a() { theirs }\n"}, "fix: a upstream")
        upstream_head = self.r.origin_head()
        mine = git(self.r.work, "rev-parse", "HEAD")
        with self.assertRaisesRegex(land.LandError, "outside the generated ledgers"):
            self.r.lander().land()
        self.assertEqual(self.r.origin_head(), upstream_head, "nothing of ours reached main")
        self.assertFalse((self.r.work / ".git" / "rebase-merge").exists(), "the rebase was aborted")
        self.assertEqual(git(self.r.work, "rev-parse", "HEAD"), mine, "our commit is left as it was")

    def test_the_ledgers_are_regenerated_before_the_first_preflight(self):
        # preflight checks the ledgers against a freshly built index; a branch
        # whose ledgers were written from an older index must not fail there
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander()
        lander.land()
        self.assertEqual(lander.calls[:2], ["regenerate", "preflight"], lander.calls)
        landed = git(self.r.work, "show", "HEAD:docs/integration-status.md")
        self.assertIn("b.rs", landed)

    def test_docs_only_upstream_does_not_rerun_preflight(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        self.r.push_upstream({"docs/oracle-status.md": "regenerated\n"}, "docs: oracle-status")
        lander = self.r.lander()
        lander.land()
        self.assertEqual(lander.preflight_runs, 1)

    def test_code_upstream_in_other_files_compiles_but_does_not_rerun_preflight(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        self.r.push_upstream({"src/c.rs": "fn c() {}\n"}, "feat: c")
        lander = self.r.lander()
        lander.land()
        self.assertEqual(lander.preflight_runs, 1)
        self.assertIn("compile", lander.calls)

    def test_upstream_touching_our_file_reruns_preflight(self):
        write(self.r.work, "src/b.rs", "fn b1() {}\n\n\n\n\nfn b2() {}\n")
        git(self.r.work, "add", "-A")
        git(self.r.work, "commit", "-q", "--author", f"{NAME} <{EMAIL}>", "-m", "feat: b base")
        git(self.r.work, "push", "-q", "origin", "HEAD:main")
        commit(self.r.work, {"src/b.rs": "fn b1() { mine }\n\n\n\n\nfn b2() {}\n",
                             "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b1")
        self.r.push_upstream({"src/b.rs": "fn b1() {}\n\n\n\n\nfn b2() { theirs }\n"}, "feat: b2")
        lander = self.r.lander()
        lander.land()
        self.assertEqual(lander.preflight_runs, 2)
        self.assertNotIn("compile", lander.calls)

    def test_upstream_changing_the_build_reruns_preflight(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        self.r.push_upstream({"Cargo.toml": "[package]\nname = \"x\"\n"}, "build: deps")
        lander = self.r.lander()
        lander.land()
        self.assertEqual(lander.preflight_runs, 2)

    def test_overlaps(self):
        self.assertFalse(land.overlaps(["src/a.rs", "CHANGELOG.md"], ["src/c.rs", "CHANGELOG.md"]))
        self.assertTrue(land.overlaps(["src/a.rs"], ["src/a.rs"]))
        self.assertTrue(land.overlaps(["src/a.rs"], ["Cargo.lock"]))
        self.assertTrue(land.overlaps(["src/a.rs"], ["src/lib.rs"]))
        self.assertFalse(land.overlaps(["docs/x.md"], ["docs/x.md"]))
        for t in ["fuzz/Cargo.toml", "deny.toml", "scripts/affected_tests.py", "src/m/mod.rs",
                  "include/alice.h", ".github/workflows/ci.yml", "examples/a.rs", "benches/b.rs",
                  "fuzz/fuzz_targets/f.rs", "bindings/c/alice.h", "unreal-plugin/x.cpp", "cbindgen.toml"]:
            self.assertTrue(land.overlaps(["src/a.rs"], [t]), t)

    def test_main_moving_before_the_push_is_retried(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        hook = lambda: self.r.push_upstream({"docs/wiring-status.md": "bot\n"}, "docs: wiring status")  # noqa: E731
        lander = self.r.lander(before_push=hook)
        sha = lander.land()
        self.assertEqual(self.r.origin_head(), sha)
        self.assertGreaterEqual(lander.calls.count("checks"), 2)

    def test_landed_commits_carry_the_fixed_committer(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        git(self.r.work, "-c", "user.email=me@machine.local", "commit", "-q", "--amend", "--no-edit")
        # main has not moved: a plain rebase would keep the commit (and its committer) as is
        self.r.lander().land()
        committers = git(self.r.work, "log", "-3", "--format=%ce").splitlines()
        self.assertTrue(all(c == EMAIL for c in committers), committers)

    def test_signature_in_a_message_is_refused(self):
        commit(self.r.work, {"docs/x.md": "x\n"}, "docs: x\n\nCo-Authored-By: Someone <a@b.c>")
        with self.assertRaisesRegex(land.LandError, "signature"):
            self.r.lander().land()

    def test_a_rewritten_changelog_line_does_not_come_back(self):
        # the base still carries `merge=union` (Repo default): the old wording must not survive
        cl = "# Changelog\n\n## [Unreleased]\n\n- start\n- old wording\n"
        self.r.push_upstream({"CHANGELOG.md": cl}, "docs: changelog")
        git(self.r.work, "pull", "-q", "--rebase", "origin", "main")
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": cl.replace("old wording", "new wording")}, "fix: b")
        self.r.push_upstream({"CHANGELOG.md": cl + "- theirs\n"}, "fix: elsewhere")
        sha = self.r.lander().land()
        self.assertEqual(self.r.origin_head(), sha)
        text = (self.r.work / "CHANGELOG.md").read_text()
        self.assertIn("- new wording", text)
        self.assertIn("- theirs", text)
        self.assertNotIn("old wording", text)
        self.assertNotIn("<<<<<<<", text)

    def test_both_sides_rewriting_the_same_line_stops(self):
        cl = "# Changelog\n\n## [Unreleased]\n\n- start\n- old wording\n"
        self.r.push_upstream({"CHANGELOG.md": cl}, "docs: changelog")
        git(self.r.work, "pull", "-q", "--rebase", "origin", "main")
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": cl.replace("old wording", "mine")}, "fix: b")
        self.r.push_upstream({"CHANGELOG.md": cl.replace("old wording", "theirs")}, "fix: elsewhere")
        upstream_head = self.r.origin_head()
        with self.assertRaisesRegex(land.LandError, "both sides changed the same lines"):
            self.r.lander().land()
        self.assertEqual(self.r.origin_head(), upstream_head)
        self.assertFalse((self.r.work / ".git" / "rebase-merge").exists(), "the rebase was aborted")

    def test_merge_conflict_hunks(self):
        add_add = "a\n<<<<<<< o\nx\n||||||| b\n=======\ny\n>>>>>>> t\nz\n"
        self.assertEqual(land.merge_conflict_hunks(add_add), "a\nx\ny\nz\n")
        rewrite_vs_add = "<<<<<<< o\nnew\n||||||| b\nold\n=======\nold\nmore\n>>>>>>> t\n"
        self.assertEqual(land.merge_conflict_hunks(rewrite_vs_add), "new\nmore\n")
        add_vs_rewrite = "<<<<<<< o\nold\nmore\n||||||| b\nold\n=======\nnew\n>>>>>>> t\n"
        self.assertEqual(land.merge_conflict_hunks(add_vs_rewrite), "more\nnew\n")
        with self.assertRaises(land.LandError):
            land.merge_conflict_hunks("<<<<<<< o\nm\n||||||| b\nold\n=======\nt\n>>>>>>> t\n")

    def test_a_ci_ref_that_already_holds_head_is_not_pushed_again(self):
        commit(self.r.work, {"scripts/x.py": "print(1)\n"}, "ci: x")
        git(self.r.work, "push", "-q", "origin", "HEAD:refs/heads/ci/T-1")
        lander = self.r.lander()
        pushes = []
        real_git = lander.git

        def spy(*a, check=True):
            if a and a[0] == "push" and any(x.startswith("HEAD:refs/heads/ci/") for x in a):
                pushes.append(a)
            return real_git(*a, check=check)
        lander.git = spy
        lander.land()
        self.assertEqual(pushes, [], "the earlier ci/T-1 push is reused, not repeated")
        self.assertIn("wait_ci ci/T-1", lander.calls)

    def test_union_merged_changelog_with_a_repeated_line_fails(self):
        cl = "# Changelog\n\n## [Unreleased]\n\n### Fixed\n- start\n"
        self.r.push_upstream({"CHANGELOG.md": cl}, "docs: changelog layout")
        git(self.r.work, "pull", "-q", "--rebase", "origin", "main")
        # the same line added next to different neighbours: the union driver keeps both copies
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": cl + "- mine\n- same fix\n"}, "fix: b")
        self.r.push_upstream({"CHANGELOG.md": cl + "- same fix\n- theirs\n"}, "fix: elsewhere")
        with self.assertRaisesRegex(land.LandError, "appears 2 times"):
            self.r.lander().land()

    def test_modules_conflict_keeps_both_hand_written_changes(self):
        head = "| Module | Summary | Integration |\n|---|---|---|\n"
        base = head + "".join(f"| `{m}` | about {m} | step |\n" for m in "acdefg")
        self.r.push_upstream({"docs/MODULES.md": base}, "docs: modules")
        git(self.r.work, "pull", "-q", "--rebase", "origin", "main")
        # ours: a new row after `a` / main: `a` relabelled (generated column) and `g` reworded
        commit(self.r.work, {"docs/MODULES.md": base.replace("| `c`", "| `b` | second | binding |\n| `c`")},
               "docs: add b")
        self.r.push_upstream({"docs/MODULES.md": base.replace("| about a | step", "| about a | world API")
                              .replace("| about g |", "| third, revised |")}, "docs: regenerate")
        self.r.lander().land()
        text = (self.r.work / "docs/MODULES.md").read_text()
        self.assertIn("`b` | second", text)
        self.assertIn("third, revised", text)

    def test_modules_conflict_in_a_hand_written_cell_stops(self):
        head = "| Module | Summary | Integration |\n|---|---|---|\n"
        base = head + "| `a` | first | step |\n"
        self.r.push_upstream({"docs/MODULES.md": base}, "docs: modules")
        git(self.r.work, "pull", "-q", "--rebase", "origin", "main")
        commit(self.r.work, {"docs/MODULES.md": base.replace("first", "mine")}, "docs: a mine")
        self.r.push_upstream({"docs/MODULES.md": base.replace("first", "theirs")}, "docs: a theirs")
        with self.assertRaisesRegex(land.LandError, "hand-written parts conflict"):
            self.r.lander().land()

    def test_ci_ref_owned_by_someone_else_is_not_overwritten(self):
        git(self.r.upstream, "pull", "-q", "--rebase", "origin", "main")
        commit(self.r.upstream, {"scripts/other.py": "x\n"}, "ci: other")
        git(self.r.upstream, "push", "-q", "origin", "HEAD:refs/heads/ci/T-1")
        commit(self.r.work, {"scripts/x.py": "print(1)\n"}, "ci: x")
        with self.assertRaisesRegex(land.LandError, "holds other commits"):
            self.r.lander().land()
        self.r.lander(replace_ci=True).land()  # explicit: an earlier attempt of ours

    def test_ci_ref_from_our_earlier_attempt_is_reused(self):
        commit(self.r.work, {"scripts/x.py": "print(1)\n"}, "ci: x")
        git(self.r.work, "push", "-q", "origin", "HEAD:refs/heads/ci/T-1")
        # the same patch with a new hash (an earlier attempt was rebased): not an ancestor, still ours
        git(self.r.work, "commit", "-q", "--amend", "--no-edit", "--date", "2001-01-01T00:00:00")
        self.r.lander().land()

    def test_dry_run_pushes_nothing(self):
        before = self.r.origin_head()
        commit(self.r.work, {"scripts/x.py": "print(1)\n"}, "ci: x")
        lander = self.r.lander(dry_run=True)
        lander.land()
        self.assertEqual(self.r.origin_head(), before)
        self.assertNotIn("wait_ci ci/T-1", lander.calls)


class WaitCi(unittest.TestCase):
    """wait_ci against a scripted run list (no network)."""

    def lander(self, answers, at="abc"):
        class L(land.Lander):
            def git(self, *a, check=True):
                return "git@github.com:o/r.git"

            def list_runs(self, slug, ref):
                return answers.pop(0) if answers else []

            def remote_sha(self, ref):
                return at
        lnd = L(Path("."), "T", log=lambda *_: None)
        lnd.poll_seconds = 0
        return lnd

    def test_success(self):
        self.lander([[], [{"headSha": "abc", "databaseId": 1, "status": "completed", "conclusion": "success"}]]
                    ).wait_ci("main", "abc")

    def test_cancelled_is_not_green(self):
        with self.assertRaisesRegex(land.LandError, "cancelled .*unverified"):
            self.lander([[{"headSha": "abc", "databaseId": 1, "status": "completed", "conclusion": "cancelled"}]]
                        ).wait_ci("main", "abc")

    def test_no_run_at_all_times_out_instead_of_looping(self):
        old = land.NO_RUN_TIMEOUT
        land.NO_RUN_TIMEOUT = -1
        try:
            with self.assertRaisesRegex(land.LandError, "no ci.yml run"):
                self.lander([]).wait_ci("main", "abc")
        finally:
            land.NO_RUN_TIMEOUT = old

    def test_a_push_that_did_not_reach_the_ref_fails_at_once(self):
        # with no run and the ref elsewhere there is nothing to wait for: not 30 minutes of polling
        with self.assertRaisesRegex(land.LandError, "points at def1234, not abc"):
            self.lander([[]] * 3, at="def1234567").wait_ci("ci/x", "abc")

    def test_a_ref_that_holds_the_sha_keeps_waiting_for_a_late_run(self):
        late = {"headSha": "abc", "databaseId": 7, "status": "completed", "conclusion": "success"}
        self.lander([[], [], [], [late]]).wait_ci("ci/x", "abc")

    def test_the_timeout_shows_what_gh_listed(self):
        other = {"headSha": "fff0000", "databaseId": 9, "status": "completed", "conclusion": "success",
                 "createdAt": "2026-10-05T12:19:24Z", "event": "push"}
        old = land.NO_RUN_TIMEOUT
        land.NO_RUN_TIMEOUT = -1
        try:
            with self.assertRaisesRegex(land.LandError, r"fff0000 push completed/success created 2026-10-05T12:19:24Z"):
                self.lander([[other]]).wait_ci("ci/x", "abc")
        finally:
            land.NO_RUN_TIMEOUT = old


if __name__ == "__main__":
    unittest.main()
