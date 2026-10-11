#!/usr/bin/env python3
"""Tests for scripts/land.py.

Each case builds a throwaway origin (a bare repository) and a clone, makes
commits, and lands them with a Lander whose heavy steps (preflight,
regeneration, checks, CI wait) are recorded instead of run. One rule per case.
"""

from __future__ import annotations

import json
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

    def wait_evidence(self, ref, sha):
        self.calls.append(f"wait_evidence {ref}")
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
        self.assertNotIn("wait_evidence ci/T-1", lander.calls)
        self.assertEqual((self.r.work / "docs/integration-status.md").read_text(), "ledger: a.rs, b.rs\n")
        self.assertIn("docs: regenerate the generated ledgers", git(self.r.work, "log", "-1", "--format=%s"))

    def test_ci_lane_waits_for_the_branch_ci(self):
        commit(self.r.work, {"scripts/x.py": "print(1)\n"}, "ci: add x")
        lander = self.r.lander()
        lander.land()
        self.assertIn("wait_evidence ci/T-1", lander.calls)

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
        self.assertIn("wait_evidence ci/T-1", lander.calls)

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
        self.assertNotIn("wait_evidence ci/T-1", lander.calls)


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


FAKE_GH = r"""import json, os, sys
path = os.environ["FAKE_GH_STATE"]
st = json.load(open(path, encoding="utf-8"))
a = sys.argv[1:]
st["calls"].append(a)
rc = 0
if a[:2] == ["run", "list"]:
    i = min(st["polls"], len(st["list"]) - 1)
    st["polls"] += 1
    print(json.dumps(st["list"][i] + st["added"], ensure_ascii=True))
elif a[:2] == ["run", "view"]:
    print(json.dumps({"jobs": st["jobs"].get(a[2], [])}, ensure_ascii=True))
elif a[:2] == ["workflow", "run"]:
    st["dispatched"].append(a[2])
    st["added"] += st["on_dispatch"].get(a[2], [])
else:
    print("unexpected gh " + " ".join(a), file=sys.stderr)
    rc = 2
json.dump(st, open(path, "w", encoding="utf-8"))
sys.exit(rc)
"""

SHA = "a" * 40
ROOT = Path(__file__).resolve().parent.parent


def ok_run(name: str, rid: int, **k) -> dict:
    return {"workflowName": name, "headSha": SHA, "databaseId": rid, "status": "completed",
            "conclusion": "success", "event": "push", "createdAt": "2026-10-12T00:00:00Z", **k}


class WaitEvidence(unittest.TestCase):
    """wait_evidence against a fake `gh` on PATH (no network)."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.gh = self.dir / "gh.py"
        self.gh.write_text(FAKE_GH, encoding="utf-8")
        self.state = self.dir / "state.json"
        self.env = {k: os.environ.get(k) for k in ("FAKE_GH_STATE",)}
        os.environ["FAKE_GH_STATE"] = str(self.state)
        self.expected = land.expected_workflows(ROOT)
        self.names = {w["file"]: w["name"] for w in self.expected}

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.dir, ignore_errors=True)

    def green(self) -> list[dict]:
        """every expected workflow green; ci.yml and the after-landing workflow still running
        with their after-landing parts started"""
        runs = []
        for i, w in enumerate(self.expected, 1):
            if w["file"] in ("ci.yml", "unsafe-and-parallel.yml"):
                runs.append(ok_run(w["name"], i, status="in_progress", conclusion=""))
            else:
                runs.append(ok_run(w["name"], i))
        return runs

    def rid(self, f: str) -> int:
        return 1 + [w["file"] for w in self.expected].index(f)

    def ci_jobs(self, leg_status="in_progress", leg_concl="") -> list[dict]:
        return [{"name": "gate", "status": "completed", "conclusion": "success"},
                {"name": "Test (ubuntu-latest)", "status": "completed", "conclusion": "success"},
                {"name": "Test (macos-latest)", "status": leg_status, "conclusion": leg_concl},
                {"name": "Clippy", "status": "completed", "conclusion": "success"}]

    def miri_jobs(self, heavy="in_progress", concl="") -> list[dict]:
        return [{"name": "Changed paths", "status": "completed", "conclusion": "success"},
                {"name": "loom (batch colouring protocol, --cfg loom)", "status": heavy, "conclusion": concl},
                {"name": "Miri (UB 検出、parallel feature)", "status": heavy, "conclusion": concl},
                {"name": "Gate", "status": "pending", "conclusion": ""}]

    def wait(self, lists, jobs=None, on_dispatch=None, at=SHA, root=ROOT):
        jobs = jobs if jobs is not None else {str(self.rid("ci.yml")): self.ci_jobs(),
                                              str(self.rid("unsafe-and-parallel.yml")): self.miri_jobs()}
        self.state.write_text(json.dumps({"list": lists, "jobs": jobs, "on_dispatch": on_dispatch or {},
                                          "added": [], "dispatched": [], "calls": [], "polls": 0}))

        class L(land.Lander):
            def git(self, *a, check=True):
                return "git@github.com:o/r.git"

            def remote_sha(self, ref):
                return at
        lnd = L(root, "T", log=lambda *_: None)
        lnd.gh_cmd = [sys.executable, str(self.gh)]
        lnd.poll_seconds = 0
        lnd.max_polls = 12
        lnd.wait_evidence("ci/T", SHA)

    def st(self) -> dict:
        return json.loads(self.state.read_text())

    # --- the expected set ----------------------------------------------------------------

    def test_the_expected_set_is_derived_from_the_workflows(self):
        files = {w["file"]: w["cls"] for w in self.expected}
        self.assertEqual(files["unsafe-and-parallel.yml"], "after-landing")
        self.assertEqual(files["security-audit.yml"], "landing")
        for absent in ("quality-deep.yml", "release.yml"):
            self.assertNotIn(absent, files)
        ci = next(w for w in self.expected if w["file"] == "ci.yml")
        self.assertEqual(ci["legs"], ["Test (macos-latest)"])  # the Intel leg is not expected on ci/**
        self.assertGreaterEqual(len(self.expected), 10)

    def tree(self, wf: dict[str, str], evidence: str) -> Path:
        d = self.dir / "tree"
        (d / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
        for k, v in wf.items():
            (d / ".github" / "workflows" / k).write_text(v, encoding="utf-8")
        (d / ".github" / "ci-evidence.toml").write_text(evidence, encoding="utf-8")
        return d

    def test_zero_expected_workflows_fails(self):
        d = self.tree({"w.yml": "name: W\non:\n  schedule:\n    - cron: '0 0 * * *'\njobs:\n  a: {}\n"},
                      '[workflows]\n"w.yml" = "schedule-only"\n')
        with self.assertRaisesRegex(land.LandError, "compared 0 workflows"):
            land.expected_workflows(d)

    def test_an_expected_workflow_without_dispatch_fails_loudly(self):
        d = self.tree({"w.yml": "name: W\non:\n  push:\n    branches: [main, 'ci/**']\njobs:\n  a: {}\n"},
                      '[workflows]\n"w.yml" = "landing"\n')
        with self.assertRaisesRegex(land.LandError, "no workflow_dispatch"):
            land.expected_workflows(d)

    def test_a_ci_workflow_without_an_evidence_class_fails(self):
        d = self.tree({"w.yml": "name: W\non:\n  push:\n    branches: ['ci/**']\n  workflow_dispatch:\njobs:\n  a: {}\n"},
                      '[workflows]\n')
        with self.assertRaisesRegex(land.LandError, "class .* is None"):
            land.expected_workflows(d)

    def test_main_only_and_tag_workflows_are_not_expected(self):
        d = self.tree({"m.yml": "name: M\non:\n  push:\n    branches: [main]\n  workflow_dispatch:\njobs:\n  a: {}\n",
                       "t.yml": "name: T\non:\n  push:\n    tags:\n      - 'v*'\njobs:\n  a: {}\n",
                       "c.yml": "name: C\non:\n  push:\n    branches: [main, 'ci/**']\n  workflow_dispatch:\njobs:\n  a: {}\n"},
                      '[workflows]\n"m.yml" = "landing"\n"t.yml" = "tag-only"\n"c.yml" = "landing"\n')
        self.assertEqual([w["file"] for w in land.expected_workflows(d)], ["c.yml"])

    # --- waiting ---------------------------------------------------------------------------

    def test_all_green_passes(self):
        self.wait([self.green()])
        self.assertEqual(self.st()["dispatched"], [])

    def test_one_red_stops(self):
        runs = self.green()
        runs[self.rid("security-audit.yml") - 1]["conclusion"] = "failure"
        with self.assertRaisesRegex(land.LandError, "is red for .*\n  security-audit.yml: run .* ended failure"):
            self.wait([runs])

    def test_a_path_gated_skip_with_a_gate_success_passes(self):
        # the run of a workflow whose heavy jobs were skipped ends success (the Gate's verdict)
        self.wait([self.green()])

    def test_a_missing_run_is_re_created_then_green(self):
        old = land.NO_RUN_GRACE
        land.NO_RUN_GRACE = -1
        try:
            runs = self.green()
            gone = runs.pop(self.rid("fuzz.yml") - 1)
            self.wait([runs], on_dispatch={"fuzz.yml": [dict(gone, databaseId=99, event="workflow_dispatch")]})
        finally:
            land.NO_RUN_GRACE = old
        self.assertEqual(self.st()["dispatched"], ["fuzz.yml"])
        self.assertIn(["workflow", "run", "fuzz.yml", "--repo", "o/r", "--ref", "ci/T"], self.st()["calls"])
        self.assertFalse(any(c[:2] not in (["run", "list"], ["run", "view"], ["workflow", "run"]) for c in self.st()["calls"]))

    def test_a_missing_run_waits_for_the_grace_before_re_creating(self):
        runs = self.green()
        runs.pop(self.rid("fuzz.yml") - 1)
        with self.assertRaisesRegex(land.LandError, "did not give its evidence in time"):
            self.wait([runs])
        self.assertEqual(self.st()["dispatched"], [])

    def test_a_cancelled_run_is_re_created(self):
        runs = self.green()
        f = self.rid("bench-gate.yml") - 1
        runs[f]["conclusion"] = "cancelled"
        again = dict(runs[f], databaseId=98, conclusion="success", event="workflow_dispatch")
        self.wait([runs], on_dispatch={"bench-gate.yml": [again]})
        self.assertEqual(self.st()["dispatched"], ["bench-gate.yml"])

    def test_re_creation_is_bounded(self):
        runs = self.green()
        f = self.rid("bench-gate.yml") - 1
        runs[f]["conclusion"] = "cancelled"
        still = dict(runs[f], databaseId=97)
        with self.assertRaisesRegex(land.LandError, "after 2 re-created runs"):
            self.wait([runs], on_dispatch={"bench-gate.yml": [still]})
        self.assertEqual(self.st()["dispatched"], ["bench-gate.yml", "bench-gate.yml"])

    def test_a_moved_ref_is_not_re_created(self):
        runs = self.green()
        runs[self.rid("bench-gate.yml") - 1]["conclusion"] = "cancelled"
        with self.assertRaisesRegex(land.LandError, "the ref moved"):
            self.wait([runs], at="b" * 40)
        self.assertEqual(self.st()["dispatched"], [])

    def test_the_newest_run_counts(self):
        runs = self.green()
        f = self.rid("bench-gate.yml") - 1
        old = dict(runs[f], databaseId=1000 + runs[f]["databaseId"], conclusion="failure")
        runs[f]["databaseId"] = 2000
        runs.append(old)  # an older failure (lower id than the newest) does not count
        old["databaseId"] = 5
        self.wait([runs])

    def test_runs_of_another_sha_do_not_count(self):
        runs = self.green()
        runs[self.rid("bench-gate.yml") - 1]["headSha"] = "c" * 40
        with self.assertRaisesRegex(land.LandError, "bench-gate.yml: no run"):
            self.wait([runs])

    def test_waits_for_a_running_landing_workflow(self):
        runs = self.green()
        pending = [dict(r) for r in runs]
        pending[self.rid("downstream.yml") - 1].update(status="in_progress", conclusion="")
        self.wait([pending, pending, runs])
        self.assertGreaterEqual(self.st()["polls"], 3)

    # --- after-landing ---------------------------------------------------------------------

    def test_after_landing_leg_not_started_blocks(self):
        jobs = {str(self.rid("ci.yml")): self.ci_jobs(leg_status="queued"),
                str(self.rid("unsafe-and-parallel.yml")): self.miri_jobs()}
        with self.assertRaisesRegex(land.LandError, "Test \\(macos-latest\\)"):
            self.wait([self.green()], jobs=jobs)

    def test_after_landing_leg_started_is_enough(self):
        self.wait([self.green()])  # Test (macos-latest) in_progress, the rest green

    def test_after_landing_leg_known_red_stops(self):
        jobs = {str(self.rid("ci.yml")): self.ci_jobs(leg_status="completed", leg_concl="failure"),
                str(self.rid("unsafe-and-parallel.yml")): self.miri_jobs()}
        with self.assertRaisesRegex(land.LandError, "is red for .*\n  ci.yml: .*macos-latest.* ended failure"):
            self.wait([self.green()], jobs=jobs)

    def test_intel_is_not_expected_on_ci(self):
        # the ci/** run has no Mac Intel job; the landing does not wait for one
        jobs = {str(self.rid("ci.yml")): self.ci_jobs(),
                str(self.rid("unsafe-and-parallel.yml")): self.miri_jobs()}
        self.assertFalse(any("intel" in j["name"] for j in jobs[str(self.rid("ci.yml"))]))
        self.wait([self.green()], jobs=jobs)

    def test_a_non_leg_ci_job_still_running_is_waited_for(self):
        ci = self.ci_jobs()
        ci[1]["status"], ci[1]["conclusion"] = "in_progress", ""
        jobs = {str(self.rid("ci.yml")): ci, str(self.rid("unsafe-and-parallel.yml")): self.miri_jobs()}
        with self.assertRaisesRegex(land.LandError, "did not give its evidence in time"):
            self.wait([self.green()], jobs=jobs)

    def test_miri_run_false_passes(self):
        jobs = {str(self.rid("ci.yml")): self.ci_jobs(),
                str(self.rid("unsafe-and-parallel.yml")): self.miri_jobs(heavy="completed", concl="skipped")}
        self.wait([self.green()], jobs=jobs)

    def test_miri_run_true_not_started_blocks(self):
        jobs = {str(self.rid("ci.yml")): self.ci_jobs(),
                str(self.rid("unsafe-and-parallel.yml")): self.miri_jobs(heavy="queued")}
        with self.assertRaisesRegex(land.LandError, "heavy jobs not started"):
            self.wait([self.green()], jobs=jobs)

    def test_miri_changes_not_done_blocks(self):
        miri = self.miri_jobs()
        miri[0].update(status="in_progress", conclusion="")
        jobs = {str(self.rid("ci.yml")): self.ci_jobs(), str(self.rid("unsafe-and-parallel.yml")): miri}
        with self.assertRaisesRegex(land.LandError, "the changes job is in_progress"):
            self.wait([self.green()], jobs=jobs)

    def test_a_cancelled_after_landing_run_is_re_created(self):
        runs = self.green()
        f = self.rid("unsafe-and-parallel.yml") - 1
        runs[f].update(status="completed", conclusion="cancelled")
        again = dict(runs[f], databaseId=96, status="in_progress", conclusion="")
        jobs = {str(self.rid("ci.yml")): self.ci_jobs(), "96": self.miri_jobs()}
        self.wait([runs], jobs=jobs, on_dispatch={"unsafe-and-parallel.yml": [again]})
        self.assertEqual(self.st()["dispatched"], ["unsafe-and-parallel.yml"])

    def test_gh_failure_is_not_an_empty_list(self):
        self.state.write_text("{}")  # the fake gh then fails

        class L(land.Lander):
            def git(self, *a, check=True):
                return "git@github.com:o/r.git"
        lnd = L(ROOT, "T", log=lambda *_: None)
        lnd.gh_cmd = [sys.executable, str(self.gh)]
        lnd.poll_seconds = 0
        with self.assertRaisesRegex(land.LandError, "gh run list --repo failed"):
            lnd.wait_evidence("ci/T", SHA)


class Verdict(unittest.TestCase):
    """run_verdict, one state per case."""
    W = {"file": "w.yml", "name": "W", "cls": "landing", "legs": [], "jobs": {}}
    L = {"file": "ci.yml", "name": "CI", "cls": "landing", "legs": ["Test (m)"], "jobs": {}}
    A = {"file": "u.yml", "name": "U", "cls": "after-landing", "legs": [],
         "jobs": {"changes": "Changed paths", "gate": "Gate", "h": "H"}}

    def v(self, w, status="completed", concl="success", jobs=None):
        return land.run_verdict(w, {"databaseId": 1, "status": status, "conclusion": concl}, jobs)[0]

    def test_plain(self):
        self.assertEqual(land.run_verdict(self.W, None, None)[0], "missing")
        self.assertEqual(self.v(self.W), "ok")
        self.assertEqual(self.v(self.W, concl="failure"), "red")
        self.assertEqual(self.v(self.W, concl="cancelled"), "dropped")
        self.assertEqual(self.v(self.W, concl="skipped"), "dropped")
        self.assertEqual(self.v(self.W, status="queued", concl=""), "waiting")

    def test_legs(self):
        j = lambda n, s, c="": {"name": n, "status": s, "conclusion": c}  # noqa: E731
        self.assertEqual(self.v(self.L, status="in_progress", concl=""), "need-jobs")
        self.assertEqual(self.v(self.L, status="in_progress", concl="", jobs=[]), "waiting")
        ok = [j("a", "completed", "success"), j("Test (m)", "in_progress")]
        self.assertEqual(self.v(self.L, status="in_progress", concl="", jobs=ok), "ok")
        self.assertEqual(self.v(self.L, status="in_progress", concl="", jobs=[ok[0], j("Test (m)", "queued")]), "waiting")
        self.assertEqual(self.v(self.L, status="in_progress", concl="", jobs=[j("a", "in_progress"), ok[1]]), "waiting")
        self.assertEqual(self.v(self.L, status="in_progress", concl="", jobs=[j("a", "completed", "failure"), ok[1]]), "red")
        self.assertEqual(self.v(self.L, status="in_progress", concl="", jobs=[j("a", "completed", "skipped"), ok[1]]), "red")
        self.assertEqual(self.v(self.L, status="in_progress", concl="", jobs=[ok[0]]), "waiting")
        self.assertEqual(self.v(self.L, concl="failure", jobs=[ok[0]]), "red")
        self.assertEqual(self.v(self.L), "ok")

    def test_after_landing(self):
        j = lambda n, s, c="": {"name": n, "status": s, "conclusion": c}  # noqa: E731
        ch = j("Changed paths", "completed", "success")
        run = dict(status="in_progress", concl="")
        self.assertEqual(self.v(self.A, **run, jobs=[ch, j("H", "in_progress"), j("Gate", "pending")]), "ok")
        self.assertEqual(self.v(self.A, **run, jobs=[ch, j("H", "completed", "skipped")]), "ok")
        self.assertEqual(self.v(self.A, **run, jobs=[ch, j("H", "queued")]), "waiting")
        self.assertEqual(self.v(self.A, **run, jobs=[ch, j("H", "completed", "failure")]), "red")
        self.assertEqual(self.v(self.A, **run, jobs=[ch, j("H", "completed", "skipped"), j("H2", "in_progress")]), "red")
        self.assertEqual(self.v(self.A, **run, jobs=[j("Changed paths", "in_progress")]), "waiting")
        self.assertEqual(self.v(self.A, **run, jobs=[j("Changed paths", "completed", "failure")]), "red")
        self.assertEqual(self.v(self.A, **run, jobs=[ch]), "waiting")
        self.assertEqual(self.v(self.A, **run, jobs=[j("H", "in_progress")]), "waiting")
        self.assertEqual(self.v(self.A, concl="cancelled"), "dropped")


if __name__ == "__main__":
    unittest.main()
