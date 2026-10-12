#!/usr/bin/env python3
"""Tests for scripts/land.py.

Each case builds a throwaway origin (a bare repository) and a clone, makes
commits, and lands them with a Lander whose heavy steps (preflight,
regeneration, checks, CI wait) are recorded instead of run. One rule per case.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import land  # noqa: E402

# the landing lock of these tests lives in a throwaway directory, not in the home cache
os.environ[land.LOCK_DIR_ENV] = tempfile.mkdtemp(prefix="land-lock-")
atexit.register(shutil.rmtree, os.environ[land.LOCK_DIR_ENV], True)

# --- the harness guards --------------------------------------------------------------
# A Lander never runs against this checkout (land.py refuses a forbidden root before its
# first git call), and every git / gh that land.py starts goes through a tripwire wrapper:
# an argument or a remote naming a network host, or any gh that is not a test's fake, writes
# TRIPWIRE and exits 97. Isolated fails a test that tripped it.
REAL_ROOT = Path(__file__).resolve().parent.parent
land.FORBIDDEN_ROOTS.add(os.path.realpath(REAL_ROOT))
TRIP_DIR = Path(tempfile.mkdtemp(prefix="land-tripwire-"))
atexit.register(shutil.rmtree, TRIP_DIR, True)
TRIPWIRE = TRIP_DIR / "tripwire"
REAL_GIT = land.pinned_git()   # the same choice as land.py (the first git new enough)
GIT_TRIPWIRE = r"""import os, re, subprocess, sys
REAL, MARKER = @REAL@, @MARKER@
args = sys.argv[1:]
NET = re.compile(r"^(?:(?:https?|ssh|git|ftps?)://|[A-Za-z0-9._-]+@[A-Za-z0-9._-]+:)|github\.com", re.I)


def trip(why):
    with open(MARKER, "a", encoding="utf-8") as fh:
        fh.write(f"git: {why} :: {' '.join(args)}\n")
    print(f"TRIPWIRE git: {why}", file=sys.stderr)
    sys.exit(97)


def local(url):
    url = url.strip()
    if url.startswith("file://"):
        url = url[len("file://"):]
    return bool(url) and not NET.search(url) and (os.path.isabs(url) or os.path.exists(url))


for a in args:
    if NET.search(a) and not a.startswith("file://"):
        trip(f"network argument {a!r}")
# git's main options (git(1), git.c handle_options), before the subcommand:
#   redirect the repository, the work tree, the programs or the config source: land.py never
#   passes them, so any of them trips (with a separate value or `=value`)
REDIRECT = ("-C", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--exec-path",
            "--config-env", "--shallow-file", "--bare")
#   take a separate value (or `=value`) and are allowed
VALUED = ("-c", "--attr-source", "--list-cmds")
#   flags without a value that change nothing about where git reads or sends
FLAGS = ("-p", "--paginate", "-P", "--no-pager", "--no-replace-objects", "--no-lazy-fetch",
         "--no-optional-locks", "--no-advice", "--literal-pathspecs", "--no-literal-pathspecs",
         "--glob-pathspecs", "--noglob-pathspecs", "--icase-pathspecs")
# anything else before the subcommand (an unknown option, a value missing at the end, an
# informational option such as --version that leaves no subcommand) trips: fail closed
i, cwd = 0, None
while i < len(args) and args[i].startswith("-"):
    opt, eq, _ = args[i].partition("=")
    if opt in REDIRECT:
        trip(f"main option {opt} redirects git")
    if opt in VALUED:
        if eq:
            i += 1
        elif i + 1 < len(args):
            i += 2
        else:
            trip(f"main option {opt} without its value")
    elif args[i] in FLAGS:
        i += 1
    else:
        trip(f"unknown main option {args[i]!r} before the subcommand")
if i >= len(args):
    trip("no subcommand")
sub = args[i]
rest = args[i + 1:]
if sub in ("fetch", "push", "pull", "ls-remote", "clone", "remote") and "--get-url" not in rest \
        and not (sub == "remote" and rest[:1] != ["update"]):
    # the repository is the first argument that is not an option (`push [opts] <repo> [refspec]`);
    # none: origin. It must resolve, as a configured remote, to a local fetch and push URL: a
    # name or path that git cannot resolve trips too (fail closed)
    name = next((x for x in rest if not x.startswith("-")), "origin")
    for q in (["ls-remote", "--get-url", name], ["remote", "get-url", "--push", name]):
        r = subprocess.run([REAL, *(["-C", cwd] if cwd else []), *q], capture_output=True, text=True)
        url = r.stdout.strip()
        if r.returncode != 0 or not url:
            trip(f"{sub}: cannot resolve {name!r} ({' '.join(q)} exit {r.returncode}): fail closed")
        if not local(url):
            trip(f"{sub} reaches {url!r}")
if os.name != "nt":
    os.execv(REAL, [REAL, *args])   # no second process on POSIX
sys.exit(subprocess.call([REAL, *args]))
"""
GH_TRIPWIRE = r"""import sys
with open(@MARKER@, "a", encoding="utf-8") as fh:
    fh.write("gh: " + " ".join(sys.argv[1:]) + "\n")
print("TRIPWIRE gh: a gh that is not a test's fake", file=sys.stderr)
sys.exit(97)
"""
for _name, _src in (("git_tripwire.py", GIT_TRIPWIRE), ("gh_tripwire.py", GH_TRIPWIRE)):
    (TRIP_DIR / _name).write_text(_src.replace("@REAL@", repr(REAL_GIT)).replace("@MARKER@", repr(str(TRIPWIRE))),
                                  encoding="utf-8")
# -I: the wrapper ignores PYTHON* variables and the user's site directory (a hostile
# PYTHONPATH cannot replace it); -S: no site import, a cheaper start (the wrapper runs for
# every git call of land.py in these tests, about 2300 per run)
land.TOOL_OVERRIDE["git"] = [sys.executable, "-I", "-S", str(TRIP_DIR / "git_tripwire.py")]
land.TOOL_OVERRIDE["gh"] = [sys.executable, "-I", "-S", str(TRIP_DIR / "gh_tripwire.py")]


def live_enabled(env) -> bool:
    """live tests (the network) run only when asked for, and never in CI"""
    return env.get("LAND_LIVE_TESTS") == "1" and not env.get("CI")


LIVE_TESTS: list[str] = []


def live(fn):
    """mark a test that reaches the network: skipped unless LAND_LIVE_TESTS=1, always in CI"""
    LIVE_TESTS.append(fn.__qualname__)
    fn.live = True
    return unittest.skipUnless(live_enabled(os.environ), "live test (network): set LAND_LIVE_TESTS=1, "
                                                         "never in CI")(fn)


def assert_local_origin(work: Path, inside: Path) -> None:
    """the effective fetch and push URLs of `work`'s origin (after insteadOf) are a local
    bare repository inside `inside`; AssertionError otherwise"""
    for q in (["remote", "get-url", "--push", "origin"], ["ls-remote", "--get-url", "origin"]):
        url = subprocess.run([REAL_GIT, "-C", str(work), *q], capture_output=True, text=True).stdout.strip()
        path = url[len("file://"):] if url.startswith("file://") else url
        real, root = os.path.realpath(path), os.path.realpath(inside)
        if not (path and os.path.isabs(path) and os.path.isdir(real) and real.startswith(root + os.sep)):
            raise AssertionError(f"the test repository's origin {url!r} ({' '.join(q)}) is not a local "
                                 f"repository inside {inside}: refusing to run the test")
NAME, EMAIL = "Moroya Sakamoto", "sakamoro@alicelaw.net"
ID = ["-c", f"user.name={NAME}", "-c", f"user.email={EMAIL}"]

# Variables a shell may export that change what git does in these tests: the identity
# (git's environment wins over `-c user.*`), the repository or index git works on, and
# injected config. Every test runs with them removed (Isolated) and restored afterwards.
AMBIENT_ENV_RE = re.compile(
    r"^(GIT_(AUTHOR|COMMITTER)_(NAME|EMAIL|DATE)|LAND_IDENTITY|GIT_DIR|GIT_WORK_TREE|GIT_INDEX_FILE"
    r"|GIT_COMMON_DIR|GIT_OBJECT_DIRECTORY|GIT_ALTERNATE_OBJECT_DIRECTORIES|GIT_NAMESPACE"
    r"|GIT_CEILING_DIRECTORIES|GIT_CONFIG.*)$")
DEFAULT_IDENTITY = f"{NAME} <{EMAIL}>"


class Isolated(unittest.TestCase):
    """Runs each test without the ambient git identity / repository variables and with
    land.py's default identity, whatever the shell exports."""

    def setUp(self):
        saved = {k: v for k, v in os.environ.items() if AMBIENT_ENV_RE.match(k)}
        for k in saved:
            del os.environ[k]
        old_identity = land.IDENTITY
        land.IDENTITY = DEFAULT_IDENTITY

        def restore():
            for k in [k for k in os.environ if AMBIENT_ENV_RE.match(k)]:
                del os.environ[k]
            os.environ.update(saved)
            land.IDENTITY = old_identity
        self.addCleanup(restore)
        self.addCleanup(self.check_tripwire)
        # the throwaway repository every Lander of the test works in
        self.r = Repo()
        self.addCleanup(lambda: self.r.close())
        assert_local_origin(self.r.work, self.r.dir)

    def check_tripwire(self):
        if TRIPWIRE.exists():
            text = TRIPWIRE.read_text(encoding="utf-8")
            TRIPWIRE.unlink()
            self.fail(f"the network / gh tripwire fired during the test:\n{text}")

    def wf_root(self) -> Path:
        """a throwaway repository holding a copy of this repository's .github (for the tests
        that read the workflows and the evidence classes)"""
        repo = Repo()
        self.addCleanup(repo.close)
        assert_local_origin(repo.work, repo.dir)
        shutil.copytree(REAL_ROOT / ".github", repo.work / ".github", dirs_exist_ok=True)
        return repo.work


# what a shell might export, all pointing elsewhere
def hostile_env(elsewhere: Path) -> dict[str, str]:
    return {"GIT_COMMITTER_NAME": "Ambient Committer", "GIT_COMMITTER_EMAIL": "ambient@committer.invalid",
            "GIT_AUTHOR_NAME": "Ambient Author", "GIT_AUTHOR_EMAIL": "ambient@author.invalid",
            "GIT_COMMITTER_DATE": "2001-01-01T00:00:00", "GIT_AUTHOR_DATE": "2001-01-01T00:00:00",
            "LAND_IDENTITY": "web-flow <ambient@identity.invalid>",
            "GIT_DIR": str(elsewhere / ".git"), "GIT_WORK_TREE": str(elsewhere),
            "GIT_INDEX_FILE": str(elsewhere / "index"),
            "GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "user.email", "GIT_CONFIG_VALUE_0": "ambient@config.invalid",
            # injected hooks: a commit or a rebase run with them leaves a marker in `elsewhere`
            "GIT_CONFIG_KEY_1": "core.hooksPath", "GIT_CONFIG_VALUE_1": str(elsewhere / "hooks"),
            "GIT_CONFIG_PARAMETERS": "'user.name'='Ambient Config'"}


def hostile_hooks(elsewhere: Path) -> Path:
    """hooks under `elsewhere` that write a marker when git runs them"""
    hooks = elsewhere / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    marker = elsewhere / "hook-ran"
    for name in ("pre-commit", "post-commit", "post-rewrite"):
        h = hooks / name
        h.write_text(f"#!/bin/sh\necho {name} >> '{marker.as_posix()}'\n", encoding="utf-8")
        h.chmod(0o755)
    return marker


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

    def __init__(self, *a, ci_result: str = "success", before_push=None, lock_wait=None, **k):
        super().__init__(*a, log=lambda *_: None, **k)
        if lock_wait is not None:
            self.lock_wait_seconds = lock_wait
        self.calls: list[str] = []
        self.ci_result = ci_result
        # main moves: one hook per round, fired at the first fetch after each round of checks
        if before_push is None:
            self.moves = []
        elif isinstance(before_push, list):
            self.moves = list(before_push)
        else:
            self.moves = [before_push]
        self.fired = 0
        self.lock_poll_seconds = 0.05
        # the test origin is a local bare repository: its own URL is the expected remote
        self.expected_remote_urls = (self.git("config", "--get", "remote.origin.url"),)

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

    def rebase(self):
        self.calls.append("rebase")
        super().rebase()

    def wait_evidence(self, ref, sha):
        self.calls.append(f"wait_evidence {ref}")
        self.evidenced = getattr(self, "evidenced", []) + [sha]
        if self.ci_result != "success":
            raise land.LandError(f"CI on {ref} ended `{self.ci_result}`")
        self.record.append("evidence: ci.yml run 1 success")

    def check_gh_auth(self):
        self.calls.append("gh-auth")

    def api_commit_identity(self, sha):
        # the GitHub API stand-in: the commit as the remote holds it
        self.calls.append("api")
        return tuple(self.git("log", "-1", "--format=%an%x00%ae%x00%cn%x00%ce", f"{self.upstream()}").split("\0")) \
            if self.git("rev-parse", self.upstream()) == sha else ("?", "?", "?", "?")

    def fetch(self):
        if self.moves and self.fired < self.calls.count("checks"):
            self.fired += 1
            self.moves.pop(0)()
        super().fetch()


class Repo:
    """origin (bare) + `upstream` clone (other sessions) + `work` clone (ours)."""

    def __init__(self, gitattributes: bool = True):
        self.dir = Path(tempfile.mkdtemp())
        self.origin = self.dir / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(self.origin)], check=True)
        # a commit, rebase, merge or received push starts `git maintenance run --auto --detach`;
        # it can still pack refs or write into .git while a test reads them or the directory is
        # removed
        git(self.origin, "config", "maintenance.auto", "false")
        self.upstream = self.dir / "upstream"
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.upstream)], check=True,
                       capture_output=True)
        git(self.upstream, "checkout", "-q", "-b", "main")
        git(self.upstream, "config", "maintenance.auto", "false")
        commit(self.upstream, {"src/a.rs": "fn a() {}\n",
                                "scripts/lib-test-exempt.txt": "src/b.rs fixture for the landing steps\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n",
                               # the inputs of a path-gated workflow: what docs_only() excludes
                               ".github/paths/w.txt": "src/**\nREADME.md\ndocs/PUBLIC_API_SNAPSHOT.txt\ndocs/wiring-status.md\n",
                               "docs/integration-status.md": "ledger: a.rs\n",
                               **({".gitattributes": "CHANGELOG.md merge=union\n"} if gitattributes else {})},
               "init")
        git(self.upstream, "push", "-q", "origin", "main")
        self.work = self.dir / "work"
        subprocess.run(["git", "clone", "-q", str(self.origin), str(self.work)], check=True, capture_output=True)
        git(self.work, "config", "maintenance.auto", "false")

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


class Pure(Isolated):
    def test_docs_only(self):
        rules = land.workflow_input_rules(ROOT)
        d = lambda *p: land.docs_only(list(p), rules)  # noqa: E731
        self.assertTrue(d("CHANGELOG.md"))
        self.assertTrue(d("README_JP.md", "docs/ROADMAP.md"))
        # the ledgers land.py rebuilds and checks after every rebase, though workflows read them
        self.assertTrue(d("docs/oracle-status.md", "docs/wiring-status.md", "docs/integration-status.md",
                          "docs/integration-levels.md", "docs/coverage/status.md"))
        self.assertTrue(d("CHANGELOG.md", "docs/oracle-status.md"))
        self.assertFalse(d("CHANGELOG.md", "src/a.rs"))
        self.assertFalse(d("docs/x.md", "src/a.rs"))
        self.assertFalse(d())
        # documents that a workflow reads (from .github/paths): not docs-only
        self.assertFalse(d("docs/PUBLIC_API_SNAPSHOT.txt"))   # security-audit public-api job
        self.assertFalse(d("README.md"))                      # cargo package / wheel metadata
        self.assertFalse(d("docs/wiring-debt.md"))            # wiring_guard
        self.assertFalse(d("CHANGELOG.md", "docs/PUBLIC_API_SNAPSHOT.txt"))
        self.assertFalse(d("docs/x.rs"))                      # **/*.rs: a source file wherever it is
        # outside the document paths
        for p in ("README.md/x", "sub/README.md", "sub/CHANGELOG.md", "Cargo.toml", ".github/paths/x.txt",
                  "scripts/x.py", "tests/a.rs"):
            self.assertFalse(d(p), p)
        # no input list at all: nothing is docs-only (fail-closed)
        with self.assertRaises(land.LandError):
            land.workflow_input_rules(Path(tempfile.mkdtemp()))

    def test_docs_only_follows_the_lists(self):
        import changed_paths as cp
        rules = [cp.compile_patterns(["docs/ROADMAP.md"])]
        self.assertFalse(land.docs_only(["docs/ROADMAP.md"], rules))
        self.assertTrue(land.docs_only(["docs/ROADMAP.md"], []))
        self.assertTrue(land.docs_only(["docs/oracle-status.md"], [cp.compile_patterns(["docs/**"])]))


    def test_direct_reason_rules(self):
        self.assertIsNone(land.direct_reason_problem(False, None))
        self.assertIn("at least 12", land.direct_reason_problem(True, None))
        self.assertIn("at least 12", land.direct_reason_problem(True, "x" * 11))
        self.assertIn("at least 12", land.direct_reason_problem(True, "  " + "x" * 11 + "   "))
        self.assertIsNone(land.direct_reason_problem(True, "x" * 12))
        self.assertIn("only for --direct", land.direct_reason_problem(False, "revert the red main"))

    def test_the_cli_refuses_direct_without_a_reason(self):
        for argv in (["--id", "x", "--direct"], ["--id", "x", "--direct", "--reason", "too short"],
                     ["--id", "x", "--reason", "revert the red main abc"]):
            self.assertEqual(land.main(argv), 2, argv)

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


class Identity(Isolated):
    def test_denied_forms(self):
        p = land.identity_problem
        for name, email in (("Some One", "noreply@github.com"), ("Some One", "NoReply@GitHub.com "),
                            ("Some One", "41898282+bot@users.noreply.github.com"),
                            ("Some One", "x@USERS.NOREPLY.GITHUB.COM"),
                            ("web-flow", "a@b.c"), ("Web-Flow", "a@b.c"), ("GitHub", "a@b.c"), (" github ", "a@b.c")):
            self.assertIsNotNone(p(name, email), (name, email))
        for name, email in ((NAME, EMAIL), ("GitHub Actions fan", "a@b.c"), ("Some One", "noreply@github.com.example"),
                            ("Some One", "users.noreply.github.com@example.org"),
                            ("Some One", "a@users.noreply.github.com.example.org"), ("webflow", "a@b.c")):
            self.assertIsNone(p(name, email), (name, email))

    def test_the_override_is_refused(self):
        old = land.IDENTITY
        try:
            for ident in ("Some One <noreply@github.com>", "Some One <9+me@users.noreply.github.com>",
                          "web-flow <a@b.c>", "GitHub <a@b.c>", "GITHUB <a@b.c>", "no email here"):
                land.IDENTITY = ident
                with self.assertRaisesRegex(land.LandError, "LAND_IDENTITY .* (is refused|is not `Name <email>`)"):
                    land.Lander(self.r.work, "T", log=lambda *_: None)
            land.IDENTITY = f"{NAME} <{EMAIL}>"
            land.Lander(self.r.work, "T", log=lambda *_: None)  # the normal identity is accepted
        finally:
            land.IDENTITY = old

    def test_the_cli_refuses_a_denied_override(self):
        old = land.IDENTITY
        try:
            land.IDENTITY = "web-flow <a@b.c>"
            self.assertEqual(land.main(["--id", "x", "--dry-run"]), 2)
        finally:
            land.IDENTITY = old


class Landing(Isolated):

    def test_a_new_source_file_without_lib_tests_stops_the_land(self):
        commit(self.r.work, {"src/c.rs": "pub fn c() {}\n",
                             "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- c\n"}, "feat: c")
        lander = self.r.lander()
        with self.assertRaises(land.LandError) as cm:
            lander.land()
        self.assertIn("lib-test gate", str(cm.exception))
        self.assertIn("src/c.rs: new source file without lib tests", str(cm.exception))

    def test_a_src_only_change_takes_the_ci_lane(self):
        commit(self.r.work, {"src/b.rs": "fn b() {}\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander()
        sha = lander.land()
        self.assertEqual(self.r.origin_head(), sha)
        self.assertIn("wait_evidence ci/T-1", lander.calls)
        self.assertIn("lane: ci (ci/T-1)", lander.record)
        self.assertIn("evidence: ci.yml run 1 success", lander.record)
        self.assertIn(f"landed SHA: {sha} on main (attempt 1)", lander.record)
        self.assertIn(f"evidence SHA: {sha} on ci/T-1", lander.record)   # the evidenced SHA itself landed
        self.assertIn("reason: main-unmoved", lander.record)
        self.assertNotIn("rebase", lander.calls)  # already on main with the fixed committer: not rewritten
        self.assertEqual((self.r.work / "docs/integration-status.md").read_text(), "ledger: a.rs, b.rs\n")
        self.assertIn("docs: regenerate the generated ledgers", git(self.r.work, "log", "-1", "--format=%s"))

    def test_direct_landing_is_recorded_with_its_reason(self):
        commit(self.r.work, {"src/b.rs": "fn b() {}\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        logged = []
        lander = FakeLander(self.r.work, "T-1", direct_reason="  revert the red main of abc1234  ")
        lander.log = logged.append
        sha = lander.land()
        self.assertEqual(self.r.origin_head(), sha)
        self.assertNotIn("wait_evidence ci/T-1", lander.calls)
        self.assertIn("lane: DIRECT (emergency, no ci/ evidence) reason: revert the red main of abc1234", lander.record)
        self.assertTrue(any("WARNING: DIRECT LANDING" in l for l in logged), logged)
        self.assertTrue(any("WARNING: reason: revert the red main of abc1234" in l for l in logged))
        self.assertTrue(any("landing record:" in l for l in logged))
        self.assertTrue(any("lane: DIRECT" in l for l in logged))
        self.assertIn("evidence SHA: -", lander.record)
        self.assertIn("reason: direct (no evidence)", lander.record)

    def test_direct_landing_without_a_good_reason_is_refused(self):
        for reason in ("", "too short", " " * 20):
            with self.assertRaisesRegex(land.LandError, "at least 12"):
                FakeLander(self.r.work, "T-1", direct_reason=reason)

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

    def test_a_denied_committer_in_the_range_is_refused(self):
        commit(self.r.work, {"docs/x.md": "x\n"}, "docs: x")
        for name, email in (("Some One", "noreply@github.com"), ("Some One", "1234+someone@users.noreply.github.com"),
                            ("web-flow", "a@b.c"), ("GitHub", "a@b.c")):
            commit(self.r.work, {"docs/y.md": name + email + "\n"}, "docs: y")
            git(self.r.work, "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "--amend",
                "--no-edit")
            sha = git(self.r.work, "rev-parse", "--short=7", "HEAD")
            with self.assertRaisesRegex(land.LandError, re.escape(f"{sha}: committer `{name} <{email}>` is refused")):
                self.r.lander().land()
            git(self.r.work, "reset", "-q", "--hard", "HEAD~1")

    def test_a_denied_author_is_refused_with_its_field(self):
        commit(self.r.work, {"docs/x.md": "x\n"}, "docs: x", author="GitHub <noreply@github.com>")
        with self.assertRaisesRegex(land.LandError, "author `GitHub <noreply@github.com>` is refused: the email"):
            self.r.lander().land()

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

    HASH_BODY = ("feat: b with a cfg path\n\n"
                 "#[path = \"../tests/b_oracle.rs\"] brings the oracle into the lib tests\n"
                 "# a line that starts with a hash\n"
                 "  #indented hash\n\n"
                 "trailing text\n")

    def commit_with_hash_lines(self) -> str:
        write(self.r.work, "src/b.rs", "x\n")
        write(self.r.work, "CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n\n- start\n- mine\n")
        git(self.r.work, "add", "-A")
        msg = self.r.dir / "msg.txt"
        msg.write_text(self.HASH_BODY, encoding="utf-8")
        subprocess.run(["git", *ID, "commit", "-q", "--cleanup=verbatim", "-F", str(msg), "--author", f"{NAME} <{EMAIL}>",
                        "--date", "2001-02-03T04:05:06+00:00"], cwd=self.r.work, check=True)
        return git(self.r.work, "rev-parse", "HEAD")

    def raw(self, repo: Path, rev: str) -> bytes:
        out = subprocess.run(["git", "cat-file", "commit", rev], cwd=repo, capture_output=True, check=True).stdout
        return out.partition(b"\n\n")[2]

    def test_a_hash_line_survives_a_conflicted_rebase(self):
        mine = self.commit_with_hash_lines()
        before = self.raw(self.r.work, mine)
        self.assertIn(b"#[path", before)
        self.r.push_upstream({"CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- theirs\n"},
                             "docs(changelog): theirs")
        class Reflog(FakeLander):
            # the reflog of land.py's own working copy, read before it is removed
            def _land(self):
                sha = super()._land()
                self.reflog = git(self.root, "reflog", "--format=%gs").splitlines()
                return sha
        lander = Reflog(self.r.work, "T-1")
        sha = lander.land()
        # the pick stopped on the conflict and was committed by land.py itself (between the
        # rebase's start and finish), not by `rebase --continue`
        # (a detached HEAD: the rebase ends without a `finish` entry; a pick finished by
        # `rebase --continue` would show as `rebase (continue): ...`)
        log = lander.reflog
        start = log.index("rebase (start): checkout origin/main")
        self.assertIn("commit: feat: b with a cfg path", log[:start], log)
        self.assertFalse([l for l in log if l.startswith("rebase (continue)")], log)
        git(self.r.upstream, "fetch", "-q")
        landed = git(self.r.upstream, "rev-list", "--reverse", "origin/main~2..origin/main").split()
        ours = [c for c in landed if git(self.r.upstream, "log", "-1", "--format=%s", c) == "feat: b with a cfg path"]
        self.assertEqual(len(ours), 1, landed)
        self.assertEqual(self.raw(self.r.upstream, ours[0]), before)   # byte-identical, '#' lines kept
        who = git(self.r.upstream, "log", "-1", "--format=%an <%ae>|%at|%cn <%ce>", ours[0])
        self.assertEqual(who, f"{NAME} <{EMAIL}>|981173106|{NAME} <{EMAIL}>")   # 2001-02-03T04:05:06Z
        self.assertEqual(self.r.origin_head(), sha)
        text = (self.r.work / "CHANGELOG.md").read_text()
        self.assertIn("- mine", text)
        self.assertIn("- theirs", text)

    def test_a_changed_message_is_refused_before_the_push(self):
        mine = self.commit_with_hash_lines()
        before_head = self.r.origin_head()

        class Rewriting(FakeLander):
            # stands for any step that would change a message (what the default cleanup did)
            def regenerate(self):
                super().regenerate()
                if not getattr(self, "rewritten", False):
                    self.rewritten = True
                    msg = Path(tempfile.mkdtemp()) / "stripped.txt"   # (the working copy's .git is a file)
                    msg.write_text("feat: b with a cfg path\n\ntrailing text\n", encoding="utf-8")
                    subprocess.run(["git", *ID, "commit", "-q", "--amend", "-F", str(msg)], cwd=self.root, check=True)
        for kw in ({}, {"direct_reason": "revert the red main of abc1234"}):   # the ci/ lane and --direct
            lander = Rewriting(self.r.work, "T-1", **kw)
            with self.assertRaisesRegex(land.LandError, "commit message of [0-9a-f]{7} \\(was [0-9a-f]{7}\\) differs from "
                                                        "before the rebase; first difference: "):
                lander.land()
            self.assertEqual(self.r.origin_head(), before_head, f"nothing was pushed ({kw})")
            self.assertNotIn("wait_evidence ci/T-1", lander.calls)
            git(self.r.work, "reset", "-q", "--hard", mine)

    def test_the_message_invariant(self):
        mine = self.commit_with_hash_lines()
        lander = self.r.lander()
        lander.fetch()
        with self.assertRaisesRegex(land.LandError, "compared nothing"):
            lander.check_messages()               # nothing recorded before the rebase
        lander.original_messages = lander.our_messages()
        lander.check_messages()                   # unchanged: passes
        # land.py's own ledger commit is not one of ours and does not count
        write(self.r.work, "docs/integration-status.md", "ledger: changed\n")
        git(self.r.work, "commit", "-qam", land.REGEN_MESSAGE)
        lander.check_messages()
        # any other change: the first differing line is named
        lander.original_messages = [(s, m.replace(b"# a line", b"# A line")) for s, m in lander.original_messages]
        with self.assertRaisesRegex(land.LandError, r"first difference: '- # A line"):
            lander.check_messages()
        lander.original_messages = lander.original_messages * 2
        with self.assertRaisesRegex(land.LandError, "2 commits before the rebase, 1 now"):
            lander.check_messages()

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
        self.assertLess(lander.calls.index("regenerate"), lander.calls.index("preflight"), lander.calls)
        landed = git(self.r.work, "show", "HEAD:docs/integration-status.md")
        self.assertIn("b.rs", landed)

    def test_docs_only_upstream_does_not_rerun_preflight(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander(before_push=lambda: self.r.push_upstream({"docs/oracle-status.md": "regenerated\n"},
                                                                         "docs: oracle-status"))
        sha = lander.land()
        self.assertEqual(lander.preflight_runs, 1)
        self.assertEqual(lander.calls.count("wait_evidence ci/T-1"), 1)
        self.assertNotIn("compile", lander.calls)
        self.assertIn("reason: docs-only-moved", lander.record)
        self.assertIn(f"evidence SHA: {lander.evidenced[0]} on ci/T-1", lander.record)
        self.assertIn(f"landed SHA: {sha} on main (attempt 1)", lander.record)
        self.assertNotEqual(lander.evidenced[0], sha)  # rebased onto the documents

    # --- what reaches main is the evidence: land.py's own working copy, the evidence rule ----

    def landing_logs(self, lander) -> list[str]:
        logged = []
        lander.log = logged.append
        return logged

    def test_the_last_line_names_evidence_landed_and_rule(self):
        commit(self.r.work, {"src/b.rs": "fn b() {}\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        source = git(self.r.work, "rev-parse", "HEAD")
        lander = self.r.lander()
        logged = self.landing_logs(lander)
        sha = lander.land()
        self.assertEqual(logged[-1], f"EVIDENCE {sha} LANDED {sha} RULE main-unmoved")
        self.assertIn(f"source SHA: {source} (the checkout's HEAD at the start)", lander.record)
        self.assertEqual(git(self.r.work, "rev-parse", "HEAD"), sha)   # the checkout follows the landing
        self.assertEqual(git(self.r.work, "worktree", "list", "--porcelain").count("worktree "), 1)  # removed
        # docs-only-moved and direct name their rule too
        commit(self.r.work, {"src/c.rs": "fn c() {}\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n- c\n",
                             "scripts/lib-test-exempt.txt": "src/b.rs fixture for the landing steps\nsrc/c.rs fixture\n"},
               "feat: c")
        lander = self.r.lander(before_push=lambda: self.r.push_upstream({"docs/oracle-status.md": "regenerated\n"},
                                                                         "docs: oracle-status"))
        logged = self.landing_logs(lander)
        sha = lander.land()
        self.assertEqual(logged[-1], f"EVIDENCE {lander.evidenced[0]} LANDED {sha} RULE docs-only-moved")
        commit(self.r.work, {"docs/d.md": "d\n"}, "docs: d")
        lander = FakeLander(self.r.work, "T-1", direct_reason="revert the red main of abc1234")
        logged = self.landing_logs(lander)
        sha = lander.land()
        self.assertEqual(logged[-1], f"EVIDENCE - LANDED {sha} RULE direct")

    def test_a_checkout_moved_during_the_wait_is_refused(self):
        # what happened once: during the CI wait the operator checked out another branch (a child of
        # the evidence) in the same checkout, and the push of HEAD took that commit to main
        commit(self.r.work, {"src/b.rs": "fn b() {}\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        before = self.r.origin_head()
        work = self.r.work

        class Meddled(FakeLander):
            def wait_evidence(self, ref, sha):
                super().wait_evidence(ref, sha)
                git(work, "checkout", "-q", "-b", "next")
                commit(work, {"docs/next.md": "unverified\n"}, "docs: next")
        lander = Meddled(work, "T-1")
        with self.assertRaisesRegex(land.LandError, "the checkout's HEAD moved during the landing"):
            lander.land()
        self.assertEqual(self.r.origin_head(), before)

    def test_a_working_copy_moved_after_the_preparation_is_refused(self):
        commit(self.r.work, {"src/b.rs": "fn b() {}\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        before = self.r.origin_head()

        class Moved(FakeLander):
            # a step between the preparation and the push that moves land.py's own HEAD
            def check_remote(self):
                super().check_remote()
                if "wait_evidence ci/T-1" in self.calls and not getattr(self, "moved", False):
                    self.moved = True
                    git(self.root, *ID, "commit", "-q", "--allow-empty", "-m", "docs: extra")
        with self.assertRaisesRegex(land.LandError, "land.py's working copy moved from the prepared [0-9a-f]{7}"):
            Moved(self.r.work, "T-1").land()
        self.assertEqual(self.r.origin_head(), before)

    def test_a_checkout_with_changes_is_not_overwritten_after_the_landing(self):
        # changes made in the checkout during the landing, in a file the landing changes: the
        # landing stands, `reset --keep` refuses and leaves the checkout and the change as they were
        commit(self.r.work, {"src/b.rs": "fn b() {}\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        source = git(self.r.work, "rev-parse", "HEAD")
        work = self.r.work

        class Edited(FakeLander):
            def wait_evidence(self, ref, sha):
                super().wait_evidence(ref, sha)
                write(work, "docs/integration-status.md", "edited by hand\n")
        lander = Edited(work, "T-1")
        logged = self.landing_logs(lander)
        sha = lander.land()
        self.assertEqual(self.r.origin_head(), sha)
        self.assertNotEqual(sha, source)   # the ledger commit: the file differs between the two
        self.assertEqual(git(work, "rev-parse", "HEAD"), source)
        self.assertEqual((work / "docs/integration-status.md").read_text(), "edited by hand\n")
        self.assertTrue(lander.checkout_lagging)
        self.assertTrue(any(l.startswith(f"landed {sha[:7]}, but the checkout stays at {source[:7]}") for l in logged), logged)
        self.assertEqual(land.landing_exit(0, lander), land.EXIT_CHECKOUT)
        self.assertEqual(land.landing_exit(land.EXIT_RED, lander), land.EXIT_RED)   # a red main comes first
        self.assertEqual(land.landing_exit(0, self.r.lander()), 0)

    def test_an_evidence_branch_moved_after_the_evidence_is_refused(self):
        commit(self.r.work, {"src/b.rs": "fn b() {}\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        before = self.r.origin_head()
        upstream = self.r.upstream

        class Moved(FakeLander):
            def wait_evidence(self, ref, sha):
                super().wait_evidence(ref, sha)
                git(upstream, "push", "-q", "--force", "origin", f"{before}:refs/heads/{ref}")
        with self.assertRaisesRegex(land.LandError, "ci/T-1 is [0-9a-f]{7}, not the evidence [0-9a-f]{7}"):
            Moved(self.r.work, "T-1").land()
        self.assertEqual(self.r.origin_head(), before)

    def test_docs_only_moved_with_a_change_beyond_the_evidence_is_refused(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        before = self.r.origin_head()

        class Extra(FakeLander):
            # a step after the evidence that slips in a change of its own (a source file in an
            # existing commit: the commit count and the messages stay as they were)
            def prepare(self, verify):
                sha = super().prepare(verify)
                if self.calls.count("checks") >= 2:
                    write(self.root, "src/z.rs", "fn z() {}\n")
                    git(self.root, "add", "src/z.rs")
                    git(self.root, *ID, "commit", "-q", "--amend", "--no-edit")
                    sha = git(self.root, "rev-parse", "HEAD")
                return sha
        lander = Extra(self.r.work, "T-1")
        lander.moves = [lambda: self.r.push_upstream({"docs/oracle-status.md": "regenerated\n"}, "docs: oracle-status")]
        with self.assertRaisesRegex(land.LandError, "differs from the evidence [0-9a-f]{7} outside the paths main "
                                                    "gained and the ledgers: \\['src/z.rs'\\]"):
            lander.land()
        self.assertEqual(self.r.origin_head(), git(self.r.upstream, "rev-parse", "HEAD"))   # only main's own doc
        self.assertNotEqual(self.r.origin_head(), before)

    def test_docs_only_moved_with_the_patches_moved_between_commits_is_refused(self):
        # the same tree, the same commits and messages, but the change of one commit moved into
        # another: only the patch comparison sees it
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        commit(self.r.work, {"src/b.rs": "y\n", "docs/c.md": "c\n"}, "feat: b then c")

        class Shuffled(FakeLander):
            def prepare(self, verify):
                sha = super().prepare(verify)
                if self.calls.count("checks") >= 2:
                    base = git(self.root, "merge-base", "HEAD", self.upstream())
                    ours = git(self.root, "rev-list", "--reverse", f"{base}..HEAD").split()
                    git(self.root, "checkout", "-q", "--detach", base)
                    write(self.root, "src/b.rs", "y\n")
                    write(self.root, "CHANGELOG.md", git(self.root, "show", f"{ours[0]}:CHANGELOG.md") + "\n")
                    git(self.root, "add", "-A")
                    git(self.root, *ID, "commit", "-q", "-C", ours[0])
                    write(self.root, "docs/c.md", "c\n")
                    git(self.root, "add", "-A")
                    git(self.root, *ID, "commit", "-q", "-C", ours[1])
                    for c in ours[2:]:
                        git(self.root, *ID, "cherry-pick", "--allow-empty", c)
                    self.assertEqual(git(self.root, "rev-parse", "HEAD^{tree}"), git(self.root, "rev-parse", f"{sha}^{{tree}}"))
                    sha = git(self.root, "rev-parse", "HEAD")
                return sha
            assertEqual = unittest.TestCase().assertEqual
        lander = Shuffled(self.r.work, "T-1")
        lander.moves = [lambda: self.r.push_upstream({"docs/oracle-status.md": "regenerated\n"}, "docs: oracle-status")]
        with self.assertRaisesRegex(land.LandError, "does not carry the patches of the evidence [0-9a-f]{7} \\(2 commits vs 2\\)"):
            lander.land()

    def test_code_upstream_in_other_files_compiles_but_does_not_rerun_preflight(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander(before_push=lambda: self.r.push_upstream({"src/c.rs": "fn c() {}\n"}, "feat: c"))
        sha = lander.land()
        self.assertEqual(lander.preflight_runs, 1)
        self.assertIn("compile", lander.calls)
        self.assertEqual(lander.calls.count("wait_evidence ci/T-1"), 2)
        self.assertEqual(lander.evidenced[-1], sha)
        self.assertIn("reason: re-evidenced (evidence rounds 2)", lander.record)
        self.assertEqual([l for l in lander.record if l.startswith("evidence: ")],
                         ["evidence: ci.yml run 1 success"])  # only the runs of the last evidence SHA

    def test_upstream_touching_our_file_reruns_preflight(self):
        write(self.r.work, "src/b.rs", "fn b1() {}\n\n\n\n\nfn b2() {}\n")
        git(self.r.work, "add", "-A")
        git(self.r.work, "commit", "-q", "--author", f"{NAME} <{EMAIL}>", "-m", "feat: b base")
        git(self.r.work, "push", "-q", "origin", "HEAD:main")
        commit(self.r.work, {"src/b.rs": "fn b1() { mine }\n\n\n\n\nfn b2() {}\n",
                             "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b1")
        lander = self.r.lander(before_push=lambda: self.r.push_upstream(
            {"src/b.rs": "fn b1() {}\n\n\n\n\nfn b2() { theirs }\n"}, "feat: b2"))
        lander.land()
        self.assertEqual(lander.preflight_runs, 2)
        self.assertNotIn("compile", lander.calls)
        self.assertEqual(lander.calls.count("wait_evidence ci/T-1"), 2)

    def test_upstream_changing_the_build_reruns_preflight(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander(before_push=lambda: self.r.push_upstream({"Cargo.toml": "[package]\nname = \"x\"\n"},
                                                                         "build: deps"))
        lander.land()
        self.assertEqual(lander.preflight_runs, 2)
        self.assertEqual(lander.calls.count("wait_evidence ci/T-1"), 2)

    def test_a_snapshot_move_is_code_and_is_evidenced_again(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander(before_push=lambda: self.r.push_upstream({"docs/PUBLIC_API_SNAPSHOT.txt": "pub fn z\n"},
                                                                         "docs: snapshot"))
        lander.land()
        self.assertEqual(lander.calls.count("wait_evidence ci/T-1"), 2)
        self.assertIn("reason: re-evidenced (evidence rounds 2)", lander.record)

    def test_a_changelog_and_source_move_is_code(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander(before_push=lambda: self.r.push_upstream(
            {"CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- c\n", "src/c.rs": "fn c() {}\n"}, "feat: c"))
        lander.land()
        self.assertEqual(lander.calls.count("wait_evidence ci/T-1"), 2)

    def test_main_moving_by_code_three_times_stops(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")

        def move(n):
            return lambda: self.r.push_upstream({f"src/m{n}.rs": f"fn m{n}() {{}}\n"}, f"feat: m{n}")
        lander = self.r.lander(before_push=[move(1), move(2), move(3)])
        with self.assertRaisesRegex(land.LandError, "main is moving: wait for a quiet main or coordinate a landing slot"):
            lander.land()
        self.assertEqual(lander.calls.count("wait_evidence ci/T-1"), land.MAX_EVIDENCE)
        subjects = subprocess.run(["git", "log", "--format=%s", "main"], cwd=self.r.origin, capture_output=True,
                                  text=True).stdout.splitlines()
        self.assertEqual(subjects[0], "feat: m3")
        self.assertNotIn("feat: b", subjects)

    def test_unreadable_workflow_inputs_treat_a_docs_move_as_code(self):
        (self.r.work / ".github/paths/w.txt").unlink()
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander(before_push=lambda: self.r.push_upstream({"docs/ROADMAP.md": "x\n"}, "docs: roadmap"))
        lander.land()
        self.assertEqual(lander.calls.count("wait_evidence ci/T-1"), 2)

    def test_two_code_moves_still_land(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")

        def move(n):
            return lambda: self.r.push_upstream({f"src/m{n}.rs": f"fn m{n}() {{}}\n"}, f"feat: m{n}")
        lander = self.r.lander(before_push=[move(1), move(2)])
        sha = lander.land()
        self.assertEqual(self.r.origin_head(), sha)
        self.assertIn("reason: re-evidenced (evidence rounds 3)", lander.record)
        self.assertEqual(lander.evidenced[-1], sha)

    def test_lock_contention_waits_then_times_out(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander()
        lander.lock_wait_seconds = 0.3
        held = lander.lock()
        with held:
            before = self.r.origin_head()
            start = time.monotonic()
            with self.assertRaisesRegex(land.LandError, rf"another landing holds .*pid {os.getpid()} started "):
                self.r.lander(lock_wait=0.3).land()
            self.assertGreaterEqual(time.monotonic() - start, 0.3)
            self.assertEqual(self.r.origin_head(), before)
        self.assertFalse(held.holder.exists(), "the holder note goes with the lock")

    def test_lock_contention_waits_for_the_holder(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        import threading
        held = self.r.lander().lock()
        held.__enter__()
        t = threading.Timer(0.4, held.__exit__, (None, None, None))
        t.start()
        start = time.monotonic()
        sha = self.r.lander(lock_wait=10).land()
        t.join()
        self.assertGreaterEqual(time.monotonic() - start, 0.3)
        self.assertEqual(self.r.origin_head(), sha)

    def test_the_lock_is_per_repository_in_the_configured_dir(self):
        lock = self.r.lander().lock()
        self.assertEqual(lock.path.parent, Path(os.environ[land.LOCK_DIR_ENV]))
        self.assertTrue(lock.path.name.endswith(".lock"))
        self.assertIn("origin.git", lock.path.name)

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

    def test_an_ambient_git_env_does_not_change_the_landed_identity(self):
        # land.py's own git processes (the rebase included) under a shell that exports
        # another identity, another repository and injected config: the landed commits still
        # carry the landing identity, and this checkout (not `elsewhere`) is landed
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        git(self.r.work, "-c", "user.email=me@machine.local", "commit", "-q", "--amend", "--no-edit")  # forces a rebase
        elsewhere = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", str(elsewhere)], check=True)
        self.addCleanup(shutil.rmtree, elsewhere, True)
        marker = hostile_hooks(elsewhere)
        hostile = hostile_env(elsewhere)
        del hostile["LAND_IDENTITY"]  # read at import; the refusal of a denied override is tested in Identity
        os.environ.update(hostile)
        lander = self.r.lander()
        sha = lander.land()
        for k in hostile:
            del os.environ[k]
        self.assertIn("rebase", lander.calls)
        self.assertEqual(self.r.origin_head(), sha)
        people = git(self.r.work, "log", "-3", "--format=%an <%ae>|%cn <%ce>|%cd", "--date=format:%Y").splitlines()
        for line in people:
            author, committer, year = line.split("|")
            self.assertEqual(committer, DEFAULT_IDENTITY, people)
            self.assertNotEqual(year, "2001", people)
        self.assertEqual(git(self.r.work, "log", "-1", "--format=%an <%ae>", "HEAD~1"), DEFAULT_IDENTITY)
        self.assertEqual(subprocess.run(["git", "-C", str(elsewhere), "rev-list", "--all"], capture_output=True,
                                        text=True).stdout, "", "nothing was written to the other repository")
        self.assertFalse(marker.exists(), "no injected hook ran")

    def test_an_ambient_git_env_does_not_force_a_rebase(self):
        # already on main with the landing identity: the evidenced SHA lands unchanged
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        mine = git(self.r.work, "rev-parse", "HEAD")
        elsewhere = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", str(elsewhere)], check=True)
        self.addCleanup(shutil.rmtree, elsewhere, True)
        hostile = hostile_env(elsewhere)
        del hostile["LAND_IDENTITY"]
        os.environ.update(hostile)
        lander = self.r.lander()
        sha = lander.land()
        for k in hostile:
            del os.environ[k]
        self.assertNotIn("rebase", lander.calls)
        self.assertEqual(sha, lander.evidenced[0])
        self.assertEqual(git(self.r.work, "rev-parse", "HEAD~1"), mine)  # ours, then the ledger commit

    def test_the_landed_identity_is_read_back_from_the_remote(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        lander = self.r.lander()
        sha = lander.land()
        lines = [l for l in lander.record if l.startswith("landed identity: ")]
        landed = git(self.r.work, "rev-list", "--reverse", f"{sha}~2..{sha}").split()
        self.assertEqual(len(lines), len(landed) + 1, lines)  # every landed commit, plus the API's tip
        for c in landed:
            self.assertIn(f"landed identity: {c[:7]} author {DEFAULT_IDENTITY} committer {DEFAULT_IDENTITY} (remote)", lines)
        self.assertIn(f"landed identity: {sha[:7]} author {DEFAULT_IDENTITY} committer {DEFAULT_IDENTITY} (GitHub API)", lines)
        self.assertIn("api", lander.calls)

    def test_a_wrong_identity_from_the_api_says_revert_or_fix_now(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")

        class Wrong(FakeLander):
            def api_commit_identity(self, sha):
                return (NAME, EMAIL, "GitHub", "noreply@github.com")
        lander = Wrong(self.r.work, "T-1")
        with self.assertRaisesRegex(land.LandError, "REVERT OR FIX NOW: landed commit identity on main .*\n  .* committer "
                                                    "`GitHub <noreply@github.com>` \\(GitHub API\\)"):
            lander.land()
        self.assertTrue(any(l.endswith("committer GitHub <noreply@github.com> (GitHub API)") for l in lander.record))

    def test_a_wrong_identity_read_by_git_says_revert_or_fix_now(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")

        class Wrong(FakeLander):
            # the remote reads a different committer than the local check saw (the backstop)
            pushed = False

            def check_push_identity(self, before):
                super().check_push_identity(before)
                self.pushed = True

            def git(self, *a, check=True):
                out = super().git(*a, check=check)
                if self.pushed and a[:3] == ("log", "-1", "--format=%an%x00%ae%x00%cn%x00%ce") \
                        and self.calls[-1:] != ["api"]:
                    return out.rsplit("\0", 2)[0] + "\0Someone Else\0someone@else.invalid"
                return out
        with self.assertRaisesRegex(land.LandError, "(?s)REVERT OR FIX NOW: landed commit identity.*committer "
                                                    "`Someone Else <someone@else.invalid>` \\(remote\\): not.*"
                                                    "fix forward with a new commit.*do not rewrite"):
            Wrong(self.r.work, "T-1").land()

    def test_a_wrong_identity_before_the_push_refuses_to_push(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        before = self.r.origin_head()
        for bad in ("Someone Else\0someone@else.invalid", "web-flow\0x@y.invalid"):
            class Wrong(FakeLander):
                def check_push_identity(self, before_push):
                    real = self.git

                    def fake(*a, check=True):
                        out = real(*a, check=check)
                        if a[:3] == ("log", "-1", "--format=%an%x00%ae%x00%cn%x00%ce"):
                            return out.rsplit("\0", 2)[0] + "\0" + bad
                        return out
                    self.git = fake
                    try:
                        super().check_push_identity(before_push)
                    finally:
                        self.git = real
            with self.assertRaisesRegex(land.LandError, "(?s)refusing to push to main: commit identity.*committer.*\\(local\\)"):
                Wrong(self.r.work, "T-1").land()
            self.assertEqual(self.r.origin_head(), before, "nothing reached main")

    def test_the_local_identity_check_reads_the_pushed_range(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        seen = []

        class Spy(FakeLander):
            def identity_rows(self, rev_range, source):
                rows = super().identity_rows(rev_range, source)
                seen.append((source, len(rows)))
                return rows
        sha = Spy(self.r.work, "T-1").land()
        self.assertEqual(seen[0], ("local", 2))   # ours and the ledger commit
        self.assertEqual(seen[1], ("remote", 2))
        self.assertEqual(self.r.origin_head(), sha)

    def test_an_unreadable_api_commit_says_revert_or_fix_now(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")

        class NoApi(FakeLander):
            def api_commit_identity(self, sha):
                raise land.LandError("HTTP 404")
        with self.assertRaisesRegex(land.LandError, "(?s)REVERT OR FIX NOW: landed commit identity.*unreadable: HTTP 404"):
            NoApi(self.r.work, "T-1").land()

    def test_landed_identity_problems(self):
        ok = ("a" * 40, NAME, EMAIL, NAME, EMAIL, "remote")
        self.assertEqual(land.landed_identity_problems([ok], DEFAULT_IDENTITY), [])
        self.assertIn("compared nothing", land.landed_identity_problems([], DEFAULT_IDENTITY)[0])
        for i, field in ((1, "author"), (3, "committer")):
            row = list(ok)
            row[i] = "Other"
            p = land.landed_identity_problems([tuple(row)], DEFAULT_IDENTITY)
            self.assertEqual(len(p), 1, p)
            self.assertIn(f"aaaaaaa {field} `Other <{EMAIL}>` (remote): not", p[0])
        # a denied identity is named as such even when LAND_IDENTITY were that identity
        p = land.landed_identity_problems([("b" * 40, "web-flow", "x@y", "web-flow", "x@y", "GitHub API")], "web-flow <x@y>")
        self.assertEqual(len(p), 2)
        self.assertIn("is a GitHub identity", p[0])

    def test_the_api_reader_parses_gh_output(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        (d / "gh.py").write_text(FAKE_GH, encoding="utf-8")
        state = d / "state.json"
        state.write_text(json.dumps({"calls": [], "api": {"repos/o/r/commits/abc": {"commit": {
            "author": {"name": NAME, "email": EMAIL}, "committer": {"name": "GitHub", "email": "noreply@github.com"}}}}}))
        os.environ["FAKE_GH_STATE"] = str(state)
        self.addCleanup(os.environ.pop, "FAKE_GH_STATE", None)

        class L(land.Lander):
            def slug(self):
                return "o/r"
        lnd = L(self.r.work, "T", log=lambda *_: None)
        lnd.gh_cmd = [sys.executable, str(d / "gh.py")]
        self.assertEqual(lnd.api_commit_identity("abc"), (NAME, EMAIL, "GitHub", "noreply@github.com"))
        with self.assertRaisesRegex(land.LandError, "gh api --hostname github.com repos/o/r/commits/def failed"):
            lnd.api_commit_identity("def")
        state.write_text(json.dumps({"calls": [], "api": {"repos/o/r/commits/abc": {"sha": "abc"}}}))
        with self.assertRaisesRegex(land.LandError, "no author / committer"):
            lnd.api_commit_identity("abc")

    def test_a_repository_chosen_by_the_environment_is_refused(self):
        import contextlib
        import io
        for var in land.REFUSED_ENV:
            os.environ[var] = str(self.r.upstream)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = land.main(["--id", "x", "--dry-run"])
            del os.environ[var]
            self.assertEqual(rc, 2, var)
            self.assertIn(f"error: {var} set in the environment", err.getvalue())
        os.environ["GIT_DIR"], os.environ["GIT_NAMESPACE"] = "/x", "n"
        self.assertEqual(land.refused_env(), ["GIT_DIR", "GIT_NAMESPACE"])
        del os.environ["GIT_DIR"], os.environ["GIT_NAMESPACE"]
        # the identity / config / ceiling variables are not refused (they are scrubbed)
        for var in ("GIT_COMMITTER_NAME", "GIT_AUTHOR_EMAIL", "GIT_CONFIG_COUNT", "GIT_CEILING_DIRECTORIES"):
            os.environ[var] = "x"
        self.assertEqual(land.refused_env(), [])

    def test_child_processes_get_the_clean_env(self):
        elsewhere = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, elsewhere, True)
        os.environ.update(hostile_env(elsewhere))
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        probe = ("import os, re, sys; bad = [k for k in os.environ if re.match("
                 "r'GIT_(AUTHOR|COMMITTER)_|GIT_DIR$|GIT_WORK_TREE$|GIT_INDEX_FILE$|GIT_CONFIG_(COUNT|KEY|VALUE|PARAMETERS)', k)]; "
                 "sys.exit(1 if bad else 0)")
        lander.sh(sys.executable, "-c", probe)  # LandError if the child saw any of them
        # nothing ambient that the scrub matches; only the fixed values land.py sets itself
        self.assertEqual(sorted(k for k in lander.git_env if land.SCRUBBED_ENV_RE.match(k)
                                and lander.git_env[k] != land.CHILD_PYTHON_ENV.get(k)), [])

    def test_landed_commits_carry_the_fixed_committer(self):
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")
        git(self.r.work, "-c", "user.email=me@machine.local", "commit", "-q", "--amend", "--no-edit")
        # main has not moved: a plain rebase would keep the commit (and its committer) as is
        lander = self.r.lander()
        lander.land()
        self.assertIn("rebase", lander.calls)
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


FAKE_GH = r"""import json, os, sys
path = os.environ["FAKE_GH_STATE"]
st = json.load(open(path, encoding="utf-8"))
a = sys.argv[1:]
st["calls"].append(a)
# what gh would read from its environment
st.setdefault("envs", []).append({k: v for k, v in os.environ.items()
                                  if k.startswith(("GH_", "GITHUB_")) or "PROXY" in k.upper() or k.startswith("SSL_")})
rc = 0
if a[:2] == ["auth", "status"]:
    ok = st.get("auth", True)
    print("github.com\n  Logged in to github.com account someone" if ok else "github.com\n  X Failed to log in",
          file=sys.stderr)
    rc = 0 if ok else 1
elif a[:2] == ["run", "list"]:
    i = min(st["polls"], len(st["list"]) - 1)
    st["polls"] += 1
    print(json.dumps(st["list"][i] + st["added"], ensure_ascii=True))
elif a[:2] == ["run", "view"]:
    print(json.dumps({"jobs": st["jobs"].get(a[2], [])}, ensure_ascii=True))
elif a[:1] == ["api"]:
    path_arg = [x for i, x in enumerate(a[1:], 1) if x != "--hostname" and a[i - 1] != "--hostname"][0]
    if path_arg in st.get("api", {}):
        print(json.dumps(st["api"][path_arg]))
    else:
        print("HTTP 404", file=sys.stderr)
        rc = 1
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


class WaitEvidence(Isolated):
    """wait_evidence against a fake `gh` on PATH (no network)."""

    def setUp(self):
        super().setUp()
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
                {"name": "Determinism tests (release, macos-latest)", "status": leg_status, "conclusion": leg_concl},
                {"name": "Clippy", "status": "completed", "conclusion": "success"}]

    def miri_jobs(self, heavy="in_progress", concl="") -> list[dict]:
        return [{"name": "Changed paths", "status": "completed", "conclusion": "success"},
                {"name": "loom (batch colouring protocol, --cfg loom)", "status": heavy, "conclusion": concl},
                {"name": "Miri (UB 検出、parallel feature)", "status": heavy, "conclusion": concl},
                {"name": "Gate", "status": "pending", "conclusion": ""}]

    def wait(self, lists, jobs=None, on_dispatch=None, at=SHA, root=None):
        root = root or self.wf_root()
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
        return lnd

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
        # the Intel leg is not expected on ci/**
        self.assertEqual(ci["legs"], ["Test (macos-latest)", "Determinism tests (release, macos-latest)"])
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

    def test_the_record_lists_each_run_relied_on(self):
        lnd = self.wait([self.green()])
        lines = [l for l in lnd.record if l.startswith("evidence: ")]
        self.assertEqual(len(lines), len(self.expected))
        self.assertIn(f"evidence: security-audit.yml run {self.rid('security-audit.yml')} success", lines)
        self.assertIn(f"evidence: ci.yml run {self.rid('ci.yml')} in_progress", lines)

    def test_a_newer_red_run_beats_an_older_green_one(self):
        for f in ("security-audit.yml", "bench-gate.yml"):
            runs = self.green()
            i = self.rid(f) - 1
            runs.append(dict(runs[i], databaseId=500, conclusion="failure", event="workflow_dispatch"))
            with self.assertRaisesRegex(land.LandError, f"is red for .*\\n  {f}: run 500 ended failure"):
                self.wait([runs])

    def test_a_push_that_did_not_reach_the_ref_fails_at_once(self):
        with self.assertRaisesRegex(land.LandError, "the push did not reach it"):
            self.wait([[]], at="d" * 40)
        self.assertEqual(self.st()["polls"], 1)

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
        self.assertIn(["workflow", "run", "fuzz.yml", "--repo", "github.com/o/r", "--ref", "ci/T"], self.st()["calls"])
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
        lnd = L(self.wf_root(), "T", log=lambda *_: None)
        lnd.gh_cmd = [sys.executable, str(self.gh)]
        lnd.poll_seconds = 0
        with self.assertRaisesRegex(land.LandError, "gh run list --repo github.com/o/r failed"):
            lnd.wait_evidence("ci/T", SHA)


class WaitAfter(Isolated):
    """wait_after on main against a fake `gh` (no network)."""

    def setUp(self):
        super().setUp()
        self.dir = Path(tempfile.mkdtemp())
        self.gh = self.dir / "gh.py"
        self.gh.write_text(FAKE_GH, encoding="utf-8")
        self.state = self.dir / "state.json"
        self.old = os.environ.get("FAKE_GH_STATE")
        os.environ["FAKE_GH_STATE"] = str(self.state)

    def tearDown(self):
        if self.old is None:
            os.environ.pop("FAKE_GH_STATE", None)
        else:
            os.environ["FAKE_GH_STATE"] = self.old
        shutil.rmtree(self.dir, ignore_errors=True)

    def st(self) -> dict:
        return json.loads(self.state.read_text())

    @staticmethod
    def job(name, status="completed", concl="success"):
        return {"name": name, "status": status, "conclusion": concl}

    def ci(self, mac="success", intel="success", mac_status="completed", intel_status="completed",
           rel="success", rel_status="completed"):
        return [self.job("Test (ubuntu-latest)"), self.job("Test (macos-latest)", mac_status, mac),
                self.job("Determinism tests (release, macos-latest)", rel_status, rel),
                self.job("Test (macos-15-intel)", intel_status, intel)]

    def miri(self, heavy="success", status="completed", changes="success"):
        return [self.job("Changed paths", "completed", changes),
                self.job("loom (batch colouring protocol, --cfg loom)", status, heavy),
                self.job("Miri (UB 検出、parallel feature)", status, heavy), self.job("Gate")]

    def go(self, ci_jobs=None, miri_jobs=None, runs=None, extra_jobs=None, polls=3):
        runs = runs if runs is not None else [ok_run("CI", 1), ok_run("Miri and loom (changed unsafe or parallel code)", 2)]
        jobs = {"1": ci_jobs if ci_jobs is not None else self.ci(), "2": miri_jobs if miri_jobs is not None else self.miri()}
        jobs.update(extra_jobs or {})
        self.state.write_text(json.dumps({"list": [runs], "jobs": jobs, "on_dispatch": {}, "added": [],
                                          "dispatched": [], "calls": [], "polls": 0}))

        class L(land.Lander):
            def git(self, *a, check=True):
                return "git@github.com:o/r.git"
        lnd = L(self.wf_root(), "T", log=lambda *_: None)
        lnd.gh_cmd = [sys.executable, str(self.gh)]
        lnd.poll_seconds = 0
        lnd.after_max_polls = polls
        return lnd.wait_after(SHA)

    def test_the_items_are_derived(self):
        labels = [land.after_item_label(i) for i in land.after_landing_items(ROOT)]
        self.assertEqual(labels, ["ci.yml: Test (macos-latest)", "ci.yml: Test (macos-15-intel)",
                                  "ci.yml: Determinism tests (release, macos-latest)", "unsafe-and-parallel.yml"])

    def test_all_green(self):
        code, lines = self.go()
        self.assertEqual(code, 0, lines)
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(l.startswith("ok: ") for l in lines), lines)

    def test_miri_that_did_not_run_is_reported(self):
        code, lines = self.go(miri_jobs=self.miri(heavy="skipped"))
        self.assertEqual(code, 0)
        self.assertIn("ok: unsafe-and-parallel.yml: run 2: did not run (the changes job said run=false)", lines)

    def test_a_red_leg_says_revert_or_fix_now(self):
        for kw, label in (({"mac": "failure"}, "Test (macos-latest)"), ({"intel": "failure"}, "Test (macos-15-intel)"),
                          ({"rel": "failure"}, "Determinism tests (release, macos-latest)")):
            code, lines = self.go(ci_jobs=self.ci(**kw))
            self.assertEqual(code, land.EXIT_RED)
            red = [l for l in lines if l.startswith("REVERT OR FIX NOW")]
            self.assertEqual(len(red), 1, lines)
            self.assertIn(label, red[0])
            self.assertIn("run 1", red[0])

    def test_red_miri_says_revert_or_fix_now(self):
        code, lines = self.go(miri_jobs=self.miri(heavy="failure"))
        self.assertEqual(code, land.EXIT_RED)
        self.assertTrue(any(l.startswith("REVERT OR FIX NOW: unsafe-and-parallel.yml") and "run 2" in l for l in lines), lines)

    def test_red_changes_job_is_red(self):
        code, _ = self.go(miri_jobs=self.miri(changes="failure"))
        self.assertEqual(code, land.EXIT_RED)

    def test_unfinished_is_pending_with_ids(self):
        code, lines = self.go(ci_jobs=self.ci(intel="", intel_status="in_progress"))
        self.assertEqual(code, land.EXIT_PENDING)
        self.assertNotEqual(land.EXIT_PENDING, land.EXIT_RED)
        self.assertIn("pending: ci.yml: Test (macos-15-intel): job 'Test (macos-15-intel)' in_progress in run 1", lines)
        self.assertEqual(self.st()["polls"], 3)  # polled up to the bound

    def test_miri_not_finished_is_pending(self):
        code, lines = self.go(miri_jobs=self.miri(heavy="", status="in_progress"))
        self.assertEqual(code, land.EXIT_PENDING)

    def test_no_run_is_pending_not_green(self):
        code, lines = self.go(runs=[ok_run("CI", 1)])
        self.assertEqual(code, land.EXIT_PENDING)
        self.assertIn("pending: unsafe-and-parallel.yml: no run for the SHA yet", lines)

    def test_a_missing_leg_job_is_pending_not_green(self):
        code, lines = self.go(ci_jobs=self.ci()[:2])
        self.assertEqual(code, land.EXIT_PENDING)
        self.assertTrue(any("no job 'Test (macos-15-intel)'" in l for l in lines), lines)

    def test_a_cancelled_leg_is_pending_not_green(self):
        code, _ = self.go(ci_jobs=self.ci(intel="cancelled"))
        self.assertEqual(code, land.EXIT_PENDING)

    def test_the_nightly_re_run_of_the_intel_leg_counts(self):
        # the push run's Intel leg was cancelled; the nightly run (newer id) passed it
        runs = [ok_run("CI", 1), ok_run("CI", 3, event="schedule"), ok_run("Miri and loom (changed unsafe or parallel code)", 2)]
        code, lines = self.go(ci_jobs=self.ci(intel="cancelled"), runs=runs,
                              extra_jobs={"3": [self.job("Test (macos-15-intel)")]})
        self.assertEqual(code, 0, lines)
        self.assertIn("ok: ci.yml: Test (macos-15-intel): job 'Test (macos-15-intel)' success in run 3", lines)

    def test_the_newest_leg_result_counts(self):
        runs = [ok_run("CI", 1), ok_run("CI", 3, event="schedule"), ok_run("Miri and loom (changed unsafe or parallel code)", 2)]
        code, _ = self.go(runs=runs, extra_jobs={"3": [self.job("Test (macos-15-intel)", "completed", "failure")]})
        self.assertEqual(code, land.EXIT_RED)

    def test_runs_of_another_sha_do_not_count(self):
        runs = [dict(ok_run("CI", 1), headSha="e" * 40), ok_run("Miri and loom (changed unsafe or parallel code)", 2)]
        code, _ = self.go(runs=runs)
        self.assertEqual(code, land.EXIT_PENDING)

    def test_red_stops_polling_early(self):
        self.go(ci_jobs=self.ci(mac="failure"), polls=5)
        self.assertEqual(self.st()["polls"], 1)

    def test_miri_changes_job_running_is_pending(self):
        miri = self.miri(heavy="", status="queued")
        miri[0].update(status="in_progress", conclusion="")
        code, lines = self.go(miri_jobs=miri)
        self.assertEqual(code, land.EXIT_PENDING)
        self.assertIn("pending: unsafe-and-parallel.yml: run 2: the changes job has not finished", lines)

    def test_a_cancelled_miri_run_is_pending_not_green(self):
        runs = [ok_run("CI", 1), ok_run("Miri and loom (changed unsafe or parallel code)", 2, conclusion="cancelled")]
        code, lines = self.go(runs=runs)
        self.assertEqual(code, land.EXIT_PENDING)
        self.assertTrue(any("ended cancelled (no result; re-run it)" in l for l in lines), lines)

    def test_an_empty_after_landing_set_fails(self):
        d = self.dir / "tree"
        (d / ".github" / "workflows").mkdir(parents=True)
        (d / ".github" / "workflows" / "w.yml").write_text("name: W\non:\n  push:\n    branches: [main]\njobs:\n  a: {}\n")
        (d / ".github" / "ci-evidence.toml").write_text('[workflows]\n"w.yml" = "landing"\n')
        with self.assertRaisesRegex(land.LandError, "compared nothing"):
            land.after_landing_items(d)

    def test_miri_partly_skipped_is_red(self):
        miri = self.miri()
        miri[1].update(conclusion="skipped")
        code, lines = self.go(miri_jobs=miri)
        self.assertEqual(code, land.EXIT_RED)
        self.assertTrue(any("heavy jobs partly skipped" in l for l in lines), lines)

    def test_the_cli_exit_codes(self):
        class L(land.Lander):
            def __init__(s, code):
                s.code, s.id, s.branch = code, "T", "main"

            def wait_after(s, sha):
                return s.code, ["x"]
        self.assertEqual(land.report_after(L(0), SHA), 0)
        self.assertEqual(land.report_after(L(land.EXIT_RED), SHA), land.EXIT_RED)
        self.assertEqual(land.report_after(L(land.EXIT_PENDING), SHA), land.EXIT_PENDING)


SUBPROCESS_FUNCS = {"run", "Popen", "check_output", "call", "check_call", "getoutput", "getstatusoutput"}


def call_sites(source: str) -> tuple[int, list[str]]:
    """(call sites, unwrapped): every subprocess.* / os.system / os.popen / os.exec* / os.spawn*
    call in `source`; a site is wrapped when it passes env= built by clean_env(): `clean_env()`,
    `self.git_env`, or a dict that spreads `**clean_env()`. os.system and os.popen cannot take
    an env, so they are always unwrapped"""
    import ast
    tree = ast.parse(source)
    sites, bad = 0, []

    def built_by_clean_env(v) -> bool:
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("clean_env", "gh_env"):
            return True
        # Lander.child_env() returns {**clean_env(cargo=True), ...} (checked in test_every_call_site_is_wrapped)
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "child_env" \
                and isinstance(v.func.value, ast.Name) and v.func.value.id in ("self", "lander"):
            return True
        if isinstance(v, ast.Attribute) and v.attr == "git_env" and isinstance(v.value, ast.Name) and v.value.id == "self":
            return True
        if isinstance(v, ast.Dict):
            return any(k is None and built_by_clean_env(val) for k, val in zip(v.keys, v.values))
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            if mod == "subprocess" or (mod == "os" and any(a.name in ("system", "popen") for a in node.names)):
                bad.append(f"line {node.lineno}: `from {mod} import ...` hides call sites from this check")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value.id if isinstance(node.func.value, ast.Name) else None
        name = node.func.attr
        if owner == "subprocess" and name in SUBPROCESS_FUNCS:
            sites += 1
            env = [k.value for k in node.keywords if k.arg == "env"]
            if not env or not built_by_clean_env(env[0]):
                bad.append(f"line {node.lineno}: subprocess.{name} without env= from clean_env()")
        elif owner == "os" and (name in ("system", "popen") or name.startswith(("exec", "spawn"))):
            sites += 1
            bad.append(f"line {node.lineno}: os.{name} (cannot pass the clean environment)")
    return sites, bad


def git_call_sites(source: str) -> tuple[int, list[str]]:
    """(git / gh calls, unpinned): a subprocess call whose argument list starts with the
    pinned git and forced config (`*self.git_argv` / `*git_argv()`) or the pinned gh
    (`*self.gh()`) is pinned; one that starts with anything else naming git or gh, or with an
    expression that is neither of those, counts as an unpinned git / gh call"""
    import ast
    n, bad = 0, []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess"):
            continue
        if not node.args or not isinstance(node.args[0], ast.List) or not node.args[0].elts:
            continue
        first = node.args[0].elts[0]
        src = ast.unparse(first)
        if src in ("*self.git_argv", "*git_argv()", "*self.gh()"):
            n += 1
            env = [ast.unparse(k.value) for k in node.keywords if k.arg == "env"]
            if src == "*self.gh()" and env != ["gh_env()"]:
                bad.append(f"line {node.lineno}: gh without env=gh_env()")
        elif "git" in src.lower() or "gh" in src.lower() or "GIT" in src:
            n += 1
            bad.append(f"line {node.lineno}: `{src}` is not the pinned git / gh with the forced config")
    return n, bad


class CallSites(Isolated):
    """Every process land.py starts gets the clean environment (a static check of the source)."""

    SOURCE = (Path(__file__).resolve().parent / "land.py").read_text(encoding="utf-8")

    def test_every_git_call_is_pinned(self):
        n, bad = git_call_sites(self.SOURCE)
        print(f"git calls: {n}, pinned: {n - len(bad)}", file=sys.stderr)
        self.assertGreater(n, 0, "no git call found (compared nothing)")
        self.assertEqual(bad, [])
        self.assertIn("self.git_argv = git_argv()", self.SOURCE)
        for snippet in ("subprocess.run(['git', 'push'], env=clean_env())",
                        "subprocess.run(['gh', 'api'], env=clean_env())",
                        "subprocess.run([GIT, 'push'], env=clean_env())",
                        "subprocess.run([pinned_git(), 'push'], env=clean_env())",
                        "subprocess.run([*self.gh(), 'api', 'x'], env=clean_env())"):
            m, b = git_call_sites(self.SOURCE + "\n\n" + snippet + "\n")
            self.assertEqual((m, len(b)), (n + 1, 1), snippet)
        self.assertEqual(git_call_sites("x = 1\n"), (0, []))

    def test_every_call_site_is_wrapped(self):
        n, bad = call_sites(self.SOURCE)
        print(f"call sites: {n}, wrapped: {n - len([b for b in bad if 'line' in b])}", file=sys.stderr)
        self.assertGreater(n, 0, "no call site found (compared nothing)")
        self.assertEqual(bad, [])
        self.assertIn("self.git_env = clean_env()", self.SOURCE)
        self.assertIn('env = {**clean_env(cargo=True), "PYTHONIOENCODING": "utf-8"}', self.SOURCE)

    def test_a_bare_call_is_found(self):
        for snippet in ("subprocess.run(['git', 'status'])", "subprocess.run(['x'], env=os.environ)",
                        "subprocess.check_output(['x'], env={**os.environ})", "os.system('git status')",
                        "subprocess.Popen(['x'], env=dict(clean_env))"):
            n, bad = call_sites(self.SOURCE + "\n\n" + snippet + "\n")
            self.assertEqual(n, call_sites(self.SOURCE)[0] + 1, snippet)
            self.assertEqual(len(bad), 1, (snippet, bad))

    def test_wrapped_forms(self):
        for snippet in ("subprocess.run(['x'], env=clean_env())", "subprocess.run(['x'], env={**clean_env(), 'A': '1'})",
                        "subprocess.run(['x'], env=self.git_env)"):
            self.assertEqual(call_sites(snippet), (1, []), snippet)

    def test_no_call_site_and_hidden_imports(self):
        self.assertEqual(call_sites("x = 1\n"), (0, []))
        self.assertTrue(call_sites("from subprocess import run\nrun(['x'])\n")[1])


class Output(Isolated):
    """land.py's lines reach a redirected stdout while it runs, not at exit."""

    def line_before_exit(self, code: str) -> str:
        """run `code` (after `import land`) with stdout redirected to a file; return what the
        file holds while the process is still running (it sleeps after writing)"""
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        out = d / "out.txt"
        script = f"import sys, time; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); import land\n{code}\ntime.sleep(30)\n"
        with open(out, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([sys.executable, "-c", script], stdout=fh, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 20
            text = ""
            while time.monotonic() < deadline:
                text = out.read_text(encoding="utf-8")
                if "MARKER" in text:
                    break
                time.sleep(0.1)
            self.assertIsNone(proc.poll(), "the process is still running")
            return text
        finally:
            proc.kill()
            proc.wait()

    def test_the_default_log_flushes(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        text = self.line_before_exit(f"land.Lander(land.Path({str(root)!r}), 'T').log('MARKER from log')")
        self.assertIn("MARKER from log", text)

    def test_main_makes_plain_prints_line_buffered(self):
        text = self.line_before_exit("land.line_buffered_output(); print('MARKER from print')")
        self.assertIn("MARKER from print", text)

    def test_without_flushing_nothing_arrives(self):
        # the control: a plain print to a file stays in the buffer while the process runs
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        out = d / "out.txt"
        with open(out, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen([sys.executable, "-c", "import time; print('MARKER unflushed'); time.sleep(30)"],
                                    stdout=fh)
        try:
            time.sleep(1.5)
            self.assertIsNone(proc.poll())
            self.assertNotIn("MARKER", out.read_text(encoding="utf-8"))
        finally:
            proc.kill()
            proc.wait()

    def test_every_print_flushes(self):
        import ast
        tree = ast.parse(CallSites.SOURCE)
        prints = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                  and c.func.id == "print"]
        self.assertGreater(len(prints), 0)
        self.assertEqual([c.lineno for c in prints if not any(k.arg == "flush" for k in c.keywords)], [])
        self.assertIn("log=say", CallSites.SOURCE)


def marker_script(path: Path, marker: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho \"$0 $*\" >> '{marker.as_posix()}'\nexit 1\n", encoding="utf-8")
    path.chmod(0o755)


class HostileConfig(Isolated):
    """A same-user environment that differs (by accident or on purpose) must not change what
    land.py pushes, where, or which programs run."""

    def setUp(self):
        super().setUp()
        self.d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, True)
        self.marker = self.d / "marker"
        commit(self.r.work, {"src/b.rs": "x\n", "CHANGELOG.md": "# Changelog\n\n## [Unreleased]\n\n- start\n- b\n"}, "feat: b")

    def home_with(self, settings: list[tuple[str, str]]) -> dict[str, str]:
        """a HOME / XDG_CONFIG_HOME whose global git config holds `settings`, written by git
        itself (`git config --file`), so paths with backslashes are quoted correctly"""
        home = self.d / "home"
        (home / ".config" / "git").mkdir(parents=True, exist_ok=True)
        for f in (home / ".gitconfig", home / ".config" / "git" / "config"):
            f.write_text("", encoding="utf-8")
            for key, value in settings:
                subprocess.run([land.pinned_git(), "config", "--file", str(f), "--add", key, value], check=True)
        return {"HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config")}

    def land_with(self, env: dict[str, str], lander=None):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            lander = lander or self.r.lander()
            return lander, lander.land()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def evil_copy(self) -> Path:
        evil = self.d / "evil.git"
        subprocess.run(["git", "clone", "-q", "--bare", str(self.r.origin), str(evil)], check=True, capture_output=True)
        return evil

    def heads(self, repo: Path) -> str:
        return subprocess.run(["git", "--git-dir", str(repo), "show-ref"], capture_output=True, text=True).stdout

    def test_insteadof_and_pushinsteadof_to_another_repository_are_refused(self):
        evil = self.evil_copy()
        origin = git(self.r.work, "config", "--get", "remote.origin.url")   # as the clone wrote it
        for key in ("insteadOf", "pushInsteadOf"):
            before_evil, before_origin = self.heads(evil), self.r.origin_head()
            env = self.home_with([(f"url.{evil.as_posix()}.{key}", origin)])
            with self.assertRaisesRegex(land.LandError, rf"refusing remote 'origin': its effective (push|fetch) URL "
                                                        rf"'{re.escape(evil.as_posix())}' is not the repository's remote"):
                # the config really redirects (a fixture that git cannot parse would prove nothing)
                self.assertEqual(subprocess.run([land.pinned_git(), "-C", str(self.r.work), "remote", "get-url", "--push",
                                                 "origin"], env={**os.environ, **env}, capture_output=True,
                                                text=True).stdout.strip(), evil.as_posix())
                self.land_with(env)
            self.assertEqual(self.heads(evil), before_evil, f"{key}: nothing reached the other repository")
            self.assertEqual(self.r.origin_head(), before_origin)

    def test_hooks_from_the_global_config_do_not_run(self):
        hooks = self.d / "hooks"
        # client-side hooks (the receiving side of these tests is a local bare repository, whose
        # own hooks are the server's business, as on GitHub)
        for name in ("pre-push", "post-commit", "post-rewrite", "pre-commit"):
            marker_script(hooks / name, self.marker)
        git(self.r.work, "-c", "user.email=me@machine.local", "commit", "-q", "--amend", "--no-edit")  # a rebase
        lander, sha = self.land_with(self.home_with([("core.hooksPath", str(hooks))]))
        self.assertIn("rebase", lander.calls)
        self.assertEqual(self.r.origin_head(), sha)
        self.assertFalse(self.marker.exists(), self.marker.read_text() if self.marker.exists() else "")

    def test_a_git_and_an_ssh_earlier_on_path_are_not_run(self):
        fake = self.d / "bin"
        marker_script(fake / "git", self.marker)
        marker_script(fake / "ssh", self.marker)
        lander, sha = self.land_with({"PATH": f"{fake}{os.pathsep}{os.environ['PATH']}"})
        self.assertEqual(self.r.origin_head(), sha)
        self.assertFalse(self.marker.exists(), self.marker.read_text() if self.marker.exists() else "")

    def test_git_program_variables_are_not_passed_on(self):
        progs = self.d / "progs"
        for name in ("ssh", "proxy", "git-receive-pack", "git-upload-pack", "git-remote-https"):
            marker_script(progs / name, self.marker)
        env = {"GIT_SSH_COMMAND": str(progs / "ssh"), "GIT_SSH": str(progs / "ssh"),
               "GIT_EXEC_PATH": str(progs), "GIT_PROXY_COMMAND": str(progs / "proxy")}
        lander, sha = self.land_with(env)
        self.assertEqual(self.r.origin_head(), sha)
        self.assertFalse(self.marker.exists(), self.marker.read_text() if self.marker.exists() else "")

    def test_children_get_no_loader_or_program_variables_and_a_fixed_path(self):
        fake = self.d / "bin"
        marker_script(fake / "git", self.marker)
        marker_script(fake / "ssh", self.marker)
        names = ["LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "GIT_SSH_COMMAND",
                 "GIT_SSH", "GIT_SSH_VARIANT", "GIT_EXEC_PATH", "GIT_PROXY_COMMAND", "GIT_ASKPASS", "SSH_ASKPASS",
                 "GIT_TEMPLATE_DIR", "GIT_EXTERNAL_DIFF", "GIT_PAGER", "PAGER", "GIT_EDITOR", "EDITOR", "VISUAL",
                 "GIT_TRACE", "GIT_TRACE_PACKET"]
        env = {k: "/nonexistent/x" for k in names}
        env["PATH"] = f"{fake}{os.pathsep}{os.environ['PATH']}"
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            lander = land.Lander(self.r.work, "T", log=lambda *_: None)
            probe = ("import os, shutil, sys; bad = [k for k in " + repr(names) + " if k in os.environ]; "
                     "g, s = shutil.which('git'), shutil.which('ssh'); "
                     f"bad += [x for x in (g, s) if x and x.startswith({str(fake)!r})]; "
                     "sys.exit(1 if bad else 0)")
            lander.sh(sys.executable, "-c", probe)   # LandError if the child saw any of them
            for e in (lander.git_env, land.clean_env(), land.clean_env(cargo=True)):
                self.assertEqual([k for k in names if k in e], [])
                self.assertNotIn(str(fake), e["PATH"])
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_git_and_gh_are_pinned(self):
        # without the tests' tripwire override: the pinned git and the forced config
        saved = dict(land.TOOL_OVERRIDE)
        land.TOOL_OVERRIDE.clear()
        self.addCleanup(land.TOOL_OVERRIDE.update, saved)
        argv = land.git_argv()
        self.assertTrue(os.path.isabs(argv[0]), argv)
        self.assertIn(argv[0], {os.path.realpath(c) for c in land.GIT_CANDIDATES})
        self.assertEqual(argv[1:], list(land.FORCED_GIT_CONFIG))
        for want in ("core.hooksPath=/dev/null", "core.fsmonitor=false", "core.sshCommand=/usr/bin/ssh",
                     "protocol.ext.allow=never"):
            self.assertIn(want, land.FORCED_GIT_CONFIG)
        with self.assertRaisesRegex(land.LandError, "PATH is not searched"):
            land.pinned_tool(("/nonexistent/git", str(self.d)))
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        self.assertEqual(lander.git_argv, argv)
        gh = [os.path.realpath(c) for c in land.GH_CANDIDATES if os.path.isfile(os.path.realpath(c))]
        if gh:
            self.assertEqual(lander.gh(), [gh[0]])
        cargo = land.clean_env(cargo=True)["PATH"].split(os.pathsep)
        self.assertEqual(cargo[1], land.cargo_bin(), cargo)   # right after the running Python

    def test_github_remote_forms(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        for url in ("https://github.com/o/r.git", "git@github.com:o/r", "ssh://git@github.com/o/r.git"):
            git(self.r.work, "remote", "set-url", "origin", url)
            self.assertEqual(lander.slug(), "o/r")
            lander.check_remote()   # the effective URLs equal the configured GitHub URL
        git(self.r.work, "remote", "set-url", "origin", "https://github.com/o/r.git")
        git(self.r.work, "remote", "set-url", "--push", "origin", "https://github.com/x/r.git")
        with self.assertRaisesRegex(land.LandError, "effective push URL 'https://github.com/x/r.git' is not the "
                                                    "repository's remote \\(https://github.com/o/r"):
            lander.check_remote()
        git(self.r.work, "remote", "set-url", "--delete", "--push", "origin", "https://github.com/x/r.git")
        git(self.r.work, "config", "url.https://github.com/evil/.insteadOf", "https://github.com/o/")
        with self.assertRaisesRegex(land.LandError, "effective push URL 'https://github.com/evil/r.git'"):
            lander.check_remote()
        # fetch redirected, push kept: the evidence would be read from another repository
        git(self.r.work, "config", "url.https://github.com/o/.pushInsteadOf", "https://github.com/o/")
        with self.assertRaisesRegex(land.LandError, "effective fetch URL 'https://github.com/evil/r.git'"):
            lander.check_remote()
        git(self.r.work, "config", "--unset", "url.https://github.com/o/.pushInsteadOf")
        git(self.r.work, "config", "--unset", "url.https://github.com/evil/.insteadOf")
        git(self.r.work, "remote", "set-url", "origin", "https://example.org/o/r.git")
        with self.assertRaisesRegex(land.LandError, "is not a GitHub owner/repo URL"):
            lander.check_remote()
        self.assertIsNone(land.remote_url_problem("https://github.com/o/r", ("https://github.com/o/r",)))
        self.assertIsNotNone(land.remote_url_problem("https://github.com/o/r2", ("https://github.com/o/r",)))


HOSTILE_GH = {"GH_HOST": "unroutable.invalid", "GH_ENTERPRISE_TOKEN": "hostile", "GITHUB_ENTERPRISE_TOKEN": "hostile",
              "GH_CONFIG_DIR": "/nonexistent/gh", "GH_PATH": "/nonexistent/gh", "GH_BROWSER": "x", "GH_PAGER": "x",
              "GH_EDITOR": "x", "GH_DEBUG": "api", "GH_REPO": "evil/repo",
              "SSL_CERT_FILE": "/nonexistent/ca.pem", "SSL_CERT_DIR": "/nonexistent", "CURL_CA_BUNDLE": "/x",
              "GIT_SSL_NO_VERIFY": "1", "GIT_SSL_CAINFO": "/x", "NODE_EXTRA_CA_CERTS": "/x"}
# GH_CONFIG_DIR is set again by land.py (the account's own), so it is checked by value
GH_SCRUBBED = [k for k in HOSTILE_GH if k not in ("GH_HOST", "GH_CONFIG_DIR")]


class GhEnvironment(Isolated):
    """gh is pinned to github.com: the environment cannot point it at another host."""

    def setUp(self):
        super().setUp()
        self.d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, True)
        (self.d / "gh.py").write_text(FAKE_GH, encoding="utf-8")
        self.state = self.d / "state.json"
        old = {k: os.environ.get(k) for k in [*HOSTILE_GH, "FAKE_GH_STATE", "GH_TOKEN", "GITHUB_TOKEN"]}

        def restore():
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(restore)
        os.environ.update(HOSTILE_GH)
        os.environ["GH_TOKEN"], os.environ["GITHUB_TOKEN"] = "token-a", "token-b"
        os.environ["FAKE_GH_STATE"] = str(self.state)

    def lander(self, auth=True, api=None, lists=None):
        self.state.write_text(json.dumps({"calls": [], "auth": auth, "api": api or {}, "list": lists or [[]],
                                          "jobs": {}, "on_dispatch": {}, "added": [], "dispatched": [], "polls": 0}))

        class L(land.Lander):
            def slug(self):
                return "o/r"
        lnd = L(self.wf_root(), "T", log=lambda *_: None)
        lnd.gh_cmd = [sys.executable, str(self.d / "gh.py")]
        return lnd

    def st(self):
        return json.loads(self.state.read_text())

    def assert_env(self, envs):
        self.assertTrue(envs)
        for e in envs:
            self.assertEqual(e.get("GH_HOST"), "github.com", e)
            self.assertEqual([k for k in GH_SCRUBBED if k in e], [], e)
            self.assertEqual((e.get("GH_TOKEN"), e.get("GITHUB_TOKEN")), ("token-a", "token-b"))
            self.assertNotEqual(e.get("GH_CONFIG_DIR"), HOSTILE_GH["GH_CONFIG_DIR"])
            if land.account_home():
                self.assertEqual(e.get("GH_CONFIG_DIR"), os.path.join(land.account_home(), ".config", "gh"))

    def test_every_gh_call_is_pinned_to_github_com(self):
        lnd = self.lander(api={"repos/o/r/commits/abc": {"commit": {"author": {"name": NAME, "email": EMAIL},
                                                                     "committer": {"name": NAME, "email": EMAIL}}}})
        lnd.check_gh_auth()
        lnd.list_branch_runs("o/r", "ci/T")
        lnd.run_jobs("o/r", 7)
        lnd.dispatch("o/r", "fuzz.yml", "ci/T")
        lnd.api_commit_identity("abc")
        calls = self.st()["calls"]
        self.assertEqual(calls[0], ["auth", "status", "--active", "--hostname", "github.com"])
        for c in calls[1:4]:
            self.assertEqual(c[c.index("--repo") + 1], "github.com/o/r", c)
        self.assertEqual(calls[4][:3], ["api", "--hostname", "github.com"])
        self.assert_env(self.st()["envs"])

    def test_auth_status_failing_refuses(self):
        with self.assertRaisesRegex(land.LandError, "gh is not logged in to github.com"):
            self.lander(auth=False).check_gh_auth()

    def test_the_landing_and_the_wait_check_auth_first(self):
        lnd = self.lander(auth=False)
        lnd.poll_seconds, lnd.after_max_polls = 0, 1   # a missing check must fail fast, not poll
        with self.assertRaisesRegex(land.LandError, "gh is not logged in"):
            lnd.wait_after(SHA)
        self.assertEqual(self.st()["calls"], [["auth", "status", "--active", "--hostname", "github.com"]])

        class NoRemote(land.Lander):
            def check_remote(self):
                pass

            def lock(self):
                import contextlib
                return contextlib.nullcontext()
        # the throwaway repository: if the check were missing, nothing real is fetched or pushed
        lnd = NoRemote(self.r.work, "T", log=lambda *_: None)
        lnd.gh_cmd = [sys.executable, str(self.d / "gh.py")]
        with self.assertRaisesRegex(land.LandError, "gh is not logged in"):
            lnd.land()

    def test_gh_env(self):
        e = land.gh_env()
        self.assertEqual(e["GH_HOST"], "github.com")
        self.assertEqual([k for k in GH_SCRUBBED if k in e], [])
        self.assertEqual((e["GH_TOKEN"], e["GITHUB_TOKEN"]), ("token-a", "token-b"))
        self.assertEqual(e["GH_PROMPT_DISABLED"], "1")
        self.assertNotEqual(e.get("GH_CONFIG_DIR"), HOSTILE_GH["GH_CONFIG_DIR"])
        self.assertEqual(land.repo_arg("o/r"), "github.com/o/r")
        for k in ("SSL_CERT_FILE", "SSL_CERT_DIR", "GIT_SSL_NO_VERIFY", "GIT_SSL_CAINFO", "CURL_CA_BUNDLE"):
            self.assertNotIn(k, land.clean_env())

    @live
    def test_a_real_gh_ignores_a_hostile_host(self):
        # safe offline: an unknown host fails to resolve, github.com may or may not be reachable;
        # either way the request must not go to the host the environment named
        gh = [os.path.realpath(c) for c in land.GH_CANDIDATES if os.path.isfile(os.path.realpath(c))]
        if not gh:
            self.skipTest("no gh at the fixed locations")
        r = subprocess.run([gh[0], "api", "--hostname", "github.com", "rate_limit", "--jq", ".resources.core.limit"],
                           capture_output=True, text=True, env=land.gh_env(), timeout=60)
        self.assertNotIn("unroutable.invalid", r.stdout + r.stderr)

    def test_windows_candidates_need_an_exe(self):
        exe = self.d / "tool.exe"
        exe.write_text("x")
        plain = self.d / "tool"
        plain.write_text("x")
        old = os.name
        try:
            land.os.name = "nt"
            self.assertEqual(land.pinned_tool((str(plain), str(exe))), os.path.realpath(exe))
            with self.assertRaisesRegex(land.LandError, "PATH is not searched"):
                land.pinned_tool((str(plain), str(self.d)))
        finally:
            land.os.name = old


class GitVersion(Isolated):
    """git is the first candidate at least GIT_MIN_VERSION: an older git earlier in the list
    (Apple's git on some macOS images) is skipped, and no new enough git is a refusal that
    names every candidate."""

    def setUp(self):
        super().setUp()
        self.d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.d, True)

    def fake_git(self, name: str, says: str) -> str:
        """an executable that prints `says` for --version / version (POSIX only: a script)"""
        f = self.d / name
        f.write_text(f"#!/bin/sh\necho '{says}'\n", encoding="utf-8")
        f.chmod(0o755)
        return str(f)

    def test_the_version_is_read_from_every_known_form(self):
        self.assertEqual(land.git_version("git version 2.39.5 (Apple Git-154)\n"), (2, 39, 5))
        self.assertEqual(land.git_version("git version 2.47.1.windows.1"), (2, 47, 1))
        self.assertEqual(land.git_version("git version 2.42"), (2, 42, 0))
        self.assertIsNone(land.git_version("hub version 2.14.2"))
        self.assertIsNone(land.git_version(""))
        self.assertGreaterEqual(land.git_version(subprocess.run([REAL_GIT, "--version"], capture_output=True,
                                                                text=True).stdout), land.GIT_MIN_VERSION)

    def test_the_minimum_is_the_newest_option_land_py_passes(self):
        # --attr-source=<tree> as a global option is in git 2.42; land.py passes it to rebase / commit
        self.assertIn('f"--attr-source={EMPTY_TREE}"', Path(land.__file__).read_text(encoding="utf-8"))
        self.assertGreaterEqual(land.GIT_MIN_VERSION, (2, 42))

    @unittest.skipIf(os.name == "nt", "the fake git is a shell script")
    def test_an_older_git_first_is_skipped(self):
        old = self.fake_git("git-old", "git version 2.39.5 (Apple Git-154)")
        self.assertEqual(land.pinned_git_from((old, REAL_GIT)), os.path.realpath(REAL_GIT))
        # pinned_git() itself makes this choice from GIT_CANDIDATES
        saved = (land.GIT_CANDIDATES, dict(land._PINNED))
        self.addCleanup(lambda: (setattr(land, "GIT_CANDIDATES", saved[0]), land._PINNED.clear(),
                                 land._PINNED.update(saved[1])))
        land.GIT_CANDIDATES = (old, REAL_GIT)
        land._PINNED.pop("git", None)
        self.assertEqual(land.pinned_git(), os.path.realpath(REAL_GIT))
        junk = self.fake_git("git-junk", "not a git")
        self.assertEqual(land.pinned_git_from((str(self.d / "missing"), junk, old, REAL_GIT)),
                         os.path.realpath(REAL_GIT))

    @unittest.skipIf(os.name == "nt", "the fake git is a shell script")
    def test_no_git_new_enough_refuses_and_names_every_candidate(self):
        old = self.fake_git("git-old", "git version 2.39.5 (Apple Git-154)")
        older = self.fake_git("git-older", "git version 2.30.1")
        missing = str(self.d / "missing")
        with self.assertRaises(land.LandError) as e:
            land.pinned_git_from((old, missing, older))
        text = str(e.exception)
        self.assertIn("no git at least 2.42", text)
        self.assertIn(f"{old}: git version 2.39.5 (Apple Git-154) (older than 2.42)", text)
        self.assertIn(f"{older}: git version 2.30.1 (older than 2.42)", text)
        self.assertIn(f"{missing}: not an executable", text)
        # a version equal to the minimum is enough
        exact = self.fake_git("git-exact", "git version 2.42.0")
        self.assertEqual(land.pinned_git_from((old, exact)), os.path.realpath(exact))

    @unittest.skipIf(os.name == "nt", "the fake git is a shell script")
    def test_the_env_smoke_fails_on_a_git_older_than_the_minimum(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        executed, failed, lines = land.env_smoke(lander, steps=("git",))
        self.assertEqual((executed, failed), (1, 0), lines)
        self.assertTrue(lines[0].startswith("ok   git: git version"), lines)
        self.assertIn(f"minimum {land.version_text(land.GIT_MIN_VERSION)}", lines[0])
        lander.git_argv = [self.fake_git("git-old", "git version 2.39.5 (Apple Git-154)")]
        executed, failed, lines = land.env_smoke(lander, steps=("git",))
        self.assertEqual((executed, failed), (1, 1), lines)
        self.assertIn("git 2.39.5 is older than 2.42 (GIT_MIN_VERSION)", lines[0])


class Harness(Isolated):
    """The guards of this test module: no Lander on this checkout, no network, no real gh."""

    def tripped(self) -> str:
        text = TRIPWIRE.read_text(encoding="utf-8") if TRIPWIRE.exists() else ""
        if TRIPWIRE.exists():
            TRIPWIRE.unlink()
        return text

    def test_the_tests_own_checkout_is_refused_before_any_git_call(self):
        def no_git():
            raise AssertionError("a git command line was built")
        old = land.git_argv
        land.git_argv = no_git
        try:
            for root in (REAL_ROOT, REAL_ROOT / "scripts" / ".."):
                with self.assertRaisesRegex(land.LandError, "forbidden root"):
                    land.Lander(root, "T", log=lambda *_: None)
                with self.assertRaisesRegex(land.LandError, "forbidden root"):
                    FakeLander(root, "T")
        finally:
            land.git_argv = old
        self.assertIn(os.path.realpath(REAL_ROOT), land.FORBIDDEN_ROOTS)

    def test_the_sandbox_origin_is_local(self):
        assert_local_origin(self.r.work, self.r.dir)
        for url in ("https://github.com/o/r.git", "git@github.com:o/r.git", str(REAL_ROOT)):
            git(self.r.work, "remote", "set-url", "origin", url)
            with self.assertRaisesRegex(AssertionError, "refusing to run the test"):
                assert_local_origin(self.r.work, self.r.dir)
        git(self.r.work, "remote", "set-url", "origin", str(self.r.origin))
        git(self.r.work, "config", f"url.https://github.com/o/r.git.insteadOf", str(self.r.origin))
        with self.assertRaisesRegex(AssertionError, "refusing to run the test"):
            assert_local_origin(self.r.work, self.r.dir)

    def test_a_network_argument_trips(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        out = lander.git("ls-remote", "https://github.com.invalid/o/r", check=False)
        self.assertEqual(out, "")
        self.assertIn("git: network argument 'https://github.com.invalid/o/r'", self.tripped())

    def test_a_remote_that_reaches_the_network_trips(self):
        git(self.r.work, "remote", "add", "far", "ssh://git@unroutable.invalid/o/r.git")
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        for args in (("fetch", "-q", "far"), ("push", "-q", "far", "HEAD:refs/heads/x"), ("ls-remote", "far")):
            lander.git(*args, check=False)
            self.assertIn("reaches 'ssh://git@unroutable.invalid/o/r.git'", self.tripped(), args)
        # the prefix is the origin URL exactly as the clone wrote it (a path with the separators
        # of the platform: a hand-built prefix did not match on Windows and the push stayed local)
        origin_url = git(self.r.work, "config", "--get", "remote.origin.url")
        git(self.r.work, "config", "url.ssh://git@unroutable.invalid/x.git.pushInsteadOf", origin_url)
        self.assertEqual(subprocess.run([REAL_GIT, "-C", str(self.r.work), "remote", "get-url", "--push", "origin"],
                                        capture_output=True, text=True).stdout.strip(),
                         "ssh://git@unroutable.invalid/x.git", "the pushInsteadOf really applies")
        lander.git("push", "-q", "origin", "HEAD:refs/heads/x", check=False)
        self.assertIn("push reaches 'ssh://git@unroutable.invalid/x.git'", self.tripped())

    def test_a_remote_that_cannot_be_resolved_trips(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        # a local path given as the repository: not a configured remote, so its push URL
        # cannot be resolved; the tripwire refuses instead of guessing that it is local
        lander.git("push", "-q", str(self.r.origin), "HEAD:refs/heads/x", check=False)
        self.assertIn("cannot resolve", self.tripped())
        self.assertEqual(git(self.r.upstream, "ls-remote", "origin", "refs/heads/x"), "", "nothing was pushed")
        # a name that is no remote and no path: resolves to itself, which is not local
        lander.git("fetch", "-q", "nosuchremote", check=False)
        self.assertIn("fetch reaches 'nosuchremote'", self.tripped())

    def test_main_options_that_redirect_git_trip(self):
        git(self.r.work, "remote", "add", "far", "ssh://git@unroutable.invalid/o/r.git")
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        net = str(self.r.dir / "net")
        for opt in ("--git-dir", "--work-tree", "--namespace", "--super-prefix", "--exec-path", "--config-env",
                    "--shallow-file", "-C"):
            for form in ([opt, net], [f"{opt}={net}"]):
                if opt == "-C" and "=" in form[0]:
                    continue
                with self.subTest(form=form):
                    lander.git(*form, "push", "-q", "far", "HEAD:refs/heads/x", check=False)
                    self.assertIn(f"main option {opt} redirects git", self.tripped())
        lander.git("--bare", "push", "-q", "origin", "HEAD:refs/heads/x", check=False)
        self.assertIn("main option --bare redirects git", self.tripped())

    def test_an_undetermined_subcommand_trips(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        for args, why in ((("--frobnicate", "push", "-q", "origin"), "unknown main option '--frobnicate'"),
                          (("-x", "status"), "unknown main option '-x'"),
                          (("--version",), "unknown main option '--version'"),
                          (("-c",), "main option -c without its value"),
                          (("-c", "a.b=c"), "no subcommand"),
                          (("--attr-source",), "main option --attr-source without its value")):
            with self.subTest(args=args):
                lander.git(*args, check=False)
                self.assertIn(why, self.tripped())

    def test_land_py_argv_shapes_pass(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        tree = land.EMPTY_TREE
        self.assertTrue(lander.git(*lander.ident_cfg, f"--attr-source={tree}", "rev-parse", "HEAD"))
        self.assertTrue(lander.git("-c", "core.editor=true", "--no-pager", "rev-parse", "HEAD"))
        self.assertTrue(lander.git("--attr-source", tree, "rev-parse", "HEAD"))
        self.assertEqual(self.tripped(), "")

    def test_local_git_passes_through(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        lander.git("fetch", "-q", "origin")
        self.assertTrue(lander.git("ls-remote", "origin", "main"))
        self.assertEqual(lander.git("remote", "get-url", "--push", "origin"), str(self.r.origin))
        self.assertEqual(self.tripped(), "")

    def test_a_real_gh_trips(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        with self.assertRaisesRegex(land.LandError, "gh is not logged in"):
            lander.check_gh_auth()
        self.assertIn("gh: auth status --active --hostname github.com", self.tripped())

    def test_a_trip_fails_the_test(self):
        class Inner(Isolated):
            def test_x(self):
                land.Lander(self.r.work, "T", log=lambda *_: None).git("ls-remote", "git@github.com:o/r", check=False)
        result = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(Inner).run(result)
        self.assertEqual(len(result.failures), 1, result.failures)
        self.assertIn("tripwire fired", result.failures[0][1])
        self.assertFalse(TRIPWIRE.exists())

    def test_no_live_test_runs_in_ci(self):
        self.assertFalse(live_enabled({"CI": "true", "LAND_LIVE_TESTS": "1"}))
        self.assertFalse(live_enabled({}))
        self.assertTrue(live_enabled({"LAND_LIVE_TESTS": "1"}))
        self.assertEqual(LIVE_TESTS, ["GhEnvironment.test_a_real_gh_ignores_a_hostile_host"])
        if os.environ.get("CI"):
            self.assertFalse(live_enabled(os.environ), "a live test would run in CI")
            for name in LIVE_TESTS:
                cls, meth = name.split(".")
                fn = getattr(globals()[cls], meth)
                self.assertTrue(getattr(fn, "__unittest_skip__", False), name)


class ChildTools(Isolated):
    """The tools land.py's children need resolve on the child PATH: python3 is the interpreter
    running land.py (>= 3.11, for tomllib), then cargo / rustc / rust-analyzer, bash, git."""

    def test_the_running_python_comes_first(self):
        for cargo in (False, True):
            first = land.safe_path(cargo).split(os.pathsep)[0]
            self.assertEqual(first, land.python_dir(), cargo)
        if os.name != "nt":
            import shutil
            exe = shutil.which("python3", path=land.safe_path(cargo=True))
            self.assertEqual(os.path.dirname(exe), land.python_dir())
            self.assertEqual(os.path.realpath(exe), os.path.realpath(sys.executable))
        self.assertIn(land.cargo_bin(), land.safe_path(cargo=True).split(os.pathsep))
        self.assertNotIn(land.cargo_bin(), land.safe_path().split(os.pathsep))
        self.assertIn(os.path.dirname(land.pinned_git()), land.safe_path().split(os.pathsep))

    @unittest.skipIf(os.name == "nt", "python3 is a POSIX name; Windows children do not run the scripts")
    def test_python3_in_the_child_env_is_this_interpreter(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        r = subprocess.run(["python3", "-c", "import sys, tomllib; print(sys.executable)"], cwd=self.r.work,
                           env=lander.child_env(), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = r.stdout.strip()
        self.assertTrue(os.path.dirname(got) == os.path.dirname(sys.executable)
                        or os.path.realpath(got) == os.path.realpath(sys.executable), (got, sys.executable))

    def test_the_env_smoke_executes_the_tools(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        executed, failed, lines = land.env_smoke(lander)
        text = "\n".join(lines)
        self.assertGreaterEqual(executed, 4, text)
        self.assertTrue(any(l.startswith("ok   python3: ") for l in lines), text)
        self.assertTrue(any(l.startswith("ok   python-env: ") for l in lines), text)
        self.assertTrue(any(l.startswith("ok   git: git version") for l in lines), text)
        self.assertEqual(failed, 0, text)
        self.assertIn(f"env-smoke: executed {executed}, skipped ", text)

    def test_nothing_executed_fails(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        executed, failed, lines = land.env_smoke(lander, steps=())
        self.assertEqual(executed, 0)
        self.assertGreater(failed, 0)
        self.assertIn("env-smoke: nothing executed (compared nothing)", lines)
        # independent of the runner's tools: git is always there (pinned)
        executed, failed, lines = land.env_smoke(lander, steps=("git",))
        self.assertEqual((executed, failed), (1, 0), lines)
        with self.assertRaisesRegex(land.LandError, "unknown step"):
            land.env_smoke(lander, steps=("make",))

    def test_a_failing_step_fails(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        old = land.python_dir
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        if os.name != "nt":
            (d / "python3").write_text("#!/bin/sh\nexit 3\n")
            (d / "python3").chmod(0o755)
        land.python_dir = lambda: str(d)
        try:
            executed, failed, lines = land.env_smoke(lander, steps=("python3",))
        finally:
            land.python_dir = old
        if os.name != "nt":
            self.assertEqual((executed, failed), (1, 1), lines)
            self.assertTrue(lines[0].startswith("FAIL python3: exit 3"), lines)

    @unittest.skipIf(os.name == "nt", "a shell script as python3")
    def test_another_python3_fails(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        (d / "python3").write_text("#!/bin/sh\necho 3.9.6\necho /usr/bin/python3\n")
        (d / "python3").chmod(0o755)
        old = land.python_dir
        land.python_dir = lambda: str(d)
        try:
            executed, failed, lines = land.env_smoke(lander, steps=("python3",))
        finally:
            land.python_dir = old
        self.assertEqual((executed, failed), (1, 1), lines)
        self.assertIn("is '/usr/bin/python3', not the interpreter running land.py", lines[0])

    def fake_cargo(self, rustup_which: str | None, analyzer_exit: int = 0) -> Path:
        """a cargo bin directory with a rustup and a rust-analyzer proxy: `rustup which
        rust-analyzer` prints `rustup_which` (exit 0) or fails like a missing component (None)"""
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        real = d / "real-rust-analyzer"
        real.write_text(f"#!/bin/sh\necho rust-analyzer 0.0.0-fake\nexit {analyzer_exit}\n")
        which = (f"echo {real}" if rustup_which == "real" else
                 "echo \"error: Unknown binary 'rust-analyzer' in official toolchain 'stable'\" >&2; exit 1")
        (d / "rustup").write_text(f"#!/bin/sh\nif [ \"$1\" = which ]; then {which}; fi\n")
        (d / "rust-analyzer").write_text("#!/bin/sh\necho \"error: Unknown binary 'rust-analyzer'\" >&2\nexit 1\n")
        for f in (real, d / "rustup", d / "rust-analyzer"):
            f.chmod(0o755)
        old = land.cargo_bin
        land.cargo_bin = lambda: str(d)
        self.addCleanup(setattr, land, "cargo_bin", old)
        return d

    @unittest.skipIf(os.name == "nt", "shell-script fakes of rustup")
    def test_rust_analyzer_proxy_without_the_component_is_skipped(self):
        self.fake_cargo(None)
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        executed, failed, lines = land.env_smoke(lander, steps=("rust-analyzer", "git"))
        self.assertEqual((executed, failed), (1, 0), lines)
        self.assertIn("skip rust-analyzer: rustup: the component is not installed in the active toolchain", lines)
        self.assertIn("env-smoke: executed 1, skipped 1", lines)

    @unittest.skipIf(os.name == "nt", "shell-script fakes of rustup")
    def test_an_installed_rust_analyzer_that_fails_fails(self):
        self.fake_cargo("real", analyzer_exit=2)
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        executed, failed, lines = land.env_smoke(lander, steps=("rust-analyzer",))
        self.assertEqual((executed, failed), (1, 1), lines)
        self.assertTrue(lines[0].startswith("FAIL rust-analyzer: exit 2"), lines)
        self.fake_cargo("real", analyzer_exit=0)
        executed, failed, lines = land.env_smoke(lander, steps=("rust-analyzer",))
        self.assertEqual((executed, failed), (1, 0), lines)
        self.assertIn("real-rust-analyzer", lines[0])

    def test_the_git_line_names_the_pinned_git(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        _, _, lines = land.env_smoke(lander, steps=("git",))
        self.assertTrue(lines[0].startswith("ok   git: git version"), lines)
        self.assertIn(f"({land.pinned_git()} via the tests' tripwire, minimum "
                      f"{land.version_text(land.GIT_MIN_VERSION)})", lines[0])

    def test_the_cli_exit_code(self):
        import contextlib
        import io
        old = land.SMOKE_STEPS
        out = io.StringIO()
        try:
            land.SMOKE_STEPS = ()
            with contextlib.redirect_stdout(out), contextlib.chdir(self.r.work):
                rc = land.main(["--env-smoke"])
        finally:
            land.SMOKE_STEPS = old
        self.assertEqual(rc, 1, out.getvalue())
        self.assertIn("env-smoke: executed 0, skipped 0", out.getvalue())


class ChildPython(Isolated):
    """A child python started by land.py runs the interpreter's own code only, whatever the
    ambient environment plants (PYTHONPATH, PYTHONHOME, PYTHONSTARTUP, user site-packages)."""

    def hostile(self, drop=()):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        env, marker, probe = land.hostile_python_env(str(d))
        land.plant_user_site(env, marker)
        for k in drop:
            env.pop(k)
        saved = {k: os.environ.get(k) for k in env}

        def restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(restore)
        os.environ.update(env)        # ambient, not a minimal environment
        return env, Path(marker), probe

    def child_env(self):
        return land.Lander(self.r.work, "T", log=lambda *_: None).child_env()

    def test_no_python_variable_reaches_a_child(self):
        self.hostile()
        env = self.child_env()
        self.assertEqual(sorted(k for k in env if k.startswith("PYTHON")),
                         sorted(land.CHILD_PYTHON_ENV))
        for k, v in land.CHILD_PYTHON_ENV.items():
            self.assertEqual(env[k], v)
        for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONINSPECT", "PYTHONSAFEPATH",
                  "PYTHONWARNINGS", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
            self.assertNotIn(k, env)
        self.assertIn("HOME", env)   # kept (credentials, gh): its user site is disabled instead

    def test_the_probe_runs_clean(self):
        env, marker, probe = self.hostile()
        r = subprocess.run([sys.executable, probe], env=self.child_env(), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(got["sibling"], "probe")
        self.assertFalse(got["user_site"])
        self.assertFalse(marker.exists(), marker.read_text() if marker.exists() else "")

    def test_a_real_script_with_a_sibling_import_runs_clean(self):
        # scripts/test_ansi.py imports its sibling ansi.py and re / unittest from the stdlib;
        # the planted directory has all three names
        env, marker, _ = self.hostile()
        r = subprocess.run([sys.executable, str(REAL_ROOT / "scripts" / "test_ansi.py")], cwd=self.r.work,
                           env=self.child_env(), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-500:])
        self.assertIn("OK", r.stderr)
        self.assertFalse(marker.exists(), marker.read_text() if marker.exists() else "")

    def test_without_the_scrub_the_plant_would_run(self):
        # the control: the same hostile variables, not scrubbed (PYTHONHOME dropped, so the
        # interpreter starts): the planted json / user-site code runs
        env, marker, probe = self.hostile(drop=("PYTHONHOME",))
        raw = {**os.environ}
        r = subprocess.run([sys.executable, probe], env=raw, capture_output=True, text=True)
        self.assertTrue(marker.exists(), r.stdout + r.stderr)

    def test_the_env_smoke_step_catches_a_broken_sanitizer(self):
        # the smoke injects the hostile environment itself: with the scrub or the user-site
        # setting broken, its python-env step fails (a minimal environment around it, as in CI,
        # changes nothing)
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        for attr, broken in (("SCRUBBED_ENV_RE", re.compile(r"^(?!PYTHON)(" + land.SCRUBBED_ENV_RE.pattern[1:] + ")")),
                             ("CHILD_PYTHON_ENV", {})):
            old = getattr(land, attr)
            setattr(land, attr, broken)
            try:
                executed, failed, lines = land.env_smoke(lander, steps=("python-env",))
            finally:
                setattr(land, attr, old)
            self.assertEqual((executed, failed), (1, 1), (attr, lines))
            self.assertTrue(lines[0].startswith("FAIL python-env: "), lines)

    def test_the_env_smoke_step(self):
        lander = land.Lander(self.r.work, "T", log=lambda *_: None)
        executed, failed, lines = land.env_smoke(lander, steps=("python-env",))
        self.assertEqual((executed, failed), (1, 0), lines)
        self.assertTrue(lines[0].startswith("ok   python-env: sibling=probe user_site=False"), lines)


class AmbientEnv(Isolated):
    """The key cases in a child process whose environment is hostile throughout."""

    KEY = ["Landing.test_a_denied_committer_in_the_range_is_refused",
           "Landing.test_the_landed_identity_is_read_back_from_the_remote",
           "Landing.test_an_ambient_git_env_does_not_force_a_rebase",
           "Landing.test_landed_commits_carry_the_fixed_committer",
           "Landing.test_a_src_only_change_takes_the_ci_lane",
           "Landing.test_code_upstream_in_other_files_compiles_but_does_not_rerun_preflight",
           "Landing.test_wrong_author_is_refused",
           "Identity.test_the_override_is_refused",
           "WaitEvidence.test_all_green_passes",
           "WaitAfter.test_all_green"]

    def test_the_key_cases_pass_under_a_hostile_environment(self):
        elsewhere = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, elsewhere, True)
        subprocess.run(["git", "init", "-q", str(elsewhere)], check=True)
        env = {**os.environ, **hostile_env(elsewhere)}
        r = subprocess.run([sys.executable, os.path.abspath(__file__), *self.KEY], env=env,
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn(f"Ran {len(self.KEY)} tests", r.stderr)


class Verdict(Isolated):
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
