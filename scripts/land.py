#!/usr/bin/env python3
"""Land the local commits on main.

Run from a checkout whose HEAD sits on top of an origin/main base. The script

  1. checks the commits: the author is LAND_IDENTITY, and no author or committer
     (nor LAND_IDENTITY itself) is a GitHub identity (identity_problem:
     noreply@github.com, *@users.noreply.github.com, the names web-flow and
     GitHub, case-insensitive); no internal vocabulary
     in the messages (the docs_lint word list and private-name hashes), a
     CHANGELOG line when src/ changes (--no-changelog for a pure refactor);
  2. takes the per-repository landing lock of this machine (LandLock, held until
     the push to main; a second landing waits up to --lock-wait-minutes);
  3. rebases on origin/main when needed (with a fixed committer; a conflict in a
     generated ledger takes main's copy, any other conflict stops), rebuilds and
     commits the ledgers, runs the checks and `scripts/preflight.sh --fast`;
  4. pushes HEAD to `ci/<id>` and waits for the evidence of every workflow that
     runs on ci/** (wait_evidence: the classes of .github/ci-evidence.toml; a
     missing or dropped run is re-created with `gh workflow run`); every landing
     takes this lane. The one exception is the emergency escape `--direct
     --reason "<why>"` (e.g. reverting a red main): it skips the ci/ push, prints
     a warning and puts the reason in the landing record;
  5. fetches main again: unmoved, it pushes the evidenced SHA itself; moved by
     documents only (docs_only), it rebases and runs the local checks; moved by
     code, it rebases, checks (preflight or a compile check), pushes the new SHA to
     `ci/<id>` and waits for the evidence again, at most MAX_EVIDENCE rounds in all,
     then stops ("main is moving");
  6. pushes the prepared SHA to main by name (`<sha>:main`, fast-forward only), going
     back to 5 when main moved meanwhile. Before that push the evidence rule must hold
     (evidence_rule): the landed SHA is the evidence SHA (main-unmoved / re-evidenced), or
     main gained documents only and our commits carry the same patches as the evidence
     with every other difference in the moved paths or the ledgers, and in every path main
     changed, each line main or the evidence added is still in the landed file (docs-only-moved);
     origin's ci/<id> must still be the evidence SHA. Before that push (and in a dry run)
     ALICE-Law is asked whether the landing moves its conformance state (check_law_state: a
     blob-less fetch of Law's main into a temporary directory runs its
     scripts/land_law_state.py with the prepared SHA of land.py's own working copy as the
     candidate): exit 0 lands; exit 1 (the state moves, so Law's state commit has to land
     first: the message names the Law SHA, the moved counts and the command), exit 2 (not
     computed) and a Law that cannot be read stop the landing with nothing pushed. The last
     line of a landing is `EVIDENCE <sha|-> LANDED <sha> RULE <rule>`.
  All of it runs in a detached working copy of land.py's own (own_copy, a `git worktree`
  made at start from the checkout's HEAD, the source SHA, read once) and removed at the end: the
  checkout itself is never pushed. If the checkout's HEAD moves during the landing, the
  landing stops before main moves; after a landing the checkout is moved to the landed
  SHA (`reset --keep`).
  7. before that push, requires author and committer of every commit about to reach
     main to be LAND_IDENTITY (check_push_identity); after it, reads the landed
     commits back from the remote (git, after a fetch) and the tip
     from the GitHub API, and requires author and committer to be LAND_IDENTITY
     ("REVERT OR FIX NOW" otherwise).

A different environment of the same user (by accident or on purpose) must not change
what land.py pushes, where, or as whom:
  - it refuses to start when the environment selects a repository (REFUSED_ENV:
    GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, ...);
  - git and gh run by absolute path from fixed locations (GIT_CANDIDATES /
    GH_CANDIDATES, never PATH; git is the first candidate at least GIT_MIN_VERSION, the
    oldest git with every option land.py passes), every git call with FORCED_GIT_CONFIG (no hooks, no
    fsmonitor, ssh as /usr/bin/ssh, no ext:: transport; command-line config wins over
    the global and system files, which stay for credential helpers);
  - every child process gets clean_env(): the ambient identity, injected config, the
    programs git or ssh would run (GIT_SSH*, GIT_EXEC_PATH, GIT_PROXY_COMMAND, askpass,
    pager, editor), GIT_TRACE* and LD_* / DYLD_* dropped, and a fixed PATH (the system
    directories, /opt/homebrew/bin, /usr/local/bin; for preflight and cargo also
    LAND_CARGO_DIR when set, else $CARGO_HOME/bin or ~/.cargo/bin);
  - before the first fetch and before every push, the effective fetch and push URLs of
    the remote (after insteadOf / pushInsteadOf / pushurl) must be the GitHub URL of
    the owner/repo that gh is told (check_remote).
  - gh runs with gh_env(): every GH_* / GITHUB_*_TOKEN variable dropped except
    GH_TOKEN and GITHUB_TOKEN, GH_HOST=github.com, prompts and update checks off, the
    config directory under the account's home from the user database; every --repo is
    host-qualified (github.com/<owner>/<repo>) and every `gh api` names --hostname
    github.com. Proxies are kept; the TLS trust overrides (SSL_CERT_FILE, SSL_CERT_DIR,
    GIT_SSL_*, CURL_CA_BUNDLE, NODE_EXTRA_CA_CERTS, NODE_TLS_REJECT_UNAUTHORIZED) are
    dropped for every child, so a proxy cannot answer for github.com. Before a landing
    and before --wait reads anything, `gh auth status --active --hostname github.com`
    must succeed (check_gh_auth).

Generated ledgers are not meant to be edited on a branch: they are rebuilt at
landing time, so two landings never conflict on them.

The landing record (printed at the end) lists the lane and, per expected
workflow, the run id and conclusion the landing relied on.

After the land, `--wait` / `--wait-only SHA` read the after-landing set on main
(after_landing_items): the after-landing legs (macos-latest), the legs that run on
main only (macos-15-intel, also re-run by the nightly safety net) and the
after-landing workflows when they ran (Miri and loom). Exit codes: 0 all green,
EXIT_RED (4) a red result ("REVERT OR FIX NOW"), EXIT_PENDING (3) not finished
within the bound, EXIT_CHECKOUT (5) landed (and green when waited for) but the checkout
could not be moved to the landed SHA (`reset --keep` refused: changes made there during the
landing; nothing in it is overwritten), 1 any other error. The last line of a landing is
`EVIDENCE <sha|-> LANDED <sha> RULE <rule>` (printed again after the --wait report).

Usage:
  python3 scripts/land.py --id AUD-A-S1W2-005            # land
  python3 scripts/land.py --id tgs-keys --dry-run         # everything but the pushes
  python3 scripts/land.py --id x --wait                   # also read the after-landing set on main
  python3 scripts/land.py --id revert-x --direct --reason "revert the red main of abc1234"
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IDENTITY = os.environ.get("LAND_IDENTITY", "Moroya Sakamoto <sakamoro@alicelaw.net>")
# identities that are never an author or committer of a landed commit, whatever
# LAND_IDENTITY says: GitHub's own (web-flow commits from the web UI, merges and bots,
# and the per-account noreply addresses), which put a GitHub account on the commit
DENIED_EMAILS = ("noreply@github.com",)
DENIED_EMAIL_SUFFIXES = ("@users.noreply.github.com",)
DENIED_NAMES = ("web-flow", "github")


# git environment variables that override the identity land.py gives with `-c user.*` (the
# environment wins over -c), point git at another repository or index, or inject config.
# land.py drops them for every git (and child) process it starts, so its `-c user.*`
# identity applies and the ambient environment of a shell cannot change who commits the
# landing, when, or which repository is landed
SCRUBBED_ENV_RE = re.compile(
    r"^(GIT_(AUTHOR|COMMITTER)_(NAME|EMAIL|DATE)|GIT_DIR|GIT_WORK_TREE|GIT_INDEX_FILE|GIT_COMMON_DIR"
    r"|GIT_OBJECT_DIRECTORY|GIT_ALTERNATE_OBJECT_DIRECTORIES|GIT_NAMESPACE|GIT_CEILING_DIRECTORIES"
    r"|GIT_CONFIG_PARAMETERS|GIT_CONFIG_COUNT|GIT_CONFIG_KEY_\d+|GIT_CONFIG_VALUE_\d+"
    # programs git or ssh would run instead of the expected ones, and loader injection
    r"|GIT_SSH_COMMAND|GIT_SSH|GIT_SSH_VARIANT|GIT_EXEC_PATH|GIT_PROXY_COMMAND|GIT_ASKPASS|SSH_ASKPASS"
    r"|GIT_TEMPLATE_DIR|GIT_EXTERNAL_DIFF|GIT_PAGER|PAGER|GIT_EDITOR|EDITOR|VISUAL|GIT_TRACE.*"
    r"|LD_.*|DYLD_.*"
    # TLS trust overrides (a proxy alone cannot forge a GitHub answer; a replaced CA can)
    r"|SSL_CERT_FILE|SSL_CERT_DIR|GIT_SSL_NO_VERIFY|GIT_SSL_CAINFO|GIT_SSL_CAPATH|CURL_CA_BUNDLE"
    r"|NODE_EXTRA_CA_CERTS|NODE_TLS_REJECT_UNAUTHORIZED"
    # Python: every PYTHON* variable (PYTHONPATH / PYTHONHOME / PYTHONSTARTUP / PYTHONUSERBASE /
    # PYTHONSAFEPATH / ... would plant or move modules for the scripts land.py starts, and for the
    # python3 that preflight.sh and scip_index.sh start), the macOS launcher variable that moves
    # sys.executable, and the virtual environment / conda activation variables (the children
    # use the interpreter running land.py, chosen by path, not an activated environment)
    r"|PYTHON.*|__PYVENV_LAUNCHER__|VIRTUAL_ENV|VIRTUAL_ENV_PROMPT|CONDA_.*)$")

# set for every child after the scrub: no user site-packages (a .pth file under the home
# directory would run code in every child python), no bytecode written into the checkout,
# UTF-8 standard streams
CHILD_PYTHON_ENV = {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}

# git and gh are run by absolute path, resolved once from fixed locations (never from PATH),
# and every child process gets a fixed PATH: a `git` or `ssh` placed earlier on the caller's
# PATH is never run. /opt/homebrew/bin and /usr/local/bin are the package prefixes of the
# machines that land; the preflight child (bash, python3, cargo, rustup) also gets the cargo
# directory (LAND_CARGO_DIR when the caller names it, else $CARGO_HOME/bin, default ~/.cargo/bin).
GIT_CANDIDATES = ("/usr/bin/git", "/opt/homebrew/bin/git", "/usr/local/bin/git",
                  r"C:\Program Files\Git\cmd\git.exe")
GH_CANDIDATES = ("/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh",
                 r"C:\Program Files\GitHub CLI\gh.exe")
SYSTEM_PATH = ("/usr/bin", "/bin", "/usr/sbin", "/sbin", "/opt/homebrew/bin", "/usr/local/bin")
# forced on every git call (command-line config wins over the global and system files, which
# are kept: credential helpers live there): no hooks, no fsmonitor program, ssh by absolute
# path, no ext:: transport
SSH = "/usr/bin/ssh"
FORCED_GIT_CONFIG = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                     "-c", f"core.sshCommand={SSH}", "-c", "protocol.ext.allow=never")


# the oldest git that has every option land.py passes; the candidates are tried in order and
# the first one at least this new is used (an older system git, such as Apple's git on some
# macOS images, is skipped, never run with an option it rejects). Raise it with the option:
#   --attr-source=<tree> (global option)  2.42
GIT_MIN_VERSION = (2, 42)


def executable(candidate: str) -> str | None:
    """the real path of the candidate if it exists and is executable"""
    real = os.path.realpath(candidate)
    if os.name == "nt":
        # os.access(X_OK) says little on Windows: an existing .exe file is required
        return real if real.lower().endswith(".exe") and os.path.isfile(real) else None
    return real if os.path.isfile(real) and os.access(real, os.X_OK) else None


def pinned_tool(candidates: tuple[str, ...]) -> str:
    """the real path of the first candidate that exists and is executable"""
    for c in candidates:
        real = executable(c)
        if real:
            return real
    raise LandError(f"none of {', '.join(candidates)} is an executable (PATH is not searched)")


def git_version(text: str) -> tuple[int, int, int] | None:
    """(major, minor, patch) from `git version 2.39.5 (Apple Git-154)` / `git version
    2.47.1.windows.1`; None for anything else"""
    m = re.match(r"git version (\d+)\.(\d+)(?:\.(\d+))?", text.strip())
    return (int(m[1]), int(m[2]), int(m[3] or 0)) if m else None


def version_text(v: tuple[int, ...]) -> str:
    return ".".join(map(str, v))


def pinned_git_from(candidates: tuple[str, ...], minimum: tuple[int, ...] = GIT_MIN_VERSION) -> str:
    """the real path of the first candidate that is executable and reports at least `minimum`;
    otherwise refuse, naming every candidate and what it was"""
    seen = []
    for c in candidates:
        real = executable(c)
        if real is None:
            seen.append(f"{c}: not an executable")
            continue
        try:
            # the git directory is left out of PATH here: it is what is being decided
            # (`git --version` starts no other program)
            r = subprocess.run([real, "--version"], env=clean_env(with_git=False), capture_output=True,
                               text=True, timeout=60)
            v = git_version(r.stdout) if r.returncode == 0 else None
            what = r.stdout.strip() or r.stderr.strip() or f"exit {r.returncode}"
        except (OSError, subprocess.TimeoutExpired) as e:
            v, what = None, str(e)
        if v is not None and v >= tuple(minimum):
            return real
        seen.append(f"{c}: {what[:80]}" + ("" if v is None else f" (older than {version_text(minimum)})"))
    raise LandError(f"no git at least {version_text(minimum)} at the fixed locations (PATH is not searched): "
                    + "; ".join(seen))


_PINNED: dict[str, str] = {}
# Test-only hooks, set from Python by scripts/test_land.py (never from the environment):
# TOOL_OVERRIDE replaces the command that starts git / gh (the tests run tripwire wrappers
# that fail on any network host), FORBIDDEN_ROOTS lists checkouts a Lander must not touch
# (the tests' own checkout).
TOOL_OVERRIDE: dict[str, list[str]] = {}
FORBIDDEN_ROOTS: set[str] = set()


def pinned_git() -> str:
    if "git" not in _PINNED:
        _PINNED["git"] = pinned_git_from(GIT_CANDIDATES)
    return _PINNED["git"]


def git_argv() -> list[str]:
    """the start of every git command line: the pinned git and the forced config"""
    return [*(TOOL_OVERRIDE.get("git") or [pinned_git()]), *FORCED_GIT_CONFIG]


_PYTHON_DIR: list[str] = []


def python_dir() -> str:
    """a directory whose `python3` is the interpreter running land.py (the scripts land.py
    starts need Python >= 3.11, and /usr/bin/python3 may be older): the interpreter's own
    directory when it holds a python3, otherwise a private directory with a python3 link"""
    if not _PYTHON_DIR:
        here = os.path.dirname(sys.executable)
        name = "python3.exe" if os.name == "nt" else "python3"
        cand = os.path.join(here, name)
        if os.name == "nt" or (os.path.isfile(cand) and os.path.realpath(cand) == os.path.realpath(sys.executable)):
            _PYTHON_DIR.append(here)
        else:
            import tempfile
            d = tempfile.mkdtemp(prefix="land-python-")
            os.symlink(sys.executable, os.path.join(d, "python3"))
            _PYTHON_DIR.append(d)
    return _PYTHON_DIR[0]


def cargo_bin() -> str:
    """the cargo directory of the cargo children: LAND_CARGO_DIR when the caller names it (the
    directory of the cargo it resolved, e.g. a toolchain manager that keeps its proxies outside
    $CARGO_HOME/bin), else $CARGO_HOME/bin, default ~/.cargo/bin; the PATH is never searched here"""
    named = os.environ.get("LAND_CARGO_DIR")
    if named:
        return named
    return os.path.join(os.environ.get("CARGO_HOME") or os.path.join(os.path.expanduser("~"), ".cargo"), "bin")


def missing_tool_reason(tool: str) -> str:
    """why a cargo child cannot start `tool`: the directories the child PATH searches for it"""
    return (f"not found on the child PATH (the cargo directory {cargo_bin()}, then the system "
            f"directories); install the Rust toolchain there, set CARGO_HOME, or name the directory "
            f"that holds cargo in LAND_CARGO_DIR")


def safe_path(cargo: bool = False, with_git: bool = True) -> str:
    """the fixed PATH of a child process: the directory of the Python running land.py first
    (python3 for the scripts), the cargo directory for the cargo children, then the system
    directories; git itself is started by absolute path, its directory comes last"""
    dirs = [python_dir()]
    if cargo:
        dirs.append(cargo_bin())
    if os.name == "nt":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        dirs += [os.path.join(root, "System32"), root]
    else:
        dirs += list(SYSTEM_PATH)
    if with_git:
        dirs.append(os.path.dirname(pinned_git()))
    seen: list[str] = []
    for d in dirs:
        if d not in seen:
            seen.append(d)
    return os.pathsep.join(seen)


# variables that select another repository, working tree, index or object store: land.py refuses
# to start when one is set (picking a repository silently is the dangerous direction);
# clean_env() still drops them for every child process
REFUSED_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
               "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_NAMESPACE")


def refused_env() -> list[str]:
    """the REFUSED_ENV variables set in this process's environment"""
    return [k for k in REFUSED_ENV if k in os.environ]


def landed_identity_problems(rows: list[tuple[str, str, str, str, str, str]], identity: str) -> list[str]:
    """rows: (sha, author name, author email, committer name, committer email, source).
    A problem per field that is not `identity` or is a denied identity, and one when no row
    was read (nothing compared)."""
    if not rows:
        return ["no landed commit was read from the remote (compared nothing)"]
    out = []
    for sha, an, ae, cn, ce, source in rows:
        for field, name, email in (("author", an, ae), ("committer", cn, ce)):
            who = f"{name} <{email}>"
            problem = identity_problem(name, email)
            if who != identity or problem:
                out.append(f"{sha[:7]} {field} `{who}` ({source})"
                           + (f": {problem}" if problem else f": not `{identity}`"))
    return out


def say(msg: str) -> None:
    """the default log: one flushed line (a landing waits for minutes; output redirected to
    a file must show each line as it happens, not at exit)"""
    print(msg, flush=True)


def line_buffered_output() -> None:
    """stdout / stderr in UTF-8 and line-buffered, also when redirected to a file or a pipe"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def clean_env(cargo: bool = False, with_git: bool = True) -> dict[str, str]:
    """os.environ without the variables SCRUBBED_ENV_RE matches, with the fixed PATH
    (with_git=False: without the pinned git's directory, for choosing that git)"""
    env = {k: v for k, v in os.environ.items() if not SCRUBBED_ENV_RE.match(k)}
    env["PATH"] = safe_path(cargo, with_git)
    env.update(CHILD_PYTHON_ENV)
    return env


# gh: every variable gh reads (GH_*, GITHUB_ENTERPRISE_TOKEN, ...) is dropped except the two
# token variables (legitimate auth for github.com); GH_HOST is forced to github.com, prompts
# and update checks are off, and the config directory is the one under the account's home
# directory from the user database (not $HOME / $XDG_CONFIG_HOME, which an environment can
# point at a config with another host or an `http_unix_socket`). Proxies are kept (networks
# need them; TLS to github.com is still verified because the CA overrides are dropped).
GH_HOSTNAME = "github.com"
# a gh failure GitHub reports as its own, on a read that can be asked again: a server error
# (HTTP 5xx) or the connection dropping / timing out; a 4xx or an auth failure is not one
GH_TRANSIENT = re.compile(r"HTTP 5\d\d\b|connection reset|connection refused|i/o timeout|TLS handshake timeout"
                          r"|unexpected EOF|error connecting to", re.IGNORECASE)
GH_KEPT = ("GH_TOKEN", "GITHUB_TOKEN")
GH_SCRUBBED_RE = re.compile(r"^(GH_.*|GITHUB_ENTERPRISE_TOKEN|GITHUB_.*_TOKEN)$")


def account_home() -> str | None:
    """the home directory of the account running land.py, from the user database"""
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError, AttributeError):
        return None


def gh_env() -> dict[str, str]:
    env = {k: v for k, v in clean_env().items() if not GH_SCRUBBED_RE.match(k) or k in GH_KEPT}
    env["GH_HOST"] = GH_HOSTNAME
    env["GH_PROMPT_DISABLED"] = "1"
    env["GH_NO_UPDATE_NOTIFIER"] = "1"
    home = account_home()
    if home:
        env["GH_CONFIG_DIR"] = os.path.join(home, ".config", "gh")
    return env


def law_state_env(gh: str | None) -> dict[str, str]:
    """the environment of ALICE-Law's land_law_state.py: gh_env() (it reads the other
    repositories' mains through `gh api`), with `git` and `gh` on PATH resolving to the pinned git
    and gh (the script starts both by name; the system directories of the fixed PATH come
    before the pinned git's otherwise) and FORCED_GIT_CONFIG given as GIT_CONFIG_COUNT (the
    script's git command lines cannot carry it)"""
    env = gh_env()
    first = [python_dir(), os.path.dirname(pinned_git())]
    if gh:
        first.append(os.path.dirname(os.path.realpath(gh)))
    rest = [d for d in env["PATH"].split(os.pathsep) if d and d not in first]
    env["PATH"] = os.pathsep.join(dict.fromkeys(first + rest))
    pairs = [FORCED_GIT_CONFIG[i + 1] for i in range(0, len(FORCED_GIT_CONFIG), 2)]
    env["GIT_CONFIG_COUNT"] = str(len(pairs))
    for n, kv in enumerate(pairs):
        key, _, value = kv.partition("=")
        env[f"GIT_CONFIG_KEY_{n}"], env[f"GIT_CONFIG_VALUE_{n}"] = key, value
    return env


def repo_arg(slug: str) -> str:
    """the --repo value: host-qualified, so GH_HOST or a config cannot pick another host"""
    return f"{GH_HOSTNAME}/{slug}"


GITHUB_URL_FORMS = ("https://github.com/{slug}", "https://github.com/{slug}.git",
                    "git@github.com:{slug}", "git@github.com:{slug}.git",
                    "ssh://git@github.com/{slug}", "ssh://git@github.com/{slug}.git")


def remote_url_problem(url: str, allowed: tuple[str, ...]) -> str | None:
    """why `url` (an effective fetch or push URL) is not one of `allowed`"""
    if url.strip() in allowed:
        return None
    return f"{url.strip()!r} is not the repository's remote ({' / '.join(allowed)})"


def identity_problem(name: str, email: str) -> str | None:
    """Why `name <email>` may not author or commit a landed commit (None: allowed).
    Compared case-insensitively, surrounding spaces ignored."""
    n, e = name.strip().lower(), email.strip().lower()
    if e in DENIED_EMAILS:
        return f"the email {email!r} is GitHub's noreply address"
    if any(e.endswith(s) for s in DENIED_EMAIL_SUFFIXES):
        return f"the email {email!r} is a GitHub account noreply address"
    if n in DENIED_NAMES:
        return f"the name {name!r} is a GitHub identity"
    return None
# the generated files, the generators and the toolchain pins live in ledgers.py
# (the preflight's stale check reads the same list)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ledgers import GENERATED, NATIVE, NIGHTLY, SEMI_GENERATED, WRITERS  # noqa: E402
# lists that both sides mostly add to: a conflict is merged by merge_append(),
# which keeps lines both sides added and a line one side rewrote, and stops when
# both sides changed the same lines. A union merge is not used: it keeps the old
# and the new version of a rewritten line side by side
APPEND_ONLY = ("CHANGELOG.md",)
DIRECT_REASON_MIN = 12  # characters of --reason required by the emergency --direct landing
EXIT_PENDING = 3        # --wait: the after-landing set did not finish within the bound
EXIT_RED = 4            # --wait: an after-landing result is red
EXIT_CHECKOUT = 5       # landed, but the checkout could not follow (reset --keep refused)
SIGNATURE_RE = re.compile(r"Co-Authored-By:|^\s*Generated with\b|\U0001F916", re.I | re.M)
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
MAX_ATTEMPTS = 6        # pushes to main (a push that loses a race to another landing retries)
REGEN_MESSAGE = "docs: regenerate the generated ledgers"   # the only commit message land.py writes
MAX_EVIDENCE = 3        # evidence rounds on ci/<id>: the first and two after main gained code
MOVING_MESSAGE = "main is moving: wait for a quiet main or coordinate a landing slot"
LOCK_DIR_ENV = "LAND_LOCK_DIR"  # where the per-repository landing lock lives (tests point it elsewhere)
LOCK_WAIT_MINUTES = 30
# ALICE-Law's conformance state: before the push to main, land.py asks Law's land_law_state.py
# whether landing the commit moves Law's recorded state (check_law_state)
LAW_URL = "https://github.com/ext-sakamoro/ALICE-Law"
LAW_STATE_SCRIPT = ("scripts", "land_law_state.py")
LAW_CANDIDATE = "ALICE-Physics"


class LandError(Exception):
    pass


# --- the evidence on the ci/<id> SHA ---------------------------------------------------
#
# Every workflow whose `on.push` runs on ci/** is expected on the ci/<id> SHA, with the
# class .github/ci-evidence.toml gives it:
#   landing        the newest run for the SHA must complete with conclusion success (for a
#                  path-gated workflow that is the Gate's verdict: skipped heavy jobs and a
#                  Gate success pass); a matrix leg declared `after-landing` only has to
#                  have started (in_progress or completed), its result is read after the
#                  landing; a completed red leg still stops the landing (a known red)
#   after-landing  the run must exist and its heavy jobs must have started when the changes
#                  job let them run (all heavy jobs skipped = the changes job said run=false);
#                  read from the jobs GitHub reports, not decided again here
#   main-after-landing (a leg), schedule-only, tag-only: not expected on ci/**
# No run for the SHA, or a cancelled / skipped run (one pending run per concurrency group:
# a middle run is dropped), is no evidence: the run is re-created with `gh workflow run`.

EVIDENCE_FILE = ".github/ci-evidence.toml"
EXPECTED_CLASSES = ("landing", "after-landing")
NOT_ON_CI_CLASSES = ("schedule-only", "tag-only")
MAX_DISPATCH = 2          # re-creations per workflow before the landing stops
NO_RUN_GRACE = 600        # seconds without a run for a workflow before it is re-created
STARTED = ("in_progress", "completed")


def _yaml_scalar(block: str, key: str, indent: int) -> str | None:
    m = re.search(rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", block, re.M)
    if not m:
        return None
    v = m.group(1)
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"" else v


def expected_workflows(root: Path) -> list[dict]:
    """The workflows expected on a ci/** push, derived from the workflow files and the
    evidence file. Each: file, name (the run's workflowName), cls, legs (after-landing matrix
    legs: job display names), jobs {id: display name}. Raises LandError on an inconsistency
    and when nothing is expected."""
    import tomllib
    import changed_paths as cp
    from ci_load_check import jobs as job_blocks
    try:
        ev = tomllib.loads((root / EVIDENCE_FILE).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise LandError(f"{EVIDENCE_FILE}: {e}") from None
    classes = ev.get("workflows", {})
    legs = ev.get("legs", [])
    out, errors = [], []
    for p in sorted((root / ".github" / "workflows").glob("*.yml")):
        text = p.read_text(encoding="utf-8")
        on = cp.on_block(text)
        push = cp.sub_block(on, "push", 2)
        if push is None:
            continue
        branches = cp.push_branches(push)
        if branches is not None and "ci/**" not in branches:
            continue
        if branches is None and cp.sub_block(push, "tags", 4) is not None:
            continue
        cls = classes.get(p.name)
        if cls not in EXPECTED_CLASSES:
            errors.append(f"{p.name} runs on ci/** but its class in {EVIDENCE_FILE} is {cls!r}"
                          f" (expected one of {', '.join(EXPECTED_CLASSES)})")
            continue
        if not any(re.match(r"^  workflow_dispatch:", l) for l in on):
            errors.append(f"{p.name} has no workflow_dispatch: a dropped run could not be re-created")
            continue
        name = _yaml_scalar(text, "name", 0)
        if not name:
            errors.append(f"{p.name}: no top-level name (the runs are listed by name)")
            continue
        blocks = job_blocks(text)
        names = {j: (_yaml_scalar(b, "name", 4) or j) for j, b in blocks.items()}
        wf_legs = []
        for leg in legs:
            if leg.get("workflow") != p.name or leg.get("class") != "after-landing":
                continue
            j = leg.get("job")
            if j not in names:
                errors.append(f"{EVIDENCE_FILE}: leg {p.name}/{j}: no such job")
                continue
            key = leg.get("key", "os")
            wf_legs.append(re.sub(r"\$\{\{\s*matrix\." + re.escape(key) + r"\s*\}\}", str(leg.get("matrix")), names[j]))
        out.append({"file": p.name, "name": name, "cls": cls, "legs": wf_legs, "jobs": names})
    if errors:
        raise LandError("the expected workflows cannot be derived:\n  " + "\n  ".join(errors))
    if not out:
        raise LandError("no workflow runs on ci/** (compared 0 workflows): nothing would be evidence")
    seen = [w["name"] for w in out]
    dup = sorted({n for n in seen if seen.count(n) > 1})
    if dup:
        raise LandError(f"two expected workflows share the run name {dup}: their runs cannot be told apart")
    return out


def waits_for(wf: dict) -> str:
    """what the landing waits for on one expected workflow (the table of docs/CI_TRIGGERS.md)"""
    if wf["cls"] == "after-landing":
        return "started: heavy jobs started, or all skipped by the changes job"
    if wf["legs"]:
        return "success; started only: " + ", ".join(wf["legs"])
    return "success"


def newest_run(runs: list[dict], name: str, sha: str) -> dict | None:
    mine = [r for r in runs if r.get("workflowName") == name and r.get("headSha") == sha]
    return max(mine, key=lambda r: r.get("databaseId", 0)) if mine else None


def run_verdict(wf: dict, run: dict | None, jobs: list[dict] | None) -> tuple[str, str]:
    """(state, why) for one expected workflow. state: ok / red / waiting / missing /
    dropped / need-jobs (the run is not finished and the decision needs its jobs)."""
    if run is None:
        return "missing", "no run for the SHA"
    status, concl = run.get("status"), run.get("conclusion")
    if status == "completed" and concl in ("cancelled", "skipped", "stale"):
        return "dropped", f"run {run.get('databaseId')} ended {concl} (no evidence)"
    if wf["cls"] == "landing" and not wf["legs"]:
        if status != "completed":
            return "waiting", f"run {run.get('databaseId')} {status}"
        if concl == "success":
            return "ok", f"run {run.get('databaseId')} success"
        return "red", f"run {run.get('databaseId')} ended {concl}"
    if status == "completed" and concl == "success":
        return "ok", f"run {run.get('databaseId')} success"
    if jobs is None:
        return "need-jobs", f"run {run.get('databaseId')} {status}/{concl}"
    rid = run.get("databaseId")
    if wf["cls"] == "landing":
        legs = set(wf["legs"])
        found = {j.get("name") for j in jobs}
        for leg in legs:
            if leg not in found:
                return "waiting" if status != "completed" else "red", f"leg {leg!r} not in run {rid}"
        for j in jobs:
            n, st, c = j.get("name"), j.get("status"), j.get("conclusion")
            if st == "completed" and c not in ("success",):
                return "red", f"job {n!r} of run {rid} ended {c}"
        pending = [j.get("name") for j in jobs if j.get("name") not in legs and j.get("status") != "completed"]
        unstarted = [j.get("name") for j in jobs if j.get("name") in legs and j.get("status") not in STARTED]
        if pending or unstarted:
            return "waiting", f"run {rid}: waiting for {pending + unstarted}"
        return "ok", f"run {rid}: every job success, after-landing legs {sorted(legs)} started"
    # after-landing workflow: the changes job decides, the heavy jobs must have started
    changes_name = wf["jobs"].get("changes")
    gate_name = wf["jobs"].get("gate")
    ch = [j for j in jobs if j.get("name") == changes_name]
    heavy = [j for j in jobs if j.get("name") not in (changes_name, gate_name)]
    if not ch:
        return "waiting", f"run {rid}: no changes job yet"
    if ch[0].get("status") != "completed":
        return "waiting", f"run {rid}: the changes job is {ch[0].get('status')}"
    if ch[0].get("conclusion") != "success":
        return "red", f"run {rid}: the changes job ended {ch[0].get('conclusion')}"
    if not heavy:
        return "waiting", f"run {rid}: no heavy job listed yet"
    for j in heavy:
        if j.get("status") == "completed" and j.get("conclusion") not in ("success", "skipped"):
            return "red", f"job {j.get('name')!r} of run {rid} ended {j.get('conclusion')}"
    skipped = [j for j in heavy if j.get("status") == "completed" and j.get("conclusion") == "skipped"]
    if len(skipped) == len(heavy):
        return "ok", f"run {rid}: the changes job said run=false (every heavy job skipped)"
    if skipped:
        return "red", f"run {rid}: heavy jobs partly skipped {[j.get('name') for j in skipped]}"
    unstarted = [j.get("name") for j in heavy if j.get("status") not in STARTED]
    if unstarted:
        return "waiting", f"run {rid}: heavy jobs not started yet {unstarted}"
    return "ok", f"run {rid}: heavy jobs started {[j.get('name') for j in heavy]} (result read after the landing)"


def direct_reason_problem(direct: bool, reason: str | None) -> str | None:
    """why the emergency --direct landing is refused (None: allowed or not asked for)"""
    if reason is not None and not direct:
        return "--reason is only for --direct"
    if not direct:
        return None
    if reason is None or len(reason.strip()) < DIRECT_REASON_MIN:
        return (f"--direct needs --reason with at least {DIRECT_REASON_MIN} characters "
                "(why main cannot wait for the ci/ evidence, e.g. reverting a red main)")
    return None


def after_landing_items(root: Path) -> list[dict]:
    """What --wait reads on main after a landing: each after-landing or main-after-landing
    matrix leg of .github/ci-evidence.toml (a job of its workflow) and each after-landing
    workflow. Each: kind (leg / workflow), file, name (the run's workflowName), job (the
    leg's job display name, or None), cls, jobs {id: display name}."""
    import tomllib
    try:
        ev = tomllib.loads((root / EVIDENCE_FILE).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise LandError(f"{EVIDENCE_FILE}: {e}") from None
    from ci_load_check import jobs as job_blocks
    out = []
    files = {p.name: p.read_text(encoding="utf-8") for p in (root / ".github" / "workflows").glob("*.yml")}
    for leg in ev.get("legs", []):
        cls, f = leg.get("class"), leg.get("workflow")
        if cls not in ("after-landing", "main-after-landing"):
            continue
        if f not in files:
            raise LandError(f"{EVIDENCE_FILE}: leg of {f!r}, which is not a workflow")
        blocks = job_blocks(files[f])
        j = leg.get("job")
        if j not in blocks:
            raise LandError(f"{EVIDENCE_FILE}: leg {f}/{j}: no such job")
        display = _yaml_scalar(blocks[j], "name", 4) or j
        key = leg.get("key", "os")
        display = re.sub(r"\$\{\{\s*matrix\." + re.escape(key) + r"\s*\}\}", str(leg.get("matrix")), display)
        out.append({"kind": "leg", "file": f, "name": _yaml_scalar(files[f], "name", 0) or f,
                    "job": display, "cls": cls, "jobs": {}})
    for f, cls in sorted(ev.get("workflows", {}).items()):
        if cls != "after-landing" or f not in files:
            continue
        blocks = job_blocks(files[f])
        out.append({"kind": "workflow", "file": f, "name": _yaml_scalar(files[f], "name", 0) or f, "job": None,
                    "cls": cls, "jobs": {j: (_yaml_scalar(b, "name", 4) or j) for j, b in blocks.items()}})
    if not out:
        raise LandError(f"no after-landing leg or workflow in {EVIDENCE_FILE} (compared nothing)")
    return out


def after_item_label(item: dict) -> str:
    return f"{item['file']}: {item['job']}" if item["kind"] == "leg" else item["file"]


def after_verdict(item: dict, runs: list[dict], jobs_of) -> tuple[str, str]:
    """(state, why) of one after-landing item on main. state: ok / red / pending.
    `runs`: the runs of the item's workflow for the SHA; `jobs_of(run_id)` lists a run's jobs.
    A leg counts in the newest run that has its job (the push run, or a nightly re-run);
    a workflow counts in its newest run. A missing run or job is pending, never ok."""
    if not runs:
        return "pending", "no run for the SHA yet"
    runs = sorted(runs, key=lambda r: r.get("databaseId", 0), reverse=True)
    if item["kind"] == "leg":
        for run in runs:
            for j in jobs_of(run["databaseId"]):
                if j.get("name") != item["job"]:
                    continue
                rid = run["databaseId"]
                if j.get("status") != "completed":
                    return "pending", f"job {item['job']!r} {j.get('status')} in run {rid}"
                if j.get("conclusion") == "success":
                    return "ok", f"job {item['job']!r} success in run {rid}"
                if j.get("conclusion") in ("cancelled", "skipped"):
                    return "pending", f"job {item['job']!r} {j.get('conclusion')} in run {rid} (no result; re-run it)"
                return "red", f"job {item['job']!r} ended {j.get('conclusion')} in run {rid}"
        return "pending", f"no job {item['job']!r} in runs {[r['databaseId'] for r in runs]} yet"
    run = runs[0]
    rid = run["databaseId"]
    if run.get("status") == "completed" and run.get("conclusion") in ("cancelled", "skipped", "stale"):
        return "pending", f"run {rid} ended {run.get('conclusion')} (no result; re-run it)"
    jobs = jobs_of(rid)
    changes_name, gate_name = item["jobs"].get("changes"), item["jobs"].get("gate")
    ch = [j for j in jobs if j.get("name") == changes_name]
    heavy = [j for j in jobs if j.get("name") not in (changes_name, gate_name)]
    if not ch or ch[0].get("status") != "completed":
        return "pending", f"run {rid}: the changes job has not finished"
    if ch[0].get("conclusion") != "success":
        return "red", f"run {rid}: the changes job ended {ch[0].get('conclusion')}"
    if heavy and all(j.get("status") == "completed" and j.get("conclusion") == "skipped" for j in heavy):
        return "ok", f"run {rid}: did not run (the changes job said run=false)"
    bad = [j for j in heavy if j.get("status") == "completed" and j.get("conclusion") not in ("success", "skipped")]
    if bad:
        return "red", f"job {bad[0].get('name')!r} ended {bad[0].get('conclusion')} in run {rid}"
    if not heavy or any(j.get("status") != "completed" for j in heavy):
        return "pending", f"run {rid}: heavy jobs not finished"
    if any(j.get("conclusion") == "skipped" for j in heavy):
        return "red", f"run {rid}: heavy jobs partly skipped"
    if run.get("status") == "completed" and run.get("conclusion") != "success":
        return "red", f"run {rid} ended {run.get('conclusion')}"
    return "ok", f"run {rid}: heavy jobs success"


def is_subsequence(short: list[str], long: list[str]) -> bool:
    it = iter(long)
    return all(any(x == y for y in it) for x in short)


def merge_conflict_hunks(merged: str) -> str:
    """Resolve the conflict hunks of `git merge-file --diff3` output for an
    append-only list. A hunk where one side only added lines to the base (the base
    is a subsequence of it) becomes the other side followed by those added lines,
    so a line the other side rewrote keeps only its new version; both sides adding
    to an empty base keeps both. Anything else raises LandError."""
    out: list[str] = []
    lines = merged.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if not lines[i].startswith("<<<<<<< "):
            out.append(lines[i])
            i += 1
            continue
        parts: dict[str, list[str]] = {"ours": [], "base": [], "theirs": []}
        cur = "ours"
        i += 1
        while not lines[i].startswith(">>>>>>> "):
            if lines[i].startswith("||||||| "):
                cur = "base"
            elif lines[i].startswith("=======") and lines[i].strip() == "=======":
                cur = "theirs"
            else:
                parts[cur].append(lines[i])
            i += 1
        i += 1
        ours, base, theirs = parts["ours"], parts["base"], parts["theirs"]
        if is_subsequence(base, theirs):
            out += ours + [x for x in theirs if x not in base]
        elif is_subsequence(base, ours):
            out += [x for x in ours if x not in base] + theirs
        else:
            first = next((x for x in base if x.strip()), "").strip()
            raise LandError(f"both sides changed the same lines (not only additions): {first[:80]}")
    return "".join(out)


def changelog_problems(text: str) -> list[str]:
    """Inside [Unreleased]: no line twice and no heading twice (two sides adding
    the same line both survive merge_append)."""
    m = re.search(r"^## \[Unreleased\]\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
    if not m:
        return ["CHANGELOG.md: no [Unreleased] section (compared nothing)"]
    seen: dict[str, int] = {}
    out = []
    for line in m.group(1).splitlines():
        if line.strip():
            seen[line] = seen.get(line, 0) + 1
    for line, n in seen.items():
        if n > 1:
            kind = "heading" if line.startswith("#") else "line"
            out.append(f"CHANGELOG.md [Unreleased]: {kind} appears {n} times: {line[:80]}")
    return out


def blank_integration(text: str) -> str:
    """docs/MODULES.md with the cells of every Integration column emptied."""
    out, col = [], None
    for line in text.split("\n"):
        cells = line.strip().strip("|").split("|") if line.startswith("|") else None
        if cells is None:
            col = None
        elif cells[0].strip() == "Module":
            col = [c.strip() for c in cells].index("Integration") if "Integration" in line else None
        elif col is not None and not set(line) <= set("|- ") and len(cells) > col:
            cells[col] = " "
            line = "|" + "|".join(cells) + "|"
        out.append(line)
    return "\n".join(out)


DOCS_RE = re.compile(r"^(docs/|README[^/]*\.md$|CHANGELOG\.md$)")
# ledgers land.py rebuilds from the tree and checks locally after every rebase (run_checks):
# a main commit that only rewrote them cannot make the landed tree's ledgers stale.
# docs/PUBLIC_API_SNAPSHOT.txt is not here: it depends on the toolchain and the platform and
# only CI's public-api job compares it
LOCAL_LEDGERS = frozenset(("docs/integration-status.md", "docs/integration-levels.md",
                           "docs/oracle-status.md", "docs/wiring-status.md", "docs/coverage/status.md"))


def workflow_input_rules(root: Path) -> list:
    """the pattern lists of .github/paths/*.txt (each: the files a path-gated workflow reads),
    compiled; LandError when there is none or one cannot be read"""
    import changed_paths as cp
    lists = sorted((root / ".github" / "paths").glob("*.txt"))
    if not lists:
        raise LandError("no .github/paths/*.txt: the workflow inputs are unknown")
    out = []
    for f in lists:
        try:
            out.append(cp.compile_patterns(cp.read_patterns(str(f), [])))
        except (cp.PatternError, OSError) as e:
            raise LandError(f"{f.name}: {e}") from None
    return out


def docs_only(paths: list[str], input_rules: list) -> bool:
    """True when every path is a document that no workflow reads: under docs/, a top-level
    README*.md or CHANGELOG.md, and not selected by any .github/paths list (the inputs of the
    path-gated workflows), except the ledgers land.py rebuilds and checks after every rebase
    (LOCAL_LEDGERS). Such a change cannot change what the tests or the gated workflows see.
    False for an empty list (nothing compared)."""
    import changed_paths as cp
    if not paths:
        return False
    for p in paths:
        if not DOCS_RE.match(p):
            return False
        if p in LOCAL_LEDGERS:
            continue
        if any(cp.selected(p, rules) for rules in input_rules):
            return False
    return True


# files whose change can alter how every other file builds or tests
BUILD_WIDE_RE = re.compile(r"((^|/)Cargo\.(toml|lock)$|(^|/)build\.rs$|^\.cargo/|^rust-toolchain|^clippy\.toml$|"
                           r"^deny\.toml$|^src/lib\.rs$|(^|/)mod\.rs$|^scripts/|^\.github/|^include/|"
                           r"^examples/|^benches/|^fuzz/|^bindings/|^unreal-plugin/|^cbindgen\.toml$)")


def overlaps(ours: list[str], theirs: list[str]) -> bool:
    """True when commits main gained since the last preflight can change what our
    commits' preflight saw: they touch a file we touch (documents aside), or a
    build-wide file. Disjoint source changes are left to a compile check here and
    to main's CI, so a landing is not re-tested in full every time main moves."""
    mine = {p for p in ours if not DOCS_RE.match(p)}
    return any(p in mine or BUILD_WIDE_RE.search(p) for p in theirs if not DOCS_RE.match(p))


class LandLock:
    """An exclusive lock per repository and machine (flock on POSIX, msvcrt on Windows),
    held from the first fetch to the push to main. A second landing waits up to `wait`
    seconds, then fails naming the holder (written next to the lock)."""

    def __init__(self, path: Path, wait: float, poll: float = 5.0, label: str = ""):
        self.path, self.wait, self.poll, self.label = path, wait, poll, label
        self.holder = path.with_suffix(".holder")
        self.fh = None

    def _try(self) -> bool:
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)  # msvcrt locks from the current position: always byte 0
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        start = time.monotonic()
        while not self._try():
            if time.monotonic() - start >= self.wait:
                try:
                    who = self.holder.read_text(encoding="utf-8").strip()
                except OSError:
                    who = "unknown holder"
                self.fh.close()
                raise LandError(f"another landing holds {self.path} ({who}); waited "
                                f"{self.wait / 60:.1f} minutes")
            time.sleep(self.poll)
        self.holder.write_text(f"pid {os.getpid()} started {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                               f"{self.label}\n", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        try:
            self.holder.unlink()
        except OSError:
            pass
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()
        return False


def message_problems(message: str) -> list[str]:
    import docs_lint
    out = []
    for i, line in enumerate(message.splitlines(), 1):
        for label, pat in docs_lint.FORBIDDEN:
            for m in pat.finditer(line):
                out.append(f"line {i}: {label} `{m.group(0)}`")
        for name in docs_lint.private_names(line):
            out.append(f"line {i}: private or internal name (`{name}`)")
    return out


class Lander:
    def __init__(self, root: Path, ident: str, *, dry_run: bool = False, no_changelog: bool = False,
                 replace_ci: bool = False, remote: str = "origin", branch: str = "main", log=say,
                 direct_reason: str | None = None):
        if os.path.realpath(root) in FORBIDDEN_ROOTS:
            raise LandError(f"refusing to land in {root}: a forbidden root (the tests' own checkout)")
        self.root, self.id, self.dry_run = root, ident, dry_run
        # the checkout land.py was started in; self.root becomes land.py's own working copy
        # for the landing (own_copy)
        self.checkout, self.source_sha = root, None
        self.checkout_lagging = False   # landed, but `reset --keep` of the checkout was refused
        self.evidence_line = ""
        self.no_changelog, self.remote, self.branch, self.log = no_changelog, remote, branch, log
        self.replace_ci = replace_ci
        m = re.fullmatch(r"(.+?) <([^<>]+)>", IDENTITY)
        if not m:
            raise LandError(f"LAND_IDENTITY {IDENTITY!r} is not `Name <email>`")
        name, email = m.groups()
        problem = identity_problem(name, email)
        if problem:
            raise LandError(f"LAND_IDENTITY {IDENTITY!r} is refused: {problem}")
        self.ident_cfg = ["-c", f"user.name={name}", "-c", f"user.email={email}"]
        # every git process: the pinned git with the forced config, and the ambient
        # identity / repository / program variables dropped
        self.git_argv = git_argv()
        self.git_env = clean_env()
        # tests compare the effective remote URLs with their local bare repository instead of
        # the GitHub forms (set in code, never from the environment)
        self.expected_remote_urls: tuple[str, ...] | None = None
        # where check_law_state clones ALICE-Law from; tests point it at a local fake (set in
        # code, never from the environment)
        self.law_url = LAW_URL
        self.preflight_runs = 0
        self.poll_seconds = 30
        self.gh_cmd: list[str] | None = None   # the pinned gh on first use; tests run a fake gh script here
        # the delays before asking a read again after a transient GitHub failure (GH_TRANSIENT)
        self.gh_retry_seconds: tuple[float, ...] = (10.0, 30.0, 90.0)
        self.max_polls = 240  # 2 hours at 30 s: the 4-OS CI run and the queued heavy runs
        self.after_max_polls = 360  # 3 hours at 30 s: the Mac Intel leg and Miri on main
        self.lock_wait_seconds = LOCK_WAIT_MINUTES * 60
        self.lock_poll_seconds = 5.0
        problem = direct_reason_problem(direct_reason is not None, direct_reason)
        if problem:
            raise LandError(problem)
        self.direct_reason = direct_reason.strip() if direct_reason is not None else None
        self.record: list[str] = []  # the landing record: lane, reason, the runs relied on
        self.original_messages: list[tuple[str, bytes]] = []

    # --- plumbing (overridden in tests) ------------------------------------------

    def git(self, *args: str, check: bool = True) -> str:
        r = subprocess.run([*self.git_argv, *args], cwd=self.root, capture_output=True, text=True, env=self.git_env)
        if check and r.returncode != 0:
            raise LandError(f"git {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout.strip()

    def child_env(self) -> dict[str, str]:
        """the environment of every tool child (preflight, scip_index, the ledger scripts, cargo);
        the cargo build directory stays the checkout's (not the temporary copy's) unless
        CARGO_TARGET_DIR names one"""
        env = {**clean_env(cargo=True), "PYTHONIOENCODING": "utf-8"}   # (clean_env sets it too)
        env.setdefault("CARGO_TARGET_DIR", str(Path(self.checkout) / "target"))
        return env

    def checkout_git(self, *args: str) -> str:
        """git in the checkout land.py was started in (not its own working copy)"""
        r = subprocess.run([*self.git_argv, *args], cwd=self.checkout, capture_output=True, text=True,
                           env=self.git_env)
        if r.returncode != 0:
            raise LandError(f"git {' '.join(args)} in {self.checkout} failed: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout.strip()

    @contextlib.contextmanager
    def own_copy(self):
        """The landing works in a detached working copy (`git worktree add --detach`) made from
        the checkout's HEAD, read once
        (the source SHA); the checkout is not read again except by check_checkout_unmoved."""
        if self.checkout_git("status", "--porcelain"):
            raise LandError("the checkout has uncommitted changes")
        self.source_sha = self.checkout_git("rev-parse", "HEAD")
        d = tempfile.mkdtemp(prefix="land-wt-")
        wt = os.path.join(d, "wt")
        self.checkout_git("worktree", "add", "-q", "--detach", wt, self.source_sha)
        self.record.append(f"source SHA: {self.source_sha} (the checkout's HEAD at the start)")
        self.root = Path(wt)
        try:
            yield
        finally:
            self.root = self.checkout
            subprocess.run([*self.git_argv, "worktree", "remove", "--force", wt], cwd=self.checkout,
                           capture_output=True, env=self.git_env)
            subprocess.run([*self.git_argv, "worktree", "prune"], cwd=self.checkout, capture_output=True,
                           env=self.git_env)
            shutil.rmtree(d, ignore_errors=True)

    def check_checkout_unmoved(self) -> None:
        now = self.checkout_git("rev-parse", "HEAD")
        if now != self.source_sha:
            raise LandError(f"the checkout's HEAD moved during the landing ({self.source_sha[:7]} -> {now[:7]}): "
                            "nothing was pushed to main; land.py works in its own working copy, but a moved "
                            "checkout means something else is acting on it")

    def sh(self, *cmd: str) -> None:
        self.log("$ " + " ".join(cmd))
        r = subprocess.run(list(cmd), cwd=self.root, env=self.child_env())
        if r.returncode != 0:
            raise LandError(f"`{' '.join(cmd)}` failed (exit {r.returncode})")

    def lib_test_gate(self) -> None:
        """New source files must bring lib tests (scripts/lib_test_gate.py): the
        per-push coverage ratchet runs `--lib` only and would turn red after the land."""
        gate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib_test_gate.py")
        r = subprocess.run([sys.executable, gate, self.upstream(), "--root", self.root],
                           cwd=self.root, capture_output=True, text=True, env=clean_env(cargo=True))
        for line in r.stdout.splitlines():
            self.log(line)
        if r.returncode != 0:
            errors = [l[len("error: "):] for l in r.stdout.splitlines() if l.startswith("error: ")]
            raise LandError("lib-test gate failed:\n  " + "\n  ".join(errors or [r.stderr.strip()]))

    def run_preflight(self) -> None:
        self.preflight_runs += 1
        self.sh("bash", "scripts/preflight.sh", "--fast")

    def run_compile_check(self) -> None:
        # clippy with -D warnings, as CI: a caller removed on each side can leave
        # dead code that only the lint (not a plain check) refuses
        self.sh("cargo", "clippy", "-q", "--all-targets", "--features", NATIVE, "--", "-D", "warnings")

    def regenerate(self) -> None:
        self.sh("bash", "scripts/scip_index.sh")
        for _, argv, _ in WRITERS:
            self.sh(sys.executable, *argv[1:])
        self.regenerate_public_api()

    def regenerate_public_api(self) -> None:
        r = subprocess.run(["cargo", f"+{NIGHTLY}", "public-api", "--features", NATIVE, "--simplified"],
                           cwd=self.root, capture_output=True, text=True, env=clean_env(cargo=True))
        if r.returncode != 0 or not r.stdout.strip():
            raise LandError(f"cargo +{NIGHTLY} public-api failed (install the toolchain and cargo-public-api): "
                            f"{r.stderr.strip()[-300:]}")
        (self.root / "docs/PUBLIC_API_SNAPSHOT.txt").write_text(r.stdout, encoding="utf-8")

    def run_checks(self) -> None:
        for cmd in ([sys.executable, "scripts/integration_levels.py", "--check"],
                    [sys.executable, "scripts/readme_sync.py", "--check"],
                    [sys.executable, "scripts/docs_lint.py", "--check"],
                    [sys.executable, "scripts/scip_reach.py", "--check-baseline"],
                    [sys.executable, "scripts/gen-oracle-status.py", "--check"],
                    [sys.executable, "scripts/wiring_guard.py"],
                    [sys.executable, "scripts/coverage_check.py"]):
            self.sh(*cmd)
        self.check_changelog()

    def check_changelog(self) -> None:
        p = self.root / "CHANGELOG.md"
        problems = changelog_problems(p.read_text(encoding="utf-8")) if p.exists() else []
        if problems:
            raise LandError("\n  ".join(problems))

    def gh(self) -> list[str]:
        if self.gh_cmd is None:
            self.gh_cmd = list(TOOL_OVERRIDE.get("gh") or [pinned_tool(GH_CANDIDATES)])
        return self.gh_cmd

    def gh_json(self, *args: str):
        """`gh <args>` parsed as JSON; LandError when gh fails (never read as an empty list).

        Every caller reads (run list / run view / api GET), so a failure GitHub reports as its own
        (an HTTP 5xx, a dropped connection or a timeout) is asked again after each delay of
        gh_retry_seconds; any other failure (4xx, auth, a bad argument) and the last transient one
        stop the landing, naming the attempts. Each retry is logged with the failure GitHub reported
        and the attempt number, so a flaky API shows in the landing record."""
        attempts = len(self.gh_retry_seconds) + 1
        for attempt in range(1, attempts + 1):
            out = subprocess.run([*self.gh(), *args], capture_output=True, text=True, env=gh_env(),
                                 encoding="utf-8", errors="replace")
            transient = GH_TRANSIENT.search(out.stderr) if out.returncode != 0 else None
            if transient is None or attempt == attempts:
                break
            delay = self.gh_retry_seconds[attempt - 1]
            self.log(f"gh {' '.join(args[:4])}: {transient.group(0)} (attempt {attempt}/{attempts}), "
                     f"asking again in {delay:g} s: {out.stderr.strip()[:120]}")
            time.sleep(delay)
        if out.returncode != 0:
            tried = f" (after {attempt} attempts)" if attempt > 1 else ""
            raise LandError(f"gh {' '.join(args[:4])} failed{tried}: {out.stderr.strip()[:300]}")
        try:
            return json.loads(out.stdout or "null")
        except ValueError as e:
            raise LandError(f"gh {' '.join(args[:3])}: not JSON: {e}") from None

    def list_branch_runs(self, slug: str, ref: str) -> list[dict]:
        runs = self.gh_json("run", "list", "--repo", repo_arg(slug), "--branch", ref, "--limit", "100", "--json",
                            "workflowName,headSha,databaseId,status,conclusion,createdAt,event")
        if not isinstance(runs, list):
            raise LandError("gh run list did not return a list")
        return runs

    def run_jobs(self, slug: str, run_id: int) -> list[dict]:
        data = self.gh_json("run", "view", str(run_id), "--repo", repo_arg(slug), "--json", "jobs")
        return [{"name": j.get("name"), "status": j.get("status"), "conclusion": j.get("conclusion")}
                for j in (data or {}).get("jobs", [])]

    def dispatch(self, slug: str, wf_file: str, ref: str) -> None:
        r = subprocess.run([*self.gh(), "workflow", "run", wf_file, "--repo", repo_arg(slug), "--ref", ref],
                           capture_output=True, text=True, env=gh_env(), encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise LandError(f"gh workflow run {wf_file} --ref {ref} failed: {r.stderr.strip()[:300]}")

    def slug(self) -> str:
        """owner/repo from the configured URL of the remote (as written, before insteadOf)"""
        repo = self.git("config", "--get", f"remote.{self.remote}.url", check=False)
        m = re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
                         r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?", repo.strip())
        return m.group(1) if m else repo

    def check_gh_auth(self) -> None:
        """gh, run with gh_env(), is logged in to github.com; refuse to start otherwise"""
        # --active: the account gh uses (another, inactive login on the host may be unreadable)
        r = subprocess.run([*self.gh(), "auth", "status", "--active", "--hostname", GH_HOSTNAME], capture_output=True,
                           text=True, env=gh_env(), encoding="utf-8", errors="replace")
        text = (r.stdout + r.stderr).strip()
        if r.returncode != 0 or GH_HOSTNAME not in text:
            raise LandError(f"gh is not logged in to {GH_HOSTNAME} (gh auth status --active --hostname {GH_HOSTNAME} "
                            f"exit {r.returncode}): {text[:300]}")

    def check_remote(self) -> None:
        """The effective fetch and push URLs of the remote (after insteadOf / pushInsteadOf and
        any pushurl) must be the repository's GitHub URL for slug() (the one gh is told), or the
        test's expected URLs; refuse otherwise, naming both."""
        if self.expected_remote_urls is not None:
            allowed = self.expected_remote_urls
        else:
            slug = self.slug()
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", slug):
                raise LandError(f"remote {self.remote!r}: {slug!r} is not a GitHub owner/repo URL")
            allowed = tuple(f.format(slug=slug) for f in GITHUB_URL_FORMS)
        for label, args in (("push URL", ("remote", "get-url", "--push", self.remote)),
                            ("fetch URL", ("ls-remote", "--get-url", self.remote))):
            url = self.git(*args)
            problem = remote_url_problem(url, allowed)
            if problem:
                raise LandError(f"refusing remote {self.remote!r}: its effective {label} {problem}")

    def wait_evidence(self, ref: str, sha: str) -> None:
        """Wait until every workflow expected on a ci/** push has its evidence for `sha` on
        `ref` (see run_verdict); re-create a missing or dropped run with `gh workflow run`
        (at most MAX_DISPATCH times per workflow); raise on a red result, on a workflow that
        cannot be re-created, and when nothing was compared."""
        expected = expected_workflows(self.root)
        slug = self.slug()
        self.log(f"waiting on {len(expected)} workflows for {sha[:7]} on {ref}: "
                 + ", ".join(f"{w['file']} ({w['cls']})" for w in expected))
        started = time.monotonic()
        # a missing run is re-created NO_RUN_GRACE seconds after the start or after its last re-creation
        last_seen = {w["file"]: started for w in expected}
        dispatched = {w["file"]: 0 for w in expected}
        states: dict[str, tuple[str, str]] = {}
        for _ in range(self.max_polls):
            runs = self.list_branch_runs(slug, ref)
            states = {}
            chosen: dict[str, dict | None] = {}
            for w in expected:
                run = newest_run(runs, w["name"], sha)
                chosen[w["file"]] = run
                state, why = run_verdict(w, run, None)
                if state == "need-jobs":
                    state, why = run_verdict(w, run, self.run_jobs(slug, run["databaseId"]))
                states[w["file"]] = (state, why)
            red = {f: why for f, (s, why) in states.items() if s == "red"}
            if red:
                raise LandError(f"CI on {ref} is red for {sha[:7]}:\n  "
                                + "\n  ".join(f"{f}: {why}" for f, why in sorted(red.items())))
            redo = [f for f, (s, _) in states.items()
                    if s == "dropped" or (s == "missing" and time.monotonic() - last_seen[f] > NO_RUN_GRACE)]
            if redo:
                at = self.remote_sha(ref)
                if at != sha:
                    raise LandError(f"{ref} on {self.remote} points at {at[:7] or 'nothing'}, not {sha[:7]}: "
                                    "the ref moved; a re-created run would test another commit")
                for f in redo:
                    if dispatched[f] >= MAX_DISPATCH:
                        raise LandError(f"{f}: no evidence for {sha[:7]} on {ref} after {MAX_DISPATCH} "
                                        f"re-created runs ({states[f][1]}): not treated as success")
                    dispatched[f] += 1
                    self.log(f"{f}: {states[f][1]}; re-creating it: gh workflow run {f} --ref {ref}")
                    self.dispatch(slug, f, ref)
                    last_seen[f] = time.monotonic()
            ok = [f for f, (s, _) in states.items() if s == "ok"]
            if len(ok) == len(expected) and ok:
                for f in sorted(ok):
                    self.log(f"  {f}: {states[f][1]}")
                self.log(f"CI on {ref}: evidence complete for {len(ok)} workflows")
                for f in sorted(ok):
                    run = chosen[f] or {}
                    concl = run.get("conclusion") or run.get("status")
                    self.record.append(f"evidence: {f} run {run.get('databaseId')} {concl}")
                return
            if not any(s != "missing" for s, _ in states.values()) and (at := self.remote_sha(ref)) != sha:
                raise LandError(f"{ref} on {self.remote} points at {at[:7] or 'nothing'}, not {sha[:7]}: "
                                "the push did not reach it, or the ref moved (no CI run to wait for)")
            time.sleep(self.poll_seconds)
        waiting = "\n  ".join(f"{f}: {why}" for f, (s, why) in sorted(states.items()) if s != "ok")
        raise LandError(f"CI on {ref} did not give its evidence in time:\n  {waiting}")

    def remote_sha(self, ref: str) -> str:
        """The commit `ref` points at on the remote now ("" when it does not exist)."""
        out = self.git("ls-remote", self.remote, f"refs/heads/{ref}", check=False).split()
        return out[0] if out else ""

    def wait_after(self, sha: str) -> tuple[int, list[str]]:
        """Read the after-landing set (after_landing_items) for `sha` on main. Returns
        (exit code, report lines): 0 every item green, EXIT_RED when one is red (the
        message says REVERT OR FIX NOW), EXIT_PENDING when one did not finish in time."""
        items = after_landing_items(self.root)
        self.check_gh_auth()
        slug = self.slug()
        states: dict[str, tuple[str, str]] = {}
        for _ in range(self.after_max_polls):
            runs = self.list_branch_runs(slug, self.branch)
            cache: dict[int, list[dict]] = {}

            def jobs_of(rid: int) -> list[dict]:
                if rid not in cache:
                    cache[rid] = self.run_jobs(slug, rid)
                return cache[rid]
            for it in items:
                mine = [r for r in runs if r.get("workflowName") == it["name"] and r.get("headSha") == sha]
                states[after_item_label(it)] = after_verdict(it, mine, jobs_of)
            if any(s == "red" for s, _ in states.values()) or all(s == "ok" for s, _ in states.values()):
                break
            time.sleep(self.poll_seconds)
        lines = []
        for label, (s, why) in states.items():
            if s == "red":
                lines.append(f"REVERT OR FIX NOW: {label}: {why} (main at {sha[:7]})")
            else:
                lines.append(f"{s}: {label}: {why}")
        if any(s == "red" for s, _ in states.values()):
            return EXIT_RED, lines
        if all(s == "ok" for s, _ in states.values()):
            return 0, lines
        return EXIT_PENDING, lines

    # --- steps -------------------------------------------------------------------

    def upstream(self) -> str:
        return f"{self.remote}/{self.branch}"

    def fetch(self) -> None:
        self.git("fetch", "-q", self.remote, self.branch)

    def commits(self) -> list[str]:
        return self.git("rev-list", "--reverse", f"{self.upstream()}..HEAD").split()

    def raw(self, *args: str) -> bytes:
        """git's stdout as bytes, unstripped (self.git strips, which changes a first or a last line)"""
        r = subprocess.run([*self.git_argv, *args], cwd=self.root, capture_output=True, env=self.git_env)
        if r.returncode != 0:
            raise LandError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout

    def paths(self, rev_range: str) -> list[str]:
        """the paths a range changes, read NUL-separated: a path with a space, a quote or a non-ASCII
        character is the path itself, not git's quoted form; a rename lists both names"""
        out = self.raw("diff", "--name-only", "--no-renames", "-z", rev_range)
        return [p.decode("utf-8", "surrogateescape") for p in out.split(b"\0") if p]

    def check_commits(self) -> list[str]:
        if self.git("status", "--porcelain"):
            raise LandError("the checkout has uncommitted changes")
        shas = self.commits()
        if not shas:
            raise LandError(f"nothing to land: HEAD has no commit beyond {self.upstream()} (compared nothing)")
        problems = []
        for sha in shas:
            an, ae, cn, ce = self.git("log", "-1", "--format=%an%x00%ae%x00%cn%x00%ce", sha).split("\0")
            for field, name, email in (("author", an, ae), ("committer", cn, ce)):
                problem = identity_problem(name, email)
                if problem:
                    problems.append(f"{sha[:7]}: {field} `{name} <{email}>` is refused: {problem}")
            if f"{an} <{ae}>" != IDENTITY:
                problems.append(f"{sha[:7]}: author `{an} <{ae}>` is not `{IDENTITY}`")
            msg = self.git("log", "-1", "--format=%B", sha)
            problems += [f"{sha[:7]}: {p}" for p in message_problems(msg)]
            if SIGNATURE_RE.search(msg):
                problems.append(f"{sha[:7]}: signature / generated-by line in the message")
        paths = self.paths(f"{self.upstream()}...HEAD")
        if any(p.startswith("src/") for p in paths) and "CHANGELOG.md" not in paths and not self.no_changelog:
            problems.append("src/ changes without a CHANGELOG.md line (pass --no-changelog for a pure refactor)")
        return problems

    def owns_ref(self, sha: str) -> bool:
        """A ci/<id> head we pushed earlier (a previous attempt of the same commits)."""
        r = subprocess.run([*self.git_argv, "cherry", "HEAD", sha], cwd=self.root, capture_output=True, text=True,
                           env=self.git_env)
        return r.returncode == 0 and all(line.startswith("-") for line in r.stdout.splitlines())

    def rebase(self) -> None:
        """Rebase on the upstream with a fixed committer; settle ledger conflicts."""
        # --force-rebase replays every commit, so each one gets the fixed committer
        # attributes are read from the empty tree: a base that still carries
        # `CHANGELOG.md merge=union` must not merge it behind merge_append()'s back
        # write the empty tree object EMPTY_TREE names (mktree reads the entries from stdin)
        subprocess.run([*self.git_argv, "mktree"], cwd=self.root, input="", capture_output=True, text=True,
                       env=self.git_env)
        r = subprocess.run([*self.git_argv, *self.ident_cfg, f"--attr-source={EMPTY_TREE}", "rebase", "--force-rebase",
                            self.upstream()],
                           cwd=self.root, capture_output=True, text=True, env=self.git_env)
        while r.returncode != 0:
            conflicted = [p.decode("utf-8", "surrogateescape")
                          for p in self.raw("diff", "--name-only", "--diff-filter=U", "-z").split(b"\0") if p]
            if not conflicted:
                self.git("rebase", "--abort", check=False)
                raise LandError(f"rebase failed: {r.stderr.strip()}")
            other = [p for p in conflicted if p not in GENERATED and p not in SEMI_GENERATED and p not in APPEND_ONLY]
            if other:
                self.git("rebase", "--abort", check=False)
                raise LandError(f"rebase conflict outside the generated ledgers: {other}")
            for p in conflicted:
                if p in SEMI_GENERATED:
                    self.merge_semi(p)
                elif p in APPEND_ONLY:
                    self.merge_append(p)
                else:
                    self.git("checkout", "--ours", "--", p)   # main's copy; regenerated after the rebase
                self.git("add", "--", p)
            # commit the resolved pick ourselves, with its message verbatim: `rebase --continue`
            # would commit it with the default cleanup, which drops body lines that start with
            # `#` (e.g. `#[path = ...]`). -C REBASE_HEAD keeps the author, the date and the message;
            # the committer is the fixed identity
            c = subprocess.run([*self.git_argv, *self.ident_cfg, f"--attr-source={EMPTY_TREE}", "commit", "-q",
                                "--no-verify", "--cleanup=verbatim", "-C", "REBASE_HEAD"],
                               cwd=self.root, capture_output=True, text=True, env=self.git_env)
            if c.returncode != 0:
                self.git("rebase", "--abort", check=False)
                raise LandError(f"committing the resolved pick failed: {(c.stderr or c.stdout).strip()[:300]}")
            r = subprocess.run([*self.git_argv, *self.ident_cfg, f"--attr-source={EMPTY_TREE}", "-c", "core.editor=true",
                                "rebase", "--continue"],
                               cwd=self.root, capture_output=True, text=True, env=self.git_env)

    def merge_stages(self, path: str, *, diff3: bool = False, transform=lambda t: t) -> bool:
        """3-way merge of the conflict stages of `path` (main, base, branch) after
        `transform`; writes the result and returns whether it merged cleanly."""
        import tempfile
        stages = {}
        for n in (1, 2, 3):
            r = subprocess.run([*self.git_argv, "show", f":{n}:{path}"], cwd=self.root, capture_output=True, text=True,
                               env=self.git_env)
            stages[n] = transform(r.stdout) if r.returncode == 0 else ""
        with tempfile.TemporaryDirectory() as d:
            files = []
            for n in (2, 1, 3):  # ours (main), base, theirs (branch)
                f = Path(d) / f"{n}"
                f.write_text(stages[n], encoding="utf-8")
                files.append(str(f))
            r = subprocess.run([*self.git_argv, "merge-file", "-p", *(["--diff3"] if diff3 else []), *files],
                               capture_output=True, text=True, env=self.git_env)
        (self.root / path).write_text(r.stdout, encoding="utf-8")
        return r.returncode == 0

    def merge_append(self, path: str) -> None:
        """3-way merge of an append-only list; conflict hunks go to merge_conflict_hunks()."""
        if self.merge_stages(path, diff3=True):
            return
        p = self.root / path
        try:
            p.write_text(merge_conflict_hunks(p.read_text(encoding="utf-8")), encoding="utf-8")
        except LandError as e:
            self.git("rebase", "--abort", check=False)
            raise LandError(f"{path}: {e}") from None

    def merge_semi(self, path: str) -> None:
        """3-way merge of a hand-written file with its generated column blanked
        (the regeneration refills it); stop when the hand-written parts clash."""
        if not self.merge_stages(path, transform=blank_integration):
            self.git("rebase", "--abort", check=False)
            raise LandError(f"{path}: the hand-written parts conflict (not only the Integration column)")

    def commit_regenerated(self) -> None:
        names = sorted(set(self.git("diff", "--name-only", "HEAD").splitlines())
                       | set(self.git("ls-files", "--others", "--exclude-standard").splitlines()))
        names = [n for n in names if n]
        if not names:
            return
        stray = [p for p in names if p not in GENERATED and p not in SEMI_GENERATED]
        if stray:
            raise LandError(f"regeneration touched files that are not ledgers: {stray}")
        self.git("add", "--", *names)
        self.git(*self.ident_cfg, "commit", "-q", "--author", IDENTITY, "-m", REGEN_MESSAGE)

    def api_commit_identity(self, sha: str) -> tuple[str, str, str, str]:
        """(author name, author email, committer name, committer email) of `sha` from the
        GitHub API (`gh api repos/<slug>/commits/<sha>`)"""
        data = self.gh_json("api", "--hostname", GH_HOSTNAME, f"repos/{self.slug()}/commits/{sha}")
        try:
            c = data["commit"]
            return c["author"]["name"], c["author"]["email"], c["committer"]["name"], c["committer"]["email"]
        except (TypeError, KeyError) as e:
            raise LandError(f"gh api commits/{sha[:7]}: no author / committer ({e})") from None

    def identity_rows(self, rev_range: str, source: str) -> list[tuple[str, str, str, str, str, str]]:
        rows = []
        for c in self.git("rev-list", "--reverse", rev_range).split():
            an, ae, cn, ce = self.git("log", "-1", "--format=%an%x00%ae%x00%cn%x00%ce", c).split("\0")
            rows.append((c, an, ae, cn, ce, source))
        return rows

    def check_push_identity(self, before: str) -> None:
        """Before the push: every commit about to reach main (before..HEAD, after the
        rebase) has author and committer IDENTITY and no denied identity; refuse otherwise."""
        problems = landed_identity_problems(self.identity_rows(f"{before}..HEAD", "local"), IDENTITY)
        if problems:
            raise LandError(f"refusing to push to {self.branch}: commit identity of {before[:7]}..HEAD:\n  "
                            + "\n  ".join(problems))

    def check_law_state(self, sha: str) -> None:
        """Before the push to main: whether landing `sha` moves ALICE-Law's conformance state.
        A blob-less clone of Law's main (in a temporary directory, with the pinned git) runs its
        scripts/land_law_state.py with this repository's own working copy at `sha` as the
        candidate. Exit 0 (the state does not move) lands; exit 1 (it moves: the Law state commit
        has to land first) and exit 2 (not computed), a Law that cannot be cloned and a missing
        script refuse the landing (LandError, nothing pushed)."""
        d = tempfile.mkdtemp(prefix="land-law-")
        law = os.path.join(d, "law")
        nothing = f"nothing was pushed to {self.branch}"
        def law_git(*args: str) -> str:
            r = subprocess.run([*self.git_argv, *args], cwd=law, capture_output=True, text=True, env=self.git_env)
            if r.returncode != 0:
                raise LandError(f"ALICE-Law's main could not be read from {self.law_url} (git {' '.join(args[:3])}: "
                                f"{(r.stderr or r.stdout).strip()[:300]}): the Law state cannot be checked; {nothing}")
            return r.stdout.strip()

        try:
            # a blob-less partial clone of main, made as init + remote + fetch so the remote's
            # effective URLs (after insteadOf / pushInsteadOf) can be checked before git reaches it
            os.makedirs(law)
            law_git("init", "-q")
            law_git("remote", "add", "origin", self.law_url)
            for label, args in (("fetch URL", ("ls-remote", "--get-url", "origin")),
                                ("push URL", ("remote", "get-url", "--push", "origin"))):
                url = law_git(*args)
                if url != self.law_url:
                    raise LandError(f"ALICE-Law's {label} resolves to {url!r}, not {self.law_url} (an insteadOf "
                                    f"in the git config?): the Law state cannot be checked; {nothing}")
            law_git("config", "remote.origin.promisor", "true")
            law_git("config", "remote.origin.partialclonefilter", "blob:none")
            law_git("fetch", "-q", "--filter=blob:none", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main")
            law_git("checkout", "-q", "--detach", "refs/remotes/origin/main")
            law_sha = law_git("rev-parse", "HEAD")
            if not re.fullmatch(r"[0-9a-f]{40}", law_sha):
                raise LandError(f"ALICE-Law's main has no commit: the Law state cannot be checked; {nothing}")
            script = os.path.join(law, *LAW_STATE_SCRIPT)
            if not os.path.isfile(script):
                raise LandError(f"ALICE-Law {law_sha[:12]} has no {'/'.join(LAW_STATE_SCRIPT)}: "
                                f"the Law state cannot be checked; {nothing}")
            gh = self.gh()
            p = subprocess.run([sys.executable, script, "--candidate", f"{LAW_CANDIDATE}={self.root}@{sha}"],
                               cwd=law, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env=law_state_env(gh[0] if len(gh) == 1 else None))
            out = "\n".join(x for x in (p.stdout.strip(), p.stderr.strip()) if x)[:3000]
            summary = next((x for x in reversed(out.splitlines()) if x.startswith("land_law_state:")), "")
            # the exit code counts only with the script's own verdict line: an exit 1 from a
            # traceback is not a computed move, and an exit 0 that printed nothing is not "unmoved"
            verdict = {ln.strip() for ln in out.splitlines()}
            said = {0: "land_law_state: state moves: no", 1: "land_law_state: state moves: yes"}.get(p.returncode)
            if said is not None and said not in verdict:
                raise LandError(f"ALICE-Law's state for {sha[:7]} could not be computed (land_law_state exit "
                                f"{p.returncode} without `{said}`, Law main {law_sha[:12]}):\n{out}\n{nothing}")
            if p.returncode == 0:
                self.log(f"ALICE-Law state: landing {sha[:7]} does not move it (Law main {law_sha[:12]}) {summary}")
                self.record.append(f"Law state: unmoved (compared with ALICE-Law main {law_sha})")
                return
            if p.returncode == 1:
                raise LandError(
                    f"landing {sha[:7]} moves ALICE-Law's conformance state (compared with Law main {law_sha[:12]}):\n"
                    f"{out}\n"
                    f"(the move may also come from another repository's main that Law has not recorded yet; "
                    f"the output above names the crates that move, and the same --write records them)\n"
                    f"record the Law state first: in a clone of ALICE-Law at {law_sha[:12]} run\n"
                    f"  python3 {'/'.join(LAW_STATE_SCRIPT)} --candidate {LAW_CANDIDATE}=<a clone of this "
                    f"repository>@{sha} --write\n"
                    f"land that commit on ALICE-Law's main, then run land.py again; {nothing}")
            raise LandError(f"ALICE-Law's state for {sha[:7]} could not be computed (land_law_state exit "
                            f"{p.returncode}, Law main {law_sha[:12]}):\n{out}\n{nothing}")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def verify_landed_identity(self, before: str, sha: str) -> None:
        """After the push (the backstop of check_push_identity): read every landed commit
        (before..sha on the fetched remote branch) with git, and the tip from the GitHub API,
        and require author and committer to be IDENTITY and not denied. A mismatch raises
        "REVERT OR FIX NOW" (main has moved: fix forward, no history rewrite)."""
        rows = self.identity_rows(f"{before}..{self.upstream()}", "remote")
        try:
            rows.append((sha, *self.api_commit_identity(sha), "GitHub API"))
        except LandError as e:
            rows.append((sha, "?", f"unreadable: {e}", "?", "?", "GitHub API"))
        for c, an, ae, cn, ce, source in rows:
            self.record.append(f"landed identity: {c[:7]} author {an} <{ae}> committer {cn} <{ce}> ({source})")
        problems = landed_identity_problems(rows, IDENTITY)
        if problems:
            for line in self.record:
                self.log(f"  {line}")
            raise LandError(f"REVERT OR FIX NOW: landed commit identity on {self.branch} "
                            f"({before[:7]}..{sha[:7]}):\n  " + "\n  ".join(problems)
                            + "\n  fix forward with a new commit (a revert or a correction); do not rewrite "
                            f"the history of {self.branch}")

    def lock(self) -> LandLock:
        base = os.environ.get(LOCK_DIR_ENV) or str(Path.home() / ".cache" / "alice-land")
        try:
            name = self.slug()
        except LandError:
            name = self.root.name
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "repo"
        return LandLock(Path(base) / f"{name}.lock", self.lock_wait_seconds, self.lock_poll_seconds,
                        label=f"landing {self.id} in {self.root}")

    def needs_rebase(self) -> bool:
        """HEAD is not on the upstream, or a commit of ours lacks the fixed committer (a
        rebase that is not needed would give the commits a new date and a new SHA)."""
        r = subprocess.run([*self.git_argv, "merge-base", "--is-ancestor", self.upstream(), "HEAD"], cwd=self.root,
                           env=self.git_env)
        if r.returncode != 0:
            return True
        who = self.git("log", "--format=%cn <%ce>", f"{self.upstream()}..HEAD").splitlines()
        return any(w != IDENTITY for w in who)

    def prepare(self, verify: str) -> str:
        """Rebase when needed, rebuild and commit the ledgers, run the checks, then
        `verify`: preflight / compile / none. Returns HEAD."""
        if self.needs_rebase():
            self.rebase()
        self.regenerate()
        self.commit_regenerated()
        self.run_checks()
        if verify == "preflight":
            self.run_preflight()
        elif verify == "compile":
            self.run_compile_check()
        return self.git("rev-parse", "HEAD")

    def raw_message(self, sha: str) -> bytes:
        """the message of `sha` exactly as stored (the commit object after its header)"""
        r = subprocess.run([*self.git_argv, "cat-file", "commit", sha], cwd=self.root, capture_output=True,
                           env=self.git_env)
        if r.returncode != 0:
            raise LandError(f"git cat-file commit {sha[:7]} failed: {r.stderr.decode('utf-8', 'replace').strip()}")
        head, sep, body = r.stdout.partition(b"\n\n")
        return body if sep else b""

    def our_messages(self) -> list[tuple[str, bytes]]:
        """(sha, message) of the commits HEAD has beyond the upstream, oldest first, without
        the ledger commits land.py makes itself"""
        out = []
        for sha in self.git("rev-list", "--reverse", f"{self.upstream()}..HEAD").split():
            msg = self.raw_message(sha)
            if msg.rstrip(b"\n") != REGEN_MESSAGE.encode():
                out.append((sha, msg))
        return out

    def check_messages(self) -> None:
        """Before every push: the messages of the commits about to be pushed are byte-identical,
        in order, to those of the commits as they were before any rebase (land.py never
        rewrites a message). Refuse otherwise, naming the commit and the first differing line,
        and when nothing was compared."""
        import difflib
        now = self.our_messages()
        before = self.original_messages
        if not before or not now:
            raise LandError(f"commit messages: compared nothing ({len(before)} before the rebase, {len(now)} now)")
        if len(now) != len(before):
            raise LandError(f"commit messages: {len(before)} commits before the rebase, {len(now)} now "
                            "(a commit was dropped or added): refusing to push")
        for (old_sha, old), (new_sha, new) in zip(before, now):
            if old != new:
                a = old.decode("utf-8", "replace").splitlines(keepends=True)
                b = new.decode("utf-8", "replace").splitlines(keepends=True)
                first = next((f"{ln!r}" for ln in difflib.ndiff(a, b) if ln[:1] in "-+"), "(line ends)")
                raise LandError(f"commit message of {new_sha[:7]} (was {old_sha[:7]}) differs from before the "
                                f"rebase; first difference: {first}: refusing to push")
        self.log(f"commit messages: {len(now)} byte-identical to before the rebase")

    def push_evidence(self, sha: str, first: bool) -> None:
        """Push `sha` to ci/<id> and wait for the expected set (wait_evidence)."""
        ref = f"ci/{self.id}"
        self.record = [l for l in self.record if not l.startswith("evidence: ")]
        if first:
            existing = self.git("ls-remote", self.remote, f"refs/heads/{ref}").split()
            if existing and existing[0] == sha:
                # an earlier run of this script pushed the same commit and stopped while
                # waiting: wait for (or take the result of) that run instead of pushing again
                self.log(f"{ref} already holds {sha[:7]}: reusing its CI runs")
                self.wait_evidence(ref, sha)
                return
            if existing:
                self.git("fetch", "-q", self.remote, existing[0], check=False)
                r = subprocess.run([*self.git_argv, "merge-base", "--is-ancestor", existing[0], "HEAD"], cwd=self.root,
                                   env=self.git_env)
                if r.returncode != 0 and not self.owns_ref(existing[0]) and not self.replace_ci:
                    raise LandError(f"{ref} holds other commits; pick another --id, or pass --replace-ci "
                                    "if it is your own earlier attempt (e.g. amended after a red CI)")
        # our own ci/<id>: a re-evidence push replaces the earlier attempt (and its pending run)
        self.check_remote()
        self.check_messages()
        if self.git("rev-parse", "HEAD") != sha:
            raise LandError(f"HEAD is not the prepared {sha[:7]}: nothing pushed to {ref}")
        self.git("push", "-q", "--force", self.remote, f"{sha}:refs/heads/{ref}")
        self.wait_evidence(ref, sha)

    def land(self) -> str:
        with self.lock():
            with self.own_copy():
                sha = self._land()
            if not self.dry_run:
                # the checkout follows the landing, as when land.py rebased it in place; `--keep`
                # refuses (and changes nothing) when the checkout has changes in files the move
                # would touch: the landing stands, the checkout is reported as behind
                try:
                    self.checkout_git("reset", "-q", "--keep", sha)
                except LandError as e:
                    self.checkout_lagging = True
                    self.log(f"landed {sha[:7]}, but the checkout stays at {self.source_sha[:7]}: {e}; "
                             f"put its changes away and run `git reset --keep {sha[:7]}` there")
            return sha

    def patch_ids(self, rev_range: str, skip: set[str]) -> list[str]:
        """`git patch-id --stable` of each commit in rev_range that is not the ledger commit,
        over the paths outside `skip`"""
        out = []
        for c in self.git("rev-list", "--reverse", rev_range).split():
            if self.git("log", "-1", "--format=%s", c) == REGEN_MESSAGE:
                continue
            spec = ["--", "."] + [f":(exclude,literal){p}" for p in sorted(skip)]
            show = subprocess.run([*self.git_argv, "show", "--format=", "--no-color", c, *spec], cwd=self.root,
                                  capture_output=True, env=self.git_env)
            pid = subprocess.run([*self.git_argv, "patch-id", "--stable"], cwd=self.root, input=show.stdout,
                                 capture_output=True, env=self.git_env)
            out.append(pid.stdout.decode().split()[0] if pid.stdout.strip() else "(empty)")
        return out

    def evidence_rule(self, lane: str, evidence_sha: str | None, evidence_base: str | None,
                      rounds: int, sha: str) -> str:
        """The rule the push to main stands on, or LandError: the landed SHA is the evidence SHA,
        or main gained documents only since the evidence and our commits carry the evidence's
        patches with every other difference in the paths main gained or in the ledgers, and with
        each line main or the evidence added to a path main gained still in the landed file"""
        if lane == "direct":
            return "direct"
        if self.dry_run:
            return "dry-run"
        if evidence_sha is None or evidence_base is None:
            raise LandError("no evidence SHA on the ci lane: nothing pushed to main")
        tip = self.git("ls-remote", self.remote, f"refs/heads/ci/{self.id}").split()
        if not tip or tip[0] != evidence_sha:
            raise LandError(f"ci/{self.id} is {tip[0][:7] if tip else 'gone'}, not the evidence {evidence_sha[:7]}: "
                            "nothing pushed to main")
        if sha == evidence_sha:
            return "re-evidenced" if rounds > 1 else "main-unmoved"
        base_now = self.git("merge-base", sha, self.upstream())
        gained = set(self.paths(f"{evidence_base}..{base_now}"))
        moved_code = [p for p in gained if not docs_only([p], workflow_input_rules(self.root))]
        if moved_code:
            raise LandError(f"landing {sha[:7]} without evidence: main gained code {moved_code[:5]} after the "
                            f"evidence {evidence_sha[:7]}")
        ledgers = set(GENERATED) | set(SEMI_GENERATED)
        stray = [p for p in self.paths(f"{evidence_sha}..{sha}") if p not in gained and p not in ledgers]
        if stray:
            raise LandError(f"{sha[:7]} differs from the evidence {evidence_sha[:7]} outside the paths main "
                            f"gained and the ledgers: {stray[:5]}")
        theirs = self.patch_ids(f"{evidence_base}..{evidence_sha}", gained | ledgers)
        ours = self.patch_ids(f"{base_now}..{sha}", gained | ledgers)
        if theirs != ours:
            raise LandError(f"{sha[:7]} does not carry the patches of the evidence {evidence_sha[:7]} "
                            f"({len(ours)} commits vs {len(theirs)})")
        self.check_kept_lines(evidence_sha, evidence_base, base_now, sha, ledgers)
        return "docs-only-moved"

    def check_kept_lines(self, evidence_sha: str, evidence_base: str, base_now: str, sha: str,
                         ledgers: set[str]) -> None:
        """the paths main changed are left out of the patch-id comparison, so a rebase that resolved a
        conflict in one of them by dropping a line would pass it: in each of those paths, every line
        main added since the evidence base and every line the evidence added must still be in the
        landed file. Lines are counted (a line added twice is kept twice; when both sides added the
        same line, the larger count is required), whitespace-only lines are not counted, and a path
        main renamed is followed to its new name. A path main renamed into a ledger (regenerated, so
        an edit of ours in it would be lost) is refused when the evidence changed it, and a path main
        deleted, or the old name of a path main renamed, must stay gone unless the evidence changed it. Paths are read NUL-separated, files
        by blob id from a full tree listing (no pathspec, so `:x.md` is a name), and lines are split
        at \\n only (\\r before it dropped), so no diff header and no other line break is parsed.
        By design the check is on line counts, not order: a line both sides added once is kept once,
        and a move main made that the landed file reverts is not seen."""
        changed = self.paths(f"{evidence_base}..{base_now}")
        renamed_to: dict[str, str] = {}
        status = self.raw("diff", "-M", "--name-status", "-z", evidence_base, base_now).split(b"\0")
        k = 0
        while k < len(status) and status[k]:
            code = status[k].decode()
            if code.startswith(("R", "C")):
                old, new = (status[k + 1].decode("utf-8", "surrogateescape"),
                            status[k + 2].decode("utf-8", "surrogateescape"))
                if code.startswith("R"):
                    renamed_to[old] = new
                k += 3
            else:
                k += 2
        ours = set(self.paths(f"{evidence_base}..{evidence_sha}"))
        trees: dict[str, dict[str, str]] = {}

        def tree(rev: str) -> dict[str, str]:
            if rev not in trees:
                listing = self.raw("ls-tree", "-r", "-z", "--full-tree", rev).split(b"\0")
                trees[rev] = {}
                for entry in listing:
                    if entry:
                        meta, _, name = entry.partition(b"\t")
                        trees[rev][name.decode("utf-8", "surrogateescape")] = meta.split()[2].decode()
            return trees[rev]

        def lines(rev: str, path: str) -> collections.Counter:
            oid = tree(rev).get(path)
            if oid is None:
                return collections.Counter()
            text = self.raw("cat-file", "blob", oid).decode("utf-8", "surrogateescape")
            return collections.Counter(line.removesuffix("\r") for line in text.split("\n") if line.strip())

        # (old name at the evidence base, name at main now) for each path main changed
        pairs, problems = [], []
        renamed_new = set(renamed_to.values())
        for p in changed:
            if p in renamed_to:
                pairs.append((p, renamed_to[p]))
            elif p not in renamed_new:
                pairs.append((p, p))
        for old, new in pairs:
            if old not in tree(evidence_base) and new not in tree(base_now):
                raise LandError(f"main changed {new}, but it is in neither the evidence base nor main "
                                "(the path could not be read)")
        dropped, compared, paths = [], 0, 0
        for old, new in sorted(pairs, key=lambda x: x[1]):
            if old != new and old not in ours and old not in tree(base_now) and old in tree(sha):
                problems.append(f"main renamed {old} to {new}, the landed tree has {old} again")
            if new in ledgers:
                if old != new and old in ours:
                    problems.append(f"main renamed {old} into the ledger {new}, which is regenerated: "
                                    "the evidence's change to it would be lost")
                continue
            if new not in tree(base_now) and old == new:
                if old not in ours and new in tree(sha):
                    problems.append(f"main deleted {new}, the landed tree has it again")
                if old not in ours:
                    continue
            paths += 1
            base = lines(evidence_base, old)
            need_main = lines(base_now, new) - base
            need_ours = (lines(evidence_sha, old) - base) if old in ours else collections.Counter()
            landed = lines(sha, new)
            compared += sum((need_main | need_ours).values())
            lost_main = sum((need_main - landed).values())
            lost_ours = sum((need_ours - landed).values())
            if lost_main or lost_ours:
                dropped.append(f"{new} (main: {lost_main} of {sum(need_main.values())} lines, "
                               f"ours: {lost_ours} of {sum(need_ours.values())} lines)")
        if dropped:
            problems.insert(0, f"{sha[:7]} drops lines in a path main changed after the evidence "
                               f"{evidence_sha[:7]}: " + ", ".join(dropped))
        if problems:
            raise LandError("; ".join(problems))
        if not paths:
            self.log("docs-only-moved: main changed no path outside the ledgers (0 compared)")
            return
        self.log(f"docs-only-moved: {compared} lines main or the evidence added in {paths} path(s) "
                 "main changed are kept")

    def _land(self) -> str:
        self.check_remote()
        if not self.dry_run:
            self.check_gh_auth()   # the evidence and the post-push identity read go through gh
        self.fetch()
        problems = self.check_commits()
        if problems:
            raise LandError("commit checks failed:\n  " + "\n  ".join(problems))
        self.original_messages = self.our_messages()   # before any rebase
        self.lib_test_gate()
        if self.direct_reason is not None:
            lane = "direct"
            for line in ("!" * 78,
                         "WARNING: DIRECT LANDING: no ci/ push, no CI evidence before main moves",
                         f"WARNING: reason: {self.direct_reason}",
                         "!" * 78):
                self.log(line)
            self.record.append(f"lane: DIRECT (emergency, no ci/ evidence) reason: {self.direct_reason}")
        else:
            lane = "ci"
            self.record.append(f"lane: ci (ci/{self.id})")
        self.log(f"lane: {lane}")
        # the ledgers are rebuilt before the first preflight: it checks them against a
        # freshly built index, so ledgers written from an older index would fail there
        sha = self.prepare("preflight")
        base = self.git("rev-parse", self.upstream())
        evidence_sha, evidence_base, rounds = None, None, 0
        if lane == "ci":
            if self.dry_run:
                self.log(f"dry run: would push {sha[:7]} to ci/{self.id} and wait for its CI")
            else:
                self.push_evidence(sha, first=True)
                evidence_sha, evidence_base, rounds = sha, base, 1
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.fetch()
            new = self.git("rev-parse", self.upstream())
            moved = self.git("rev-list", f"{base}..{new}").split()
            if moved:
                theirs = sorted({p for c in moved for p in self.paths(f"{c}^!")})
                ours = self.paths(f"{self.upstream()}...HEAD")
                try:
                    code_moved = not docs_only(theirs, workflow_input_rules(self.root))
                except LandError as e:
                    self.log(f"the workflow inputs cannot be read ({e}): treating the move as code")
                    code_moved = True
                verify = "none" if not code_moved else ("preflight" if overlaps(ours, theirs) else "compile")
                if code_moved and lane == "ci" and not self.dry_run:
                    if rounds >= MAX_EVIDENCE:
                        raise LandError(f"{MOVING_MESSAGE} (main gained code after {rounds} evidence rounds "
                                        f"on ci/{self.id}; last evidence {evidence_sha[:7]})")
                    self.log(f"main gained code commits ({len(moved)}): rebasing and gathering the "
                             f"evidence again (round {rounds + 1} of {MAX_EVIDENCE})")
                    sha = self.prepare(verify)
                    base = new
                    self.push_evidence(sha, first=False)
                    evidence_sha, evidence_base, rounds = sha, new, rounds + 1
                    continue
                self.log("main gained " + ("documents only" if not code_moved else "code (direct lane)")
                         + f" ({len(moved)} commits): rebasing, local checks")
                sha = self.prepare(verify)
                base = new
            sha = self.git("rev-parse", "HEAD")
            # every attempt: Law's main or the candidate may have moved since the last one
            self.check_law_state(sha)
            if self.dry_run:
                self.log(f"dry run: would push {sha[:7]} to {self.branch}")
                return sha
            before_push = self.git("rev-parse", self.upstream())
            self.check_push_identity(before_push)
            self.check_messages()
            self.check_remote()
            self.check_checkout_unmoved()
            if self.git("rev-parse", "HEAD") != sha:
                raise LandError(f"land.py's working copy moved from the prepared {sha[:7]}: nothing pushed to main")
            rule = self.evidence_rule(lane, evidence_sha, evidence_base, rounds, sha)
            r = subprocess.run([*self.git_argv, "push", "-q", self.remote, f"{sha}:refs/heads/{self.branch}"],
                               cwd=self.root, capture_output=True, text=True, env=self.git_env)
            self.fetch()
            if r.returncode == 0 and self.git("rev-parse", self.upstream()) == sha:
                self.verify_landed_identity(before_push, sha)
                why = {"direct": "direct (no evidence)",
                       "re-evidenced": f"re-evidenced (evidence rounds {rounds})"}.get(rule, rule)
                self.log(f"landed {sha[:7]} on {self.branch} (attempt {attempt})")
                self.record += [f"evidence SHA: {evidence_sha or '-'}" + (f" on ci/{self.id}" if evidence_sha else ""),
                                f"landed SHA: {sha} on {self.branch} (attempt {attempt})",
                                f"reason: {why}"]
                self.log("landing record:")
                for line in self.record:
                    self.log(f"  {line}")
                if lane == "ci":
                    self.git("push", "-q", self.remote, f":refs/heads/ci/{self.id}", check=False)
                self.evidence_line = f"EVIDENCE {evidence_sha or '-'} LANDED {sha} RULE {rule}"
                self.log(self.evidence_line)
                return sha
            self.log(f"{self.branch} moved during attempt {attempt}; checking what it gained")
        raise LandError(f"could not land in {MAX_ATTEMPTS} attempts ({MOVING_MESSAGE})")

# --env-smoke: the tools the landing's children need, started the way the landing starts them
# (child_env(): clean_env(cargo=True), safe_path), each one executed; CI runs it in a minimal
# environment on every OS so a change of the sanitizer cannot leave the real landing without
# a working python3 / bash / cargo / rustc / rust-analyzer / git
SMOKE_STEPS: tuple[str, ...] = ("python3", "python-env", "bash", "cargo", "rustc", "rust-analyzer", "git")


def hostile_python_env(d: str) -> tuple[dict[str, str], str, str]:
    """An ambient environment that plants code for a child python (the --env-smoke step and the
    tests use it): PYTHONPATH with a `json` that shadows the stdlib, a garbage PYTHONHOME,
    PYTHONSTARTUP / PYTHONINSPECT, a PYTHONUSERBASE and a HOME whose user site-packages holds a
    .pth that runs code. Returns (variables, marker file, a probe script that imports json and a
    sibling module and prints what it found)."""
    marker = os.path.join(d, "planted-code-ran")
    plant = os.path.join(d, "plant")
    os.makedirs(plant, exist_ok=True)
    note = f"open({marker!r}, 'a').write(__name__ + '\\n')\n"
    for name in ("json.py", "re.py", "unittest.py", "sibling.py", "ansi.py", "sitecustomize.py", "usercustomize.py"):
        with open(os.path.join(plant, name), "w", encoding="utf-8") as fh:
            fh.write(note)
    home = os.path.join(d, "home")
    env = {"HOME": home, "USERPROFILE": home, "PYTHONUSERBASE": os.path.join(d, "userbase"),
           "PYTHONPATH": plant, "PYTHONHOME": os.path.join(d, "no-such-home"),
           "PYTHONSTARTUP": os.path.join(plant, "json.py"), "PYTHONINSPECT": "1", "PYTHONSAFEPATH": "1",
           "PYTHONWARNINGS": "error", "VIRTUAL_ENV": os.path.join(d, "venv"), "__PYVENV_LAUNCHER__": "/x"}
    probe_dir = os.path.join(d, "probe")
    os.makedirs(probe_dir, exist_ok=True)
    with open(os.path.join(probe_dir, "sibling.py"), "w", encoding="utf-8") as fh:
        fh.write("ORIGIN = 'probe'\n")
    probe = os.path.join(probe_dir, "probe.py")
    with open(probe, "w", encoding="utf-8") as fh:
        fh.write("import json, os, site, sys\nimport sibling\n"
                 "print(f'sibling={sibling.ORIGIN} user_site={site.ENABLE_USER_SITE} json from the stdlib')\n"
                 "print(json.dumps({'sibling': sibling.ORIGIN, 'json': os.path.dirname(json.__file__),"
                 " 'user_site': site.ENABLE_USER_SITE, 'path': sys.path}))\n")
    return env, marker, probe


def plant_user_site(env: dict[str, str], marker: str) -> None:
    """a .pth that writes `marker` in every user site-packages the probe python could read,
    for both HOME and PYTHONUSERBASE (plain user site, macOS framework builds)"""
    import sysconfig
    ver = f"{sys.version_info[0]}.{sys.version_info[1]}"
    dirs = set()
    for base in (env["HOME"], env["PYTHONUSERBASE"], os.path.join(env["HOME"], ".local"),
                 os.path.join(env["HOME"], "Library", "Python", ver)):
        for scheme in ("posix_user", "osx_framework_user", "nt_user"):
            try:
                dirs.add(sysconfig.get_path("purelib", scheme, vars={"userbase": base}))
            except KeyError:
                pass
    for site_dir in dirs:
        os.makedirs(site_dir, exist_ok=True)
        with open(os.path.join(site_dir, "planted.pth"), "w", encoding="utf-8") as fh:
            fh.write(f"import os; open({marker!r}, 'a').write('pth\\n')\n")


def env_smoke(lander: "Lander", steps: tuple[str, ...] | None = None) -> tuple[int, int, list[str]]:
    """(executed, failed, report lines); a step skipped (not available on this machine) is
    reported and counted apart, never as executed"""
    import shutil
    steps = SMOKE_STEPS if steps is None else steps
    path = lander.child_env()["PATH"]
    executed, failed, skipped, lines = 0, 0, 0, []

    def run(name: str, argv: list[str], check=None, shown: str | None = None):
        nonlocal executed, failed
        executed += 1
        try:
            r = subprocess.run(argv, cwd=lander.root, env=lander.child_env(), capture_output=True, text=True,
                               timeout=120)
            out = (r.stdout or r.stderr).strip().splitlines()
            first = out[0] if out else ""
            problem = None if r.returncode == 0 else f"exit {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"
            if problem is None and check:
                problem = check(r.stdout)
        except (OSError, subprocess.TimeoutExpired) as e:
            first, problem = "", str(e)
        if problem:
            failed += 1
            lines.append(f"FAIL {name}: {problem} ({' '.join(argv[:3])})")
        else:
            lines.append(f"ok   {name}: {first} ({shown or argv[0]})")

    def skip(name: str, why: str):
        nonlocal skipped
        skipped += 1
        lines.append(f"skip {name}: {why}")

    for step in steps:
        if step == "python3":
            exe = shutil.which("python3", path=path) or (shutil.which("python", path=path) if os.name == "nt" else None)
            if exe is None:
                run("python3", ["python3", "-V"])
                continue
            want = os.path.dirname(sys.executable)

            def same_python(out: str, want=want) -> str | None:
                got = out.strip().splitlines()[-1] if out.strip() else ""
                if os.path.dirname(got) == want or \
                        os.path.dirname(os.path.realpath(got)) == os.path.dirname(os.path.realpath(sys.executable)):
                    return None
                return f"python3 on the child PATH is {got!r}, not the interpreter running land.py ({sys.executable})"
            run("python3", [exe, "-c", "import sys, tomllib; assert sys.version_info >= (3, 11), sys.version; "
                                       "print(sys.version.split()[0]); print(sys.executable)"], same_python)
        elif step == "python-env":
            # a hostile ambient environment injected here, so a minimal environment around the
            # smoke (CI runs it under env -i) cannot hide a sanitizer that lets PYTHON* through
            import json
            import tempfile
            d = tempfile.mkdtemp(prefix="land-env-smoke-")
            try:
                hostile, marker, probe = hostile_python_env(d)
                plant_user_site(hostile, marker)
                env = {}

                def clean_child(out: str, marker=marker, env=env) -> str | None:
                    if os.path.exists(marker):
                        with open(marker, encoding="utf-8") as fh:
                            return f"planted code ran in the child: {fh.read().split()}"
                    try:
                        got = json.loads(out.strip().splitlines()[-1])
                    except (ValueError, IndexError):
                        return f"probe output unreadable: {out[:200]!r}"
                    if got["sibling"] != "probe" or got["user_site"]:
                        return f"child python saw {got}"
                    if "plant" in json.dumps(got["path"]):
                        return f"the planted directory is on the child's sys.path: {got['path']}"
                    leaked = sorted(k for k in env if k.startswith("PYTHON") and k not in CHILD_PYTHON_ENV)
                    return f"PYTHON variables reach the child: {leaked}" if leaked else None
                saved = {k: os.environ.get(k) for k in hostile}
                os.environ.update(hostile)   # ambient while the child is started (run() reads child_env())
                try:
                    env.update(lander.child_env())
                    run("python-env", [sys.executable, probe], clean_child)
                finally:
                    for k, v in saved.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
            finally:
                import shutil as _sh
                _sh.rmtree(d, ignore_errors=True)
        elif step == "bash":
            if os.name == "nt":
                skip("bash", "no bash child on Windows (preflight and scip_index run on POSIX)")
                continue
            exe = shutil.which("bash", path=path)
            run("bash", [exe or "bash", "--version"])
        elif step in ("cargo", "rustc"):
            # resolved on the child PATH as every cargo child is; a tool that is not there is a
            # failure that names where it was looked for, not the OS error of starting a bare name
            exe = shutil.which(step, path=path)
            if exe is None:
                failed += 1
                lines.append(f"FAIL {step}: {missing_tool_reason(step)}")
                continue
            run(step, [exe, "-V"])
        elif step == "rust-analyzer":
            # installed or not is decided by rustup, not by reading an error message: with a
            # rustup on the cargo PATH, `rustup which rust-analyzer` (in the checkout, so its
            # toolchain file applies) names the real binary when the component is installed;
            # without rustup, a rust-analyzer on the cargo PATH is a direct install
            rustup = shutil.which("rustup", path=path)
            exe = None
            if rustup:
                w = subprocess.run([rustup, "which", "rust-analyzer"], cwd=lander.root, env=lander.child_env(),
                                   capture_output=True, text=True, timeout=120)
                if w.returncode == 0 and w.stdout.strip():
                    exe = w.stdout.strip()
                else:
                    skip("rust-analyzer", "rustup: the component is not installed in the active toolchain")
                    continue
            else:
                exe = shutil.which("rust-analyzer", path=path)
                if exe is None:
                    skip("rust-analyzer", "not on the cargo PATH, and no rustup")
                    continue
            run("rust-analyzer", [exe, "--version"])
        elif step == "git":
            def new_enough(out: str) -> str | None:
                v = git_version(out)
                if v is None:
                    return f"no version in {out.strip()[:80]!r}"
                if v < GIT_MIN_VERSION:
                    return f"git {version_text(v)} is older than {version_text(GIT_MIN_VERSION)} (GIT_MIN_VERSION)"
                return None
            run("git", [*lander.git_argv, "version"], check=new_enough,
                shown=pinned_git() + (" via the tests' tripwire" if TOOL_OVERRIDE.get("git") else "")
                + f", minimum {version_text(GIT_MIN_VERSION)}")
        else:
            raise LandError(f"env-smoke: unknown step {step!r}")
    lines.append(f"env-smoke: executed {executed}, skipped {skipped}")
    if executed == 0:
        lines.append("env-smoke: nothing executed (compared nothing)")
        failed += 1
    return executed, failed, lines


def report_after(lander: "Lander", sha: str) -> int:
    code, lines = lander.wait_after(sha)
    for line in lines:
        print(line, file=sys.stderr if code else sys.stdout, flush=True)
    if code == EXIT_RED:
        print(f"error: an after-landing result on {lander.branch} is red for {sha[:7]}: REVERT OR FIX NOW",
              file=sys.stderr, flush=True)
    elif code == EXIT_PENDING:
        print(f"pending: the after-landing set on {lander.branch} did not finish for {sha[:7]}; "
              f"run again: python3 scripts/land.py --id {lander.id} --wait-only {sha}", file=sys.stderr, flush=True)
    else:
        print(f"after-landing set on {lander.branch} green for {sha[:7]}", flush=True)
    return code


def landing_exit(rc: int, lander: "Lander") -> int:
    """the exit code after a landing: the --wait result first (red / pending), then a checkout
    that could not follow"""
    if rc != 0:
        return rc
    return EXIT_CHECKOUT if lander.checkout_lagging else 0


def main(argv: list[str] | None = None) -> int:
    line_buffered_output()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", help="issue / defect id or group name (used for ci/<id>)")
    ap.add_argument("--env-smoke", action="store_true",
                    help="only start each tool the landing's children need, the way the landing starts them")
    ap.add_argument("--dry-run", action="store_true", help="run every step except the pushes")
    ap.add_argument("--no-changelog", action="store_true", help="src/ change without a CHANGELOG line")
    ap.add_argument("--replace-ci", action="store_true",
                    help="overwrite ci/<id> although it holds other commits (your own earlier attempt)")
    ap.add_argument("--direct", action="store_true",
                    help="emergency only (e.g. reverting a red main): land without the ci/ evidence; "
                         "needs --reason, prints a warning and records the reason")
    ap.add_argument("--reason", help=f"why --direct (at least {DIRECT_REASON_MIN} characters)")
    ap.add_argument("--lock-wait-minutes", type=float, default=LOCK_WAIT_MINUTES,
                    help="how long a landing waits for another landing on this machine (default 30)")
    ap.add_argument("--wait", action="store_true", help="after landing, read the after-landing set on main")
    ap.add_argument("--wait-only", metavar="SHA", help="only read the after-landing set on main for SHA")
    args = ap.parse_args(argv)
    if sys.version_info < (3, 11):
        print(f"error: land.py needs Python 3.11 or newer (running {sys.version.split()[0]})", file=sys.stderr, flush=True)
        return 2
    if args.env_smoke:
        args.id = args.id or "env-smoke"
    if args.id is None:
        print("error: --id is required", file=sys.stderr, flush=True)
        return 2
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.id):
        print("error: --id may hold letters, digits, '.', '_' and '-' only", file=sys.stderr, flush=True)
        return 2
    problem = direct_reason_problem(args.direct, args.reason)
    if problem:
        print(f"error: {problem}", file=sys.stderr, flush=True)
        return 2
    refused = refused_env()
    if refused:
        print(f"error: {', '.join(refused)} set in the environment: land.py does not land a repository "
              "chosen by the environment; unset it and run from the checkout", file=sys.stderr, flush=True)
        return 2
    root = Path(subprocess.run([*git_argv(), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               env=clean_env()).stdout.strip())
    try:
        lander = Lander(root, args.id, dry_run=args.dry_run, no_changelog=args.no_changelog,
                        replace_ci=args.replace_ci, direct_reason=args.reason if args.direct else None)
    except LandError as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 2
    lander.lock_wait_seconds = args.lock_wait_minutes * 60
    if args.env_smoke:
        executed, failed, lines = env_smoke(lander)
        for line in lines:
            print(line, flush=True)
        return 1 if failed else 0
    try:
        if args.wait_only:
            return report_after(lander, args.wait_only)
        sha = lander.land()
        if args.dry_run:
            return 0
        rc = 0
        if args.wait:
            rc = report_after(lander, sha)
            print(lander.evidence_line, flush=True)
        else:
            print(f"pushed {sha[:7]}; the after-landing set on {lander.branch} is NOT read yet: "
                  f"python3 scripts/land.py --id {args.id} --wait-only {sha}", flush=True)
        return landing_exit(rc, lander)
    except LandError as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        if "CI on" in str(e):
            print("note: a red workflow is not \"an unrelated existing red\" until the same "
                  "workflow's run on the commit before it (by headSha) is shown green; if it was green, "
                  "the red is this land's", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
