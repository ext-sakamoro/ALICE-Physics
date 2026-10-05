#!/usr/bin/env python3
"""Land the local commits on main.

Run from a checkout whose HEAD sits on top of an origin/main base. The script

  1. checks the commits: author and committer identity, no internal vocabulary
     in the messages (the docs_lint word list and private-name hashes), a
     CHANGELOG line when src/ changes (--no-changelog for a pure refactor);
  2. runs `scripts/preflight.sh --fast` once;
  3. CI lane: when the change touches .github/, scripts/, Cargo.toml / .lock or
     bindings/ (the parts whose failures were OS / runner specific), pushes
     HEAD to `ci/<id>` and waits for that ci.yml run to succeed; other changes
     go straight on (their post-push CI on main has been green);
  4. rebases on the latest origin/main with a fixed committer; a conflict in a
     generated ledger is resolved by taking main's copy (it is regenerated
     next), any other conflict stops the landing;
  5. regenerates the generated ledgers on that base (SCIP index first) and
     commits them, then runs the document / ledger checks;
  6. pushes HEAD to main, starting again from 4 when main moved meanwhile
     (preflight is re-run only when the new main commits touch more than documents).

Generated ledgers are not meant to be edited on a branch: they are rebuilt at
landing time, so two landings never conflict on them.

Usage:
  python3 scripts/land.py --id AUD-A-S1W2-005            # land
  python3 scripts/land.py --id tgs-keys --dry-run         # everything but the pushes
  python3 scripts/land.py --id x --wait                   # also poll the main CI
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IDENTITY = os.environ.get("LAND_IDENTITY", "Moroya Sakamoto <sakamoro@alicelaw.net>")
GENERATED = (
    "docs/integration-status.md",
    "docs/integration-levels.md",
    "docs/oracle-status.md",
    "docs/wiring-status.md",
    "docs/PUBLIC_API_SNAPSHOT.txt",
)
# hand-written, with a generated column (Integration) that the regeneration
# rewrites: a conflict is merged with that column blanked, never dropped
SEMI_GENERATED = ("docs/MODULES.md",)
# append-only lists: a conflict keeps the lines of both sides (the union merge),
# done here rather than through .gitattributes so that it also holds in a clone
# or on a base without that file; changelog_problems() then catches a clash
UNION = ("CHANGELOG.md",)
# changes whose failures have been OS / runner / toolchain specific
CI_LANE_RE = re.compile(
    r"^(\.github/|scripts/|Cargo\.(toml|lock)$|bindings/|fuzz/|deny\.toml$|rust-toolchain|include/"
    r"|unreal-plugin/|\.gitattributes$|\.cargo/)")
# added code that behaves differently per OS / locale / terminal
OS_CODE_RE = re.compile(r"cfg\((?:[^)]*\b)?(windows|unix|target_os|target_arch|target_family)\b|std::env::var|std::fs::")
SIGNATURE_RE = re.compile(r"Co-Authored-By:|^\s*Generated with\b|\U0001F916", re.I | re.M)
NIGHTLY = "nightly-2026-09-26"  # the security-audit pin
NATIVE = "std,simd,parallel,ffi,gpu-solver-bridge"
NO_RUN_TIMEOUT = 1800  # seconds without any run for the pushed SHA (the GitHub queue can be slow)
MAX_ATTEMPTS = 5


class LandError(Exception):
    pass


def lane_of(paths: list[str], added: str = "") -> str:
    """`ci` when a path belongs to the CI lane or an added line is OS-specific
    code, otherwise `direct`."""
    if any(CI_LANE_RE.match(p) for p in paths) or OS_CODE_RE.search(added):
        return "ci"
    return "direct"


def changelog_problems(text: str) -> list[str]:
    """Inside [Unreleased]: no line twice and no heading twice. The union merge
    of CHANGELOG.md keeps both sides silently, so this is where it shows."""
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


def docs_only(paths: list[str]) -> bool:
    """True for a change to documents only (ledger bot commits included): it cannot
    change what the tests see, so preflight need not run again for it."""
    return bool(paths) and all(DOCS_RE.match(p) for p in paths)


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
                 replace_ci: bool = False, remote: str = "origin", branch: str = "main", log=print):
        self.root, self.id, self.dry_run = root, ident, dry_run
        self.no_changelog, self.remote, self.branch, self.log = no_changelog, remote, branch, log
        self.replace_ci = replace_ci
        name, email = re.fullmatch(r"(.*) <(.*)>", IDENTITY).groups()
        self.ident_cfg = ["-c", f"user.name={name}", "-c", f"user.email={email}"]
        self.preflight_runs = 0
        self.poll_seconds = 30

    # --- plumbing (overridden in tests) ------------------------------------------

    def git(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise LandError(f"git {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout.strip()

    def sh(self, *cmd: str) -> None:
        self.log("$ " + " ".join(cmd))
        r = subprocess.run(list(cmd), cwd=self.root, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode != 0:
            raise LandError(f"`{' '.join(cmd)}` failed (exit {r.returncode})")

    def run_preflight(self) -> None:
        self.preflight_runs += 1
        self.sh("bash", "scripts/preflight.sh", "--fast")

    def regenerate(self) -> None:
        self.sh("bash", "scripts/scip_index.sh")
        self.sh("python3", "scripts/integration_levels.py", "--write")
        self.sh("python3", "scripts/scip_reach.py", "--write", "docs/integration-status.md")
        self.sh("python3", "scripts/gen-oracle-status.py")
        self.sh("python3", "scripts/gen-wiring-status.py")
        self.regenerate_public_api()

    def regenerate_public_api(self) -> None:
        r = subprocess.run(["cargo", f"+{NIGHTLY}", "public-api", "--features", NATIVE, "--simplified"],
                           cwd=self.root, capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            raise LandError(f"cargo +{NIGHTLY} public-api failed (install the toolchain and cargo-public-api): "
                            f"{r.stderr.strip()[-300:]}")
        (self.root / "docs/PUBLIC_API_SNAPSHOT.txt").write_text(r.stdout, encoding="utf-8")

    def run_checks(self) -> None:
        for cmd in (["python3", "scripts/integration_levels.py", "--check"],
                    ["python3", "scripts/readme_sync.py", "--check"],
                    ["python3", "scripts/docs_lint.py", "--check"],
                    ["python3", "scripts/scip_reach.py", "--check-baseline"],
                    ["python3", "scripts/gen-oracle-status.py", "--check"],
                    ["python3", "scripts/wiring_guard.py"]):
            self.sh(*cmd)
        self.check_changelog()

    def check_changelog(self) -> None:
        p = self.root / "CHANGELOG.md"
        problems = changelog_problems(p.read_text(encoding="utf-8")) if p.exists() else []
        if problems:
            raise LandError("\n  ".join(problems))

    def list_runs(self, slug: str, ref: str) -> list[dict]:
        out = subprocess.run(
            ["gh", "run", "list", "--repo", slug, "--workflow", "ci.yml", "--branch", ref, "--limit", "20",
             "--json", "headSha,databaseId,status,conclusion"], capture_output=True, text=True)
        return json.loads(out.stdout or "[]") if out.returncode == 0 else []

    def wait_ci(self, ref: str, sha: str) -> None:
        """Wait for the ci.yml run of `sha` on `ref`; raise unless it succeeds."""
        repo = self.git("remote", "get-url", self.remote)
        m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", repo)
        slug = m.group(1) if m else repo
        started = time.monotonic()
        for _ in range(180):
            runs = [r for r in self.list_runs(slug, ref) if r["headSha"] == sha]
            if not runs and time.monotonic() - started > NO_RUN_TIMEOUT:
                raise LandError(f"no ci.yml run for {sha[:7]} on {ref} after {NO_RUN_TIMEOUT // 60} minutes "
                                "(push failed, or the workflow did not trigger): not treated as success")
            if runs and runs[0]["status"] == "completed":
                c = runs[0]["conclusion"]
                if c != "success":
                    why = {"cancelled": "cancelled (a later push or a manual cancel; unverified, not green)",
                           "skipped": "skipped (nothing ran; unverified, not green)"}.get(c, c)
                    raise LandError(f"CI on {ref} ended {why} (run {runs[0]['databaseId']})")
                self.log(f"CI on {ref} succeeded (run {runs[0]['databaseId']})")
                return
            time.sleep(self.poll_seconds)
        raise LandError(f"CI on {ref} did not finish in 90 minutes")

    # --- steps -------------------------------------------------------------------

    def upstream(self) -> str:
        return f"{self.remote}/{self.branch}"

    def fetch(self) -> None:
        self.git("fetch", "-q", self.remote, self.branch)

    def commits(self) -> list[str]:
        return self.git("rev-list", "--reverse", f"{self.upstream()}..HEAD").split()

    def paths(self, rev_range: str) -> list[str]:
        out = self.git("diff", "--name-only", rev_range)
        return [p for p in out.splitlines() if p]

    def check_commits(self) -> list[str]:
        if self.git("status", "--porcelain"):
            raise LandError("the checkout has uncommitted changes")
        shas = self.commits()
        if not shas:
            raise LandError(f"nothing to land: HEAD has no commit beyond {self.upstream()} (compared nothing)")
        problems = []
        for sha in shas:
            who = self.git("log", "-1", "--format=%an <%ae>|%cn <%ce>", sha).split("|")
            if who[0] != IDENTITY:
                problems.append(f"{sha[:7]}: author `{who[0]}` is not `{IDENTITY}`")
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
        r = subprocess.run(["git", "cherry", "HEAD", sha], cwd=self.root, capture_output=True, text=True)
        return r.returncode == 0 and all(line.startswith("-") for line in r.stdout.splitlines())

    def rebase(self) -> None:
        """Rebase on the upstream with a fixed committer; settle ledger conflicts."""
        # --force-rebase replays every commit, so each one gets the fixed committer
        r = subprocess.run(["git", *self.ident_cfg, "rebase", "--force-rebase", self.upstream()],
                           cwd=self.root, capture_output=True, text=True)
        while r.returncode != 0:
            conflicted = [p for p in self.git("diff", "--name-only", "--diff-filter=U").splitlines() if p]
            if not conflicted:
                self.git("rebase", "--abort", check=False)
                raise LandError(f"rebase failed: {r.stderr.strip()}")
            other = [p for p in conflicted if p not in GENERATED and p not in SEMI_GENERATED and p not in UNION]
            if other:
                self.git("rebase", "--abort", check=False)
                raise LandError(f"rebase conflict outside the generated ledgers: {other}")
            for p in conflicted:
                if p in SEMI_GENERATED:
                    self.merge_semi(p)
                elif p in UNION:
                    self.merge_stages(p, union=True)
                else:
                    self.git("checkout", "--ours", "--", p)   # main's copy; regenerated after the rebase
                self.git("add", "--", p)
            r = subprocess.run(["git", *self.ident_cfg, "-c", "core.editor=true", "rebase", "--continue"],
                               cwd=self.root, capture_output=True, text=True)

    def merge_stages(self, path: str, *, union: bool = False, transform=lambda t: t) -> bool:
        """3-way merge of the conflict stages of `path` (main, base, branch) after
        `transform`; writes the result and returns whether it merged cleanly."""
        import tempfile
        stages = {}
        for n in (1, 2, 3):
            r = subprocess.run(["git", "show", f":{n}:{path}"], cwd=self.root, capture_output=True, text=True)
            stages[n] = transform(r.stdout) if r.returncode == 0 else ""
        with tempfile.TemporaryDirectory() as d:
            files = []
            for n in (2, 1, 3):  # ours (main), base, theirs (branch)
                f = Path(d) / f"{n}"
                f.write_text(stages[n], encoding="utf-8")
                files.append(str(f))
            cmd = ["git", "merge-file", "-p", *(["--union"] if union else []), *files]
            r = subprocess.run(cmd, capture_output=True, text=True)
        (self.root / path).write_text(r.stdout, encoding="utf-8")
        return r.returncode == 0

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
        self.git(*self.ident_cfg, "commit", "-q", "--author", IDENTITY, "-m",
                 "docs: regenerate the generated ledgers")

    def land(self) -> str:
        self.fetch()
        problems = self.check_commits()
        if problems:
            raise LandError("commit checks failed:\n  " + "\n  ".join(problems))
        added = "\n".join(l[1:] for l in self.git("diff", "-U0", f"{self.upstream()}...HEAD").splitlines()
                          if l.startswith("+") and not l.startswith("+++"))
        lane = lane_of(self.paths(f"{self.upstream()}...HEAD"), added)
        self.log(f"lane: {lane}")
        self.run_preflight()
        if lane == "ci":
            ref = f"ci/{self.id}"
            sha = self.git("rev-parse", "HEAD")
            if self.dry_run:
                self.log(f"dry run: would push {sha[:7]} to {ref} and wait for its CI")
            else:
                existing = self.git("ls-remote", self.remote, f"refs/heads/{ref}").split()
                if existing:
                    self.git("fetch", "-q", self.remote, existing[0], check=False)
                    r = subprocess.run(["git", "merge-base", "--is-ancestor", existing[0], "HEAD"], cwd=self.root)
                    if r.returncode != 0 and not self.owns_ref(existing[0]) and not self.replace_ci:
                        raise LandError(f"{ref} holds other commits; pick another --id, or pass --replace-ci "
                                        "if it is your own earlier attempt (e.g. amended after a red CI)")
                self.git("push", "-q", "--force", self.remote, f"HEAD:refs/heads/{ref}")
                self.wait_ci(ref, sha)
        verified_base = self.git("merge-base", "HEAD", self.upstream())
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.fetch()
            new = self.git("rev-parse", self.upstream())
            moved = self.git("rev-list", f"{verified_base}..{new}").split()
            needs_preflight = any(not docs_only(self.paths(f"{c}^!")) for c in moved)
            self.rebase()
            self.regenerate()
            self.commit_regenerated()
            self.run_checks()
            if needs_preflight:
                self.log("main gained commits beyond documents: preflight again")
                self.run_preflight()
            verified_base = new
            sha = self.git("rev-parse", "HEAD")
            if self.dry_run:
                self.log(f"dry run: would push {sha[:7]} to {self.branch}")
                return sha
            r = subprocess.run(["git", "push", "-q", self.remote, f"HEAD:{self.branch}"], cwd=self.root,
                               capture_output=True, text=True)
            self.fetch()
            if r.returncode == 0 and self.git("rev-parse", self.upstream()) == sha:
                self.log(f"landed {sha[:7]} on {self.branch} (attempt {attempt})")
                if lane == "ci":
                    self.git("push", "-q", self.remote, f":refs/heads/ci/{self.id}", check=False)
                return sha
            self.log(f"{self.branch} moved during attempt {attempt}; rebasing again")
        raise LandError(f"could not land in {MAX_ATTEMPTS} attempts")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", required=True, help="issue / defect id or group name (used for ci/<id>)")
    ap.add_argument("--dry-run", action="store_true", help="run every step except the pushes")
    ap.add_argument("--no-changelog", action="store_true", help="src/ change without a CHANGELOG line")
    ap.add_argument("--replace-ci", action="store_true",
                    help="overwrite ci/<id> although it holds other commits (your own earlier attempt)")
    ap.add_argument("--wait", action="store_true", help="after landing, wait for the main CI run")
    ap.add_argument("--wait-only", metavar="SHA", help="only wait for the main CI run of SHA")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.id):
        print("error: --id may hold letters, digits, '.', '_' and '-' only", file=sys.stderr)
        return 2
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip())
    lander = Lander(root, args.id, dry_run=args.dry_run, no_changelog=args.no_changelog,
                    replace_ci=args.replace_ci)
    if args.wait_only:
        try:
            lander.wait_ci(lander.branch, args.wait_only)
        except LandError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"CI on {lander.branch} green for {args.wait_only[:7]}")
        return 0
    try:
        sha = lander.land()
        if args.dry_run:
            pass
        elif args.wait:
            lander.wait_ci(lander.branch, sha)
            print(f"CI on {lander.branch} green for {sha[:7]}")
        else:
            print(f"pushed {sha[:7]}; CI on {lander.branch} NOT confirmed yet: "
                  f"python3 scripts/land.py --id {args.id} --wait-only {sha}")
    except LandError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
