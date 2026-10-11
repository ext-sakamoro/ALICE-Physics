#!/usr/bin/env python3
"""Land the local commits on main.

Run from a checkout whose HEAD sits on top of an origin/main base. The script

  1. checks the commits: author and committer identity, no internal vocabulary
     in the messages (the docs_lint word list and private-name hashes), a
     CHANGELOG line when src/ changes (--no-changelog for a pure refactor);
  2. regenerates the ledgers (they are checked against a freshly built index)
     and runs `scripts/preflight.sh --fast` once;
  3. pushes HEAD to `ci/<id>` and waits for the evidence of every workflow that
     runs on ci/** (wait_evidence: the classes of .github/ci-evidence.toml; a
     missing or dropped run is re-created with `gh workflow run`); every landing
     takes this lane. The one exception is the emergency escape `--direct
     --reason "<why>"` (e.g. reverting a red main): it skips the ci/ push, prints
     a warning and puts the reason in the landing record;
  4. rebases on the latest origin/main with a fixed committer; a conflict in a
     generated ledger is resolved by taking main's copy (it is regenerated
     next), any other conflict stops the landing;
  5. regenerates the generated ledgers on that base (SCIP index first) and
     commits them, then runs the document / ledger checks;
  6. pushes HEAD to main, starting again from 4 when main moved meanwhile
     (preflight is re-run only when the new main commits touch more than documents).

Generated ledgers are not meant to be edited on a branch: they are rebuilt at
landing time, so two landings never conflict on them.

The landing record (printed at the end) lists the lane and, per expected
workflow, the run id and conclusion the landing relied on.

After the land, `--wait` / `--wait-only SHA` read the after-landing set on main
(after_landing_items): the after-landing legs (macos-latest), the legs that run on
main only (macos-15-intel, also re-run by the nightly safety net) and the
after-landing workflows when they ran (Miri and loom). Exit codes: 0 all green,
EXIT_RED (4) a red result ("REVERT OR FIX NOW"), EXIT_PENDING (3) not finished
within the bound, 1 any other error.

Usage:
  python3 scripts/land.py --id AUD-A-S1W2-005            # land
  python3 scripts/land.py --id tgs-keys --dry-run         # everything but the pushes
  python3 scripts/land.py --id x --wait                   # also read the after-landing set on main
  python3 scripts/land.py --id revert-x --direct --reason "revert the red main of abc1234"
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
    "docs/coverage/status.md",
)
# hand-written, with a generated column (Integration) that the regeneration
# rewrites: a conflict is merged with that column blanked, never dropped
SEMI_GENERATED = ("docs/MODULES.md",)
# lists that both sides mostly add to: a conflict is merged by merge_append(),
# which keeps lines both sides added and a line one side rewrote, and stops when
# both sides changed the same lines. A union merge is not used: it keeps the old
# and the new version of a rewritten line side by side
APPEND_ONLY = ("CHANGELOG.md",)
DIRECT_REASON_MIN = 12  # characters of --reason required by the emergency --direct landing
EXIT_PENDING = 3        # --wait: the after-landing set did not finish within the bound
EXIT_RED = 4            # --wait: an after-landing result is red
SIGNATURE_RE = re.compile(r"Co-Authored-By:|^\s*Generated with\b|\U0001F916", re.I | re.M)
NIGHTLY = "nightly-2026-09-26"  # the security-audit pin
NATIVE = "std,simd,parallel,ffi,gpu-solver-bridge"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
MAX_ATTEMPTS = 5


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


def docs_only(paths: list[str]) -> bool:
    """True for a change to documents only (ledger bot commits included): it cannot
    change what the tests see, so preflight need not run again for it."""
    return bool(paths) and all(DOCS_RE.match(p) for p in paths)


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
                 replace_ci: bool = False, remote: str = "origin", branch: str = "main", log=print,
                 direct_reason: str | None = None):
        self.root, self.id, self.dry_run = root, ident, dry_run
        self.no_changelog, self.remote, self.branch, self.log = no_changelog, remote, branch, log
        self.replace_ci = replace_ci
        name, email = re.fullmatch(r"(.*) <(.*)>", IDENTITY).groups()
        self.ident_cfg = ["-c", f"user.name={name}", "-c", f"user.email={email}"]
        self.preflight_runs = 0
        self.poll_seconds = 30
        self.gh_cmd = ["gh"]   # tests run a fake gh script here (no network)
        self.max_polls = 240  # 2 hours at 30 s: the 4-OS CI run and the queued heavy runs
        self.after_max_polls = 360  # 3 hours at 30 s: the Mac Intel leg and Miri on main
        problem = direct_reason_problem(direct_reason is not None, direct_reason)
        if problem:
            raise LandError(problem)
        self.direct_reason = direct_reason.strip() if direct_reason is not None else None
        self.record: list[str] = []  # the landing record: lane, reason, the runs relied on

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

    def lib_test_gate(self) -> None:
        """New source files must bring lib tests (scripts/lib_test_gate.py): the
        per-push coverage ratchet runs `--lib` only and would turn red after the land."""
        gate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib_test_gate.py")
        r = subprocess.run([sys.executable, gate, self.upstream(), "--root", self.root],
                           cwd=self.root, capture_output=True, text=True)
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
        self.sh("python3", "scripts/integration_levels.py", "--write")
        self.sh("python3", "scripts/scip_reach.py", "--write", "docs/integration-status.md")
        self.sh("python3", "scripts/gen-oracle-status.py")
        self.sh("python3", "scripts/gen-wiring-status.py")
        self.sh("python3", "scripts/coverage_check.py", "--write-status")
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
                    ["python3", "scripts/wiring_guard.py"],
                    ["python3", "scripts/coverage_check.py"]):
            self.sh(*cmd)
        self.check_changelog()

    def check_changelog(self) -> None:
        p = self.root / "CHANGELOG.md"
        problems = changelog_problems(p.read_text(encoding="utf-8")) if p.exists() else []
        if problems:
            raise LandError("\n  ".join(problems))

    def gh_json(self, *args: str):
        """`gh <args>` parsed as JSON; LandError when gh fails (never read as an empty list)."""
        out = subprocess.run([*self.gh_cmd, *args], capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
        if out.returncode != 0:
            raise LandError(f"gh {' '.join(args[:3])} failed: {out.stderr.strip()[:300]}")
        try:
            return json.loads(out.stdout or "null")
        except ValueError as e:
            raise LandError(f"gh {' '.join(args[:3])}: not JSON: {e}") from None

    def list_branch_runs(self, slug: str, ref: str) -> list[dict]:
        runs = self.gh_json("run", "list", "--repo", slug, "--branch", ref, "--limit", "100", "--json",
                            "workflowName,headSha,databaseId,status,conclusion,createdAt,event")
        if not isinstance(runs, list):
            raise LandError("gh run list did not return a list")
        return runs

    def run_jobs(self, slug: str, run_id: int) -> list[dict]:
        data = self.gh_json("run", "view", str(run_id), "--repo", slug, "--json", "jobs")
        return [{"name": j.get("name"), "status": j.get("status"), "conclusion": j.get("conclusion")}
                for j in (data or {}).get("jobs", [])]

    def dispatch(self, slug: str, wf_file: str, ref: str) -> None:
        r = subprocess.run([*self.gh_cmd, "workflow", "run", wf_file, "--repo", slug, "--ref", ref],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise LandError(f"gh workflow run {wf_file} --ref {ref} failed: {r.stderr.strip()[:300]}")

    def slug(self) -> str:
        repo = self.git("remote", "get-url", self.remote)
        m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", repo)
        return m.group(1) if m else repo

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
        # attributes are read from the empty tree: a base that still carries
        # `CHANGELOG.md merge=union` must not merge it behind merge_append()'s back
        # write the empty tree object EMPTY_TREE names (mktree reads the entries from stdin)
        subprocess.run(["git", "mktree"], cwd=self.root, input="", capture_output=True, text=True)
        r = subprocess.run(["git", *self.ident_cfg, f"--attr-source={EMPTY_TREE}", "rebase", "--force-rebase",
                            self.upstream()],
                           cwd=self.root, capture_output=True, text=True)
        while r.returncode != 0:
            conflicted = [p for p in self.git("diff", "--name-only", "--diff-filter=U").splitlines() if p]
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
            r = subprocess.run(["git", *self.ident_cfg, f"--attr-source={EMPTY_TREE}", "-c", "core.editor=true",
                                "rebase", "--continue"],
                               cwd=self.root, capture_output=True, text=True)

    def merge_stages(self, path: str, *, diff3: bool = False, transform=lambda t: t) -> bool:
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
            cmd = ["git", "merge-file", "-p", *(["--diff3"] if diff3 else []), *files]
            r = subprocess.run(cmd, capture_output=True, text=True)
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
        self.git(*self.ident_cfg, "commit", "-q", "--author", IDENTITY, "-m",
                 "docs: regenerate the generated ledgers")

    def land(self) -> str:
        self.fetch()
        problems = self.check_commits()
        if problems:
            raise LandError("commit checks failed:\n  " + "\n  ".join(problems))
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
        # preflight checks the ledgers against a freshly built index, so a branch
        # whose ledgers were written from an older index (or not at all) would fail
        # there; regenerate them first, as is done again after every rebase
        self.regenerate()
        self.commit_regenerated()
        self.run_preflight()
        if lane == "ci":
            ref = f"ci/{self.id}"
            sha = self.git("rev-parse", "HEAD")
            self.record.append(f"evidence SHA: {sha} on {ref}")
            if self.dry_run:
                self.log(f"dry run: would push {sha[:7]} to {ref} and wait for its CI")
            else:
                existing = self.git("ls-remote", self.remote, f"refs/heads/{ref}").split()
                if existing and existing[0] == sha:
                    # an earlier run of this script pushed the same commit and stopped while
                    # waiting: wait for (or take the result of) that run instead of pushing again
                    self.log(f"{ref} already holds {sha[:7]}: reusing its CI runs")
                    self.wait_evidence(ref, sha)
                    existing = None
                if existing:
                    self.git("fetch", "-q", self.remote, existing[0], check=False)
                    r = subprocess.run(["git", "merge-base", "--is-ancestor", existing[0], "HEAD"], cwd=self.root)
                    if r.returncode != 0 and not self.owns_ref(existing[0]) and not self.replace_ci:
                        raise LandError(f"{ref} holds other commits; pick another --id, or pass --replace-ci "
                                        "if it is your own earlier attempt (e.g. amended after a red CI)")
                if existing is not None:
                    self.git("push", "-q", "--force", self.remote, f"HEAD:refs/heads/{ref}")
                    self.wait_evidence(ref, sha)
        verified_base = self.git("merge-base", "HEAD", self.upstream())
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.fetch()
            new = self.git("rev-parse", self.upstream())
            moved = self.git("rev-list", f"{verified_base}..{new}").split()
            ours = self.paths(f"{self.upstream()}...HEAD")
            theirs = sorted({p for c in moved for p in self.paths(f"{c}^!")})
            code_moved = any(not docs_only(self.paths(f"{c}^!")) for c in moved)
            self.rebase()
            self.regenerate()
            self.commit_regenerated()
            self.run_checks()
            if code_moved and overlaps(ours, theirs):
                self.log("main gained commits touching our files or the build: preflight again")
                self.run_preflight()
            elif code_moved:
                self.log("main gained commits in other files: compile check (main's CI runs their tests)")
                self.run_compile_check()
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
                self.record.append(f"landed: {sha} on {self.branch} (attempt {attempt})")
                self.log("landing record:")
                for line in self.record:
                    self.log(f"  {line}")
                if lane == "ci":
                    self.git("push", "-q", self.remote, f":refs/heads/ci/{self.id}", check=False)
                return sha
            self.log(f"{self.branch} moved during attempt {attempt}; rebasing again")
        raise LandError(f"could not land in {MAX_ATTEMPTS} attempts")


def report_after(lander: "Lander", sha: str) -> int:
    code, lines = lander.wait_after(sha)
    for line in lines:
        print(line, file=sys.stderr if code else sys.stdout)
    if code == EXIT_RED:
        print(f"error: an after-landing result on {lander.branch} is red for {sha[:7]}: REVERT OR FIX NOW",
              file=sys.stderr)
    elif code == EXIT_PENDING:
        print(f"pending: the after-landing set on {lander.branch} did not finish for {sha[:7]}; "
              f"run again: python3 scripts/land.py --id {lander.id} --wait-only {sha}", file=sys.stderr)
    else:
        print(f"after-landing set on {lander.branch} green for {sha[:7]}")
    return code


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
    ap.add_argument("--direct", action="store_true",
                    help="emergency only (e.g. reverting a red main): land without the ci/ evidence; "
                         "needs --reason, prints a warning and records the reason")
    ap.add_argument("--reason", help=f"why --direct (at least {DIRECT_REASON_MIN} characters)")
    ap.add_argument("--wait", action="store_true", help="after landing, read the after-landing set on main")
    ap.add_argument("--wait-only", metavar="SHA", help="only read the after-landing set on main for SHA")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.id):
        print("error: --id may hold letters, digits, '.', '_' and '-' only", file=sys.stderr)
        return 2
    problem = direct_reason_problem(args.direct, args.reason)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip())
    lander = Lander(root, args.id, dry_run=args.dry_run, no_changelog=args.no_changelog,
                    replace_ci=args.replace_ci, direct_reason=args.reason if args.direct else None)
    try:
        if args.wait_only:
            return report_after(lander, args.wait_only)
        sha = lander.land()
        if args.dry_run:
            return 0
        if args.wait:
            return report_after(lander, sha)
        print(f"pushed {sha[:7]}; the after-landing set on {lander.branch} is NOT read yet: "
              f"python3 scripts/land.py --id {args.id} --wait-only {sha}")
    except LandError as e:
        print(f"error: {e}", file=sys.stderr)
        if "CI on" in str(e):
            print("note: a red workflow is not \"an unrelated existing red\" until the same "
                  "workflow's run on the commit before it (by headSha) is shown green; if it was green, "
                  "the red is this land's", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
