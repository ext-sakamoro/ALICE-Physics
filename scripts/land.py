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
  6. pushes HEAD to main (fast-forward only), going back to 5 when main moved
     meanwhile.
  7. before that push, requires author and committer of every commit about to reach
     main to be LAND_IDENTITY (check_push_identity); after it, reads the landed
     commits back from the remote (git, after a fetch) and the tip
     from the GitHub API, and requires author and committer to be LAND_IDENTITY
     ("REVERT OR FIX NOW" otherwise).

land.py refuses to start when the environment selects a repository (REFUSED_ENV:
GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, ...), and runs every child process with
clean_env(), which also drops the ambient identity and injected config.

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
    r"|GIT_CONFIG_PARAMETERS|GIT_CONFIG_COUNT|GIT_CONFIG_KEY_\d+|GIT_CONFIG_VALUE_\d+)$")


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


def clean_env() -> dict[str, str]:
    """os.environ without the variables SCRUBBED_ENV_RE matches"""
    return {k: v for k, v in os.environ.items() if not SCRUBBED_ENV_RE.match(k)}


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
MAX_ATTEMPTS = 6        # pushes to main (a push that loses a race to another landing retries)
MAX_EVIDENCE = 3        # evidence rounds on ci/<id>: the first and two after main gained code
MOVING_MESSAGE = "main is moving: wait for a quiet main or coordinate a landing slot"
LOCK_DIR_ENV = "LAND_LOCK_DIR"  # where the per-repository landing lock lives (tests point it elsewhere)
LOCK_WAIT_MINUTES = 30


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
        self.root, self.id, self.dry_run = root, ident, dry_run
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
        # every git process: the ambient identity / repository / config variables dropped
        self.git_env = clean_env()
        self.preflight_runs = 0
        self.poll_seconds = 30
        self.gh_cmd = ["gh"]   # tests run a fake gh script here (no network)
        self.max_polls = 240  # 2 hours at 30 s: the 4-OS CI run and the queued heavy runs
        self.after_max_polls = 360  # 3 hours at 30 s: the Mac Intel leg and Miri on main
        self.lock_wait_seconds = LOCK_WAIT_MINUTES * 60
        self.lock_poll_seconds = 5.0
        problem = direct_reason_problem(direct_reason is not None, direct_reason)
        if problem:
            raise LandError(problem)
        self.direct_reason = direct_reason.strip() if direct_reason is not None else None
        self.record: list[str] = []  # the landing record: lane, reason, the runs relied on

    # --- plumbing (overridden in tests) ------------------------------------------

    def git(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, env=self.git_env)
        if check and r.returncode != 0:
            raise LandError(f"git {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
        return r.stdout.strip()

    def sh(self, *cmd: str) -> None:
        self.log("$ " + " ".join(cmd))
        r = subprocess.run(list(cmd), cwd=self.root, env={**clean_env(), "PYTHONIOENCODING": "utf-8"})
        if r.returncode != 0:
            raise LandError(f"`{' '.join(cmd)}` failed (exit {r.returncode})")

    def lib_test_gate(self) -> None:
        """New source files must bring lib tests (scripts/lib_test_gate.py): the
        per-push coverage ratchet runs `--lib` only and would turn red after the land."""
        gate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib_test_gate.py")
        r = subprocess.run([sys.executable, gate, self.upstream(), "--root", self.root],
                           cwd=self.root, capture_output=True, text=True, env=clean_env())
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
                           cwd=self.root, capture_output=True, text=True, env=clean_env())
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
        out = subprocess.run([*self.gh_cmd, *args], capture_output=True, text=True, env=clean_env(), encoding="utf-8",
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
                           capture_output=True, text=True, env=clean_env(), encoding="utf-8", errors="replace")
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
        r = subprocess.run(["git", "cherry", "HEAD", sha], cwd=self.root, capture_output=True, text=True,
                           env=self.git_env)
        return r.returncode == 0 and all(line.startswith("-") for line in r.stdout.splitlines())

    def rebase(self) -> None:
        """Rebase on the upstream with a fixed committer; settle ledger conflicts."""
        # --force-rebase replays every commit, so each one gets the fixed committer
        # attributes are read from the empty tree: a base that still carries
        # `CHANGELOG.md merge=union` must not merge it behind merge_append()'s back
        # write the empty tree object EMPTY_TREE names (mktree reads the entries from stdin)
        subprocess.run(["git", "mktree"], cwd=self.root, input="", capture_output=True, text=True,
                       env=self.git_env)
        r = subprocess.run(["git", *self.ident_cfg, f"--attr-source={EMPTY_TREE}", "rebase", "--force-rebase",
                            self.upstream()],
                           cwd=self.root, capture_output=True, text=True, env=self.git_env)
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
                               cwd=self.root, capture_output=True, text=True, env=self.git_env)

    def merge_stages(self, path: str, *, diff3: bool = False, transform=lambda t: t) -> bool:
        """3-way merge of the conflict stages of `path` (main, base, branch) after
        `transform`; writes the result and returns whether it merged cleanly."""
        import tempfile
        stages = {}
        for n in (1, 2, 3):
            r = subprocess.run(["git", "show", f":{n}:{path}"], cwd=self.root, capture_output=True, text=True,
                               env=self.git_env)
            stages[n] = transform(r.stdout) if r.returncode == 0 else ""
        with tempfile.TemporaryDirectory() as d:
            files = []
            for n in (2, 1, 3):  # ours (main), base, theirs (branch)
                f = Path(d) / f"{n}"
                f.write_text(stages[n], encoding="utf-8")
                files.append(str(f))
            cmd = ["git", "merge-file", "-p", *(["--diff3"] if diff3 else []), *files]
            r = subprocess.run(cmd, capture_output=True, text=True, env=self.git_env)
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

    def api_commit_identity(self, sha: str) -> tuple[str, str, str, str]:
        """(author name, author email, committer name, committer email) of `sha` from the
        GitHub API (`gh api repos/<slug>/commits/<sha>`)"""
        data = self.gh_json("api", f"repos/{self.slug()}/commits/{sha}")
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
        r = subprocess.run(["git", "merge-base", "--is-ancestor", self.upstream(), "HEAD"], cwd=self.root,
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
                r = subprocess.run(["git", "merge-base", "--is-ancestor", existing[0], "HEAD"], cwd=self.root,
                                   env=self.git_env)
                if r.returncode != 0 and not self.owns_ref(existing[0]) and not self.replace_ci:
                    raise LandError(f"{ref} holds other commits; pick another --id, or pass --replace-ci "
                                    "if it is your own earlier attempt (e.g. amended after a red CI)")
        # our own ci/<id>: a re-evidence push replaces the earlier attempt (and its pending run)
        self.git("push", "-q", "--force", self.remote, f"HEAD:refs/heads/{ref}")
        self.wait_evidence(ref, sha)

    def land(self) -> str:
        with self.lock():
            return self._land()

    def _land(self) -> str:
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
        # the ledgers are rebuilt before the first preflight: it checks them against a
        # freshly built index, so ledgers written from an older index would fail there
        sha = self.prepare("preflight")
        base = self.git("rev-parse", self.upstream())
        evidence_sha, rounds, moved_any = None, 0, False
        if lane == "ci":
            if self.dry_run:
                self.log(f"dry run: would push {sha[:7]} to ci/{self.id} and wait for its CI")
            else:
                self.push_evidence(sha, first=True)
                evidence_sha, rounds = sha, 1
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.fetch()
            new = self.git("rev-parse", self.upstream())
            moved = self.git("rev-list", f"{base}..{new}").split()
            if moved:
                moved_any = True
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
                    evidence_sha, rounds = sha, rounds + 1
                    continue
                self.log("main gained " + ("documents only" if not code_moved else "code (direct lane)")
                         + f" ({len(moved)} commits): rebasing, local checks")
                sha = self.prepare(verify)
                base = new
            sha = self.git("rev-parse", "HEAD")
            if self.dry_run:
                self.log(f"dry run: would push {sha[:7]} to {self.branch}")
                return sha
            before_push = self.git("rev-parse", self.upstream())
            self.check_push_identity(before_push)
            r = subprocess.run(["git", "push", "-q", self.remote, f"HEAD:{self.branch}"], cwd=self.root,
                               capture_output=True, text=True, env=self.git_env)
            self.fetch()
            if r.returncode == 0 and self.git("rev-parse", self.upstream()) == sha:
                self.verify_landed_identity(before_push, sha)
                if lane == "direct":
                    why = "direct (no evidence)"
                elif rounds > 1:
                    why = f"re-evidenced (evidence rounds {rounds})"
                elif moved_any:
                    why = "docs-only-moved"
                else:
                    why = "main-unmoved"
                self.log(f"landed {sha[:7]} on {self.branch} (attempt {attempt})")
                self.record += [f"evidence SHA: {evidence_sha or '-'}" + (f" on ci/{self.id}" if evidence_sha else ""),
                                f"landed SHA: {sha} on {self.branch} (attempt {attempt})",
                                f"reason: {why}"]
                self.log("landing record:")
                for line in self.record:
                    self.log(f"  {line}")
                if lane == "ci":
                    self.git("push", "-q", self.remote, f":refs/heads/ci/{self.id}", check=False)
                return sha
            self.log(f"{self.branch} moved during attempt {attempt}; checking what it gained")
        raise LandError(f"could not land in {MAX_ATTEMPTS} attempts ({MOVING_MESSAGE})")

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


def main(argv: list[str] | None = None) -> int:
    line_buffered_output()
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
    ap.add_argument("--lock-wait-minutes", type=float, default=LOCK_WAIT_MINUTES,
                    help="how long a landing waits for another landing on this machine (default 30)")
    ap.add_argument("--wait", action="store_true", help="after landing, read the after-landing set on main")
    ap.add_argument("--wait-only", metavar="SHA", help="only read the after-landing set on main for SHA")
    args = ap.parse_args(argv)
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
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               env=clean_env()).stdout.strip())
    try:
        lander = Lander(root, args.id, dry_run=args.dry_run, no_changelog=args.no_changelog,
                        replace_ci=args.replace_ci, direct_reason=args.reason if args.direct else None)
    except LandError as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 2
    lander.lock_wait_seconds = args.lock_wait_minutes * 60
    try:
        if args.wait_only:
            return report_after(lander, args.wait_only)
        sha = lander.land()
        if args.dry_run:
            return 0
        if args.wait:
            return report_after(lander, sha)
        print(f"pushed {sha[:7]}; the after-landing set on {lander.branch} is NOT read yet: "
              f"python3 scripts/land.py --id {args.id} --wait-only {sha}", flush=True)
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
