#!/usr/bin/env python3
"""The plan of the in-diff mutation runs of .github/workflows/quality-deep.yml: how many shards a night's diff
takes, which commit the diff starts from, and which ranges were set aside for the weekly run.

How many mutants one job gets through in its time limit depends on that night's tests (11-17 to 11-21: 115-180
outcomes in 270 min at -j 2; 11-22, with a 155 s build and a 264 s test run unmutated: 73 of 96, the job was
stopped). So the plan job runs the unmutated baseline of each axis first and a shard is given what fits in the
limit at the slowest measured mutant (`per_from_baseline`); a night's plan is cut into such shards, at most CAP
per axis. A night whose plan is larger than CAP shards is not run: it is recorded as `over-cap` with its range
base..head and its counts, and the weekly run tests that range (without the cap). Every plan writes a record
(`mutants-record.json`, uploaded as an artifact of the run); nothing is committed.

The base of a night's diff is the head of the latest earlier nightly run that is either

  * measured: its record says `accepted` and every shard job it planned ended its `Measured` step with
    success (every planned mutant of the shard has an outcome; a missed mutant does not change this), or
    its record says `empty` (the diff held no mutant), or
  * over-cap: its range is carried to the weekly run.

A run that was cancelled, whose unmutated tree failed, or that has a killed or unmeasured shard does not
move the base: its lines are in the next night's diff again. With no such run in the window the night has no
base: the plan fails with that reason and nothing is tested (there is no base by the clock: until 2026-11-25
the base was then the commit of 25 hours ago, and the lines between the last measured head and that commit
were tested by no run). The way on is a dispatch with `diff_base` and `set_aside`: it tests nothing, records
base..head as an open range for the weekly run (kind `aside`), and its head is the base of the next night.

A night's record is trusted as a base only when its own base came from a record (`base_from`): one planned
on a base by the clock is passed over, and so is one whose base was the head of such a record (the run its
`base_reason` names). The reason printed with the base says how many were passed over.

An over-cap range stays open until a weekly run whose record lists it has every shard job of that range
measured. Open ranges are carried in every nightly and weekly record (`open`), so one does not leave with the
window; the carried list is read from the latest such record (a pull request's or a plain diff_base dispatch's
record carries none and is passed over). A range
that has been open for more than MAX_AGE_DAYS fails the weekly run's age job. A record that is listed but
expired or unreadable is an error (exit 2), never "no open range".

    mutants_nightly.py fetch --repo OWNER/REPO --self RUN_ID --out runs.json
    mutants_nightly.py base  --runs runs.json                         # prints the base sha; exit 3 = no base
    mutants_nightly.py plan  --mode nightly|pr|aside --base-from record|pull-request|dispatch \\
        --runs runs.json --base B --head H --count default=N --count features=M --per default=P --per features=Q \\
        --out-record R --out-matrix M                                 # exit 5 = over the cap
    mutants_nightly.py per   --out mutants.out                        # the shard size from a baseline
    mutants_nightly.py weekly --runs runs.json --head H --now ISO --count default=N --count features=M \\
        --per default=P --per features=Q --out-record R --out-matrix M   # exit 5 = the core plan is over the cap
    mutants_nightly.py core-jobs --record R                           # the ratchet's count of output directories
    mutants_nightly.py age   --record R --now ISO                     # exit 1 = a range open for too long

Author: Moroya Sakamoto
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

STOP_S = 270 * 60    # the `timeout ... 270m` around cargo mutants in the shard step
JOBS = 2             # its `-j 2`
# a mutant against the unmutated baseline of its own run, the slowest measured: build 232 s on 155 s (run
# 44989179739, 11-22), test 2.6x (nightlies 11-10 to 11-20, the spread in the shard step's comment)
BUILD_SLOW = 1.5
TEST_SLOW = 2.6
# for a range recorded without its measure (a record written before 2026-11-23): what the 11-22 baseline
# (155 s build + 264 s test) gives
PER_UNMEASURED = 34
CAP = 8              # shards per feature axis in one night
MAX_AGE_DAYS = 8     # an over-cap range older than this fails the weekly age job
WINDOW_DAYS = 21     # how far back the runs are read (the record artifacts are kept 30 days)
AXES = {"default": "", "features": "parallel,gpu-solver-bridge,simd"}
# the weekly core shards are sized like a night's (`per_from_baseline`, from the weekly plan job's baseline of
# the core files). Until 2026-11-23 they were 16 an axis whatever the baseline: with 25 s build + 4 s test
# (11-08, 260 mutants a shard) every shard ended in 46-170 min; with 85 s + 13 s (11-15 and 11-22, 304 a shard,
# after the mutants moved to the opt-level 2 profile) one shard of 32 ended, in 243 min, and the others were
# stopped at the job limit. The cap is what WEEKLY_PARALLEL jobs at a time end in WEEKLY_BUDGET_H, the time
# from the weekly start to the next nightly, when every job runs to JOB_MIN; a larger core plan is not run and
# the weekly plan fails with the counts. The cap counts the core jobs only: the ranges set aside and the other
# weekly jobs (Miri, loom, bench, the ignored tests, coverage) use the same runners, so a week at the cap that
# also takes ranges can run past the next nightly, which then waits in the group (it is not replaced or lost).
# WEEKLY_PARALLEL 6 and the cap it gives are for the baseline of 2026-11 (85 s build + 13 s test, the opt-level 2
# profile the two slow self-contact sweeps of the lib suite made necessary): when the mutants are back on the
# cheaper profile, derive both again from the new baseline (4 jobs at a time is the number to return to).
WEEKLY_BUDGET_H = 48
WEEKLY_PARALLEL = 6  # `max-parallel` of the shard matrix in a weekly run
NIGHTLY_PARALLEL = 4  # and in any other run
JOB_MIN = 280        # a shard job: the 270 min stop and the setup around it
CORE_CAP = (WEEKLY_BUDGET_H * 60 // JOB_MIN) * WEEKLY_PARALLEL // len(AXES)
RECORD_ARTIFACT = "mutants-record"
RECORD_FILE = "mutants-record.json"
MEASURED_STEP = "Measured (every planned mutant of this shard has an outcome)"
CARRYING = ("nightly", "weekly", "aside")   # the record kinds whose `open` list is the carried list
# what the plan step is told about where the base came from, and the one each mode takes
BASE_FROM = {"nightly": ("record",), "pr": ("pull-request", "dispatch"), "aside": ("dispatch",)}
BY_THE_CLOCK = "falling back to the commit of"   # in the base_reason of a record planned on a base by the clock
EVENTS = ("schedule", "workflow_dispatch")   # the runs that are read back (a pull request's run is never a base)
MAX_PAGES = 10       # of 100 runs per event: more than this in the window is an error, not a cut
OVER_CAP, NO_BASE, ERROR = 5, 3, 2


class RecordError(Exception):
    """a record that exists but cannot be read"""


def per_from_baseline(build_s: float, test_s: float) -> int:
    """the mutants one shard job is given: what fits between the baseline and the stop when every mutant
    builds and tests as slowly as the slowest measured one (a missed mutant runs the whole test set)"""
    if build_s <= 0 or test_s <= 0:
        raise ValueError(f"baseline build {build_s} s / test {test_s} s: not a measurement")
    each = (BUILD_SLOW * build_s + TEST_SLOW * test_s) / JOBS
    left = STOP_S - build_s - test_s
    if left < each:
        raise ValueError(f"the unmutated baseline takes {build_s:.0f} s build + {test_s:.0f} s test: not one mutant "
                         f"({each:.0f} s) fits in the {STOP_S} s of a shard job")
    return int(left // each)


def read_baseline(out: Path) -> tuple[float, float]:
    """(build seconds, test seconds) of the unmutated baseline of a cargo-mutants output directory"""
    root = out / "mutants.out" if (out / "mutants.out").is_dir() else out
    result = json.loads((root / "outcomes.json").read_text(encoding="utf-8"))
    base = [o for o in result.get("outcomes", []) if o.get("scenario") == "Baseline"]
    if len(base) != 1 or base[0].get("summary") != "Success":
        raise ValueError(f"{root}: the unmutated baseline did not succeed "
                         f"({[o.get('summary') for o in base] or 'no baseline outcome'})")
    took = {p.get("phase"): float(p.get("duration", 0)) for p in base[0].get("phase_results", [])
            if p.get("process_status") == "Success"}
    if set(took) != {"Build", "Test"}:
        raise ValueError(f"{root}: the baseline has phases {sorted(took)}, not Build and Test")
    return took["Build"], took["Test"]


def per_of(o: dict) -> dict[str, int]:
    """the shard size per axis a recorded range was planned with"""
    per = o.get("per") or {}
    return {a: int(per.get(a, PER_UNMEASURED)) for a in AXES}


def shards_for(count: int, per: int) -> int:
    """the number of shards a plan of `count` mutants takes (0 for an empty plan)"""
    if count < 0 or per <= 0:
        raise ValueError(f"count {count} / per {per}")
    return -(-count // per)


def range_id(base: str, head: str) -> str:
    return f"{base[:12]}..{head[:12]}"


def diff_job_label(base: str, head: str, shard: int, of: int, axis: str) -> str:
    return f"in-diff {range_id(base, head)} shard {shard}/{of}, {axis}"


def measured(run: dict, base: str, head: str, jobs: int) -> bool:
    """whether every one of the `jobs` shard jobs of base..head in this run ended its Measured step with success"""
    tag = f"in-diff {range_id(base, head)} "
    mine = [j for j in run.get("jobs", []) if tag in j.get("name", "")]
    if jobs <= 0 or len(mine) != jobs:
        return False
    return all(any(s.get("name") == MEASURED_STEP and s.get("conclusion") == "success" for s in j.get("steps", []))
               for j in mine)


def check_records(runs: list[dict]) -> None:
    bad = [f"run {r.get('id')}: {r['record_error']}" for r in runs if r.get("record_error")]
    if bad:
        raise RecordError("; ".join(bad))


def by_time(runs: list[dict]) -> list[dict]:
    return sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)


def source_run(rec: dict) -> int | None:
    """the run whose record gave this record its base (the `run N ...` its base_reason starts with)"""
    m = re.match(r"run (\d+) ", rec.get("base_reason") or "")
    return int(m.group(1)) if m else None


def trusted(run: dict, by_id: dict[int, dict]) -> bool:
    """whether a night's record may be a base: its own base came from a record, and that record (when it is
    still in the window) may be a base too. A record from before `base_from` is read by its base_reason."""
    seen = set()
    while True:
        rec = run.get("record") or {}
        if rec.get("kind") == "aside":
            return True
        if rec.get("kind") != "nightly" or run.get("id") in seen:
            return False
        seen.add(run.get("id"))
        if BY_THE_CLOCK in (rec.get("base_reason") or "") or rec.get("base_from", "record") != "record":
            return False
        src = by_id.get(source_run(rec))
        if src is None:   # (the base came from a run that has left the window)
            return True
        run = src


def pick_base(runs: list[dict]) -> tuple[str | None, str]:
    """(the head of the latest nightly run that measured or was set aside, why); None = no such run"""
    check_records(runs)
    skipped = passed = 0
    by_id = {r.get("id"): r for r in runs}

    def said(why: str) -> str:
        return why + (f" ({passed} later night(s) passed over: planned on a base no record gave)" if passed else "")

    for r in by_time(runs):
        rec = r.get("record")
        if not rec or rec.get("kind") not in ("nightly", "aside"):
            continue
        if rec["kind"] == "aside":
            return rec["head"], said(f"run {r['id']} set its range aside for the weekly run (a dispatch)")
        if not trusted(r, by_id):
            passed += 1
            continue
        state = rec.get("state")
        if state == "over-cap":
            return rec["head"], said(f"run {r['id']} set its range aside for the weekly run (over the cap)")
        if state == "empty":
            return rec["head"], said(f"run {r['id']} had no mutant in its diff")
        if state == "accepted" and measured(r, rec["base"], rec["head"], rec.get("jobs", 0)):
            return rec["head"], said(f"run {r['id']} measured every shard")
        skipped += 1
    return None, (f"no nightly run in the window measured ({skipped} unmeasured, {passed} planned on a base no "
                  f"record gave, {len(runs)} run(s) read)")


def open_ranges(runs: list[dict]) -> list[dict]:
    """the over-cap ranges no weekly run has measured: those of the records in the window and those the latest
    record carries, oldest first"""
    check_records(runs)
    found: dict[str, dict] = {}
    ordered = by_time(runs)
    for r in ordered:   # (the carried list of the latest nightly or weekly record: a pr record carries none)
        rec = r.get("record")
        if rec and rec.get("kind") in CARRYING:
            for o in rec.get("open", []):
                found.setdefault(range_id(o["base"], o["head"]), dict(o))
            break
    for r in ordered:
        rec = r.get("record")
        if rec and (rec.get("kind"), rec.get("state")) in (("nightly", "over-cap"), ("aside", "set-aside")):
            found.setdefault(range_id(rec["base"], rec["head"]),
                             {"base": rec["base"], "head": rec["head"], "counts": rec["counts"],
                              **({"per": rec["per"]} if rec.get("per") else {}), "since": r["created_at"]})
    for r in ordered:
        rec = r.get("record")
        if not rec or rec.get("kind") != "weekly":
            continue
        for t in rec.get("taken", []):
            rid = range_id(t["base"], t["head"])
            if rid in found and measured(r, t["base"], t["head"], t.get("jobs", 0)):
                del found[rid]
    return sorted(found.values(), key=lambda o: o["since"])


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def stale(ranges: list[dict], now: str, max_age_days: int = MAX_AGE_DAYS) -> list[dict]:
    limit = parse_time(now) - timedelta(days=max_age_days)
    return [o for o in ranges if parse_time(o["since"]) < limit]


def diff_entries(base: str, head: str, counts: dict[str, int], per: dict[str, int]) -> list[dict]:
    out = []
    for axis, features in AXES.items():
        n = shards_for(counts.get(axis, 0), per[axis])
        out += [{"kind": "diff", "shard": k, "of": n, "features": features, "base": base, "head": head,
                 "label": diff_job_label(base, head, k, n, axis),
                 "artifact": f"mutants-diff-{base[:12]}-{head[:12]}-{k}-{axis}"} for k in range(n)]
    return out


def nightly_record(mode: str, base: str, head: str, counts: dict[str, int], carried: list[dict], why: str,
                   per: dict[str, int], cap: int = CAP, base_from: str | None = None) -> tuple[dict, list[dict]]:
    """(the record of a nightly / pr / aside plan, its matrix entries); `per` is the shard size of each axis
    that has a mutant, from that axis' baseline of tonight. An aside plan has no entry whatever its size: its
    range is for the weekly run."""
    base_from = base_from or BASE_FROM[mode][0]
    if base_from not in BASE_FROM[mode]:
        raise ValueError(f"a {mode} plan takes its base from {' or '.join(BASE_FROM[mode])}, not from {base_from}")
    unknown = sorted(set(counts) - set(AXES))
    if unknown or set(counts) != set(AXES):
        raise ValueError(f"counts must name exactly {sorted(AXES)}, got {sorted(counts)}")
    unmeasured = sorted(a for a in AXES if counts[a] > 0 and a not in per)
    if unmeasured or set(per) - set(AXES):
        raise ValueError(f"an axis with mutants needs its shard size: missing {unmeasured}, got {sorted(per)}")
    per = {a: int(per.get(a, PER_UNMEASURED)) for a in AXES}
    shards = {a: shards_for(counts[a], per[a]) for a in AXES}
    if sum(counts.values()) == 0:
        state, entries = "empty", []
    elif mode == "aside":
        state, entries = "set-aside", []
    elif max(shards.values()) > cap:
        state, entries = "over-cap", []
    else:
        state, entries = "accepted", diff_entries(base, head, counts, per)
    rec = {"v": 1, "kind": mode, "state": state, "base": base, "head": head, "base_from": base_from,
           "base_reason": why, "counts": counts,
           "per": per, "shards": shards, "jobs": len(entries), "open": carried}
    return rec, entries


def core_entries(head: str, shards: dict[str, int]) -> list[dict]:
    return [{"kind": "core", "shard": k, "of": shards[a], "features": f, "base": "", "head": head,
             "label": f"core shard {k}/{shards[a]}, {a}",
             "artifact": f"mutants-out-{k}-{'default' if a == 'default' else 'parallel'}"}
            for a, f in AXES.items() for k in range(shards[a])]


def weekly_record(head: str, ranges: list[dict], now: str, counts: dict[str, int], per: dict[str, int],
                  cap: int = CORE_CAP) -> tuple[dict, list[dict]]:
    """(the record of a weekly plan, its matrix entries): the core shards, sized by `per` (each axis' baseline
    of this run) for the `counts` core mutants of each axis, and every open range. A core plan over `cap`
    shards on an axis has no core entry and the state `over-cap`; the open ranges are still taken."""
    if set(counts) != set(AXES) or set(per) != set(AXES):
        raise ValueError(f"the core counts and shard sizes must name exactly {sorted(AXES)}, "
                         f"got {sorted(counts)} / {sorted(per)}")
    if min(counts.values()) <= 0:
        raise ValueError(f"an axis has no core mutant to test (counts {counts}): nothing would be compared")
    shards = {a: shards_for(counts[a], per[a]) for a in AXES}
    over = max(shards.values()) > cap
    entries = [] if over else core_entries(head, shards)
    core_jobs = len(entries)
    taken = []
    for o in ranges:
        mine = diff_entries(o["base"], o["head"], o["counts"], per_of(o))
        if not mine:
            raise ValueError(f"open range {range_id(o['base'], o['head'])} has no mutant to test (counts {o['counts']})")
        entries += mine
        taken.append({"base": o["base"], "head": o["head"], "jobs": len(mine)})
    rec = {"v": 1, "kind": "weekly", "state": "over-cap" if over else "accepted", "head": head,
           "core": {"counts": counts, "per": per, "shards": shards, "jobs": core_jobs}, "taken": taken,
           "open": ranges, "stale": [range_id(o["base"], o["head"]) for o in stale(ranges, now)]}
    return rec, entries


# --- reading the runs (the only part that talks to GitHub) -----------------------------------------------------

def gh_json(args: list[str]):
    out = subprocess.run(["gh", "api", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gh api {' '.join(args)}: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout)


def read_record(repo: str, run_id: int) -> tuple[dict | None, str | None]:
    """(record, error) of one run: (None, None) when the run uploaded no record"""
    arts = gh_json([f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"]).get("artifacts", [])
    mine = [a for a in arts if a.get("name") == RECORD_ARTIFACT]
    if not mine:
        return None, None
    if mine[0].get("expired"):
        return None, "its record artifact has expired"
    with tempfile.TemporaryDirectory() as t:
        z = Path(t) / "r.zip"
        with open(z, "wb") as f:
            got = subprocess.run(["gh", "api", f"repos/{repo}/actions/artifacts/{mine[0]['id']}/zip"], stdout=f,
                                 stderr=subprocess.PIPE)
        if got.returncode != 0:
            return None, f"its record artifact could not be downloaded ({got.stderr.decode()[:200].strip()})"
        try:
            with zipfile.ZipFile(z) as zf:
                rec = json.loads(zf.read(RECORD_FILE).decode("utf-8"))
        except (zipfile.BadZipFile, KeyError, ValueError) as e:
            return None, f"its record artifact is not readable ({e})"
    if not isinstance(rec, dict) or rec.get("v") != 1 or rec.get("kind") not in ("nightly", "weekly", "pr", "aside"):
        return None, "its record is not a version 1 record"
    return rec, None


def list_runs(repo: str, since: str, workflow: str, api=None) -> list[dict]:
    """every schedule / dispatch run of the workflow created after `since`: each event is listed on its own (the
    pull request runs do not take places on a page) and every page is followed; more than MAX_PAGES pages is an
    error (the window would not be covered)"""
    api = api or gh_json
    out = []
    for event in EVENTS:
        for page in range(1, MAX_PAGES + 2):
            if page > MAX_PAGES:
                raise RuntimeError(f"more than {MAX_PAGES * 100} {event} runs of {workflow} since {since}: the "
                                   "window is not covered")
            got = api([f"repos/{repo}/actions/workflows/{workflow}/runs?per_page=100&page={page}&event={event}"
                       f"&created=%3E{since}"]).get("workflow_runs", [])
            out += got
            if len(got) < 100:
                break
    return out


def fetch(repo: str, self_id: int, now: datetime, workflow: str = "quality-deep.yml") -> list[dict]:
    since = (now - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = []
    for r in list_runs(repo, since, workflow):
        if r["id"] == self_id:
            continue
        jobs, page = [], 1
        while True:
            got = gh_json([f"repos/{repo}/actions/runs/{r['id']}/jobs?per_page=100&page={page}"]).get("jobs", [])
            jobs += [{"name": j["name"], "conclusion": j.get("conclusion"),
                      "steps": [{"name": s["name"], "conclusion": s.get("conclusion")} for s in j.get("steps", [])]}
                     for j in got]
            if len(got) < 100:
                break
            page += 1
        rec, err = read_record(repo, r["id"])
        runs.append({"id": r["id"], "head_sha": r["head_sha"], "created_at": r["created_at"],
                     "conclusion": r.get("conclusion"), "jobs": jobs, "record": rec, "record_error": err})
    return runs


# --- command line ----------------------------------------------------------------------------------------------

def write_out(args, rec: dict, entries: list[dict]) -> None:
    Path(args.out_record).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    Path(args.out_matrix).write_text(json.dumps({"include": entries}) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--repo", required=True)
    f.add_argument("--self", dest="self_id", type=int, required=True)
    f.add_argument("--out", required=True)
    b = sub.add_parser("base")
    b.add_argument("--runs", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--mode", choices=tuple(BASE_FROM), required=True)
    p.add_argument("--base-from", choices=sorted({f for fs in BASE_FROM.values() for f in fs}), required=True,
                   help="where the base came from: a nightly plan takes one a record gave and no other")
    p.add_argument("--base", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--base-reason", default="")
    p.add_argument("--now", required=True)
    w = sub.add_parser("weekly")
    w.add_argument("--head", required=True)
    w.add_argument("--now", required=True)
    for x in (p, w):
        x.add_argument("--count", action="append", default=[], metavar="AXIS=N")
        x.add_argument("--per", action="append", default=[], metavar="AXIS=N",
                       help="the shard size of an axis (the `per` subcommand's output); needed for an axis with mutants")
        x.add_argument("--runs", required=True)
        x.add_argument("--out-record", required=True)
        x.add_argument("--out-matrix", required=True)
    m = sub.add_parser("per")
    m.add_argument("--out", required=True, help="a cargo-mutants output directory holding the baseline")
    c = sub.add_parser("core-jobs", help="the number of core shard jobs of a record (0 for a night's)")
    c.add_argument("--record", required=True)
    a = sub.add_parser("age")
    a.add_argument("--record", required=True)
    a.add_argument("--now", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "fetch":
            runs = fetch(args.repo, args.self_id, datetime.now(timezone.utc))
            Path(args.out).write_text(json.dumps(runs), encoding="utf-8")
            print(f"read {len(runs)} run(s), {sum(1 for r in runs if r['record'])} with a record")
            return 0
        if args.cmd == "per":
            build_s, test_s = read_baseline(Path(args.out))
            print(per_from_baseline(build_s, test_s))
            print(f"unmutated baseline {build_s:.0f} s build + {test_s:.0f} s test", file=sys.stderr)
            return 0
        if args.cmd == "core-jobs":
            rec = json.loads(Path(args.record).read_text(encoding="utf-8"))
            if rec.get("kind") == "weekly":
                print(int(rec["core"]["jobs"]))
            elif rec.get("kind") in ("nightly", "pr", "aside"):
                print(0)
            else:
                raise ValueError(f"{args.record}: not a record (kind {rec.get('kind')!r})")
            return 0
        if args.cmd == "age":
            rec = json.loads(Path(args.record).read_text(encoding="utf-8"))
            old = stale(rec.get("open", []), args.now)
            for o in old:
                print(f"error: range {range_id(o['base'], o['head'])} (counts {o['counts']}) has been open since "
                      f"{o['since']}: more than {MAX_AGE_DAYS} days without a weekly run measuring it", file=sys.stderr)
            print(f"compared: open ranges {len(rec.get('open', []))}, older than {MAX_AGE_DAYS} days {len(old)}")
            return 1 if old else 0
        runs = json.loads(Path(args.runs).read_text(encoding="utf-8"))
        if args.cmd == "base":
            sha, why = pick_base(runs)
            print(sha or "")
            print(why if sha else f"error: {why}", file=sys.stderr)
            return 0 if sha else NO_BASE
        counts = {k: int(v) for k, v in (c.split("=", 1) for c in args.count)}
        per = {k: int(v) for k, v in (c.split("=", 1) for c in args.per)}
        if args.cmd == "weekly":
            rec, entries = weekly_record(args.head, open_ranges(runs), args.now, counts, per)
            write_out(args, rec, entries)
            core = rec["core"]
            print(f"weekly: {rec['state']}, core mutants {core['counts']}, per shard {core['per']}, core shards "
                  f"{core['shards']}, {len(entries)} job(s), {len(rec['taken'])} open range(s) taken, "
                  f"{len(rec['stale'])} older than {MAX_AGE_DAYS} days")
            if rec["state"] == "over-cap":
                print(f"error: the core plan takes {max(core['shards'].values())} shards on an axis, over the cap of "
                      f"{CORE_CAP} ({WEEKLY_PARALLEL} jobs at a time for {WEEKLY_BUDGET_H} h): no core shard is run "
                      "this week", file=sys.stderr)
                return OVER_CAP
            return 0
        rec, entries = nightly_record(args.mode, args.base, args.head, counts,
                                      open_ranges(runs) if args.mode in CARRYING else [], args.base_reason, per,
                                      base_from=args.base_from)
        if (args.mode, rec["state"]) in (("nightly", "over-cap"), ("aside", "set-aside")):
            rec["open"] = rec["open"] + [{"base": args.base, "head": args.head, "counts": counts,
                                          "per": rec["per"], "since": args.now}]
        write_out(args, rec, entries)
        print(f"{args.mode}: {rec['state']}, {range_id(args.base, args.head)}, mutants {counts}, "
              f"per shard {rec['per']}, shards {rec['shards']}"
              + (": not run here, set aside for the weekly run" if rec["state"] == "set-aside" else ""))
        if rec["state"] == "over-cap":
            print(f"error: the plan of {range_id(args.base, args.head)} takes {max(rec['shards'].values())} shards on "
                  f"an axis, over the cap of {CAP} ({' / '.join(f'{a} {CAP * n}' for a, n in rec['per'].items())} mutants): not run tonight"
                  + (", set aside for the weekly run" if args.mode == "nightly" else ""), file=sys.stderr)
            return OVER_CAP
        return 0
    except (RecordError, ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return ERROR


if __name__ == "__main__":
    sys.exit(main())
