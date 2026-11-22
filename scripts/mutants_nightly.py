#!/usr/bin/env python3
"""The plan of the in-diff mutation runs of .github/workflows/quality-deep.yml: how many shards a night's diff
takes, which commit the diff starts from, and which ranges were set aside for the weekly run.

One job tests about 100 mutants of a diff in its time limit (measured 11-17 to 11-21: 115-180 outcomes in 270
min at -j 2), so a night's plan is cut into shards of at most PER mutants per feature axis, at most CAP shards
per axis. A night whose plan is larger than CAP shards is not run: it is recorded as `over-cap` with its range
base..head and its counts, and the weekly run tests that range (without the cap). Every plan writes a record
(`mutants-record.json`, uploaded as an artifact of the run); nothing is committed.

The base of a night's diff is the head of the latest earlier nightly run that is either

  * measured: its record says `accepted` and every shard job it planned ended its `Measured` step with
    success (every planned mutant of the shard has an outcome; a missed mutant does not change this), or
    its record says `empty` (the diff held no mutant), or
  * over-cap: its range is carried to the weekly run.

A run that was cancelled, whose unmutated tree failed, or that has a killed or unmeasured shard does not
move the base: its lines are in the next night's diff again. With no such run in the window the base is the
commit of 25 hours ago, and the summary says so.

An over-cap range stays open until a weekly run whose record lists it has every shard job of that range
measured. Open ranges are carried in every nightly and weekly record (`open`), so one does not leave with the
window; the carried list is read from the latest such record (a pull request's or a diff_base dispatch's record
carries none and is passed over). A range
that has been open for more than MAX_AGE_DAYS fails the weekly run's age job. A record that is listed but
expired or unreadable is an error (exit 2), never "no open range".

    mutants_nightly.py fetch --repo OWNER/REPO --self RUN_ID --out runs.json
    mutants_nightly.py base  --runs runs.json                         # prints the base sha, or nothing
    mutants_nightly.py plan  --mode nightly|pr --runs runs.json --base B --head H \\
        --count default=N --count features=M --out-record R --out-matrix M   # exit 5 = over the cap
    mutants_nightly.py weekly --runs runs.json --head H --now ISO --out-record R --out-matrix M
    mutants_nightly.py age   --record R --now ISO                     # exit 1 = a range open for too long

Author: Moroya Sakamoto
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

PER = 100            # mutants one shard job is given
CAP = 8              # shards per feature axis in one night
MAX_AGE_DAYS = 8     # an over-cap range older than this fails the weekly age job
WINDOW_DAYS = 21     # how far back the runs are read (the record artifacts are kept 30 days)
AXES = {"default": "", "features": "parallel,gpu-solver-bridge,simd"}
CORE_SHARDS = 16
RECORD_ARTIFACT = "mutants-record"
RECORD_FILE = "mutants-record.json"
MEASURED_STEP = "Measured (every planned mutant of this shard has an outcome)"
CARRYING = ("nightly", "weekly")   # the record kinds whose `open` list is the carried list
EVENTS = ("schedule", "workflow_dispatch")   # the runs that are read back (a pull request's run is never a base)
MAX_PAGES = 10       # of 100 runs per event: more than this in the window is an error, not a cut
OVER_CAP, ERROR = 5, 2


class RecordError(Exception):
    """a record that exists but cannot be read"""


def shards_for(count: int, per: int = PER) -> int:
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


def pick_base(runs: list[dict]) -> tuple[str | None, str]:
    """(the head of the latest nightly run that measured or was set aside, why); None = no such run"""
    check_records(runs)
    skipped = 0
    for r in by_time(runs):
        rec = r.get("record")
        if not rec or rec.get("kind") != "nightly":
            continue
        state = rec.get("state")
        if state == "over-cap":
            return rec["head"], f"run {r['id']} set its range aside for the weekly run (over the cap)"
        if state == "empty":
            return rec["head"], f"run {r['id']} had no mutant in its diff"
        if state == "accepted" and measured(r, rec["base"], rec["head"], rec.get("jobs", 0)):
            return rec["head"], f"run {r['id']} measured every shard"
        skipped += 1
    return None, f"no nightly run in the window measured ({skipped} unmeasured, {len(runs)} run(s) read)"


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
        if rec and rec.get("kind") == "nightly" and rec.get("state") == "over-cap":
            found.setdefault(range_id(rec["base"], rec["head"]),
                             {"base": rec["base"], "head": rec["head"], "counts": rec["counts"], "since": r["created_at"]})
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


def diff_entries(base: str, head: str, counts: dict[str, int], per: int = PER) -> list[dict]:
    out = []
    for axis, features in AXES.items():
        n = shards_for(counts.get(axis, 0), per)
        out += [{"kind": "diff", "shard": k, "of": n, "features": features, "base": base, "head": head,
                 "label": diff_job_label(base, head, k, n, axis),
                 "artifact": f"mutants-diff-{base[:12]}-{head[:12]}-{k}-{axis}"} for k in range(n)]
    return out


def nightly_record(mode: str, base: str, head: str, counts: dict[str, int], carried: list[dict], why: str,
                   per: int = PER, cap: int = CAP) -> tuple[dict, list[dict]]:
    """(the record of a nightly / pr plan, its matrix entries)"""
    unknown = sorted(set(counts) - set(AXES))
    if unknown or set(counts) != set(AXES):
        raise ValueError(f"counts must name exactly {sorted(AXES)}, got {sorted(counts)}")
    shards = {a: shards_for(counts[a], per) for a in AXES}
    if sum(counts.values()) == 0:
        state, entries = "empty", []
    elif max(shards.values()) > cap:
        state, entries = "over-cap", []
    else:
        state, entries = "accepted", diff_entries(base, head, counts, per)
    rec = {"v": 1, "kind": mode, "state": state, "base": base, "head": head, "base_reason": why, "counts": counts,
           "shards": shards, "jobs": len(entries), "open": carried}
    return rec, entries


def weekly_record(head: str, ranges: list[dict], now: str, per: int = PER) -> tuple[dict, list[dict]]:
    """(the record of a weekly plan, its matrix entries): the core shards and every open range"""
    entries = [{"kind": "core", "shard": k, "of": CORE_SHARDS, "features": f, "base": "", "head": head,
                "label": f"core shard {k}/{CORE_SHARDS}, {a}",
                "artifact": f"mutants-out-{k}-{'default' if a == 'default' else 'parallel'}"}
               for k in range(CORE_SHARDS) for a, f in AXES.items()]
    taken = []
    for o in ranges:
        mine = diff_entries(o["base"], o["head"], o["counts"], per)
        if not mine:
            raise ValueError(f"open range {range_id(o['base'], o['head'])} has no mutant to test (counts {o['counts']})")
        entries += mine
        taken.append({"base": o["base"], "head": o["head"], "jobs": len(mine)})
    rec = {"v": 1, "kind": "weekly", "state": "accepted", "head": head, "taken": taken, "open": ranges,
           "stale": [range_id(o["base"], o["head"]) for o in stale(ranges, now)]}
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
    if not isinstance(rec, dict) or rec.get("v") != 1 or rec.get("kind") not in ("nightly", "weekly", "pr"):
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
    p.add_argument("--mode", choices=("nightly", "pr"), required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--base-reason", default="")
    p.add_argument("--count", action="append", default=[], metavar="AXIS=N")
    p.add_argument("--now", required=True)
    w = sub.add_parser("weekly")
    w.add_argument("--head", required=True)
    w.add_argument("--now", required=True)
    for x in (p, w):
        x.add_argument("--runs", required=True)
        x.add_argument("--out-record", required=True)
        x.add_argument("--out-matrix", required=True)
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
            print(why, file=sys.stderr)
            return 0
        if args.cmd == "weekly":
            rec, entries = weekly_record(args.head, open_ranges(runs), args.now)
            write_out(args, rec, entries)
            print(f"weekly: {len(entries)} job(s), {len(rec['taken'])} open range(s) taken, {len(rec['stale'])} older "
                  f"than {MAX_AGE_DAYS} days")
            return 0
        counts = {k: int(v) for k, v in (c.split("=", 1) for c in args.count)}
        rec, entries = nightly_record(args.mode, args.base, args.head, counts,
                                      open_ranges(runs) if args.mode == "nightly" else [], args.base_reason)
        if rec["state"] == "over-cap" and args.mode == "nightly":
            rec["open"] = rec["open"] + [{"base": args.base, "head": args.head, "counts": counts,
                                          "since": args.now}]
        write_out(args, rec, entries)
        print(f"{args.mode}: {rec['state']}, {range_id(args.base, args.head)}, mutants {counts}, shards {rec['shards']}")
        if rec["state"] == "over-cap":
            print(f"error: the plan of {range_id(args.base, args.head)} takes {max(rec['shards'].values())} shards on "
                  f"an axis, over the cap of {CAP} ({CAP * PER} mutants): not run tonight"
                  + (", set aside for the weekly run" if args.mode == "nightly" else ""), file=sys.stderr)
            return OVER_CAP
        return 0
    except (RecordError, ValueError, KeyError, OSError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return ERROR


if __name__ == "__main__":
    sys.exit(main())
