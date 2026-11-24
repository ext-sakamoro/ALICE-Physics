#!/usr/bin/env python3
"""Tests of scripts/mutants_nightly.py: the shard count, the base of a night's diff, the ranges set aside for the
weekly run and the age rule.

Author: Moroya Sakamoto
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutants_nightly as mn  # noqa: E402

A, B, C, D = ("a" * 40, "b" * 40, "c" * 40, "d" * 40)
P = {"default": 100, "features": 100}   # the shard size the plans of these cases were measured at
OK = [{"name": "Set up job", "conclusion": "success"}, {"name": mn.MEASURED_STEP, "conclusion": "success"}]
KILLED = [{"name": "Set up job", "conclusion": "success"}, {"name": mn.MEASURED_STEP, "conclusion": "failure"}]


OUT = 9000   # run ids from here on are of runs outside the window of a case


def core_steps(e: dict, conclusion: str = "success") -> list[dict]:
    return [OK[0], {"name": f"{mn.CORE_STEP}8 module, shard {e['shard']}/{e['of']}, lib test のみ)", "conclusion": conclusion}]


def jobs_of(entries: list[dict], steps=None) -> list[dict]:
    """the jobs a plan's entries become: a range shard ends its Measured step, a core shard its mutants step
    (`steps` replaces those of the range shards)"""
    return [{"name": f"Mutation testing ({e['label']})", "conclusion": "success",
             "steps": core_steps(e) if e["kind"] == "core" else (steps or OK)} for e in entries]


def night(run_id: int, day: int, base: str, head: str, counts=None, steps=None, drop: int = 0, carried=None) -> dict:
    """a nightly run of 2026-11-<day> with its record and the jobs its plan asked for (`drop` of them missing)"""
    counts = counts or {"default": 150, "features": 150}
    # (its base came from the record of a run that has left the window)
    rec, entries = mn.nightly_record("nightly", base, head, counts, carried or [], f"run {OUT + run_id} measured every shard",
                                     P, base_run=OUT + run_id)
    if rec["state"] == "over-cap":
        rec["open"] = rec["open"] + [{"base": base, "head": head, "counts": counts, "per": rec["per"],
                                     "since": f"2026-11-{day:02d}T22:00:00Z"}]
    return {"id": run_id, "head_sha": head, "created_at": f"2026-11-{day:02d}T22:00:00Z", "conclusion": "success",
            "jobs": jobs_of(entries, steps)[drop:], "record": rec, "record_error": None}


def by_the_clock(run_id: int, day: int, base: str, head: str, counts=None) -> dict:
    """a night as they were planned until 2026-11-25 when no record gave a base: on the commit of 25 hours ago"""
    run = night(run_id, day, base, head, counts=counts or {"default": 0, "features": 0})
    del run["record"]["base_from"], run["record"]["base_run"]
    run["record"]["base_reason"] = ("no nightly run in the window measured (5 unmeasured, 21 run(s) read): "
                                    "falling back to the commit of 25 hours ago")
    return run


def after(run: dict, source: dict, old: bool = False) -> dict:
    """`run` with the base_reason of a night whose base was the head of `source`"""
    run["record"]["base_reason"] = f"run {source['id']} had no mutant in its diff"
    run["record"]["base_run"] = source["id"]
    if old:
        del run["record"]["base_from"], run["record"]["base_run"]
    return run


def aside(run_id: int, day: int, base: str, head: str, counts=None, carried=None) -> dict:
    """a dispatch with diff_base and set_aside of 2026-11-<day>: no job, its range recorded as open"""
    counts = counts or BIG
    rec, entries = mn.nightly_record("aside", base, head, counts, carried or [], "the dispatch's diff_base", P)
    assert not entries
    if rec["state"] == "set-aside":
        rec["open"] = rec["open"] + [{"base": base, "head": head, "counts": counts, "per": rec["per"],
                                     "since": f"2026-11-{day:02d}T03:00:00Z"}]
    return {"id": run_id, "head_sha": head, "created_at": f"2026-11-{day:02d}T03:00:00Z", "conclusion": "success",
            "jobs": [], "record": rec, "record_error": None}


def cancelled(run_id: int, day: int, head: str) -> dict:
    return {"id": run_id, "head_sha": head, "created_at": f"2026-11-{day:02d}T22:00:00Z", "conclusion": "cancelled",
            "jobs": [], "record": None, "record_error": None}


# the core of 11-15 and 11-22: 16 shards of 304 an axis, a baseline of 85 s build + 13 s test
CORE_N = {"default": 4864, "features": 4864}
CORE_PER = {"default": 199, "features": 199}
CORE_ARGS = ("--count", "default=4864", "--count", "features=4864", "--per", "default=199", "--per", "features=199")


def weekly(run_id: int, day: int, head: str, ranges: list[dict], steps=None, skip: bool = False) -> dict:
    rec, entries = mn.weekly_record(head, ranges, f"2026-11-{day:02d}T18:00:00Z", CORE_N, CORE_PER)
    jobs = jobs_of([e for e in entries if e["kind"] == "core" or not skip], steps)
    return {"id": run_id, "head_sha": head, "created_at": f"2026-11-{day:02d}T18:00:00Z", "conclusion": "success",
            "jobs": jobs, "record": rec, "record_error": None}


BIG = {"default": 1634, "features": 1634}


def weekly_over(run_id: int, day: int, head: str) -> dict:
    """a weekly run whose core plan is over the cap: no core job"""
    n = {"default": 199 * (mn.CORE_CAP + 1), "features": 199}
    rec, entries = mn.weekly_record(head, [], f"2026-11-{day:02d}T18:00:00Z", n, CORE_PER)
    assert rec["state"] == "over-cap" and not entries
    return {"id": run_id, "head_sha": head, "created_at": f"2026-11-{day:02d}T18:00:00Z", "conclusion": "failure",
            "jobs": [], "record": rec, "record_error": None}


class Shards(unittest.TestCase):
    def test_a_shard_holds_at_most_per_mutants(self):
        self.assertEqual([mn.shards_for(n, 100) for n in (0, 1, 100, 101, 388, 800, 801, 1634)], [0, 1, 1, 2, 4, 8, 9, 17])

    def test_the_cap_is_per_axis_and_an_over_cap_plan_has_no_job(self):
        rec, entries = mn.nightly_record("nightly", A, B, {"default": 800, "features": 105}, [], "", P, base_run=1)
        self.assertEqual((rec["state"], rec["shards"], len(entries), rec["jobs"]), ("accepted", {"default": 8, "features": 2}, 10, 10))
        rec, entries = mn.nightly_record("nightly", A, B, {"default": 105, "features": 801}, [], "", P, base_run=1)
        self.assertEqual((rec["state"], entries, rec["jobs"]), ("over-cap", [], 0))

    def test_an_empty_plan_is_empty_not_accepted(self):
        rec, entries = mn.nightly_record("nightly", A, B, {"default": 0, "features": 0}, [], "", P, base_run=1)
        self.assertEqual((rec["state"], entries), ("empty", []))

    def test_every_entry_names_its_range_shard_and_axis(self):
        _, entries = mn.nightly_record("nightly", A, B, {"default": 101, "features": 1}, [], "", P, base_run=1)
        self.assertEqual([(e["shard"], e["of"], e["features"]) for e in entries],
                         [(0, 2, ""), (1, 2, ""), (0, 1, "parallel,gpu-solver-bridge,simd")])
        self.assertEqual(entries[0]["label"], f"in-diff {'a' * 12}..{'b' * 12} shard 0/2, default")
        self.assertTrue(all(e["base"] == A and e["head"] == B and e["kind"] == "diff" for e in entries))
        self.assertEqual(entries[2]["artifact"], f"mutants-diff-{'a' * 12}-{'b' * 12}-0-features")
        self.assertEqual(len({e["artifact"] for e in entries}), 3)

    def test_counts_must_name_both_axes(self):
        for counts in ({"default": 1}, {"default": 1, "features": 1, "other": 1}):
            with self.assertRaises(ValueError):
                mn.nightly_record("nightly", A, B, counts, [], "", P)


class Base(unittest.TestCase):
    def test_the_latest_measured_night_is_the_base(self):
        self.assertEqual(mn.pick_base([night(1, 18, A, B), night(2, 19, B, C)])[0], C)

    def test_a_missed_mutant_does_not_stop_a_night_from_being_the_base(self):
        run = night(2, 19, B, C)
        for j in run["jobs"]:
            j["conclusion"] = "failure"   # (a missed mutant fails the job; the Measured step still passed)
        self.assertEqual(mn.pick_base([night(1, 18, A, B), run])[0], C)

    def test_a_cancelled_night_is_passed_over_for_the_one_before(self):
        self.assertEqual(mn.pick_base([night(1, 18, A, B), cancelled(2, 19, C)])[0], B)

    def test_a_night_with_a_killed_shard_or_a_missing_shard_job_is_passed_over(self):
        one_killed = night(2, 19, B, C)
        one_killed["jobs"][1]["steps"] = KILLED
        self.assertEqual(mn.pick_base([night(1, 18, A, B), one_killed])[0], B)
        self.assertEqual(mn.pick_base([night(1, 18, A, B), night(2, 19, B, C, drop=1)])[0], B)
        self.assertEqual(mn.pick_base([night(1, 18, A, B), night(2, 19, B, C, steps=[OK[0]])])[0], B)

    def test_an_over_cap_night_moves_the_base(self):
        self.assertEqual(mn.pick_base([night(1, 18, A, B), night(2, 19, B, C, counts=BIG)])[0], C)

    def test_an_empty_night_moves_the_base(self):
        self.assertEqual(mn.pick_base([night(1, 18, A, B), night(2, 19, B, C, counts={"default": 0, "features": 0})])[0], C)

    def test_a_weekly_or_a_pr_record_is_never_the_base(self):
        pr = night(3, 20, C, D)
        pr["record"]["kind"] = "pr"
        self.assertEqual(mn.pick_base([night(1, 18, A, B), weekly(2, 19, C, []), pr])[0], B)

    def test_no_measured_night_gives_no_base_and_says_how_many_were_read(self):
        sha, why = mn.pick_base([cancelled(1, 18, B), night(2, 19, B, C, steps=KILLED)])
        self.assertIsNone(sha)
        self.assertIn("1 unmeasured, 0 planned on a base no record gave, 2 run(s) read", why)
        self.assertIsNone(mn.pick_base([])[0])

    def test_an_empty_night_planned_on_a_base_by_the_clock_is_passed_over_and_the_reason_says_so(self):
        # the nights of 11-18 to 11-24: every one empty on the commit of 25 hours ago, so each moved the base
        # and the lines between the last measured head and that commit were in no night's diff
        sha, why = mn.pick_base([night(1, 18, A, B), by_the_clock(2, 19, C, D)])
        self.assertEqual(sha, B)
        self.assertIn("run 1 measured every shard", why)
        self.assertIn("1 later night(s) passed over", why)
        sha, why = mn.pick_base([by_the_clock(2, 19, C, D)])
        self.assertIsNone(sha)
        self.assertIn("0 unmeasured, 1 planned on a base no record gave, 1 run(s) read", why)

    def test_a_night_whose_base_was_the_head_of_such_a_night_is_passed_over_too(self):
        first = by_the_clock(2, 19, B, C)
        for old in (True, False):
            with self.subTest(old=old):
                second = after(night(3, 20, C, D), first, old=old)
                third = after(night(4, 21, D, "e" * 40), second, old=old)
                sha, why = mn.pick_base([night(1, 18, A, B), first, second, third])
                self.assertEqual(sha, B)
                self.assertIn("3 later night(s) passed over", why)
        # (a night whose base came from a run that has left the window is not judged by it)
        self.assertEqual(mn.pick_base([after(night(3, 20, C, D), {"id": 999})])[0], D)

    def test_a_night_that_names_no_run_for_its_base_is_passed_over_and_one_whose_run_left_the_window_is_not(self):
        # a record gave the base, it says, and names no run: nothing ties it to one (read as "left the window",
        # it would be trusted on its word)
        for field in (None, "", "12", 12.0, True):
            with self.subTest(base_run=field):
                run = night(2, 19, B, C)
                run["record"]["base_run"] = field
                sha, why = mn.pick_base([night(1, 18, A, B), run])
                self.assertEqual(sha, B)
                self.assertIn("1 later night(s) passed over", why)
        run = night(2, 19, B, C)
        del run["record"]["base_run"]          # (the field is read, not the prose, once the record has base_from)
        self.assertEqual(mn.pick_base([night(1, 18, A, B), run])[0], B)
        # a record from before the fields: its prose names the run, or it names none
        old = after(night(2, 19, B, C), {"id": 999}, old=True)
        self.assertEqual(mn.pick_base([night(1, 18, A, B), old])[0], C, "run 999 has left the window: not judged")
        for prose in ("", "the run before", "run measured every shard", "Run 5 measured every shard"):
            with self.subTest(prose=prose):
                old["record"]["base_reason"] = prose
                self.assertEqual(mn.pick_base([night(1, 18, A, B), old])[0], B)
        # the id is the field's even when the prose names another run
        run = night(3, 20, C, D)
        run["record"]["base_run"] = 2
        run["record"]["base_reason"] = "run 1 measured every shard"
        self.assertEqual(mn.pick_base([night(1, 18, A, B), by_the_clock(2, 19, B, C), run])[0], B)

    def test_a_plan_names_the_run_of_its_base_exactly_when_a_record_gave_it(self):
        with self.assertRaisesRegex(ValueError, "names the run of that record"):
            mn.nightly_record("nightly", A, B, {"default": 1, "features": 1}, [], "", P)
        for mode in ("pr", "aside"):
            with self.assertRaisesRegex(ValueError, "no other base names one"):
                mn.nightly_record(mode, A, B, {"default": 1, "features": 1}, [], "", P, base_run=5)
        rec, _ = mn.nightly_record("nightly", A, B, {"default": 1, "features": 1}, [], "", P, base_run=5)
        self.assertEqual((rec["base_from"], rec["base_run"]), ("record", 5))
        self.assertIsNone(mn.nightly_record("aside", A, B, {"default": 1, "features": 1}, [], "", P)[0]["base_run"])
        self.assertEqual(mn.pick([night(1, 18, A, B), night(2, 19, B, C)])[2], 2, "the run whose record gave the base")
        self.assertIsNone(mn.pick([])[2])

    def test_the_anchor_is_the_base_a_night_would_take_and_else_the_latest_weekly_head(self):
        runs = [night(1, 12, A, B), by_the_clock(2, 19, C, D)]
        self.assertEqual(mn.anchor(runs)[0], B, "not the head of the night planned by the clock")
        self.assertEqual(mn.anchor(runs + [aside(3, 20, B, D)])[0], D)
        only_clock = [by_the_clock(2, 19, C, D)]
        sha, why = mn.anchor(only_clock + [weekly(5, 14, A, []), weekly(6, 21, B, [])])
        self.assertEqual(sha, B)
        self.assertIn("run 6 is the latest weekly run that measured its core", why)
        self.assertNotIn("passed over", why.split("; run 6")[1])
        sha, why = mn.anchor(only_clock)
        self.assertIsNone(sha)
        self.assertIn("no weekly run in the window measured its core (0 weekly record(s) read)", why)

    def test_a_weekly_run_that_did_not_measure_its_core_is_not_an_anchor(self):
        only_clock = [by_the_clock(2, 19, C, D)]
        good = weekly(5, 14, A, [])
        self.assertTrue(mn.core_measured(good))

        def broken(how: str) -> dict:
            w = weekly(6, 21, B, [])
            if how == "a core shard stopped at the limit":
                w["jobs"][3]["steps"] = core_steps({"shard": 3, "of": 25}, "failure")
            elif how == "a core shard cancelled before its step":
                w["jobs"][3]["steps"] = [OK[0]]
            elif how == "a core shard job missing":
                del w["jobs"][3]
            elif how == "cancelled before any job":
                w["jobs"] = []
            elif how == "only the Measured step of a range shard":
                w["jobs"][3]["steps"] = OK
            elif how == "over the cap":
                w = weekly_over(6, 21, B)
            elif how == "a record that is not of an accepted plan":
                w["record"]["state"] = "over-cap"       # (whatever its jobs say)
            elif how == "a record that plans no core job":
                w["record"]["core"]["jobs"] = 0
                w["jobs"] = []
            return w
        for how in ("a core shard stopped at the limit", "a core shard cancelled before its step", "a core shard job missing",
                    "cancelled before any job", "only the Measured step of a range shard", "over the cap",
                    "a record that is not of an accepted plan",
                    "a record that plans no core job"):
            with self.subTest(how=how):
                bad = broken(how)
                self.assertFalse(mn.core_measured(bad))
                sha, why = mn.anchor(only_clock + [good, bad])
                self.assertEqual(sha, A, "the weekly run before it, which measured")
                self.assertIn("run 5 is the latest weekly run that measured its core (1 later weekly run(s) passed over", why)
                sha, why = mn.anchor(only_clock + [bad])
                self.assertIsNone(sha)
                self.assertIn("no weekly run in the window measured its core (1 weekly record(s) read)", why)
        # a missed mutant fails the job, not the step: the core was still measured
        missed = weekly(6, 21, B, [])
        for j in missed["jobs"]:
            j["conclusion"] = "failure"
        self.assertEqual(mn.anchor(only_clock + [good, missed])[0], B)
        # range shards of the same run do not count as core shards, in either direction
        with_range = weekly(7, 21, B, [{"base": A, "head": B, "counts": BIG, "per": P, "since": "2026-11-19T22:00:00Z"}], steps=KILLED)
        self.assertTrue(mn.core_measured(with_range), "its range shards were killed, its core ran")

    def test_a_record_that_says_its_base_came_from_elsewhere_is_never_a_base(self):
        run = night(2, 19, B, C)
        for other in ("dispatch", "pull-request", "clock", ""):
            with self.subTest(base_from=other):
                run["record"]["base_from"] = other
                self.assertEqual(mn.pick_base([night(1, 18, A, B), run])[0], B)
        with self.assertRaisesRegex(ValueError, "a nightly plan takes its base from record, not from dispatch"):
            mn.nightly_record("nightly", A, B, {"default": 1, "features": 1}, [], "", P, base_from="dispatch")
        with self.assertRaisesRegex(ValueError, "aside plan takes its base from dispatch, not from record"):
            mn.nightly_record("aside", A, B, {"default": 1, "features": 1}, [], "", P, base_from="record")

    def test_a_range_set_aside_by_a_dispatch_moves_the_base_past_the_nights_passed_over(self):
        runs = [night(1, 12, A, B), by_the_clock(2, 19, C, D), aside(3, 20, B, D)]
        sha, why = mn.pick_base(runs)
        self.assertEqual(sha, D)
        self.assertIn("run 3 set its range aside for the weekly run (a dispatch)", why)
        nxt = night(4, 20, D, "e" * 40)
        nxt["record"]["base_reason"] = why
        self.assertEqual(mn.pick_base(runs + [nxt])[0], "e" * 40)

    def test_another_range_s_shard_jobs_do_not_count_for_this_one(self):
        run = night(2, 19, B, C)
        other = night(9, 19, A, D)
        run["jobs"] = other["jobs"]
        self.assertEqual(mn.pick_base([night(1, 18, A, B), run])[0], B)


class Open(unittest.TestCase):
    def test_an_over_cap_range_is_carried_into_the_weekly_plan_without_the_cap(self):
        runs = [night(1, 17, A, B), night(2, 19, B, C, counts=BIG)]
        ranges = mn.open_ranges(runs)
        self.assertEqual([(o["base"], o["head"], o["counts"], o["since"]) for o in ranges], [(B, C, BIG, "2026-11-19T22:00:00Z")])
        rec, entries = mn.weekly_record(D, ranges, "2026-11-22T18:00:00Z", CORE_N, CORE_PER)
        diff = [e for e in entries if e["kind"] == "diff"]
        self.assertEqual((len(entries) - len(diff), len(diff)), (50, 34))
        self.assertTrue(all(e["base"] == B and e["head"] == C for e in diff))
        self.assertEqual(rec["taken"], [{"base": B, "head": C, "jobs": 34}])
        self.assertEqual(rec["stale"], [])
        # (the ratchet job reads the 32 core artifacts by this name)
        self.assertEqual(sorted(e["artifact"] for e in entries if e["kind"] == "core"),
                         sorted(f"mutants-out-{k}-{x}" for k in range(25) for x in ("default", "parallel")))
        self.assertEqual(len({e["artifact"] for e in entries}), 84)

    def test_a_range_set_aside_by_a_dispatch_is_carried_to_the_weekly_plan_and_closed_by_its_measure(self):
        runs = [night(1, 12, A, B), by_the_clock(2, 19, C, D), aside(3, 20, B, D)]
        ranges = mn.open_ranges(runs)
        self.assertEqual([(o["base"], o["head"], o["counts"], o["since"]) for o in ranges],
                         [(B, D, BIG, "2026-11-20T03:00:00Z")])
        rec, entries = mn.weekly_record("e" * 40, ranges, "2026-11-21T18:00:00Z", CORE_N, CORE_PER)
        mine = [e for e in entries if e["kind"] == "diff"]
        self.assertEqual(len(mine), 2 * 17)   # (1634 mutants at 100 a shard on each axis: no cap in the weekly run)
        self.assertEqual(rec["taken"], [{"base": B, "head": D, "jobs": 34}])
        # the nights after it carry it; a weekly run that measured it closes it, one that did not leaves it
        later = night(4, 20, D, "e" * 40, carried=ranges)
        self.assertEqual(len(mn.open_ranges(runs + [later])), 1)
        self.assertEqual(mn.open_ranges(runs + [later, weekly(5, 21, "e" * 40, ranges)]), [])
        self.assertEqual(len(mn.open_ranges(runs + [later, weekly(5, 21, "e" * 40, ranges, skip=True)])), 1)
        # and the age job counts it from the day of the dispatch
        self.assertEqual(len(mn.stale(ranges, "2026-11-28T03:00:01Z")), 1)
        self.assertEqual(mn.stale(ranges, "2026-11-28T03:00:00Z"), [])

    def test_a_range_set_aside_by_a_dispatch_is_read_from_its_record_when_no_list_carries_it(self):
        run = aside(3, 20, B, D)
        run["record"]["open"] = []        # as if only the record's own range were known
        self.assertEqual([(o["base"], o["head"], o["per"], o["since"]) for o in mn.open_ranges([run])],
                         [(B, D, P, "2026-11-20T03:00:00Z")])

    def rng(self, base: str, head: str, n: int, day: int) -> dict:
        return {"base": base, "head": head, "counts": {"default": n, "features": n}, "per": P,
                "since": f"2026-11-{day:02d}T03:00:00Z"}

    def test_a_weekly_plan_never_has_more_entries_than_a_matrix_holds_and_what_does_not_fit_stays_open(self):
        self.assertEqual(mn.MATRIX_MAX, 256)
        # the core is 50 jobs; ranges of 100, 120 and 40 jobs, oldest first
        ranges = [self.rng(A, B, 5000, 10), self.rng(B, C, 6000, 11), self.rng(C, D, 2000, 12)]
        rec, entries = mn.weekly_record("e" * 40, ranges, "2026-11-14T18:00:00Z", CORE_N, CORE_PER)
        self.assertLessEqual(len(entries), mn.MATRIX_MAX)
        self.assertEqual(len(entries), 50 + 100 + 40)
        self.assertEqual([(t["base"], t["jobs"]) for t in rec["taken"]], [(A, 100), (C, 40)],
                         "the oldest first, and a later smaller one that still fits")
        self.assertEqual(rec["deferred"], [{"base": B, "head": C, "jobs": 120}])
        self.assertEqual(rec["open"], ranges, "nothing leaves the open list by not fitting")
        # the week after, with the first and the third measured, the deferred one is taken
        w = weekly(5, 14, "e" * 40, ranges)
        left = mn.open_ranges([w])
        self.assertEqual([(o["base"], o["head"]) for o in left], [(B, C)])
        rec, entries = mn.weekly_record("f" * 40, left, "2026-11-21T18:00:00Z", CORE_N, CORE_PER)
        self.assertEqual((len(entries), rec["deferred"]), (50 + 120, []))
        # a deferred range is in the next weekly's input as long as it is not measured, and the age job counts it
        self.assertEqual(mn.open_ranges([w, weekly(6, 21, "f" * 40, left, skip=True)]), left)
        old = mn.weekly_record("f" * 40, left, "2026-11-19T03:00:01Z", CORE_N, CORE_PER)[0]
        self.assertEqual(old["stale"], [mn.range_id(B, C)], "open since 11-11: past the 8 days")
        self.assertEqual(mn.weekly_record("f" * 40, left, "2026-11-19T03:00:00Z", CORE_N, CORE_PER)[0]["stale"], [])
        # exactly at the limit is taken, one job more is not
        fits = mn.weekly_record(D, [self.rng(A, B, 10300, 10)], "2026-11-14T18:00:00Z", CORE_N, CORE_PER)
        self.assertEqual((len(fits[1]), fits[0]["deferred"]), (256, []))
        over = mn.weekly_record(D, [self.rng(A, B, 10301, 10)], "2026-11-14T18:00:00Z", CORE_N, CORE_PER)
        self.assertEqual((len(over[1]), [d["jobs"] for d in over[0]["deferred"]]), (50, [208]))

    def test_a_range_larger_than_a_matrix_is_refused_where_it_is_recorded_and_by_the_weekly_plan(self):
        with self.assertRaisesRegex(ValueError, "takes 258 jobs .*more than the 256 a matrix holds.*in pieces"):
            mn.nightly_record("aside", A, B, {"default": 12900, "features": 12900}, [], "", P)
        rec, _ = mn.nightly_record("aside", A, B, {"default": 12800, "features": 12800}, [], "", P)
        self.assertEqual((rec["state"], sum(rec["shards"].values())), ("set-aside", 256))
        # the range of 2026-11-25 (8824 mutants an axis at 34 a shard: 520 jobs) is such a one
        with self.assertRaisesRegex(ValueError, "takes 520 jobs"):
            mn.nightly_record("aside", A, B, {"default": 8824, "features": 8824}, [], "", {"default": 34, "features": 34})
        with self.assertRaisesRegex(ValueError, "takes 258 jobs.*no weekly run can take it"):
            mn.weekly_record(D, [self.rng(A, B, 12900, 10)], "2026-11-14T18:00:00Z", CORE_N, CORE_PER)

    def test_a_dispatch_that_sets_aside_a_range_without_a_mutant_opens_nothing(self):
        run = aside(3, 20, B, D, counts={"default": 0, "features": 0})
        self.assertEqual(run["record"]["state"], "empty")
        self.assertEqual(mn.open_ranges([run]), [])
        self.assertEqual(mn.pick_base([run])[0], D)

    def test_a_weekly_run_that_measured_the_range_closes_it(self):
        over = night(2, 19, B, C, counts=BIG)
        w = weekly(3, 22, D, mn.open_ranges([over]))
        self.assertEqual(mn.open_ranges([over, w]), [])

    def test_a_weekly_run_that_skipped_or_did_not_finish_the_range_leaves_it_open(self):
        over = night(2, 19, B, C, counts=BIG)
        ranges = mn.open_ranges([over])
        for w in (weekly(3, 22, D, ranges, skip=True), weekly(3, 22, D, ranges, steps=KILLED), weekly(3, 22, D, [])):
            self.assertEqual(len(mn.open_ranges([over, w])), 1)
        one_short = weekly(3, 22, D, ranges)
        one_short["jobs"].pop()
        self.assertEqual(len(mn.open_ranges([over, one_short])), 1)

    def test_a_range_stays_open_through_the_carried_list_when_its_night_left_the_window(self):
        over = night(2, 1, B, C, counts=BIG)
        later = night(5, 25, C, D, carried=mn.open_ranges([over]))
        self.assertEqual([o["since"] for o in mn.open_ranges([later])], ["2026-11-01T22:00:00Z"])

    def test_the_carried_list_is_the_latest_record_s_only(self):
        over = night(2, 1, B, C, counts=BIG)
        old = night(4, 20, C, D, carried=mn.open_ranges([over]))
        new = night(5, 25, C, D, carried=[])
        self.assertEqual(mn.open_ranges([old, new]), [])

    def test_a_newer_pr_record_does_not_empty_the_carried_list(self):
        over = night(2, 1, B, C, counts=BIG)
        carrying = night(4, 20, C, D, carried=mn.open_ranges([over]))
        pr = night(5, 21, C, D)
        pr["record"]["kind"], pr["record"]["open"] = "pr", []
        self.assertEqual([o["since"] for o in mn.open_ranges([carrying, pr])], ["2026-11-01T22:00:00Z"])

    def test_an_unreadable_record_is_an_error_for_the_base_and_for_the_open_ranges(self):
        bad = cancelled(3, 20, C)
        bad["record_error"] = "its record artifact has expired"
        for f in (mn.pick_base, mn.open_ranges):
            with self.assertRaises(mn.RecordError) as e:
                f([night(1, 18, A, B), bad])
            self.assertIn("run 3: its record artifact has expired", str(e.exception))

    def test_a_weekly_plan_refuses_an_open_range_without_a_mutant(self):
        with self.assertRaises(ValueError):
            mn.weekly_record(D, [{"base": B, "head": C, "counts": {"default": 0, "features": 0}, "since": "2026-11-19T22:00:00Z"}],
                             "2026-11-22T18:00:00Z", CORE_N, CORE_PER)


class Core(unittest.TestCase):
    """the weekly core shards are cut by the baseline of the week, under a cap"""
    NOW = "2026-11-22T18:00:00Z"

    def core(self, counts, per, **kw):
        rec, entries = mn.weekly_record(D, [], self.NOW, counts, per, **kw)
        return rec, [e for e in entries if e["kind"] == "core"]

    def test_the_weeks_that_ended_and_the_weeks_that_were_stopped(self):
        # 11-08: 25 s build + 4 s test, 4160 mutants an axis: the 16 shards of 260 ended in 46-170 min
        quick = mn.per_from_baseline(25, 4)
        self.assertGreaterEqual(quick, 260, "the shards that ended are within what the baseline gives")
        self.assertLessEqual(mn.shards_for(4160, quick), 16)
        # 11-15 and 11-22: 85 s + 13 s, 4864 an axis: 16 shards of 304 were stopped at the limit
        slow = mn.per_from_baseline(85, 13)
        self.assertEqual(slow, 199)
        self.assertLess(slow, 304, "a shard of 304 is more than the baseline gives")
        rec, core = self.core(CORE_N, {"default": slow, "features": slow})
        self.assertEqual((rec["state"], rec["core"]["shards"], len(core)), ("accepted", {"default": 25, "features": 25}, 50))

    def test_each_axis_is_cut_by_its_own_count_and_size(self):
        rec, core = self.core({"default": 400, "features": 401}, {"default": 100, "features": 200})
        self.assertEqual([(e["shard"], e["of"], e["features"]) for e in core],
                         [(k, 4, "") for k in range(4)] + [(k, 3, mn.AXES["features"]) for k in range(3)])
        self.assertEqual([e["label"] for e in core][3:5], ["core shard 3/4, default", "core shard 0/3, features"])
        self.assertEqual([e["artifact"] for e in core][3:5], ["mutants-out-3-default", "mutants-out-0-parallel"])
        self.assertEqual(len({e["artifact"] for e in core}), 7)
        self.assertTrue(all(e["head"] == D and e["base"] == "" for e in core))
        self.assertEqual(rec["core"], {"counts": {"default": 400, "features": 401}, "per": {"default": 100, "features": 200},
                                       "shards": {"default": 4, "features": 3}, "jobs": 7})

    def test_the_cap_is_what_the_weekly_jobs_at_a_time_end_before_the_next_night(self):
        rounds = mn.WEEKLY_BUDGET_H * 60 // mn.JOB_MIN
        self.assertEqual(mn.CORE_CAP * len(mn.AXES), rounds * mn.WEEKLY_PARALLEL)
        self.assertLessEqual(mn.CORE_CAP * len(mn.AXES) * mn.JOB_MIN, mn.WEEKLY_BUDGET_H * 60 * mn.WEEKLY_PARALLEL)
        self.assertGreater(mn.JOB_MIN * 60, mn.STOP_S, "a job is the stop and the setup around it")
        self.assertLess(mn.JOB_MIN, 300, "under the job's own limit")

    def test_a_core_plan_over_the_cap_on_one_axis_has_no_core_job_and_still_takes_the_open_ranges(self):
        at = {"default": 100 * mn.CORE_CAP, "features": 1}
        self.assertEqual(self.core(at, {"default": 100, "features": 100})[0]["state"], "accepted")
        over = {"default": 100 * mn.CORE_CAP + 1, "features": 1}
        o = {"base": B, "head": C, "counts": {"default": 150, "features": 0}, "per": {"default": 100, "features": 100},
             "since": "2026-11-19T22:00:00Z"}
        rec, entries = mn.weekly_record(D, [o], self.NOW, over, {"default": 100, "features": 100})
        self.assertEqual(rec["state"], "over-cap")
        self.assertEqual([e["kind"] for e in entries], ["diff", "diff"])
        self.assertEqual((rec["core"]["jobs"], rec["core"]["shards"]["default"]), (0, mn.CORE_CAP + 1))
        self.assertEqual(rec["taken"], [{"base": B, "head": C, "jobs": 2}])

    def test_an_axis_without_a_core_mutant_or_without_its_size_is_an_error(self):
        for counts, per in (({"default": 0, "features": 10}, {"default": 5, "features": 5}),
                            ({"default": 10}, {"default": 5, "features": 5}),
                            ({"default": 10, "features": 10}, {"default": 5}),
                            ({"default": 10, "features": 10}, {"default": 5, "features": 0})):
            with self.subTest(counts=counts, per=per), self.assertRaises(ValueError):
                mn.weekly_record(D, [], self.NOW, counts, per)


class Age(unittest.TestCase):
    RANGE = {"base": B, "head": C, "counts": BIG, "since": "2026-11-10T22:00:00Z"}

    def test_a_range_is_stale_after_eight_days_not_at_eight(self):
        self.assertEqual(mn.stale([self.RANGE], "2026-11-18T22:00:00Z"), [])
        self.assertEqual(mn.stale([self.RANGE], "2026-11-18T22:00:01Z"), [self.RANGE])

    def test_the_weekly_record_names_the_stale_ranges(self):
        rec, _ = mn.weekly_record(D, [self.RANGE], "2026-11-22T18:00:00Z", CORE_N, CORE_PER)
        self.assertEqual(rec["stale"], [f"{'b' * 12}..{'c' * 12}"])


class Per(unittest.TestCase):
    """The shard size comes from the unmutated baseline of the night, not from a constant."""

    def out(self, outcomes: list[dict]) -> Path:
        d = Path(tempfile.mkdtemp()) / "mutants.out"
        (d / "mutants.out").mkdir(parents=True)
        (d / "mutants.out" / "outcomes.json").write_text(json.dumps({"outcomes": outcomes}), encoding="utf-8")
        return d

    @staticmethod
    def baseline(build=155.2, test=264.4, summary="Success", test_status="Success") -> dict:
        return {"scenario": "Baseline", "summary": summary, "phase_results": [
            {"phase": "Build", "duration": build, "process_status": "Success"},
            {"phase": "Test", "duration": test, "process_status": test_status}]}

    def test_the_night_of_11_22_gets_shards_its_slowest_job_would_have_finished(self):
        # run 44989179739: 155 s build + 264 s test unmutated; the default 1/2 job was given 96 and stopped at 73
        per = mn.per_from_baseline(155, 264)
        self.assertEqual(per, 34)
        self.assertLess(per, 73)
        # every mutant as slow as the slowest measured one still ends before the stop
        each = (mn.BUILD_SLOW * 155 + mn.TEST_SLOW * 264) / mn.JOBS
        self.assertLessEqual(155 + 264 + per * each, mn.STOP_S)
        self.assertGreater(155 + 264 + (per + 1) * each, mn.STOP_S)

    def test_a_quick_baseline_gets_larger_shards_and_a_slow_one_smaller(self):
        self.assertEqual([mn.per_from_baseline(b, t) for b, t in ((20, 30), (155, 264), (300, 900))], [299, 34, 10])

    def test_a_baseline_no_mutant_fits_behind_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "not one mutant"):
            mn.per_from_baseline(1800, 9000)
        for b, t in ((0, 10), (10, 0), (-1, 5)):
            with self.assertRaisesRegex(ValueError, "not a measurement"):
                mn.per_from_baseline(b, t)

    def test_the_baseline_is_read_from_the_outcomes_and_only_a_success_counts(self):
        mutant = {"scenario": {"Mutant": {}}, "summary": "CaughtMutant", "phase_results": [
            {"phase": "Build", "duration": 1.0, "process_status": "Success"},
            {"phase": "Test", "duration": 2.0, "process_status": "Failure"}]}
        self.assertEqual(mn.read_baseline(self.out([self.baseline(), mutant])), (155.2, 264.4))
        for bad in ([mutant], [self.baseline(summary="Failure")], [self.baseline(test_status="Timeout")],
                    [self.baseline(), self.baseline()]):
            with self.assertRaises(ValueError):
                mn.read_baseline(self.out(bad))

    def test_the_stop_and_the_jobs_are_the_ones_of_the_shard_step(self):
        text = Workflow.TEXT
        step = text[text.index("- name: Mutants (in-diff shard, a missed mutant fails)"):]
        step = step[:step.index("\n      - name: ", 10)]
        run = re.search(r"timeout --signal=INT --kill-after=10m (\d+)m \\\n\s+cargo mutants --in-diff [^\n]* -j (\d+) ", step)
        self.assertIsNotNone(run)
        self.assertEqual((int(run.group(1)) * 60, int(run.group(2))), (mn.STOP_S, mn.JOBS))

    def test_an_axis_with_mutants_and_no_shard_size_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "features"):
            mn.nightly_record("nightly", A, B, {"default": 10, "features": 10}, [], "", {"default": 34}, base_run=1)
        with self.assertRaisesRegex(ValueError, "extra"):
            mn.nightly_record("nightly", A, B, {"default": 10, "features": 0}, [], "", {"default": 34, "extra": 1}, base_run=1)
        rec, entries = mn.nightly_record("nightly", A, B, {"default": 10, "features": 0}, [], "", {"default": 34}, base_run=1)
        self.assertEqual((rec["state"], len(entries), rec["per"]["default"]), ("accepted", 1, 34))

    def test_each_axis_is_cut_by_its_own_shard_size(self):
        rec, entries = mn.nightly_record("nightly", A, B, {"default": 193, "features": 193}, [], "",
                                         {"default": 34, "features": 50}, base_run=1)
        self.assertEqual(rec["shards"], {"default": 6, "features": 4})
        self.assertEqual(sorted({(e["features"] == "", e["of"]) for e in entries}), [(False, 4), (True, 6)])
        rec, entries = mn.nightly_record("nightly", A, B, {"default": 273, "features": 10}, [], "",
                                         {"default": 34, "features": 50}, base_run=1)
        self.assertEqual((rec["state"], rec["shards"]["default"], entries), ("over-cap", 9, []))

    def test_a_range_set_aside_is_cut_in_the_weekly_run_by_the_size_it_was_measured_at(self):
        o = {"base": A, "head": B, "counts": {"default": 300, "features": 0}, "per": {"default": 30, "features": 100},
             "since": "2026-11-19T22:00:00Z"}
        _, entries = mn.weekly_record(D, [o], "2026-11-22T18:00:00Z", CORE_N, CORE_PER)
        self.assertEqual(len([e for e in entries if e["kind"] == "diff"]), 10)
        # a range recorded before the measure existed: the size of the 11-22 baseline
        del o["per"]
        _, entries = mn.weekly_record(D, [o], "2026-11-22T18:00:00Z", CORE_N, CORE_PER)
        self.assertEqual(len([e for e in entries if e["kind"] == "diff"]), -(-300 // mn.PER_UNMEASURED))
        self.assertEqual(mn.PER_UNMEASURED, mn.per_from_baseline(155, 264))

    def test_an_over_cap_night_is_read_back_with_its_shard_size(self):
        n = night(1, 19, A, B, {"default": 900, "features": 10})
        self.assertEqual(n["record"]["state"], "over-cap")
        n["record"]["open"] = []          # as if only the record's own range were known
        self.assertEqual(mn.open_ranges([n])[0]["per"], P)


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = mn.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def runs(self, runs: list[dict]) -> str:
        p = self.d / "runs.json"
        p.write_text(json.dumps(runs), encoding="utf-8")
        return str(p)

    def plan(self, runs: list[dict], default: int, features: int, mode: str = "nightly",
             per=("--per", "default=100", "--per", "features=100"), base_from: str | None = None,
             base_run: bool = True):
        base_from = base_from or mn.BASE_FROM[mode][0]
        code, out, err = self.run_cli("plan", "--mode", mode, "--base-from", base_from,
                                      *(("--base-run", "77") if base_from == "record" and base_run else ()),
                                      "--runs", self.runs(runs), "--base", B, "--head", C,
                                      "--count", f"default={default}", "--count", f"features={features}",
                                      *per,
                                      "--now", "2026-11-19T22:00:00Z",
                                      "--out-record", str(self.d / "r.json"), "--out-matrix", str(self.d / "m.json"))
        rec = json.loads((self.d / "r.json").read_text()) if (self.d / "r.json").exists() else None
        matrix = json.loads((self.d / "m.json").read_text()) if (self.d / "m.json").exists() else None
        return code, rec, matrix, err

    def test_per_prints_the_shard_size_of_a_baseline_and_fails_without_one(self):
        d = self.d / "mutants.out"
        d.mkdir()
        (d / "outcomes.json").write_text(json.dumps({"outcomes": [Per.baseline()]}), encoding="utf-8")
        code, out, err = self.run_cli("per", "--out", str(d))
        self.assertEqual((code, out), (0, "34\n"))
        self.assertIn("155 s build + 264 s test", err)
        (d / "outcomes.json").write_text(json.dumps({"outcomes": [Per.baseline(summary="Failure")]}), encoding="utf-8")
        code, out, err = self.run_cli("per", "--out", str(d))
        self.assertEqual((code, out), (mn.ERROR, ""))

    def test_a_plan_without_the_shard_size_of_an_axis_with_mutants_is_an_error(self):
        code, rec, matrix, err = self.plan([], 10, 10, per=("--per", "default=34"))
        self.assertEqual(code, mn.ERROR)
        self.assertIn("features", err)

    def test_the_plan_cuts_by_the_given_size_and_records_it(self):
        code, rec, matrix, err = self.plan([], 193, 193, per=("--per", "default=34", "--per", "features=50"))
        self.assertEqual((code, rec["per"], rec["shards"]), (0, {"default": 34, "features": 50}, {"default": 6, "features": 4}))
        self.assertEqual(len(matrix["include"]), 10)

    def test_base_prints_the_sha_and_fails_with_the_reason_when_no_record_gives_one(self):
        code, out, err = self.run_cli("base", "--runs", self.runs([night(1, 18, A, B)]))
        self.assertEqual((code, out), (0, f"{B}\n1\n"), "the sha, then the run whose record gave it")
        for runs in ([cancelled(1, 18, B)], [by_the_clock(2, 19, C, D)], []):
            code, out, err = self.run_cli("base", "--runs", self.runs(runs))
            self.assertEqual((code, out), (mn.NO_BASE, "\n"))
            self.assertIn("error: no nightly run in the window measured", err)
        self.assertNotIn(mn.NO_BASE, (0, mn.ERROR, mn.OVER_CAP))

    def test_anchor_prints_the_last_trusted_head_and_fails_without_one(self):
        code, out, err = self.run_cli("anchor", "--runs", self.runs([night(1, 12, A, B), by_the_clock(2, 19, C, D)]))
        self.assertEqual((code, out), (0, B + "\n"))
        code, out, err = self.run_cli("anchor", "--runs", self.runs([by_the_clock(2, 19, C, D)]))
        self.assertEqual((code, out), (mn.NO_BASE, "\n"))
        self.assertIn("error: no nightly run in the window measured", err)

    def test_a_nightly_plan_that_names_no_run_for_its_base_is_refused(self):
        code, rec, matrix, err = self.plan([], 10, 10, base_run=False)
        self.assertEqual((code, rec, matrix), (mn.ERROR, None, None))
        self.assertIn("names the run of that record", err)
        self.assertEqual(self.plan([], 10, 10)[1]["base_run"], 77)

    def test_a_nightly_plan_on_a_base_no_record_gave_is_refused_and_writes_no_record(self):
        for other in ("dispatch", "pull-request"):
            code, rec, matrix, err = self.plan([], 10, 10, base_from=other)
            self.assertEqual((code, rec, matrix), (mn.ERROR, None, None))
            self.assertIn("a nightly plan takes its base from record", err)
        self.assertEqual(self.plan([], 10, 10, mode="pr", base_from="record")[0], mn.ERROR)
        self.assertEqual(self.plan([], 10, 10)[1]["base_from"], "record")

    def test_a_dispatch_that_sets_a_range_aside_runs_no_job_and_records_the_range_as_open(self):
        earlier = night(1, 17, A, B, counts=BIG)
        for n in (1634, 10):   # (over a night's cap or under it: an aside plan runs neither)
            code, rec, matrix, err = self.plan([earlier], n, 10, mode="aside")
            self.assertEqual((code, rec["kind"], rec["state"], rec["base_from"], matrix), (0, "aside", "set-aside", "dispatch", {"include": []}))
            self.assertEqual([(o["base"], o["head"]) for o in rec["open"]], [(A, B), (B, C)])
            self.assertEqual(rec["open"][1], {"base": B, "head": C, "counts": {"default": n, "features": 10},
                                              "per": {"default": 100, "features": 100}, "since": "2026-11-19T22:00:00Z"})
            self.assertEqual(self.run_cli("core-jobs", "--record", str(self.d / "r.json"))[:2], (0, "0\n"))
        # the record is read back as the base of the next night and as an open range of the weekly plan
        run = {"id": 7, "head_sha": C, "created_at": "2026-11-19T22:00:00Z", "conclusion": "success", "jobs": [],
               "record": rec, "record_error": None}
        self.assertEqual(self.run_cli("base", "--runs", self.runs([earlier, run]))[:2], (0, f"{C}\n7\n"))
        code, out, err, wrec, wmatrix = self.weekly_cli(*CORE_ARGS, runs=[earlier, run])
        self.assertEqual(code, 0)
        self.assertEqual([(t["base"], t["head"]) for t in wrec["taken"]], [(A, B), (B, C)])
        self.assertEqual(len([e for e in wmatrix if e["kind"] == "diff" and e["base"] == B]), 2)
        code, rec, _, _ = self.plan([], 0, 0, mode="aside")
        self.assertEqual((code, rec["state"], rec["open"]), (0, "empty", []))

    def test_an_accepted_plan_writes_its_record_and_matrix(self):
        code, rec, matrix, _ = self.plan([], 388, 388)
        self.assertEqual((code, rec["state"], len(matrix["include"])), (0, "accepted", 8))

    def test_an_over_cap_night_exits_5_with_the_count_and_records_its_range_as_open(self):
        code, rec, matrix, err = self.plan([], 1634, 1634)
        self.assertEqual((code, rec["state"], matrix), (mn.OVER_CAP, "over-cap", {"include": []}))
        self.assertIn("takes 17 shards on an axis, over the cap of 8 (default 800 / features 800 mutants)", err)
        self.assertEqual(rec["open"], [{"base": B, "head": C, "counts": BIG, "per": P, "since": "2026-11-19T22:00:00Z"}])

    def test_a_nightly_record_carries_the_ranges_still_open_and_a_pr_record_none(self):
        earlier = night(1, 17, A, B, counts=BIG)
        _, rec, _, _ = self.plan([earlier], 10, 10)
        self.assertEqual([o["head"] for o in rec["open"]], [B])
        code, rec, _, _ = self.plan([earlier], 1634, 10, mode="pr")
        self.assertEqual((code, rec["open"]), (mn.OVER_CAP, []))

    def test_an_unreadable_record_fails_the_plan_the_weekly_plan_and_the_base(self):
        bad = cancelled(3, 18, C)
        bad["record_error"] = "its record artifact is not readable (x)"
        self.assertEqual(self.plan([bad], 10, 10)[0], mn.ERROR)
        self.assertEqual(self.run_cli("base", "--runs", self.runs([bad]))[0], mn.ERROR)
        code, _, err = self.run_cli("weekly", "--runs", self.runs([bad]), "--head", D, "--now", "2026-11-22T18:00:00Z", *CORE_ARGS,
                                    "--out-record", str(self.d / "r.json"), "--out-matrix", str(self.d / "m.json"))
        self.assertEqual(code, mn.ERROR)
        self.assertIn("run 3", err)

    def weekly_cli(self, *core: str, runs=()):
        code, out, err = self.run_cli("weekly", "--runs", self.runs(list(runs)), "--head", D, "--now", "2026-11-22T18:00:00Z", *core,
                                      "--out-record", str(self.d / "r.json"), "--out-matrix", str(self.d / "m.json"))
        rec = json.loads((self.d / "r.json").read_text()) if (self.d / "r.json").exists() else None
        matrix = json.loads((self.d / "m.json").read_text())["include"] if (self.d / "m.json").exists() else None
        return code, out, err, rec, matrix

    def test_the_weekly_plan_cuts_the_core_by_the_given_counts_and_sizes(self):
        code, out, _, rec, matrix = self.weekly_cli(*CORE_ARGS)
        self.assertEqual((code, len(matrix), rec["core"]["jobs"]), (0, 50, 50))
        self.assertIn("core shards {'default': 25, 'features': 25}", out)
        self.assertEqual({e["of"] for e in matrix}, {25})

    def test_a_weekly_core_over_the_cap_exits_5_with_the_count_and_no_core_job(self):
        n = 199 * mn.CORE_CAP + 1
        code, _, err, rec, matrix = self.weekly_cli("--count", f"default={n}", "--count", "features=10",
                                                    "--per", "default=199", "--per", "features=199")
        self.assertEqual((code, matrix, rec["state"]), (mn.OVER_CAP, [], "over-cap"))
        self.assertIn(f"takes {mn.CORE_CAP + 1} shards on an axis, over the cap of {mn.CORE_CAP}", err)

    def test_core_jobs_prints_the_core_shard_jobs_of_a_record_and_not_the_ranges_(self):
        over = night(2, 19, B, C, counts=BIG)
        self.run_cli("weekly", "--runs", self.runs([over]), "--head", D, "--now", "2026-11-22T18:00:00Z", *CORE_ARGS,
                     "--out-record", str(self.d / "r.json"), "--out-matrix", str(self.d / "m.json"))
        self.assertEqual(len(json.loads((self.d / "m.json").read_text())["include"]), 84)
        self.assertEqual(self.run_cli("core-jobs", "--record", str(self.d / "r.json"))[:2], (0, "50\n"))
        self.plan([], 150, 0)
        self.assertEqual(self.run_cli("core-jobs", "--record", str(self.d / "r.json"))[:2], (0, "0\n"))
        (self.d / "r.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.run_cli("core-jobs", "--record", str(self.d / "r.json"))[0], mn.ERROR)
        (self.d / "r.json").write_text('{"kind": "weekly"}', encoding="utf-8")
        self.assertEqual(self.run_cli("core-jobs", "--record", str(self.d / "r.json"))[0], mn.ERROR)

    def test_the_weekly_plan_says_which_range_it_deferred_and_a_plan_over_the_matrix_is_not_written(self):
        big = {"default": 6000, "features": 6000}
        runs = [night(1, 12, A, B, counts=big), night(2, 13, B, C, counts=big), night(3, 14, C, D, counts=big)]
        code, out, err, rec, matrix = self.weekly_cli(*CORE_ARGS, runs=runs)
        self.assertEqual((code, len(matrix)), (0, 50 + 120))
        self.assertEqual(out.count("deferred: range "), 2)
        self.assertIn(f"deferred: range {mn.range_id(B, C)} (120 jobs) does not fit this week's matrix of 256 entries", out)
        self.assertEqual(len(rec["open"]), 3)
        # a deferred range past the age limit turns the age job red, deferred or not
        self.assertEqual(self.run_cli("age", "--record", str(self.d / "r.json"), "--now", "2026-11-22T18:00:00Z")[0], 1)
        code, out, err = self.run_cli("age", "--record", str(self.d / "r.json"), "--now", "2026-11-20T18:00:00Z")
        self.assertEqual((code, out), (0, "compared: open ranges 3, older than 8 days 0\n"))
        # an aside plan over the limit: exit 2, nothing written
        (self.d / "r.json").unlink()
        (self.d / "m.json").unlink()
        code, rec, matrix, err = self.plan([], 12900, 12900, mode="aside")
        self.assertEqual((code, rec, matrix), (mn.ERROR, None, None))
        self.assertIn("more than the 256 a matrix holds", err)
        # and the last guard, for a plan no rule above stopped
        class Args:
            out_record, out_matrix = str(self.d / "x.json"), str(self.d / "y.json")
        with self.assertRaisesRegex(ValueError, "257 matrix entries.*not written"):
            mn.write_out(Args, {}, [{}] * 257)
        self.assertFalse((self.d / "x.json").exists() or (self.d / "y.json").exists())
        mn.write_out(Args, {}, [{}] * 256)
        self.assertEqual(len(json.loads((self.d / "y.json").read_text())["include"]), 256)

    def test_a_weekly_plan_without_the_core_counts_or_sizes_is_an_error(self):
        self.assertEqual(self.weekly_cli()[0], mn.ERROR)
        self.assertEqual(self.weekly_cli("--count", "default=10", "--count", "features=10")[0], mn.ERROR)
        self.assertIsNone(self.weekly_cli()[4], "no matrix is written")

    def test_the_weekly_plan_takes_the_open_range_and_the_age_job_fails_on_an_old_one(self):
        over = night(2, 10, B, C, counts=BIG)
        code, out, _ = self.run_cli("weekly", "--runs", self.runs([over]), "--head", D, "--now", "2026-11-22T18:00:00Z", *CORE_ARGS,
                                    "--out-record", str(self.d / "r.json"), "--out-matrix", str(self.d / "m.json"))
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads((self.d / "m.json").read_text())["include"]), 84)
        code, out, err = self.run_cli("age", "--record", str(self.d / "r.json"), "--now", "2026-11-22T18:00:00Z")
        self.assertEqual(code, 1)
        self.assertIn("open since 2026-11-10T22:00:00Z", err)
        self.assertIn("open ranges 1, older than 8 days 1", out)
        code, out, _ = self.run_cli("age", "--record", str(self.d / "r.json"), "--now", "2026-11-15T18:00:00Z")
        self.assertEqual((code, out.strip()), (0, "compared: open ranges 1, older than 8 days 0"))


class Listing(unittest.TestCase):
    def pages(self, counts: dict[str, list[int]]):
        """a fake of the runs listing: per event, the number of runs on each page"""
        calls = []

        def api(args):
            url = args[0]
            event = url.split("&event=")[1].split("&")[0]
            page = int(url.split("&page=")[1].split("&")[0])
            calls.append((event, page))
            sizes = counts.get(event, [])
            n = sizes[page - 1] if page <= len(sizes) else 0
            return {"workflow_runs": [{"id": f"{event}-{page}-{k}"} for k in range(n)]}
        return api, calls

    def test_each_event_is_listed_on_its_own_and_every_page_is_followed(self):
        api, calls = self.pages({"schedule": [100, 100, 7], "workflow_dispatch": [3]})
        runs = mn.list_runs("o/r", "2026-11-01T00:00:00Z", "quality-deep.yml", api)
        self.assertEqual(len(runs), 210)
        self.assertEqual(calls, [("schedule", 1), ("schedule", 2), ("schedule", 3), ("workflow_dispatch", 1)])

    def test_a_pull_request_run_is_never_listed(self):
        api, calls = self.pages({"pull_request": [100]})
        self.assertEqual(mn.list_runs("o/r", "2026-11-01T00:00:00Z", "quality-deep.yml", api), [])
        self.assertEqual({e for e, _ in calls}, {"schedule", "workflow_dispatch"})

    def test_a_window_that_does_not_fit_the_pages_is_an_error_not_a_cut(self):
        api, _ = self.pages({"schedule": [100] * (mn.MAX_PAGES + 1)})
        with self.assertRaises(RuntimeError) as e:
            mn.list_runs("o/r", "2026-11-01T00:00:00Z", "quality-deep.yml", api)
        self.assertIn("the window is not covered", str(e.exception))
        api, _ = self.pages({"schedule": [100] * (mn.MAX_PAGES - 1) + [99]})
        self.assertEqual(len(mn.list_runs("o/r", "2026-11-01T00:00:00Z", "quality-deep.yml", api)), 100 * mn.MAX_PAGES - 1)

    def test_the_listing_asks_for_the_window(self):
        seen = []

        def api(args):
            seen.append(args[0])
            return {"workflow_runs": []}
        mn.list_runs("o/r", "2026-11-01T00:00:00Z", "quality-deep.yml", api)
        self.assertTrue(all("created=%3E2026-11-01T00:00:00Z" in u and "/workflows/quality-deep.yml/runs?" in u for u in seen), seen)


class Workflow(unittest.TestCase):
    """the names the plan reads back are the ones quality-deep.yml writes"""
    TEXT = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "quality-deep.yml").read_text(encoding="utf-8")

    def test_the_shard_job_has_the_measured_step_under_the_name_the_plan_reads(self):
        self.assertEqual(self.TEXT.count(f"      - name: {mn.MEASURED_STEP}\n        if: always() && matrix.kind == 'diff'\n"), 1)
        self.assertEqual(self.TEXT.count("touch /tmp/measured"), 4)

    def test_the_core_shard_step_has_the_name_the_anchor_reads(self):
        named = [ln for ln in self.TEXT.splitlines() if ln.startswith(f"      - name: {mn.CORE_STEP}")]
        self.assertEqual(len(named), 1, named)
        at = self.TEXT.index(named[0])
        self.assertTrue(self.TEXT[at + len(named[0]):].startswith("\n        if: matrix.kind == 'core'\n"))
        label = mn.core_entries(D, {"default": 1, "features": 1})[0]["label"]
        self.assertIn("    name: Mutation testing (${{ matrix.label }})\n", self.TEXT)
        self.assertIn("(core shard ", f"Mutation testing ({label})", "what core_measured looks for in a job's name")
        # the step fails when the shard was stopped or the unmutated tree failed, and not for a missed mutant
        step = self.TEXT[at:self.TEXT.index("\n      - name: ", at + 10)]
        self.assertIn('if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then', step)
        self.assertIn("case $rc in 0|2|3) ;;", step)

    def test_the_job_name_is_the_plan_s_label_and_the_matrix_is_the_plan_s(self):
        self.assertIn("    name: Mutation testing (${{ matrix.label }})\n", self.TEXT)
        self.assertIn("      matrix: ${{ fromJSON(needs.mutants-plan.outputs.matrix) }}\n", self.TEXT)
        self.assertIn('--shard "${{ matrix.shard }}/${{ matrix.of }}"', self.TEXT)

    def test_the_record_is_uploaded_under_its_name_and_kept_longer_than_the_window(self):
        block = self.TEXT.split(f"          name: {mn.RECORD_ARTIFACT}\n          path: {mn.RECORD_FILE}\n", 1)
        self.assertEqual(len(block), 2)
        days = int(block[1].split("retention-days:", 1)[1].split()[0])
        self.assertGreater(days, mn.WINDOW_DAYS)
        self.assertGreater(mn.WINDOW_DAYS, mn.MAX_AGE_DAYS + 7, "a weekly run falls inside the window after the age limit")

    def test_the_two_schedules_are_the_ones_the_plan_tells_apart(self):
        self.assertEqual(self.TEXT.count("    - cron: '30 17 * * 1-5'"), 1)
        self.assertEqual(self.TEXT.count("    - cron: '30 17 * * 6'"), 1)
        self.assertIn("""elif [ "$SCHEDULE" = '30 17 * * 1-5' ]; then mode="nightly\"""", self.TEXT)
        self.assertNotIn("1-6", self.TEXT)

    def test_a_diff_base_dispatch_is_never_a_nightly_record_and_runs_no_weekly_job(self):
        self.assertIn("""          elif [ "$EVENT" = pull_request ] || [ -n "$DIFF_BASE" ]; then mode="pr\"""", self.TEXT)
        weekly_only = [ln for ln in self.TEXT.splitlines() if ln.startswith("    if:") and "github.event.schedule" in ln]
        self.assertEqual(len(weekly_only), 8)
        self.assertTrue(all("inputs.diff_base == ''" in ln for ln in weekly_only), weekly_only)

    def plan_step(self) -> str:
        step = self.TEXT[self.TEXT.index("      - name: Plan\n        id: plan\n"):]
        return step[:step.index("\n      - uses: actions/upload-artifact")]

    def test_a_night_without_a_base_from_a_record_stops_the_plan_and_no_base_is_taken_by_the_clock(self):
        step = self.plan_step()
        got = step.split('picked=$(python3 scripts/mutants_nightly.py base --runs /tmp/runs.json 2> /tmp/why) || {\n', 1)
        self.assertEqual(len(got), 2, "the base of a night is the script's, and its failure is handled where it is read")
        handler = got[1].split("\n              }\n", 1)[0].splitlines()
        self.assertEqual(len(handler), 2, handler)
        self.assertIn('echo "::error::no base for tonight\'s diff: $(sed \'s/^error: //\' /tmp/why)', handler[0])
        self.assertEqual(handler[1].strip(), "exit 1")
        after = got[1].split("\n            fi\n", 1)[0]
        self.assertIn('if ! git merge-base --is-ancestor "$base" HEAD; then\n', after)
        self.assertEqual(after.count("exit 1"), 2)   # (no base, and a base that is not an ancestor of HEAD)
        # the sha and the run that gave it are the two lines the script printed, and nothing else sets them
        self.assertIn('base=$(printf \'%s\\n\' "$picked" | sed -n 1p); base_run=$(printf \'%s\\n\' "$picked" | sed -n 2p)\n', after)
        self.assertEqual(len(re.findall(r"(?<![\w-])base_run=", step)), 2, "emptied before the branches, set from the script")
        self.assertIn("          over=false\n          base_run=\n", step)
        # no commit is picked by its date anywhere in the workflow, and `base` is assigned in three places only
        for clock in ("--before", "--since", "--until", "hours ago", "days ago", "falling back"):
            self.assertNotIn(clock, self.TEXT, clock)
        self.assertEqual(len(re.findall(r"(?<![\w-])base=", step)), 3)
        self.assertEqual(step.count("git rev-list"), 0)

    def test_the_plan_is_told_where_the_base_came_from_and_only_a_record_s_base_is_a_night_s(self):
        step = self.plan_step()
        self.assertEqual(step.count('plan --mode "$mode" --base-from "$from" ${base_run:+--base-run "$base_run"} '
                                    '--runs /tmp/runs.json --base "$base"'), 1)
        self.assertEqual(sorted(re.findall(r"(?<![\w-])from=([a-z-]+)", step)), ["dispatch", "pull-request", "record"])
        self.assertIn('why="the dispatch\'s diff_base"; from=dispatch\n', step)
        self.assertIn('why="the base of the pull request"; from=pull-request\n', step)
        self.assertIn("why=$(cat /tmp/why); from=record\n", step)

    def test_a_range_is_set_aside_only_from_the_last_tested_head_or_before_it(self):
        step = self.plan_step()
        got = step.split('              if [ "$mode" = aside ]; then\n', 1)
        self.assertEqual(len(got), 2)
        self.assertTrue(got[0].endswith('why="the dispatch\'s diff_base"; from=dispatch\n'), "in the diff_base branch")
        block = got[1].split("\n              fi\n", 1)[0]
        self.assertIn('anchor=$(python3 scripts/mutants_nightly.py anchor --runs /tmp/runs.json 2> /tmp/why) || {\n', block)
        self.assertIn('if ! git merge-base --is-ancestor "$base" "$anchor"; then\n', block)
        self.assertIn("diff_base $base is neither the last tested head $anchor", block)
        self.assertEqual(block.count("exit 1"), 2, "no anchor, and a diff_base after it: both stop the plan")
        self.assertEqual(step.count("mutants_nightly.py anchor"), 1)
        # (the refusals come before the plan is written: the record of an aside run is only ever of an accepted range)
        self.assertLess(step.index("mutants_nightly.py anchor"), step.index('plan --mode "$mode"'))

    def test_set_aside_is_a_dispatch_input_that_needs_diff_base_and_makes_the_aside_plan(self):
        inputs = self.TEXT.split("  pull_request:\n", 1)[0]
        self.assertIn("      set_aside:\n", inputs)
        self.assertIn("        default: false\n        type: boolean\n", inputs.split("      set_aside:\n", 1)[1])
        step = self.plan_step()
        self.assertIn("          SET_ASIDE: ${{ inputs.set_aside }}\n", step)
        self.assertIn('          if [ "$SET_ASIDE" = true ]; then\n'
                      '            if [ -z "$DIFF_BASE" ]; then echo "::error::set_aside needs diff_base (the range to record is diff_base..HEAD)"; exit 1; fi\n'
                      '            mode="aside"\n'
                      '          elif [ "$EVENT" = pull_request ]', step)
        # an aside run is a diff_base dispatch: its own group, no weekly job; and its plan has no shard job
        self.assertIn("needs.mutants-plan.outputs.jobs != '0'", self.TEXT)

    def test_the_ranges_a_weekly_run_takes_run_in_the_one_matrix_whose_jobs_at_a_time_are_bounded(self):
        # an open range has no cap on its shards (no mutant is dropped); what is bounded is how many of
        # its jobs run at once: they are entries of the same matrix as the core shards, under its max-parallel
        use = "      matrix: ${{ fromJSON(needs.mutants-plan.outputs.matrix) }}\n"
        self.assertEqual(self.TEXT.count(use), 1)
        self.assertEqual(self.TEXT.count("fromJSON(needs.mutants-plan.outputs.matrix)"), 1)
        job = self.TEXT[:self.TEXT.index(use)].rsplit("\n    strategy:\n", 1)
        self.assertEqual(len(job), 2)
        bound = (f"      max-parallel: ${{{{ needs.mutants-plan.outputs.mode == 'weekly' && {mn.WEEKLY_PARALLEL} || "
                 f"{mn.NIGHTLY_PARALLEL} }}}}\n")
        self.assertEqual(job[1].count(bound), 1, "in the strategy block that takes the plan's matrix")
        self.assertNotIn("\n  ", job[1].replace("\n      ", "\n"), "nothing but that strategy block in between")
        rec, entries = mn.weekly_record(D, [{"base": A, "head": B, "counts": BIG, "per": P, "since": "2026-11-19T22:00:00Z"}],
                                        "2026-11-22T18:00:00Z", CORE_N, CORE_PER)
        self.assertEqual({e["kind"] for e in entries}, {"core", "diff"}, "one matrix carries both")
        self.assertEqual(len([e for e in entries if e["kind"] == "diff"]), 34, "and the range is not cut to a cap")

    def test_the_nightly_and_the_weekly_schedule_share_one_group_that_queues(self):
        # both crons are in this one file, under one group for the whole workflow: a weekly run that is
        # still going makes the nightly wait (cancel-in-progress false: the running one is never cut).
        # Only a diff_base dispatch leaves that group, for the one group of such dispatches: the first
        # operand of the suffix is false on every other event, so the schedules' group is the workflow's
        # name alone
        on, rest = self.TEXT.split("\non:\n", 1)[1].split("\nconcurrency:\n", 1)
        self.assertEqual(on.count("    - cron: "), 2)
        block = rest.split("\n\n", 1)[0]
        self.assertEqual(block, "  group: ${{ github.workflow }}${{ github.event_name == 'workflow_dispatch' && "
                                "inputs.diff_base != '' && '-diff' || '' }}\n  cancel-in-progress: false")
        self.assertEqual(self.TEXT.count("\nconcurrency:"), 1)
        self.assertNotIn("    concurrency:", self.TEXT, "no job overrides the group")

    def test_the_feature_axes_of_the_plan_step_are_the_script_s(self):
        self.assertEqual(self.TEXT.count(f'for axis in "default=" "features={mn.AXES["features"]}"; do'), 2,
                         "the night's plan and the weekly core plan")

    def test_the_core_shards_are_the_plan_s_in_the_shard_step_the_baseline_and_the_ratchet(self):
        self.assertNotIn("/16", self.TEXT)
        self.assertNotIn("--expect-dirs 32", self.TEXT)
        # the files are named once; the count, the baseline and the shard step read that one list
        core = self.TEXT.split("\n  CORE_FILES: >-\n", 1)[1].split("\n\n", 1)[0]
        self.assertEqual(core.count("--file src/"), 8)
        self.assertEqual(self.TEXT.count("--file src/"), 9, "CORE_FILES and the example of the scope input")
        self.assertEqual(self.TEXT.count("$CORE_FILES"), 3)
        self.assertIn('cargo mutants --shard "${{ matrix.shard }}/${{ matrix.of }}" -j 2 ', self.TEXT)
        self.assertIn('$CORE_FILES --cargo-arg=--lib 2>&1 | tee /tmp/mutants.log', self.TEXT)
        self.assertIn('${features:+--features "$features"} $CORE_FILES --cargo-arg=--lib > "/tmp/baseline-$name.log" 2>&1', self.TEXT)
        self.assertIn('counts+=(--count "$name=$n" --per "$name=$per")', self.TEXT)
        self.assertIn('--head "$head" --now "$now" "${counts[@]}" \\\n', self.TEXT)
        self.assertIn("      core: ${{ steps.plan.outputs.core }}\n", self.TEXT)
        self.assertIn('echo "core=$(python3 scripts/mutants_nightly.py core-jobs --record mutants-record.json)"', self.TEXT)
        self.assertIn("          CORE_JOBS: ${{ needs.mutants-plan.outputs.core }}\n", self.TEXT)
        self.assertIn('python3 scripts/mutants_ratchet.py --out mutants-runs/* --expect-dirs "$CORE_JOBS"', self.TEXT)
        self.assertIn('if [ "${CORE_JOBS:-0}" -lt 1 ]; then', self.TEXT)
        self.assertIn("    needs: [mutants-plan, mutants]\n", self.TEXT)

    def test_the_jobs_of_a_plan_over_the_cap_still_run_and_a_plan_without_outputs_runs_none(self):
        # the plan job ends red at its cap step with its outputs written; `needs` alone would skip the shards
        job = self.TEXT.split("\n  mutants:\n", 1)[1].split("\n    name:", 1)[0]
        cond = [ln for ln in job.splitlines() if ln.startswith("    if:")]
        self.assertEqual(cond, ["    if: ${{ !cancelled() && needs.mutants-plan.outputs.jobs != '' && "
                                "needs.mutants-plan.outputs.jobs != '0' }}"])
        # `always()` would start the waiting shard jobs of a run that was cancelled
        self.assertNotIn("always() &&", "".join(ln for ln in self.TEXT.splitlines() if ln.startswith("    if:")))
        ratchet = self.TEXT.split("\n  mutants-ratchet:\n", 1)[1].split("\n    runs-on:", 1)[0]
        self.assertIn("\n    if: ${{ !cancelled() && inputs.diff_base == '' && ", ratchet)
        cap = self.TEXT.split("      - name: The plan is within the cap\n", 1)
        self.assertEqual(len(cap), 2)
        self.assertLess(self.TEXT.index('>> "$GITHUB_OUTPUT"'), self.TEXT.index("      - name: The plan is within the cap\n"),
                        "the outputs are written before the step that fails")

    def test_the_stop_of_both_shard_steps_is_the_one_the_shard_size_is_computed_for(self):
        self.assertEqual(self.TEXT.count(f"timeout --signal=INT --kill-after=10m {mn.STOP_S // 60}m \\\n"), 2)
        self.assertEqual(self.TEXT.count("    timeout-minutes: 300\n      max-parallel") , 0)
        self.assertIn(f"      max-parallel: ${{{{ needs.mutants-plan.outputs.mode == 'weekly' && {mn.WEEKLY_PARALLEL} || "
                      f"{mn.NIGHTLY_PARALLEL} }}}}\n", self.TEXT)
        self.assertEqual(self.TEXT.count("max-parallel:"), 1)


STUB_CARGO = """#!/usr/bin/env python3
import json, os, sys
a = sys.argv[1:]
assert a[0] == "mutants", a
core = os.environ["CORE_FILES"].split()
assert [x for x in a if x in core] == core, ("the core files are not all passed", a)
if "--list" in a:
    print("\\n".join(f"src/math.rs:{k}: replace + with -" for k in range(int(os.environ["STUB_N"]))))
    sys.exit(0)
n = int(os.environ["STUB_N"])
assert a[a.index("--shard") + 1] == f"0/{n}", a
assert "--cargo-arg=--lib" in a, a
out = os.path.join(a[a.index("--output") + 1], "mutants.out")
os.makedirs(out)
build, test = os.environ["STUB_BASELINE"].split(",")
phases = [{"phase": "Build", "duration": float(build), "process_status": "Success"},
          {"phase": "Test", "duration": float(test), "process_status": os.environ.get("STUB_TEST", "Success")}]
json.dump({"outcomes": [{"scenario": "Baseline", "summary": "Success", "phase_results": phases}]},
          open(os.path.join(out, "outcomes.json"), "w"))
print(f"ok       Unmutated baseline in {build}s build + {test}s test")
"""


@unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("git"), "needs bash and git")
class WeeklyPlanStep(unittest.TestCase):
    """the weekly branch of the plan job's step, run as written, with cargo, gh and timeout replaced: the
    schedule runs it once a week, so a mistake in it would otherwise first show on a Saturday night"""
    ROOT = Path(__file__).resolve().parent.parent

    def step(self) -> str:
        text = (self.ROOT / ".github" / "workflows" / "quality-deep.yml").read_text(encoding="utf-8")
        body = text.split("        run: |\n          now=$(date -u", 1)[1].split("\n      - uses: actions/upload-artifact@v6\n", 1)[0]
        return "\n".join(ln[10:] for ln in ("          now=$(date -u" + body).split("\n"))

    def run_step(self, n: int, baseline: str, **env: str):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            work, tmp, stubs = d / "work", d / "tmp", d / "bin"
            for x in (work, tmp, stubs):
                x.mkdir()
            (stubs / "cargo").write_text(STUB_CARGO, encoding="utf-8")
            (stubs / "gh").write_text("#!/bin/sh\necho '{\"workflow_runs\": []}'\n", encoding="utf-8")
            (stubs / "timeout").write_text("#!/bin/sh\nshift 3\nexec \"$@\"\n", encoding="utf-8")
            for f in stubs.iterdir():
                f.chmod(0o755)
            (work / "scripts").symlink_to(self.ROOT / "scripts")
            git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-C", str(work)]
            subprocess.run([*git, "init", "-q"], check=True)
            subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "c"], check=True)
            (d / "step.sh").write_text(self.step().replace("/tmp/", f"{tmp}/"), encoding="utf-8")
            core = "--file src/solver.rs --file src/math.rs"
            got = subprocess.run(
                ["bash", "-e", str(d / "step.sh")], cwd=work, capture_output=True, text=True,
                env={**os.environ, "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}", "EVENT": "schedule",
                     "SCHEDULE": "30 17 * * 6", "BASE_REF": "", "DIFF_BASE": "", "CORE_FILES": core,
                     "GITHUB_REPOSITORY": "o/r", "GITHUB_RUN_ID": "1", "GITHUB_OUTPUT": str(d / "out"),
                     "GITHUB_STEP_SUMMARY": str(d / "summary"), "STUB_N": str(n), "STUB_BASELINE": baseline, **env})
            out = dict(ln.split("=", 1) for ln in (d / "out").read_text().splitlines()) if (d / "out").exists() else {}
            rec = json.loads((work / "mutants-record.json").read_text()) if (work / "mutants-record.json").exists() else None
            summary = (d / "summary").read_text() if (d / "summary").exists() else ""
            return got, out, rec, summary

    def test_the_core_is_counted_timed_and_cut_on_both_axes(self):
        got, out, rec, summary = self.run_step(4864, "85,13")
        self.assertEqual(got.returncode, 0, got.stderr[-2000:])
        self.assertEqual({k: out[k] for k in ("mode", "over", "jobs", "core")},
                         {"mode": "weekly", "over": "false", "jobs": "50", "core": "50"})
        matrix = json.loads(out["matrix"])["include"]
        self.assertEqual(sorted({(e["features"], e["of"]) for e in matrix}), [("", 25), (mn.AXES["features"], 25)])
        self.assertEqual(rec["core"], {"counts": CORE_N, "per": CORE_PER, "shards": {"default": 25, "features": 25}, "jobs": 50})
        self.assertIn("default axis: 4864 core mutants, unmutated baseline 85 s build + 13 s test, 199 mutants a shard", summary)
        self.assertIn("features axis: 4864 core mutants", summary)

    def test_a_core_over_the_cap_is_marked_and_has_no_core_job(self):
        got, out, rec, _ = self.run_step(199 * mn.CORE_CAP + 1, "85,13")
        self.assertEqual(got.returncode, 0, got.stderr[-2000:])
        self.assertEqual((out["over"], out["core"], out["jobs"], rec["state"]), ("true", "0", "0", "over-cap"))
        self.assertIn(f"::error::the core plan takes {mn.CORE_CAP + 1} shards on an axis", got.stdout)

    def test_no_listed_core_mutant_or_a_failed_baseline_stops_the_plan(self):
        got, out, rec, _ = self.run_step(0, "85,13")
        self.assertNotEqual(got.returncode, 0)
        self.assertIn("lists no mutant in the core files", got.stdout)
        self.assertEqual((out, rec), ({}, None))
        got, out, rec, _ = self.run_step(100, "85,13", STUB_TEST="Failure")
        self.assertNotEqual(got.returncode, 0)
        self.assertIn("no unmutated baseline to size the core shards from", got.stdout)
        self.assertEqual((out, rec), ({}, None))


if __name__ == "__main__":
    unittest.main()
