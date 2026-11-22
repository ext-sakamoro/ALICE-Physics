#!/usr/bin/env python3
"""Tests of scripts/mutants_nightly.py: the shard count, the base of a night's diff, the ranges set aside for the
weekly run and the age rule.

Author: Moroya Sakamoto
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutants_nightly as mn  # noqa: E402

A, B, C, D = ("a" * 40, "b" * 40, "c" * 40, "d" * 40)
OK = [{"name": "Set up job", "conclusion": "success"}, {"name": mn.MEASURED_STEP, "conclusion": "success"}]
KILLED = [{"name": "Set up job", "conclusion": "success"}, {"name": mn.MEASURED_STEP, "conclusion": "failure"}]


def jobs_of(entries: list[dict], steps=None) -> list[dict]:
    return [{"name": f"Mutation testing ({e['label']})", "conclusion": "success", "steps": steps or OK} for e in entries]


def night(run_id: int, day: int, base: str, head: str, counts=None, steps=None, drop: int = 0, carried=None) -> dict:
    """a nightly run of 2026-11-<day> with its record and the jobs its plan asked for (`drop` of them missing)"""
    counts = counts or {"default": 150, "features": 150}
    rec, entries = mn.nightly_record("nightly", base, head, counts, carried or [], "test")
    if rec["state"] == "over-cap":
        rec["open"] = rec["open"] + [{"base": base, "head": head, "counts": counts, "since": f"2026-11-{day:02d}T22:00:00Z"}]
    return {"id": run_id, "head_sha": head, "created_at": f"2026-11-{day:02d}T22:00:00Z", "conclusion": "success",
            "jobs": jobs_of(entries, steps)[drop:], "record": rec, "record_error": None}


def cancelled(run_id: int, day: int, head: str) -> dict:
    return {"id": run_id, "head_sha": head, "created_at": f"2026-11-{day:02d}T22:00:00Z", "conclusion": "cancelled",
            "jobs": [], "record": None, "record_error": None}


def weekly(run_id: int, day: int, head: str, ranges: list[dict], steps=None, skip: bool = False) -> dict:
    rec, entries = mn.weekly_record(head, ranges, f"2026-11-{day:02d}T18:00:00Z")
    jobs = jobs_of([e for e in entries if e["kind"] == "core" or not skip], steps)
    return {"id": run_id, "head_sha": head, "created_at": f"2026-11-{day:02d}T18:00:00Z", "conclusion": "success",
            "jobs": jobs, "record": rec, "record_error": None}


BIG = {"default": 1634, "features": 1634}


class Shards(unittest.TestCase):
    def test_a_shard_holds_at_most_per_mutants(self):
        self.assertEqual([mn.shards_for(n) for n in (0, 1, 100, 101, 388, 800, 801, 1634)], [0, 1, 1, 2, 4, 8, 9, 17])

    def test_the_cap_is_per_axis_and_an_over_cap_plan_has_no_job(self):
        rec, entries = mn.nightly_record("nightly", A, B, {"default": 800, "features": 105}, [], "")
        self.assertEqual((rec["state"], rec["shards"], len(entries), rec["jobs"]), ("accepted", {"default": 8, "features": 2}, 10, 10))
        rec, entries = mn.nightly_record("nightly", A, B, {"default": 105, "features": 801}, [], "")
        self.assertEqual((rec["state"], entries, rec["jobs"]), ("over-cap", [], 0))

    def test_an_empty_plan_is_empty_not_accepted(self):
        rec, entries = mn.nightly_record("nightly", A, B, {"default": 0, "features": 0}, [], "")
        self.assertEqual((rec["state"], entries), ("empty", []))

    def test_every_entry_names_its_range_shard_and_axis(self):
        _, entries = mn.nightly_record("nightly", A, B, {"default": 101, "features": 1}, [], "")
        self.assertEqual([(e["shard"], e["of"], e["features"]) for e in entries],
                         [(0, 2, ""), (1, 2, ""), (0, 1, "parallel,gpu-solver-bridge,simd")])
        self.assertEqual(entries[0]["label"], f"in-diff {'a' * 12}..{'b' * 12} shard 0/2, default")
        self.assertTrue(all(e["base"] == A and e["head"] == B and e["kind"] == "diff" for e in entries))
        self.assertEqual(entries[2]["artifact"], f"mutants-diff-{'a' * 12}-{'b' * 12}-0-features")
        self.assertEqual(len({e["artifact"] for e in entries}), 3)

    def test_counts_must_name_both_axes(self):
        for counts in ({"default": 1}, {"default": 1, "features": 1, "other": 1}):
            with self.assertRaises(ValueError):
                mn.nightly_record("nightly", A, B, counts, [], "")


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
        self.assertIn("1 unmeasured, 2 run(s) read", why)
        self.assertIsNone(mn.pick_base([])[0])

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
        rec, entries = mn.weekly_record(D, ranges, "2026-11-22T18:00:00Z")
        diff = [e for e in entries if e["kind"] == "diff"]
        self.assertEqual((len(entries) - len(diff), len(diff)), (32, 34))
        self.assertTrue(all(e["base"] == B and e["head"] == C for e in diff))
        self.assertEqual(rec["taken"], [{"base": B, "head": C, "jobs": 34}])
        self.assertEqual(rec["stale"], [])
        # (the ratchet job reads the 32 core artifacts by this name)
        self.assertEqual(sorted(e["artifact"] for e in entries if e["kind"] == "core"),
                         sorted(f"mutants-out-{k}-{x}" for k in range(16) for x in ("default", "parallel")))
        self.assertEqual(len({e["artifact"] for e in entries}), 66)

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
                             "2026-11-22T18:00:00Z")


class Age(unittest.TestCase):
    RANGE = {"base": B, "head": C, "counts": BIG, "since": "2026-11-10T22:00:00Z"}

    def test_a_range_is_stale_after_eight_days_not_at_eight(self):
        self.assertEqual(mn.stale([self.RANGE], "2026-11-18T22:00:00Z"), [])
        self.assertEqual(mn.stale([self.RANGE], "2026-11-18T22:00:01Z"), [self.RANGE])

    def test_the_weekly_record_names_the_stale_ranges(self):
        rec, _ = mn.weekly_record(D, [self.RANGE], "2026-11-22T18:00:00Z")
        self.assertEqual(rec["stale"], [f"{'b' * 12}..{'c' * 12}"])


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

    def plan(self, runs: list[dict], default: int, features: int, mode: str = "nightly"):
        code, out, err = self.run_cli("plan", "--mode", mode, "--runs", self.runs(runs), "--base", B, "--head", C,
                                      "--count", f"default={default}", "--count", f"features={features}",
                                      "--now", "2026-11-19T22:00:00Z",
                                      "--out-record", str(self.d / "r.json"), "--out-matrix", str(self.d / "m.json"))
        rec = json.loads((self.d / "r.json").read_text()) if (self.d / "r.json").exists() else None
        matrix = json.loads((self.d / "m.json").read_text()) if (self.d / "m.json").exists() else None
        return code, rec, matrix, err

    def test_base_prints_the_sha_or_an_empty_line(self):
        code, out, err = self.run_cli("base", "--runs", self.runs([night(1, 18, A, B)]))
        self.assertEqual((code, out), (0, B + "\n"))
        code, out, err = self.run_cli("base", "--runs", self.runs([cancelled(1, 18, B)]))
        self.assertEqual((code, out), (0, "\n"))
        self.assertIn("no nightly run in the window measured", err)

    def test_an_accepted_plan_writes_its_record_and_matrix(self):
        code, rec, matrix, _ = self.plan([], 388, 388)
        self.assertEqual((code, rec["state"], len(matrix["include"])), (0, "accepted", 8))

    def test_an_over_cap_night_exits_5_with_the_count_and_records_its_range_as_open(self):
        code, rec, matrix, err = self.plan([], 1634, 1634)
        self.assertEqual((code, rec["state"], matrix), (mn.OVER_CAP, "over-cap", {"include": []}))
        self.assertIn("takes 17 shards on an axis, over the cap of 8 (800 mutants)", err)
        self.assertEqual(rec["open"], [{"base": B, "head": C, "counts": BIG, "since": "2026-11-19T22:00:00Z"}])

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
        code, _, err = self.run_cli("weekly", "--runs", self.runs([bad]), "--head", D, "--now", "2026-11-22T18:00:00Z",
                                    "--out-record", str(self.d / "r.json"), "--out-matrix", str(self.d / "m.json"))
        self.assertEqual(code, mn.ERROR)
        self.assertIn("run 3", err)

    def test_the_weekly_plan_takes_the_open_range_and_the_age_job_fails_on_an_old_one(self):
        over = night(2, 10, B, C, counts=BIG)
        code, out, _ = self.run_cli("weekly", "--runs", self.runs([over]), "--head", D, "--now", "2026-11-22T18:00:00Z",
                                    "--out-record", str(self.d / "r.json"), "--out-matrix", str(self.d / "m.json"))
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads((self.d / "m.json").read_text())["include"]), 66)
        code, out, err = self.run_cli("age", "--record", str(self.d / "r.json"), "--now", "2026-11-22T18:00:00Z")
        self.assertEqual(code, 1)
        self.assertIn("open since 2026-11-10T22:00:00Z", err)
        self.assertIn("open ranges 1, older than 8 days 1", out)
        code, out, _ = self.run_cli("age", "--record", str(self.d / "r.json"), "--now", "2026-11-15T18:00:00Z")
        self.assertEqual((code, out.strip()), (0, "compared: open ranges 1, older than 8 days 0"))


class Workflow(unittest.TestCase):
    """the names the plan reads back are the ones quality-deep.yml writes"""
    TEXT = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "quality-deep.yml").read_text(encoding="utf-8")

    def test_the_shard_job_has_the_measured_step_under_the_name_the_plan_reads(self):
        self.assertEqual(self.TEXT.count(f"      - name: {mn.MEASURED_STEP}\n        if: always() && matrix.kind == 'diff'\n"), 1)
        self.assertEqual(self.TEXT.count("touch /tmp/measured"), 4)

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
        self.assertIn("""if [ "$EVENT" = pull_request ] || [ -n "$DIFF_BASE" ]; then mode="pr\"""", self.TEXT)
        weekly_only = [ln for ln in self.TEXT.splitlines() if ln.startswith("    if:") and "github.event.schedule" in ln]
        self.assertEqual(len(weekly_only), 8)
        self.assertTrue(all("inputs.diff_base == ''" in ln for ln in weekly_only), weekly_only)

    def test_the_nightly_and_the_weekly_schedule_share_one_group_that_queues(self):
        # both crons are in this one file, under one group for the whole workflow: a weekly run that is
        # still going makes the nightly wait (cancel-in-progress false: the running one is never cut)
        on, rest = self.TEXT.split("\non:\n", 1)[1].split("\nconcurrency:\n", 1)
        self.assertEqual(on.count("    - cron: "), 2)
        block = rest.split("\n\n", 1)[0]
        self.assertEqual(block, "  group: ${{ github.workflow }}\n  cancel-in-progress: false")
        self.assertEqual(self.TEXT.count("\nconcurrency:"), 1)
        self.assertNotIn("    concurrency:", self.TEXT, "no job overrides the group")

    def test_the_feature_axes_of_the_plan_step_are_the_script_s(self):
        self.assertIn(f'for axis in "default=" "features={mn.AXES["features"]}"; do', self.TEXT)


if __name__ == "__main__":
    unittest.main()
