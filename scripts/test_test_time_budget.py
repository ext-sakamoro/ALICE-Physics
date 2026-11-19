#!/usr/bin/env python3
"""Tests of scripts/test_time_budget.py: the projection refuses what would not fit."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_time_budget as t  # noqa: E402

CAL = t.CALIBRATION
FLUID = "tests/analytic_fluid_sdf_body.rs"


def runs(*totals, build=2.0):
    return [{"id": str(100 + k), "sha": f"{k:040x}", "total": float(x), "build": build}
            for k, x in enumerate(totals)]


def text(mac=(3300, 3360, 3330), ubuntu=(2400, 2380, 2390), build=2.0, fluid=None):
    bins = {CAL: [50.0, 42.0], "tests/small.rs": [0.5, 0.5]}
    if fluid:
        bins[FLUID] = fluid
    return t.format_ledger([
        {"name": "Test (macos-latest)", "limit": 120, "runs": runs(*mac, build=build), "bins": bins},
        {"name": "Test (ubuntu-latest)", "limit": 120, "runs": runs(*ubuntu, build=build),
         "bins": {CAL: [34.0, 34.0]}},
    ])


def ledger(**k):
    return t.parse_ledger(text(**k))


# the slowest per-push job as it was before the fluid file existed: 56 min at most, of 120
LEDGER = text()


class Projection(unittest.TestCase):
    def test_the_900_frame_fluid_file_is_refused(self):
        # 600 s here, the runner 2.0 times slower (46 / 23), two passes: +40 min on 56
        report, refusals = t.project(ledger(), {CAL: 23.0, FLUID: 600.0}, [FLUID])
        self.assertEqual(refusals, [], "96 min of 120 is inside the 15-minute margin")
        self.assertIn("= 96 min of 120", report[0])
        _, refusals = t.project(ledger(mac=(6300, 6360, 6330)), {CAL: 23.0, FLUID: 600.0}, [FLUID])
        self.assertEqual(len(refusals), 1)
        self.assertIn("Test (macos-latest)", refusals[0])
        self.assertIn(FLUID, refusals[0])

    def test_a_file_already_in_the_ledger_counts_only_its_change(self):
        # in the ledger at 600 s a pass on the runner; here 80 s -> 160 s a pass there
        had = ledger(mac=(6300, 6360, 6330), fluid=[600.0, 600.0])
        report, refusals = t.project(had, {CAL: 23.0, FLUID: 80.0}, [FLUID])
        self.assertEqual(refusals, [])
        self.assertIn("-14.7 min", report[0])
        _, refusals = t.project(had, {CAL: 23.0, FLUID: 600.0}, [FLUID])
        self.assertEqual(len(refusals), 1, "the same file twice as slow as the ledger has it")

    def test_nothing_compared_is_an_error(self):
        with self.assertRaises(t.BudgetError):
            t.parse_ledger("# only a comment\n")
        with self.assertRaises(t.BudgetError):
            t.project(ledger(), {FLUID: 600.0}, [FLUID])  # no calibration time
        with self.assertRaises(t.BudgetError):
            t.project(ledger(), {CAL: 0.0, FLUID: 600.0}, [FLUID])
        with self.assertRaises(t.BudgetError):
            t.project(ledger(), {CAL: 23.0}, [FLUID])  # the changed binary not timed
        no_cal = t.parse_ledger(t.format_ledger([
            {"name": "Test (x)", "limit": 120, "runs": runs(600, 600, 600), "bins": {"tests/a.rs": [3.0]}}]))
        with self.assertRaises(t.BudgetError):
            t.project(no_cal, {CAL: 23.0, FLUID: 1.0}, [FLUID])
        with self.assertRaises(t.BudgetError):  # an added binary that is not a changed one
            t.project(ledger(), {CAL: 23.0, FLUID: 1.0}, [FLUID], added=["tests/other.rs"])

    def test_the_margin_is_what_refuses(self):
        tight = ledger(mac=(6300, 6360, 6330))
        _, refusals = t.project(tight, {CAL: 23.0, FLUID: 600.0}, [FLUID], margin=0)
        self.assertEqual(len(refusals), 1, "146 min is over 120 whatever the margin")
        _, refusals = t.project(ledger(), {CAL: 23.0, FLUID: 600.0}, [FLUID], margin=30)
        self.assertEqual(len(refusals), 1, "96 min is over 120 - 30")

    def test_the_baseline_is_the_slowest_run_not_the_newest_or_the_fastest(self):
        # one commit on a fast and a slow runner: 60, 82 and 61 min; the tests add 20 min
        spread = ledger(mac=(3600, 4920, 3660))
        report, refusals = t.project(spread, {CAL: 23.0, FLUID: 300.0}, [FLUID])
        # 82 + 20 = 102 of 120; the margin is the spread, 22 min: budget 98
        self.assertIn("at most 82 min on main (3 runs, spread 22 min)", report[0])
        self.assertIn("(margin 22, budget 98)", report[0])
        self.assertEqual(len(refusals), 1, "from the fastest run (60 + 20 = 80) this would pass")
        self.assertIn("projected 102 min is over 98 min", refusals[0])
        # three runs close together at the slow one's time: 102 is inside 120 - 15
        _, refusals = t.project(ledger(mac=(4900, 4920, 4910)), {CAL: 23.0, FLUID: 300.0}, [FLUID])
        self.assertEqual(refusals, [])

    def test_only_the_newest_five_runs_count(self):
        old = ledger(mac=(9000, 3300, 3360, 3330, 3340, 3350))
        self.assertEqual(len(old["Test (macos-latest)"]["runs"]), 5, "the ledger keeps five")
        report, _ = t.project(old, {CAL: 23.0, FLUID: 600.0}, [FLUID])
        self.assertIn("at most 56 min on main (5 runs", report[0])
        # a ledger read with a sixth, older run (written by hand): still the newest five
        old["Test (macos-latest)"]["runs"].insert(0, runs(9000)[0])
        report, _ = t.project(old, {CAL: 23.0, FLUID: 600.0}, [FLUID])
        self.assertIn("at most 56 min on main (5 runs", report[0])

    def test_fewer_than_three_runs_of_a_job_is_not_projected(self):
        for few in ((3300,), (3300, 3360)):
            with self.assertRaisesRegex(t.BudgetError, "at least 3"):
                t.project(ledger(mac=few), {CAL: 23.0, FLUID: 1.0}, [FLUID])
        with self.assertRaisesRegex(t.BudgetError, "at least 3"):
            t.project(ledger(ubuntu=(2400,)), {CAL: 23.0, FLUID: 1.0}, [FLUID])

    def test_every_added_binary_costs_its_build_in_every_pass(self):
        # four added files that run in no time, where one test binary takes 240 s to build
        # for a pass: 4 x 2 passes x 240 s = 32 min on 75
        new = [f"tests/new_{k}.rs" for k in range(4)]
        local = {CAL: 23.0, **{n: 0.0 for n in new}}
        slow = ledger(mac=(4500, 4500, 4500), build=240.0)
        report, refusals = t.project(slow, local, new, added=new)
        self.assertIn("32.0 min of it the build of 4 added", report[0])
        self.assertEqual(len(refusals), 1, "75 + 32 = 107 is over 120 - 15; without the build, 75")
        self.assertIn("projected 107 min is over 105 min", refusals[0])
        # changed, not added: no build is counted
        _, refusals = t.project(slow, local, new, added=[])
        self.assertEqual(refusals, [])


CI = """
    strategy:
      matrix:
        os: ${{ fromJSON(github.event_name == 'schedule' && '["macos-15-intel"]' || '["macos-latest", "ubuntu-latest"]') }}
"""


class Stale(unittest.TestCase):
    def test_a_ledger_of_other_jobs_than_the_workflows_is_refused(self):
        ledger = t.parse_ledger(LEDGER)  # macos-latest, ubuntu-latest
        problems = t.staleness(ledger, CI, 0)
        self.assertEqual(len(problems), 1)  # ci.yml also has macos-15-intel
        self.assertIn("Test (macos-15-intel)", problems[0])
        whole = LEDGER + "job\tTest (macos-15-intel)\t360\n"
        self.assertEqual(t.staleness(t.parse_ledger(whole), CI, 0), [])
        # a job split in two leaves the ledger describing the old one
        split = CI.replace('"macos-latest"', '"macos-latest-default", "macos-latest-parallel"')
        problems = t.staleness(t.parse_ledger(whole), split, 0)
        self.assertEqual(len(problems), 1)
        self.assertIn("only in the ledger: Test (macos-latest)", problems[0])
        with self.assertRaises(t.BudgetError):
            t.staleness(t.parse_ledger(whole), "jobs:\n  test:\n    runs-on: x\n", 0)

    def test_a_ledger_too_far_behind_is_refused(self):
        whole = t.parse_ledger(LEDGER + "job\tTest (macos-15-intel)\t360\n")
        self.assertEqual(t.staleness(whole, CI, t.MAX_BEHIND), [])
        problems = t.staleness(whole, CI, t.MAX_BEHIND + 1)
        self.assertEqual(len(problems), 1)
        self.assertIn("commits behind", problems[0])

    def test_the_ledgers_commit_is_read_from_its_sha_line(self):
        self.assertEqual(t.ledger_sha(LEDGER), f"{2:040x}", "the newest run's commit")
        with self.assertRaises(t.BudgetError):
            t.ledger_sha("job\tTest (x)\t120\n")


class LocalTimes(unittest.TestCase):
    OUT = "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in {}s\n"

    def test_one_result_line_on_stdout_is_the_binarys_time(self):
        asked = []

        def run(name):
            asked.append(name)
            return 0, "running 2 tests\n" + self.OUT.format("12.50")

        self.assertEqual(t.time_locally(["tests/a_b.rs"], run), {"tests/a_b.rs": 12.5})
        self.assertEqual(asked, ["a_b"])

    def test_a_failed_run_or_no_time_or_two_times_is_an_error(self):
        for code, out in [(101, self.OUT.format("1.00")), (0, "running 0 tests\n"),
                          (0, self.OUT.format("1.00") + self.OUT.format("2.00"))]:
            with self.assertRaises(t.BudgetError):
                t.time_locally(["tests/a.rs"], lambda _name, c=code, o=out: (c, o))


class Logs(unittest.TestCase):
    def test_a_jobs_log_gives_each_pass_of_each_binary(self):
        log = "\n".join([
            "2026-11-18T05:54:00Z \x1b[1m\x1b[92m     Running\x1b[0m tests/a.rs (target/x/a-1)",
            "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.50s",
            "     Running unittests src/lib.rs (target/x/lib-1)",
            "test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 100.00s",
            "     Running tests\\a.rs (target\\x\\a-2)",
            "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 13.00s",
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.00s",
        ])
        self.assertEqual(t.binary_times(log), {"tests/a.rs": [12.5, 13.0], "unit src/lib.rs": [100.0]})

    def test_the_ledger_round_trips_and_drops_the_quick_binaries(self):
        ledger = t.parse_ledger(LEDGER)
        self.assertEqual(ledger["Test (macos-latest)"]["limit"], 120.0)
        self.assertEqual(ledger["Test (macos-latest)"]["bins"][CAL], [50.0, 42.0])
        self.assertNotIn("tests/small.rs", ledger["Test (macos-latest)"]["bins"])
        with self.assertRaises(t.BudgetError):
            t.parse_ledger("bin\tTest (x)\ttests/a.rs\t3.00\n")
        with self.assertRaises(t.BudgetError):
            t.parse_ledger("job\tTest (x)\t120\t600\n")  # (the one-run form of the ledger)
        with self.assertRaises(t.BudgetError):
            t.parse_ledger("run\tTest (x)\t1\tabc\t600\t2.0\n")  # a run before its job
        self.assertEqual([r["total"] for r in ledger["Test (macos-latest)"]["runs"]], [3300.0, 3360.0, 3330.0])
        self.assertEqual(ledger["Test (macos-latest)"]["runs"][1]["build"], 2.0)

    def test_a_job_the_new_run_did_not_have_keeps_its_rows(self):
        known = ledger()
        new = [{"name": "Test (ubuntu-latest)", "limit": 120, "runs": runs(1, 2, 3, 4), "bins": {CAL: [1.0]}}]
        merged = t.parse_ledger(t.format_ledger(t.merge_jobs(known, new)))
        self.assertEqual(len(merged["Test (ubuntu-latest)"]["runs"]), 4, "the run's own job as read")
        self.assertEqual(merged["Test (macos-latest)"], known["Test (macos-latest)"], "kept whole")

    def test_the_build_time_per_binary_is_what_the_full_passes_took_beyond_their_tests(self):
        ok = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in {}s"
        log = "\n".join([
            "2026-11-19T05:00:00.0000000Z ##[group]Run python scripts/x.py",
            "2026-11-19T05:01:00.0000000Z ##[group]Run cargo test --profile ci-test",
            "2026-11-19T05:01:01.0000000Z      Running tests/a.rs (target/x/a-1)",
            "2026-11-19T05:03:00.0000000Z " + ok.format("30.00"),
            "2026-11-19T05:03:00.0000000Z      Running tests/b.rs (target/x/b-1)",
            "2026-11-19T05:04:00.0000000Z " + ok.format("50.00"),
            # a step that runs one binary by name is not a full pass
            "2026-11-19T05:05:00.0000000Z ##[group]Run scripts/run_tests.sh a=1 -- cargo test --test a",
            "2026-11-19T05:05:01.0000000Z      Running tests/a.rs (target/x/a-1)",
            "2026-11-19T05:05:30.0000000Z " + ok.format("29.00"),
            "2026-11-19T05:06:00.0000000Z ##[group]Run cargo test --profile ci-test --features \"parallel\"",
            "2026-11-19T05:06:01.0000000Z      Running tests/a.rs (target/x/a-2)",
            "2026-11-19T05:08:00.0000000Z " + ok.format("40.00"),
            "2026-11-19T05:09:00.0000000Z Post job cleanup.",
        ])
        # the two full passes took 240 s + 180 s, their three test runs 30 + 50 + 40 s
        self.assertEqual(t.build_seconds(log), (420.0 - 120.0) / 3)
        with self.assertRaisesRegex(t.BudgetError, "no `cargo test --profile ci-test` step"):
            t.build_seconds("2026-11-19T05:00:00.0000000Z ##[group]Run python scripts/x.py\n")
        with self.assertRaisesRegex(t.BudgetError, "no time stamps"):
            t.build_seconds("##[group]Run cargo test --profile ci-test\n")


if __name__ == "__main__":
    unittest.main()
