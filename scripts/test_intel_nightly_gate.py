#!/usr/bin/env python3
"""Tests for scripts/intel_nightly_gate.py (one rule per case, synthetic API responses)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import intel_nightly_gate as g  # noqa: E402

CURRENT = 999


def job(conclusion: str | None, status: str = "completed", name: str = g.JOB) -> dict:
    return {"name": name, "status": status, "conclusion": conclusion}


def decide(jobs_by_run: dict[int, list[dict]]) -> tuple[bool, str]:
    runs = [{"id": rid} for rid in jobs_by_run]
    return g.decide(runs, lambda rid: jobs_by_run[rid], CURRENT)


class Decide(unittest.TestCase):
    def test_a_successful_intel_job_skips_the_nightly(self) -> None:
        self.assertFalse(decide({1: [job("success")]})[0])

    def test_a_failed_intel_job_runs_the_nightly(self) -> None:
        self.assertTrue(decide({1: [job("failure")]})[0])

    def test_a_cancelled_intel_job_runs_the_nightly(self) -> None:
        self.assertTrue(decide({1: [job("cancelled")]})[0])

    def test_a_timed_out_intel_job_runs_the_nightly(self) -> None:
        self.assertTrue(decide({1: [job("timed_out")]})[0])

    def test_an_intel_job_in_progress_is_left_to_finish(self) -> None:
        self.assertFalse(decide({1: [job(None, status="in_progress")]})[0])

    def test_a_queued_intel_job_is_left_to_finish(self) -> None:
        self.assertFalse(decide({1: [job(None, status="queued")]})[0])

    def test_no_run_of_the_commit_runs_the_nightly(self) -> None:
        run, why = decide({})
        self.assertTrue(run)
        self.assertIn("no Test (macos-15-intel) job", why)

    def test_runs_without_the_intel_job_run_the_nightly(self) -> None:
        # a ci/** run of the same commit has only the four other legs
        self.assertTrue(decide({1: [job("success", name="Test (ubuntu-latest)")]})[0])

    def test_one_success_among_failures_skips(self) -> None:
        self.assertFalse(decide({1: [job("failure")], 2: [job("success")]})[0])

    def test_the_calling_run_is_left_out(self) -> None:
        runs = [{"id": CURRENT}]
        run, _ = g.decide(runs, lambda rid: [job("success")], CURRENT)
        self.assertTrue(run, "the nightly's own run must not count as the commit's Intel result")


class Main(unittest.TestCase):
    def run_main(self, fetch) -> str:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            env = {"GITHUB_REPOSITORY": "o/r", "GITHUB_SHA": "abc", "GITHUB_RUN_ID": str(CURRENT),
                   "GITHUB_TOKEN": "t", "GITHUB_OUTPUT": str(out)}
            with mock.patch.dict(os.environ, env), mock.patch.object(g, "fetch", fetch):
                self.assertEqual(g.main(), 0)
            return out.read_text(encoding="utf-8")

    def test_an_api_error_runs_the_leg(self) -> None:
        def fail(path: str, token: str) -> dict:
            raise OSError("network down")
        self.assertEqual(self.run_main(fail), "run=true\n")

    def test_a_malformed_response_runs_the_leg(self) -> None:
        self.assertEqual(self.run_main(lambda path, token: {"unexpected": []}), "run=true\n")

    def test_the_api_paths_name_the_commit_and_the_runs(self) -> None:
        seen: list[str] = []

        def fetch(path: str, token: str) -> dict:
            seen.append(path)
            if "/workflows/ci.yml/runs?head_sha=abc" in path:
                return {"workflow_runs": [{"id": 5}]}
            return {"jobs": [job("success")]}
        self.assertEqual(self.run_main(fetch), "run=false\n")
        self.assertEqual(len(seen), 2, "one runs query and one jobs query (compared something)")
        self.assertIn("/actions/runs/5/jobs", seen[1])


if __name__ == "__main__":
    unittest.main()
