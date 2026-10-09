#!/usr/bin/env python3
"""Tests for scripts/gate_oracle_check.py.

The runner's own self-test (docs/GATE_ORACLES.md section 4): a stub gate
that always exits 0 must be reported as failing its must-red control, and
a stub that always exits 1 as failing its must-green control. These two
fixtures live in scripts/gate_oracles/runner/ so they are the same files
the real runner would use if pointed at them; this file is what actually
checks that pointing the runner at them produces the expected report."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import gate_oracle_check as goc  # noqa: E402

RUNNER_DIR = "scripts/gate_oracles/runner"


def control(gate_id: str, case: str, expect: str, reason: str | None = None) -> goc.Control:
    return goc.Control(
        gate_id=gate_id,
        case=case,
        expect=expect,
        reason=reason,
        control_dir=REPO_ROOT / RUNNER_DIR / case,
    )


class SelfTest(unittest.TestCase):
    """The runner's own controls, per docs/GATE_ORACLES.md section 4."""

    def test_always_exit_0_fails_its_must_red_control(self):
        gate = goc.Gate(
            id="runner-self-test-0",
            kind="script",
            source=f"{RUNNER_DIR}/always_exit_0.py",
            run="python3 {script}",
            cost="fast",
            controls=[control("runner-self-test-0", "always-exit-0", "fail")],
        )
        errors = goc.run_control(gate, gate.controls[0])
        self.assertTrue(errors, "a gate that always exits 0 must fail a must-red control")
        self.assertIn("expected to fail", errors[0])

    def test_always_exit_1_fails_its_must_green_control(self):
        gate = goc.Gate(
            id="runner-self-test-1",
            kind="script",
            source=f"{RUNNER_DIR}/always_exit_1.py",
            run="python3 {script}",
            cost="fast",
            controls=[control("runner-self-test-1", "always-exit-1", "pass")],
        )
        errors = goc.run_control(gate, gate.controls[0])
        self.assertTrue(errors, "a gate that always exits 1 must fail a must-green control")
        self.assertIn("expected to pass", errors[0])

    def test_reason_mismatch_is_reported_even_when_the_exit_status_matches(self):
        gate = goc.Gate(
            id="runner-self-test-0",
            kind="script",
            source=f"{RUNNER_DIR}/always_exit_0.py",
            run="python3 {script}",
            cost="fast",
            controls=[
                control(
                    "runner-self-test-0",
                    "always-exit-0",
                    "pass",
                    reason="this string never appears in always_exit_0.py's output",
                )
            ],
        )
        errors = goc.run_control(gate, gate.controls[0])
        self.assertTrue(errors)
        self.assertIn("did not contain reason", errors[0])


class LoadGates(unittest.TestCase):
    def write(self, tmp: Path, text: str) -> Path:
        path = tmp / "gates.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_empty_file_is_zero_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(Path(tmp), "# nothing yet\n")
            self.assertEqual(goc.load_gates(path), [])

    def test_missing_file_is_an_error(self):
        with self.assertRaises(goc.GateOracleError):
            goc.load_gates(Path("/nonexistent/gates.toml"))

    def test_nonexistent_source_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(
                Path(tmp),
                '[[gate]]\nid = "x"\nsource = "scripts/does_not_exist.py"\nrun = "true"\n',
            )
            with self.assertRaises(goc.GateOracleError) as ctx:
                goc.load_gates(path)
            self.assertIn("does not exist", str(ctx.exception))

    def test_invalid_expect_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            control_dir = tmp_path / "x" / "c"
            control_dir.mkdir(parents=True)
            path = self.write(
                tmp_path,
                f'[[gate]]\nid = "x"\nsource = "{RUNNER_DIR}/always_exit_0.py"\n'
                'run = "true"\n\n[[gate.control]]\ncase = "c"\nexpect = "maybe"\n',
            )
            with self.assertRaises(goc.GateOracleError) as ctx:
                goc.load_gates(path)
            self.assertIn("expect must be", str(ctx.exception))

    def test_missing_control_directory_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(
                Path(tmp),
                f'[[gate]]\nid = "x"\nsource = "{RUNNER_DIR}/always_exit_0.py"\n'
                'run = "true"\n\n[[gate.control]]\ncase = "no-such-dir"\nexpect = "fail"\n',
            )
            with self.assertRaises(goc.GateOracleError) as ctx:
                goc.load_gates(path)
            self.assertIn("no such directory", str(ctx.exception))


class Check(unittest.TestCase):
    def test_zero_gates_is_ran_no_control_at_all(self):
        errors = goc.check([], {"fast"})
        self.assertEqual(errors, ["ran no control at all (compared nothing)"])

    def test_a_gate_with_only_a_must_red_control_is_an_error(self):
        gate = goc.Gate(
            id="x",
            kind="script",
            source=f"{RUNNER_DIR}/always_exit_0.py",
            run="python3 {script}",
            cost="fast",
            controls=[control("x", "always-exit-0", "fail")],
        )
        errors = goc.check([gate], {"fast"})
        self.assertTrue(any("no must-green control" in e for e in errors))

    def test_a_gate_with_only_a_must_green_control_is_an_error(self):
        gate = goc.Gate(
            id="x",
            kind="script",
            source=f"{RUNNER_DIR}/always_exit_1.py",
            run="python3 {script}",
            cost="fast",
            controls=[control("x", "always-exit-1", "pass")],
        )
        errors = goc.check([gate], {"fast"})
        self.assertTrue(any("no must-red control" in e for e in errors))

    def test_a_well_behaved_gate_with_both_controls_passes(self):
        # grep itself as the toy gate under test: the run command greps
        # each control's own copied marker.txt for a fixed pattern, so
        # the must-red control's marker lacks it (grep exits 1) and the
        # must-green control's marker has it (grep exits 0).
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            red_dir = tmp_path / "x" / "red"
            green_dir = tmp_path / "x" / "green"
            red_dir.mkdir(parents=True)
            green_dir.mkdir(parents=True)
            (red_dir / "marker.txt").write_text("nothing here\n")
            (green_dir / "marker.txt").write_text("PATTERN is here\n")
            gate = goc.Gate(
                id="grep-toy",
                kind="script",
                source=f"{RUNNER_DIR}/always_exit_0.py",
                run="grep -q PATTERN {dir}/marker.txt",
                cost="fast",
                controls=[
                    goc.Control("grep-toy", "red", "fail", None, red_dir),
                    goc.Control("grep-toy", "green", "pass", None, green_dir),
                ],
            )
            self.assertEqual(goc.check([gate], {"fast"}), [])

    def test_cargo_gates_are_skipped_unless_asked_for(self):
        gate = goc.Gate(
            id="x",
            kind="script",
            source=f"{RUNNER_DIR}/always_exit_0.py",
            run="python3 {script}",
            cost="cargo",
            controls=[control("x", "always-exit-0", "fail")],
        )
        errors = goc.check([gate], {"fast"})
        self.assertEqual(errors, ["ran no control at all (compared nothing)"])


if __name__ == "__main__":
    unittest.main()
