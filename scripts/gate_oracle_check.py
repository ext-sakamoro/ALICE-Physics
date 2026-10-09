#!/usr/bin/env python3
"""Check that gates can fail: run every gate's must-red and must-green
controls and compare the exit status (and, when given, a reason substring)
against what `scripts/gate_oracles/gates.toml` says to expect.

A gate that cannot turn red reports green for code nobody checked. This
runs each gate against a small fixed input built to be wrong (the
must-red control) and a small fixed input built to be right (the
must-green control), so a gate that always exits 0 — or compares nothing —
is caught here instead of by a later audit. See docs/GATE_ORACLES.md.

For every `[[gate]]` entry and each of its `[[gate.control]]` entries:

  1. copy the control's directory to a fresh temporary directory;
  2. run `run` (with `{script}` and `{dir}` substituted) under a time
     limit (60s for `cost = "fast"`, 900s for `cost = "cargo"`);
  3. compare the exit status with `expect` ("fail" -> nonzero, "pass" ->
     zero); when the control also gives `reason`, require that substring
     in stdout+stderr together, so a gate failing for an unrelated reason
     (a missing file, a traceback) does not count as the expected red.

Fails when any control has the wrong exit status or is missing its
`reason`; when a gate has no must-red control or no must-green control;
when `gates.toml` names a `source` that does not exist; or when it ran no
control at all (its own empty-input case).

  python3 scripts/gate_oracle_check.py                    # all fast gates
  python3 scripts/gate_oracle_check.py --cost cargo        # cargo gates too
  python3 scripts/gate_oracle_check.py --gates-toml <path> # a different file (tests)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GATES_TOML = REPO_ROOT / "scripts" / "gate_oracles" / "gates.toml"
TIME_LIMIT_S = {"fast": 60, "cargo": 900}

# Forced for every control of every gate, unconditionally (2026-10-09:
# ci.yml's CARGO_TERM_COLOR: always broke three gates that read an
# external tool's output with a plain-text regex -- each had been tested
# only without color, so the break was invisible here and only showed up
# as a red main). A gate whose own `run` already pins color back to
# "never" (see COLOR_PIN_MARKERS below) overrides this via its own shell
# assignment, which always wins for that command's own execution; this
# is the baseline a gate that does not override it is actually run under.
FORCED_ENV = {**os.environ, "CARGO_TERM_COLOR": "always"}

# Literal substrings that prove a gate's own `run` pins color itself,
# making it safe regardless of FORCED_ENV. Checked at load time (see
# load_gates): a gate whose `run` contains none of these must have a
# must-green (expect = "pass") control, which -- now that every control
# always runs under FORCED_ENV -- is the proof that gate tolerates it.
COLOR_PIN_MARKERS = ("CARGO_TERM_COLOR=never", "--color never", "--colors never")


@dataclass(frozen=True)
class Control:
    gate_id: str
    case: str
    expect: str  # "fail" | "pass"
    reason: str | None
    control_dir: Path


@dataclass(frozen=True)
class Gate:
    id: str
    kind: str
    source: str
    run: str
    cost: str
    controls: list[Control]


class GateOracleError(Exception):
    """A malformed gates.toml, not a failed control."""


def load_gates(gates_toml: Path) -> list[Gate]:
    if not gates_toml.is_file():
        raise GateOracleError(f"no such file: {gates_toml}")
    with gates_toml.open("rb") as f:
        doc = tomllib.load(f)
    gates_dir = gates_toml.parent
    gates: list[Gate] = []
    for raw in doc.get("gate", []):
        gate_id = raw["id"]
        source = raw["source"]
        if not (REPO_ROOT / source).exists():
            raise GateOracleError(f"{gate_id}: source does not exist: {source}")
        controls = []
        for raw_c in raw.get("control", []):
            case = raw_c["case"]
            expect = raw_c["expect"]
            if expect not in ("fail", "pass"):
                raise GateOracleError(
                    f"{gate_id}/{case}: expect must be 'fail' or 'pass', got {expect!r}"
                )
            control_dir = gates_dir / gate_id / case
            if not control_dir.is_dir():
                raise GateOracleError(f"{gate_id}/{case}: no such directory: {control_dir}")
            controls.append(
                Control(
                    gate_id=gate_id,
                    case=case,
                    expect=expect,
                    reason=raw_c.get("reason"),
                    control_dir=control_dir,
                )
            )
        run = raw["run"]
        if not any(marker in run for marker in COLOR_PIN_MARKERS) and not any(
            c.expect == "pass" for c in controls
        ):
            raise GateOracleError(
                f"{gate_id}: run does not pin color ({COLOR_PIN_MARKERS}) and has no "
                "must-green control -- every control now runs under FORCED_ENV "
                "(CARGO_TERM_COLOR=always), so an unpinned gate needs a passing "
                "control to prove it tolerates that"
            )
        gates.append(
            Gate(
                id=gate_id,
                kind=raw.get("kind", "script"),
                source=source,
                run=run,
                cost=raw.get("cost", "fast"),
                controls=controls,
            )
        )
    return gates


def run_control(gate: Gate, control: Control) -> list[str]:
    """Run one control; return a list of error strings (empty if it passed)."""
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"gate-oracle-{gate.id}-") as tmp:
        tmp_dir = Path(tmp)
        for item in control.control_dir.iterdir():
            dest = tmp_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        command = gate.run.format(script=str(REPO_ROOT / gate.source), dir=str(tmp_dir))
        timeout = TIME_LIMIT_S[gate.cost]
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=FORCED_ENV,
            )
            exit_code, output = proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as e:
            exit_code, output = None, (e.stdout or "") + (e.stderr or "")

        if exit_code is None:
            errors.append(f"{gate.id}/{control.case}: timed out after {timeout}s")
        else:
            got_pass = exit_code == 0
            want_pass = control.expect == "pass"
            if got_pass != want_pass:
                errors.append(
                    f"{gate.id}/{control.case}: expected to {control.expect}, "
                    f"exit code was {exit_code}"
                )
        if control.reason is not None and control.reason not in output:
            errors.append(
                f"{gate.id}/{control.case}: output did not contain reason "
                f"{control.reason!r} (output: {output[:500]!r})"
            )
    return errors


def check(gates: list[Gate], cost_filter: set[str]) -> list[str]:
    errors: list[str] = []
    tested = 0
    for gate in gates:
        if gate.cost not in cost_filter:
            continue
        expects = {c.expect for c in gate.controls}
        if "fail" not in expects:
            errors.append(f"{gate.id}: no must-red control")
        if "pass" not in expects:
            errors.append(f"{gate.id}: no must-green control")
        for control in gate.controls:
            tested += 1
            errors.extend(run_control(gate, control))
    if tested == 0:
        errors.append("ran no control at all (compared nothing)")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gates-toml", type=Path, default=DEFAULT_GATES_TOML)
    ap.add_argument(
        "--cost",
        choices=["fast", "cargo"],
        default="fast",
        help="'fast' runs only cost=fast gates (the default, for every push); "
        "'cargo' runs cost=fast and cost=cargo gates",
    )
    args = ap.parse_args(argv)
    cost_filter = {"fast"} if args.cost == "fast" else {"fast", "cargo"}

    try:
        gates = load_gates(args.gates_toml)
    except GateOracleError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    errors = check(gates, cost_filter)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: gates {len(gates)}, controls run against cost in {sorted(cost_filter)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
