#!/usr/bin/env python3
"""Keep the heavy CI analyses from crowding out the runs the landing script waits on.

Runners are shared: when the weekly mutation shards and per-push analyses ran
together they held the runners for hours, and the CI of the ci/** branches that
scripts/land.py waits on stayed queued. The rules:

  * quality-deep.yml (weekly mutants) never triggers on push (so never on ci/**),
    has one concurrency group per workflow (and one more for its diff_base
    dispatches, QD_GROUP; see the amendment of 2026-11-23), and caps its mutation
    matrix with `max-parallel` of at most MAX_PARALLEL (WEEKLY_MAX_PARALLEL in the
    weekly run);
  * bench-gate.yml and integration-status.yml use exactly
    the two-class concurrency of TWO_CLASS_GROUP with `cancel-in-progress: false`:
    at most one ci/** run at a time and one run per main commit, a new run waits
    instead of cancelling, so a ci/** push never cancels the main run (a group per
    ref would let every branch start its own copy; a missing group or
    `cancel-in-progress: true` lets a run be cut off or doubled; see the
    amendment of 2026-11-25 for main and the manual runs);
  * unsafe-and-parallel.yml uses one group per ref, per commit on main and per
    run for a manual run (PER_REF_GROUP), with `cancel-in-progress`
    PER_REF_CANCEL: a newer push on the same ci/** branch cancels that branch's
    older run, another branch's run is never cancelled or replaced, and a main
    run is never cut off (see the amendments of 2026-11-09 and 2026-11-25);
  * unsafe-and-parallel.yml (Miri and loom, 45-95 min) may run on ci/** only behind
    its path gate: the workflow passes `--patterns-file
    .github/paths/unsafe-and-parallel.txt` to scripts/changed_paths.py, every job
    other than `changes` and `gate` runs only when the changes job says so, and a
    src file outside the unsafe / parallel code does not match the patterns.

Amendment, 2026-10-12. Amended rule: heavy jobs stay off every ci/** push, and
their concurrency group does not include the ref. Reason: the landing evidence on
a ci/** branch was missing 8 workflows (their push triggers were main-only or
path-filtered), and main went red in Security & Hygiene after a landing. The rule
now reads: a heavy job may run on a ci/** push only behind a path gate (the
changes job of scripts/changed_paths.py; for unsafe-and-parallel.yml the gate is
narrowed to the unsafe / parallel code and checked here), and concurrency limits
the parallel heavy runs (one main run and one ci/** run per workflow) instead of
keeping them off ci/**. quality-deep.yml stays off push entirely.

Amendment, 2026-11-09. Amended rule: unsafe-and-parallel.yml has one concurrency
group per ref instead of the two-class group. Reason: with one ci/** slot (one
running and one waiting run for every branch together) each new ci/** push or
re-created run replaced the waiting run of another branch; in one hour six waiting
Miri runs of five branches were cancelled this way, and a landing held the landing
lock about 9 h re-creating its run. The rule now reads: each branch has its own
group, a newer push on that branch cancels its own older run (the older SHA is
superseded), main runs wait instead of being cut off, and the number of parallel
runs is bounded by the path gate (only a change of the unsafe / parallel code or
its tests starts one), which stays required on ci/**.

Amendment, 2026-11-23. Amended rule: quality-deep.yml had one group and at most
MAX_PARALLEL mutation jobs at a time. Reason: a group holds one waiting run and a
newer one replaces it, so a diff_base dispatch that waited behind a weekly run was
replaced by the next nightly (or replaced it); and the weekly core, cut by the
measured baseline, is 25 shards an axis where 16 no longer end in the job limit,
which four jobs at a time do not finish in the 48 h before the next nightly. The
rule now reads: the scheduled runs keep the one group; the diff_base dispatches
have one group of their own, for every ref together (one running and one waiting,
never a group per ref); the mutation matrix runs at most MAX_PARALLEL jobs at a
time, WEEKLY_MAX_PARALLEL in the weekly run, written as QD_PARALLEL_RE.

Amendment, 2026-11-25. Amended rule: unsafe-and-parallel.yml had one group per
ref for every event. Reason: a manual run (workflow_dispatch) of a ref shared the
group of the push run of the same ref, so on a ci/** branch one cancelled the
other, and the landing evidence reads a cancelled push run as red. The rule now
reads: a manual run has a group of its own (`dispatch-<run id>`); a push run has
one group per ref off main, as before, and one group per commit on main (a main
run is neither cut off nor replaced while it waits; how many run at once is
bounded by the path gate, as on ci/**).
The same day, for bench-gate.yml and integration-status.yml: the `main` class was
one group for every main commit, so a third main push replaced the waiting run of
the second and that commit had no run of its own; and a manual run waited in the
class of its ref, where it could replace a waiting push run. The rule now reads:
on main one group per commit (landings are one at a time, so the main runs are as
many as the landings), off main the one `ci` group as before, and a manual run has
a group of its own.

The workflows are read as text (no YAML library on the runners): the `on:`
block, the top-level `concurrency:` block and the job headers are inspected.
Fails when no heavy workflow was compared.

  python3 scripts/ci_load_check.py            # exit 1 on a finding
  python3 scripts/ci_load_check.py --root DIR
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import changed_paths as cp  # noqa: E402

QUALITY_DEEP = "quality-deep.yml"
TWO_CLASS = ("bench-gate.yml", "integration-status.yml")
PER_REF = ("unsafe-and-parallel.yml",)
PATH_GATED = ("unsafe-and-parallel.yml",)
HEAVY = (QUALITY_DEEP,) + TWO_CLASS + PER_REF
MAX_PARALLEL = 4
WEEKLY_MAX_PARALLEL = 6
QD_GROUPS = ("${{ github.workflow }}",
             "${{ github.workflow }}${{ github.event_name == 'workflow_dispatch' && inputs.diff_base != '' && '-diff' || '' }}")
# `max-parallel` of the mutation matrix: a number, or the weekly number and the number of the other runs
QD_PARALLEL_RE = re.compile(r"^\s*max-parallel:\s*(?:(\d+)|\$\{\{ needs\.mutants-plan\.outputs\.mode == 'weekly' && (\d+) \|\| (\d+) \}\})\s*$", re.M)
TWO_CLASS_GROUP = ("${{ github.workflow }}-${{ github.event_name == 'workflow_dispatch' && format('dispatch-{0}', github.run_id) "
                   "|| (github.ref == 'refs/heads/main' && github.sha || 'ci') }}")
PER_REF_GROUP = ("${{ github.workflow }}-${{ github.event_name == 'workflow_dispatch' && format('dispatch-{0}', github.run_id) "
                 "|| (github.ref == 'refs/heads/main' && github.sha || github.ref) }}")
PER_REF_CANCEL = "${{ github.ref != 'refs/heads/main' }}"
RUN_IF = "needs.changes.outputs.run == 'true'"
# a src file that is not unsafe or parallel code: the path gate must not select it
UNRELATED_SRC = "src/zz_unrelated_module.rs"


def top_block(text: str, key: str) -> str:
    """The lines of a top-level `key:` block (until the next top-level key)."""
    out, inside = [], False
    for line in text.splitlines():
        if re.match(rf"^{re.escape(key)}:\s*(#.*)?$", line):
            inside = True
            continue
        if inside and re.match(r"^[A-Za-z_]", line):
            break
        if inside:
            out.append(line)
    return "\n".join(out)


def scalar(block: str, key: str) -> str | None:
    m = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", block, re.M)
    return m.group(1) if m else None


def jobs(text: str) -> dict[str, str]:
    """job id -> its block text (the job's lines under `jobs:`)."""
    out: dict[str, str] = {}
    cur = None
    for line in top_block(text, "jobs").splitlines():
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*(#.*)?$", line)
        if m:
            cur = m.group(1)
            out[cur] = ""
        elif cur is not None:
            out[cur] += line + "\n"
    return out


def path_gate_errors(root: Path, name: str, text: str) -> list[str]:
    stem = name.rsplit(".", 1)[0]
    rel = f".github/paths/{stem}.txt"
    errors = []
    if f"--patterns-file {rel}" not in text:
        errors.append(f"{name}: runs on ci/** without its path gate (no `--patterns-file {rel}`)")
        return errors
    pfile = root / rel
    if not pfile.is_file():
        return [f"{name}: path gate {rel} is missing"]
    try:
        rules = cp.compile_patterns(cp.read_patterns(str(pfile), []))
    except (cp.PatternError, OSError) as e:
        return [f"{name}: path gate {rel}: {e}"]
    if cp.selected(UNRELATED_SRC, rules):
        errors.append(f"{name}: path gate {rel} selects {UNRELATED_SRC} (a src file outside the "
                      "unsafe / parallel code would start the heavy run)")
    js = jobs(text)
    heavy = [j for j in js if j not in ("changes", "gate")]
    if "changes" not in js:
        errors.append(f"{name}: runs on ci/** without a `changes` job")
    if not heavy:
        errors.append(f"{name}: no job besides changes / gate (compared nothing)")
    for j in heavy:
        if RUN_IF not in js[j]:
            errors.append(f"{name}: job {j} does not wait for the path gate (`if: {RUN_IF}`)")
    return errors


def check(root: Path) -> tuple[list[str], int]:
    errors, seen = [], 0
    wf = root / ".github" / "workflows"
    for name in HEAVY:
        p = wf / name
        if not p.is_file():
            errors.append(f"{name}: missing (the check compares nothing for it)")
            continue
        seen += 1
        text = p.read_text(encoding="utf-8")
        on = top_block(text, "on")
        if not on:
            errors.append(f"{name}: no `on:` block found")
        on_ci = bool(re.search(r"ci/\*\*", on))
        conc = top_block(text, "concurrency")
        group = scalar(conc, "group")
        cancel = scalar(conc, "cancel-in-progress")
        if group is None:
            errors.append(f"{name}: no top-level concurrency group (two runs can hold runners at once)")
        if name == QUALITY_DEEP:
            if on_ci:
                errors.append(f"{name}: triggers on ci/** pushes; it runs on a schedule or dispatch only")
            if re.search(r"^  push:", on, re.M):
                errors.append(f"{name}: triggers on push; it runs on a schedule or dispatch only")
            if group is not None and "github.ref" in group:
                errors.append(f"{name}: concurrency group {group} is per ref; use one group per workflow")
            elif group is not None and group not in QD_GROUPS:
                errors.append(f"{name}: concurrency group {group} is not the workflow's one group, nor that group "
                              "with the one group of the diff_base dispatches")
            if cancel != "false":
                errors.append(f"{name}: cancel-in-progress is {cancel!r}; a running mutation run is never cut off")
            lines = re.findall(r"^\s*max-parallel:.*$", text, re.M)
            found = QD_PARALLEL_RE.findall(text)
            if not lines:
                errors.append(f"{name}: the mutation matrix has no max-parallel")
            elif len(found) != len(lines):
                errors.append(f"{name}: a max-parallel is neither a number nor the weekly / other pair (not read)")
            for plain, weekly, other in found:
                if plain and int(plain) > MAX_PARALLEL:
                    errors.append(f"{name}: max-parallel {plain} exceeds {MAX_PARALLEL}")
                if weekly and int(weekly) > WEEKLY_MAX_PARALLEL:
                    errors.append(f"{name}: max-parallel {weekly} of the weekly run exceeds {WEEKLY_MAX_PARALLEL}")
                if other and int(other) > MAX_PARALLEL:
                    errors.append(f"{name}: max-parallel {other} exceeds {MAX_PARALLEL}")
            continue
        if name in PER_REF:
            if group is not None and group != PER_REF_GROUP:
                errors.append(f"{name}: concurrency group {group} is not the per-ref group "
                              f"{PER_REF_GROUP} (another branch's waiting run would be replaced)")
            if cancel != PER_REF_CANCEL:
                errors.append(f"{name}: cancel-in-progress is {cancel!r}; it must be {PER_REF_CANCEL} "
                              "(a newer push cancels its own branch's older run, a main run is not cut off)")
            if on_ci:
                # a group per ref is bounded only by the path gate
                errors.extend(path_gate_errors(root, name, text))
            continue
        if group is not None and group != TWO_CLASS_GROUP:
            errors.append(f"{name}: concurrency group {group} is not the two-class group "
                          f"{TWO_CLASS_GROUP} (one ci/** run at a time, one run per main commit)")
        if cancel != "false":
            errors.append(f"{name}: cancel-in-progress is {cancel!r}; it must be false (a ci/** push "
                          "must not cancel the main run, and a run must not be cut off)")
        if name in PATH_GATED and on_ci:
            errors.extend(path_gate_errors(root, name, text))
    if seen == 0:
        errors.append("no heavy workflow found (compared nothing)")
    return errors, seen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args(argv)
    errors, seen = check(Path(args.root))
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(f"compared: heavy workflows {seen}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
