#!/usr/bin/env python3
"""Keep the heavy CI analyses from crowding out the runs the landing script waits on.

Runners are shared: when the weekly mutation shards and per-push analyses ran
together they held the runners for hours, and the CI of the ci/** branches that
scripts/land.py waits on stayed queued. The rules:

  * quality-deep.yml (weekly mutants) never triggers on push (so never on ci/**),
    has one concurrency group per workflow, and caps its mutation matrix with
    `max-parallel` of at most MAX_PARALLEL;
  * bench-gate.yml, unsafe-and-parallel.yml and integration-status.yml use exactly
    the two-class concurrency of TWO_CLASS_GROUP with `cancel-in-progress: false`:
    at most one main run and one ci/** run at a time, a new run waits instead of
    cancelling, so a ci/** push never cancels the main run (a group per ref would
    let every branch start its own copy; a missing group or `cancel-in-progress:
    true` lets a run be cut off or doubled);
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
TWO_CLASS = ("bench-gate.yml", "unsafe-and-parallel.yml", "integration-status.yml")
PATH_GATED = ("unsafe-and-parallel.yml",)
HEAVY = (QUALITY_DEEP,) + TWO_CLASS
MAX_PARALLEL = 4
TWO_CLASS_GROUP = "${{ github.workflow }}-${{ github.ref == 'refs/heads/main' && 'main' || 'ci' }}"
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
            mp = [int(x) for x in re.findall(r"^\s*max-parallel:\s*(\d+)", text, re.M)]
            if not mp:
                errors.append(f"{name}: the mutation matrix has no max-parallel")
            elif max(mp) > MAX_PARALLEL:
                errors.append(f"{name}: max-parallel {max(mp)} exceeds {MAX_PARALLEL}")
            continue
        if group is not None and group != TWO_CLASS_GROUP:
            errors.append(f"{name}: concurrency group {group} is not the two-class group "
                          f"{TWO_CLASS_GROUP} (one main run and one ci/** run at a time)")
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
