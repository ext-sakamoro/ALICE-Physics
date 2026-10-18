# CI triggers and landing evidence

A change lands in two steps: `scripts/land.py` pushes the commits to a `ci/<id>`
branch, waits for its CI, then moves `main` (fast-forward, after a rebase when `main`
moved meanwhile). The runs on the
`ci/<id>` SHA are the evidence for the landing. This page lists, for every workflow,
what starts it and what its run on a `ci/**` branch means.

`.github/ci-evidence.toml` declares the class of each workflow (and of the legs listed
below); `scripts/ci_triggers_check.py` fails when this page, that file and the workflow
files disagree, or when it compares no row (CI test job and `scripts/preflight.sh`).

## Evidence classes

| class | meaning |
|---|---|
| landing | must be green on the `ci/<id>` SHA before `main` is fast-forwarded to it |
| after-landing | started on the `ci/<id>` SHA (when its changes job outputs `run=true`); its result is read after the landing and does not hold it |
| main-after-landing | (a matrix leg only) not started on `ci/**`; runs on the push to `main` (and the schedule), read on `main` after the landing |
| schedule-only | never runs on a `ci/**` push (schedule, manual dispatch, pull request) |
| tag-only | runs on a version tag |

## Path gates

A workflow with a path gate has no `paths` filter on `on.push`. GitHub's filter
compares only the commits that are new to the repository when a push creates a branch,
so a `ci/**` branch created at a SHA that another branch already has got an empty diff
and the workflow did not run. Instead, a `changes` job runs `scripts/changed_paths.py`
with the patterns in `.github/paths/<workflow>.txt`:

- a new or force-pushed branch is compared with `merge-base(origin/main)..SHA`, another
  push with `before..after`, a pull request with `base...head` (the pull request keeps
  the same list as its `paths` filter);
- a schedule or a manual dispatch always runs;
- any error, an empty diff on a new branch, or a non-empty diff in which no file was
  compared runs the jobs (fail-closed);
- the heavy jobs `need` the changes job and run when it outputs `run=true`;
- the last job, `gate` (shown as `Gate`), runs `changed_paths.py gate` and fails when
  `run=true` and a heavy job was skipped, cancelled or failed, and when `run=false` and a
  heavy job ran. `ci.yml` has no path gate and no such job: its own `gate` job (shown as
  `gate`) decides whether the nightly schedule runs the Mac Intel leg
  (`scripts/intel_nightly_gate.py`), and is a different job. `scripts/ci_triggers_check.py`
  requires the guard's `gate` job in every path-gated workflow and in no other.

Each list names every file a job of the workflow executes or reads (manifests, lock,
toolchain file, scripts it calls and the scripts they import, baselines, committed
ledgers, `#[path]` and `include_*!` targets, the workflow and the list itself).
`scripts/changed_paths.py sync` checks that no workflow outside `main` filters
`on.push` by paths and that each list equals its workflow's `pull_request` paths.

## Workflows

<!-- ci-triggers:workflows:begin -->
| workflow | push | path gate | pull_request | schedule | dispatch | ci/** push | class |
|---|---|---|---|---|---|---|---|
| `bench-gate.yml` | `main, ci/**` | `.github/paths/bench-gate.txt` | - | - | yes | runs when the paths match | landing |
| `ci.yml` | `main, ci/**` | - | `main` | yes | yes | runs | landing |
| `downstream.yml` | `main, ci/**` | `.github/paths/downstream.txt` | `main (paths)` | - | yes | runs when the paths match | landing |
| `eol.yml` | `main, ci/**` | - | `main` | - | yes | runs | landing |
| `fuzz.yml` | `main, ci/**` | `.github/paths/fuzz.txt` | `main (paths)` | yes | yes | runs when the paths match | landing |
| `g2-comparisons.yml` | `main, ci/**` | `.github/paths/g2-comparisons.txt` | - | - | yes | runs when the paths match | landing |
| `integration-status.yml` | `main, ci/**` | `.github/paths/integration-status.txt` | - | - | yes | runs when the paths match | landing |
| `oracle-status.yml` | `main, ci/**` | `.github/paths/oracle-status.txt` | - | - | yes | runs when the paths match | landing |
| `public-text.yml` | `main, ci/**` | - | `main` | - | yes | runs | landing |
| `python-binding.yml` | `main, ci/**` | `.github/paths/python-binding.txt` | `main (paths)` | - | yes | runs when the paths match | landing |
| `quality-deep.yml` | - | - | `main (paths)` | yes | yes | - | schedule-only |
| `reexport_check.yml` | `main, ci/**` | - | `main` | - | yes | runs | landing |
| `release.yml` | `tags v*` | - | - | - | - | - | tag-only |
| `security-audit.yml` | `main, ci/**` | `.github/paths/security-audit.txt` | `main (paths)` | yes | yes | runs when the paths match | landing |
| `unsafe-and-parallel.yml` | `main, ci/**` | `.github/paths/unsafe-and-parallel.txt` | - | - | yes | runs when the paths match | after-landing |
| `wiring-status.yml` | `main, ci/**` | `.github/paths/wiring-status.txt` | - | - | yes | runs when the paths match | landing |
<!-- ci-triggers:workflows:end -->

## Legs of landing workflows read after the landing

The macOS legs of the test matrix take the longest. The landing does not wait for
either of them, and a release (crates.io, PyPI) needs every leg green.

- `macos-latest` (class `after-landing`): starts on the `ci/<id>` SHA with the rest of
  the run; its result is read after the landing. It is the arm64 leg that holds the
  golden hashes: a red result there after a landing is fixed or reverted at once.
- `macos-15-intel` (class `main-after-landing`): does **not** run on a `ci/**` push, a
  pull request or a manual run. It runs on the push to `main` (all 5 OS) and on the
  nightly schedule (that leg alone, when `main`'s HEAD has no successful Intel job); its
  result is read on `main` after the landing.

### Mac Intel on a `ci/**` push (landing script changes)

When the branch changes `scripts/land.py` or `scripts/test_land.py`, the Mac Intel leg
also runs on the `ci/**` push and on a manual run, as an ordinary job of the run: the
landing waits for it like the other legs. Those files start the runner's own git and
tools, and the Intel image has different ones (its system git was older than the
`--attr-source` option land.py passes, which only showed after a landing). The `gate`
job of `ci.yml` decides it with `scripts/changed_paths.py decide` (a new or force-pushed
branch and a manual run are compared with `merge-base(origin/main)`, a later push with
`before..after`; an error or a step without an answer runs the leg) and outputs
`landing_scripts`. The cost is one Intel job (about 100 min) per such branch. In
`.github/ci-evidence.toml` the leg's `events_when_landing_scripts` lists the events it
runs on then, and `scripts/ci_triggers_check.py` evaluates the matrix with that output
true and false and compares both.

The `test` matrix is computed from the event: the nightly schedule runs Mac Intel only,
a push to `main` runs all 5 OS, and every other event, a manual run (`workflow_dispatch`,
on any branch) included, runs the 4 others. The `events` column is checked against that
expression: `scripts/ci_triggers_check.py` evaluates the matrix for each event and fails
when a leg's declared events differ, or when a leg that runs on a `main` push but not on
a `ci/**` push is not declared.

<!-- ci-triggers:after-landing-legs:begin -->
| workflow | job | matrix | class | events |
|---|---|---|---|---|
| `ci.yml` | `test` | `macos-latest` | after-landing | push main, push ci/**, pull_request, workflow_dispatch |
| `ci.yml` | `test` | `macos-15-intel` | main-after-landing | push main, schedule |
| `ci.yml` | `determinism-release` | `macos-latest` | after-landing | push main, push ci/**, pull_request, workflow_dispatch |
<!-- ci-triggers:after-landing-legs:end -->

## What the landing waits on

Every landing takes the `ci/<id>` lane: `scripts/land.py` pushes the commits to
`ci/<id>` and waits on every workflow whose `on.push` runs on `ci/**`, with the class of
`.github/ci-evidence.toml` (the list is derived from the workflow files, not written in
the script), before `main` moves. A change of `src/` only is no exception.

- **landing**: the newest run for the SHA completes with conclusion `success`. For a
  path-gated workflow that is the `Gate` job's verdict, so a run whose heavy jobs were
  skipped and whose `Gate` succeeded passes. A matrix leg of class `after-landing` only has
  to have started (`in_progress` or `completed`); every other job of the run must succeed.
- **after-landing**: the run exists, its changes job succeeded, and its heavy jobs have
  started, or were all skipped because the changes job said `run=false` (read from the jobs
  GitHub reports, not decided again locally). Its result is not awaited.
- A known red result stops the landing, also on an after-landing leg or workflow.
- A workflow with no run for the SHA (after 10 minutes), or whose run was cancelled or
  skipped, has no evidence: the script checks that `ci/<id>` still points at the SHA and
  re-creates the run with `gh workflow run <workflow> --ref ci/<id>`, at most twice per
  workflow, then stops. Absence is never read as success, and a landing that would wait on
  0 workflows fails.
- The newest run for the SHA counts: a newer red run (a re-run or a manual run) stops the
  landing even when an older run of the same workflow was green.
- One `gh run list` per poll (every 30 s) plus one `gh run view` per unfinished run whose
  decision needs its jobs; the wait is bounded at 2 hours.
- The landing record printed at the end names the lane, one line per expected workflow
  (`evidence: <workflow> run <id> <conclusion>`, for the last evidence SHA), the evidence
  SHA, the landed SHA and a reason class, so it can be pasted as evidence:
  - `main-unmoved`: `main` did not move after the evidence; the evidenced SHA itself lands;
  - `docs-only-moved`: `main` gained documents only; the change is rebased onto them and
    the local checks run again (no new evidence); in every document `main` changed, each line
    `main` or the evidence added must still be in the landed file (counted, renames followed),
    a document `main` renamed into a ledger must not carry a change of the evidence, and a
    document `main` deleted (or the old name of one it renamed) must stay gone unless the evidence
    changed it, or nothing lands;
  - `re-evidenced (evidence rounds N)`: `main` gained code; see below;
  - `direct (no evidence)`: the emergency escape.

### When `main` moves after the evidence

After the evidence, `land.py` fetches `main` again before pushing:

- **Documents only**: every changed file is under `docs/`, a top-level `README*.md` or
  `CHANGELOG.md`, and no `.github/paths/*.txt` list (the inputs of the path-gated
  workflows) selects it. The ledgers `land.py` rebuilds and checks after every rebase
  (`docs/integration-status.md`, `docs/integration-levels.md`, `docs/oracle-status.md`,
  `docs/wiring-status.md`, `docs/coverage/status.md`) count as documents although
  workflows read them. Not documents: `docs/PUBLIC_API_SNAPSHOT.txt`, `README.md` and
  `docs/wiring-debt.md` (workflow inputs), a `.rs` file anywhere, and a change that mixes
  a document with any other file. When the lists cannot be read, every move counts as
  code. The change is rebased, the ledgers rebuilt and the local checks (documents,
  ledgers, `CHANGELOG`) run again.
- **Code**: the change is rebased and checked (`preflight.sh --fast` when `main` touched
  our files or the build, otherwise a compile check), the new SHA is pushed to our own
  `ci/<id>` (a force update of the earlier attempt) and the expected set is awaited again
  for the **new SHA only** (the newest run of each workflow for that SHA). At most 3
  evidence rounds in all; when `main` gains code a third time the landing stops with
  `main is moving: wait for a quiet main or coordinate a landing slot`.
- A re-evidence push follows the heavy-job rules: Bench gate, Miri and loom and
  Integration Status use one `ci` concurrency group with `cancel-in-progress: false`, so a
  new `ci/**` run waits for the running one, and GitHub keeps one pending run per group: the
  pending run of the earlier attempt is replaced, and no extra heavy run starts in
  parallel.

### Landing lock

`land.py` takes an exclusive lock per repository on this machine
(`~/.cache/alice-land/<repository>.lock`; the directory can be set with
`LAND_LOCK_DIR`) from the first fetch to the push to `main`. A second landing waits up to
`--lock-wait-minutes` (default 30), then stops naming the holder (pid, start time,
landing id). The lock is per machine: landings from different machines are serialized
only by the fast-forward-only push to `main` and the re-evidence loop above.

`scripts/ci_triggers_check.py` compares this table with what `scripts/land.py` derives:

<!-- ci-triggers:landing-waits:begin -->
| workflow | class | waits for |
|---|---|---|
| `bench-gate.yml` | landing | success |
| `ci.yml` | landing | success; started only: Test (macos-latest), Determinism tests (release, macos-latest) |
| `downstream.yml` | landing | success |
| `eol.yml` | landing | success |
| `fuzz.yml` | landing | success |
| `g2-comparisons.yml` | landing | success |
| `integration-status.yml` | landing | success |
| `oracle-status.yml` | landing | success |
| `public-text.yml` | landing | success |
| `python-binding.yml` | landing | success |
| `reexport_check.yml` | landing | success |
| `security-audit.yml` | landing | success |
| `unsafe-and-parallel.yml` | after-landing | started: heavy jobs started, or all skipped by the changes job |
| `wiring-status.yml` | landing | success |
<!-- ci-triggers:landing-waits:end -->

### Emergency escape: `--direct`

`python3 scripts/land.py --id <id> --direct --reason "<why, at least 12 characters>"`
lands without the `ci/<id>` push and without any CI evidence, for an emergency such as
reverting a red `main`. It is refused without a reason or with a shorter one. It prints a
warning with the reason, and the landing record carries
`lane: DIRECT (emergency, no ci/ evidence) reason: <why>`. There is no environment
variable or default that skips the `ci/<id>` lane.

## What is read after the landing

`python3 scripts/land.py --id <id> --wait` (after the landing) or
`--wait-only <SHA>` reads the after-landing set on `main` for the landed SHA and reports
each item on its own line:

- each matrix leg of class `after-landing` or `main-after-landing` (the macOS test legs):
  the job in the newest run of its workflow that has it (the push to `main`, or a nightly
  re-run of the Mac Intel leg);
- each workflow of class `after-landing` (Miri and loom): its newest run; when its
  changes job said `run=false` it is reported as not run.

Exit codes: `0` every item green; `4` an item is red, and the line reads
`REVERT OR FIX NOW: <workflow>: <job> ... run <id>`; `3` an item had not finished within
3 hours (pending, with the run ids); `1` any other error. A missing run or job, or a
cancelled one, is pending, never green.

`scripts/ci_triggers_check.py` compares this table with what `scripts/land.py` derives:

<!-- ci-triggers:after-landing-wait:begin -->
| item | class |
|---|---|
| ci.yml: Test (macos-latest) | after-landing |
| ci.yml: Determinism tests (release, macos-latest) | after-landing |
| ci.yml: Test (macos-15-intel) | main-after-landing |
| unsafe-and-parallel.yml | after-landing |
<!-- ci-triggers:after-landing-wait:end -->

## Dropped runs

GitHub keeps one pending run per concurrency group: while a run of a group is in
progress, a newer run waits, and a third one replaces the waiting one. The run of a
middle SHA can therefore be dropped without a result.

For the landing check, an expected workflow with **no run** for the `ci/<id>` SHA, or with
a **cancelled** run, is **no evidence** (neither green nor red). Re-create the run and wait
for it:

```sh
gh workflow run <workflow>.yml --ref ci/<id>   # or: gh run rerun <run-id>
```

A manual run always runs every job (its changes job outputs `run=true`), so the `Gate`
job of a path-gated workflow then needs every job to succeed. Bench gate compares a
manual run with `merge-base(origin/main)` (a manual run has no previous SHA). The
workflows that can be dispatched are listed below; every landing and after-landing
workflow must be among them (`scripts/ci_triggers_check.py` fails otherwise):

<!-- ci-triggers:dispatchable:begin -->
| workflow |
|---|
| `bench-gate.yml` |
| `ci.yml` |
| `downstream.yml` |
| `eol.yml` |
| `fuzz.yml` |
| `g2-comparisons.yml` |
| `integration-status.yml` |
| `oracle-status.yml` |
| `public-text.yml` |
| `python-binding.yml` |
| `quality-deep.yml` |
| `reexport_check.yml` |
| `security-audit.yml` |
| `unsafe-and-parallel.yml` |
| `wiring-status.yml` |
<!-- ci-triggers:dispatchable:end -->

## Concurrency

- A workflow that runs on `ci/**` never lets a `ci/**` push cancel a `main` run: its
  concurrency group includes the ref, or `cancel-in-progress` is false
  (`scripts/ci_triggers_check.py`). The light workflows use a group per ref and cancel
  an older run of the same branch only off `main`.
- `bench-gate.yml`, `unsafe-and-parallel.yml` and `integration-status.yml` use two
  groups per workflow, `main` and `ci`, with `cancel-in-progress: false`: at most one
  `main` run and one `ci/**` run at a time, and a new run waits instead of cancelling
  the running one (GitHub keeps one waiting run per group, so a newer push replaces an
  older waiting run). `scripts/ci_load_check.py` requires exactly this shape.
- `g2-comparisons.yml` (four `cargo check` runs) uses a group per SHA on `main` and a
  group per ref elsewhere: a `main` run is never cancelled, and is never dropped by the
  one-pending-run-per-group rule; a newer push to a `ci/**` branch cancels the older run
  of that branch.
- `quality-deep.yml` (the weekly mutation testing) does not trigger on push.

## Amendment, 2026-10-12

The rule amended: heavy jobs stay off every `ci/**` push, and their concurrency group
does not include the ref. Reason: the landing evidence on a `ci/**` branch was missing
8 workflows (their push triggers were `main`-only or path-filtered), and `main` went red
in Security & Hygiene (an unused dependency and the line-coverage ratchet) after a
landing. The rule now reads: a heavy job may run on a `ci/**` push only behind a path
gate (Miri and loom behind a gate narrowed to the unsafe and parallel code, checked by
`scripts/ci_load_check.py`), and concurrency limits the parallel heavy runs to one
`main` run and one `ci/**` run per workflow instead of keeping them off `ci/**`.
The weekly mutation testing stays off push.
