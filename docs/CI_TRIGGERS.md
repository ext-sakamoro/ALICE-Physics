# CI triggers and landing evidence

A change lands in two steps: `scripts/land.py` pushes the commits to a `ci/<id>`
branch, waits for its CI, then fast-forwards `main` to the same SHA. The runs on the
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
- the last job, `Gate`, fails when `run=true` and a heavy job was skipped, cancelled or
  failed, and when `run=false` and a heavy job ran.

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
| `ci.yml` | `main, ci/**` | - | `main` | - | yes | runs | landing |
| `downstream.yml` | `main, ci/**` | `.github/paths/downstream.txt` | `main (paths)` | - | yes | runs when the paths match | landing |
| `fuzz.yml` | `main, ci/**` | `.github/paths/fuzz.txt` | `main (paths)` | yes | yes | runs when the paths match | landing |
| `integration-status.yml` | `main, ci/**` | `.github/paths/integration-status.txt` | - | - | yes | runs when the paths match | landing |
| `oracle-status.yml` | `main, ci/**` | `.github/paths/oracle-status.txt` | - | - | yes | runs when the paths match | landing |
| `python-binding.yml` | `main, ci/**` | `.github/paths/python-binding.txt` | `main (paths)` | - | yes | runs when the paths match | landing |
| `quality-deep.yml` | - | - | `main (paths)` | yes | yes | - | schedule-only |
| `reexport_check.yml` | `main, ci/**` | - | `main` | - | yes | runs | landing |
| `release.yml` | `tags v*` | - | - | - | - | - | tag-only |
| `security-audit.yml` | `main, ci/**` | `.github/paths/security-audit.txt` | `main (paths)` | yes | yes | runs when the paths match | landing |
| `unsafe-and-parallel.yml` | `main, ci/**` | `.github/paths/unsafe-and-parallel.txt` | - | - | yes | runs when the paths match | after-landing |
| `wiring-status.yml` | `main, ci/**` | `.github/paths/wiring-status.txt` | - | - | yes | runs when the paths match | landing |
<!-- ci-triggers:workflows:end -->

## After-landing legs of landing workflows

The macOS legs of the test matrix take the longest. The landing does not wait for
either of them: they start with the rest of the run, and their result is read after the
landing. A release (crates.io, PyPI) needs every leg green.

The `macos-latest` leg is the arm64 leg that holds the golden hashes: a red result there
after a landing is fixed or reverted at once.

<!-- ci-triggers:after-landing-legs:begin -->
| workflow | job | matrix |
|---|---|---|
| `ci.yml` | `test` | `macos-latest` |
| `ci.yml` | `test` | `macos-15-intel` |
<!-- ci-triggers:after-landing-legs:end -->

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
| `fuzz.yml` |
| `integration-status.yml` |
| `oracle-status.yml` |
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
