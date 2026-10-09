#!/usr/bin/env python3
"""Run the `#[ignore]`d tests and check each one against what its reason claims.

`#[ignore]` carries two incompatible meanings in this repository, and the reason
string is what tells them apart:

| reason prefix                                     | meaning                        | expected |
|---------------------------------------------------|--------------------------------|----------|
| `runtime:` / `runtime only:`                      | passes, but too slow for CI    | **pass** |
| `src gap:` / `src bug:` / `the red is correct`    | does not pass yet (a target)   | **fail** |
| `manual:`                                         | passes, but longer than the job | listed, not run |
| anything else                                     | not yet triaged                | reported |

So a single `--ignored` run cannot be read as "green or red": a `runtime:` test
that fails is a regression, while a `src gap:` test that *passes* is the reversal
condition its doc comment asks about — the gap has been closed and the twin
convention now needs the two attributes flipped. This script separates the two
and reports them differently:

* `runtime:` fails            → `::error` and a non-zero exit (the job goes red)
* `src gap:` passes           → `::notice` and a prominent summary section, exit 0
* `src gap:` fails            → as documented, nothing to do
* `manual:`                   → listed and skipped (`--skip`): run it by hand, it must pass
* anything else               → listed with its outcome, no verdict
* table/run mismatch          → `::warning` (the expectation table has drifted)
* a run that checked nothing  → `::error`: an empty table, an axis that built no
  test binary, a binary that crashed before its `test result:` line, or a
  `runtime:` test that did not run

The expectation table is derived from the source at run time rather than stored
in a file, so it cannot drift on its own; what can drift is this parser's view of
the source, which is why a test that runs without a table entry (or an entry with
no run) is reported rather than ignored.

The feature sets come from `scripts/preflight.sh` (`NATIVE`) instead of being
written out here: regenerating anything with a feature set that is one feature
away from preflight's has cost this repository a 126-line public-API deletion.

usage:
  scripts/run_ignored.py           # derive the table, build, run, report
  scripts/run_ignored.py --list    # print the table only (no build, no run)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RUNTIME_PREFIXES = ("runtime:", "runtime only:")
EXPECTED_RED_PREFIXES = ("src gap:", "src bug:", "the red is correct")
MANUAL_PREFIXES = ("manual:",)

from ansi import strip as strip_escapes  # noqa: E402  (libtest colours its words with --color always)

CAT_RUNTIME = "runtime"
CAT_EXPECTED_RED = "expected-red"
CAT_UNTRIAGED = "untriaged"
CAT_MANUAL = "manual"

FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)")
RESULT_RE = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED|ignored)\s*$")
SUMMARY_RE = re.compile(r"^test result:\s+(\w+)\.\s+(\d+) passed;\s+(\d+) failed;")


# ---------------------------------------------------------------- source table


def _scan_attribute(text: str, start: int) -> tuple[str, int] | None:
    """Return (attribute source, index just past it) for the `#[...]` at `start`.

    Brackets inside string literals are skipped, so a reason containing `]` does
    not truncate the attribute.
    """
    open_at = text.find("[", start)
    if open_at < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    i = open_at
    while i < len(text):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
        i += 1
    return None


def _unescape(s: str) -> str:
    """Decode the escapes a Rust string literal may carry.

    A backslash before a newline continues the literal and eats the indentation
    of the next line, which is how the longer reasons in this repository wrap.
    """
    simple = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'", "0": "\0"}
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if nxt == "\n":
            i += 2
            while i < len(s) and s[i] in " \t":
                i += 1
            continue
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
            continue
        i += 1  # unknown escape: drop the backslash, keep going
    return "".join(out)


def _reason_of(attr: str) -> str:
    """Pull the reason literal out of `#[ignore = "..."]`; `""` if there is none."""
    q = attr.find('"')
    if q < 0:
        return ""
    i = q + 1
    esc = False
    while i < len(attr):
        c = attr[i]
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            return _unescape(attr[q + 1 : i])
        i += 1
    return ""


def classify(reason: str) -> str:
    low = reason.strip().lower()
    if low.startswith(RUNTIME_PREFIXES):
        return CAT_RUNTIME
    if low.startswith(EXPECTED_RED_PREFIXES):
        return CAT_EXPECTED_RED
    if low.startswith(MANUAL_PREFIXES):
        return CAT_MANUAL
    return CAT_UNTRIAGED


def _binary_of(path: Path) -> str:
    """The cargo test target a file's tests end up in."""
    rel = path.relative_to(REPO)
    if rel.parts[0] == "tests":
        return rel.stem
    return "<lib>"


def collect_table() -> tuple[list[dict], list[str]]:
    """Derive the expectation table from every `#[ignore]` attribute in the tree."""
    entries: list[dict] = []
    notes: list[str] = []
    files = sorted(REPO.glob("tests/**/*.rs")) + sorted(REPO.glob("src/**/*.rs"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"(?m)^[ \t]*#\[\s*ignore\b", text):
            hash_at = text.index("#", m.start())
            scanned = _scan_attribute(text, hash_at)
            if scanned is None:
                notes.append(f"{path.relative_to(REPO)}: unterminated #[ignore] attribute")
                continue
            attr, after = scanned
            line = text.count("\n", 0, hash_at) + 1
            name = None
            for tail in text[after:].splitlines():
                stripped = tail.strip()
                if not stripped or stripped.startswith(("#[", "#![", "//", "/*", "*")):
                    continue
                fn = FN_RE.match(tail)
                name = fn.group(1) if fn else None
                break
            if name is None:
                notes.append(
                    f"{path.relative_to(REPO)}:{line}: #[ignore] with no `fn` after it"
                )
                continue
            reason = _reason_of(attr)
            entries.append(
                {
                    "file": str(path.relative_to(REPO)),
                    "line": line,
                    "binary": _binary_of(path),
                    "name": name,
                    "reason": reason,
                    "category": classify(reason),
                }
            )
    seen: dict[tuple[str, str], dict] = {}
    for e in entries:
        key = (e["binary"], e["name"])
        if key in seen:
            notes.append(
                f"ambiguous: {e['binary']} has two ignored tests named `{e['name']}` "
                f"({seen[key]['file']}:{seen[key]['line']} and {e['file']}:{e['line']}); "
                "outcomes for it cannot be attributed"
            )
        seen[key] = e
    return entries, notes


# ------------------------------------------------------------------- execution


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO, text=True, **kw)


def native_features() -> str:
    """The `NATIVE` feature set preflight uses, read from preflight.sh."""
    src = (REPO / "scripts" / "preflight.sh").read_text(encoding="utf-8")
    m = re.search(r"(?m)^NATIVE='([^']*)'", src)
    if not m:
        raise SystemExit("scripts/preflight.sh no longer defines NATIVE; refusing to guess")
    return m.group(1)


def build_binaries(extra: list[str]) -> list[tuple[str, str]]:
    """`cargo test --no-run` and return the (target name, executable) pairs."""
    cmd = ["cargo", "test", "--release", "--no-run", "--message-format=json", *extra]
    proc = _run(cmd, stdout=subprocess.PIPE)
    if proc.returncode != 0:
        raise SystemExit(f"build failed: {' '.join(cmd)} exited {proc.returncode}")
    out: list[tuple[str, str]] = []
    for raw in proc.stdout.splitlines():
        if not raw.startswith("{"):
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-artifact" or not msg.get("executable"):
            continue
        if not msg.get("profile", {}).get("test"):
            continue
        target = msg.get("target", {})
        kinds = target.get("kind", [])
        if any(k in ("bench", "example", "custom-build") for k in kinds):
            continue
        label = target.get("name", "?") if "test" in kinds else "<lib>"
        out.append((label, msg["executable"]))
    return out


def parse_outcomes(output: str) -> dict[str, str]:
    """`{test name: "ok" | "FAILED"}` from a libtest run's output."""
    outcomes: dict[str, str] = {}
    for raw in strip_escapes(output).splitlines():
        m = RESULT_RE.match(raw.strip())
        if m and m.group(2) != "ignored":
            outcomes[m.group(1).split("::")[-1]] = m.group(2)
    return outcomes


def crashed(returncode: int, output: str) -> bool:
    """Whether a test binary stopped before reporting: libtest exits 0 (all
    passed) or 101 (some failed) after its `test result:` line; anything else,
    or no such line, is a crash (signal, abort, killed) and the tests after the
    crashing one never ran."""
    reported = any(SUMMARY_RE.match(line.strip()) for line in strip_escapes(output).splitlines())
    return returncode not in (0, 101) or not reported


# A binary is killed after this many seconds and reported by name: one slow
# binary used to run the whole job into its 180-minute limit, which says nothing
# about which test hung. Override with RUN_IGNORED_BINARY_TIMEOUT (seconds).
BINARY_TIMEOUT = int(os.environ.get("RUN_IGNORED_BINARY_TIMEOUT", "1800"))
TIMED_OUT = -1000  # exit code run_binary returns for a binary it killed


def run_binary(
    label: str, exe: str, skip: list[str], timeout: int | None = None
) -> tuple[dict[str, str], str, int]:
    """Run only the ignored tests in one binary; return outcomes, output and exit code.

    `skip` names the `manual:` tests of this binary, passed as libtest `--skip`
    so they are neither run nor counted as missing. A binary that runs past
    `timeout` seconds (BINARY_TIMEOUT by default) is killed; the exit code is
    then TIMED_OUT and the outcomes are those reported before the kill.
    """
    cmd = [exe, "--ignored", "--test-threads=1"]
    for name in skip:
        cmd += ["--skip", name]
    try:
        proc = _run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    timeout=BINARY_TIMEOUT if timeout is None else timeout)
        out, rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as e:
        partial = e.stdout or ""
        out = partial if isinstance(partial, str) else partial.decode("utf-8", "replace")
        rc = TIMED_OUT
    outcomes = parse_outcomes(out)
    print(f"  [{label}] {len(outcomes)} ignored test(s) ran", flush=True)
    return outcomes, out, rc


# ---------------------------------------------------------------------- report


def emit(lines: list[str]) -> None:
    """Print the report, and also append it to the step summary when in CI.

    A copy goes to `target/ignored-report.md` (inside the already-ignored build
    directory, so it does not show up in `git status`) for the weekly run to
    upload: the classification is half of what this job produces, and a step
    summary is not something a later session can diff against.
    """
    body = "\n".join(lines) + "\n"
    sys.stdout.write(body)
    out = REPO / "target" / "ignored-report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(body)


def table_markdown(entries: list[dict]) -> list[str]:
    order = {CAT_RUNTIME: 0, CAT_EXPECTED_RED: 1, CAT_MANUAL: 2, CAT_UNTRIAGED: 3}
    lines = ["| category | test | where | reason (first 110 chars) |", "|---|---|---|---|"]
    for e in sorted(entries, key=lambda e: (order[e["category"]], e["file"], e["line"])):
        reason = e["reason"].replace("\n", " ").replace("|", "\\|")
        if len(reason) > 110:
            reason = reason[:107] + "..."
        lines.append(
            f"| `{e['category']}` | `{e['name']}` | `{e['file']}:{e['line']}` | {reason or '_(no reason)_'} |"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print the derived table and stop")
    args = ap.parse_args()

    entries, notes = collect_table()
    counts = {c: sum(1 for e in entries if e["category"] == c) for c in (CAT_RUNTIME, CAT_EXPECTED_RED, CAT_MANUAL, CAT_UNTRIAGED)}

    header = [
        "## `#[ignore]`d tests",
        "",
        f"{len(entries)} ignored test(s): "
        f"**{counts[CAT_RUNTIME]}** `runtime:` (must pass) / "
        f"**{counts[CAT_EXPECTED_RED]}** `src gap:` family (expected to fail) / "
        f"**{counts[CAT_MANUAL]}** `manual:` (longer than this job; listed, not run) / "
        f"**{counts[CAT_UNTRIAGED]}** untriaged (reported, no verdict)",
        "",
        *table_markdown(entries),
    ]
    if args.list:
        # stdout only: the run below owns the step summary, and writing the
        # table twice would put it in there twice.
        sys.stdout.write("\n".join(header) + "\n")
        for n in notes:
            print(f"note: {n}")
        return 0

    axes = [("default features", []), (f"--lib --features {native_features()}", ["--lib", "--features", native_features()])]

    # a run that checked nothing is a failure, not a quiet pass
    broken: list[str] = []
    if not entries:
        broken.append("no `#[ignore]` test was found in the source (the table is empty)")

    outcomes: dict[tuple[str, str], str] = {}
    logs: dict[str, str] = {}
    for axis_label, extra in axes:
        print(f"\n===== axis: {axis_label}", flush=True)
        binaries = build_binaries(extra)
        if not binaries:
            broken.append(f"axis `{axis_label}`: the build produced no test binary")
        for label, exe in binaries:
            skip = [e["name"] for e in entries if e["category"] == CAT_MANUAL and e["binary"] == label]
            got, log, rc = run_binary(label, exe, skip)
            if rc == TIMED_OUT:
                broken.append(
                    f"axis `{axis_label}`: `{label}` ran past {BINARY_TIMEOUT} s and was killed; "
                    "the ignored tests after the slow one did not run"
                )
            elif crashed(rc, log):
                broken.append(
                    f"axis `{axis_label}`: `{label}` stopped before reporting (exit {rc}); "
                    "the ignored tests after the crash did not run"
                )
            for name, verdict in got.items():
                outcomes[(label, name)] = verdict
                logs[f"{label}::{name}"] = log

    errors: list[dict] = []
    inversions: list[dict] = []
    as_documented: list[dict] = []
    untriaged: list[dict] = []
    manual: list[dict] = []
    missing: list[dict] = []

    for e in entries:
        verdict = outcomes.pop((e["binary"], e["name"]), None)
        row = dict(e, verdict=verdict)
        if e["category"] == CAT_MANUAL:
            # Skipped on purpose above; a verdict here would mean `--skip` failed.
            (manual if verdict is None else errors).append(row)
        elif verdict is None:
            missing.append(row)
            if e["category"] == CAT_RUNTIME:
                # its reason claims it passes; a claim that was not run is not checked
                broken.append(f"`runtime:` test did not run: `{e['name']}` ({e['file']}:{e['line']})")
        elif e["category"] == CAT_RUNTIME:
            (as_documented if verdict == "ok" else errors).append(row)
        elif e["category"] == CAT_EXPECTED_RED:
            (inversions if verdict == "ok" else as_documented).append(row)
        else:
            untriaged.append(row)

    report = list(header)

    if inversions:
        report += [
            "",
            "## ⚠️ reversal condition met — an expected-red test now passes",
            "",
            "These carry a `src gap:` / `src bug:` / `the red is correct` reason, which says"
            " they cannot pass yet. They pass. Read each one's doc comment: it states what to"
            " flip (usually: drop the `#[ignore]` here and put `#[ignore = \"superseded\"]` on"
            " its non-ignored twin, in the same diff).",
            "",
            "| test | where | reason |",
            "|---|---|---|",
        ]
        for r in inversions:
            reason = r["reason"].replace("\n", " ").replace("|", "\\|")
            report.append(f"| `{r['name']}` | `{r['file']}:{r['line']}` | {reason} |")
            print(f"::notice file={r['file']},line={r['line']}::reversal condition met: {r['name']} passes although its reason says it cannot")

    if errors:
        report += [
            "",
            "## ❌ a `runtime:` test failed",
            "",
            "A `runtime:` reason says the test passes and is only ignored for cost. These"
            " failed, so either the claim was wrong or something regressed.",
            "",
            "| test | where | reason |",
            "|---|---|---|",
        ]
        for r in errors:
            reason = r["reason"].replace("\n", " ").replace("|", "\\|")
            report.append(f"| `{r['name']}` | `{r['file']}:{r['line']}` | {reason} |")
            print(f"::error file={r['file']},line={r['line']}::runtime-only ignored test failed: {r['name']}")
        report += ["", "<details><summary>failure output</summary>", "", "```"]
        for r in errors:
            log = logs.get(f"{r['binary']}::{r['name']}", "")
            report += [line for line in log.splitlines() if r["name"] in line or line.startswith(("thread ", "assertion", "  left", "  right", "test result:"))][:40]
        report += ["```", "", "</details>"]

    if manual:
        report += ["", "### manual (longer than this job — skipped here, must pass when run by hand with `--exact`)", "", "| test | where | reason |", "|---|---|---|"]
        for r in sorted(manual, key=lambda r: r["file"]):
            reason = r["reason"].replace("\n", " ").replace("|", "\\|")
            report.append(f"| `{r['name']}` | `{r['file']}:{r['line']}` | {reason} |")

    if untriaged:
        report += ["", "### untriaged reasons (no verdict, listed for the backlog)", "", "| test | outcome | reason |", "|---|---|---|"]
        for r in sorted(untriaged, key=lambda r: r["file"]):
            reason = r["reason"].replace("\n", " ").replace("|", "\\|")
            report.append(f"| `{r['name']}` | {r['verdict']} | {reason} |")

    if as_documented:
        report += ["", f"### as documented ({len(as_documented)})", ""]
        for r in sorted(as_documented, key=lambda r: r["file"]):
            report.append(f"- `{r['name']}` — {r['verdict']} ({r['category']})")

    drift = list(notes)
    for r in missing:
        drift.append(
            f"in the table but did not run: `{r['name']}` ({r['file']}:{r['line']}) — "
            "feature-gated, renamed, or this parser mis-read the source"
        )
    for (binary, name), verdict in sorted(outcomes.items()):
        drift.append(f"ran but has no table entry: `{name}` in `{binary}` ({verdict})")
    if drift:
        report += ["", "### ⚠️ table drift", ""]
        for d in drift:
            report.append(f"- {d}")
            print(f"::warning::{d}")

    if broken:
        report += ["", "## ❌ the run did not check what it should", ""]
        for b in broken:
            report.append(f"- {b}")
            print(f"::error::{b}")

    emit(report)
    return 1 if errors or broken else 0


if __name__ == "__main__":
    sys.exit(main())
