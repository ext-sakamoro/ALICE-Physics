#!/usr/bin/env python3
"""scripts/preflight.sh and the landing workflows run the same commands, both ways.

preflight.sh is the local copy of CI that gates a push; CI is what a landing is judged
by. A command one side runs and the other does not is a gate that only half the time
exists: a CI command missing locally is found after the push (the parallel-gated tests of
29 test files were run by CI only), and a preflight command missing from CI disappears
once landing is decided by CI evidence alone.

Compared: the cargo commands that build or check (`cargo [+toolchain] test | clippy |
build | check | doc | rustc | deny | fmt | public-api | mutants`) and the repository's
scripts (`python3 scripts/<x>.py`, `scripts/<x>.sh`), in scripts/preflight.sh and in the
`run:` steps of every workflow .github/ci-evidence.toml classes as `landing`. Each shell
line is split into its commands (`&&`, `||`, `;`, the first command of a pipeline,
`if` / `then` / `!` / `(` dropped, redirections dropped). A command is normalised before
it is compared:

  * the variables preflight.sh defines (`NATIVE=...`) are expanded, also inside a longer
    word (`"$NATIVE,neural"`);
  * the assignments that change what is built (BUILD_ENV) are part of the command, from
    a `VAR=value` prefix or from the `env:` of the step, the job or the workflow, in any
    order; other assignments (colour, network retries) are not;
  * the directory it runs in is part of it: a workflow step's `working-directory:` or a
    `cd <dir>` earlier in the same run block, a `cd <dir> &&` earlier on the same line;
  * quotes, `--no-fail-fast`, `--quiet`, `--locked` and repeated spaces are dropped,
    `python` is `python3` and `bash scripts/x.sh` is `scripts/x.sh`.

EXEMPT lists the CI commands preflight does not run (each with the reason: CI plumbing or
a check that needs the CI context); LOCAL_ONLY lists the preflight commands no landing
workflow runs, each `non-gating` (it decides nothing a CI command does not) or `gating`
with the reason. A gating local-only command is a gate that vanishes when landing is
judged by CI evidence: their count is printed, and `--opt-in` fails while it is not 0.

Fails (exit 1) on a command missing on either side and not listed, on a listed command
that no longer occurs or now has a counterpart (a stale entry), and when either side has
no command (a parse that finds nothing must not pass).

Usage: `python3 scripts/preflight_ci_parity.py [--opt-in]` (from the repository root).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# CI commands that preflight.sh does not run, with the reason (exact normalised commands)
EXEMPT: dict[str, str] = {
    "python3 scripts/intel_nightly_gate.py":
        "reads the GitHub runs of main to decide whether tonight's Intel leg is needed",
    "CARGO_PROFILE_BENCH_STRIP=false RUSTFLAGS='--cfg alice_instruction_bench' python3 scripts/bench_counts_check.py bench.txt":
        "reads the instruction counts of the bench run (valgrind on the Linux runner)",
    "python3 scripts/line_coverage_ratchet.py --summary coverage-summary.txt":
        "reads the llvm-cov summary the coverage job writes",
    "python3 scripts/semver_baseline_has.py $base $pkg":
        "reads the published baseline the semver-checks job fetches",
    "scripts/public_text.sh $base":
        "the pushed range's base; preflight runs the same script with the base defaulted to the merge-base",
    "cargo hack build --depth 2 --exclude-features python,neural,analytics,wasm --feature-powerset --lib":
        "cost (every pair of features); CI runs it on every ci/** push and is the landing evidence after opt-in",
    "[fuzz] cargo +nightly build --bins":
        "cost (the nightly build of every fuzz target); CI runs it on every ci/** push its path gate selects "
        "and is the landing evidence after opt-in",
}

# the EXEMPT commands left out of preflight for their cost alone: the exemption holds only while
# a landing workflow runs them on a ci/** push, so the `if:` of each job that runs one must be
# one of LANDING_IFS (none, the schedule exclusion, or the workflow's path gate)
EXEMPT_FOR_COST = (
    "cargo hack build --depth 2 --exclude-features python,neural,analytics,wasm --feature-powerset --lib",
    "[fuzz] cargo +nightly build --bins",
)
LANDING_IFS = ("", "github.event_name != 'schedule'", "needs.changes.outputs.run == 'true'")

# families of CI commands that preflight.sh does not run: a regular expression the whole
# normalised command matches, with the reason; each must match at least one CI command
EXEMPT_FAMILIES: dict[str, str] = {
    r"(?:\S+=(?:'[^']*'|\S+) )*python3 scripts/changed_paths\.py (?:decide|gate) .*":
        "the path gate of a workflow: decides from the push event which jobs run",
    r"scripts/downstream_check\.sh --(?:fetch-only|no-fetch)":
        "fetches and builds the published dependents against this tree",
    r"\[\S+\] cargo (?:check|test) .*\$\{patches\[@\]\}.*":
        "builds the packaged .crate unpacked in a temporary directory",
}

# preflight.sh commands that no landing workflow runs: (gating or non-gating, the reason,
# the normalised CI commands (regular expressions) that decide what it decides). A
# non-gating entry names at least one, and each must match a landing-workflow command, so
# "covered by CI" is checked, not claimed; a gating entry has none (it is what CI lacks).
LOCAL_ONLY: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "python3 scripts/preflight_tools_check.py --only targets": (
        "non-gating", "checks this machine has the targets the later steps build for",
        (r"cargo build --lib --target wasm32-unknown-unknown",)),
    "python3 scripts/ledgers.py --check": (
        "non-gating", "regenerates every ledger at once; CI checks each generator on its own",
        (r"python3 scripts/integration_levels\.py .*", r"python3 scripts/scip_reach\.py .*",
         r"python3 scripts/gen-oracle-status\.py.*", r"python3 scripts/gen-wiring-status\.py.*",
         r"python3 scripts/coverage_check\.py", r"python3 scripts/f32_modules\.py --check",
         r"python3 scripts/readme_sync\.py --check", r"python3 scripts/version_sync\.py --check",
         r"python3 scripts/fuzz_targets\.py --check", r"python3 scripts/binding_build_commands\.py --check",
         r"cargo \+nightly-2026-09-26 public-api .*")),
    "python3 scripts/ledgers.py --check --allow-missing-tools": (
        "non-gating", "the --quick / --fast form of ledgers.py --check (skips a generator whose tool is missing)",
        (r"python3 scripts/gen-oracle-status\.py.*", r"python3 scripts/gen-wiring-status\.py.*")),
    "cargo +1.85 check --lib": (
        "non-gating", "the --quick / --fast form of the MSRV lane CI tests",
        (r"cargo \+1\.85 test --lib",)),
    "cargo +1.87 check --features privacy-crypto --lib": (
        "non-gating", "the --quick / --fast form of the MSRV lane CI tests",
        (r"cargo \+1\.87 test --features privacy-crypto --lib",)),
    "cargo +1.87 check --features neural --lib": (
        "non-gating", "the --quick / --fast form of the MSRV lane CI tests",
        (r"cargo \+1\.87 test --features neural --lib",)),
    "cargo +nightly-2026-09-26 public-api --version": (
        "non-gating", "probes whether cargo-public-api is installed before the snapshot step",
        (r"cargo \+nightly-2026-09-26 public-api --features \S+ --simplified",)),
    "python3 scripts/affected_tests.py --features std,simd,parallel,ffi,gpu-solver-bridge --run": (
        "non-gating", "runs the tests of the changed modules only (--fast); CI runs the whole suites",
        (r"cargo test", r"cargo test --features parallel")),
}

# the assignments that change what a command builds or checks
BUILD_ENV = re.compile(r"^(?:RUSTFLAGS|RUSTDOCFLAGS|CLIPPY_CONF_DIR|CARGO_PROFILE_\w+|CARGO_BUILD_\w+|CARGO_TARGET_\w+_RUSTFLAGS"
                       r"|CARGO_ENCODED_RUSTFLAGS|MIRIFLAGS)$")
ASSIGN = re.compile(r"""([A-Z_][A-Z0-9_]*)=("[^"]*"|'[^']*'|\S*)""")
LEADING = re.compile(r"""^((?:[A-Z_][A-Z0-9_]*=(?:"[^"]*"|'[^']*'|\S*)\s+)*)(.*)$""", re.S)
KIND = re.compile(r"^(?:cargo (?:\+\S+ )?(?:test|clippy|build|check|doc|rustc|deny|fmt|public-api|mutants|hack)\b"
                  r"|python3 scripts/[\w.-]+\.py\b|scripts/[\w./-]+\.sh\b)")
REDIRECT = re.compile(r"""\s*(?:\d?>>?&?\s*\S+|<<<\s*("[^"]*"|\S+)|<\s*\S+)""")


def split_commands(line: str) -> list[str]:
    """the simple commands of one shell line (first command of each pipeline)"""
    out, cur, i, q = [], "", 0, ""
    while i < len(line):
        ch = line[i]
        if q:
            cur += ch
            if ch == q:
                q = ""
        elif ch in "'\"":
            q = ch
            cur += ch
        elif line.startswith(("&&", "||"), i):
            out.append(cur)
            cur = ""
            i += 1
        elif ch == ";":
            out.append(cur)
            cur = ""
        elif ch == "|":
            out.append(cur)
            cur = ""
            # the rest of the pipeline reads this command's output (tee, grep): not a command run here
            while i + 1 < len(line) and not line.startswith(("&&", "||", ";"), i + 1):
                i += 1
        elif ch == "#" and (not cur or cur[-1] in " \t"):
            break
        else:
            cur += ch
        i += 1
    out.append(cur)
    cmds = []
    for c in out:
        c = c.strip()
        while True:
            n = re.sub(r"^(?:if|then|else|elif|do|while|until|!|time|\(|\{)\s+|^\(", "", c).strip()
            if n == c:
                break
            c = n
        c = re.sub(r"\s*\)+\s*$|\s*\}\s*$", "", c).strip()
        if c:
            cmds.append(c)
    return cmds


def expand(text: str, var: dict[str, str]) -> str:
    for k in sorted(var, key=len, reverse=True):
        text = text.replace(f"${{{k}}}", var[k])
        text = re.sub(rf"\${k}(?![A-Za-z0-9_])", lambda _m, v=var[k]: v, text)
    return text


def quote(v: str) -> str:
    return f"'{v}'" if re.search(r"\s", v) else v


def normalise(cmd: str, env: dict[str, str], cwd: str = "", var: dict[str, str] | None = None) -> str | None:
    """the comparable form of a command, or None when it is not of a compared kind"""
    cmd = expand(cmd, var or {})
    m = LEADING.match(cmd.strip())
    pairs = dict(env)
    for k, v in ASSIGN.findall(m.group(1)):
        pairs[k] = v
    body = REDIRECT.sub("", m.group(2))
    body = body.replace('"', "").replace("'", "")
    body = re.sub(r"^python\s", "python3 ", body)
    body = re.sub(r"^bash\s+(scripts/)", r"\1", body)
    body = re.sub(r"^\./(scripts/)", r"\1", body)
    body = re.sub(r"\s--(?:no-fail-fast|quiet|locked)\b", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not KIND.match(body):
        return None
    body = canonical_args(body)
    pre = " ".join(f"{k}={quote(v.strip(chr(34) + chr(39)))}" for k, v in sorted(pairs.items()) if BUILD_ENV.match(k))
    return (f"[{cwd}] " if cwd and cwd != "." else "") + (pre + " " if pre else "") + body


# cargo options that take a value: the option and its value are one unit when sorting
VALUE_OPTIONS = {"--features", "-F", "--target", "--test", "--bench", "--bin", "--example", "-p", "--package",
                 "--manifest-path", "--profile", "--exclude-features", "--depth", "--target-dir", "--file", "-f",
                 "--color", "--message-format"}


def canonical_args(body: str) -> str:
    """a cargo command with its options before ` -- ` in one order (cargo reads them in any order;
    `--test a --test b` and `--test b --test a` build the same), the part after ` -- ` kept as written"""
    if not body.startswith("cargo "):
        return body
    head, sep, tail = body.partition(" -- ")
    words = head.split(" ")
    i = 2 if words[1].startswith("+") else 1
    if words[i] == "hack" and len(words) > i + 1:
        i += 1  # `cargo hack <subcommand>`
    lead, rest = words[: i + 1], words[i + 1:]
    units, k = [], 0
    while k < len(rest):
        w = rest[k]
        if w in VALUE_OPTIONS and k + 1 < len(rest):
            units.append(w + " " + rest[k + 1])
            k += 2
            continue
        units.append(w)
        k += 1
    return " ".join(lead + sorted(units)) + (sep + tail if sep else "")


def shell_lines(text: str) -> list[str]:
    """the lines of a shell text with continuations joined and comment lines dropped"""
    text = re.sub(r"\\\n\s*", " ", text)
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def commands_of_block(block: str, env: dict[str, str], cwd: str, var: dict[str, str]) -> list[tuple[str, str]]:
    """(raw, normalised) of the compared commands of a shell block; `cd <dir>` alone on a line
    changes the directory of the lines after it, `cd <dir> &&` the commands after it on its line"""
    out = []
    for line in shell_lines(block):
        here = cwd
        for c in split_commands(line):
            m = re.match(r"^cd\s+(\S+)$", c)
            if m:
                here = os.path.normpath(os.path.join(here or ".", m.group(1).replace('"', "").replace("'", "")))
                continue
            n = normalise(c, env, here, var)
            if n:
                out.append((c, n))
        if len(split_commands(line)) == 1 and (m := re.match(r"^cd\s+(\S+)$", line.strip())):
            cwd = here
    return out


def preflight_commands(text: str) -> tuple[list[tuple[str, str]], dict[str, str]]:
    var = dict(re.findall(r"^([A-Z_][A-Z0-9_]*)='([^']*)'", text, re.M))
    var.update(re.findall(r'^([A-Z_][A-Z0-9_]*)="([^"$]*)"', text, re.M))
    return commands_of_block(text, {}, "", var), var


def indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def env_map(lines: list[str], i: int) -> tuple[dict[str, str], int]:
    """the literal `KEY: value` pairs of the `env:` mapping at lines[i]"""
    base, out, j = indent(lines[i]), {}, i + 1
    while j < len(lines) and (not lines[j].strip() or indent(lines[j]) > base):
        m = re.match(r"^\s*([A-Z_][A-Z0-9_]*):\s*(.*)$", lines[j])
        if m and "${{" not in m.group(2):
            out[m.group(1)] = m.group(2).strip().strip("\"'")
        j += 1
    return out, j


def block_scalar(lines: list[str], i: int, first: str) -> tuple[str, int]:
    """the text of `run: <first>` at lines[i]: a plain or quoted scalar, `|` or `>` block"""
    first = first.strip()
    if first[:1] in "|>":
        j, body = i + 1, []
        while j < len(lines) and (not lines[j].strip() or indent(lines[j]) > indent(lines[i]) - (2 if lines[i].lstrip().startswith("- ") else 0)):
            body.append(lines[j])
            j += 1
        while body and not body[-1].strip():
            body.pop()
        pad = min((indent(l) for l in body if l.strip()), default=0)
        body = [l[pad:] for l in body]
        if first[0] == ">":
            return " ".join(l.strip() for l in body if l.strip()), j
        return "\n".join(body), j
    if first[:1] in "'\"" and first.endswith(first[0]) and len(first) > 1:
        first = first[1:-1].replace("''", "'") if first[0] == "'" else first[1:-1]
    return first, i + 1


def workflow_commands(text: str) -> list[tuple[str, str, str]]:
    """(job, raw, normalised) of the compared commands of every `run:` step of a workflow"""
    lines = text.splitlines()
    out: list[tuple[str, str, str]] = []
    wf_env: dict[str, str] = {}
    job, job_env, in_jobs = "", {}, False
    step_env: dict[str, str] = {}
    step_cwd, step_runs = "", []
    step_indent = -1

    def flush():
        for run in step_runs:
            for raw, n in commands_of_block(run, {**wf_env, **job_env, **step_env}, step_cwd, {}):
                out.append((job, raw, n))

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        if indent(line) == 0:
            in_jobs = s == "jobs:"
            if s == "env:":
                wf_env, i = env_map(lines, i)
                continue
            i += 1
            continue
        if in_jobs and indent(line) == 2 and re.match(r"^[\w-]+:\s*$", s):
            flush()
            step_runs, step_env, step_cwd, step_indent = [], {}, "", -1
            job, job_env = s[:-1], {}
            i += 1
            continue
        if in_jobs and indent(line) == 4 and s == "env:":
            job_env, i = env_map(lines, i)
            continue
        if in_jobs and s.startswith("- ") and (step_indent < 0 or indent(line) <= step_indent):
            flush()
            step_runs, step_env, step_cwd = [], {}, ""
            step_indent = indent(line)
        key = re.match(r"^(?:-\s+)?(run|env|working-directory):\s*(.*)$", s)
        if in_jobs and key:
            if key.group(1) == "run":
                body, i = block_scalar(lines, i, key.group(2))
                step_runs.append(body)
                continue
            if key.group(1) == "env":
                step_env, i = env_map(lines, i)
                continue
            step_cwd = key.group(2).strip().strip("\"'")
        i += 1
    flush()
    return out


def job_conditions(text: str) -> dict[str, str]:
    """the `if:` of each job of a workflow ("" when it has none)"""
    out: dict[str, str] = {}
    job, in_jobs = "", False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if indent(line) == 0:
            in_jobs = s == "jobs:"
            continue
        if in_jobs and indent(line) == 2 and re.match(r"^[\w-]+:\s*$", s):
            job = s[:-1]
            out[job] = ""
        elif in_jobs and job and indent(line) == 4 and s.startswith("if:"):
            c = s[3:].strip()
            if len(c) > 1 and c[0] in "'\"" and c[-1] == c[0]:
                c = c[1:-1]
            out[job] = c.removeprefix("${{").removesuffix("}}").strip()
    return out


def landing_workflows(root: str) -> list[str]:
    with open(os.path.join(root, ".github", "ci-evidence.toml"), "rb") as f:
        classes = tomllib.load(f).get("workflows", {})
    return sorted(w for w, c in classes.items() if c == "landing")


def check(root: str = ROOT, exempt: dict[str, str] | None = None,
          local_only: dict[str, tuple[str, str, tuple[str, ...]]] | None = None,
          families: dict[str, str] | None = None,
          for_cost: tuple[str, ...] | None = None) -> tuple[list[str], dict[str, int]]:
    exempt = EXEMPT if exempt is None else exempt
    for_cost = EXEMPT_FOR_COST if for_cost is None else for_cost
    families = EXEMPT_FAMILIES if families is None else families
    local_only = LOCAL_ONLY if local_only is None else local_only
    with open(os.path.join(root, "scripts", "preflight.sh"), encoding="utf-8") as f:
        local, var = preflight_commands(f.read())
    ci: list[tuple[str, str, str, str]] = []
    conditions: dict[tuple[str, str], str] = {}
    for wf in landing_workflows(root):
        with open(os.path.join(root, ".github", "workflows", wf), encoding="utf-8") as f:
            text = f.read()
        ci += [(wf, job, raw, expand(n, var)) for job, raw, n in workflow_commands(text)]
        conditions.update({(wf, j): c for j, c in job_conditions(text).items()})
    local_n = {n for _, n in local}
    ci_n = {n for *_, n in ci}
    errors = []
    if not ci:
        errors.append("the landing workflows have no compared command (compared nothing)")
    if not local:
        errors.append("scripts/preflight.sh has no compared command (compared nothing)")
    seen: set[str] = set()
    family_hits = {f: 0 for f in families}
    for wf, job, raw, n in ci:
        fam = [f for f in families if re.fullmatch(f, n)]
        for f in fam:
            family_hits[f] += 1
        if n in local_n or n in exempt or fam or n in seen:
            continue
        seen.add(n)
        errors.append(f"{wf} / {job} runs `{raw}` but scripts/preflight.sh does not "
                      f"(add it, or list `{n}` in EXEMPT with the reason)")
    for raw, n in local:
        if n in ci_n or n in local_only or n in seen:
            continue
        seen.add(n)
        errors.append(f"scripts/preflight.sh runs `{raw}` but no landing workflow does "
                      f"(add it to one, or list `{n}` in LOCAL_ONLY with the reason)")
    for n in exempt:
        if n not in ci_n:
            errors.append(f"EXEMPT `{n}`: no landing workflow runs it (a stale entry)")
        elif n in local_n:
            errors.append(f"EXEMPT `{n}`: scripts/preflight.sh runs it now (a stale entry)")
    for f, hits in family_hits.items():
        if len(families[f].strip()) < 12:
            errors.append(f"EXEMPT_FAMILIES `{f}`: needs a reason")
        if not hits:
            errors.append(f"EXEMPT_FAMILIES `{f}`: matches no landing-workflow command (a stale entry)")
    for n, reason in exempt.items():
        if len(reason.strip()) < 12:
            errors.append(f"EXEMPT `{n}`: needs a reason")
    for n in for_cost:
        if n not in exempt:
            errors.append(f"EXEMPT_FOR_COST `{n}`: not in EXEMPT")
        for wf, job, _raw, c in ci:
            if c == n and conditions.get((wf, job), "") not in LANDING_IFS:
                errors.append(f"EXEMPT `{n}`: {wf} / {job} runs it only `if: {conditions[(wf, job)]}`, which can "
                              "skip it on a ci/** push (left out of preflight for its cost, CI must run it)")
    for n, (kind, reason, covered) in local_only.items():
        if kind not in ("gating", "non-gating") or len(reason.strip()) < 12:
            errors.append(f"LOCAL_ONLY `{n}`: needs gating or non-gating and a reason")
        if kind == "non-gating" and not covered:
            errors.append(f"LOCAL_ONLY `{n}`: a non-gating entry names the CI commands that decide what it decides")
        if kind == "gating" and covered:
            errors.append(f"LOCAL_ONLY `{n}`: a gating entry is one CI lacks, it names no CI command")
        for c in covered:
            if not any(re.fullmatch(c, x) for x in ci_n):
                errors.append(f"LOCAL_ONLY `{n}`: no landing workflow runs `{c}` (the claimed counterpart)")
        if n not in local_n:
            errors.append(f"LOCAL_ONLY `{n}`: scripts/preflight.sh does not run it (a stale entry)")
        elif n in ci_n:
            errors.append(f"LOCAL_ONLY `{n}`: a landing workflow runs it now (a stale entry)")
    gating = sum(1 for k, *_ in local_only.values() if k == "gating")
    return errors, {"ci": len(ci), "preflight": len(local), "exempt": len(exempt) + len(families),
                    "local_only": len(local_only), "gating_local_only": gating}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--opt-in", action="store_true",
                    help="also fail while a gating command runs in preflight.sh only")
    a = ap.parse_args(argv)
    errors, n = check()
    if a.opt_in and n["gating_local_only"]:
        errors.append(f"{n['gating_local_only']} gating command(s) run in scripts/preflight.sh only "
                      "(landing by CI evidence would drop them)")
    for e in errors:
        print(f"error: {e}")
    print(f"compared: {n['ci']} landing-workflow and {n['preflight']} preflight commands; "
          f"{n['exempt']} CI-only (EXEMPT), {n['local_only']} local-only, of them gating: {n['gating_local_only']}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
