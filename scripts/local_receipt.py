#!/usr/bin/env python3
"""The workflows' cheap steps, run here before a series goes to ci/**, and the receipt of that run.

A landing's cheap prepass does not build, so a red clippy line, a broken doc link or a stale
review table used to be found by the CI only, a full round later. This script reads every
`run:` step of every workflow (.github/workflows/*.yml) and sorts each one by the first rule
of scripts/local-receipt-rules.tsv that matches its command:

    local   the step is run here, with the env and working directory the workflow gives it
    ci      the step is the CI's (a test leg, a toolchain install, a step that reads the event);
            the rule says why

A step no rule matches is an error, so a step added to a workflow is run here from that day or
is written down as the CI's. A rule that matches no step is an error too.

    local_receipt.py steps     print every step with its class and rule
    local_receipt.py run       run the local steps on HEAD (a clean checkout), write the receipt
    local_receipt.py check     exit 0 only with a receipt of HEAD's code: every local step run, all green

The receipt is keyed by the blob list of HEAD without `docs/` and `*.md` (the ledgers a landing
regenerates are there, and none of them is compiled): a change of any other file, a workflow or
the rules among them, asks for a new run. No local step, or nothing read, is an error, not a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ".github/workflows"
RULES = "scripts/local-receipt-rules.tsv"
RUN = re.compile(r"^(\s*)(- )?run:\s*(.*)$")
PAIR = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*):\s*(.*?)\s*$")
NOT_COMPILED = ("docs/",)
CLASSES = ("local", "slow", "ci")
# a series that changes a file under one of these needs the `slow` steps too (the lib coverage ratchet)
SLOW_PATHS = ("src/",)
BASE = "origin/main"
# what the runner gives a step and this machine does not: a local step may not read it
RUNNER_ONLY = re.compile(r"\$\{\{|\$\{?GITHUB_|\$\{?RUNNER_")


class ReceiptError(Exception):
    pass


def indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def live(line: str) -> bool:
    return bool(line.strip()) and not line.lstrip().startswith("#")


def block(lines: list[str], start: int, depth: int) -> list[str]:
    """the lines after `start` that are deeper than `depth` (blank lines kept)"""
    out = []
    for line in lines[start + 1:]:
        if line.strip() and indent(line) <= depth:
            break
        out.append(line)
    return out


def unquote(value: str) -> str:
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'" else value


def env_at(lines: list[str], first: int, last: int, depth: int) -> dict[str, str]:
    """the pairs of every `env:` key at `depth` between two lines"""
    env: dict[str, str] = {}
    for k in range(first, last):
        if lines[k].strip() == "env:" and indent(lines[k]) == depth:
            for pair in block(lines, k, depth):
                m = PAIR.match(pair)
                if m and live(pair) and indent(pair) > depth:
                    env[m.group(1)] = unquote(m.group(2))
    return env


def steps(text: str, name: str) -> list[dict]:
    """every `run:` step of one workflow, in file order: its command (all lines), the env of
    the workflow, the job and the step (the later wins), its shell and working directory"""
    lines = text.splitlines()
    found: list[dict] = []
    for k, line in enumerate(lines):
        m = RUN.match(line)
        if not m or line.lstrip().startswith("#"):
            continue
        item = indent(line) if m.group(2) else None
        depth = indent(line) + (2 if m.group(2) else 0)   # where the step's keys are
        rest = m.group(3).strip()
        if rest[:1] in ("|", ">"):
            if rest not in ("|", "|-", "|+", ">", ">-", ">+"):
                raise ReceiptError(f"{name}:{k + 1}: a block scalar header this script does not read: {rest!r}")
            body = block(lines, k, depth)
            cut = min((indent(b) for b in body if b.strip()), default=0)
            if rest[0] == "|":   # the script as bash gets it: the block's own indent removed, nothing else
                cmd = "\n".join(b[cut:] for b in body).strip("\n")
            else:
                cmd = " ".join(b.strip() for b in body if b.strip())
        else:
            cmd = unquote(rest)
        if not cmd:
            raise ReceiptError(f"{name}:{k + 1}: a `run:` with no command")
        # the step: from its `- ` line to the next line at or above that indent
        first = k
        if item is None:
            while first > 0 and not (lines[first].lstrip().startswith("- ") and indent(lines[first]) == depth - 2):
                first -= 1
            item = depth - 2
        last = first + 1
        while last < len(lines) and not (live(lines[last]) and indent(lines[last]) <= item):
            last += 1
        # the job: the key two levels above the steps list
        job = first
        while job > 0 and not (live(lines[job]) and indent(lines[job]) < item - 2):
            job -= 1
        job_end = job + 1
        while job_end < len(lines) and not (live(lines[job_end]) and indent(lines[job_end]) <= indent(lines[job])):
            job_end += 1
        env = {**env_at(lines, 0, len(lines), 0), **env_at(lines, job, job_end, indent(lines[job]) + 2),
               **env_at(lines, first, last, depth)}
        attrs = {}
        for row in lines[first:last]:
            key = row.lstrip()
            p = PAIR.match(key[2:] if key.startswith("- ") else key)
            if p and live(row) and indent(row) + (2 if key.startswith("- ") else 0) == depth:
                attrs[p.group(1)] = unquote(p.group(2))
        found.append({"workflow": name, "line": k + 1, "cmd": cmd, "env": env,
                      "shell": attrs.get("shell", "bash"), "cwd": attrs.get("working-directory", "")})
    return found


def workflow_steps(root: Path) -> list[dict]:
    files = sorted((root / WORKFLOWS).glob("*.yml"))
    found: list[dict] = []
    for f in files:
        found += steps(f.read_text(encoding="utf-8"), f.name)
    if not found:
        raise ReceiptError(f"no `run:` step read from {len(files)} workflow file(s) of {root}: "
                           "nothing to sort is an error, not a pass")
    return found


def read_rules(text: str) -> list[dict]:
    rules = []
    for n, row in enumerate(text.splitlines(), 1):
        if not row.strip() or row.startswith("#"):
            continue
        cols = row.split("\t")
        if len(cols) != 3 or cols[0] not in CLASSES or len(cols[2].strip()) < 12:
            raise ReceiptError(f"{RULES}:{n}: a rule is `local|slow|ci<TAB>regex<TAB>reason (12 characters or more)`")
        try:
            rules.append({"class": cols[0], "re": re.compile(cols[1], re.M), "src": cols[1], "reason": cols[2], "n": n})
        except re.error as e:
            raise ReceiptError(f"{RULES}:{n}: {e}") from e
    if not rules:
        raise ReceiptError(f"{RULES}: no rule")
    return rules


def rule_text(step: dict) -> str:
    """what a rule's regex is searched in: the command, then one `env: KEY=value` line per pair"""
    return "\n".join([step["cmd"], *(f"env: {k}={v}" for k, v in sorted(step["env"].items()))])


def sort_steps(found: list[dict], rules: list[dict]) -> list[dict]:
    """each step with the first rule that matches it; errors for a step without a rule, a rule
    without a step, and a local step that reads what only the runner has"""
    used: set[int] = set()
    problems: list[str] = []
    out = []
    for s in found:
        rule = next((r for r in rules if r["re"].search(rule_text(s))), None)
        where = f"{s['workflow']}:{s['line']}"
        if rule is None:
            problems.append(f"{where}: no rule for this step (run it here: a `local` rule; the CI's: a `ci` rule "
                            f"with the reason): {s['cmd'].splitlines()[0][:100]}")
            continue
        used.add(rule["n"])
        if rule["class"] != "ci":
            reads = [v for v in (s["cmd"], s["cwd"], *s["env"].values()) if RUNNER_ONLY.search(v)]
            if reads or s["shell"] != "bash":
                problems.append(f"{where}: a `{rule['class']}` rule (line {rule['n']}) matches a step that reads the runner "
                                f"({(reads or [s['shell']])[0][:60]!r}): {s['cmd'].splitlines()[0][:80]}")
                continue
        out.append({**s, "class": rule["class"], "rule": rule["n"]})
    problems += [f"{RULES}:{r['n']}: the rule matches no step (stale): {r['src']}" for r in rules if r["n"] not in used]
    if problems:
        raise ReceiptError(f"{len(problems)} problem(s) sorting {len(found)} steps:\n  " + "\n  ".join(problems))
    return out


def local_steps(root: Path, cls: str = "local") -> list[dict]:
    """the steps of one class to run here, in the workflows' order, each command once (the
    first workflow that has it names it)"""
    rules = read_rules((root / RULES).read_text(encoding="utf-8"))
    seen: set[str] = set()
    out = []
    for s in sort_steps(workflow_steps(root), rules):
        unit = {"cmd": s["cmd"], "env": s["env"], "cwd": s["cwd"]}
        key = json.dumps(unit, sort_keys=True)
        if s["class"] == cls and key not in seen:
            seen.add(key)
            out.append(unit)
    if not out:
        raise ReceiptError(f"no {cls} step: nothing to run is an error, not a pass")
    return out


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        raise ReceiptError(f"git {' '.join(args)}: {r.stderr.strip()[-200:]}")
    return r.stdout


def compiled(path: str) -> bool:
    return not path.startswith(NOT_COMPILED) and not path.endswith(".md")


def code_key(root: Path) -> str:
    """the blobs of HEAD that a step can read (the workflows and the rules among them):
    what a receipt is a receipt of"""
    listed = [row for row in git(root, "ls-tree", "-r", "HEAD").splitlines() if compiled(row.split("\t", 1)[1])]
    if not listed:
        raise ReceiptError("HEAD lists no file")
    return hashlib.sha256("\n".join(listed).encode()).hexdigest()


def receipt_dir() -> Path:
    base = os.environ.get("LOCAL_RECEIPT_DIR")
    return Path(base) if base else Path.home() / ".cache" / "alice-siblings" / "local-receipts"


def receipt_path(key: str, cls: str = "local") -> Path:
    return receipt_dir() / (f"{key}.json" if cls == "local" else f"{key}.{cls}.json")


def needs_slow(root: Path, base: str | None = None) -> list[str]:
    """the files under SLOW_PATHS that HEAD changes since it left `base` (a base that cannot
    be read is an error: the answer is not "nothing changed")"""
    fork = git(root, "merge-base", base or BASE, "HEAD").strip()
    changed = git(root, "diff", "--name-only", fork, "HEAD").splitlines()
    return [f for f in changed if f.startswith(SLOW_PATHS)]


def shell(root: Path, logs: Path):
    """a step as the runner's bash runs it (`-e -o pipefail`), its output in logs/NNN.log; the
    workflows call `python`, which a machine with only `python3` does not have: then `python`
    is this interpreter, through a directory put first on PATH"""
    path = os.environ.get("PATH", "")
    if shutil.which("python") is None:
        shim = logs / "bin"
        shim.mkdir(exist_ok=True)
        if not (shim / "python").exists():
            (shim / "python").symlink_to(sys.executable)
        path = f"{shim}{os.pathsep}{path}"

    def runner(n: int, unit: dict) -> int:
        with open(logs / f"{n:03d}.log", "w", encoding="utf-8") as log:
            return subprocess.run(["bash", "-e", "-o", "pipefail", "-c", unit["cmd"]], cwd=root / unit["cwd"],
                                  env={**os.environ, "PATH": path, **unit["env"]}, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
    return runner


def run(root: Path, runner=None, only: str | None = None, cls: str = "local") -> tuple[Path | None, list[dict]]:
    """run every step of the class to the end (a red step does not stop the next) and write
    the receipt; with `only` (a regex) run the matching steps and write nothing"""
    dirty = [row for row in git(root, "status", "--porcelain", "--untracked-files=no").splitlines()
             if compiled(row[3:])]
    if dirty and only is None:
        raise ReceiptError(f"{len(dirty)} tracked file(s) differ from HEAD ({dirty[0][3:]} ...): "
                           "the receipt is of a commit, commit first")
    units = local_steps(root, cls)
    key = code_key(root)
    logs = receipt_dir() / (f"{key}.logs" if cls == "local" else f"{key}.{cls}.logs")
    if runner is None:
        logs.mkdir(parents=True, exist_ok=True)
        runner = shell(root, logs)
    ran = []
    for n, unit in enumerate(units, 1):
        if only is not None and not re.search(only, unit["cmd"]):
            continue
        t0 = time.monotonic()
        rc = runner(n, unit)
        print(f"local_receipt: {cls} {n}/{len(units)} rc={rc} {time.monotonic() - t0:.0f}s :: "
              f"{unit['cmd'].splitlines()[0][:110]}", flush=True)
        ran.append({**unit, "rc": rc})
    if only is not None:
        return None, ran
    path = receipt_path(key, cls)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "head": git(root, "rev-parse", "HEAD").strip(), "steps": ran,
                                "compared": len(ran)}, indent=1) + "\n", encoding="utf-8")
    return path, ran


def check_class(root: Path, cls: str) -> str:
    units = local_steps(root, cls)
    key = code_key(root)
    path = receipt_path(key, cls)
    flag = "" if cls == "local" else f" --{cls}"
    if not path.exists():
        raise ReceiptError(f"no {'' if cls == 'local' else cls + ' '}receipt for the code of HEAD ({key[:12]}): run `python3 scripts/local_receipt.py "
                           f"run{flag}` on this commit (the workflows' {len(units)} {cls} steps) before the landing")
    data = json.loads(path.read_text(encoding="utf-8"))
    ran = data.get("steps") or []
    if data.get("key") != key or [{k: r.get(k) for k in ("cmd", "env", "cwd")} for r in ran] != units:
        raise ReceiptError(f"the receipt {path.name} is not of these {len(units)} steps: run it again")
    red = [r["cmd"].splitlines()[0] for r in ran if r.get("rc") != 0]
    if red:
        raise ReceiptError(f"{len(red)} of {len(ran)} {cls} step(s) not green in the receipt: {red[0]}")
    return f"{cls} receipt: {len(ran)} of {len(units)} steps green on the code of HEAD ({key[:12]})"


def check(root: Path, base: str | None = None) -> str:
    """the line land.py prints, or ReceiptError: no receipt, another set of steps, a red step,
    nothing run; the `slow` receipt too when the series changes a file under SLOW_PATHS"""
    line = check_class(root, "local")
    touched = needs_slow(root, base)
    if not touched:
        return f"{line}; slow receipt not asked (no file under {'/'.join(SLOW_PATHS)} changed since {base or BASE})"
    return f"{line}; {check_class(root, 'slow')} ({len(touched)} file(s) under {' '.join(SLOW_PATHS)} changed)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("steps", "run", "check"))
    ap.add_argument("--only", help="run: a regex; only the matching steps are run and no receipt is written")
    ap.add_argument("--slow", action="store_true", help="run: the `slow` steps (the lib coverage ratchet, about "
                    "half an hour) in place of the local ones; their receipt is asked when the series changes src/")
    ap.add_argument("--base", default=BASE, help=f"check: the branch the series left (default {BASE})")
    args = ap.parse_args(argv)
    try:
        if args.what == "steps":
            rules = read_rules((ROOT / RULES).read_text(encoding="utf-8"))
            sorted_steps = sort_steps(workflow_steps(ROOT), rules)
            for s in sorted_steps:
                print(f"{s['class']}\t{s['workflow']}:{s['line']}\trule {s['rule']}\t{s['cmd'].splitlines()[0][:100]}")
            n = sum(1 for s in sorted_steps if s["class"] == "local")
            slow = sum(1 for s in sorted_steps if s["class"] == "slow")
            print(f"local_receipt: {len(sorted_steps)} steps: {n} local ({len(local_steps(ROOT))} distinct), "
                  f"{slow} slow ({len(local_steps(ROOT, 'slow'))} distinct), {len(sorted_steps) - n - slow} the CI's")
        elif args.what == "run":
            path, ran = run(ROOT, only=args.only, cls="slow" if args.slow else "local")
            red = sum(1 for r in ran if r["rc"] != 0)
            print(f"local_receipt: {len(ran) - red} of {len(ran)} steps green, "
                  + (f"receipt {path}" if path else "no receipt written (--only)"))
            return 1 if red or not ran else 0
        else:
            print(check(ROOT, args.base))
    except ReceiptError as e:
        print(f"local_receipt: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
