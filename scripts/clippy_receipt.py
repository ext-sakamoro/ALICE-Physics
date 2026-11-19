#!/usr/bin/env python3
"""The CI's clippy lines, run here before a series goes to ci/**, and the receipt of that run.

A landing's cheap prepass does not build, so a clippy diagnostic used to be found by the CI
only, a full round later. This script reads every `cargo clippy` line the workflows run
(.github/workflows/*.yml, with the step's `env:`), runs each one to the end in this checkout,
and writes a receipt keyed by the code of HEAD. scripts/land.py
asks for that receipt (`check`) before it pushes anything.

    clippy_receipt.py lines     print the lines read from the workflows
    clippy_receipt.py run       run them on HEAD (a clean checkout) and write the receipt
    clippy_receipt.py check     exit 0 only with a receipt of HEAD's code, every line run, all green

The key is the blob list of HEAD without `docs/` and `*.md` (the ledgers a landing
regenerates are there, and none of them is compiled): a change of any other file, a
workflow and its clippy lines among them, asks for a new run. No line read from
the workflows, or a `cargo clippy` this script cannot read as a line, is an error, not a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ".github/workflows"
RUN = re.compile(r"^(\s*)(?:- )?run:\s*(.*)$")
STEP = re.compile(r"^(\s*)- ")
ENV_PAIR = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*):\s*(.+?)\s*$")
NOT_COMPILED = ("docs/",)


class ReceiptError(Exception):
    pass


def indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def block(lines: list[str], start: int, depth: int) -> list[str]:
    """the lines after `start` that are deeper than `depth` (blank lines kept)"""
    out = []
    for line in lines[start + 1:]:
        if line.strip() and indent(line) <= depth:
            break
        out.append(line)
    return out


def step_env(lines: list[str], at: int, run_depth: int) -> dict[str, str]:
    """the `env:` of the step whose `run:` is on line `at` (the keys beside `run:`)"""
    first = at
    while first > 0 and not (STEP.match(lines[first]) and indent(lines[first]) < run_depth):
        first -= 1
    env: dict[str, str] = {}
    k = first
    while k < len(lines):
        line = lines[k]
        if k > first and line.strip() and indent(line) < run_depth and not line.lstrip().startswith("#"):
            break
        if line.strip() == "env:" and indent(line) == run_depth:
            for pair in block(lines, k, run_depth):
                m = ENV_PAIR.match(pair)
                if m and not pair.lstrip().startswith("#"):
                    env[m.group(1)] = m.group(2).strip("\"'")
        k += 1
    return env


def commands(text: str, name: str) -> list[dict]:
    """every `cargo clippy` command of one workflow, in file order, with its step's env"""
    lines = text.splitlines()
    found: list[dict] = []
    for k, line in enumerate(lines):
        m = RUN.match(line)
        if not m or line.lstrip().startswith("#"):
            continue
        depth = len(m.group(1)) + (2 if "- run:" in line[:len(m.group(1)) + 7] else 0)
        rest = m.group(2).strip()
        if rest in ("|", "|-", "|+"):
            body = [b.strip() for b in block(lines, k, depth) if b.strip()]
        elif rest in (">", ">-", ">+"):
            body = [" ".join(b.strip() for b in block(lines, k, depth) if b.strip())]
        else:
            body = [rest]
        if not any("cargo clippy" in b for b in body):
            continue
        env = step_env(lines, k, depth)
        for b in body:
            if b.startswith("#") or "cargo clippy" not in b:
                continue
            if not b.startswith("cargo clippy") or b.endswith("\\") or "${{" in b or "$(" in b:
                raise ReceiptError(f"{name}:{k + 1}: a `cargo clippy` this script cannot read as one line: {b!r}")
            found.append({"cmd": b, "env": env})
    return found


def workflow_lines(root: Path) -> list[dict]:
    files = sorted((root / WORKFLOWS).glob("*.yml"))
    found: list[dict] = []
    for f in files:
        found += commands(f.read_text(encoding="utf-8"), f.name)
    if not found:
        raise ReceiptError(f"no `cargo clippy` line read from {len(files)} workflow file(s) of {root}: "
                           "nothing to run is an error, not a pass")
    return found


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        raise ReceiptError(f"git {' '.join(args)}: {r.stderr.strip()[-200:]}")
    return r.stdout


def compiled(path: str) -> bool:
    return not path.startswith(NOT_COMPILED) and not path.endswith(".md")


def code_key(root: Path) -> str:
    """the blobs of HEAD that a build can read (the workflows, and so their clippy lines, among them):
    what a receipt is a receipt of"""
    listed = [row for row in git(root, "ls-tree", "-r", "HEAD").splitlines() if compiled(row.split("\t", 1)[1])]
    if not listed:
        raise ReceiptError("HEAD lists no file")
    h = hashlib.sha256("\n".join(listed).encode())
    return h.hexdigest()


def receipt_dir() -> Path:
    base = os.environ.get("CLIPPY_RECEIPT_DIR")
    return Path(base) if base else Path.home() / ".cache" / "alice-siblings" / "clippy-receipts"


def receipt_path(key: str) -> Path:
    return receipt_dir() / f"{key}.json"


def run(root: Path, runner=None) -> tuple[Path, list[dict]]:
    """run every line to the end (a red line does not stop the next), write the receipt"""
    dirty = [row for row in git(root, "status", "--porcelain", "--untracked-files=no").splitlines()
             if compiled(row[3:])]
    if dirty:
        raise ReceiptError(f"{len(dirty)} tracked file(s) differ from HEAD ({dirty[0][3:]} ...): "
                           "the receipt is of a commit, commit first")
    lines = workflow_lines(root)
    key = code_key(root)
    runner = runner or (lambda cmd, env: subprocess.run(["bash", "-c", cmd], cwd=root, env={**os.environ, **env}).returncode)
    ran = []
    for line in lines:
        rc = runner(line["cmd"], line["env"])
        print(f"clippy_receipt: rc={rc} :: {line['cmd']}" + (f"  (env {line['env']})" if line["env"] else ""), flush=True)
        ran.append({**line, "rc": rc})
    path = receipt_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": key, "head": git(root, "rev-parse", "HEAD").strip(), "lines": ran,
                                "compared": len(ran)}, indent=1) + "\n", encoding="utf-8")
    return path, ran


def check(root: Path) -> str:
    """the line land.py prints, or ReceiptError: no receipt, another set of lines, a red line, nothing run"""
    lines = workflow_lines(root)
    key = code_key(root)
    path = receipt_path(key)
    if not path.exists():
        raise ReceiptError(f"no clippy receipt for the code of HEAD ({key[:12]}): run "
                           "`python3 scripts/clippy_receipt.py run` on this commit (the CI's "
                           f"{len(lines)} clippy lines) before the landing")
    data = json.loads(path.read_text(encoding="utf-8"))
    ran = data.get("lines") or []
    if data.get("key") != key or [{"cmd": r.get("cmd"), "env": r.get("env")} for r in ran] != lines:
        raise ReceiptError(f"the clippy receipt {path.name} is not of these {len(lines)} lines: run it again")
    if not ran or data.get("compared") != len(ran):
        raise ReceiptError(f"the clippy receipt {path.name} ran 0 lines")
    red = [r["cmd"] for r in ran if r.get("rc") != 0]
    if red:
        raise ReceiptError(f"{len(red)} of {len(ran)} clippy line(s) not green in the receipt: {red[0]}")
    return f"clippy receipt: {len(ran)} of {len(lines)} lines green on the code of HEAD ({key[:12]})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("lines", "run", "check"))
    args = ap.parse_args(argv)
    try:
        if args.what == "lines":
            lines = workflow_lines(ROOT)
            for line in lines:
                print(line["cmd"] + (f"  (env {line['env']})" if line["env"] else ""))
            print(f"clippy_receipt: {len(lines)} lines")
        elif args.what == "run":
            path, ran = run(ROOT)
            red = sum(1 for r in ran if r["rc"] != 0)
            print(f"clippy_receipt: {len(ran) - red} of {len(ran)} lines green, receipt {path}")
            return 1 if red else 0
        else:
            print(check(ROOT))
    except ReceiptError as e:
        print(f"clippy_receipt: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
