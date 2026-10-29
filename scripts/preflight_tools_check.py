#!/usr/bin/env python3
"""Every program, toolchain and target the preflight calls is installed, checked before its first step.

A tool the preflight lacks otherwise surfaces dozens of steps in (in verify.yml's full job, after 14 minutes). The
needs are read from scripts/preflight.sh itself, so a new tool there is checked without a list to keep:

  programs    the command words of its lines (not shell builtins, the functions it defines, assignments or
              paths), `command -v X`, `( cd … && X`, `$(X …`
  toolchains  `cargo +<toolchain>`
  targets     `--target <triple>` in the preflight and in the scripts/*.py it runs (not the tests), except those it installs
              itself (`rustup target add`)
  public API  `cargo +<toolchain> public-api` (the cargo subcommand of that toolchain)

  python3 scripts/preflight_tools_check.py                 # all of it (verify.yml's full job)
  python3 scripts/preflight_tools_check.py --only targets  # the targets (the local preflight's first step)

Exit 1 naming each missing one; 1 when it read no program (compared nothing).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHELL = {"if", "then", "else", "elif", "fi", "for", "do", "done", "while", "case", "esac", "in", "set", "echo",
         "printf", "exit", "export", "local", "declare", "trap", "cd", "command", "env", "true", "false", "return",
         "shift", "read", "test", "eval", "exec", "source", "unset", "rm", "grep", "mkdir", "cat", "sed", "awk", "tr",
         "cut", "head", "tail", "sort", "uniq", "wc", "mktemp", "cp", "mv", "ls", "touch", "tee", "xargs", "find",
         "diff", "basename", "dirname", "sleep", "date", "time", "timeout"}
# a command word: followed by a space, an operator or the line end, not by `=` (an assignment) or `/` (a path)
WORD = r"([a-z][a-z0-9_.-]*)(?=[ \t;|&)]|$)"
TARGET = re.compile(r"""--target["',\s=]+["']?([a-z0-9_]+-[a-z0-9_-]+)""")


def needs(root: Path) -> tuple[set[str], set[str], set[str], str | None]:
    """(programs, toolchains, targets, the toolchain of its `cargo +<toolchain> public-api` or None) of
    scripts/preflight.sh"""
    text = (root / "scripts" / "preflight.sh").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    defined = set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)", code, re.M))
    tools = set(re.findall(r"command -v ([A-Za-z0-9_.-]+)", code))
    tools |= {w for w in re.findall(r"^\s*" + WORD, code, re.M) if w not in SHELL | defined}
    tools |= set(re.findall(r"\(\s*cd [^)]*&&\s*" + WORD, code, re.M))
    tools |= set(re.findall(r"\$\(\s*" + WORD, code, re.M)) - SHELL
    chains = set(re.findall(r"cargo \+([0-9a-z.-]+)", code))
    texts = [code] + [(root / "scripts" / n).read_text(encoding="utf-8", errors="replace")
                      for n in sorted(set(re.findall(r"scripts/([A-Za-z0-9_.-]+\.py)", code)))
                      if (root / "scripts" / n).is_file() and not n.startswith("test_")]
    # (a test's fixtures name targets of their own cases, not ones the preflight builds for)
    # (a target the preflight installs itself with `rustup target add` is not needed beforehand)
    added = {t for line in re.findall(r"rustup target add ([^\n|;&)]+)", code) for t in line.split()}
    targets = {t for x in texts for t in TARGET.findall(x)} - added
    api = re.search(r"cargo \+([0-9a-z.-]+) public-api", code)
    return tools, chains, targets, api.group(1) if api else None


def installed_toolchains() -> str:
    return subprocess.run(["rustup", "toolchain", "list"], capture_output=True, text=True).stdout \
        if shutil.which("rustup") else ""


def installed_targets() -> list[str]:
    return subprocess.run(["rustup", "target", "list", "--installed"], capture_output=True, text=True).stdout.split() \
        if shutil.which("rustup") else []


def public_api_runs(toolchain: str) -> bool:
    return shutil.which("cargo") is not None and subprocess.run(
        ["cargo", f"+{toolchain}", "public-api", "--version"], capture_output=True).returncode == 0


def missing(root: Path, only: str | None = None, which=shutil.which, toolchains=installed_toolchains,
            targets=installed_targets, public_api=public_api_runs) -> tuple[list[str], tuple[set, set, set, str | None]]:
    tools, chains, tgts, api = needs(root)
    out: list[str] = []
    if only in (None, "programs"):
        out += sorted(t for t in tools if which(t) is None)
    if only in (None, "toolchains"):
        have = toolchains()
        out += sorted(f"toolchain {c}" for c in chains if not re.search(rf"^{re.escape(c)}[-.]", have, re.M))
    if only in (None, "targets"):
        have_t = set(targets())
        out += sorted(f"target {t}" for t in tgts if t not in have_t)
    if only is None and api and not public_api(api):
        out.append(f"cargo +{api} public-api")
    return out, (tools, chains, tgts, api)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--only", choices=("programs", "toolchains", "targets"))
    a = ap.parse_args(argv)
    gone, (tools, chains, tgts, _api) = missing(a.root, a.only)
    print(f"preflight calls {len(tools)} program(s), {len(chains)} toolchain(s) and {len(tgts)} target(s): "
          f"{', '.join(sorted(tools))}; {', '.join(sorted(chains))}; {', '.join(sorted(tgts))}")
    if not tools:
        print("error: read no program from scripts/preflight.sh (compared nothing)", file=sys.stderr)
        return 1
    if gone:
        print(f"error: not installed: {', '.join(gone)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
