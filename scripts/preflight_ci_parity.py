#!/usr/bin/env python3
"""Every `cargo test` command of ci.yml is also run by scripts/preflight.sh.

preflight.sh is the local copy of CI that gates a push; a `cargo test` step
added to ci.yml and not to preflight.sh is a set of tests nobody runs before
pushing (the parallel-gated tests of 29 test files were run by CI only). The
commands are compared after normalising: the variables preflight.sh defines
(`NATIVE=...`) are expanded, quotes and `--no-fail-fast` are dropped, spaces
collapsed. A CI command that cannot run locally is listed in EXEMPT with the
reason.

Fails (exit 1) on a missing command, and when ci.yml has no `cargo test` at all
(a parse that finds nothing must not pass).

Usage: `python3 scripts/preflight_ci_parity.py` (from the repository root).
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# CI commands that preflight.sh does not run, with the reason
EXEMPT: dict[str, str] = {}


# `VAR=value ... cargo test ...`: leading environment assignments change what the
# command builds (RUSTFLAGS, features through env), so they are part of it
CARGO_TEST = re.compile(r"""^((?:[A-Z_][A-Z0-9_]*=(?:"[^"]*"|'[^']*'|\S+)\s+)*)cargo test\b""")
ASSIGN = re.compile(r"""([A-Z_][A-Z0-9_]*)=("[^"]*"|'[^']*'|\S+)""")


def normalise(cmd: str, env: dict[str, str]) -> str:
    for k, v in env.items():
        cmd = cmd.replace(f'"${k}"', v).replace(f"${{{k}}}", v).replace(f"${k}", v)
    m = CARGO_TEST.match(cmd)
    prefix = ""
    if m and m.group(1):
        # the same assignments in any order are the same command
        pairs = sorted((k, v.strip("\"'")) for k, v in ASSIGN.findall(m.group(1)))
        prefix = " ".join(f"{k}={v}" for k, v in pairs) + " "
        cmd = cmd[len(m.group(1)):]
    cmd = cmd.replace('"', "").replace("'", "")
    cmd = re.sub(r"\s--no-fail-fast\b", "", cmd)
    return prefix + re.sub(r"\s+", " ", cmd).strip()


def ci_commands(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        s = re.sub(r"^(?:-\s*)?run:\s*", "", s)
        if s.startswith(("'", '"')) and s.endswith(s[0]):
            s = s[1:-1]
        if CARGO_TEST.match(s):
            out.append(s)
    return out


def preflight(text: str) -> tuple[set[str], dict[str, str]]:
    env = dict(re.findall(r"^([A-Z_][A-Z0-9_]*)='([^']*)'", text, re.M))
    env.update(re.findall(r'^([A-Z_][A-Z0-9_]*)="([^"]*)"', text, re.M))
    cmds = {normalise(l.strip(), env) for l in text.splitlines() if CARGO_TEST.match(l.strip())}
    return cmds, env


def check(root: str = ROOT) -> tuple[list[str], int]:
    with open(os.path.join(root, ".github", "workflows", "ci.yml"), encoding="utf-8") as f:
        ci = f.read()
    with open(os.path.join(root, "scripts", "preflight.sh"), encoding="utf-8") as f:
        pf = f.read()
    local, env = preflight(pf)
    commands = ci_commands(ci)
    errors = []
    if not commands:
        errors.append("ci.yml has no `cargo test` command (compared nothing)")
    for c in commands:
        n = normalise(c, env)
        if n not in local and n not in EXEMPT:
            errors.append(f"ci.yml runs `{c}` but scripts/preflight.sh does not (add it, or list it in EXEMPT with the reason)")
    return errors, len(commands)


def main() -> int:
    errors, n = check()
    for e in errors:
        print(f"error: {e}")
    print(f"compared: ci.yml cargo test commands {n}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
