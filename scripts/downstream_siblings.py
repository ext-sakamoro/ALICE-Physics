"""The sibling repositories a downstream consumer needs next to it, and the commit of each.

scripts/downstream_check.sh builds each direct consumer of alice-physics (ALICE-SDF,
ALICE-LOL, ALICE-TRT) at its main in a layout directory of its own, with this checkout
linked in as ALICE-Physics (the candidate) and every other repository the layout takes
by path placed next to it. cargo reads the manifest of every path dependency, optional
ones included, so each such repository must be placed.

Every tracked Cargo.toml of every checkout in the layout is read (the consumer and the
siblings placed so far), and each `path` dependency (any dependency table, target
tables, [workspace.dependencies] and [patch.*]) that leaves its checkout names a
repository of the layout (`../../ALICE-Sign` -> ALICE-Sign). The consumer itself and
ALICE-Physics are never placed by this script. A needed repository that is not in the
layout yet is printed as "<Repo> <ref>":
  - the commit the consumer pins in its .github/alice-siblings ("<Repo> <40-hex SHA>"
    per line), the commit the consumer itself is built and tested with,
  - or, only with --unpinned-main and only when the consumer has no pin file, `main`
    (a notice naming each such repository goes to stderr, for the job log).
The caller places what is printed and runs the script again until nothing is printed
(siblings can need further siblings).

Fails (exit 1) when
  - a needed repository is not pinned (the pin file is missing without --unpinned-main,
    empty, or does not list it),
  - a line of the pin file is not "<Repo> <40-hex SHA>", or names a repository twice,
  - a repository given with --require is not taken by path (nothing was compared),
  - a path dependency leaves the layout directory,
  - with --verify: a placed repository the consumer pins is not at its pin (nothing is
    printed; one placed at main is named in a notice).

run: python3 scripts/downstream_siblings.py --layout <dir> --consumer ALICE-LOL --require ALICE-Physics
     python3 scripts/downstream_siblings.py --layout <dir> --consumer ALICE-TRT --unpinned-main
     ... --verify
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib

PIN_FILE = os.path.join(".github", "alice-siblings")
CANDIDATE = "ALICE-Physics"
SHA = re.compile(r"[0-9a-f]{40}")
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def _dep_specs(manifest: dict):
    for t in DEP_TABLES:
        yield from manifest.get(t, {}).values()
    for body in manifest.get("target", {}).values():
        for t in DEP_TABLES:
            yield from body.get(t, {}).values()
    yield from manifest.get("workspace", {}).get("dependencies", {}).values()
    for deps in manifest.get("patch", {}).values():
        yield from deps.values()


def outside_repos(root: str, manifests: dict[str, str]) -> dict[str, list[str]]:
    """{repository: [manifest paths]} for each path dependency that leaves root.

    manifests maps a manifest path relative to root to its text."""
    root = os.path.realpath(root)
    parent = os.path.dirname(root)
    found: dict[str, list[str]] = {}
    for rel, text in sorted(manifests.items()):
        for spec in _dep_specs(tomllib.loads(text)):
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            target = os.path.normpath(os.path.join(root, os.path.dirname(rel), spec["path"]))
            if target == root or target.startswith(root + os.sep):
                continue
            if not target.startswith(parent + os.sep):
                raise ValueError(f"{rel}: path {spec['path']!r} leaves the layout directory")
            repo = os.path.relpath(target, parent).split(os.sep)[0]
            found.setdefault(repo, []).append(rel)
    return found


def read_pins(text: str | None) -> tuple[dict[str, str], list[str]]:
    pins: dict[str, str] = {}
    problems: list[str] = []
    if text is None:
        return pins, problems
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or not SHA.fullmatch(parts[1]):
            problems.append(f"{PIN_FILE}:{n}: {line!r} is not '<Repo> <40-hex SHA>'")
            continue
        if parts[0] in pins:
            problems.append(f"{PIN_FILE}:{n}: {parts[0]} pinned twice")
            continue
        pins[parts[0]] = parts[1]
    return pins, problems


def plan(consumer: str, needed: dict[str, list[str]], present: set[str], pins: dict[str, str],
         pin_file_present: bool, unpinned_main: bool, require: list[str]
         ) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """(to place as (repo, ref), problems, notices)"""
    problems, notices, out = [], [], []
    for repo in require:
        if repo not in needed:
            problems.append(f"{consumer} takes no path dependency on {repo} (0 compared)")
    for repo, where in sorted(needed.items()):
        if repo in (consumer, CANDIDATE) or repo in present:
            continue
        users = ", ".join(where)
        if repo in pins:
            out.append((repo, pins[repo]))
        elif not pin_file_present and unpinned_main:
            out.append((repo, "main"))
            notices.append(f"{consumer} has no {PIN_FILE}: {repo} (path dependency of {users}) placed at main")
        elif pin_file_present:
            problems.append(f"{repo} (path dependency of {users}) is not pinned in {consumer}'s {PIN_FILE}")
        else:
            problems.append(f"{repo} (path dependency of {users}) is not placed and {consumer}'s {PIN_FILE} is missing")
    return out, problems, notices


def _git(d: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", d, *args], capture_output=True, text=True)


def _manifests(repo_dir: str) -> dict[str, str]:
    r = _git(repo_dir, "ls-files", "--", "Cargo.toml", "*/Cargo.toml")
    if r.returncode != 0:
        raise ValueError(f"{repo_dir}: not a git checkout")
    out = {}
    for rel in r.stdout.split():
        with open(os.path.join(repo_dir, rel), encoding="utf-8") as f:
            out[rel] = f.read()
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", required=True)
    ap.add_argument("--consumer", required=True)
    ap.add_argument("--require", action="append", default=[])
    ap.add_argument("--unpinned-main", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    layout = os.path.realpath(args.layout)
    consumer_dir = os.path.join(layout, args.consumer)
    present = {n for n in os.listdir(layout)
               if n not in (args.consumer, CANDIDATE) and os.path.isdir(os.path.join(layout, n, ".git"))}
    problems: list[str] = []
    needed: dict[str, list[str]] = {}
    n_manifests = 0
    try:
        for repo in sorted({args.consumer, *present}):
            ms = _manifests(os.path.join(layout, repo))
            n_manifests += len(ms)
            for dep, where in outside_repos(os.path.join(layout, repo), ms).items():
                needed.setdefault(dep, []).extend(f"{repo}/{w}" for w in where)
    except ValueError as e:
        problems.append(str(e))
    pin_path = os.path.join(consumer_dir, PIN_FILE)
    pin_text = None
    if os.path.isfile(pin_path):
        with open(pin_path, encoding="utf-8") as f:
            pin_text = f.read()
    pins, more = read_pins(pin_text)
    problems += more
    place, more, notices = plan(args.consumer, needed, present, pins, pin_text is not None,
                                args.unpinned_main, args.require)
    problems += more
    if args.verify and not problems:
        for repo in sorted(present):
            head = _git(os.path.join(layout, repo), "rev-parse", "HEAD").stdout.strip()
            if repo in pins:
                if head != pins[repo]:
                    problems.append(f"{repo}: HEAD {head} is not {args.consumer}'s pin {pins[repo]}")
            elif pin_text is None and args.unpinned_main:
                notices.append(f"{args.consumer} has no {PIN_FILE}: {repo} is at main ({head[:12]})")
            else:
                problems.append(f"{repo}: placed but not pinned in {args.consumer}'s {PIN_FILE}")
        if place:
            problems.append(f"not placed yet: {', '.join(r for r, _ in place)}")
    for n in notices:
        print(f"downstream_siblings: notice: {n}", file=sys.stderr)
    for p in problems:
        print(f"downstream_siblings: {p}", file=sys.stderr)
    print(f"downstream_siblings: {args.consumer}: {n_manifests} manifest(s), {len(needed)} repositor(ies) taken by "
          f"path ({', '.join(sorted(needed))}), {len(place)} to place, {len(problems)} problem(s)", file=sys.stderr)
    if problems:
        return 1
    if not args.verify:
        for repo, ref in place:
            print(f"{repo} {ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
