#!/usr/bin/env python3
"""Pick the release tag of a repository that ALICE-LOL needs next to it.

scripts/downstream_check.sh places ALICE-LOL and ALICE-SDF (their main
branches) next to this checkout. ALICE-LOL also needs other repositories by
path; for each of those this picks the newest release tag that fits both ways:

  * forward   every version requirement on the crate in the Cargo.toml files of
              the ALICE-LOL workspace is satisfied by the tag's version
  * reverse   every version requirement in the tag's own Cargo.toml files on a
              package that ALICE-LOL or ALICE-SDF provide (alice-lol, alice-sdf,
              ...) is satisfied by the version placed next to it

Pre-release tags are skipped. When no tag fits, the requirements and the tags
(with the reason each candidate was rejected) are printed and the exit status
is 1: falling back to a branch would test a combination nobody declared.

  python3 scripts/downstream_select_tag.py --url URL --repo DIR --crate NAME \
      --workspace LOL_DIR --provider LOL_DIR --provider SDF_DIR

Prints `refs/tags/<tag>` on stdout; the explanation goes to stderr. The tag is
fetched into DIR (a git repository, created when missing) to read its
manifests.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

Version = tuple[int, int, int]


def parse(v: str) -> tuple[int, int | None, int | None]:
    m = re.fullmatch(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", v.strip())
    if not m:
        raise ValueError(f"unsupported version `{v}`")
    a, b, c = m.groups()
    return int(a), None if b is None else int(b), None if c is None else int(c)


def full(v: str) -> Version:
    """A package version (`1.2.3`, pre-release and build metadata dropped)."""
    a, b, c = parse(re.split(r"[-+]", v.strip())[0])
    return a, b or 0, c or 0


def bounds(req: str) -> list[tuple[Version, Version | None]]:
    """A Cargo version requirement as (lower inclusive, upper exclusive) bounds."""
    out: list[tuple[Version, Version | None]] = []
    for part in req.split(","):
        m = re.fullmatch(r"\s*(\^|~|=|>=|<=|>|<)?\s*([0-9.*]+)\s*", part)
        if not m:
            raise ValueError(f"unsupported requirement `{req}`")
        op, v = m.group(1) or "^", m.group(2)
        if v.endswith(".*"):
            v = v[:-2]
            op = "=" if op == "^" else op
        major, minor, patch = parse(v)
        lo = (major, minor or 0, patch or 0)
        nxt_minor = (major + 1, 0, 0) if minor is None else (major, minor + 1, 0)
        nxt_patch = nxt_minor if patch is None else (major, minor or 0, patch + 1)
        if op == "^":
            if major > 0 or minor is None:
                hi = (major + 1, 0, 0)
            elif minor > 0 or patch is None:
                hi = (0, minor + 1, 0)
            else:
                hi = (0, 0, patch + 1)
            out.append((lo, hi))
        elif op == "~":
            out.append((lo, nxt_minor))
        elif op == "=":
            out.append((lo, nxt_patch))
        elif op == ">=":
            out.append((lo, None))
        elif op == ">":
            out.append((nxt_patch, None))
        elif op == "<":
            out.append(((0, 0, 0), lo))
        else:  # <=
            out.append(((0, 0, 0), nxt_patch))
    return out


def satisfies(version: Version, req: str) -> bool:
    return all(lo <= version and (hi is None or version < hi) for lo, hi in bounds(req))


def dependency_requirements(text: str, crate: str) -> list[tuple[int, str | None]]:
    """(line, requirement) of each inline dependency on `crate` in a manifest
    (`crate = "1.2"` or `crate = { ..., version = "1.2" }`); requirement None
    when the entry has no version."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        m = re.match(rf"^\s*{re.escape(crate)}\s*=\s*(.*)$", line)
        if not m:
            continue
        rhs = m.group(1)
        v = re.search(r'\bversion\s*=\s*"([^"]+)"', rhs) or re.fullmatch(r'"([^"]+)"\s*(#.*)?', rhs.strip())
        out.append((i, v.group(1) if v else None))
    return out


def package_of(text: str, workspace_version: str | None) -> tuple[str, str] | None:
    """(name, version) of the `[package]` of a manifest."""
    m = re.search(r"^\[package\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if not m:
        return None
    name = re.search(r'^name\s*=\s*"([^"]+)"', m.group(1), re.M)
    ver = re.search(r'^version\s*=\s*"([^"]+)"', m.group(1), re.M)
    if not name:
        return None
    if ver:
        return name.group(1), ver.group(1)
    if re.search(r"^version(\.workspace\s*=\s*true|\s*=\s*\{\s*workspace\s*=\s*true)", m.group(1), re.M) and workspace_version:
        return name.group(1), workspace_version
    return None


def manifests_on_disk(root: str) -> list[tuple[str, str]]:
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in (".git", "target"))
        if "Cargo.toml" in files:
            p = os.path.join(d, "Cargo.toml")
            with open(p, encoding="utf-8") as f:
                out.append((os.path.relpath(p, root).replace(os.sep, "/"), f.read()))
    return out


def provided_packages(manifest_sets: list[list[tuple[str, str]]]) -> dict[str, str]:
    """package name -> version for every package in the given trees."""
    out: dict[str, str] = {}
    for manifests in manifest_sets:
        root = dict(manifests).get("Cargo.toml", "")
        wv = re.search(r"^\[workspace\.package\]\s*$(.*?)(?=^\[|\Z)", root, re.M | re.S)
        wver = re.search(r'^version\s*=\s*"([^"]+)"', wv.group(1), re.M).group(1) if wv and \
            re.search(r'^version\s*=\s*"([^"]+)"', wv.group(1), re.M) else None
        for _, text in manifests:
            pkg = package_of(text, wver)
            if pkg:
                out.setdefault(pkg[0], pkg[1])
    return out


def release_tags(refs: list[str], crate: str) -> dict[Version, str]:
    tags: dict[Version, str] = {}
    for ref in refs:
        name = ref.removeprefix("refs/tags/")
        m = re.fullmatch(rf"(?:{re.escape(crate)}-)?v?(\d+\.\d+\.\d+)", name)
        if m:  # a pre-release (`v1.0.0-beta.1`) does not match
            tags[full(m.group(1))] = name
    return tags


def select(crate: str, workspace: list[tuple[str, str]], tags: dict[Version, str],
           provided: dict[str, str], tag_manifests) -> tuple[str | None, list[str]]:
    """The tag to use and the explanation lines. `tag_manifests(tag)` returns the
    (path, text) manifests of a tag; it is called newest first, only for tags
    that pass the forward check."""
    reqs = []
    for path, text in workspace:
        for line, req in dependency_requirements(text, crate):
            if req is None:
                return None, [f"error: {path}:{line}: `{crate}` has no version requirement"]
            reqs.append((f"{path}:{line}", req))
    if not reqs:
        return None, [f"error: no Cargo.toml of the workspace requires `{crate}`"]
    rejected = []
    for v in sorted(tags, reverse=True):
        tag = tags[v]
        if not all(satisfies(v, r) for _, r in reqs):
            continue
        clash = []
        for path, text in tag_manifests(tag):
            for pkg, have in sorted(provided.items()):
                if pkg == crate:
                    continue
                for line, req in dependency_requirements(text, pkg):
                    if req is not None and not satisfies(full(have), req):
                        clash.append(f"{path}:{line} requires {pkg} {req}, placed {have}")
        if not clash:
            why = ", ".join(f"{r} ({w})" for w, r in reqs)
            return tag, [f"{crate}: {tag} satisfies {why}, and its manifests accept the placed packages"]
        rejected.append(f"    {tag}: " + "; ".join(clash))
    lines = [f"error: no release tag of `{crate}` fits the placed workspace", "  requirements on it:"]
    lines += [f"    {w}: {r}" for w, r in reqs]
    if rejected:
        lines += ["  rejected (its manifests do not accept the placed packages):"] + rejected
    lines += ["  release tags: " + (", ".join(tags[v] for v in sorted(tags)) or "(none)")]
    return None, lines


def git(*args: str, cwd: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="repository URL")
    ap.add_argument("--repo", required=True, help="local git directory to fetch tags into")
    ap.add_argument("--crate", required=True)
    ap.add_argument("--workspace", required=True, help="the ALICE-LOL checkout")
    ap.add_argument("--provider", action="append", default=[], help="checkout whose packages are placed")
    args = ap.parse_args(argv)

    refs = [line.split()[1] for line in git("ls-remote", "--tags", "--refs", args.url).splitlines() if line.strip()]
    if not os.path.isdir(os.path.join(args.repo, ".git")):
        git("init", "-q", args.repo)
        git("remote", "add", "origin", args.url, cwd=args.repo)

    def tag_manifests(tag: str) -> list[tuple[str, str]]:
        git("fetch", "-q", "--depth", "1", "origin", f"refs/tags/{tag}", cwd=args.repo)
        paths = [p for p in git("ls-tree", "-r", "--name-only", "FETCH_HEAD", cwd=args.repo).splitlines()
                 if p == "Cargo.toml" or p.endswith("/Cargo.toml")]
        return [(p, git("show", f"FETCH_HEAD:{p}", cwd=args.repo)) for p in paths]

    provided = provided_packages([manifests_on_disk(p) for p in args.provider])
    tag, lines = select(args.crate, manifests_on_disk(args.workspace), release_tags(refs, args.crate),
                        provided, tag_manifests)
    print("\n".join(lines), file=sys.stderr)
    if tag is None:
        return 1
    print(f"refs/tags/{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
