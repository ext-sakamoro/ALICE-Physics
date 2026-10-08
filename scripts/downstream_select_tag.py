#!/usr/bin/env python3
"""Pick the release tag of a repository placed next to ALICE-LOL, and check the
version requirements of the placed repositories.

scripts/downstream_check.sh places ALICE-LOL, ALICE-SDF and ALICE-TRT (their
main branches) next to this checkout. ALICE-LOL also needs other repositories by
path; for each of those this picks the newest release tag that fits both ways:

  * forward   every version requirement on the crate in the Cargo.toml files of
              the ALICE-LOL workspace is satisfied by the tag's version
  * reverse   every version requirement of a path dependency in the tag's own
              Cargo.toml files on a package that is already placed (alice-physics from this checkout,
              alice-lol, alice-sdf, ... and the repositories selected before
              this one) is satisfied by the version placed next to it

Pre-release tags are skipped. When no tag fits, the requirements and the tags
(with the reason each candidate was rejected) are printed and the exit status
is 1: falling back to a branch would test a combination nobody declared.

  python3 scripts/downstream_select_tag.py --url URL --repo DIR --crate NAME \
      --workspace LOL_DIR --provider DIR [--provider DIR ...]

prints `refs/tags/<tag>` on stdout; the explanation goes to stderr. The tag is
fetched into DIR (a git repository, created when missing) to read its
manifests.

  python3 scripts/downstream_select_tag.py --check TREE [--check TREE ...] \
      --provider DIR [--provider DIR ...]

checks that every requirement (path or registry: a registry dependency is
replaced by `[patch]` only when the requirement accepts the patched version) in
the Cargo.toml files under each TREE on a package of the providers (for downstream_check.sh: alice-physics of this
checkout) is satisfied by the provider's version, and fails with each
requirement that is not (a downstream that still requires alice-physics 1.x
when this checkout is 2.0.0). A TREE with no requirement on any provided
package also fails.

Manifests are read with tomllib (Python 3.11 or newer; `tomli` is used on older
versions when installed, otherwise the script stops with an error rather than
reading the manifests by pattern). Dependencies are read from `[dependencies]`,
`[dev-dependencies]`, `[build-dependencies]`, their `[target.'cfg(..)'.*]`
forms and `[workspace.dependencies]`, written inline or as tables
(`[dependencies.x]`), under their package name when renamed (`package = "x"`).
`[features]` entries such as `x = ["dep:x"]` are not requirements. A member's
`x.workspace = true` takes the requirement written in `[workspace.dependencies]`,
which is read from the root manifest. Requirements follow Cargo: a pre-release
version (`0.3.0-beta.2`) satisfies a requirement only when one of its
comparators names a pre-release of the same major.minor.patch.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        sys.exit("error: reading Cargo.toml needs Python 3.11 or newer (tomllib) or the tomli package")

Core = tuple[int, int, int]
# sort key of a version: (core, 1, ()) for a release, (core, 0, identifiers) for a
# pre-release, so a pre-release sorts below its release and (core, 0, ()) below
# every version of that core
Key = tuple[Core, int, tuple]

DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies", "dev_dependencies", "build_dependencies")


def _pre_ids(pre: str) -> tuple:
    return tuple((0, int(x), "") if x.isdigit() else (1, 0, x) for x in pre.split("."))


def parse(v: str) -> tuple[int, int | None, int | None, str | None]:
    """`1`, `1.2`, `1.2.3` or `1.2.3-pre` (build metadata dropped)."""
    m = re.fullmatch(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?", v.strip())
    if not m:
        raise ValueError(f"unsupported version `{v}`")
    a, b, c, pre = m.groups()
    if pre is not None and c is None:
        raise ValueError(f"unsupported version `{v}`")
    return int(a), None if b is None else int(b), None if c is None else int(c), pre


def version_key(v: str) -> Key:
    """A package version (`1.2.3`, `0.3.0-beta.2`)."""
    a, b, c, pre = parse(v)
    core = (a, b or 0, c or 0)
    return (core, 0, _pre_ids(pre)) if pre else (core, 1, ())


def _floor(core: Core) -> Key:
    return (core, 0, ())


def comparators(req: str) -> list[tuple[str, int, int | None, int | None, str | None]]:
    out = []
    for part in req.split(","):
        m = re.fullmatch(r"\s*(\^|~|=|>=|<=|>|<)?\s*([0-9A-Za-z.*+-]+)\s*", part)
        if not m:
            raise ValueError(f"unsupported requirement `{req}`")
        op, v = m.group(1) or "^", m.group(2)
        if v == "*":
            out.append(("*", 0, None, None, None))
            continue
        if v.endswith(".*"):
            v = v[:-2]
            op = "=" if op == "^" else op
        out.append((op, *parse(v)))
    return out


def _holds(key: Key, op: str, major: int, minor: int | None, patch: int | None, pre: str | None) -> bool:
    if op == "*":
        return True
    lo_core = (major, minor or 0, patch or 0)
    lo = (lo_core, 0, _pre_ids(pre)) if pre else (lo_core, 1, ())
    nxt_minor = (major + 1, 0, 0) if minor is None else (major, minor + 1, 0)
    nxt = nxt_minor if patch is None else (major, minor or 0, patch + 1)
    full = patch is not None
    if op == "^":
        if major > 0 or minor is None:
            hi = (major + 1, 0, 0)
        elif minor > 0 or patch is None:
            hi = (0, minor + 1, 0)
        else:
            hi = (0, 0, patch + 1)
        return lo <= key < _floor(hi)
    if op == "~":
        return lo <= key < _floor(nxt_minor)
    if op == "=":
        return key == lo if full else _floor(lo_core) <= key < _floor(nxt)
    if op == ">=":
        return key >= lo
    if op == ">":
        return key > lo if full else key >= _floor(nxt)
    if op == "<":
        return key < lo
    return key <= lo if full else key < _floor(nxt)  # <=


def satisfies(version: str | Core, req: str) -> bool:
    """Whether `version` matches the Cargo requirement `req`."""
    key = (version, 1, ()) if isinstance(version, tuple) else version_key(version)
    comps = comparators(req)
    if key[1] == 0:
        # Cargo: a pre-release matches only through a comparator that names a
        # pre-release of the same major.minor.patch
        if not any(pre is not None and (a, b, c) == key[0] for _, a, b, c, pre in comps):
            return False
    return all(_holds(key, *c) for c in comps)


def load(text: str, where: str = "Cargo.toml") -> dict:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{where}: not valid TOML: {e}") from None


def dependency_tables(doc: dict):
    for t in DEP_TABLES:
        if isinstance(doc.get(t), dict):
            yield t, doc[t]
    for cfg, sub in (doc.get("target") or {}).items():
        for t in DEP_TABLES:
            if isinstance(sub, dict) and isinstance(sub.get(t), dict):
                yield f"target.{cfg}.{t}", sub[t]
    ws = doc.get("workspace") or {}
    if isinstance(ws.get("dependencies"), dict):
        yield "workspace.dependencies", ws["dependencies"]


def dependency_requirements(text: str, crate: str, where: str = "Cargo.toml",
                            path_only: bool = False) -> list[tuple[str, str | None]]:
    """(location, requirement) of each dependency on the package `crate` in a
    manifest; requirement None when the entry has no version. An entry that
    inherits from the workspace (`workspace = true`) is skipped: its requirement
    is the `[workspace.dependencies]` entry of the root manifest. With
    `path_only`, only entries with a `path` (those that the placed checkout
    satisfies) are returned."""
    out = []
    for table, deps in dependency_tables(load(text, where)):
        for key, spec in deps.items():
            if isinstance(spec, str):
                pkg, req, has_path = key, spec, False
            elif isinstance(spec, dict):
                if spec.get("workspace") is True:
                    continue
                pkg, req, has_path = spec.get("package", key), spec.get("version"), "path" in spec
            else:
                continue
            if path_only and not has_path:
                continue
            if pkg == crate:
                out.append((f"[{table}] {key}", req))
    return out


def package_of(text: str, workspace_version: str | None, where: str = "Cargo.toml") -> tuple[str, str] | None:
    """(name, version) of the `[package]` of a manifest."""
    pkg = load(text, where).get("package")
    if not isinstance(pkg, dict) or not isinstance(pkg.get("name"), str):
        return None
    ver = pkg.get("version")
    if isinstance(ver, str):
        return pkg["name"], ver
    if isinstance(ver, dict) and ver.get("workspace") is True and workspace_version:
        return pkg["name"], workspace_version
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
        root = load(dict(manifests).get("Cargo.toml", ""))
        wver = ((root.get("workspace") or {}).get("package") or {}).get("version")
        for path, text in manifests:
            pkg = package_of(text, wver if isinstance(wver, str) else None, path)
            if pkg:
                out.setdefault(pkg[0], pkg[1])
    return out


def release_tags(refs: list[str], crate: str) -> dict[Core, str]:
    tags: dict[Core, str] = {}
    for ref in refs:
        name = ref.removeprefix("refs/tags/")
        m = re.fullmatch(rf"(?:{re.escape(crate)}-)?v?(\d+)\.(\d+)\.(\d+)", name)
        if m:  # a pre-release (`v1.0.0-beta.1`) does not match
            tags[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = name
    return tags


def clashes(manifests: list[tuple[str, str]], provided: dict[str, str], skip: str | None = None,
            path_only: bool = False) -> tuple[list[str], int]:
    """Requirements in `manifests` on a provided package that the placed version
    does not satisfy, and the number of requirements compared."""
    out, n = [], 0
    for path, text in manifests:
        for pkg, have in sorted(provided.items()):
            if pkg == skip:
                continue
            for where, req in dependency_requirements(text, pkg, path, path_only):
                n += 1
                if req is not None and not satisfies(have, req):
                    out.append(f"{path} {where} requires {pkg} {req}, placed {have}")
    return out, n


def select(crate: str, workspace: list[tuple[str, str]], tags: dict[Core, str],
           provided: dict[str, str], tag_manifests) -> tuple[str | None, list[str]]:
    """The tag to use and the explanation lines. `tag_manifests(tag)` returns the
    (path, text) manifests of a tag; it is called newest first, only for tags
    that pass the forward check."""
    reqs = []
    for path, text in workspace:
        for where, req in dependency_requirements(text, crate, path):
            if req is None:
                return None, [f"error: {path} {where}: `{crate}` has no version requirement"]
            reqs.append((f"{path} {where}", req))
    if not reqs:
        return None, [f"error: no Cargo.toml of the workspace requires `{crate}`"]
    rejected = []
    for v in sorted(tags, reverse=True):
        tag = tags[v]
        if not all(satisfies(v, r) for _, r in reqs):
            continue
        # a registry dependency of the tag is not served by the placed checkout
        clash, _ = clashes(tag_manifests(tag), provided, skip=crate, path_only=True)
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


def check(trees: list[tuple[str, list[tuple[str, str]]]], provided: dict[str, str]) -> tuple[bool, list[str]]:
    """Every requirement in each tree on a provided package must be satisfied, and
    each tree must have at least one."""
    ok, lines = True, []
    names = ", ".join(f"{p} {v}" for p, v in sorted(provided.items()))
    for name, manifests in trees:
        bad, n = clashes(manifests, provided)
        if n == 0:
            ok = False
            lines.append(f"error: {name}: no Cargo.toml requires any of {names}")
        elif bad:
            ok = False
            lines += [f"error: {name}/{b}" for b in bad]
        else:
            lines.append(f"{name}: {n} requirement(s) on {names} satisfied")
    return ok, lines


def git(*args: str, cwd: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", help="repository URL")
    ap.add_argument("--repo", help="local git directory to fetch tags into")
    ap.add_argument("--crate")
    ap.add_argument("--workspace", help="the ALICE-LOL checkout")
    ap.add_argument("--provider", action="append", default=[], help="checkout whose packages are placed")
    ap.add_argument("--check", action="append", default=[], help="tree whose requirements on the providers are checked")
    args = ap.parse_args(argv)

    try:
        provided = provided_packages([manifests_on_disk(p) for p in args.provider])
        if args.check:
            # the providers' own packages (alice-physics), not their fuzz or tool crates
            provided = provided_packages([[m for m in manifests_on_disk(p) if m[0] == "Cargo.toml"]
                                          for p in args.provider])
            ok, lines = check([(os.path.basename(os.path.normpath(t)), manifests_on_disk(t)) for t in args.check],
                              provided)
            print("\n".join(lines), file=sys.stderr)
            return 0 if ok else 1
        if not (args.url and args.repo and args.crate and args.workspace):
            ap.error("--url, --repo, --crate and --workspace are required without --check")

        refs = [line.split()[1] for line in git("ls-remote", "--tags", "--refs", args.url).splitlines() if line.strip()]
        if not os.path.isdir(os.path.join(args.repo, ".git")):
            git("init", "-q", args.repo)
            git("remote", "add", "origin", args.url, cwd=args.repo)

        def tag_manifests(tag: str) -> list[tuple[str, str]]:
            git("fetch", "-q", "--depth", "1", "origin", f"refs/tags/{tag}", cwd=args.repo)
            paths = [p for p in git("ls-tree", "-r", "--name-only", "FETCH_HEAD", cwd=args.repo).splitlines()
                     if p == "Cargo.toml" or p.endswith("/Cargo.toml")]
            return [(p, git("show", f"FETCH_HEAD:{p}", cwd=args.repo)) for p in paths]

        tag, lines = select(args.crate, manifests_on_disk(args.workspace), release_tags(refs, args.crate),
                            provided, tag_manifests)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("\n".join(lines), file=sys.stderr)
    if tag is None:
        return 1
    print(f"refs/tags/{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
