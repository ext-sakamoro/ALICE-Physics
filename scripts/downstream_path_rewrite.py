#!/usr/bin/env python3
"""Point the downstream path dependencies on alice-physics at the facade, in the fetched copies.

The repository root is the engine package (alice-physics-core); alice-physics is the facade
in crates/alice-physics. A downstream manifest written before that layout depends on
`alice-physics = { path = "../ALICE-Physics", ... }`, which now names the engine's directory
(another package name), so cargo cannot resolve it. scripts/downstream_check.sh builds the
downstream crates against this checkout; this script makes, in the fetched copies only, the
one edit each downstream repository makes itself: the path of every dependency on the
facade's package that names this checkout's root gets `/crates/alice-physics` appended.

  * a dependency counts when its package (the key, or `package = "..."`) is the facade's and
    its path, resolved from the manifest's directory, is the provider's root (the same
    directory, symlinks resolved);
  * after the edit the manifest is parsed again and must equal the original with only those
    path fields changed; anything else stops the script (exit 1);
  * a dependency that already names the facade, or names another directory, is left alone.

Each changed line is printed as a diff with the number rewritten (0 is not an error: the
downstream repositories carry the new path once they have been updated).

  python3 scripts/downstream_path_rewrite.py --provider DIR --tree DIR [--tree DIR ...]
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import downstream_select_tag as st  # noqa: E402

FACADE_REL = "crates/alice-physics"
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def facade_package(provider: str) -> str:
    with open(os.path.join(provider, FACADE_REL, "Cargo.toml"), encoding="utf-8") as f:
        pkg = st.package_of(f.read(), None, f"{FACADE_REL}/Cargo.toml")
    if pkg is None:
        raise ValueError(f"{provider}/{FACADE_REL}/Cargo.toml has no [package] name and version")
    return pkg[0]


def dep_tables(doc: dict) -> list[dict]:
    out = [doc.get(t) for t in DEP_TABLES]
    out.append((doc.get("workspace") or {}).get("dependencies"))
    for spec in (doc.get("target") or {}).values():
        if isinstance(spec, dict):
            out += [spec.get(t) for t in DEP_TABLES]
    return [t for t in out if isinstance(t, dict)]


def same_dir(a: str, b: str) -> bool:
    return os.path.isdir(a) and os.path.isdir(b) and os.path.samefile(a, b)


def targets(doc: dict, package: str, manifest_dir: str, provider: str) -> set[str]:
    """the path strings of the dependencies on `package` that name the provider's root"""
    out = set()
    for table in dep_tables(doc):
        for key, spec in table.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
                continue
            if spec.get("package", key) != package:
                continue
            if same_dir(os.path.join(manifest_dir, spec["path"]), provider):
                out.add(spec["path"])
    return out


def rewrite_text(text: str, paths: set[str]) -> tuple[str, list[str]]:
    lines, diff = text.splitlines(keepends=True), []
    for i, line in enumerate(lines):
        for p in sorted(paths):
            pat = re.compile(r'(\bpath\s*=\s*)"' + re.escape(p) + r'"')
            if pat.search(line):
                new = pat.sub(lambda m: f'{m.group(1)}"{p.rstrip("/")}/{FACADE_REL}"', line)
                diff += [f"-{line.rstrip()}", f"+{new.rstrip()}"]
                lines[i] = line = new
    return "".join(lines), diff


def rewrite_tree(root: str, provider: str, package: str) -> list[str]:
    report = []
    for rel, text in st.manifests_on_disk(root):
        doc = st.load(text, rel)
        manifest_dir = os.path.dirname(os.path.join(root, rel))
        paths = targets(doc, package, manifest_dir, provider)
        if not paths:
            continue
        new, diff = rewrite_text(text, paths)
        # parsed again: equal to the original with only the rewritten paths changed
        want = copy.deepcopy(doc)
        for table in dep_tables(want):
            for key, spec in table.items():
                if isinstance(spec, dict) and spec.get("path") in paths and spec.get("package", key) == package:
                    spec["path"] = f'{spec["path"].rstrip("/")}/{FACADE_REL}'
        if st.load(new, rel) != want:
            raise ValueError(f"{os.path.basename(root)}/{rel}: the path of {package} could not be rewritten "
                             "on its own line (the parsed manifest differs from the expected one)")
        with open(os.path.join(root, rel), "w", encoding="utf-8") as f:
            f.write(new)
        report.append(f"--- {os.path.basename(os.path.normpath(root))}/{rel}")
        report += diff
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--provider", required=True, help="this checkout (the engine at its root)")
    ap.add_argument("--tree", action="append", default=[], required=True, help="fetched downstream repository")
    args = ap.parse_args(argv)
    try:
        package = facade_package(args.provider)
        n = 0
        for tree in args.tree:
            report = rewrite_tree(tree, args.provider, package)
            n += sum(1 for r in report if r.startswith("+"))
            print("\n".join(report) if report else
                  f"{os.path.basename(os.path.normpath(tree))}: no path dependency on {package} names the root")
        print(f"{n} path dependency line(s) on {package} pointed at {FACADE_REL}")
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
