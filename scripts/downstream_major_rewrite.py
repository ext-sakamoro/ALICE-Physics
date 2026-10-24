#!/usr/bin/env python3
"""On a major bump of a crate, raise the downstream requirements on it to the new
version in the fetched copies, so the downstream build tests the new major.

scripts/downstream_check.sh builds ALICE-SDF, ALICE-LOL and ALICE-TRT against
this checkout. Their manifests require the alice-physics they were written for;
a commit that raises the major version (2.x -> 3.0.0) is rejected by every such
requirement, and until the new version is published the downstream repositories
cannot raise the requirement themselves (crates.io has nothing to resolve it
to). This script makes that edit in the fetched copies only, so the downstream
code is compiled and tested against the new major with nothing changed but the
requirement:

  * major bump   the checkout's version is newer than the newest published
                 release (crates.io index, yanked versions and pre-releases
                 skipped) and outside its caret range (2.2.0 -> 3.0.0, 0.4.1 ->
                 0.5.0). Anything else is not a bump and no manifest is touched
  * rewritten    a requirement on the crate that accepts the newest published
                 release and rejects the checkout's version: its version string
                 becomes the checkout's version (`"2.2"` -> `"3.0.0"`, the caret
                 range of the new major). A requirement that rejects the
                 published release too (a downstream still on an older major)
                 and a dependency without a version are left as they are, so
                 the requirement check that follows still fails on the former

On a bump, at least one requirement in the trees must accept the new version
after the edit (the copies may already be edited by an earlier run, and a range
such as ">=3.0, <5" accepts both majors without an edit); none fails, since the
build would test the bump against no consumer. Each changed line is printed as a diff. After the edit the manifest is parsed
again and must equal the original with only those version fields replaced; a
line that cannot be edited that way stops the script (exit 1) instead of
leaving a requirement half-rewritten. A failure to read the index also exits
1: without the published version the bump cannot be decided.

  python3 scripts/downstream_major_rewrite.py --provider DIR --tree DIR [--tree DIR ...]
      [--index-file FILE]

--provider is this checkout (its root Cargo.toml names the crate and version);
--index-file reads the crates.io index entry from a file instead of
https://index.crates.io (the tests use it).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import downstream_select_tag as st  # noqa: E402

INDEX_URL = "https://index.crates.io"


def index_path(crate: str) -> str:
    """Path of a crate's entry in the crates.io sparse index."""
    n = crate.lower()
    if len(n) <= 2:
        return f"{len(n)}/{n}"
    if len(n) == 3:
        return f"3/{n[0]}/{n}"
    return f"{n[:2]}/{n[2:4]}/{n}"


def newest_published(index_text: str) -> str:
    """Newest release in an index entry (one JSON object per line), skipping
    yanked versions and pre-releases."""
    best: str | None = None
    for line in index_text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        v = rec.get("vers")
        if not isinstance(v, str) or rec.get("yanked") or "-" in v.split("+", 1)[0]:
            continue
        if best is None or st.version_key(v) > st.version_key(best):
            best = v
    if best is None:
        raise ValueError("the index lists no published release")
    return best


def is_major_bump(published: str, version: str) -> bool:
    return st.version_key(version) > st.version_key(published) and not st.satisfies(version, f"^{published}")


def eligible(req: str | None, published: str, version: str) -> bool:
    return req is not None and st.satisfies(published, req) and not st.satisfies(version, req)


HEADER = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")


def _quoted(s: str) -> str:
    return re.escape(f'"{s}"')


def rewrite_text(text: str, crate: str, published: str, version: str, where: str = "Cargo.toml") \
        -> tuple[str, list[str]]:
    """The manifest with each eligible requirement on `crate` set to `version`, and
    the diff lines. Raises ValueError when the result is not the original with
    exactly those version fields replaced."""
    doc = st.load(text, where)
    want = copy.deepcopy(doc)
    targets: set[str] = set()
    for table, deps in st.dependency_tables(doc):
        for key, spec in deps.items():
            if isinstance(spec, str):
                pkg, req = key, spec
            elif isinstance(spec, dict) and spec.get("workspace") is not True:
                pkg, req = spec.get("package", key), spec.get("version")
            else:
                continue
            if pkg != crate or not eligible(req, published, version):
                continue
            targets.add(req)
            # the same entry in the copy that the edited text must parse to
            for t2, d2 in st.dependency_tables(want):
                if t2 == table:
                    if isinstance(d2[key], str):
                        d2[key] = version
                    else:
                        d2[key]["version"] = version
    if not targets:
        return text, []

    lines = text.splitlines(keepends=True)
    out, diff = [], []
    section_is_crate = False  # inside [dependencies.<crate>] or a table naming package = crate
    sec_lines: list[int] = []
    key_re = re.compile(rf'^\s*(?:"{re.escape(crate)}"|{re.escape(crate)})\s*=')
    for i, line in enumerate(lines):
        new = line
        m = HEADER.match(line)
        if m:
            name = m.group(1).strip()
            last = name.rsplit(".", 1)[-1].strip().strip('"\'')
            section_is_crate = last == crate and re.search(r"dependencies\.", name) is not None
            sec_lines = []
            # a renamed table form ([dependencies.phys] + package = "crate") is found below
        else:
            sec_lines.append(i)
            for req in targets:
                if key_re.match(line) or re.search(rf'package\s*=\s*{_quoted(crate)}', line):
                    new = re.sub(rf'^(\s*(?:"{re.escape(crate)}"|{re.escape(crate)})\s*=\s*){_quoted(req)}',
                                 rf'\g<1>"{version}"', new)
                    new = re.sub(rf'(\bversion\s*=\s*){_quoted(req)}', rf'\g<1>"{version}"', new)
                elif section_is_crate:
                    new = re.sub(rf'^(\s*version\s*=\s*){_quoted(req)}', rf'\g<1>"{version}"', new)
        out.append(new)
    # renamed table form: a section with `package = "crate"` and its own `version = "req"` line
    text2 = "".join(out)
    if st.load(text2, where) != want:
        sections: list[list[int]] = []
        cur: list[int] = []
        for i, line in enumerate(out):
            if HEADER.match(line):
                sections.append(cur)
                cur = [i]
            else:
                cur.append(i)
        sections.append(cur)
        for sec in sections:
            if any(re.search(rf'^\s*package\s*=\s*{_quoted(crate)}', out[i]) for i in sec):
                for i in sec:
                    for req in targets:
                        out[i] = re.sub(rf'^(\s*version\s*=\s*){_quoted(req)}', rf'\g<1>"{version}"', out[i])
        text2 = "".join(out)
    if st.load(text2, where) != want:
        raise ValueError(f"{where}: could not rewrite the requirement on {crate} to {version} "
                         "(the edited manifest is not the original with only that version changed)")
    for a, b in zip(lines, out):
        if a != b:
            diff += [f"-{a.rstrip()}", f"+{b.rstrip()}"]
    return text2, diff


def rewrite_tree(root: str, crate: str, published: str, version: str) -> list[str]:
    """Rewrite every manifest under root in place; the report lines."""
    report = []
    for rel, text in st.manifests_on_disk(root):
        new, diff = rewrite_text(text, crate, published, version, f"{os.path.basename(root)}/{rel}")
        if diff:
            with open(os.path.join(root, rel), "w", encoding="utf-8") as f:
                f.write(new)
            report.append(f"--- {os.path.basename(root)}/{rel}")
            report += diff
    return report


def fetch_index(crate: str) -> str:
    with urllib.request.urlopen(f"{INDEX_URL}/{index_path(crate)}", timeout=30) as r:
        return r.read().decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--provider", required=True, help="this checkout")
    ap.add_argument("--tree", action="append", default=[], required=True, help="fetched downstream repository")
    ap.add_argument("--index-file", help="crates.io index entry of the crate (default: fetched)")
    args = ap.parse_args(argv)
    try:
        with open(os.path.join(args.provider, "Cargo.toml"), encoding="utf-8") as f:
            pkg = st.package_of(f.read(), None, "Cargo.toml")
        if pkg is None:
            raise ValueError(f"{args.provider}/Cargo.toml has no [package] name and version")
        crate, version = pkg
        if args.index_file:
            with open(args.index_file, encoding="utf-8") as f:
                index = f.read()
        else:
            index = fetch_index(crate)
        published = newest_published(index)
        if not is_major_bump(published, version):
            print(f"{crate} {version}: not a major bump over the published {published}; "
                  "downstream manifests left as fetched")
            return 0
        print(f"{crate} {version}: major bump over the published {published}; downstream requirements "
              f"that accept {published} are raised to \"{version}\" in the fetched copies")
        n = 0
        for tree in args.tree:
            report = rewrite_tree(tree, crate, published, version)
            n += sum(1 for r in report if r.startswith("+"))
            print("\n".join(report) if report else f"{os.path.basename(os.path.normpath(tree))}: no requirement rewritten")
        print(f"{n} requirement line(s) rewritten")
        # after the edit (or on a rerun over copies already edited) some downstream requirement
        # must accept the new version; none means the bump would be checked against no
        # consumer. A requirement that already accepts it (a range such as ">=3.0, <5", which
        # accepts the published release too) counts: it needs no edit and the build tests it
        on_new = sum(1 for tree in args.tree for rel, text in st.manifests_on_disk(tree)
                     for _, req in st.dependency_requirements(text, crate, rel)
                     if req is not None and st.satisfies(version, req))
        if on_new == 0:
            raise ValueError(f"major bump to {version}, but no downstream requirement on {crate} accepts it "
                             "(0 rewritten and none already accepting)")
        print(f"{on_new} downstream requirement(s) on {crate} accept {version}")
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
