#!/usr/bin/env python3
"""Every file the SCIP index's readers open is an input scripts/land.py knows about.

    python3 scripts/scip_inputs_check.py --scip <index dir> [--patterns scripts/land-scip-inputs.txt]

scripts/land.py skips the SCIP index when a landing's range touches none of the patterns of
scripts/land-scip-inputs.txt. That is only sound if the patterns cover every file the readers of
the index depend on. This runs each reader (the checks land.py runs with --scip, and the
integration-status writer into a temporary file) in its own interpreter under an audit hook,
collects the files of the repository each one opens and the directories it lists, and fails (exit 1) when:
  - a file one of them opened is matched by no pattern (add it to the patterns file);
  - a directory one of them listed is one no pattern can reach;
  - a reader opened no file of the repository, or did not read the index (it compared nothing);
  - a reader did not exit 0 (a crash hides the reads after it) or started a process (not traced);
  - the patterns file cannot be read, holds no pattern, or holds one anchored with a leading or trailing /.
Imports are traced as source: the readers run without bytecode caches.
Byte-compiled caches (__pycache__), .git and the index directory itself (--scip, which CI builds inside the
repository at target/scip) are not inputs: the index is what the readers read, built from the inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import changed_paths as cp  # noqa: E402

TRACER = r"""
import json, os, runpy, sys
root, out, script, *args = sys.argv[1:]
root = os.path.realpath(root)
seen, listed, globbed, spawned = set(), set(), set(), []
def inside(*names):
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if any(n in fn for n in names):
            return True
        f = f.f_back
    return False
def rel(x):
    p = os.path.realpath(os.fsdecode(x))
    return p, (os.path.relpath(p, root).replace(os.sep, "/") if p == root or p.startswith(root + os.sep) else None)
def hook(event, a):
    try:
        if event == "open" and a and isinstance(a[0], (str, bytes, os.PathLike)):
            p, r = rel(a[0])
            seen.add(r if r is not None else "<outside>" + p)
        elif event in ("os.listdir", "os.scandir") and a and isinstance(a[0], (str, bytes, os.PathLike)):
            # the import system's directory scans (sys.path) and glob's own walk are not listings of the reader:
            # imports are traced as the source files they open, a glob as its pattern below
            if not inside("importlib._bootstrap", "importlib/_bootstrap", "zipimport", "glob.py", "pathlib"):
                p, r = rel(a[0])
                if r is not None:
                    listed.add(r)
        elif event == "glob.glob" and a:
            g = os.fsdecode(a[0])
            g = g if os.path.isabs(g) else os.path.join(os.getcwd(), g)
            if g.startswith(root + os.sep):
                globbed.add(os.path.relpath(g, root).replace(os.sep, "/"))
        elif event in ("pathlib.Path.glob", "pathlib.Path.rglob") and len(a) >= 2:
            base = os.path.realpath(os.fsdecode(a[0]))
            pat = ("**/" if event.endswith("rglob") else "") + os.fsdecode(a[1])
            if base == root or base.startswith(root + os.sep):
                globbed.add((os.path.relpath(base, root).replace(os.sep, "/") + "/" + pat).lstrip("./") if base != root else pat)
        elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn"):
            spawned.append(str(a[0] if a else event))
    except Exception:
        pass
sys.addaudithook(hook)
sys.argv = [script, *args]
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
except BaseException as e:
    code = repr(e)
with open(out, "w", encoding="utf-8") as f:
    json.dump({"exit": code, "read": sorted(seen), "listed": sorted(listed), "globbed": sorted(globbed),
               "spawned": spawned}, f)
"""


def readers(scip: str, tmp: str) -> list[list[str]]:
    """the readers of the index, as land.py and CI run them (the writer into a temporary file)"""
    return [["scripts/integration_levels.py", "--check", "--scip", scip],
            ["scripts/scip_reach.py", "--check-baseline", "--scip", scip],
            ["scripts/scip_reach.py", "--write", str(Path(tmp) / "integration-status.md"), "--scip", scip]]


def trace(root: Path, argv: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "trace.json"
        # no bytecode cache: an import then opens its .py, which the hook sees (a cached .pyc hides the source)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPYCACHEPREFIX=str(Path(d) / "pycache"))
        subprocess.run([sys.executable, "-c", TRACER, str(root), str(out), *argv], cwd=root,
                       capture_output=True, text=True, env=env)
        if not out.exists():
            return {"exit": None, "read": [], "listed": [], "globbed": [], "spawned": []}
        return json.loads(out.read_text(encoding="utf-8"))


def is_input(path: str, index_dir: str | None = None) -> bool:
    parts = path.split("/")
    if index_dir and (path == index_dir or path.startswith(index_dir.rstrip("/") + "/")):
        return False
    return "__pycache__" not in parts and parts[0] != ".git"


def index_rel(root: Path, scip: str) -> str | None:
    """the index directory relative to the repository, None when it lies outside"""
    p = Path(scip) if Path(scip).is_absolute() else root / scip
    try:
        return os.path.relpath(os.path.realpath(p), os.path.realpath(root)).replace(os.sep, "/") \
            if os.path.realpath(p).startswith(os.path.realpath(root) + os.sep) else None
    except ValueError:
        return None


PROBE = "zz_probe"


def listing_covered(d: str, rules) -> bool:
    """a directory listed without a filter is covered when any new file in it is an input: a file of any name and any
    extension directly under it is matched"""
    prefix = "" if d in ("", ".") else d.rstrip("/") + "/"
    return all(cp.selected(prefix + name, rules) for name in (PROBE, PROBE + ".qq"))


def glob_covered(g: str, rules) -> bool:
    """a glob is covered when the files it could pick up are inputs: a probe path the glob matches (each * and ** taken
    as one probe name, ** also as two levels) is matched"""
    probes = {g.replace("**", PROBE).replace("*", PROBE)}
    if "**" in g:
        probes.add(g.replace("**", f"{PROBE}/{PROBE}").replace("*", PROBE))
    return all(cp.selected(p, rules) for p in probes)


def problems(read_by: dict[str, list[str]], patterns: list[str], index_dir: str | None = None,
             traces: dict[str, dict] | None = None, index_abs: str | None = None) -> list[str]:
    """why the patterns do not cover what the readers read (empty: they do)"""
    out = []
    if not patterns:
        return ["the patterns file holds no pattern"]
    bad = [p for p in patterns if p.lstrip("!").startswith("/") or p.endswith("/")]
    if bad:
        return [f"a pattern with a leading or trailing / matches nothing: {bad[0]!r}"]
    rules = cp.compile_patterns(patterns)
    for reader, files in read_by.items():
        t = (traces or {}).get(reader, {})
        if traces is not None:
            if t.get("exit") != 0:
                out.append(f"{reader}: exited {t.get('exit')!r}, not 0 (a crash hides the reads after it)")
            for x in t.get("spawned", []):
                out.append(f"{reader}: started a process ({x}), whose reads are not traced")
            for d in t.get("listed", []):
                if not is_input(d + "/x", index_dir):
                    continue
                if not listing_covered(d, rules):
                    out.append(f"{reader}: lists {d or '.'}/, and a new file there is matched by no pattern")
            for g in t.get("globbed", []):
                if not glob_covered(g, rules):
                    out.append(f"{reader}: globs {g}, whose files no pattern matches")
            boundary = (index_dir.rstrip("/") + "/") if index_dir else None
            idx_reads = [f for f in files if (boundary and f.startswith(boundary))
                         or (index_abs and f.startswith("<outside>" + index_abs.rstrip("/") + "/"))]
            if not idx_reads:
                out.append(f"{reader}: did not read the index (compared nothing)")
        repo_files = [f for f in files if not f.startswith("<outside>") and is_input(f, index_dir)]
        if not repo_files:
            out.append(f"{reader}: opened no file of the repository (compared nothing)")
        for f in repo_files:
            if not cp.selected(f, rules):
                out.append(f"{reader}: reads {f}, which no pattern of the inputs file matches")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scip", required=True)
    ap.add_argument("--patterns", default="scripts/land-scip-inputs.txt")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    try:
        patterns = cp.read_patterns(str(root / a.patterns), [])
    except (OSError, UnicodeDecodeError) as e:
        print(f"scip_inputs_check: {a.patterns} cannot be read: {e}", file=sys.stderr)
        return 1
    read_by: dict[str, list[str]] = {}
    traces: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for argv_ in readers(a.scip, tmp):
            t = trace(root, argv_)
            name = " ".join(argv_[:2])
            read_by[name], traces[name] = t["read"], t
    idx = index_rel(root, a.scip)
    index_abs = os.path.realpath(a.scip if Path(a.scip).is_absolute() else root / a.scip)
    errs = problems(read_by, patterns, idx, traces, index_abs)
    for e in errs:
        print(f"scip_inputs_check: {e}", file=sys.stderr)
    n = sum(len([f for f in v if not f.startswith("<outside>") and is_input(f, idx)]) for v in read_by.values())
    print(f"scip_inputs_check: {len(read_by)} readers, {n} files read, {len(patterns)} patterns, {len(errs)} problems")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
