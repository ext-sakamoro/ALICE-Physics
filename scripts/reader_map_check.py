#!/usr/bin/env python3
"""Every tracked file is read by a check that runs on a ci/** push (.github/reader-map.toml).

A path filter shows that a workflow started, not that anything in it read a given file. The
reader map names, for every tracked file, the jobs that read it, and this check holds the
map to the workflows:

  * every tracked file is matched by at least one entry (an unmapped file fails);
  * every entry matches at least one tracked file (a stale entry fails);
  * every reader `<workflow>:<job>` is a job of a workflow whose class in
    .github/ci-evidence.toml is landing or after-landing (it runs on ci/**);
  * every entry's `evidence` (a string) appears in each of its readers: in the job's own
    text (its steps' run / uses / with), or in a file under scripts/ that the job names,
    followed through the scripts/ files those name in turn;
  * an entry with `scripts = true` is matched file by file instead: each file under
    scripts/ it covers must be run by some landing job (named in its steps, or imported by
    a script it runs), or be named by such a script (its data: baselines, lists, tables);
    a file no job reaches fails, naming it;
  * `paths-list = "<name>"` takes the entry's patterns from .github/paths/<name>.txt
    (the path gate's own list), so the lists are not copied here;
  * an entry with `crate-tests = "<dir>"` is read by the tests of the cargo crate at <dir>:
    each reader must run `cargo test` with `--manifest-path <dir>/Cargo.toml`, a source file
    of the crate under <dir>/src must name the entry's `evidence` (the directory its tests
    read), and with `read-ext = "<ext>"` every matched file must have that extension (the
    tests read the `.<ext>` files of the directory: a file of another kind would be matched
    here but never read), at least one;
  * a `[[unread]]` entry names files no ci/** job reads, with the `reason` they stay so
    (a decision, at least 12 characters): they are counted and printed, not passed silently,
    and a stale one (matching no file, or a file some reader entry covers) fails.

Fails (exit 1) on a finding, and when no tracked file or no entry was read.

  python3 scripts/reader_map_check.py           # exit 1 on a finding
  python3 scripts/reader_map_check.py --root DIR
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import changed_paths as cp  # noqa: E402

RUNS_ON_CI = {"landing", "after-landing"}
SCRIPT_NAME = re.compile(r"(?<![\w./-])(?:\./)?scripts/([A-Za-z0-9_.-]+)")
MODULE_IMPORT = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


def tracked(root: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, check=True)
    return [f for f in out.stdout.split("\0") if f]


def job_text(job) -> str:
    """the text of a job a reader can be seen in: each step's run, uses and with values"""
    parts = []
    for s in (job.get("steps") or []) if isinstance(job, dict) else []:
        if isinstance(s, dict):
            for k in ("run", "uses"):
                if isinstance(s.get(k), str):
                    parts.append(s[k])
            if isinstance(s.get("with"), dict):
                parts.extend(str(v) for v in s["with"].values())
    return "\n".join(parts)


def scripts_named(text: str, scripts: dict[str, str]) -> set[str]:
    """the files under scripts/ that `text` names (by path, or a python import of a module there)"""
    found = {n for n in SCRIPT_NAME.findall(text) if n in scripts}
    stems = {Path(n).stem: n for n in scripts if n.endswith(".py")}
    found |= {stems[m] for m in MODULE_IMPORT.findall(text) if m in stems}
    return found


def reach(text: str, scripts: dict[str, str]) -> set[str]:
    """the scripts/ files a job runs: those its own text names, and the python modules those
    import, followed through their imports (a script that only mentions another file's path,
    as land.py and the ledgers do for dozens, does not make the job a reader of it)"""
    stems = {Path(n).stem: n for n in scripts if n.endswith(".py")}
    seen: set[str] = set()
    todo = list(scripts_named(text, scripts))
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        todo.extend({stems[m] for m in MODULE_IMPORT.findall(scripts[n]) if m in stems} - seen)
    return seen


def check(root: Path) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    files = tracked(root)
    counts = {"files": len(files), "entries": 0, "readers": 0, "scripts": 0, "unread": 0}
    if not files:
        return ["no tracked file was read (compared 0)"], counts
    mapfile = root / ".github" / "reader-map.toml"
    try:
        data = tomllib.loads(mapfile.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return [f"{mapfile.relative_to(root)}: {e}"], counts
    entries = data.get("entry", [])
    counts["entries"] = len(entries)
    if not entries:
        return ["the reader map has no entry (compared 0)"], counts
    classes = tomllib.loads((root / ".github" / "ci-evidence.toml").read_text(encoding="utf-8")).get("workflows", {})
    scripts = {}
    for f in files:
        if f.startswith("scripts/") and "/" not in f[len("scripts/"):]:
            try:
                scripts[f[len("scripts/"):]] = (root / f).read_text(encoding="utf-8", errors="replace")
            except OSError:
                scripts[f[len("scripts/"):]] = ""
    jobs: dict[str, str] = {}
    landing_reach: set[str] = set()
    for wf, cls in classes.items():
        try:
            tree = cp.load_yaml((root / ".github" / "workflows" / wf).read_text(encoding="utf-8"))
        except (OSError, cp.FormError) as e:
            errors.append(f"{wf}: cannot read ({e})")
            continue
        for jid, job in (tree.get("jobs") or {}).items():
            text = job_text(job)
            jobs[f"{wf}:{jid}"] = (cls, text)
            if cls in RUNS_ON_CI:
                ran = reach(text, scripts)
                # a script a job runs reads the scripts/ files it names (its baselines, lists, tables)
                landing_reach |= ran | {m for n in ran for m in scripts_named(scripts[n], scripts)} \
                    | {m for m in scripts if any(re.search(rf"(?<![\w.-]){re.escape(m)}(?![\w-])", scripts[n])
                                                 for n in ran)}
    covered = [False] * len(files)
    unreached: dict[int, str] = {}  # scripts/ files the `scripts = true` entry could not show read
    for i, e in enumerate(entries):
        where = f"entry {i + 1} ({e.get('name', '?')})"
        pats = list(e.get("paths", []))
        if "paths-list" in e:
            lst = root / ".github" / "paths" / f"{e['paths-list']}.txt"
            if not lst.is_file():
                errors.append(f"{where}: no .github/paths/{e['paths-list']}.txt")
                continue
            pats += cp.read_patterns(str(lst), [])
        try:
            rules = cp.compile_patterns(pats)
        except cp.PatternError as err:
            errors.append(f"{where}: {err}")
            continue
        hits = [k for k, f in enumerate(files) if cp.selected(f, rules)]
        if not hits:
            errors.append(f"{where}: matches no tracked file (a stale entry)")
            continue
        if e.get("scripts") is True:
            for k in hits:
                f = files[k]
                name = f[len("scripts/"):] if f.startswith("scripts/") else None
                counts["scripts"] += 1
                if name is not None and name in landing_reach:
                    covered[k] = True
                else:
                    unreached[k] = where
            continue
        readers = e.get("readers", [])
        evidence = e.get("evidence")
        crate = e.get("crate-tests")
        if not readers:
            errors.append(f"{where}: no reader")
            continue
        if not isinstance(evidence, str) or not evidence:
            errors.append(f"{where}: no evidence string")
            continue
        ok = True
        for r in readers:
            counts["readers"] += 1
            if r not in jobs:
                errors.append(f"{where}: reader {r} is not a job of a workflow in .github/ci-evidence.toml")
                ok = False
                continue
            cls, text = jobs[r]
            if cls not in RUNS_ON_CI:
                errors.append(f"{where}: reader {r} is class {cls} (it does not run on ci/**)")
                ok = False
                continue
            if crate is not None:
                if not re.search(rf"cargo test\b[^\n]*--manifest-path {re.escape(crate)}/Cargo\.toml", text):
                    errors.append(f"{where}: reader {r} does not run `cargo test --manifest-path {crate}/Cargo.toml`")
                    ok = False
                continue
            seen = text + "\n" + "\n".join(scripts[n] for n in reach(text, scripts))
            if evidence not in seen:
                errors.append(f"{where}: reader {r} does not show `{evidence}` (its steps and the scripts they run)")
                ok = False
        if crate is not None:
            srcs = [f for f in files if f.startswith(f"{crate}/src/") and f.endswith(".rs")]
            if not srcs:
                errors.append(f"{where}: {crate}/src has no tracked .rs file")
                ok = False
            elif not any(evidence in (root / f).read_text(encoding="utf-8", errors="replace") for f in srcs):
                errors.append(f"{where}: no source of {crate} names `{evidence}` (its tests do not read the files)")
                ok = False
            ext = e.get("read-ext")
            if ext:
                odd = [files[k] for k in hits if not files[k].endswith(f".{ext}")]
                if odd:
                    errors.append(f"{where}: {len(odd)} matched file(s) are not .{ext}, which the tests skip: "
                                  f"{', '.join(odd[:3])}")
                    ok = False
            counts["crate_files"] = counts.get("crate_files", 0) + len(hits)
        if ok:
            for k in hits:
                covered[k] = True
    acknowledged = 0
    for i, u in enumerate(data.get("unread", [])):
        where = f"unread {i + 1}"
        reason = u.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 12:
            errors.append(f"{where}: no reason (at least 12 characters)")
            continue
        try:
            rules = cp.compile_patterns(list(u.get("paths", [])))
        except cp.PatternError as err:
            errors.append(f"{where}: {err}")
            continue
        hits = [k for k, f in enumerate(files) if cp.selected(f, rules)]
        if not hits:
            errors.append(f"{where}: matches no tracked file (a stale entry)")
        for k in hits:
            if covered[k]:
                errors.append(f"{where}: {files[k]} has a reader (the unread entry is stale for it)")
            else:
                covered[k] = True
                unreached.pop(k, None)
                acknowledged += 1
    counts["unread"] = acknowledged
    for k, f in enumerate(files):
        if covered[k]:
            continue
        if k in unreached:
            errors.append(f"{unreached[k]}: {f} is run or named by no job that runs on ci/** (and no other entry "
                          "names a reader)")
        else:
            errors.append(f"{f}: no reader (add it to .github/reader-map.toml with the job that reads it)")
    return errors, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args(argv)
    errors, c = check(a.root)
    for e in errors:
        print(f"reader-map: {e}", file=sys.stderr)
    print(f"compared: tracked files {c['files']}, entries {c['entries']}, readers {c['readers']}, scripts {c['scripts']}, "
          f"unread by decision {c['unread']}")
    if c["files"] == 0 or c["entries"] == 0:
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
