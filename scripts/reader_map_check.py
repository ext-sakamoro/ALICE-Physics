#!/usr/bin/env python3
"""Every tracked file is read by a job that runs on a ci/** push.

A path filter shows that a workflow started, not that anything in it read a given file. This check finds, for every
tracked file, a job of a landing or after-landing workflow (.github/ci-evidence.toml) that reads it. The reading of
"names a file" is the shared scripts/reader_map_core.py: the file's whole path as one token, in a run step outside
its shell comments, or in a script the job runs (named in a run step, or imported by one: Python string values, not
comments or docstrings; shell outside comments).

Derived first: a file is read when a landing job's run step, or a script that job runs, names its path.

Declared (.github/reader-map.toml) only for reads no path names, each with a token check:
  kind = "cargo"        the files cargo reads through a manifest (src/, tests/, benches/, examples/, the manifests):
                        a reader's run step runs cargo with `--manifest-path <manifest>` (`manifest = "Cargo.toml"`
                        is the root workspace: a cargo command with no --manifest-path also reads it)
  kind = "crate-tests"  the files a crate's tests read from a directory: a reader runs `cargo test --manifest-path
                        <crate>/Cargo.toml`, a string literal of a source under <crate>/src names `dir` (its path, or
                        its last component), and every matched file has the `read-ext` extension the tests read
  kind = "dir"          a script a reader runs names the directory `dir` itself (it walks it)
  kind = "all-tracked"  a script a reader runs lists every tracked file (`ls-files` among its string values): for
                        text that a whole-tree check reads (public vocabulary, line endings)
  kind = "pinned-checker"  files a checker of another repository reads, fetched at a pinned revision by a script a
                        reader runs: the script names the `pin` file (`rev <40 hex>`, `checker-sha256 <64 hex>`), the
                        `checker` fetched from `repo` at that rev has that sha256, and its string values name each file
                        (READER_MAP_GIT_BASE overrides https://github.com/ext-sakamoro for the fetch)
  kind = "scheduled"    files only a schedule-only job reads (the weekly run): the same name check against that job,
                        counted apart ("read on schedule only"), since it does not run on ci/**
A `[[unread]]` entry names files no job reads, with the reason (12 characters at least); they are counted, and a
stale one fails.

Fails on an unmapped file, a stale entry, an unknown or non-ci/** reader, a failed token check, 0 tracked files and
0 landing jobs.

  python3 scripts/reader_map_check.py [--root DIR]
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import changed_paths as cp  # noqa: E402
import reader_map_core as core  # noqa: E402  (shared with ALICE-Law)

RUNS_ON_CI = {"landing", "after-landing"}
CARGO = re.compile(r"(?<![\w-])cargo(?:\s+\+\S+)?\s+(test|build|check|clippy|run|doc|rustc|hack|bench|fmt|mutants|deny|fuzz)\b")
SHELL_VAR_DIR = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?/")
RUST_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
SELF = {"reader_map_check.py", "test_reader_map_check.py", "reader_map_core.py", "test_reader_map_core.py"}


def tracked(root: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, check=True)
    return [f for f in out.stdout.split("\0") if f]


def job_runs(job) -> list[str]:
    """the run steps of a job, without their shell comments"""
    steps = job.get("steps") if isinstance(job, dict) else None
    # (a step's `working-directory` is where its run step starts: read as a `cd` in front of it)
    return [SHELL_VAR_DIR.sub(" ", (f"cd {s['working-directory']}\n" if isinstance(s.get("working-directory"), str) else "")
                              + core.without_shell_comments(s["run"])) for s in steps or []
            if isinstance(s, dict) and isinstance(s.get("run"), str)]


def ran_scripts(runs: list[str], scripts: dict[str, Path]) -> tuple[set[str], set[str]]:
    """(the scripts/ files a job runs by name in its run steps, the scripts/ modules those import): an imported module
    is read (it is executed), but its paths count only through the names a script uses from it (core semantics:
    importing a module for something else does not read its files)"""
    stems = {p.stem: n for n, p in scripts.items() if n.endswith(".py")}
    named = {n for r in runs for n in core.RUN_SCRIPT.findall(r) if n in scripts}
    todo = set(named)
    seen: set[str] = set()
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        if n.endswith(".py"):
            try:
                tree = ast.parse(scripts[n].read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                mods = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                    else []
                todo |= {stems[m] for m in mods if m in stems} - seen
    return named, seen - named


def corpus(script: Path, scripts: dict[str, Path], cache: dict[str, list[str]]) -> str:
    """the text a script uses paths in, read once: for Python its string values (core.python_strings) and the string
    values of the module-level names it uses from a scripts/ module it imports (core.uses_imported_path's reading,
    collected instead of asked per path); for a shell script its text outside comments (core.without_shell_comments)"""
    source = script.read_text(encoding="utf-8", errors="replace")
    if script.suffix != ".py":
        # (`"$root/conformance/a.toml"`: a variable that holds a directory, then the path from there)
        return SHELL_VAR_DIR.sub(" ", core.without_shell_comments(source))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    out = core.strings_in(tree)

    def bound(module: str) -> dict[str, list[str]]:
        if module not in cache:
            names: dict[str, list[str]] = {}
            path = script.parent / f"{module}.py"
            try:
                mtree = ast.parse(path.read_text(encoding="utf-8", errors="replace")) if path.is_file() else None
            except SyntaxError:
                mtree = None
            for n in (mtree.body if mtree else []):
                if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for t in targets:
                        if isinstance(t, ast.Name):
                            names[t.id] = core.strings_in(n.value)
            cache[module] = names  # type: ignore[assignment]
        return cache[module]  # type: ignore[return-value]

    aliases: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            b = bound(n.module.split(".")[0])
            for a in n.names:
                out += b.get(a.name, [])
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
            out += bound(aliases[n.value.id]).get(n.attr, [])
    return "\n".join(out)


def fetch_pinned(repo: str, rev: str, path: str) -> bytes | None:
    """the bytes of `path` at `rev` of `repo` (a blob-less fetch of that one commit, as scripts/g11_table.sh does)"""
    import os as _os
    import tempfile
    base = _os.environ.get("READER_MAP_GIT_BASE", "https://github.com/ext-sakamoro")
    with tempfile.TemporaryDirectory() as d:
        if subprocess.run(["git", "init", "-q", d], capture_output=True).returncode != 0:
            return None
        if subprocess.run(["git", "-C", d, "fetch", "-q", "--depth", "1", f"{base}/{repo}", rev],
                          capture_output=True).returncode != 0:
            return None
        p = subprocess.run(["git", "-C", d, "show", f"FETCH_HEAD:{path}"], capture_output=True)
        return p.stdout if p.returncode == 0 else None


def pinned_checker_problem(root: Path, e: dict, files: list[str], hits: list[int], runs: list[str], ran: set[str],
                           texts: dict[str, str]) -> str | None:
    import hashlib
    pin, repo, checker = e.get("pin", ""), e.get("repo", ""), e.get("checker", "")
    if not (pin and repo and checker):
        return "needs pin, repo and checker"
    if not (any(core.token(pin).search(x) for x in runs) or any(core.token(pin).search(texts.get(s, "")) for s in ran)):
        return f"runs nothing that names {pin}"
    try:
        fields = dict(line.split(None, 1) for line in (root / pin).read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, ValueError):
        return f"{pin} is not readable"
    rev, want = fields.get("rev", "").strip(), fields.get("checker-sha256", "").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", rev) or not re.fullmatch(r"[0-9a-f]{64}", want):
        return f"{pin} needs `rev <40 hex>` and `checker-sha256 <64 hex>`"
    data = fetch_pinned(repo, rev, checker)
    if data is None:
        return f"cannot fetch {repo} {rev[:12]}:{checker}"
    if hashlib.sha256(data).hexdigest() != want:
        return f"{repo} {rev[:12]}:{checker} does not have the pinned sha256"
    strings = "\n".join(core.python_strings(data.decode("utf-8", errors="replace")))
    missing = [files[k] for k in hits if not core.token(files[k]).search(strings)]
    return f"the pinned checker does not name {', '.join(missing[:3])}" if missing else None


def rust_strings(root: Path, files: list[str], crate: str) -> list[str]:
    out = []
    for f in files:
        if f.startswith(f"{crate}/src/") and f.endswith(".rs"):
            text = (root / f).read_text(encoding="utf-8", errors="replace")
            text = re.sub(r"(?m)^\s*//.*$", "", text)  # (a comment line holds no literal)
            out += RUST_STRING.findall(text)
    return out


def check(root: Path) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    files = tracked(root)
    counts = {"files": len(files), "jobs": 0, "derived": 0, "declared": 0, "scheduled": 0, "unread": 0}
    if not files:
        return ["no tracked file was read (compared 0)"], counts
    classes = tomllib.loads((root / ".github" / "ci-evidence.toml").read_text(encoding="utf-8")).get("workflows", {})
    scripts = {f[len("scripts/"):]: root / f for f in files
               if f.startswith("scripts/") and "/" not in f[len("scripts/"):] and f.endswith((".py", ".sh"))}
    jobs: dict[str, tuple[str, list[str], set[str]]] = {}
    for wf, cls in classes.items():
        try:
            tree = cp.load_yaml((root / ".github" / "workflows" / wf).read_text(encoding="utf-8"))
        except (OSError, cp.FormError) as e:
            errors.append(f"{wf}: cannot read ({e})")
            continue
        for jid, job in (tree.get("jobs") or {}).items():
            runs = job_runs(job)
            named, imported = ran_scripts(runs, scripts)
            jobs[f"{wf}:{jid}"] = (cls, runs, named, imported)
    landing = {k: v for k, v in jobs.items() if v[0] in RUNS_ON_CI}
    all_imported = {s for _, _, _, imp in landing.values() for s in imp}
    counts["jobs"] = len(landing)
    if not landing:
        return errors + ["no job of a landing workflow was read (compared 0)"], counts
    run_text = "\n".join(r for _, runs, _, _ in landing.values() for r in runs)
    all_ran = {s for _, _, ran, _ in landing.values() for s in ran}
    cache: dict[str, list[str]] = {}
    texts: dict[str, str] = {}
    # a script under scripts/ names a sibling by its file name (`$here/x.py`, `HERE / "x.txt"`): a sibling script it
    # names is run by it (and read in turn), a sibling data file is read
    data_names = {f[len("scripts/"):] for f in files if f.startswith("scripts/") and "/" not in f[len("scripts/"):]}
    sibling_dirs = {f[len("scripts/"):].split("/")[0] for f in files if f.count("/") >= 2 and f.startswith("scripts/")}
    sibling_read: set[str] = set()
    todo = sorted(all_ran)
    while todo:
        n = todo.pop()
        if n in texts:
            continue
        texts[n] = corpus(scripts[n], scripts, cache)
        if n in SELF:
            continue
        for d in sibling_dirs:
            if core.token(d).search(texts[n]):
                sibling_read.add(d + "/")
        for m in data_names:
            if m != n and core.token(m).search(texts[n]):
                sibling_read.add(m)
                if m in scripts and m not in all_ran:
                    all_ran.add(m)
                    todo.append(m)
    readers = sorted(all_ran - SELF)
    used = run_text + "\n" + "\n".join(texts[s] for s in readers)
    covered = [False] * len(files)
    cargo_read = [False] * len(files)
    if "reader_map_check.py" in all_ran and ".github/reader-map.toml" in files:
        covered[files.index(".github/reader-map.toml")] = True  # (the map is read by the check that runs on it)
    for k, f in enumerate(files):
        name = f[len("scripts/"):] if f.startswith("scripts/") else None
        if name in all_ran or name in all_imported or name in sibling_read or \
                (name and name.split("/")[0] + "/" in sibling_read):
            covered[k] = True  # a script a landing job runs is read by running it; a sibling it names is read
        elif core.token(f).search(used):
            covered[k] = True
        if covered[k]:
            counts["derived"] += 1
    # a directory path of two components or more that a reader uses (`ROOT / "docs" / "coverage"`) is read whole
    dirs = sorted({str(Path(f).parents[i]) for f in files for i in range(len(Path(f).parents) - 1)
                   if str(Path(f).parents[i]).count("/") >= 1})
    for d in dirs:
        if core.token(d).search(used):
            for k, f in enumerate(files):
                if f.startswith(d + "/") and not covered[k]:
                    covered[k] = True
                    counts["derived"] += 1
    try:
        data = tomllib.loads((root / ".github" / "reader-map.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return errors + [f".github/reader-map.toml: {e}"], counts
    for i, e in enumerate(data.get("entry", [])):
        where = f"entry {i + 1} ({e.get('name', '?')})"
        try:
            rules = cp.compile_patterns(list(e.get("paths", [])))
        except cp.PatternError as err:
            errors.append(f"{where}: {err}")
            continue
        hits = [k for k, f in enumerate(files) if cp.selected(f, rules)]
        if not hits:
            errors.append(f"{where}: matches no tracked file (a stale entry)")
            continue
        kind, rs = e.get("kind"), e.get("readers", [])
        if not rs:
            errors.append(f"{where}: no reader")
            continue
        ok = True
        for r in rs:
            if r not in jobs:
                errors.append(f"{where}: reader {r} is not a job of a workflow in .github/ci-evidence.toml")
                ok = False
                continue
            cls, runs, ran, _ = jobs[r]
            if kind == "scheduled":
                if cls != "schedule-only":
                    errors.append(f"{where}: reader {r} is class {cls}, not schedule-only (a landing reader is found by name)")
                    ok = False
                    continue
                text = "\n".join(runs) + "\n" + "\n".join(corpus(scripts[x], scripts, cache) for x in ran if x not in SELF)
                sib = "\n".join(corpus(scripts[x], scripts, cache) for x in ran if x not in SELF)
                missing = [files[k] for k in hits if not core.token(files[k]).search(text)
                           and not (files[k].startswith("scripts/") and "/" not in files[k][len("scripts/"):]
                                    and core.token(files[k][len("scripts/"):]).search(sib))]
                if missing:
                    errors.append(f"{where}: reader {r} does not name {', '.join(missing[:3])}")
                    ok = False
                continue
            if cls not in RUNS_ON_CI:
                errors.append(f"{where}: reader {r} is class {cls} (it does not run on ci/**)")
                ok = False
                continue
            if kind == "cargo":
                manifest = e.get("manifest", "")
                if manifest == "Cargo.toml":
                    good = any(CARGO.search(x) and "--manifest-path" not in x for x in runs)
                else:
                    d = str(Path(manifest).parent)
                    good = bool(manifest) and any(CARGO.search(x) and (core.token(manifest).search(x) and "--manifest-path" in x
                                                                       or re.search(rf"(?m)^\s*cd {re.escape(d)}\s*(?:&&|$)", x))
                                                  for x in runs)
                if not good:
                    errors.append(f"{where}: reader {r} runs no cargo command on {manifest or '(no manifest)'}")
                    ok = False
            elif kind == "crate-tests":
                crate = e.get("crate", "")
                if not any(re.search(r"cargo\s+test\b", x) and core.token(f"{crate}/Cargo.toml").search(x) for x in runs):
                    errors.append(f"{where}: reader {r} does not run `cargo test --manifest-path {crate}/Cargo.toml`")
                    ok = False
            elif kind == "dir":
                d = e.get("dir", "")
                if not d or not (any(core.token(d).search(x) for x in runs)
                                 or any(core.token(d).search(texts[s]) for s in ran if s not in SELF)):
                    errors.append(f"{where}: reader {r} names no `{d}` (its run steps and the scripts they run)")
                    ok = False
            elif kind == "pinned-checker":
                why = pinned_checker_problem(root, e, files, hits, runs, ran, texts)
                if why:
                    errors.append(f"{where}: reader {r}: {why}")
                    ok = False
            elif kind == "all-tracked":
                if not any("ls-files" in core.python_strings(scripts[s].read_text(encoding="utf-8", errors="replace"))
                           for s in ran if s.endswith(".py")):
                    errors.append(f"{where}: reader {r} runs no script that lists the tracked files (git ls-files)")
                    ok = False
            else:
                errors.append(f"{where}: unknown kind {kind!r}")
                ok = False
        if kind == "crate-tests" and ok:
            crate, d, ext = e.get("crate", ""), e.get("dir", ""), e.get("read-ext")
            lits = rust_strings(root, files, crate)
            last = d.rstrip("/").split("/")[-1] if d else ""
            if not lits:
                errors.append(f"{where}: no string literal in {crate}/src")
                ok = False
            elif not d or not any(core.token(d).search(x) or core.token(last).search(x) for x in lits):
                errors.append(f"{where}: no string literal of {crate}/src names `{d}`")
                ok = False
            odd = [files[k] for k in hits if ext and not files[k].endswith(f".{ext}")]
            if odd:
                errors.append(f"{where}: {len(odd)} matched file(s) are not .{ext}, which the tests skip: "
                              f"{', '.join(odd[:3])}")
                ok = False
        if ok:
            for k in hits:
                if not covered[k]:
                    counts["scheduled" if kind == "scheduled" else "declared"] += 1
                covered[k] = True
                if kind == "cargo":
                    cargo_read[k] = True
    # a Rust source cargo reads (covered by a cargo entry) reads the files its string literals name relative to its own
    # directory (`include_bytes!("fixtures/a.bin")`, `#[path = "support/x.rs"]`) or to its crate's manifest directory
    # (`env!("CARGO_MANIFEST_DIR")` joined with "../probes"): a directory named so is read whole (a read_dir)
    index = {f: k for k, f in enumerate(files)}
    manifests = sorted({str(Path(f).parent) for f in files if f.endswith("Cargo.toml")}, key=len, reverse=True)
    for k, f in enumerate(files):
        if not (covered[k] and f.endswith(".rs") and cargo_read[k]):
            continue
        crate = next((m for m in manifests if m == "." or f.startswith(m + "/")), ".")
        text = re.sub(r"(?m)^\s*//.*$", "", (root / f).read_text(encoding="utf-8", errors="replace"))
        for lit in set(RUST_STRING.findall(text)):
            if not lit or lit.startswith(("http", "-")) or " " in lit:
                continue
            for base in {str(Path(f).parent), crate, "."}:
                target = os.path.normpath(os.path.join(base, lit))
                if target in index and not covered[index[target]]:
                    covered[index[target]] = True
                    counts["derived"] += 1
                elif target + "/" and any(g.startswith(target + "/") for g in files):
                    for g in files:
                        if g.startswith(target + "/") and not covered[index[g]]:
                            covered[index[g]] = True
                            counts["derived"] += 1
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
                counts["unread"] += 1
    for k, f in enumerate(files):
        if not covered[k]:
            errors.append(f"{f}: no reader (no landing job names it; declare the read in .github/reader-map.toml)")
    return errors, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    a = ap.parse_args(argv)
    errors, c = check(a.root)
    for e in errors:
        print(f"reader-map: {e}", file=sys.stderr)
    print(f"compared: tracked files {c['files']}, landing jobs {c['jobs']}, read by name {c['derived']}, "
          f"declared {c['declared']}, read on schedule only {c['scheduled']}, unread by decision {c['unread']}")
    if c["files"] == 0 or c["jobs"] == 0:
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
