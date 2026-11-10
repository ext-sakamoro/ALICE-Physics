#!/usr/bin/env python3
"""Second pass of a mutation run: the mutants `--cargo-arg=--lib` missed, run again
with the integration test targets that can reach them.

The first pass (cargo-mutants with `--cargo-arg=--lib`) tests every mutant with the
lib tests only, which is cheap. A mutant it misses may still be killed by an
integration test in tests/. A test target can only kill a mutant whose line it
executes, so the second pass runs each missed mutant once, with the lib tests and
the targets that cover it. The result is the one a run with every target would
give, at a fraction of the cost: a mutant the lib tests catch is caught either way.

Which targets cover a mutant comes from a coverage map (`map`): every test binary
of `cargo test --lib --tests` is run alone under llvm instrumentation, and the map
records, for each source line, the targets that executed it, and for each function
(as cargo-mutants names it, with its span from `cargo mutants --list --json`), the
targets that executed any of its lines. `run` selects, for each missed mutant:

  * the targets of its line, when the map was taken at the commit being tested
    (the weekly run builds the map at its own commit);
  * otherwise (the nightly in-diff run reads the last weekly map) the targets of
    its function, which is a superset of the line's, so the result stays exact; a
    function the map does not know (added since) takes every target;
  * in both cases, every target whose files changed between the map's commit and
    the commit being tested: a test added or edited since the map is not in it.
    A changed file under tests/ that is not one target's (a shared helper module)
    takes every target.

Each mutant is run alone (`cargo mutants --re '^<name>$' --baseline=skip`); only the
outcome of that mutant is read (cargo-mutants can add other mutants to a filtered
run). The output directory is the first pass's, with its missed mutants replaced by
the second pass's outcomes, so scripts/mutants_ratchet.py reads it as one run. A
missed mutant of the first pass without an outcome in the second fails `run`.

  python3 scripts/mutants_pass2.py map --out coverage-map.json [--features F]
  python3 scripts/mutants_pass2.py run --map coverage-map.json --pass1 mutants.out \\
      --out mutants-pass2.out [--features F]
  python3 scripts/mutants_pass2.py run --map - ...    # no map yet: every target
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTCOMES = ("caught", "missed", "timeout", "unviable")
POS_RE = re.compile(r"^(?P<file>\S+?\.rs):(?P<line>\d+):(?P<col>\d+): (?P<what>.*)$")
FN_RE = re.compile(r"^.* in (?P<fn>\S.*)$")
MAP_VERSION = 1


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def parse_mutant(name: str) -> tuple[str, int, str | None]:
    """(file, line, function) of a cargo-mutants mutant name."""
    m = POS_RE.match(name.strip())
    if not m:
        raise ValueError(f"not a mutant name: {name!r}")
    f = FN_RE.match(m["what"])
    return m["file"], int(m["line"]), f["fn"] if f else None


def target_of_test_path(path: str) -> str | None:
    """The integration test target a tests/ path belongs to: tests/<t>.rs and
    tests/<t>/main.rs are target <t>; any other path under tests/ is shared (None)."""
    parts = path.split("/")
    if len(parts) == 2 and parts[0] == "tests" and parts[1].endswith(".rs"):
        return parts[1][:-3]
    if len(parts) == 3 and parts[0] == "tests" and parts[2] == "main.rs":
        return parts[1]
    return None


def changed_targets(changed: list[str], all_targets: list[str]) -> set[str]:
    """Targets to run because their files changed since the map (see the module doc)."""
    picked: set[str] = set()
    for p in changed:
        if not p.startswith("tests/"):
            continue
        t = target_of_test_path(p)
        if t is None:
            return set(all_targets)
        picked.add(t)
    return picked


def select(cov: dict, name: str, same_commit: bool, changed: set[str]) -> list[str]:
    """The integration targets to run one missed mutant with."""
    file, line, fn = parse_mutant(name)
    names = cov["targets"]
    if same_commit:
        idx = cov["lines"].get(file, {}).get(str(line), [])
        base = {names[i] for i in idx}
    else:
        fns = cov["fns"].get(file, {})
        if fn is None or fn not in fns:
            base = set(names)
        else:
            base = {names[i] for i in fns[fn]}
    # a changed target may be new (not in the map's list): it is run all the same
    return sorted(base | changed)


def changed_since(commit: str, head: str, cwd: Path = ROOT) -> list[str]:
    """Paths under tests/ that differ between the map's commit and head (added, edited,
    renamed or removed)."""
    if commit == head:
        return []
    return git("diff", "--name-only", "--no-renames", f"{commit}..{head}", "--", "tests", cwd=cwd).split()


# --- map -----------------------------------------------------------------------------

def llvm_bin() -> Path:
    sysroot = subprocess.run(["rustc", "--print", "sysroot"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
    found = sorted(Path(sysroot).glob("lib/rustlib/*/bin/llvm-profdata"))
    if not found:
        raise SystemExit("error: llvm-profdata not found (rustup component add llvm-tools)")
    return found[0].parent


def test_binaries(env: dict, features: str) -> list[tuple[str, str]]:
    """(target name, executable) of every lib / integration test binary."""
    cmd = ["cargo", "test", "--lib", "--tests", "--no-run", "--message-format=json"]
    if features:
        cmd += ["--features", features]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"error: building the test binaries failed:\n{r.stderr[-2000:]}")
    out = []
    for line in r.stdout.splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("reason") == "compiler-artifact" and m.get("executable") and m["profile"]["test"]:
            kind = m["target"]["kind"][0]
            out.append(("LIB" if kind == "lib" else m["target"]["name"], m["executable"]))
    return out


def function_spans(features: str) -> dict[str, dict[str, tuple[int, int]]]:
    """{file: {function name as cargo-mutants writes it: (first line, last line)}}."""
    cmd = ["cargo", "mutants", "--list", "--json"]
    if features:
        cmd += ["--features", features]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"error: cargo mutants --list failed:\n{r.stderr[-2000:]}")
    spans: dict[str, dict[str, tuple[int, int]]] = {}
    for m in json.loads(r.stdout):
        f = m.get("function")
        if not f:
            continue
        s = f["span"]
        spans.setdefault(m["file"], {})[f["function_name"]] = (s["start"]["line"], s["end"]["line"])
    return spans


def covered_lines(lcov: str) -> dict[str, set[int]]:
    """{src/ path: lines executed at least once} from an lcov export."""
    out: dict[str, set[int]] = {}
    cur = None
    for line in lcov.splitlines():
        if line.startswith("SF:"):
            p = line[3:]
            i = p.rfind("/src/")
            cur = "src/" + p[i + 5:] if i >= 0 else None
        elif line.startswith("DA:") and cur:
            ln, count = line[3:].split(",")[:2]
            if int(count) > 0:
                out.setdefault(cur, set()).add(int(ln))
    return out


def cmd_map(args: argparse.Namespace) -> int:
    env = dict(os.environ)
    shown = subprocess.run(["cargo", "llvm-cov", "show-env", "--sh"], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout
    for line in shown.splitlines():
        m = re.match(r'^export (\w+)=(.*)$', line.strip())
        if m:
            v = m[2]
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            env[m[1]] = v
    binaries = test_binaries(env, args.features)
    if not binaries:
        raise SystemExit("error: no test binary was built (compared nothing)")
    tools = llvm_bin()
    names = sorted({n for n, _ in binaries if n != "LIB"})
    index = {n: i for i, n in enumerate(names)}
    lines: dict[str, dict[str, set[int]]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n, exe in binaries:
            if n == "LIB":
                continue
            raw = Path(tmp) / n
            raw.mkdir()
            run_env = dict(env, LLVM_PROFILE_FILE=str(raw / "%p-%m.profraw"))
            r = subprocess.run([exe, "-q"], cwd=ROOT, env=run_env, capture_output=True,
                               timeout=args.target_timeout)
            if r.returncode != 0:
                raise SystemExit(f"error: test target {n} failed under coverage (exit {r.returncode})")
            profs = list(raw.glob("*.profraw"))
            if not profs:
                raise SystemExit(f"error: test target {n} wrote no profile")
            prof = Path(tmp) / f"{n}.profdata"
            subprocess.run([str(tools / "llvm-profdata"), "merge", "-sparse", *map(str, profs),
                            "-o", str(prof)], check=True, capture_output=True)
            ex = subprocess.run([str(tools / "llvm-cov"), "export", "-format=lcov",
                                 "-instr-profile", str(prof), exe], capture_output=True, text=True)
            if ex.returncode != 0 and "no coverage data found" in ex.stderr:
                # a target whose tests are all behind a feature this axis does not turn on
                # (0 tests ran): it executes no line
                lcov = ""
            elif ex.returncode != 0:
                raise SystemExit(f"error: llvm-cov export of {n} failed: {ex.stderr.strip()[-500:]}")
            else:
                lcov = ex.stdout
            for f, ls in covered_lines(lcov).items():
                per = lines.setdefault(f, {})
                for ln in ls:
                    per.setdefault(str(ln), set()).add(index[n])
            shutil.rmtree(raw)
            prof.unlink()
    spans = function_spans(args.features)
    fns: dict[str, dict[str, list[int]]] = {}
    for f, by_fn in spans.items():
        per = lines.get(f, {})
        for fn, (a, b) in by_fn.items():
            hit: set[int] = set()
            for ln in range(a, b + 1):
                hit |= per.get(str(ln), set())
            fns.setdefault(f, {})[fn] = sorted(hit)
    covered = sum(len(v) for v in lines.values())
    if covered == 0:
        raise SystemExit("error: the map covers no line (compared nothing)")
    doc = {
        "version": MAP_VERSION,
        "commit": git("rev-parse", "HEAD").strip(),
        "features": args.features,
        "targets": names,
        "lines": {f: {ln: sorted(t) for ln, t in sorted(per.items(), key=lambda x: int(x[0]))}
                  for f, per in sorted(lines.items())},
        "fns": fns,
    }
    Path(args.out).write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    print(f"coverage map: {len(names)} targets, {len(lines)} files, {covered} covered lines, "
          f"{sum(len(v) for v in fns.values())} functions, commit {doc['commit'][:12]}")
    return 0


# --- run -----------------------------------------------------------------------------

def read_list(p: Path) -> list[str]:
    return [l.strip() for l in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()] if p.is_file() else []


def outcome_of(out: Path, name: str) -> str | None:
    """The outcome cargo-mutants gave this exact mutant (other mutants it added are ignored)."""
    for k in OUTCOMES:
        if name in read_list(out / f"{k}.txt"):
            return k
    return None


def run_one(name: str, targets: list[str], features: str, out: Path, timeout: int) -> str | None:
    file, _, _ = parse_mutant(name)
    cmd = ["cargo", "mutants", "-f", file, "--re", "^" + re.escape(name) + "$", "--baseline=skip",
           "-j", "1", "--timeout", str(timeout), "--build-timeout", str(timeout),
           "--output", str(out), "--cargo-arg=--lib", *[f"--cargo-arg=--test={t}" for t in targets]]
    if features:
        cmd += ["--features", features]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return outcome_of(out / "mutants.out", name)


def no_map(features: str) -> dict:
    """The map to use when there is none yet (`--map -`): it knows no function, so every
    mutant runs with every integration target (exact, slower)."""
    names = sorted({t for p in (ROOT / "tests").rglob("*.rs")
                    if (t := target_of_test_path(p.relative_to(ROOT).as_posix()))})
    if not names:
        raise SystemExit("error: no integration test target under tests/ (compared nothing)")
    return {"version": MAP_VERSION, "commit": "", "features": features, "targets": names,
            "lines": {}, "fns": {}}


def cmd_run(args: argparse.Namespace) -> int:
    cov = no_map(args.features) if args.map == "-" else json.loads(Path(args.map).read_text(encoding="utf-8"))
    if cov.get("version") != MAP_VERSION:
        raise SystemExit(f"error: coverage map version {cov.get('version')}, expected {MAP_VERSION}")
    if cov.get("features", "") != args.features:
        raise SystemExit(f"error: the map is for features {cov.get('features')!r}, this run is {args.features!r}")
    head = git("rev-parse", "HEAD").strip()
    same = cov["commit"] == head
    changed = changed_targets(changed_since(cov["commit"], head), cov["targets"]) if cov["commit"] else set()
    pass1 = Path(args.pass1)
    p1 = pass1 / "mutants.out" if (pass1 / "mutants.out").is_dir() else pass1
    missed = read_list(p1 / "missed.txt")
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(p1, out)
    results: dict[str, list[str]] = {k: [] for k in OUTCOMES}
    missing = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, name in enumerate(missed):
            targets = select(cov, name, same, changed)
            if not targets:
                results["missed"].append(name)       # no target executes it: nothing more to run
                print(f"[{i + 1}/{len(missed)}] missed (no covering target): {name}", flush=True)
                continue
            got = run_one(name, targets, args.features, Path(tmp) / str(i), args.timeout)
            if got is None:
                missing.append(name)
                continue
            results[got].append(name)
            print(f"[{i + 1}/{len(missed)}] {got} ({len(targets)} targets): {name}", flush=True)
    for k in OUTCOMES:
        base = [] if k == "missed" else read_list(out / f"{k}.txt")
        (out / f"{k}.txt").write_text("".join(f"{l}\n" for l in base + results[k]), encoding="utf-8")
    summary = {"map_commit": cov["commit"], "commit": head, "same_commit": same,
               "changed_targets": sorted(changed), "pass1_missed": len(missed),
               **{k: len(v) for k, v in results.items()}, "without_outcome": missing}
    (out / "pass2.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"pass 2: {len(missed)} lib-missed: " + ", ".join(f"{k} {len(v)}" for k, v in results.items()))
    if missing:
        for m in missing:
            print(f"error: no pass-2 outcome for {m}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map", help="build the coverage map of every test target")
    m.add_argument("--out", required=True)
    m.add_argument("--features", default="")
    m.add_argument("--target-timeout", type=int, default=3600)
    r = sub.add_parser("run", help="run the first pass's missed mutants with their covering targets")
    r.add_argument("--map", required=True, help="the coverage map, or - for none (every target)")
    r.add_argument("--pass1", required=True, help="the first pass's mutants.out (or its parent)")
    r.add_argument("--out", required=True)
    r.add_argument("--features", default="")
    r.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args(argv)
    return cmd_map(args) if args.cmd == "map" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
