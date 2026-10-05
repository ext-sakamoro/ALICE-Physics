#!/usr/bin/env python3
"""Check the hand-written coverage tables in docs/coverage/*.toml against the code.

Each table lists the capabilities a field expects (from textbooks, reference
implementations and standard benchmarks), one `[[item]]` per capability, with a
status. The tables are written by hand; this script keeps them honest:

  * every item has the required keys, a well-formed unique id and one of the
    five statuses
  * every `src/...rs:line` and `tests/...rs::test_fn` in `evidence` exists
  * `implemented+oracle`: the evidence names at least one test that runs (not
    `#[ignore]`d); a cited ignored test is allowed only when its reason starts
    with `known defect` (a recorded defect of an implemented capability)
  * `limitation` is `<file>:<line> '<verbatim quote>'` and the quote is in that
    file; `partial` requires it
  * `partial` <-> `LIMITATION(<id>)` comments in src/, both ways: a partial item
    has at least one comment (unless listed in MARKER_EXEMPT with a reason), and
    every comment names an existing partial item

Every check must compare something: no table, a table without items, or no
source file to scan for comments is a failure, not a pass.

  python3 scripts/coverage_check.py            # exit 1 on any finding
  python3 scripts/coverage_check.py --root DIR # another tree (used by the tests)
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The one definition of an item id: COV-<DOMAIN>-NNN.
ID_RE = re.compile(r"COV-[A-Z]+-\d{3}")
STATUSES = ("implemented+oracle", "implemented-no-oracle", "partial", "missing", "out-of-scope")
REQUIRED_KEYS = ("id", "axis", "item", "source", "status", "evidence", "oracle_candidate", "limitation")

# `src/a/b.rs:123` or `tests/x.rs::test_fn` (also `src/x.rs::unit_test_fn`).
REF_RE = re.compile(r"(?<![\w/.-])((?:src|tests)/[\w/.-]+?\.rs)(?:::([A-Za-z_]\w*)|:(\d+))?")
LIMITATION_RE = re.compile(r"^((?:src|tests)/[\w/.-]+?\.rs):(\d+) '(.+)'$", re.S)
MARKER_RE = re.compile(r"LIMITATION\(([^)]*)\)")

# Partial items whose LIMITATION comment cannot be placed in the source yet,
# with the reason. Each entry must name a partial item that has no comment;
# a stale entry fails.
MARKER_EXEMPT: dict[str, str] = {}


def _load_test_parser():
    spec = importlib.util.spec_from_file_location("gen_oracle_status", HERE / "gen-oracle-status.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_test_metadata


_COMMENT_PREFIX = re.compile(r"^\s*(?://[!/]?)?\s?")


def _flatten(text: str) -> str:
    """Source text with comment markers removed and whitespace collapsed, so a
    quote that spans several `///` lines still matches."""
    lines = [_COMMENT_PREFIX.sub("", ln) for ln in text.splitlines()]
    return " ".join(" ".join(lines).split())


class Checker:
    def __init__(self, root: Path, exempt: dict[str, str] | None = None):
        self.root = root
        self.exempt = MARKER_EXEMPT if exempt is None else exempt
        self.errors: list[str] = []
        self._tests: dict[str, dict[str, dict]] = {}
        self._lines: dict[str, list[str]] = {}
        self._parse_tests = _load_test_parser()

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def _file_lines(self, rel: str) -> list[str] | None:
        if rel not in self._lines:
            p = self.root / rel
            self._lines[rel] = p.read_text(encoding="utf-8").splitlines() if p.is_file() else None
        return self._lines[rel]

    def _tests_in(self, rel: str) -> dict[str, dict]:
        if rel not in self._tests:
            self._tests[rel] = {t["name"]: t for t in self._parse_tests(self.root / rel)}
        return self._tests[rel]

    # ---- tables
    def load_items(self) -> list[tuple[str, dict]]:
        tables = sorted((self.root / "docs" / "coverage").glob("*.toml"))
        if not tables:
            self.err("no table: docs/coverage/*.toml matched nothing")
            return []
        items: list[tuple[str, dict]] = []
        for t in tables:
            rel = t.relative_to(self.root).as_posix()
            try:
                data = tomllib.loads(t.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as e:
                self.err(f"{rel}: not valid TOML: {e}")
                continue
            got = data.get("item", [])
            if not isinstance(got, list) or not got:
                self.err(f"{rel}: no [[item]]")
                continue
            items += [(rel, it) for it in got]
        if not items:
            self.err("no item in any table")
        return items

    def check_item(self, rel: str, it: dict) -> None:
        iid = it.get("id", "")
        where = f"{rel}: {iid or '<no id>'}"
        for k in REQUIRED_KEYS:
            if k not in it:
                self.err(f"{where}: missing key `{k}`")
            elif not isinstance(it[k], str):
                self.err(f"{where}: `{k}` is not a string")
        if not isinstance(iid, str) or not ID_RE.fullmatch(iid):
            self.err(f"{where}: id does not match {ID_RE.pattern}")
        status = it.get("status", "")
        if status not in STATUSES:
            self.err(f"{where}: status {status!r} is not one of {', '.join(STATUSES)}")

        cited_tests: list[tuple[str, str, dict | None]] = []
        for m in REF_RE.finditer(it.get("evidence", "") if isinstance(it.get("evidence"), str) else ""):
            path, fn, line = m.group(1), m.group(2), m.group(3)
            lines = self._file_lines(path)
            if lines is None:
                self.err(f"{where}: evidence names {path}, which does not exist")
                continue
            if line is not None and not 1 <= int(line) <= len(lines):
                self.err(f"{where}: evidence {path}:{line} is past the end ({len(lines)} lines)")
            if fn is not None:
                t = self._tests_in(path).get(fn)
                if t is None:
                    self.err(f"{where}: evidence names test {path}::{fn}, which is not a #[test] fn there")
                cited_tests.append((path, fn, t))

        if status == "implemented+oracle":
            running = [c for c in cited_tests if c[2] is not None and not c[2]["is_ignored"]]
            for path, fn, t in cited_tests:
                if t is not None and t["is_ignored"] and not t["ignore_reason"].lower().startswith("known defect"):
                    self.err(f"{where}: oracle {path}::{fn} is #[ignore]d ({t['ignore_reason'][:40]}...), "
                             "so it does not run")
            if not running:
                self.err(f"{where}: implemented+oracle but the evidence names no test that runs")

        lim = it.get("limitation", "")
        if isinstance(lim, str) and lim:
            m = LIMITATION_RE.match(lim)
            if not m:
                self.err(f"{where}: limitation is not `<file>:<line> '<quote>'`")
            else:
                lines = self._file_lines(m.group(1))
                if lines is None:
                    self.err(f"{where}: limitation names {m.group(1)}, which does not exist")
                elif " ".join(m.group(3).split()) not in _flatten("\n".join(lines)):
                    self.err(f"{where}: limitation quote is not in {m.group(1)}")
        elif status == "partial":
            self.err(f"{where}: partial needs a verbatim limitation from the source")

    # ---- LIMITATION comments
    def scan_markers(self) -> dict[str, list[str]]:
        files = sorted((self.root / "src").rglob("*.rs"))
        if not files:
            self.err("no source file under src/ to scan for LIMITATION comments")
        found: dict[str, list[str]] = {}
        for f in files:
            for n, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                for m in MARKER_RE.finditer(ln):
                    found.setdefault(m.group(1), []).append(f"{f.relative_to(self.root).as_posix()}:{n}")
        return found

    def run(self) -> tuple[list[tuple[str, dict]], dict[str, list[str]]]:
        items = self.load_items()
        seen: dict[str, str] = {}
        for rel, it in items:
            iid = it.get("id")
            if isinstance(iid, str) and iid in seen:
                self.err(f"{rel}: id {iid} is also used in {seen[iid]}")
            elif isinstance(iid, str):
                seen[iid] = rel
            self.check_item(rel, it)
        status_of = {it.get("id"): it.get("status") for _, it in items}
        markers = self.scan_markers()
        for mid, locs in sorted(markers.items()):
            if not ID_RE.fullmatch(mid):
                self.err(f"{locs[0]}: LIMITATION({mid}) does not name an id of the form {ID_RE.pattern}")
            elif mid not in status_of:
                self.err(f"{locs[0]}: LIMITATION({mid}) names no item in docs/coverage/")
            elif status_of[mid] != "partial":
                self.err(f"{locs[0]}: LIMITATION({mid}) names an item whose status is {status_of[mid]}, not partial")
        for iid, st in status_of.items():
            if st != "partial":
                continue
            if iid in self.exempt:
                if iid in markers:
                    self.err(f"{iid}: exempt from the LIMITATION comment but has one ({markers[iid][0]}); "
                             "drop the exemption")
            elif iid not in markers:
                self.err(f"{iid}: partial item without a LIMITATION({iid}) comment in src/")
        for iid in self.exempt:
            if status_of.get(iid) != "partial":
                self.err(f"{iid}: exemption names no partial item")
        return items, markers


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(HERE.parent))
    args = ap.parse_args(argv)
    c = Checker(Path(args.root))
    items, markers = c.run()
    if c.errors:
        for e in c.errors:
            print(f"coverage: {e}", file=sys.stderr)
        print(f"coverage: {len(c.errors)} finding(s)", file=sys.stderr)
        return 1
    by_table: dict[str, Counter] = {}
    for rel, it in items:
        by_table.setdefault(rel, Counter())[it["status"]] += 1
    for rel, cnt in by_table.items():
        parts = " / ".join(f"{s} {cnt[s]}" for s in STATUSES)
        print(f"{rel}: {sum(cnt.values())} items: {parts}")
    n_markers = sum(len(v) for v in markers.values())
    print(f"LIMITATION comments: {n_markers} in src/ for {len(markers)} items; "
          f"exempt: {len(c.exempt)}" + "".join(f"\n  {k}: {v}" for k, v in sorted(c.exempt.items())))
    print("coverage: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
