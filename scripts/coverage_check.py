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
  * `limitation` is `<file>::<symbol> '<verbatim quote>'` and the quote is in
    that item (its doc comments, attributes and body), or `<file> '<quote>'` for
    the module documentation; `partial` requires it
  * `partial` <-> `LIMITATION(<id>)` comments in src/, both ways: a partial item
    has at least one comment (unless listed in MARKER_EXEMPT with a reason), and
    every comment names an existing partial item
  * test -> table: a `// covers: COV-X-NNN[, ...]` line just above a test fn says
    the test checks that capability. A test that runs and covers an item whose
    status is still `missing` / `implemented-no-oracle` fails (update the table
    when the capability lands); a covered id must exist; `covers` must sit on a
    #[test] fn. A test ignored with `src gap: COV-X-NNN` is the oracle written
    ahead of the implementation, and fails once the item is `implemented+oracle`
    (remove the ignore)
  * references are `file::symbol` (or a bare file), never `file:line`: lines
    inserted above a line reference move it, and with many tables every source
    edit would break some of them. `src/x.rs::name` may name any item defined in
    that file (fn, struct, enum, trait, const, ...)
  * docs/coverage/status.md, the per-table and per-axis counts, matches the
    tables (`--write-status` regenerates it)

Tables are added per domain (docs/coverage/<domain>.toml, ids COV-<DOMAIN>-NNN);
nothing here names a domain.

Every check must compare something: no table, a table without items, or no
source file to scan for comments is a failure, not a pass.

  python3 scripts/coverage_check.py                 # exit 1 on any finding
  python3 scripts/coverage_check.py --write-status  # regenerate docs/coverage/status.md
  python3 scripts/coverage_check.py --root DIR      # another tree (used by the tests)
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

# `src/a/b.rs::symbol`, `tests/x.rs::test_fn` or a bare file. A line number
# (`src/a/b.rs:123`) is matched only to be refused: lines inserted above it move
# the reference, and with many tables every source edit would break some of them
# (scripts/coverage_refs_to_symbols.py converts line references).
REF_RE = re.compile(r"(?<![\w/.-])((?:src|tests)/[\w/.-]+?\.rs)(?:::([A-Za-z_]\w*)|:(\d+))?")
LIMITATION_RE = re.compile(r"^((?:src|tests)/[\w/.-]+?\.rs)(?:::([A-Za-z_]\w*))? '(.+)'$", re.S)
LINE_LIMITATION_RE = re.compile(r"^((?:src|tests)/[\w/.-]+?\.rs):(\d+) '")
LINE_REF_HINT = "use `file::symbol` (scripts/coverage_refs_to_symbols.py converts line references)"
MARKER_RE = re.compile(r"LIMITATION\(([^)]*)\)")
COVERS_RE = re.compile(r"^\s*//[/!]?\s*covers:\s*(COV-.*)$")
FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)")
SRC_GAP_RE = re.compile(r"^src gap:\s*(COV-[A-Z]+-\d{3})\b")
STATUS_DOC = "docs/coverage/status.md"
# statuses a test that runs may not leave an item in
NOT_YET = ("missing", "implemented-no-oracle")


DEF_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:(?:unsafe|async|const|extern\s+\"[^\"]*\")\s+)*"
                    r"(?:fn|struct|enum|trait|const|static|type|mod|union|macro_rules!)\s+([A-Za-z_]\w*)")


def defines(lines: list[str], name: str) -> bool:
    return any((m := DEF_RE.match(ln)) and m.group(1) == name for ln in lines)


def item_spans(lines: list[str], name: str) -> list[tuple[int, int]]:
    """0-based inclusive line spans of every item `name` defines: its doc comments
    and attributes above, through the end of its body (brace-matched) or its `;`."""
    spans = []
    for d, ln in enumerate(lines):
        m = DEF_RE.match(ln)
        if not m or m.group(1) != name:
            continue
        start = d
        while start > 0 and lines[start - 1].strip().startswith(("///", "//", "#[", "#!")):
            start -= 1
        depth, end, opened = 0, d, False
        for k in range(d, min(len(lines), d + 5000)):
            code = lines[k].split("//")[0]
            depth += code.count("{") - code.count("}")
            opened = opened or "{" in code
            end = k
            if (opened and depth <= 0) or (not opened and code.rstrip().endswith(";")):
                break
        spans.append((start, end))
    return spans

# Partial items whose LIMITATION comment cannot be placed in the source yet,
# with the reason. Each entry must name a partial item that has no comment;
# a stale entry fails.
MARKER_EXEMPT: dict[str, str] = {
}


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
    def __init__(self, root: Path, exempt: dict[str, str] | None = None, write_status: bool = False):
        self.root = root
        self.exempt = MARKER_EXEMPT if exempt is None else exempt
        self.errors: list[str] = []
        self.write_status = write_status
        self.n_covers = self.n_src_gaps = 0
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
            if line is not None:
                self.err(f"{where}: evidence {path}:{line} is a line reference; {LINE_REF_HINT}")
                continue
            if fn is not None:
                t = self._tests_in(path).get(fn)
                if t is None and defines(lines, fn):
                    continue  # an item of that file (a source symbol or a test helper), not a test
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
            if LINE_LIMITATION_RE.match(lim):
                self.err(f"{where}: limitation uses a line reference; {LINE_REF_HINT}")
            elif not m:
                self.err(f"{where}: limitation is not `<file>::<symbol> '<quote>'` (or `<file> '<quote>'` "
                         "for the module documentation)")
            else:
                path, sym, quote = m.group(1), m.group(2), " ".join(m.group(3).split())
                lines = self._file_lines(path)
                if lines is None:
                    self.err(f"{where}: limitation names {path}, which does not exist")
                elif sym is None:
                    if quote not in _flatten("\n".join(lines)):
                        self.err(f"{where}: limitation quote is not in {path}")
                else:
                    spans = item_spans(lines, sym)
                    if not spans:
                        self.err(f"{where}: limitation names {path}::{sym}, which that file does not define")
                    elif not any(quote in _flatten("\n".join(lines[a:b + 1])) for a, b in spans):
                        self.err(f"{where}: limitation quote is not in {path}::{sym} "
                                 "(its doc comments, attributes and body)")
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

    # ---- test -> table
    def _rust_files(self) -> list[Path]:
        return sorted(list((self.root / "src").rglob("*.rs")) + list((self.root / "tests").rglob("*.rs")))

    def scan_covers(self) -> list[tuple[str, str, int, str]]:
        """(file, test fn, line of the comment, id) for every `// covers:` line."""
        out: list[tuple[str, str, int, str]] = []
        for f in self._rust_files():
            text = f.read_text(encoding="utf-8")
            if "covers:" not in text:
                continue
            rel = f.relative_to(self.root).as_posix()
            lines = text.splitlines()
            for n, ln in enumerate(lines):
                m = COVERS_RE.match(ln)
                if not m:
                    continue
                ids = ID_RE.findall(m.group(1))
                if not ids or ID_RE.sub("", m.group(1)).strip(" ,"):
                    self.err(f"{rel}:{n + 1}: `covers:` takes ids of the form {ID_RE.pattern}, comma-separated")
                    continue
                fn = None
                for nxt in lines[n + 1:n + 40]:
                    fm = FN_RE.match(nxt)
                    if fm:
                        fn = fm.group(1)
                        break
                    if nxt.strip() and not nxt.strip().startswith(("#", "//", ")", "]", '"')):
                        break
                if fn is None:
                    self.err(f"{rel}:{n + 1}: `covers:` is not followed by a fn")
                    continue
                out += [(rel, fn, n + 1, i) for i in ids]
        return out

    def check_covers(self, status_of: dict) -> int:
        links = self.scan_covers()
        for rel, fn, line, iid in links:
            where = f"{rel}:{line}"
            if iid not in status_of:
                self.err(f"{where}: covers {iid}, which is no item in docs/coverage/")
                continue
            t = self._tests_in(rel).get(fn)
            if t is None:
                self.err(f"{where}: covers {iid} on `{fn}`, which is not a #[test] fn")
                continue
            if not t["is_ignored"] and status_of[iid] in NOT_YET:
                self.err(f"{where}: {rel}::{fn} runs and covers {iid}, whose status is still "
                         f"{status_of[iid]}: update the table (implemented+oracle, or partial with its limitation)")
        return len(links)

    def check_src_gaps(self, status_of: dict) -> int:
        n = 0
        for f in self._rust_files():
            if "src gap: COV-" not in f.read_text(encoding="utf-8"):
                continue
            rel = f.relative_to(self.root).as_posix()
            for t in self._tests_in(rel).values():
                m = SRC_GAP_RE.match(t["ignore_reason"]) if t["is_ignored"] else None
                if not m:
                    continue
                n += 1
                iid = m.group(1)
                if iid not in status_of:
                    self.err(f"{rel}::{t['name']}: ignored as `src gap: {iid}`, which is no item in docs/coverage/")
                elif status_of[iid] == "implemented+oracle":
                    self.err(f"{rel}::{t['name']}: still ignored as `src gap: {iid}`, but {iid} is "
                             "implemented+oracle: remove the ignore (or correct the table)")
        return n

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
        self.n_covers = self.check_covers(status_of)
        self.n_src_gaps = self.check_src_gaps(status_of)
        if items:
            want = status_text(items)
            p = self.root / STATUS_DOC
            have = p.read_text(encoding="utf-8").replace("\r\n", "\n") if p.is_file() else None
            if self.write_status:
                p.write_text(want, encoding="utf-8", newline="\n")
            elif have != want:
                self.err(f"{STATUS_DOC} is {'missing' if have is None else 'stale'}: "
                         "run `python3 scripts/coverage_check.py --write-status`")
        return items, markers


def status_text(items: list[tuple[str, dict]]) -> str:
    """docs/coverage/status.md: counts per table, then per axis inside each table."""
    head = "| " + " | ".join(STATUSES) + " |"
    rule = "|" + "---:|" * len(STATUSES)
    out = ["# Coverage status", "",
           "Generated from docs/coverage/*.toml by `python3 scripts/coverage_check.py --write-status`;",
           "do not edit by hand. Each table lists the capabilities a field expects, with what the",
           "crate has today.", "",
           "| table | items " + head, "|---|---:" + rule]
    tables: dict[str, list[dict]] = {}
    for rel, it in items:
        tables.setdefault(rel, []).append(it)
    total = Counter()
    for rel, its in sorted(tables.items()):
        cnt = Counter(it.get("status") for it in its)
        total += cnt
        out.append(f"| `{rel}` | {len(its)} | " + " | ".join(str(cnt[s]) for s in STATUSES) + " |")
    if len(tables) > 1:
        out.append(f"| **total** | {sum(len(v) for v in tables.values())} | "
                   + " | ".join(str(total[s]) for s in STATUSES) + " |")
    for rel, its in sorted(tables.items()):
        out += ["", f"## `{rel}`", "", "| axis | items " + head, "|---|---:" + rule]
        axes: dict[str, Counter] = {}
        for it in its:
            axes.setdefault(str(it.get("axis", "")), Counter())[it.get("status")] += 1
        for ax, cnt in sorted(axes.items()):
            out.append(f"| {ax} | {sum(cnt.values())} | " + " | ".join(str(cnt[s]) for s in STATUSES) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None, exempt: dict[str, str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument("--write-status", action="store_true", help=f"regenerate {STATUS_DOC}")
    args = ap.parse_args(argv)
    root = Path(args.root)
    # MARKER_EXEMPT names items of this repository; another tree is checked without it
    if exempt is None and root.resolve() != HERE.parent.resolve():
        exempt = {}
    c = Checker(root, exempt=exempt, write_status=args.write_status)
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
    print(f"covers links: {c.n_covers}; src gap oracles: {c.n_src_gaps}")
    print("coverage: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
