#!/usr/bin/env python3
"""
Generate ALICE-Physics oracle status report.

Scans tests/ for oracle tests and classifies them by implementation status:
- 🟢 Implemented: green (no #[ignore], implementation exists)
- 🟡 Partial: red or pending (implementation exists but incomplete)
- 🔴 Pending: not implemented (#[ignore] or placeholder)
"""

import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"

DOCS_DIR.mkdir(exist_ok=True)

DEFECT_ID_RE = re.compile(r'AUD-[A-Z]-S\d+(?:W\d+)?-\d+')
# `// PIN: AUD-…` before a test: it pins today's (defective) behaviour on purpose,
# so fixing that defect turns it red. Several ids may follow, comma-separated.
PIN_RE = re.compile(r'^\s*//[/!]?\s*PIN:\s*(.*)$')
# `root: external <crate> <version>` in a known-defect reason: the cause is in a
# dependency, at that version; a different version in Cargo.lock means re-check.
EXTERNAL_RE = re.compile(r'root:\s*external\s+([A-Za-z0-9_-]+)\s+([0-9][0-9A-Za-z.+-]*)')


def _strip_line_comment(line):
    """`//` 以降を落とす (コメントに書かれた #[test] / fn を数えない)."""
    return line.split('//', 1)[0] if line.lstrip().startswith('//') else line


def _read_attribute(lines, i):
    """lines[i] から始まる属性 `#[...]` を、閉じる `]` まで連結して返す (行継続 `\\` も畳む).

    返り値: (属性の本文, 次に読む行の index).  実物の `#[ignore = "…… \\\n ……"]` は複数行に
    またがり、1 行ずつ読むと理由文が落ちて「pending」と誤読される.
    """
    buf = lines[i].strip()
    j = i + 1
    while buf.count('[') > buf.count(']') or buf.count('"') % 2 == 1:
        if j >= len(lines):
            break
        nxt = lines[j].strip()
        buf = (buf[:-1] if buf.endswith('\\') else buf + ' ') + nxt
        j += 1
    return buf, j


def _ignore_reason(attr):
    """`#[ignore]` / `#[ignore = "…"]` の理由 (空の ignore は 'pending')."""
    m = re.match(r'#\[ignore\s*=\s*"(.*)"\s*\]\s*$', attr, flags=re.S)
    return re.sub(r'\s+', ' ', m.group(1)).strip() if m else 'pending'


def extract_test_metadata(test_file):
    """Extract test functions and their ignore status from a test file.

    属性は `#[test]` の前後どちらに `#[ignore]` があっても拾い、`#[test]` から `fn` までの
    距離に上限を設けない (長い doc comment / 複数行の `#[should_panic]` を落とさない).
    """
    tests = []
    lines = Path(test_file).read_text(encoding='utf-8').split('\n')

    pending_attrs = []  # 直前までに読んだ属性 (空行・コメントでは切れない)
    pending_pins = []   # 直前の `// PIN:` 行の中身
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith('//') or not stripped:
            pm = PIN_RE.match(line)
            if pm:
                pending_pins.append(pm.group(1).strip())
            i += 1
            continue
        if stripped.startswith('#['):
            attr, i = _read_attribute(lines, i)
            pending_attrs.append(attr)
            continue
        fn_match = re.match(r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(', stripped)
        if fn_match:
            is_test = any(re.match(r'#\[(?:\w+::)?test\]$', a) for a in pending_attrs)
            ignores = [a for a in pending_attrs if a.startswith('#[ignore')]
            if is_test:
                tests.append({
                    'name': fn_match.group(1),
                    'file': Path(test_file).name,
                    'is_ignored': bool(ignores),
                    'ignore_reason': _ignore_reason(ignores[0]) if ignores else '',
                    'pins': list(pending_pins),
                })
        pending_attrs = []
        pending_pins = []
        i += 1

    return tests


def classify_ignored(reason):
    """`#[ignore]` の理由から 3 分類する.

    - red:     意図して red のまま残している oracle (「the red is correct」/「src gap」/「known defect: AUD-…」= 監査台帳の欠陥)
               実装側が追いつけば #[ignore] を外す  実装を足す対象であって、期待値を緩めない
    - gated:   実行が長い / 診断表を出すだけ / 手動 (runtime / diagnostic / manual / run with --release)
    - pending: 理由の無い bare な #[ignore]
    """
    r = reason.strip().lower()
    if r == 'pending' or not r:
        return 'pending'
    if r.startswith(('the red is correct', 'src gap', 'known defect')):
        return 'red'
    return 'gated'


def cargo_lock_versions(root=PROJECT_ROOT):
    """crate name -> set of versions resolved in Cargo.lock."""
    out = defaultdict(set)
    p = Path(root) / 'Cargo.lock'
    if not p.exists():
        return out
    name = None
    for line in p.read_text(encoding='utf-8').split('\n'):
        m = re.match(r'name = "([^"]+)"', line)
        if m:
            name = m.group(1)
            continue
        m = re.match(r'version = "([^"]+)"', line)
        if m and name:
            out[name].add(m.group(1))
            name = None
    return out


def _cargo_metadata_versions(root):
    """crate name -> versions from `cargo metadata --all-features`, or None if cargo fails.

    `--all-features`: the default resolve leaves out optional dependencies, and a
    root cause is often in one (alice-db only enters through `replay`).
    """
    import json
    import subprocess
    cmd = ['cargo', 'metadata', '--format-version', '1', '--all-features',
           '--manifest-path', str(Path(root) / 'Cargo.toml')]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', check=True).stdout
        packages = json.loads(out)['packages']
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        return None
    versions = defaultdict(set)
    for pkg in packages:
        versions[pkg['name']].add(pkg['version'])
    return versions


def resolved_versions(root=PROJECT_ROOT):
    """Dependency versions as Cargo resolves them, or None when they cannot be determined.

    Cargo.lock is read when it exists. This repository ignores it (a library), so a CI
    checkout has none: then `cargo metadata` resolves the graph, which is the version CI
    tests against.
    """
    if (Path(root) / 'Cargo.lock').exists():
        return cargo_lock_versions(root)
    return _cargo_metadata_versions(root)


def audit_links(tests, lock_versions):
    """Pins, external root causes and id-less known defects, with their problems.

    Returns (pins, externals, no_id, problems). `problems` are what `--check` fails on:
    a PIN with no id, a PIN whose id is not an open known defect (fixed, or a typo:
    the pinned behaviour is no longer a defect, so the pin is stale), and an external
    root cause naming a crate the dependency resolution does not contain (or any external
    root cause at all when the versions could not be resolved: `lock_versions` is None).
    """
    open_defects = {}
    for t in tests:
        r = t['ignore_reason']
        if r.lower().startswith('known defect'):
            m = DEFECT_ID_RE.search(r)
            if m:
                open_defects.setdefault(m.group(0), t)
    pins, externals, no_id, problems = [], [], [], []
    for t in tests:
        for raw in t.get('pins', []):
            ids = DEFECT_ID_RE.findall(raw)
            if not ids:
                problems.append(f"PIN without a defect id: {t['file']}::{t['test_name']} ({raw!r})")
            for i in ids:
                d = open_defects.get(i)
                pins.append((i, t, d))
                if d is None:
                    problems.append(f"PIN {i} on {t['file']}::{t['test_name']} names no open known defect (stale pin)")
        r = t['ignore_reason']
        if r.lower().startswith('known defect'):
            if not DEFECT_ID_RE.search(r):
                no_id.append(t)
            m = EXTERNAL_RE.search(r)
            if m:
                crate, ver = m.group(1), m.group(2)
                locked = sorted((lock_versions or {}).get(crate, ()))
                externals.append((DEFECT_ID_RE.search(r).group(0) if DEFECT_ID_RE.search(r) else '', t, crate, ver, locked))
                if lock_versions is None:
                    problems.append(f"cannot resolve dependency versions (no Cargo.lock and `cargo metadata` failed) "
                                    f"to check {crate} ({t['file']}::{t['test_name']})")
                elif not locked:
                    problems.append(f"external root cause names {crate}, which the dependency resolution does not "
                                    f"contain ({t['file']}::{t['test_name']})")
    return pins, externals, no_id, problems


def run_tests_and_categorize():
    """Run tests and categorize by status."""
    all_tests = defaultdict(list)

    for test_file in sorted(TESTS_DIR.glob('*.rs')):
        tests = extract_test_metadata(test_file)
        for test in tests:
            all_tests[test_file.name].append(test)

    # Categorize
    implemented = []
    partial = []
    pending = []

    for test_file_name, tests in sorted(all_tests.items()):
        for test in tests:
            test_info = {
                'test_name': test['name'],
                'file': test['file'],
                'ignore_reason': test['ignore_reason'],
                'pins': test.get('pins', []),
            }

            if test['is_ignored']:
                test_info['reason'] = test['ignore_reason'] or 'pending implementation'
                pending.append(test_info)
            else:
                # Not ignored = assume implemented
                implemented.append(test_info)

    return {
        'implemented': implemented,
        'partial': partial,
        'pending': pending,
    }


def _line(test):
    reason = test['ignore_reason']
    if reason and reason != 'pending':
        short = reason[:110] + ('…' if len(reason) > 110 else '')
        return f"- `{test['test_name']}` ({test['file']}) — {short}\n"
    return f"- `{test['test_name']}` ({test['file']})\n"


def generate_markdown_report(categorized, lock_versions=None):
    """Generate markdown report (a pure function: no timestamp, so it changes only when the tests do)."""
    all_tests = categorized['implemented'] + categorized['partial'] + categorized['pending']
    pins, externals, no_id, _problems = audit_links(all_tests, lock_versions)
    ignored = categorized['pending']
    by_class = {'red': [], 'gated': [], 'pending': []}
    for t in ignored:
        by_class[classify_ignored(t['ignore_reason'])].append(t)
    total = sum(len(v) for v in categorized.values())

    report = f"""# ALICE-Physics Oracle Status

_Generated from `tests/*.rs` (no timestamp: the file changes only when its content does)._

## Summary

| Category | Count |
|----------|-------|
| 🟢 Not ignored (run by CI) | {len(categorized['implemented'])} |
| 🔴 Red by design | {len(by_class['red'])} |
| ⏱ Gated (runtime / diagnostic / manual) | {len(by_class['gated'])} |
| ⚪ Pending (bare `#[ignore]`) | {len(by_class['pending'])} |
| **Total** | **{total}** |

`Not ignored` means only that the test carries no `#[ignore]`: this report does not run it.
CI's `cargo test` is what says whether it passes.

"""

    if by_class['red']:
        report += f"""## 🔴 Red by design ({len(by_class['red'])})

Oracles kept red on purpose: the implementation is not there yet, and a companion test pins
today's behaviour so CI coverage is not lost. The fix is in `src/`; the expected value is never loosened.

"""
        for t in sorted(by_class['red'], key=lambda x: x['test_name']):
            report += _line(t)
        report += "\n"

    report += f"""## 📌 Pins: tests that turn red when a known defect is fixed ({len(pins)})

Tests marked `// PIN: <defect id>` assert today's defective behaviour on purpose. Fixing the
defect makes them red; update them in the same change, after checking that the new behaviour
is the intended one.

"""
    if pins:
        report += "| Defect | Pin test | Defect test |\n|--------|----------|-------------|\n"
        for i, t, d in sorted(pins, key=lambda x: (x[0], x[1]['file'], x[1]['test_name'])):
            dt = f"`{d['test_name']}` ({d['file']})" if d else "⚠️ no open known defect with this id"
            report += f"| {i} | `{t['test_name']}` ({t['file']}) | {dt} |\n"
    else:
        report += "- (none)\n"
    report += "\n"

    report += f"""## 🌐 Root cause outside this repository ({len(externals)})

Known defects whose reason says `root: external <crate> <version>`: the fix belongs in that
dependency. When Cargo resolves a different version (Cargo.lock, or `cargo metadata --all-features`
when the lock is not committed), re-check whether the defect remains.

"""
    if externals:
        report += "| Defect | Test | Crate | Reason says | Resolved | Status |\n|--------|------|-------|-------------|----------|--------|\n"
        for i, t, crate, ver, locked in sorted(externals, key=lambda x: (x[0], x[1]['test_name'])):
            lk = ", ".join(locked) or "—"
            st = "✅ same" if ver in locked else ("⚠️ re-check" if locked else "⚠️ not resolved")
            report += f"| {i} | `{t['test_name']}` ({t['file']}) | `{crate}` | {ver} | {lk} | {st} |\n"
    else:
        report += "- (none)\n"
    report += "\n"

    if no_id:
        report += f"""## ⚠️ Known defects without an id ({len(no_id)})

A known-defect reason should start with `AUD-…` so pins, external causes and
`scripts/audit_refs.py` can refer to it.

"""
        for t in sorted(no_id, key=lambda x: (x['file'], x['test_name'])):
            report += _line(t)
        report += "\n"

    if by_class['gated']:
        report += f"""## ⏱ Gated ({len(by_class['gated'])})

Correct tests that are too slow for every push, or that print a measurement table.
Run them with `python3 scripts/run_ignored.py` or `cargo test --release -- --ignored`.

"""
        for t in sorted(by_class['gated'], key=lambda x: x['test_name']):
            report += _line(t)
        report += "\n"

    if by_class['pending']:
        report += f"""## ⚪ Pending ({len(by_class['pending'])})

`#[ignore]` with no reason: not yet implemented, or forgotten.

"""
        for t in sorted(by_class['pending'], key=lambda x: x['test_name']):
            report += _line(t)
        report += "\n"

    if categorized['partial']:
        report += f"""## 🟡 Partial ({len(categorized['partial'])})

"""
        for t in sorted(categorized['partial'], key=lambda x: x['test_name']):
            report += f"- `{t['test_name']}` ({t['file']}) — {t.get('reason', 'partial implementation')}\n"
        report += "\n"

    report += f"""## 🟢 Not ignored ({len(categorized['implemented'])})

Per-file counts (the test names are in `tests/`):

| File | Tests |
|------|-------|
"""
    per_file = defaultdict(int)
    for t in categorized['implemented']:
        per_file[t['file']] += 1
    for name, n in sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0])):
        report += f"| `{name}` | {n} |\n"

    report += """
---

## How to Contribute

When an oracle goes green:
1. Remove `#[ignore]` from the test (and the companion test that pins the old behaviour, if the reason says so)
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify

For details: [CLAUDE.md](../CLAUDE.md)
"""

    return report


def main(argv=None):
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="ALICE-Physics oracle status report")
    ap.add_argument('--check', action='store_true',
                    help='do not write; fail on a stale or id-less PIN, or an external cause Cargo does not resolve')
    args = ap.parse_args(argv)
    print("Scanning ALICE-Physics oracle tests...", file=sys.stderr)

    categorized = run_tests_and_categorize()
    lock = resolved_versions()
    if args.check:
        all_tests = categorized['implemented'] + categorized['partial'] + categorized['pending']
        pins, externals, no_id, problems = audit_links(all_tests, lock)
        n = sum(len(v) for v in categorized.values())
        print(f"compared: tests {n}, pins {len(pins)}, external causes {len(externals)}, known defects without id {len(no_id)}")
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        if n == 0:
            print("error: no tests found (compared nothing)", file=sys.stderr)
            return 1
        return 1 if problems else 0
    report = generate_markdown_report(categorized, lock)

    output_file = DOCS_DIR / "oracle-status.md"
    output_file.write_text(report, encoding='utf-8')

    print(f"✅ Generated {output_file}", file=sys.stderr)
    print(f"   Implemented: {len(categorized['implemented'])}", file=sys.stderr)
    print(f"   Partial: {len(categorized['partial'])}", file=sys.stderr)
    print(f"   Pending: {len(categorized['pending'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
