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
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"

DOCS_DIR.mkdir(exist_ok=True)


def extract_test_metadata(test_file):
    """Extract test functions and their ignore status from a test file."""
    tests = []

    with open(test_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Look for #[test] or #[tokio::test]
        if '#[test]' in line or '#[tokio::test]' in line:
            # Check next lines for #[ignore] and fn
            is_ignored = False
            ignore_reason = ''
            test_name = None

            j = i + 1
            while j < len(lines) and j < i + 10:  # Look ahead up to 10 lines
                next_line = lines[j]

                # Check for #[ignore]
                if '#[ignore' in next_line:
                    is_ignored = True
                    match = re.search(r'#\[ignore\s*=\s*"([^"]*)"', next_line)
                    if match:
                        ignore_reason = match.group(1)
                    else:
                        ignore_reason = 'pending'

                # Check for fn definition
                fn_match = re.search(r'(?:async\s+)?fn\s+(\w+)\s*\(', next_line)
                if fn_match:
                    test_name = fn_match.group(1)
                    break

                j += 1

            if test_name:
                tests.append({
                    'name': test_name,
                    'file': test_file.name,
                    'is_ignored': is_ignored,
                    'ignore_reason': ignore_reason,
                })

        i += 1

    return tests


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


def generate_markdown_report(categorized):
    """Generate markdown report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    report = f"""# ALICE-Physics Oracle Status

**Last updated:** {timestamp}

## Summary

| Category | Count |
|----------|-------|
| 🟢 Implemented | {len(categorized['implemented'])} |
| 🟡 Partial | {len(categorized['partial'])} |
| 🔴 Pending | {len(categorized['pending'])} |
| **Total** | **{sum(len(v) for v in categorized.values())}** |

"""

    if categorized['pending']:
        report += f"""## 🔴 Pending ({len(categorized['pending'])})

Oracle tests not yet implemented (marked with `#[ignore]`).

"""
        for test in sorted(categorized['pending'], key=lambda x: x['test_name']):
            if test['ignore_reason']:
                # Truncate long reasons
                reason = test['ignore_reason'][:80]
                if len(test['ignore_reason']) > 80:
                    reason += "…"
                report += f"- `{test['test_name']}` ({test['file']}) — {reason}\n"
            else:
                report += f"- `{test['test_name']}` ({test['file']})\n"

    if categorized['implemented']:
        report += f"""
## 🟢 Implemented ({len(categorized['implemented'])})

Oracle tests with implementation complete and passing.

"""
        for test in sorted(categorized['implemented'], key=lambda x: x['test_name'])[:30]:
            report += f"- `{test['test_name']}` ({test['file']})\n"

        if len(categorized['implemented']) > 30:
            report += f"\n... and {len(categorized['implemented']) - 30} more\n"

    if categorized['partial']:
        report += f"""
## 🟡 Partial ({len(categorized['partial'])})

Oracle tests with incomplete or partial implementation.

"""
        for test in sorted(categorized['partial'], key=lambda x: x['test_name']):
            reason = test.get('reason', 'partial implementation')
            report += f"- `{test['test_name']}` ({test['file']}) — {reason}\n"

    report += """
---

## How to Contribute

When implementing a pending oracle:
1. Remove `#[ignore]` from the test
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify
4. The oracle status will auto-update on next CI run

For details: [CLAUDE.md](../CLAUDE.md)
"""

    return report


def main():
    import sys
    print("Scanning ALICE-Physics oracle tests...", file=sys.stderr)

    categorized = run_tests_and_categorize()
    report = generate_markdown_report(categorized)

    output_file = DOCS_DIR / "oracle-status.md"
    output_file.write_text(report)

    print(f"✅ Generated {output_file}", file=sys.stderr)
    print(f"   Implemented: {len(categorized['implemented'])}", file=sys.stderr)
    print(f"   Partial: {len(categorized['partial'])}", file=sys.stderr)
    print(f"   Pending: {len(categorized['pending'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
