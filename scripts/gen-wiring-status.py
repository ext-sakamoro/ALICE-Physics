#!/usr/bin/env python3
"""
Generate ALICE-Physics wiring status report.

Runs wiring_guard.py and generates a markdown report of unwired/dead-code items.
"""

import subprocess
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
WIRING_GUARD = PROJECT_ROOT / "scripts" / "wiring_guard.py"

DOCS_DIR.mkdir(exist_ok=True)


def run_wiring_guard():
    """Execute wiring_guard and capture output."""
    try:
        result = subprocess.run(
            ["python3", str(WIRING_GUARD)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running wiring_guard: {e}\n"


def parse_wiring_output(output):
    """Parse wiring_guard output into structured data."""
    violations = {
        'dead_code': [],
        'unwired': [],
        'stale_baseline': [],
        'unbalanced_braces': [],
        'ok': False,
    }

    # Check for ok status
    if 'wiring-guard: ok' in output:
        violations['ok'] = True
        return violations

    # Parse violations (lines starting with "wiring-guard:")
    for line in output.split('\n'):
        if line.startswith('wiring-guard:'):
            # Extract violation type and file:line
            if 'dead_code' in line:
                violations['dead_code'].append(line)
            elif 'unwired' in line:
                violations['unwired'].append(line)
            elif 'stale_baseline' in line:
                violations['stale_baseline'].append(line)
            elif 'unbalanced_braces' in line:
                violations['unbalanced_braces'].append(line)

    return violations


def generate_markdown_report(violations):
    """Generate markdown report from violations."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    if violations['ok']:
        status_summary = "✅ **All clear** — No wiring violations detected"
        return f"""# ALICE-Physics Wiring Status

**Last updated:** {timestamp}

## Status

{status_summary}

### Checks

- ✅ **Dead Code Guard**: No unchecked `#[allow(dead_code)]`
- ✅ **Unwired Items**: No unused public items
- ✅ **Stale Baseline**: No obsolete baseline entries
- ✅ **Brace Balance**: No unbalanced braces

---

## What is the Wiring Guard?

The wiring guard ensures that all public items in `src/` are actually called from production code:

- **Dead Code Guard**: Verifies that `#[allow(dead_code)]` has a documented reason
- **Unwired Items**: Detects public functions, structs, etc. that are never called (except in tests)
- **Stale Baseline**: Ensures baseline entries are still needed
- **Brace Balance**: Checks syntax integrity

Violations are tracked in `scripts/wiring-baseline.txt` and must be explicitly allowed.

For details: see `scripts/wiring_guard.py`
"""

    else:
        # Build violation report
        sections = []

        if violations['dead_code']:
            sections.append(f"""## 🔴 Dead Code Guard ({len(violations['dead_code'])})

Items with `#[allow(dead_code)]` but missing reasons:

```
{chr(10).join(violations['dead_code'])}
```
""")

        if violations['unwired']:
            sections.append(f"""## 🔴 Unwired Items ({len(violations['unwired'])})

Public items never called from production code:

```
{chr(10).join(violations['unwired'])}
```
""")

        if violations['stale_baseline']:
            sections.append(f"""## 🔴 Stale Baseline ({len(violations['stale_baseline'])})

Baseline entries that are no longer needed:

```
{chr(10).join(violations['stale_baseline'])}
```
""")

        if violations['unbalanced_braces']:
            sections.append(f"""## 🔴 Syntax Errors ({len(violations['unbalanced_braces'])})

Files with unbalanced braces:

```
{chr(10).join(violations['unbalanced_braces'])}
```
""")

        return f"""# ALICE-Physics Wiring Status

**Last updated:** {timestamp}

## Status

❌ **Violations detected** — Fix wiring issues before push

{chr(10).join(sections)}

---

## How to Fix

1. **Dead Code**: Add a `// ALLOW-DEAD: <reason>` comment or remove `#[allow(dead_code)]`
2. **Unwired**: Either call the item from production, or add `// ALLOW-UNWIRED: <reason>`
3. **Stale Baseline**: Remove the entry from `scripts/wiring-baseline.txt`
4. **Syntax**: Fix unbalanced braces in the named files

For details: see `scripts/wiring_guard.py`
"""


def main():
    import sys

    print("Running wiring guard...", file=sys.stderr)
    output = run_wiring_guard()

    violations = parse_wiring_output(output)
    report = generate_markdown_report(violations)

    output_file = DOCS_DIR / "wiring-status.md"
    output_file.write_text(report)

    print(f"✅ Generated {output_file}", file=sys.stderr)

    if violations['ok']:
        print("   Status: OK", file=sys.stderr)
        return 0
    else:
        print("   Status: VIOLATIONS", file=sys.stderr)
        print(f"   - Dead Code: {len(violations['dead_code'])}", file=sys.stderr)
        print(f"   - Unwired: {len(violations['unwired'])}", file=sys.stderr)
        print(f"   - Stale Baseline: {len(violations['stale_baseline'])}", file=sys.stderr)
        print(f"   - Syntax Errors: {len(violations['unbalanced_braces'])}", file=sys.stderr)
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
