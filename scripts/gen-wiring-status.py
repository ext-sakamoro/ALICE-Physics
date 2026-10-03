#!/usr/bin/env python3
"""
Generate ALICE-Physics wiring status report.

Scans baseline and wiring_guard output to report permitted vs new violations.
"""

import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
WIRING_GUARD = PROJECT_ROOT / "scripts" / "wiring_guard.py"
BASELINE_FILE = PROJECT_ROOT / "scripts" / "wiring-baseline.txt"

DOCS_DIR.mkdir(exist_ok=True)


def load_baseline():
    """Load baseline (permitted violations)."""
    baseline = {
        'dead_code': [],
        'unwired': [],
    }

    if not BASELINE_FILE.exists():
        return baseline

    with open(BASELINE_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                violation_type = parts[0]
                if violation_type in baseline:
                    baseline[violation_type].append(line)

    return baseline


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


def parse_violations(output):
    """Parse wiring_guard error output."""
    violations = {
        'dead_code': [],
        'unwired': [],
        'stale_baseline': [],
        'unbalanced_braces': [],
    }

    # Parse violation lines (lines starting with "wiring-guard:")
    for line in output.split('\n'):
        if line.startswith('wiring-guard:'):
            if 'dead_code' in line:
                violations['dead_code'].append(line)
            elif 'unwired' in line:
                violations['unwired'].append(line)
            elif 'stale_baseline' in line:
                violations['stale_baseline'].append(line)
            elif 'unbalanced_braces' in line:
                violations['unbalanced_braces'].append(line)

    return violations


def generate_markdown_report(baseline, violations):
    """Generate markdown report combining baseline and current violations."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Check if there are any NEW (non-baseline) violations
    has_new_violations = bool(violations['dead_code'] or violations['unwired'] or
                               violations['stale_baseline'] or violations['unbalanced_braces'])

    total_baseline = len(baseline['dead_code']) + len(baseline['unwired'])

    if has_new_violations:
        status = "❌ **NEW violations detected** — Must be resolved or added to baseline"
    elif total_baseline > 0:
        status = f"🟡 **{total_baseline} baseline items** — Permitted violations, ratchet in place"
    else:
        status = "✅ **All clear** — No violations"

    report = f"""# ALICE-Physics Wiring Status

**Last updated:** {timestamp}

## Status

{status}

---

"""

    if has_new_violations:
        if violations['dead_code']:
            report += f"""## 🔴 NEW: Dead Code Guard ({len(violations['dead_code'])})

```
{chr(10).join(violations['dead_code'])}
```

"""

        if violations['unwired']:
            report += f"""## 🔴 NEW: Unwired Items ({len(violations['unwired'])})

```
{chr(10).join(violations['unwired'])}
```

"""

        if violations['stale_baseline']:
            report += f"""## 🔴 NEW: Stale Baseline ({len(violations['stale_baseline'])})

```
{chr(10).join(violations['stale_baseline'])}
```

"""

        if violations['unbalanced_braces']:
            report += f"""## 🔴 NEW: Syntax Errors ({len(violations['unbalanced_braces'])})

```
{chr(10).join(violations['unbalanced_braces'])}
```

"""

    if total_baseline > 0:
        report += f"""## 📋 Baseline ({total_baseline} permitted)

Violations explicitly allowed via `scripts/wiring-baseline.txt`.
Must resolve or remove from baseline to reduce ratchet.

"""

        if baseline['dead_code']:
            report += f"""### Dead Code ({len(baseline['dead_code'])})

```
{chr(10).join(baseline['dead_code'])}
```

"""

        if baseline['unwired']:
            report += f"""### Unwired Items ({len(baseline['unwired'])})

```
{chr(10).join(baseline['unwired'])}
```

"""

    report += """---

## What is the Wiring Guard?

The wiring guard ensures that all public items in `src/` are actually called from production code:

- **Dead Code Guard**: Verifies that `#[allow(dead_code)]` has a documented reason
- **Unwired Items**: Detects public functions, structs, etc. that are never called (except in tests)
- **Stale Baseline**: Ensures baseline entries are still needed
- **Brace Balance**: Checks syntax integrity

### Resolving Violations

1. **New violations**: Either implement/wire the item, or add to `scripts/wiring-baseline.txt`
2. **Baseline cleanup**: Remove lines from baseline as violations are resolved
3. **Comments**: Add `// ALLOW-DEAD:` or `// ALLOW-UNWIRED:` with reason (12+ chars)

For details: see `scripts/wiring_guard.py`
"""

    return report


def main():
    import sys

    print("Scanning wiring status...", file=sys.stderr)
    baseline = load_baseline()
    guard_output = run_wiring_guard()
    violations = parse_violations(guard_output)

    report = generate_markdown_report(baseline, violations)

    output_file = DOCS_DIR / "wiring-status.md"
    output_file.write_text(report)

    print(f"✅ Generated {output_file}", file=sys.stderr)
    print(f"   Baseline: {len(baseline['dead_code']) + len(baseline['unwired'])}", file=sys.stderr)
    print(f"   New violations: {len(violations['dead_code']) + len(violations['unwired']) + len(violations['stale_baseline']) + len(violations['unbalanced_braces'])}", file=sys.stderr)

    return 0 if not (violations['dead_code'] or violations['unwired'] or violations['stale_baseline'] or violations['unbalanced_braces']) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
