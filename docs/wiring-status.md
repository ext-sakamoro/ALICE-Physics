# ALICE-Physics Wiring Status

**Last updated:** 2026-10-03 10:15

## Status

✅ **All clear** — No wiring violations detected

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
