# ALICE-Physics Wiring Status

_Generated from `scripts/wiring-baseline.txt` and `scripts/wiring_guard.py` (no timestamp: the file changes only when its content does)._

## Status

🟡 **5 baseline items** — Permitted violations, ratchet in place

---

## 📋 Baseline (5 permitted)

Violations explicitly allowed via `scripts/wiring-baseline.txt`.
Must resolve or remove from baseline to reduce ratchet.

### By file

| File | Baseline lines |
|------|----------------|
| `src/solver_tgs.rs` | 2 |
| `src/solver_tgs_hooks_6dof_oriented_scoped.rs` | 2 |
| `src/solver_tgs_hooks_6dof_oriented.rs` | 1 |

### Dead Code (3)

```
dead_code src/solver_tgs.rs 1
dead_code src/solver_tgs_hooks_6dof_oriented.rs 1
dead_code src/solver_tgs_hooks_6dof_oriented_scoped.rs 1
```

### Unwired Items (2)

```
unwired src/solver_tgs.rs::par_dispatch_islands
unwired src/solver_tgs_hooks_6dof_oriented_scoped.rs::solve_oriented_islands_parallel
```

---

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
