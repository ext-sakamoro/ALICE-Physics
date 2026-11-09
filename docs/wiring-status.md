# ALICE-Physics Wiring Status

_Generated from `scripts/wiring-baseline.txt` and `scripts/wiring_guard.py` (no timestamp: the file changes only when its content does)._

## Status

❌ **NEW violations detected** — Must be resolved or added to baseline

---

## 🔴 NEW: Unwired Items (10)

```
unwired: src/axis_lock.rs::LINEAR_X: LINEAR_X: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/axis_lock.rs::LINEAR_Y: LINEAR_Y: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/axis_lock.rs::LINEAR_Z: LINEAR_Z: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/axis_lock.rs::ANGULAR_X: ANGULAR_X: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/axis_lock.rs::ANGULAR_Y: ANGULAR_Y: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/axis_lock.rs::ANGULAR_Z: ANGULAR_Z: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/axis_lock.rs::LINEAR: LINEAR: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/axis_lock.rs::ANGULAR: ANGULAR: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/solver.rs::set_body_axis_lock: set_body_axis_lock: production code から 1 度も参照されていない (test / doc / use のみ)
unwired: src/solver.rs::body_axis_lock: body_axis_lock: production code から 1 度も参照されていない (test / doc / use のみ)
```

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
