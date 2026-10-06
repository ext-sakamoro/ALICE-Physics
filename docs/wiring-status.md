# ALICE-Physics Wiring Status

_Generated from `scripts/wiring-baseline.txt` and `scripts/wiring_guard.py` (no timestamp: the file changes only when its content does)._

## Status

🟡 **17 baseline items** — Permitted violations, ratchet in place

---

## 📋 Baseline (17 permitted)

Violations explicitly allowed via `scripts/wiring-baseline.txt`.
Must resolve or remove from baseline to reduce ratchet.

### By file

| File | Baseline lines |
|------|----------------|
| `src/solver_tgs_hooks.rs` | 5 |
| `src/solver_tgs_hooks_6dof_scoped.rs` | 4 |
| `src/solver_tgs.rs` | 2 |
| `src/solver_tgs_hooks_6dof.rs` | 2 |
| `src/solver_tgs_hooks_6dof_oriented.rs` | 2 |
| `src/solver_tgs_hooks_6dof_oriented_scoped.rs` | 2 |

### Dead Code (6)

```
dead_code src/solver_tgs.rs 1
dead_code src/solver_tgs_hooks.rs 1
dead_code src/solver_tgs_hooks_6dof.rs 1
dead_code src/solver_tgs_hooks_6dof_oriented.rs 1
dead_code src/solver_tgs_hooks_6dof_oriented_scoped.rs 1
dead_code src/solver_tgs_hooks_6dof_scoped.rs 1
```

### Unwired Items (11)

```
unwired src/solver_tgs.rs::par_dispatch_islands
unwired src/solver_tgs_hooks.rs::PgsConfig
unwired src/solver_tgs_hooks.rs::PgsHooks
unwired src/solver_tgs_hooks.rs::SimpleBodyState
unwired src/solver_tgs_hooks.rs::SimpleContact
unwired src/solver_tgs_hooks_6dof.rs::Pgs6DofHooks
unwired src/solver_tgs_hooks_6dof_oriented.rs::local_to_world
unwired src/solver_tgs_hooks_6dof_oriented_scoped.rs::solve_oriented_islands_parallel
unwired src/solver_tgs_hooks_6dof_scoped.rs::solve_island_isolated
unwired src/solver_tgs_hooks_6dof_scoped.rs::solve_islands_parallel
unwired src/solver_tgs_hooks_6dof_scoped.rs::solve_islands_serial
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
