# ALICE-Physics Wiring Status

_Generated from `scripts/wiring-baseline.txt` and `scripts/wiring_guard.py` (no timestamp: the file changes only when its content does)._

## Status

🟡 **40 baseline items** — Permitted violations, ratchet in place

---

## 📋 Baseline (40 permitted)

Violations explicitly allowed via `scripts/wiring-baseline.txt`.
Must resolve or remove from baseline to reduce ratchet.

### By file

| File | Baseline lines |
|------|----------------|
| `src/eulerian_grid.rs` | 10 |
| `src/bvh.rs` | 5 |
| `src/fatigue.rs` | 4 |
| `src/motor.rs` | 4 |
| `src/solver_tgs_hooks_6dof_scoped.rs` | 4 |
| `src/buckling.rs` | 2 |
| `src/plastic.rs` | 2 |
| `src/solver_tgs.rs` | 2 |
| `src/solver_tgs_hooks_6dof_oriented.rs` | 2 |
| `src/solver_tgs_hooks_6dof_oriented_scoped.rs` | 2 |
| `src/creep_longterm.rs` | 1 |
| `src/solver_tgs_hooks.rs` | 1 |
| `src/solver_tgs_hooks_6dof.rs` | 1 |

### Dead Code (7)

```
dead_code src/bvh.rs 2
dead_code src/solver_tgs.rs 1
dead_code src/solver_tgs_hooks.rs 1
dead_code src/solver_tgs_hooks_6dof.rs 1
dead_code src/solver_tgs_hooks_6dof_oriented.rs 1
dead_code src/solver_tgs_hooks_6dof_oriented_scoped.rs 1
dead_code src/solver_tgs_hooks_6dof_scoped.rs 1
```

### Unwired Items (33)

```
unwired src/buckling.rs::plate_buckling_mpa
unwired src/buckling.rs::snap_through_load_n
unwired src/bvh.rs::build_dynamic
unwired src/bvh.rs::clear_dynamic
unwired src/bvh.rs::insert_dynamic
unwired src/bvh.rs::query_pairs
unwired src/creep_longterm.rs::petg_25c_moderate
unwired src/eulerian_grid.rs::bytes
unwired src/eulerian_grid.rs::bytes
unwired src/eulerian_grid.rs::bytes
unwired src/eulerian_grid.rs::bytes
unwired src/eulerian_grid.rs::enforce_slab_face_boundaries_over
unwired src/eulerian_grid.rs::project_pressure_decomposed_on_rank
unwired src/eulerian_grid.rs::project_pressure_slab_local_on_rank
unwired src/eulerian_grid.rs::set_u
unwired src/eulerian_grid.rs::set_v
unwired src/eulerian_grid.rs::set_w
unwired src/fatigue.rs::aluminum_a5052
unwired src/fatigue.rs::analyze_spectrum
unwired src/fatigue.rs::steel_sus304
unwired src/fatigue.rs::stress_at_cycles
unwired src/motor.rs::apply_motors
unwired src/motor.rs::disable
unwired src/motor.rs::set_rotation_target
unwired src/motor.rs::set_velocity_target
unwired src/plastic.rs::petg_room_temp
unwired src/plastic.rs::with_hardening
unwired src/solver_tgs.rs::par_dispatch_islands
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
