# Wiring debt

Items that are implemented and tested but not yet called from the production
path carry `// ALLOW-UNWIRED: wiring debt <key>` in the source. Each key is one
row here: what is missing and what closes it. `scripts/wiring_guard.py` checks
both directions (a marker whose key has no row, and a row no marker uses, both
fail), so this table and the markers stay in step. Remove the row together with
the last marker that uses it.

| key | what is not wired | closes when |
|---|---|---|
| `equilibrated-cg-stopping-norm` | `coupled_iteration::EquilibrationScale` and the equilibrated residual norm exist with their own oracle (`tests/analytic_coupled_wiring.rs`), but the conjugate-gradient solvers still stop on the plain residual norm | the CG stopping test uses the equilibrated norm (changes converged results, so the determinism goldens are regenerated with it) |
| `sub-iteration-report-best-residual` | `SubIterationState::best_residual` is tracked and tested (`tests/analytic_added_mass_coupling.rs`), but `SubIterationReport` has no field for it | 2.0.0, when `SubIterationReport` can gain a field |
| `structural-pub-crate-residue` | crate-internal material presets and fields (the PETG creep fits in `creep_longterm` / `plastic`, `hardening_type`) that `structural_solver::new` never selects: it installs the PLA preset only | the structural solver selects the preset from the material, or the presets become part of the public material API |
| `eulerian-grid-cross-process-primitive-residue` | crate-internal primitives of the distributed pressure projection that only a process-per-rank run reaches; the single-process drivers never need them, and examples cannot reach `pub(crate)` items | a public multi-process driver calls them (they are exercised by the cross-process tests in `src/eulerian_grid.rs`) |
