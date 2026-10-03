# ALICE-Physics Oracle Status

_Generated from `tests/*.rs` (no timestamp: the file changes only when its content does)._

## Summary

| Category | Count |
|----------|-------|
| 🟢 Not ignored (run by CI) | 2203 |
| 🔴 Red by design | 3 |
| ⏱ Gated (runtime / diagnostic / manual) | 24 |
| ⚪ Pending (bare `#[ignore]`) | 0 |
| **Total** | **2230** |

`Not ignored` means only that the test carries no `#[ignore]`: this report does not run it.
CI's `cargo test` is what says whether it passes.

## 🔴 Red by design (3)

Oracles kept red on purpose: the implementation is not there yet, and a companion test pins
today's behaviour so CI coverage is not lost. The fix is in `src/`; the expected value is never loosened.

- `rigid_rotation_produces_zero_stress` (analytic_large_rotation.rs) — the red is correct: a rigid rotation must carry no stress, and the small-strain solver cannot deliver that yet…
- `the_answer_does_not_depend_on_the_increment_count` (analytic_corotational.rs) — the red is correct: bit-identical displacements across increment counts need an exact fixed point of the frame…
- `the_reattachment_length_matches_gartling` (armaly_backward_step.rs) — src gap: measured 2026-10-02 on this scene (ny = 8, default SemiLagrangian, L = 16 so nx = 128 is a power of t…

## ⏱ Gated (24)

Correct tests that are too slow for every push, or that print a measurement table.
Run them with `python3 scripts/run_ignored.py` or `cargo test --release -- --ignored`.

- `adaptive_cubic_beats_uniform_per_node` (analytic_adaptive_refinement_high_order.rs) — runtime: about 40 s in release (P3 reference at two uniform passes, a uniform coarse solve and a six-round ada…
- `adaptive_quadratic_beats_uniform_per_node` (analytic_adaptive_refinement_high_order.rs) — runtime: about 20 s in release (P2 reference at two uniform passes, a uniform coarse solve and a six-round ada…
- `amplification_growth_is_problem_size_or_element_shape` (mesh_to_fem_stress.rs) — diagnostic: run when the amplification threshold is in question
- `cantilever_order_estimates_agree` (analytic_fem_convergence.rs) — 25,600 tets at cell 0.25; run with --release, see the doc comment
- `cantilever_without_preconditioner` (analytic_fem_convergence.rs) — 25,600 tets at cell 0.25; the A/B partner of the test above
- `cavity_gap_to_ghia_shrinks_with_resolution` (analytic_cfd_wall_bc.rs) — resolution sweep: three cavity runs to t = 30, minutes in a debug build
- `cavity_size_sweep` (mesh_quality.rs) — diagnostic: the table behind a_cavity_narrower_than_a_cell_leaves_no_trace
- `cavity_trace_by_size` (mesh_quality.rs) — diagnostic: the evidence table for what each cavity size leaves behind
- `channel_error_is_second_order_in_the_cell_size` (analytic_cfd_flow_bc.rs) — runtime only: 43k steps across three resolutions, release-only; the accuracy it asks for is reached, see the d…
- `duct_profile_develops_toward_the_parabola` (analytic_cfd_flow_bc.rs) — runtime only: 4096 steps on 24x16 cells with 200 GS sweeps, release-only; the accuracy it asks for is reached,…
- `finest_level_residual_floor_by_preconditioner` (analytic_fem_convergence.rs) — 4 solves at 25,600 tets; the stopping-rule diagnostic
- `gs_control_rates` (analytic_multigrid.rs) — diagnostic: GS control rates for oracle 2
- `gs_satisfies_exact_solution_oracle` (analytic_multigrid.rs) — diagnostic: oracle validity against the existing GS
- `iterations_to_same_precision` (analytic_multigrid.rs) — diagnostic: cycles vs GS iterations to the same precision (use --release)
- `maccormack_converges_and_the_schemes_bracket_the_limit_at_ny_8_16_32` (armaly_backward_step.rs) — manual: about 4 h in release (ny = 8 / 16 / 32, both schemes, each to t_settle), longer than the 180-min ignor…
- `mg_rate_report` (analytic_multigrid.rs) — diagnostic: multigrid rates used to fix the thresholds
- `p2_error_decreases_with_refinement_at_better_than_second_order` (analytic_hyperelastic_mms_order.rs) — runtime: about 5 s in release, about 60 s in debug (P2 n = 2, 3, 4 with a Fix128 Newton solve each); run by ru…
- `p3_separates_from_p2_at_large_amplitude` (analytic_hyperelastic_mms_order.rs) — runtime: about 35 s in release (P3 n = 3 at A = 0.08 with the Newton-Krylov step, plus P2 n = 3); run by run_i…
- `p3_separates_from_p2_in_order_and_in_error` (analytic_hyperelastic_mms_order.rs) — runtime: about 30 s in release (P3 n = 2 and 3 plus P2 n = 3, Fix128 Newton on 20-node elements); run by run_i…
- `reattachment_lengthens_under_grid_refinement` (armaly_backward_step.rs) — runtime only: 256x16 cells for 8192 steps, release-only; the measured values are in the module header
- `the_two_schemes_bracket_the_reattachment_length_at_ny_8_and_16` (armaly_backward_step.rs) — runtime: about 30 min in release (ny = 8 and 16, both advection schemes, each to t_settle at Courant 0.75); ru…
- `tolerance_floor_rises_as_the_mesh_refines` (analytic_fem_convergence.rs) — 4 solves; the tolerance-floor measurement
- `tolerance_measurement` (analytic_step_multigrid.rs) — diagnostic: the measurements the two tolerances above are fixed from
- `x_1_time_trace` (armaly_backward_step.rs) — diagnostic: x_1(t) trace for one resolution and scheme, settings from ARM_NY / ARM_SCHEME / ARM_DT_RECIP / ARM…

## 🟢 Not ignored (2203)

Per-file counts (the test names are in `tests/`):

| File | Tests |
|------|-------|
| `integration_physics.rs` | 75 |
| `engineering_oracles_fluid.rs` | 40 |
| `analytic_maxwell_fdtd.rs` | 38 |
| `analytic_raycast_wiring.rs` | 37 |
| `analytic_sdf_character_up_axis.rs` | 35 |
| `analytic_world_api.rs` | 35 |
| `engineering_oracles_solid.rs` | 34 |
| `analytic_static_collider.rs` | 27 |
| `engineering_oracles_misc.rs` | 27 |
| `analytic_maxwell_wiring.rs` | 26 |
| `analytic_character_wiring.rs` | 25 |
| `analytic_p2g.rs` | 25 |
| `analytic_plastic_dissipation.rs` | 25 |
| `analytic_compound_wiring.rs` | 24 |
| `analytic_query_wiring.rs` | 24 |
| `analytic_transient_thermal_wiring.rs` | 24 |
| `analytic_compound.rs` | 23 |
| `analytic_convex_contact.rs` | 23 |
| `analytic_reactions.rs` | 22 |
| `determinism_semantic.rs` | 22 |
| `analytic_adaptive_refinement.rs` | 21 |
| `analytic_corotational.rs` | 21 |
| `analytic_multiphase_wiring.rs` | 21 |
| `analytic_neural_wiring.rs` | 21 |
| `analytic_rope_wiring.rs` | 21 |
| `analytic_fluid_netcode_wiring.rs` | 20 |
| `analytic_joint_extra_wiring.rs` | 20 |
| `analytic_wave_ship_wiring.rs` | 20 |
| `analytic_compressible_wiring.rs` | 19 |
| `analytic_elastoplastic_fem.rs` | 19 |
| `analytic_heatmap_wiring.rs` | 19 |
| `analytic_laminate_wiring.rs` | 19 |
| `analytic_prestressed_wiring.rs` | 19 |
| `analytic_rolling_contact_wiring.rs` | 19 |
| `analytic_tgs_wiring.rs` | 19 |
| `analytic_filament_db_wiring.rs` | 18 |
| `analytic_flip.rs` | 18 |
| `analytic_hyperelastic_wiring.rs` | 18 |
| `analytic_multi_world_wiring.rs` | 18 |
| `analytic_sdf_force_wiring.rs` | 18 |
| `analytic_thin_wall_wiring.rs` | 18 |
| `analytic_wind_zone_wiring.rs` | 18 |
| `analytic_acoustic_wave_wiring.rs` | 17 |
| `analytic_anomaly_wiring.rs` | 17 |
| `analytic_ccd_wiring.rs` | 17 |
| `analytic_debug_render_wiring.rs` | 17 |
| `analytic_gpu_sdf_wiring.rs` | 17 |
| `analytic_interpolation_wiring.rs` | 17 |
| `analytic_modal_wiring.rs` | 17 |
| `analytic_audio_physics_wiring.rs` | 16 |
| `analytic_damping_rayleigh_wiring.rs` | 16 |
| `analytic_linear_elastic_fem.rs` | 16 |
| `analytic_profiling_wiring.rs` | 16 |
| `analytic_shape.rs` | 16 |
| `armaly_backward_step.rs` | 16 |
| `analytic_articulation_wiring.rs` | 15 |
| `analytic_dynamic_fem.rs` | 15 |
| `analytic_multibody_dynamics.rs` | 15 |
| `analytic_pipeline_wiring.rs` | 15 |
| `analytic_privacy_wiring.rs` | 15 |
| `analytic_sdf_body_collider.rs` | 15 |
| `analytic_self_contact.rs` | 15 |
| `analytic_thermoplastic_coupling.rs` | 15 |
| `analytic_collision_mesh.rs` | 14 |
| `analytic_convex_decompose.rs` | 14 |
| `analytic_joint_wiring.rs` | 14 |
| `analytic_sdf_destruction_wiring.rs` | 14 |
| `analytic_sdf_fem_mesh_wiring.rs` | 14 |
| `analytic_smoke_fire_wiring.rs` | 14 |
| `analytic_anisotropic_wiring.rs` | 13 |
| `analytic_cubic_elastic_fem_wiring.rs` | 13 |
| `analytic_filter_wiring.rs` | 13 |
| `analytic_kinematic_loop_wiring.rs` | 13 |
| `analytic_math_wiring.rs` | 13 |
| `analytic_physics.rs` | 13 |
| `analytic_rans.rs` | 13 |
| `analytic_sim_modifier_wiring.rs` | 13 |
| `analytic_sketch_wiring.rs` | 13 |
| `default_configs.rs` | 13 |
| `determinism_golden_f32.rs` | 13 |
| `analytic_collision_mesh_gen_wiring.rs` | 12 |
| `analytic_contact_cache_wiring.rs` | 12 |
| `analytic_layer_adhesion_wiring.rs` | 12 |
| `analytic_animation_blend_wiring.rs` | 11 |
| `analytic_contact_viz_wiring.rs` | 11 |
| `analytic_coupled_field.rs` | 11 |
| `analytic_flow_viz_wiring.rs` | 11 |
| `analytic_mass_properties.rs` | 11 |
| `analytic_polar_decomposition.rs` | 11 |
| `analytic_pressure_wiring.rs` | 11 |
| `analytic_replay_wiring.rs` | 11 |
| `analytic_temperature_rise.rs` | 11 |
| `analytic_vibration_wall_wiring.rs` | 11 |
| `analytic_cfd_flow_bc.rs` | 10 |
| `analytic_cubic_hyperelastic.rs` | 10 |
| `analytic_db_bridge_wiring.rs` | 10 |
| `analytic_fsi_advanced_wiring.rs` | 10 |
| `analytic_linear_elastic_fem_wiring_additional.rs` | 10 |
| `analytic_pressure_solvers.rs` | 10 |
| `coupling_channel_inventory.rs` | 10 |
| `fix128_vs_f64_coupling_hypotheses.rs` | 10 |
| `analytic_adaptive_dt.rs` | 9 |
| `analytic_anisotropic_friction_wiring.rs` | 9 |
| `analytic_contact_forces.rs` | 9 |
| `analytic_flip_scatter.rs` | 9 |
| `analytic_laminate_failure_wiring.rs` | 9 |
| `analytic_material_wiring.rs` | 9 |
| `analytic_quadratic_fem.rs` | 9 |
| `analytic_quadratic_hyperelastic.rs` | 9 |
| `analytic_thermoplastic_softening.rs` | 9 |
| `determinism_golden.rs` | 9 |
| `analytic_added_mass_coupling.rs` | 8 |
| `analytic_broadphase.rs` | 8 |
| `analytic_cubic_fem.rs` | 8 |
| `analytic_fillet_stress_wiring.rs` | 8 |
| `analytic_metric_broadphase.rs` | 8 |
| `analytic_netcode_prediction.rs` | 8 |
| `analytic_step_multigrid.rs` | 8 |
| `analytic_wall_model.rs` | 8 |
| `mesh_conformity.rs` | 8 |
| `analytic_ccd_adaptive_substeps_wiring.rs` | 7 |
| `analytic_coupled_wiring.rs` | 7 |
| `analytic_hyperelastic_degenerate.rs` | 7 |
| `analytic_multigrid.rs` | 7 |
| `analytic_non_newtonian_wiring.rs` | 7 |
| `analytic_piezoelectric_wiring.rs` | 7 |
| `analytic_soft_body_cut_wiring.rs` | 7 |
| `elastoplastic_increment_api.rs` | 7 |
| `fsi_advanced_sub_iteration.rs` | 7 |
| `mesh_quality.rs` | 7 |
| `wm07_reset_and_rollback_contract.rs` | 7 |
| `analytic_analytics_bridge_wiring.rs` | 6 |
| `analytic_bvh_leaf_aabb_wiring.rs` | 6 |
| `analytic_geometry_helpers.rs` | 6 |
| `analytic_thermoelastic.rs` | 6 |
| `analytic_thermoelastic_channel.rs` | 6 |
| `cloth_fluid_sub_iteration.rs` | 6 |
| `hanging_node_effect.rs` | 6 |
| `p3_quadrature_fix128.rs` | 6 |
| `wm01_overflow_is_not_silent.rs` | 6 |
| `wm08_checksum_coverage.rs` | 6 |
| `analytic_cfd_wall_bc.rs` | 5 |
| `analytic_hyperelastic_mms_order.rs` | 5 |
| `analytic_math_ln.rs` | 5 |
| `analytic_solver_tgs_dispatch_wiring.rs` | 5 |
| `engineering_oracles.rs` | 5 |
| `mms_linear_elastic.rs` | 5 |
| `reduction_order_independence.rs` | 5 |
| `analytic_adaptive_refinement_high_order.rs` | 4 |
| `analytic_step_default_projection.rs` | 4 |
| `locking_p1.rs` | 4 |
| `p2_oracle_design.rs` | 4 |
| `p2_quadrature_fix128.rs` | 4 |
| `refinement_conformity.rs` | 4 |
| `analytic_boundary_faces.rs` | 3 |
| `parallel_batch_coloring.rs` | 3 |
| `wm01_flag_survives_rollback.rs` | 3 |
| `wm08_prev_state_coverage.rs` | 3 |
| `wm08_state_coverage.rs` | 3 |
| `analytic_large_rotation.rs` | 2 |
| `mesh_to_fem_stress.rs` | 2 |
| `wm07_rollback_event_parity.rs` | 2 |
| `analytic_fem_convergence.rs` | 1 |

---

## How to Contribute

When an oracle goes green:
1. Remove `#[ignore]` from the test (and the companion test that pins the old behaviour, if the reason says so)
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify

For details: [CLAUDE.md](../CLAUDE.md)
