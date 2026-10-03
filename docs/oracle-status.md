# ALICE-Physics Oracle Status

**Last updated:** 2026-10-03 18:16

## Summary

| Category | Count |
|----------|-------|
| 🟢 Implemented | 903 |
| 🟡 Partial | 0 |
| 🔴 Pending | 22 |
| **Total** | **925** |

## 🔴 Pending (22)

Oracle tests not yet implemented (marked with `#[ignore]`).

- `amplification_growth_is_problem_size_or_element_shape` (mesh_to_fem_stress.rs) — diagnostic: run when the amplification threshold is in question
- `cantilever_order_estimates_agree` (analytic_fem_convergence.rs) — 25,600 tets at cell 0.25; run with --release, see the doc comment
- `cantilever_without_preconditioner` (analytic_fem_convergence.rs) — 25,600 tets at cell 0.25; the A/B partner of the test above
- `cavity_gap_to_ghia_shrinks_with_resolution` (analytic_cfd_wall_bc.rs) — resolution sweep: three cavity runs to t = 30, minutes in a debug build
- `cavity_size_sweep` (mesh_quality.rs) — diagnostic: the table behind a_cavity_narrower_than_a_cell_leaves_no_trace
- `cavity_trace_by_size` (mesh_quality.rs) — diagnostic: the evidence table for what each cavity size leaves behind
- `channel_error_is_second_order_in_the_cell_size` (analytic_cfd_flow_bc.rs) — runtime only: 43k steps across three resolutions, release-only; the accuracy it …
- `duct_profile_develops_toward_the_parabola` (analytic_cfd_flow_bc.rs) — runtime only: 4096 steps on 24x16 cells with 200 GS sweeps, release-only; the ac…
- `finest_level_residual_floor_by_preconditioner` (analytic_fem_convergence.rs) — 4 solves at 25,600 tets; the stopping-rule diagnostic
- `gs_control_rates` (analytic_multigrid.rs) — diagnostic: GS control rates for oracle 2
- `gs_satisfies_exact_solution_oracle` (analytic_multigrid.rs) — diagnostic: oracle validity against the existing GS
- `iterations_to_same_precision` (analytic_multigrid.rs) — diagnostic: cycles vs GS iterations to the same precision (use --release)
- `maccormack_converges_and_the_schemes_bracket_the_limit_at_ny_8_16_32` (armaly_backward_step.rs) — manual: about 4 h in release (ny = 8 / 16 / 32, both schemes, each to t_settle),…
- `mg_rate_report` (analytic_multigrid.rs) — diagnostic: multigrid rates used to fix the thresholds
- `reattachment_lengthens_under_grid_refinement` (armaly_backward_step.rs) — runtime only: 256x16 cells for 8192 steps, release-only; the measured values are…
- `rigid_rotation_produces_zero_stress` (analytic_large_rotation.rs) — pending
- `the_answer_does_not_depend_on_the_increment_count` (analytic_corotational.rs) — pending
- `the_reattachment_length_matches_gartling` (armaly_backward_step.rs) — src gap: measured 2026-10-02 on this scene (ny = 8, default SemiLagrangian, L = …
- `the_two_schemes_bracket_the_reattachment_length_at_ny_8_and_16` (armaly_backward_step.rs) — runtime: about 30 min in release (ny = 8 and 16, both advection schemes, each to…
- `tolerance_floor_rises_as_the_mesh_refines` (analytic_fem_convergence.rs) — 4 solves; the tolerance-floor measurement
- `tolerance_measurement` (analytic_step_multigrid.rs) — diagnostic: the measurements the two tolerances above are fixed from
- `x_1_time_trace` (armaly_backward_step.rs) — diagnostic: x_1(t) trace for one resolution and scheme, settings from ARM_NY / A…

## 🟢 Implemented (903)

Oracle tests with implementation complete and passing.

- `a_body_flush_with_the_grid_deposits_its_whole_heat_into_the_dual_ledger` (analytic_plastic_dissipation.rs)
- `a_boundary_index_past_the_end_is_refused_and_does_not_panic` (analytic_hyperelastic_degenerate.rs)
- `a_budget_of_one_sweep_is_reported_as_not_converged` (analytic_thermoplastic_coupling.rs)
- `a_cavity_narrower_than_a_cell_leaves_no_trace` (mesh_quality.rs)
- `a_clamped_push_lands_in_the_analytic_free_band` (analytic_sdf_character_up_axis.rs)
- `a_closed_box_is_no_slip_except_where_it_is_one_cell_thick` (analytic_cfd_flow_bc.rs)
- `a_closed_current_loop_creates_no_charge` (analytic_maxwell_fdtd.rs)
- `a_collision_free_move_reports_the_sampled_distance_as_best` (analytic_sdf_character_up_axis.rs)
- `a_configuration_without_a_law_is_refused` (analytic_cubic_hyperelastic.rs)
- `a_configuration_without_a_law_is_refused` (analytic_quadratic_hyperelastic.rs)
- `a_configuration_without_a_law_is_refused_on_both_elements` (analytic_hyperelastic_degenerate.rs)
- `a_contact_does_not_add_velocity_away_from_the_surface` (analytic_sdf_character_up_axis.rs)
- `a_contracting_splitting_stays_contracting_at_every_time_step` (analytic_added_mass_coupling.rs)
- `a_contracting_splitting_trips_no_guard` (analytic_added_mass_coupling.rs)
- `a_converged_step_multigrid_matches_a_converged_step` (analytic_step_multigrid.rs)
- `a_crumpled_cloth_does_not_pass_through_itself` (analytic_self_contact.rs)
- `a_cube_metric_clearance_is_missed_without_the_expansion_and_caught_with_it` (analytic_metric_broadphase.rs)
- `a_current_inside_the_absorber_is_rejected` (analytic_maxwell_fdtd.rs)
- `a_current_on_a_magnetic_face_is_rejected` (analytic_maxwell_fdtd.rs)
- `a_current_on_a_pec_edge_is_rejected` (analytic_maxwell_fdtd.rs)
- `a_current_that_ends_deposits_exactly_the_charge_that_left` (analytic_maxwell_fdtd.rs)
- `a_degenerate_axis_is_refused_rather_than_silently_rescaled` (analytic_plastic_dissipation.rs)
- `a_degenerate_grid_axis_is_refused` (analytic_thermoplastic_coupling.rs)
- `a_degenerate_up_axis_falls_back_to_y_instead_of_producing_nan` (analytic_sdf_character_up_axis.rs)
- `a_deposit_onto_a_mismatched_mesh_is_rejected` (analytic_plastic_dissipation.rs)
- `a_diverging_iteration_reports_a_zero_l2_residual` (analytic_added_mass_coupling.rs)
- `a_field_that_does_not_cover_the_mesh_is_refused` (analytic_thermoelastic.rs)
- `a_field_that_does_not_cover_the_mesh_is_refused` (analytic_thermoplastic_coupling.rs)
- `a_field_that_does_not_cover_the_mesh_is_refused` (analytic_thermoplastic_softening.rs)
- `a_field_that_varies_along_the_bar_is_sampled_per_element` (analytic_thermoplastic_softening.rs)

... and 873 more

---

## How to Contribute

When implementing a pending oracle:
1. Remove `#[ignore]` from the test
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify
4. The oracle status will auto-update on next CI run

For details: [CLAUDE.md](../CLAUDE.md)
