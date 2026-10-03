# ALICE-Physics Oracle Status

**Last updated:** 2026-10-03 11:04

## Summary

| Category | Count |
|----------|-------|
| 🟢 Implemented | 2195 |
| 🟡 Partial | 0 |
| 🔴 Pending | 27 |
| **Total** | **2222** |

## 🔴 Pending (27)

Oracle tests not yet implemented (marked with `#[ignore]`).

- `adaptive_cubic_beats_uniform_per_node` (analytic_adaptive_refinement_high_order.rs) — runtime: about 40 s in release (P3 reference at two uniform passes, a uniform co…
- `adaptive_quadratic_beats_uniform_per_node` (analytic_adaptive_refinement_high_order.rs) — runtime: about 20 s in release (P2 reference at two uniform passes, a uniform co…
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
- `p2_error_decreases_with_refinement_at_better_than_second_order` (analytic_hyperelastic_mms_order.rs) — runtime: about 5 s in release, about 60 s in debug (P2 n = 2, 3, 4 with a Fix128…
- `p3_separates_from_p2_at_large_amplitude` (analytic_hyperelastic_mms_order.rs) — runtime: about 35 s in release (P3 n = 3 at A = 0.08 with the Newton-Krylov step…
- `p3_separates_from_p2_in_order_and_in_error` (analytic_hyperelastic_mms_order.rs) — runtime: about 30 s in release (P3 n = 2 and 3 plus P2 n = 3, Fix128 Newton on 2…
- `reattachment_lengthens_under_grid_refinement` (armaly_backward_step.rs) — runtime only: 256x16 cells for 8192 steps, release-only; the measured values are…
- `rigid_rotation_produces_zero_stress` (analytic_large_rotation.rs) — pending
- `the_answer_does_not_depend_on_the_increment_count` (analytic_corotational.rs) — pending
- `the_reattachment_length_matches_gartling` (armaly_backward_step.rs) — src gap: measured 2026-10-02 on this scene (ny = 8, default SemiLagrangian, L = …
- `the_two_schemes_bracket_the_reattachment_length_at_ny_8_and_16` (armaly_backward_step.rs) — runtime: about 30 min in release (ny = 8 and 16, both advection schemes, each to…
- `tolerance_floor_rises_as_the_mesh_refines` (analytic_fem_convergence.rs) — 4 solves; the tolerance-floor measurement
- `tolerance_measurement` (analytic_step_multigrid.rs) — diagnostic: the measurements the two tolerances above are fixed from
- `x_1_time_trace` (armaly_backward_step.rs) — diagnostic: x_1(t) trace for one resolution and scheme, settings from ARM_NY / A…

## 🟢 Implemented (2195)

Oracle tests with implementation complete and passing.

- `a_ball_meshes_to_a_closed_sphere` (analytic_collision_mesh.rs)
- `a_body_dropped_on_a_floor_rests_one_radius_above_it` (analytic_static_collider.rs)
- `a_body_flush_with_the_grid_deposits_its_whole_heat_into_the_dual_ledger` (analytic_plastic_dissipation.rs)
- `a_body_rests_on_the_mesh_of_a_ball` (analytic_collision_mesh.rs)
- `a_body_s_own_collision_radius_is_the_sphere_tested` (analytic_static_collider.rs)
- `a_body_without_a_radius_uses_the_default_collision_radius` (analytic_static_collider.rs)
- `a_boundary_index_past_the_end_is_refused_and_does_not_panic` (analytic_hyperelastic_degenerate.rs)
- `a_box_beside_a_ball_reaches_in_by_its_nearest_face_centre` (analytic_sdf_body_collider.rs)
- `a_box_dropped_on_a_floor_box_rests_on_its_top_face` (analytic_convex_contact.rs)
- `a_box_has_the_textbook_mass_and_inertia` (analytic_mass_properties.rs)
- `a_box_lands_flat_on_a_ball_by_its_face_centre` (analytic_sdf_body_collider.rs)
- `a_box_mesh_has_the_boxs_bounding_box` (analytic_collision_mesh.rs)
- `a_box_on_a_static_box_is_lifted_by_the_whole_depth` (analytic_convex_contact.rs)
- `a_budget_of_one_sweep_is_reported_as_not_converged` (analytic_thermoplastic_coupling.rs)
- `a_capsule_matches_a_quadrature_of_its_solid` (analytic_mass_properties.rs)
- `a_cavity_narrower_than_a_cell_leaves_no_trace` (mesh_quality.rs)
- `a_cell_carried_past_the_last_cell_is_lost_and_the_volume_drops_by_it` (analytic_multiphase_wiring.rs)
- `a_charge_on_the_high_wall_is_rejected` (analytic_maxwell_wiring.rs)
- `a_charge_on_the_low_wall_is_rejected` (analytic_maxwell_wiring.rs)
- `a_child_aabb_follows_the_body_pose` (analytic_compound.rs)
- `a_clamped_push_lands_in_the_analytic_free_band` (analytic_sdf_character_up_axis.rs)
- `a_closed_box_is_no_slip_except_where_it_is_one_cell_thick` (analytic_cfd_flow_bc.rs)
- `a_closed_current_loop_creates_no_charge` (analytic_maxwell_fdtd.rs)
- `a_collision_free_move_reports_the_sampled_distance_as_best` (analytic_sdf_character_up_axis.rs)
- `a_compound_body_has_the_mass_and_inertia_of_its_children` (analytic_compound.rs)
- `a_compound_collides_as_its_children_not_as_its_hull` (analytic_compound.rs)
- `a_compound_without_volume_or_density_is_refused` (analytic_compound.rs)
- `a_cone_and_a_wedge_sit_on_the_floor_by_their_centre_of_mass` (analytic_convex_contact.rs)
- `a_configuration_without_a_law_is_refused` (analytic_cubic_hyperelastic.rs)
- `a_configuration_without_a_law_is_refused` (analytic_quadratic_hyperelastic.rs)

... and 2165 more

---

## How to Contribute

When implementing a pending oracle:
1. Remove `#[ignore]` from the test
2. Implement the corresponding functionality in `src/`
3. Run `cargo test <test_name>` to verify
4. The oracle status will auto-update on next CI run

For details: [CLAUDE.md](../CLAUDE.md)
